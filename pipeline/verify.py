"""
Step 3: a second LLM call per paper that checks the values Step 2 extracted.

The model gets the text and a numbered list of the extracted values and answers three questions per
value: is it an emission to or removal from the air, what does it cover, what is its mass basis and
which sector does it come from. The list leaves out Step 2's sector words ('all sources' misleads).

Input:  output/prepared.csv, output/<run>/extracted.csv
Output: output/<run>/verified.json   (per paper: uid, raw response)
        output/<run>/verified.csv    (uid, vid, is_emission, coverage, basis, group); vid = row order in the paper

Usage:
    python pipeline/verify.py --run test
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
MODEL = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """\
You check values extracted from a scientific abstract. For each numbered value, answer from the text.
"""

FIELDS = {
    "id":          (None, "Number of the value in the list."),
    "is_emission": (["yes", "no"],
                    "yes if the value is an amount of a greenhouse gas or air pollutant emitted to or removed "
                    "from the air, or its change, share, intensity, target or uncertainty; no for carbon "
                    "stocks or storage, concentrations, fuel, energy or material amounts, costs, counts, "
                    "probabilities, fluxes to water and model statistics."),
    "coverage":    (["total", "sector", "gas subset", "entity", "part of place", "sink", "other"],
                    "What the value covers. total = all emissions of a country, province, state or city "
                    "from every source (its energy-related or fossil-fuel CO2 also counts); sector = one "
                    "sector, fuel or activity; gas subset = only some gases (e.g. non-CO2, F-gases); "
                    "entity = a company, facility, park, project, event, product or group of people; part "
                    "of place = only part of the country, province or city the value is for (e.g. some of "
                    "its districts or provinces); sink = removal "
                    "or sequestration; other = exported, embodied or consumption-based emissions, "
                    "scenario differences and anything else."),
    "basis":       (["CO2", "carbon", "CO2-eq", "not a mass"],
                    "Mass basis of the number: carbon if it is tonnes of carbon (units such as t C, or "
                    "'carbon emissions' whose size fits tonnes of carbon rather than CO2); CO2-eq for "
                    "several gases in CO2 equivalents; not a mass for percentages, ratios and intensities."),
    "group":       (["Energy", "Industry", "Transport", "Building", "Agriculture", "LULUCF", "Waste", "Other",
                     "all sectors"],
                    "Sector the emissions come from; the paper's topic when the sentence does not say. "
                    "Energy = power, heat, fuel production and supply; Industry = manufacturing, cement, "
                    "steel, chemicals, F-gases; Transport; Building = residential and commercial energy use, "
                    "appliances; Agriculture = livestock, crops, fertilizer; LULUCF = land use, forests, "
                    "soils, fires, ecosystems; Waste = landfill, wastewater, incineration; Other = health "
                    "care, institutions, events, services; all sectors = the whole economy."),
}


def build_schema() -> types.Schema:
    props = {k: types.Schema(type=types.Type.INTEGER if k == "id" else types.Type.STRING, description=d,
                             **({"enum": e} if e else {})) for k, (e, d) in FIELDS.items()}
    item = types.Schema(type=types.Type.OBJECT, properties=props, required=list(FIELDS),
                        property_ordering=list(FIELDS))
    return types.Schema(type=types.Type.OBJECT,
                        properties={"checks": types.Schema(type=types.Type.ARRAY, items=item)},
                        required=["checks"])


def value_list(rows: pd.DataFrame) -> str:
    return "\n".join(f"{i}. {r.value_text} | unit: {r.unit} | {r.measure} | {r.country} {r.place} | "
                     f"{r.year}" for i, r in enumerate(rows.itertuples(), 1))


def checks(resp) -> list:
    """The checks in one response; a few responses are the bare list instead of {"checks": [...]}."""
    return resp if isinstance(resp, list) else resp.get("checks", [])


def main(run: str, workers: int, model: str):
    load_dotenv(BASE / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out = BASE / "output" / run
    ext = pd.read_csv(out / "extracted.csv", dtype=str, keep_default_na=False)
    text = pd.read_csv(BASE / "output" / "prepared.csv", usecols=["uid", "text"]).set_index("uid").text
    out_json = out / "verified.json"
    done = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    todo = [u for u in ext.uid.unique() if int(u) not in {d["uid"] for d in done}]
    print(f"{model} | verify '{run}' | {ext.uid.nunique()} papers, {len(todo)} to do", flush=True)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0,
                                         response_mime_type="application/json", response_schema=build_schema())
    lock, tok = threading.Lock(), [0, 0]

    def call(uid):
        rows = ext[ext.uid == uid]
        prompt = f"TEXT:\n{text[int(uid)]}\n\nVALUES:\n{value_list(rows)}"
        r = client.models.generate_content(model=model, contents=prompt, config=config)
        return int(uid), json.loads(r.text or "{}"), r.usage_metadata

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(call, u) for u in todo]):
            try:
                uid, resp, usage = fut.result()
            except Exception as e:                       # retried on the next run
                print(f"[error] {str(e)[:150]}", flush=True)
                continue
            with lock:
                done.append({"uid": uid, "response": resp})
                tok[0] += usage.prompt_token_count
                tok[1] += usage.candidates_token_count or 0
    out_json.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [{"uid": d["uid"], "vid": c.get("id"), **{k: c.get(k, "") for k in FIELDS if k != "id"}}
            for d in done for c in checks(d["response"])]
    pd.DataFrame(rows).to_csv(out / "verified.csv", index=False, encoding="utf-8-sig")
    print(f"done: {len(done)} papers | tokens in {tok[0]:,} out {tok[1]:,}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 3: LLM check of extracted values")
    p.add_argument("--run", default="full")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    main(a.run, a.workers, a.model)
