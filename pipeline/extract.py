"""
Step 2: LLM extraction with Gemini 2.5 Flash Lite, one call per paper.

The model lists every quantitative emission value in the title, keywords and abstract, copying
evidence, value and unit, and classifying each value (fields in FIELDS).

Input:  output/prepared.csv
Output: output/<run>/extracted.json   (per paper: uid, raw response)
        output/<run>/extracted.csv    (one row per value)

Re-running the same --run resumes: papers already in the JSON are skipped, failed calls are retried.

Usage:
    python pipeline/extract.py --run test --uids-file eval/gold/uids.txt
    python pipeline/extract.py                                   # all papers -> output/full
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
PREPARED = BASE / "output" / "prepared.csv"
MODEL = "gemini-2.5-flash-lite"
PRICE_IN, PRICE_OUT = 0.10, 0.40       # USD per 1M tokens, for the running estimate only
SAVE_EVERY = 100

SYSTEM_PROMPT = """List every quantitative value in the text about greenhouse gas or air pollutant emissions to the air, one object
per value: amounts emitted or removed, changes, shares, intensities, targets and uncertainties. "From A in 2000
to B in 2010" gives two values, A and B. Skip values that are not about emissions (money, energy, area, counts,
concentrations, fluxes to water, carbon stocks, storage capacity, model fit). Copy words exactly from the text; do not calculate.
"""

# field -> (enum values or None, description); order = output order
FIELDS = {
    "evidence":      (None, "Sentence containing the value, copied."),
    "value_text":    (None, "Value exactly as written, with qualifiers and multiplier words, e.g. '>25', "
                            "'about 2', '1.9 billion to 3.2 billion'."),
    "unit":          (None, "Unit as written."),
    "measure":       (None, "What the value measures, in a few words."),
    "quantity_type": (["emission", "change", "share", "intensity", "target", "uncertainty", "other"],
                      "emission = amount emitted or removed; change = reduction, increase, saving or avoided "
                      "amount; share = fraction of a total; intensity = per unit (per capita, per GDP, per ha, "
                      "emission factor); uncertainty = error or confidence range; other = not about emissions."),
    "gas":           (["CO2", "CH4", "N2O", "GHG (CO2-eq)", "carbon (C)", "other GHG", "air pollutant", "other"],
                      "CO2 includes 'carbon emissions' and 'carbon dioxide'; carbon (C) only for a mass of "
                      "carbon (t C, Gt C, carbon equivalent); other GHG = F-gases; air pollutant = SO2, NOx, PM, "
                      "CO, VOC, NH3, black carbon; other = not an emission to the air."),
    "country":       (None, "Country the value refers to, English short name, also when only a place in it is "
                            "named ('South China' -> China) or when it is the country of the study area "
                            "named elsewhere in the text; 'none' only if global or several countries."),
    "place":         (None, "Place within the country the value refers to (province, state, city, county, "
                            "region, or the study area), as named; '' if the whole country."),
    "year":          (None, "Year or period of the value, YYYY or YYYY-YYYY; when the sentence gives none, the "
                            "study period stated elsewhere in the text if the value belongs to it; '' if no year "
                            "applies."),
    "sector":        (None, "Sector, activity or product the emissions come from, in a few words, e.g. power, "
                            "paper industry, forest fires, pig farming; when the sentence does not say, the "
                            "paper's topic. 'all sources' only when the value is the total emissions of a "
                            "country, region or city from every sector (e.g. 'China's CO2 emissions')."),
}


def build_schema() -> types.Schema:
    props = {k: types.Schema(type=types.Type.STRING, description=d, **({"enum": e} if e else {}))
             for k, (e, d) in FIELDS.items()}
    value = types.Schema(type=types.Type.OBJECT, properties=props, required=list(FIELDS),
                         property_ordering=list(FIELDS))
    return types.Schema(type=types.Type.OBJECT,
                        properties={"values": types.Schema(type=types.Type.ARRAY, items=value)},
                        required=["values"])


def save(done: list, out_json: Path, out_csv: Path):
    out_json.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = [{"uid": d["uid"], **{k: v.get(k, "") for k in FIELDS}}
            for d in done for v in d["response"].get("values", [])]
    pd.DataFrame(rows, columns=["uid", *FIELDS]).to_csv(out_csv, index=False, encoding="utf-8-sig")


def main(run: str, uids_file: str | None, workers: int):
    load_dotenv(BASE / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    df = pd.read_csv(PREPARED, usecols=["uid", "text"])
    if uids_file:
        df = df[df.uid.isin({int(x) for x in open(uids_file).read().split()})]

    out = BASE / "output" / run
    out.mkdir(parents=True, exist_ok=True)
    out_json, out_csv = out / "extracted.json", out / "extracted.csv"
    done = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    todo = df[~df.uid.isin({d["uid"] for d in done})]
    print(f"{MODEL} | run '{run}' | {len(df)} papers, {len(todo)} to do", flush=True)

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0,
                                         response_mime_type="application/json", response_schema=build_schema())
    lock = threading.Lock()
    tok_in = tok_out = errors = 0
    t0 = time.perf_counter()

    def call(uid, text):
        r = client.models.generate_content(model=MODEL, contents=f"TEXT:\n{text}", config=config)
        return json.loads(r.text or "{}"), r.usage_metadata.prompt_token_count, r.usage_metadata.candidates_token_count

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call, int(u), str(t)): int(u) for u, t in zip(todo.uid, todo.text)}
        for i, fut in enumerate(as_completed(futures), 1):
            uid = futures[fut]
            with lock:
                try:
                    resp, ti, to = fut.result()
                    done.append({"uid": uid, "response": resp})
                    tok_in, tok_out = tok_in + ti, tok_out + (to or 0)
                except Exception as e:                        # not stored: retried on the next run
                    errors += 1
                    print(f"[error] uid={uid}: {str(e)[:200]}", flush=True)
                if i % 100 == 0:
                    print(f"[{i}/{len(todo)}] errors {errors}", flush=True)
                if i % SAVE_EVERY == 0:
                    save(done, out_json, out_csv)

    save(done, out_json, out_csv)
    cost = tok_in * PRICE_IN / 1e6 + tok_out * PRICE_OUT / 1e6
    print(f"done: {len(done)} papers, {errors} failed | tokens in {tok_in:,} out {tok_out:,} | ~${cost:.3f} | "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 2: LLM extraction")
    p.add_argument("--run", default="full")
    p.add_argument("--uids-file")
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    main(a.run, a.uids_file, a.workers)
