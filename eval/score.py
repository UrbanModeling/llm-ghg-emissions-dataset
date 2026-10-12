"""
Score a run against the reference set (eval/gold/gold.psv) and against OWID.

Matching: within a paper, a record and a gold value match when the numbers in their value texts
agree (commas, multiplier words and 'x 10^n' ignored); equal sets first, then one set inside the
other; one-to-one. Gold cells may list alternatives separated by ';' ('' = empty is accepted).

Per matched pair: value (equal numbers), unit (present), location (finest resolved place, or the
country, among the gold alternatives), year, quantity_type, gas, sector group, is_total (gold coverage total).
Recall is over gold values with a location (the dataset drops global values).

OWID: records with is_total, a country only, quantity_type emission, one year <= 2023 and a
unit that converts to Mt/yr, compared with the OWID national value for that gas and year.

Usage:
    python eval/score.py --run full   # -> eval/scores/<run>_pairs.csv, _summary.csv, _owid.csv
"""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from owid import BASIS, owid_table, record_mt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from places import country_iso3  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
EXP_RE = re.compile(r"(?:x|×|\*)\s*10\s*\^?\s*\(?[-−]?\d+\)?|(?<=\d)e[+-]?\d+|\(\s*[-−]\s*\d+\s*\)", re.I)
NUM_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")
GAS = {"GHG (CO2-eq)": "GHG", "carbon (C)": "carbon"}
FIELDS = ["value", "unit", "location", "year", "quantity_type", "gas", "sector", "total"]
SUFFIX_RE = re.compile(r"\s+(city|province|state|region|county|district|prefecture|municipality)$", re.I)


def numbers(s) -> tuple:
    return tuple(sorted(round(float(x), 6) for x in NUM_RE.findall(EXP_RE.sub(" ", str(s).replace(",", "")))))


def key(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def alts(cell) -> list[str]:
    return [key(a) for a in str(cell).split(";")]


def judge(r, g, exact: bool) -> dict:
    places = [r.city, r.subnational, SUFFIX_RE.sub("", r.place) if r.spatial_level == "local" else "", r.country]
    finest = next(key(p) for p in places if str(p).strip())
    g_iso3 = {country_iso3(a) for a in str(g.location).split(";")} - {""}
    total = r.is_total == "True"
    return dict(
        value=exact,
        unit=bool(str(r.unit).strip()) or not g.unit,
        location=finest in alts(g.location) or (r.spatial_level == "country" and r.country_iso3 in g_iso3),
        country=r.country_iso3 in g_iso3 or finest in alts(g.location),
        year=key(r.year) in alts(g.time),
        quantity_type=key(r.quantity_type) in alts(g.quantity_type),
        gas=key(GAS.get(r.gas, r.gas)) in alts(g.gas),
        sector=key(r.sector_group) in alts(g.sector),
        total=("total" in alts(g.coverage)) if total else bool({"sector", "part"} & set(alts(g.coverage))),
    )


def main(run: str, gold_dir: str):
    gdir = BASE / "eval" / gold_dir
    gold = pd.read_csv(gdir / "gold.psv", sep="|", dtype=str, keep_default_na=False)
    # the dataset covers greenhouse gases only: drop gold values that can only be an air pollutant
    gold = gold[gold.gas.map(lambda c: set(alts(c)) != {"air pollutant"})].reset_index(drop=True)
    gold["uid"] = gold.uid.astype(int)
    gold["gid"] = range(len(gold))
    uids = {int(x) for x in open(gdir / "uids.txt").read().split()}
    rec = pd.read_csv(BASE / "output" / run / "emission_records.csv", dtype=str, keep_default_na=False)
    rec["uid"] = rec.uid.astype(int)
    rec = rec[rec.uid.isin(uids)].reset_index(drop=True)

    pairs, used_g = [], set()
    for uid, rg in rec.groupby("uid"):
        gg = gold[gold.uid == uid]
        cands = []
        for ri, r in rg.iterrows():
            rn = numbers(r.value)
            for gi, g in gg.iterrows():
                gn = numbers(g.value_text)
                if rn and gn and (rn == gn or set(rn) <= set(gn) or set(gn) <= set(rn)):
                    cands.append((rn != gn, ri, gi))
        used_r = set()
        for partial, ri, gi in sorted(cands):
            if ri not in used_r and gi not in used_g:
                used_r.add(ri)
                used_g.add(gi)
                pairs.append({"ri": ri, "gid": gold.at[gi, "gid"], **judge(rec.loc[ri], gold.loc[gi], not partial)})
    p = pd.DataFrame(pairs, columns=["ri", "gid", *FIELDS, "country"])
    out = BASE / "eval" / "scores"
    out.mkdir(exist_ok=True)
    rec.join(p.set_index("ri").add_prefix("ok_").rename(columns={"ok_gid": "gid"}), how="left").merge(gold.drop(columns="uid"), on="gid", how="left",
                                                   suffixes=("", "_gold")) \
        .to_csv(out / f"{run}_pairs.csv", index=False, encoding="utf-8-sig")
    missed = gold[~gold.gid.isin(used_g)]
    missed.to_csv(out / f"{run}_missed.csv", index=False, encoding="utf-8-sig")

    located = gold.location.map(lambda c: any(alts(c)))
    n_loc = int(located.sum())
    print(f"{run}: {len(rec)} records from {rec.uid.nunique()} of {len(uids)} papers | gold {len(gold)} values "
          f"({n_loc} with a location)")
    print(f"  recall (gold with a location)   {gold[located].gid.isin(used_g).mean():6.1%}  "
          f"({int(gold[located].gid.isin(used_g).sum())}/{n_loc})")
    print(f"  records matching gold           {len(p) / max(len(rec), 1):6.1%}  ({len(p)}/{len(rec)})")
    for f in FIELDS:
        print(f"  {f:14s} correct | matched  {p[f].mean():6.1%}")
    print(f"  (country right | matched     {p.country.mean():6.1%})")
    allok = p[FIELDS].all(axis=1)
    print(f"  all fields     correct | matched  {allok.mean():6.1%}")
    summary = [("recall", int(gold[located].gid.isin(used_g).sum()), n_loc), ("matching", len(p), len(rec)),
               *[(f, int(p[f].sum()), len(p)) for f in FIELDS], ("all fields", int(allok.sum()), len(p))]
    pd.DataFrame(summary, columns=["metric", "k", "n"]).to_csv(out / f"{run}_summary.csv", index=False)

    # OWID
    o = owid_table()
    t = rec[(rec.is_total == "True") & (rec.spatial_level == "country") & (rec.quantity_type == "emission")
            & rec.year.str.fullmatch(r"\d{4}") & rec.gas.isin(list(BASIS))].copy()
    t["value_num"] = t.value_num.astype(float)
    t["mt"] = t.apply(record_mt, axis=1)
    t["year"] = t.year.astype(int)
    t = t[t.mt.notna() & (t.year <= 2023)]
    t["column"] = t.gas.map(BASIS)
    t = t.merge(o, on=["country_iso3", "year", "column"], how="left")
    t["ratio"] = t.mt / t.owid_mt
    t[["uid", "value_text", "unit", "gas", "country", "year", "is_total", "sector", "mt", "owid_mt", "ratio",
       "evidence"]].to_csv(out / f"{run}_owid.csv", index=False, encoding="utf-8-sig")
    ratio = t.ratio.dropna()
    err = (ratio - 1).abs()
    print(f"  OWID: {len(ratio)} total records paired ({t.uid.nunique()} papers) | within ±25% "
          f"{(err <= 0.25).mean():.1%}, ±50% {(err <= 0.5).mean():.1%} | median ratio {ratio.median():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="test")
    ap.add_argument("--gold", default="gold", help="eval/<gold>/gold.psv and uids.txt")
    a = ap.parse_args()
    main(a.run, a.gold)
