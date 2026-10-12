"""
Table 5 of the paper: flagged national totals compared with PRIMAP-hist, EDGAR and the Global Carbon Budget.

Records: is_total, country level, quantity_type emission, one year 1950-2023, a unit that converts to Mt/yr. Writes the pairs and the summary table to eval/scores.

Usage:
    python eval/refcheck.py --run full
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from owid import record_mt
from reference import GAS_BASIS, SOURCES, reference_table

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
GAS_ORDER = ["CO2", "carbon (C)", "GHG (CO2-eq)", "CH4", "N2O"]


def candidates(rec: pd.DataFrame) -> pd.DataFrame:
    t = rec[(rec.is_total == "True") & (rec.spatial_level == "country") & (rec.quantity_type == "emission")
            & rec.year.str.fullmatch(r"\d{4}") & rec.gas.isin(list(GAS_BASIS))].copy()
    t["value_num"] = t.value_num.astype(float)
    t["mt"] = t.apply(record_mt, axis=1)
    t["year"] = t.year.astype(int)
    t = t[t.mt.notna() & t.year.between(1950, 2023)]
    t["basis"] = t.gas.map(GAS_BASIS)
    return t


def pairs(rec: pd.DataFrame) -> pd.DataFrame:
    """One row per record and reference source that has a value for its country, year and basis."""
    p = candidates(rec).merge(reference_table(), on=["country_iso3", "year", "basis"], how="inner")
    p["ratio"] = p.mt / p.ref_mt
    return p


def summary(g: pd.DataFrame) -> dict:
    err = (g.ratio - 1).abs()
    return {"records": len(g), "papers": g.uid.nunique(), "countries": g.country_iso3.nunique(),
            "years": f"{g.year.min()}–{g.year.max()}" if len(g) else "",
            "within_10": 100 * (err <= 0.10).mean(), "within_25": 100 * (err <= 0.25).mean(),
            "within_50": 100 * (err <= 0.50).mean(), "median_ratio": g.ratio.median(),
            "ratio_q1": g.ratio.quantile(0.25), "ratio_q3": g.ratio.quantile(0.75),
            "spearman": g[["mt", "ref_mt"]].corr("spearman").iloc[0, 1] if len(g) > 2 else float("nan")}


def main(run: str):
    rec = pd.read_csv(BASE / "output" / run / "emission_records.csv", dtype=str, keep_default_na=False)
    cand = candidates(rec)
    p = pairs(rec)
    rows = []
    for src in SOURCES:
        s = p[p.source == src]
        if s.empty:
            continue
        rows.append((src, "all", summary(s)))
        for gas in GAS_ORDER:
            if (s.gas == gas).any():
                rows.append((src, gas, summary(s[s.gas == gas])))
        rows.append((src, "China", summary(s[s.country_iso3 == "CHN"])))
        rows.append((src, "Other countries", summary(s[s.country_iso3 != "CHN"])))
    out = pd.DataFrame([{"source": a, "subset": b, **x} for a, b, x in rows])

    # disagreement between the references themselves, on the records both cover
    w = p.pivot_table(index=["uid", "vid"], columns="source", values="ref_mt")
    spread = (w.EDGAR / w.PRIMAP).dropna()
    # a record agrees with "any" reference if it is within the bound of at least one of them
    best = p.assign(err=(p.ratio - 1).abs()).groupby(["uid", "vid"]).err.min()
    print(f"{run}: {len(cand)} candidate totals, {p[['uid', 'vid']].drop_duplicates().shape[0]} paired with ≥1 reference")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"EDGAR/PRIMAP on the same records: n={len(spread)}, median {spread.median():.3f}, "
          f"IQR {spread.quantile(.25):.3f}–{spread.quantile(.75):.3f}, "
          f"within ±10% {(abs(spread - 1) <= .1).mean():.1%}")
    print(f"within ±25% / ±50% of at least one reference: {(best <= .25).mean():.1%} / {(best <= .5).mean():.1%} "
          f"(n={len(best)})")
    scores = BASE / "eval" / "scores"
    p.to_csv(scores / f"{run}_refcheck_pairs.csv", index=False, encoding="utf-8-sig")
    out.to_csv(scores / f"{run}_refcheck_table.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    main(ap.parse_args().run)
