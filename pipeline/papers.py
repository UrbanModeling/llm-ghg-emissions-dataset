"""
papers.csv: one row per paper in the dataset, linking uid to the Web of Science record.

Input:  output/prepared.csv (uid, wos_id), output/<run>/emission_records.csv, Web of Science export (*.xls)
Output: output/<run>/papers.csv (uid, wos_id, doi, title, publication_year)

Usage:
    python pipeline/papers.py --run full
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
WOS_RAW = BASE / "data" / "raw"


def main(run: str):
    out = BASE / "output" / run
    uids = pd.read_csv(out / "emission_records.csv", usecols=["uid"]).uid.unique()
    ids = pd.read_csv(BASE / "output" / "prepared.csv", usecols=["uid", "wos_id"])
    wos = pd.concat([pd.read_excel(f, usecols=["UT (Unique WOS ID)", "DOI", "Article Title", "Publication Year"])
                     for f in sorted(WOS_RAW.glob("*.xls"))])
    p = ids[ids.uid.isin(uids)].merge(wos, left_on="wos_id", right_on="UT (Unique WOS ID)", how="left")
    p = p.rename(columns={"DOI": "doi", "Article Title": "title", "Publication Year": "publication_year"})
    p = p[["uid", "wos_id", "doi", "title", "publication_year"]].sort_values("uid")
    p["publication_year"] = p.publication_year.astype("Int64")
    p.to_csv(out / "papers.csv", index=False, encoding="utf-8-sig")
    print(f"-> papers.csv: {len(p)} papers | without DOI {int(p.doi.isna().sum())} | "
          f"without WoS match {int(p.title.isna().sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    main(ap.parse_args().run)
