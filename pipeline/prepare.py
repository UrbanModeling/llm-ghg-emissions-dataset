"""
Merge the Web of Science exports into one table of texts.

uid = row number after concatenating data/raw/*.xls in file-name order, so it is stable as long as
data/raw does not change (the reference set relies on it).

Input:  data/raw/*.xls   (Web of Science "Full Record" exports; not redistributed)
Output: output/prepared.csv  (uid, wos_id, title, keywords, abstract, text)

Usage:
    python pipeline/prepare.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PREPARED = BASE / "output" / "prepared.csv"


def build_text(row) -> str:
    parts = []
    for label, col in (("Title", "title"), ("Keywords", "keywords"), ("Abstract", "abstract")):
        if isinstance(row[col], str) and row[col].strip():
            parts.append(f"{label}: {row[col].strip()}")
    return "\n".join(parts)


def main():
    xls_files = sorted((BASE / "data" / "raw").glob("*.xls"))
    merged = pd.concat([pd.read_excel(f) for f in xls_files], ignore_index=True)
    print(f"{len(xls_files)} files, {len(merged)} rows")

    df = merged[["UT (Unique WOS ID)", "Article Title", "Author Keywords", "Abstract"]].copy()
    df.columns = ["wos_id", "title", "keywords", "abstract"]
    df.index.name = "uid"
    df = df.reset_index()
    df["text"] = df.apply(build_text, axis=1)

    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPARED, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows -> {PREPARED.name}")


if __name__ == "__main__":
    main()
