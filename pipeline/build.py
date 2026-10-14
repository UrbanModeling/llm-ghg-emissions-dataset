"""
Step 4: check, normalize and filter the extracted values into the dataset.

  - drops values whose numbers are not in the text, values without a number, bare years, and values
    Step 3 says are not emissions
  - unit: a multiplier already in the value is removed from the unit ('948 million' + 'Mt' -> 't')
  - value -> value_num (mean of a range), value_low, value_high, value_qualifier
  - year -> 'YYYY' / 'YYYY-YYYY' / ''; (country, place) -> country, ISO3, subnational unit or city
  - quantity_type: an emission per unit -> intensity; an emission the measure calls a reduction -> change
  - gas and Step 3 basis outside their fixed lists: 'GHG', 'GHG (CO2-e)' / 'CO2-e' -> CO2-eq, 'CO2-C' ->
    carbon, a named air pollutant (CO, NH3) -> air pollutant, anything else -> other (gas) or '' (basis)
  - gas: a unit in mass of carbon -> carbon (C); a unit in CO2-eq -> GHG; Step 3 basis switches CO2 / carbon
  - is_total: Step 3 coverage 'total' and sector 'all sectors' (not consumption-based, embodied or a
    company's), and Step 2's sector all sources or fossil / energy CO2; sector_group from Step 3
  - Step 1 checks (chk_year, chk_place): a year the text does not mention is cleared, a subnational unit
    or city it does not mention is widened to the country, a country it does not mention is cleared
  - chk_magnitude: a national total outside 0.05-60 t CO2(-eq) per person is not a total
  - keeps values with a greenhouse gas (not air pollutants, which the search does not cover) and a
    quantity type, not in money, area, energy or people units,
    a country and a parsed number

Input:  output/prepared.csv, output/candidates.csv, output/<run>/extracted.csv, output/<run>/verified.csv
Output: output/<run>/emission_records.csv, output/<run>/steps.csv (values kept after each filter)

Usage:
    python pipeline/build.py --run test
"""
import argparse
import re
import sys
from pathlib import Path

import geonamescache
import pandas as pd

from places import _key, geocode
from sectors import AR6_SECTOR, sector_group
from units import to_mt
from values import (MASS_PREFIX_RE, MULT_ANY_RE, MULT_RE, in_text, normalize_time, nums, numbers_in,
                    parse_value, value_qualifier)

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
COLUMNS = ["uid", "measure", "value_text", "value_num", "value_low", "value_high", "value_qualifier", "unit",
           "quantity_type", "gas", "is_total", "country", "country_iso3", "subnational", "city", "place",
           "spatial_level", "lat", "lon", "year", "sector", "sector_group", "ar6_sector", "evidence"]
ALL_SOURCES_RE = re.compile(r"all (?:sources|sectors)|economy-wide|whole economy", re.I)
# a CO2 value from these sources is the place's CO2 total as inventories report it
FOSSIL_CO2_RE = re.compile(r"(?:commercial )?(?:fossil[- ]fuels?|energy[- ]related|energy consumption)"
                           r"(?: (?:combustion|consumption|use|burning))?(?: and cement.*)?", re.I)
# a unit in mass of carbon: Mt C, Mt-C, MtC, Tg C yr-1, t C-eq, carbon equivalent (not tCO2, carbon dioxide)
CARBON_UNIT_RE = re.compile(r"(?:^|(?<=[\s\-(/tTgG]))C(?![a-zA-Z0-9(])|carbon(?! dioxide)", re.I)
# not the place's territorial emissions: consumption-based or company accounts, sinks, quotas, capture
NOT_TOTAL_RE = re.compile(r"consumer|consumption-based|embodied|footprint|scope 1|company|corporate|"
                          r"sink|remov|sequest|uptake|assimilat|quota|allowance|exceed|captur|CDR", re.I)
# units of things that are not emissions (area, energy, people, money, oil)
NOT_EMISSION_UNIT_RE = re.compile(r"^\s*(?:km|ha\b|hectares?|m2|m\(2\)|people|persons?|deaths?|years?\b|"
                                  r"[kMGTPE]?J\b|[kMGT]?Wh|MMbbl|barrels?|francs|swiss)", re.I)
# per-unit values (per capita, per ha, per km2 ...); a time unit alone (Mt/yr) is not per-unit
PER_UNIT_RE = re.compile(r"\bper\b(?!\s+(?:year|annum|yr|day|month)\b)|/(?!\s*(?:yr|year|a|y|d|day|month)\b)|"
                         r"\((?!-1\))-?\d\)|(?:km|ha|m|capita|cap)\s*\(?-\d", re.I)
CHANGE_RE = re.compile(r"reduc|mitigat|saving|saved|avoid|abat|decreas|declin|increas|cut", re.I)
POPULATION = {c["iso3"]: c["population"] for c in geonamescache.GeonamesCache().get_countries().values()
              if c["population"]}
MONEY_RE = re.compile(r"[$€£¥]|\b(?:usd|eur|euros?|dollars?|yuan|rmb|cny)\b", re.I)
# fixed lists of Steps 2-3; a model without schema-constrained output (JSON mode only) can answer outside them
GASES = ["CO2", "CH4", "N2O", "GHG (CO2-eq)", "carbon (C)", "other GHG", "air pollutant", "other"]
GAS_CO2EQ_RE = re.compile(r"^(?:GHGs?(?:\s*\(?\s*CO2[\s-]*e(?:q|quivalent)?\s*\)?)?|CO2[\s-]*e(?:q|quivalent)?)$", re.I)
GAS_CARBON_RE = re.compile(r"^(?:CO2-C|C|carbon)$", re.I)
AIR_POLLUTANT_RE = re.compile(r"^(?:CO|NH3|SO2|SOx|NOx|NO2|NO|PM\s*[\d.]*|VOCs?|NMVOCs?|BC|black carbon|O3)$", re.I)
BASES = ["CO2", "carbon", "CO2-eq", "not a mass"]


def fixed_gas(g: str) -> str:
    """Step 2's gas within GASES: 'GHG', 'GHG (CO2-e)', 'CO2-eq' -> GHG (CO2-eq); 'CO2-C' -> carbon (C); a named
    air pollutant (CO, NH3) -> air pollutant; any other answer -> other."""
    if g in GASES:
        return g
    if GAS_CO2EQ_RE.match(g.strip()):
        return "GHG (CO2-eq)"
    if GAS_CARBON_RE.match(g.strip()):
        return "carbon (C)"
    return "air pollutant" if AIR_POLLUTANT_RE.match(g.strip()) else "other"


def strip_unit(value_text: str, unit: str) -> tuple[str, str]:
    """(value without the unit, unit without a multiplier the value already has)."""
    if MULT_RE.search(value_text) and MULT_RE.search(unit):
        unit = MULT_RE.sub("", unit, count=1).strip()
    value = re.sub(r"\s+", " ", re.sub(re.escape(unit.strip()), "", value_text, flags=re.I)).strip() \
        if unit.strip() else value_text
    if MULT_ANY_RE.search(value) and MASS_PREFIX_RE.search(unit):
        unit = MASS_PREFIX_RE.sub(lambda m: "t" if m.group().strip().endswith("g") else "", unit, count=1).strip()
    return value, unit


def check_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """chk_year / chk_place: the model's year and place are among those the text mentions (Step 1 lists).
    A year the text does not mention is cleared; a subnational unit or city it does not mention is
    widened to the country; a country it does not mention is cleared (the value then has no place)."""
    path = BASE / "output" / "candidates.csv"
    if not path.exists():
        return df
    cand = pd.read_csv(path, dtype=str, keep_default_na=False).astype({"uid": int}).set_index("uid")
    years = df.uid.map(cand.years).fillna("").str.split("|").map(set)
    countries = df.uid.map(cand.countries).fillna("").str.split("|").map(set)
    places = df.uid.map(cand.places).fillna("").str.split("|").map(set)
    df["chk_year"] = [y == "" or y in ys or all(p in ys for p in y.split("-")) for y, ys in zip(df.year, years)]
    country_ok = pd.Series([lvl == "none" or iso in cs for lvl, iso, cs in
                            zip(df.spatial_level, df.country_iso3, countries)], index=df.index)
    sub_ok = pd.Series([lvl not in ("subnational", "city") or f"{iso}:{_key(s or c)}" in ps for lvl, iso, s, c, ps in
                        zip(df.spatial_level, df.country_iso3, df.subnational, df.city, places)], index=df.index)
    df["chk_place"] = country_ok & sub_ok
    df.loc[~df.chk_year, "year"] = ""
    widen = country_ok & ~sub_ok
    df.loc[widen, ["subnational", "city", "lat", "lon"]] = ["", "", None, None]
    df.loc[widen, "spatial_level"] = "country"
    df.loc[~country_ok, "spatial_level"] = "none"
    return df


def check_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """chk_magnitude: a national total must be 0.05-60 t CO2(-eq) per person; otherwise it is not a total."""
    t = df.is_total & (df.spatial_level == "country") & (df.quantity_type == "emission") \
        & df.gas.isin(["CO2", "carbon (C)", "GHG (CO2-eq)"])
    mt = pd.Series([to_mt(r) for _, r in df[t].iterrows()], index=df.index[t], dtype=float)
    mt[df.gas[t] == "carbon (C)"] *= 44 / 12
    per_capita = mt * 1e6 / df.country_iso3[t].map(POPULATION)
    df["chk_magnitude"] = ~per_capita.reindex(df.index).pipe(lambda x: x.notna() & ~x.between(0.05, 60))
    df.loc[~df.chk_magnitude, "is_total"] = False
    return df


def main(run: str):
    out = BASE / "output" / run
    df = pd.read_csv(out / "extracted.csv", dtype=str, encoding="utf-8-sig").fillna("")
    df["uid"] = df.uid.astype(int)
    df["vid"] = df.groupby("uid").cumcount() + 1
    ver_path = out / "verified.csv"
    verified = ver_path.exists()
    if verified:                                        # Step 3 answers, by paper and value number
        ver = pd.read_csv(ver_path, dtype=str, keep_default_na=False).drop_duplicates(["uid", "vid"])
        ver = ver[ver.vid.str.fullmatch(r"\d+")].astype({"uid": int, "vid": int})
        df = df.merge(ver, on=["uid", "vid"], how="left").fillna("")
        df["basis"] = df.basis.replace({"CO2-e": "CO2-eq"}).where(lambda b: b.isin(BASES + [""]), "")
    df["gas"] = df.gas.map(fixed_gas)
    text = pd.read_csv(BASE / "output" / "prepared.csv", usecols=["uid", "text"]).set_index("uid").text
    full = df.uid.map(text).fillna("").astype(str)
    n0 = len(df)

    df["chk_evidence"] = [in_text(e, t) for e, t in zip(df.evidence, full)]
    df["chk_value_in_text"] = [numbers_in(v, t) for v, t in zip(df.value_text, full)]
    df[["value", "unit"]] = [strip_unit(v, u) for v, u in zip(df.value_text, df.unit)]
    parsed = df.value.map(parse_value)
    df["value_num"], df["value_low"], df["value_high"] = zip(*parsed) if len(df) else ([], [], [])
    df["value_qualifier"] = df.value.map(value_qualifier)
    df["year"] = [normalize_time(y) for y in df.year]
    # emission amounts per unit are intensities, emission amounts described as cuts are changes
    em = df.quantity_type == "emission"
    df.loc[em & (df.unit.str.contains(PER_UNIT_RE) | df.measure.str.contains(r"\bper\b|intensity", case=False)),
           "quantity_type"] = "intensity"
    df.loc[(df.quantity_type == "emission") & df.measure.str.contains(CHANGE_RE), "quantity_type"] = "change"
    geo = pd.DataFrame([geocode(c, p) for c, p in zip(df.country.str.strip(), df.place.str.strip())],
                       index=df.index)
    df = df.drop(columns="country").join(geo)
    # CO2 whose unit is a mass of carbon is carbon
    df.loc[(df.gas == "CO2") & df.unit.str.contains(CARBON_UNIT_RE), "gas"] = "carbon (C)"
    # CO2 with a unit in CO2 equivalents is a GHG total
    df.loc[(df.gas == "CO2") & df.unit.str.contains(r"CO2\s*-?\s*e(?:q|\b)|CO\(2\)\s*-?e(?:q|\b)|\beq", case=False), "gas"] = "GHG (CO2-eq)"
    if verified:
        # Step 3: basis switches CO2 / carbon; coverage and group decide the total
        df.loc[(df.gas == "CO2") & (df.basis == "carbon"), "gas"] = "carbon (C)"
        df.loc[(df.gas == "carbon (C)") & (df.basis == "CO2") & ~df.unit.str.contains(CARBON_UNIT_RE), "gas"] = "CO2"
        df["is_total"] = (df.coverage == "total") & (df.group == "all sectors") & ~df.measure.str.contains(NOT_TOTAL_RE)
    # Step 2's sector: all sources, or fossil / energy CO2 (a named sector is not the place's total)
    all_sources = df.sector.str.contains(ALL_SOURCES_RE) | (
        df.gas.isin(["CO2", "carbon (C)"]) & df.sector.str.strip().str.fullmatch(FOSSIL_CO2_RE))
    if verified:
        df["is_total"] &= all_sources
    else:
        df["is_total"] = all_sources & ~df.measure.str.contains(NOT_TOTAL_RE)
    df = check_candidates(df)
    df = check_magnitude(df)
    # 'all sources' names no sector unless the value is a total
    if verified:
        df["sector_group"] = ["Multi-sector" if t or g == "all sectors" else g for t, g in zip(df.is_total, df.group)]
    else:
        df["sector_group"] = ["Multi-sector" if t else sector_group("" if ALL_SOURCES_RE.search(s) else s, m)
                              for t, s, m in zip(df.is_total, df.sector, df.measure)]
    df["ar6_sector"] = df.sector_group.map(AR6_SECTOR).fillna("")

    bare_year = df.value.str.fullmatch(r"\s*(?:before |by |after )?(?:19|20)\d{2}\+?\s*", case=False)
    steps = [
        ("numbers in the text", df.chk_value_in_text),
        ("an emission (Step 3)", df.get("is_emission", pd.Series("", index=df.index)) != "no"),
        ("has a number, not a bare year", df.value.map(lambda v: bool(nums(v))) & ~bare_year),
        ("greenhouse gas and quantity type", ~df.gas.isin(["other", "air pollutant"]) & (df.quantity_type != "other")),
        ("not money, area, energy, people", ~(df.value_text + " " + df.unit).str.contains(MONEY_RE)
         & ~df.unit.str.contains(NOT_EMISSION_UNIT_RE)),
        ("has a country", df.spatial_level != "none"),
        ("value parsed", df.value_num.notna()),
    ]
    keep = pd.Series(True, index=df.index)
    print(f"{n0} values from {df.uid.nunique()} papers")
    counts = [("extracted", n0)]
    for label, mask in steps:
        keep &= mask
        counts.append((label, int(keep.sum())))
        print(f"  after {label:32s} {int(keep.sum())}")
    pd.DataFrame(counts, columns=["step", "values"]).to_csv(out / "steps.csv", index=False)
    rec = df[keep]
    rec = rec[COLUMNS + [c for c in rec.columns if c not in COLUMNS]]
    rec.to_csv(out / "emission_records.csv", index=False, encoding="utf-8-sig")
    print(f"-> emission_records.csv: {len(rec)} records, {rec.uid.nunique()} papers | "
          f"levels {rec.spatial_level.value_counts().to_dict()} | totals {int(rec.is_total.sum())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 4: build the dataset")
    p.add_argument("--run", default="full")
    main(p.parse_args().run)
