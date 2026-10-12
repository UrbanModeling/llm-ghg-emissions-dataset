"""
Convert a record to Mt/yr of its gas and look up the OWID national value (unit helpers).
CO2 and carbon (x 44/12) -> OWID co2; GHG -> total_ghg_excluding_lucf;
CH4 / N2O -> methane / nitrous_oxide (Mt CO2-eq; GWP100 27.9 / 273 when the unit is not CO2-eq).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

OWID = Path(__file__).resolve().parent.parent / "data" / "external" / "owid-co2-data.csv"
MULT_RE = re.compile(r"(?<![a-z])(thousand|million|billion|trillion|mln)", re.I)
# mass prefix -> factor to Mt (applied to the base unit t)
_END = r"(?=$|[\s./(\-]|co|c\b|ce\b|e\b|pa\b)"   # prefix directly followed by the gas ("MtCO2")
PREFIX = [
    (r"^(pg|petagrams?|gt|gigatonn?e?s?|gigatons?|billion (metric )?tonn?e?s?|bn ?t)" + _END, 1e3),
    (r"^(tg|teragrams?|mt|mmt|megatonn?e?s?|megatons?|million (metric )?tonn?e?s?)" + _END, 1.0),
    (r"^(gg|gigagrams?|kt|kilotonn?e?s?|kilotons?|thousand (metric )?tonn?e?s?)" + _END, 1e-3),
    (r"^((metric )?tonn?e?s?|tons?|t)" + _END, 1e-6),
    (r"^(kg|kilograms?)" + _END, 1e-9),
]
# annual expressions are removed before testing EXCLUDE_RE
ANNUAL_RE = re.compile(r"\s*((per|/)\s*(yr|year|annum|a)\b|yr\s*\(?-1\)?|a\(-1\)|\bpa\b|annual(ly)?)", re.I)
# units that are not an annual national mass
EXCLUDE_RE = re.compile(
    r"%|per|/|cap|ha\b|km|m\(?[23]|mu|µ|\bd\(-1\)|day|month|kwh|mwh|gj|"
    r"\bh\b|hour|head|person|gdp|usd|\$|yuan|kg\(-1\)|t\(-1\)|g\(-1\)", re.I)


def to_mt(row):
    """Record value in Mt of the unit's gas per year, or NaN if the unit does not convert."""
    u = str(row["unit"]).strip().lower()
    if EXCLUDE_RE.search(ANNUAL_RE.sub("", u)):
        return np.nan
    for pat, f in PREFIX:
        if re.search(pat, u):
            factor = f
            break
    else:
        return np.nan
    v = row["value_num"]
    if MULT_RE.search(str(row["value_text"])):
        # the multiplier word is already in value_num; a prefix in the unit ("Mt") repeats it
        factor = 1e-6 if factor >= 1e-6 else factor
    return v * factor


def is_carbon_unit(row):
    u = str(row["unit"])
    return row["gas"] == "carbon (C)" or bool(re.search(r"(?<![A-Za-z])C(?![A-Za-z(O])|carbon(?! dioxide)", u))


BASIS = {"CO2": "co2", "carbon (C)": "co2", "GHG (CO2-eq)": "total_ghg_excluding_lucf",
         "CH4": "methane", "N2O": "nitrous_oxide"}
CO2EQ_RE = re.compile(r"eq|co2e|co\(2\)e|co2-e|\bce\b", re.I)


def owid_table() -> pd.DataFrame:
    """(country_iso3, year, column) -> value, Mt."""
    o = pd.read_csv(OWID)
    o = o[o.iso_code.notna() & ~o.iso_code.str.startswith("OWID", na=False)]
    return o.melt(id_vars=["iso_code", "year"], value_vars=sorted(set(BASIS.values())),
                  var_name="column", value_name="owid_mt").rename(columns={"iso_code": "country_iso3"})


def record_mt(row) -> float:
    """Mt/yr on the OWID basis of the record's gas, NaN if it does not convert."""
    mt = to_mt(row)
    if np.isnan(mt) or row["gas"] not in BASIS:
        return np.nan
    if row["gas"] in ("CO2", "carbon (C)") and is_carbon_unit(row):
        mt *= 44 / 12
    if row["gas"] == "CH4" and not CO2EQ_RE.search(str(row["unit"])):
        mt *= 27.9
    if row["gas"] == "N2O" and not CO2EQ_RE.search(str(row["unit"])):
        mt *= 273
    return mt
