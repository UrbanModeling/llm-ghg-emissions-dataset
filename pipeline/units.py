"""
Convert a value to Mt per year of its gas (same helpers as eval/owid.py).
"""
import re

import numpy as np

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
