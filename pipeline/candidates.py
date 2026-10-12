"""
Step 1: list the years and places each paper mentions, with rules only (no LLM).

The lists are not shown to the model, so that its answers are not confined to them; build.py uses them to
check the model's year and place against the text.

  years:     every year 1900-2100 and every period ('2000-2013', 'from 2000 to 2013', '2009-10') in the text
  countries: ISO3 of countries named in the text (names, common aliases, demonyms), and of the
             countries of the subdivisions and cities named in it
  places:    subdivisions and cities (>= 100k people) named in the text, as 'ISO3:key'

Input:  output/prepared.csv
Output: output/candidates.csv   (uid, years, countries, places; '|'-separated)

Usage:
    python pipeline/candidates.py
"""
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pycountry

from places import CITIES, SUBDIVISIONS, _cc, _key
from values import RANGE_RE, YEAR_RE, normalize_time

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
ALIASES = {"USA": "USA", "U.S.": "USA", "US": "USA", "U.S.A.": "USA", "America": "USA", "UK": "GBR", "U.K.": "GBR",
           "Britain": "GBR", "England": "GBR", "PRC": "CHN", "UAE": "ARE", "Korea": "KOR", "Russia": "RUS",
           "Iran": "IRN", "Vietnam": "VNM", "Turkey": "TUR", "Taiwan": "TWN", "Hong Kong": "HKG", "Macau": "MAC"}
DEMONYMS = {"Chinese": "CHN", "American": "USA", "British": "GBR", "German": "DEU", "French": "FRA",
            "Japanese": "JPN", "Korean": "KOR", "Indian": "IND", "Brazilian": "BRA", "Russian": "RUS",
            "Canadian": "CAN", "Australian": "AUS", "Italian": "ITA", "Spanish": "ESP", "Mexican": "MEX",
            "Indonesian": "IDN", "Turkish": "TUR", "Iranian": "IRN", "Pakistani": "PAK", "Nigerian": "NGA",
            "Thai": "THA", "Vietnamese": "VNM", "Malaysian": "MYS", "Polish": "POL", "Dutch": "NLD",
            "Swedish": "SWE", "Norwegian": "NOR", "Danish": "DNK", "Finnish": "FIN", "Swiss": "CHE",
            "Austrian": "AUT", "Belgian": "BEL", "Greek": "GRC", "Egyptian": "EGY", "Saudi": "SAU",
            "Chilean": "CHL", "Argentine": "ARG", "Colombian": "COL", "Peruvian": "PER", "Irish": "IRL",
            "Portuguese": "PRT", "Czech": "CZE", "Hungarian": "HUN", "Romanian": "ROU", "Ukrainian": "UKR",
            "Kenyan": "KEN", "Ethiopian": "ETH", "Bangladeshi": "BGD", "Filipino": "PHL", "Taiwanese": "TWN"}
# names that are ordinary words or directions, never matched
GENERIC = {"north", "south", "east", "west", "central", "northern", "southern", "eastern", "western", "delta",
           "capital", "coast", "gulf", "free", "centre", "center", "mountain", "national", "federal", "union", "georgia"}

COUNTRY_NAMES: dict[str, str] = {}                                  # name -> ISO3
for _c in pycountry.countries:
    for _n in {_c.name, getattr(_c, "common_name", None), getattr(_c, "official_name", None),
               _cc.convert(names=_c.alpha_3, src="ISO3", to="name_short", not_found=None)} - {None}:
        COUNTRY_NAMES[str(_n)] = _c.alpha_3
COUNTRY_NAMES.update(ALIASES)
COUNTRY_NAMES.update(DEMONYMS)
COUNTRY_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(map(re.escape, COUNTRY_NAMES), key=len, reverse=True))
                        + r")(?![A-Za-z])")

def _fold(s: str) -> str:
    """Without accents, case kept ('Semnān' -> 'Semnan')."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


# subdivision and city names (case-sensitive), indexed by the first word of their key
PLACE_INDEX: dict[str, list[tuple[str, str, re.Pattern]]] = defaultdict(list)
for _iso, _d in list(SUBDIVISIONS.items()) + list(CITIES.items()):
    for _k, _v in _d.items():
        if len(_k) >= 4 and _k not in GENERIC and _k[0].isalpha():       # the full name, first letter capital
            _pat = rf"(?<![A-Za-z]){_k[0].upper()}(?![A-Z])(?i:{re.escape(_k[1:])})(?![A-Za-z])"
            PLACE_INDEX[_k.split()[0]].append((_iso, _k, re.compile(_pat)))


# places inside the names of agreements are not study places
AGREEMENT_RE = re.compile(r"\b(?:Paris|Kyoto|Montreal|Copenhagen|Glasgow|Cancun|Doha|Bali|Durban|Marrakech|Warsaw)\s+"
                          r"(?:Climate\s+)?(?:Agreement|Protocol|Accord|Amendment|Pact|Conference|Summit)\b|\bKaya\s+(?:identity|equation)\b", re.I)


def years_in(text: str) -> set[str]:
    out = {y for y, _ in YEAR_RE.findall(text) if 1900 <= int(y) <= 2100}
    out |= {normalize_time(m.group()) for m in RANGE_RE.finditer(text)} - {""}
    return out


def places_in(text: str) -> tuple[set[str], set[str]]:
    text = AGREEMENT_RE.sub(" ", text)
    countries = {COUNTRY_NAMES[m] for m in COUNTRY_RE.findall(text)}
    folded = _fold(text)
    hits = [(m.start(), m.end(), iso, k) for w in set(re.findall(r"[a-z]+", folded.lower())) & PLACE_INDEX.keys()
            for iso, k, pat in PLACE_INDEX[w] for m in pat.finditer(folded)]
    # a name inside a longer name is not a place ('York' in 'New York')
    places = {f"{iso}:{k}" for a, b, iso, k in hits
              if not any(x <= a and b <= y and y - x > b - a for x, y, *_ in hits)}
    # a place in a country the text names; otherwise keep it only if its name is unique to one country
    by_key = defaultdict(set)
    for p in places:
        iso, k = p.split(":", 1)
        by_key[k].add(iso)
    kept = {f"{iso}:{k}" for k, isos in by_key.items() for iso in isos if iso in countries or len(isos) == 1}
    return countries | {p.split(":")[0] for p in kept}, kept


def main():
    df = pd.read_csv(BASE / "output" / "prepared.csv", usecols=["uid", "text"])
    rows = []
    for uid, text in zip(df.uid, df.text.astype(str)):
        countries, places = places_in(text)
        rows.append({"uid": uid, "years": "|".join(sorted(years_in(text))), "countries": "|".join(sorted(countries)),
                     "places": "|".join(sorted(places))})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "output" / "candidates.csv", index=False, encoding="utf-8-sig")
    print(f"{len(out)} papers | with a year {(out.years != '').mean():.1%} | with a country "
          f"{(out.countries != '').mean():.1%} | with a subdivision or city {(out.places != '').mean():.1%}")


if __name__ == "__main__":
    main()
