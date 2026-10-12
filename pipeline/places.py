"""
Geocode the model's (country, place) pair.

country -> ISO3 with country_converter. place is looked up only among that country's first-level
subdivisions (pycountry, plus English names of Chinese provinces) and cities with >= 100k people
(geonamescache): the exact name first, else the longest such name inside the place text.
A name that is both a subdivision and a city is a city if the city has >= 500k people, unless the
place says 'province', 'state', 'region' ...
A place that matches nothing is kept as written (spatial_level 'local'); a place without a capital
letter ('urban areas') is not a name and is ignored.
"""
import logging
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

import country_converter as coco
import geonamescache
import pycountry

logging.getLogger("country_converter").setLevel(logging.ERROR)
_cc = coco.CountryConverter()
MIN_CITY_POP, BIG_CITY_POP = 100_000, 500_000
NONE = {"", "none", "global", "world", "n/a", "several", "multiple"}
_SUFFIX_RE = re.compile(r"\s+(sheng|shi|sar|zizhiqu|province|provincial|state|region|prefecture|oblast|krai|"
                        r"autonomous region|governorate|municipality|city|metropolitan area)$", re.I)
EXTRA_ADM1 = {"CHN": ["Inner Mongolia", "Tibet", "Ningxia", "Xinjiang", "Guangxi"]}



def _key(name: str) -> str:
    """Lower case without accents ('Semnān' -> 'semnan')."""
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().strip()


SUBDIVISIONS: dict[str, dict[str, str]] = defaultdict(dict)       # iso3 -> {key: name}
for _s in pycountry.subdivisions:
    _c = pycountry.countries.get(alpha_2=_s.country_code)
    if _c and _s.parent_code is None:
        for _n in {_s.name, _SUFFIX_RE.sub("", _s.name)}:
            SUBDIVISIONS[_c.alpha_3][_key(_n)] = _SUFFIX_RE.sub("", _s.name)
for _iso, _names in EXTRA_ADM1.items():
    SUBDIVISIONS[_iso].update({_key(n): n for n in _names})

CITIES: dict[str, dict[str, dict]] = defaultdict(dict)            # iso3 -> {key: city}
for _v in geonamescache.GeonamesCache().get_cities().values():
    _c = pycountry.countries.get(alpha_2=_v["countrycode"])
    if _c and _v["population"] >= MIN_CITY_POP:
        _k = _key(_v["name"])
        if _k not in CITIES[_c.alpha_3] or _v["population"] > CITIES[_c.alpha_3][_k]["population"]:
            CITIES[_c.alpha_3][_k] = _v


def country_iso3(name: str) -> str:
    name = str(name).strip()
    if name.lower() in NONE:
        return ""
    iso3 = _cc.convert(names=name, to="ISO3", not_found=None)
    return iso3 if isinstance(iso3, str) and len(iso3) == 3 else ""


def _lookup(iso3: str, name: str, prefer_sub: bool = False) -> tuple[str, dict | None]:
    sub = SUBDIVISIONS[iso3].get(name)
    city = CITIES[iso3].get(name)
    if city and (not sub or (city["population"] >= BIG_CITY_POP and not prefer_sub)):
        return "", city
    return sub or "", None


def _is_country(name: str) -> bool:
    """A country's own name or abbreviation ('UK', 'mainland China'), not a part of it ('northeast China')."""
    name = re.sub(r"^(?:the|mainland|whole|entire)\s+", "", name.strip(), flags=re.I)
    iso3 = country_iso3(name)
    if not iso3:
        return False
    if re.fullmatch(r"[A-Z.]{2,5}", name):                           # UK, USA, U.S., PRC, UAE
        return True
    own = {_key(_cc.convert(names=iso3, src="ISO3", to=t)) for t in ("name_short", "name_official")}
    if _key(name) in own:                                            # 'South Korea', 'Russian Federation'
        return True
    return not re.search(r"\b(?:north|south|east|west|central|inner|outer|upper|lower|triangle|region|delta|"
                         r"coast|basin|province|area|part)", name, re.I)


def country_of_place(place: str) -> str:
    """ISO3 for a place named without its country: a country ('Hong Kong'), a subdivision or city
    name found in one country only ('Mato Grosso'), else a country name inside it ('South China')."""
    p = str(place).split(",")[0].strip()
    if not re.search(r"[A-Z]", p):
        return ""
    try:
        return pycountry.countries.lookup(p).alpha_3
    except LookupError:
        pass
    k = _key(_SUFFIX_RE.sub("", p))
    hits = {c for c, d in SUBDIVISIONS.items() if k in d} | {c for c, d in CITIES.items() if k in d}
    if hits:
        return hits.pop() if len(hits) == 1 else ""
    return country_iso3(p)


@lru_cache(maxsize=None)
def geocode(country: str, place: str) -> dict:
    """-> country, country_iso3, subnational, city, lat, lon, spatial_level, place_how."""
    iso3 = country_iso3(country) or (country_of_place(place) if str(country).strip().lower() in NONE else "")
    out = dict(country="", country_iso3=iso3, subnational="", city="", lat=None, lon=None,
               spatial_level="none", place_how="")
    if not iso3:
        return out
    out.update(country=_cc.convert(names=iso3, src="ISO3", to="name_short"), spatial_level="country")
    p = re.sub(r"^the\s+", "", str(place).strip(), flags=re.I)
    if (p.lower() in NONE or not re.search(r"[A-Z]", p)          # 'urban areas' is not a name
            or p.lower() == out["country"].lower() or country_of_place(p) == iso3 and _is_country(p)):
        return out
    first = p.split(",")[0].strip()
    prefer_sub = bool(re.search(r"\b(province|state|region|oblast|governorate)$", first, re.I))
    for key in (_key(first), _key(_SUFFIX_RE.sub("", first))):     # 'Mexico City' before 'Mexico'
        sub, city = _lookup(iso3, key, prefer_sub)
        if sub or city:
            break
    how = "exact"
    if not (sub or city):                                           # longest known name inside the text
        names = sorted(set(SUBDIVISIONS[iso3]) | set(CITIES[iso3]), key=lambda n: (-len(n), n))  # name breaks ties
        hit = next((n for n in names if len(n) >= 4 and re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", key)),
                   None)
        if hit:
            sub, city = _lookup(iso3, hit, prefer_sub)
            how = "contained"
    if city:
        out.update(city=city["name"], lat=city["latitude"], lon=city["longitude"], spatial_level="city",
                   place_how=how)
    elif sub:
        out.update(subnational=sub, spatial_level="subnational", place_how=how)
    else:
        out.update(spatial_level="local", place_how="kept")
    return out
