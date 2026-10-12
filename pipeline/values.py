"""
Value, number and year helpers shared by build.py (value parser and year normalization).
"""
import re

import numpy as np
import pandas as pd

YEAR_MIN, YEAR_MAX = 1900, 2100
DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-", "’": "'", "“": '"', "”": '"'})
EXP_RE = re.compile(r"(?:x|×|\*)\s*10\s*\^?\s*\(?[-]?\d+\)?|(?<=\d)e[+-]?\d+|\(\s*-\s*\d+\s*\)", re.I)
NUM_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")          # skips digits inside CO2, N2O, PM2.5
MULT_RE = re.compile(r"\b(?:thousand|million|billion|trillion)\b", re.I)
# multiplier prefix of a tonne-based unit: Mt, MtCO2, MTons, Mtoe, Mtce, Gt, kt, Tg, Gg ...
MASS_PREFIX_RE = re.compile(r"^\s*(?:[MGk]|Mm|MM|mm|m|K)(?=[tT](?![a-z])|[tT](?:oe|ce|ons?|onnes?)\b)|^\s*[TGP]g(?![a-z])")
MULT_ANY_RE = re.compile(r"(?<![a-z])(?:thousand|million|billion|trillion)\b", re.I)   # also '17973.86million'
YEAR_RE = re.compile(r"(?<!\d)((?:19\d{2}|20\d{2}|2100))(s?)(?!\d)")
RANGE_RE = re.compile(
    r"(?<!\d)((?:19\d{2}|20\d{2}|2100))s?\s*(?:-|/|to|through|until|till)\s*((?:19\d{2}|20\d{2}|2100)|\d{2})(s?)(?!\d)"
    r"|between\s+((?:19\d{2}|20\d{2}|2100))\s+and\s+((?:19\d{2}|20\d{2}|2100))", re.I)


MULTIPLIERS = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "mln": 1e6, "thousands": 1e3, "thousand": 1e3,
               "thous": 1e3}
EMBEDDED_UNIT_RE = re.compile(
    r"\s+(?:Gt|Mt|Mton|kt|Gg|Tg|Pg|tons?|tonnes?|CO2|GHG|C|eq|yr|year|/yr|/year|%)\b.*$", re.I)
QUALIFIER_RE = re.compile(
    r"^\s*(?:in\s+between|about|approximately|approximate\s+to|around|nearly|roughly|almost|estimated|similar\s+to|between|"
    r"close\s+to|ca\.?|"
    r"not\s+(?:greater|more)\s+than|no\s+more\s+than|at\s+most|more\s+than\s+an?|"
    r"more\s+than|less\s+than|greater\s+than|up\s+to|at\s+least|exceeding|"
    r"over|under|above|below|>=|<=|>|<|~)\s*", re.I)
SKIP_RE = re.compile(
    r"^\s*(?:by\s+\d{4}|before\s+\d{4}|after\s+\d{4}|unknown|n/?a|not\s+(?:available|specified|reported))\s*$",
    re.I)
# qualifier words -> symbol; longer phrases first
QUALIFIER_SYMBOLS = [
    (r">=|≥|at\s+least|no\s+less\s+than", ">="),
    (r"<=|≤|up\s+to|at\s+most|not\s+(?:greater|more)\s+than|no\s+more\s+than", "<="),
    (r">|more\s+than|greater\s+than|over|above|exceed(?:s|ing)?", ">"),
    (r"<|less\s+than|under|below", "<"),
    (r"~|about|approximate|around|nearly|roughly|almost|some|ca\.?\s|close\s+to|similar\s+to", "~"),
]
NAN3 = (np.nan, np.nan, np.nan)


def value_qualifier(raw) -> str:
    s = str(raw).strip().lower()
    for pat, sym in QUALIFIER_SYMBOLS:
        if re.match(rf"^\s*(?:{pat})", s):
            return sym
    return ""


def _float(s: str) -> float:
    try:
        return float(s.strip().replace(",", ""))
    except ValueError:
        return np.nan


def parse_value(raw) -> tuple[float, float, float]:
    """(value, low, high); low == high == value unless raw is a range; NaNs if unparseable."""
    if pd.isna(raw):
        return NAN3
    s = str(raw).strip()
    if SKIP_RE.match(s):
        return NAN3
    s = re.sub(r"&\s*times;", "×", s)                                             # HTML '&times;'
    s = re.sub(r"\s*%?\s*(?:95\s*%\s*)?ci\s*[\[(].*$", "", s, flags=re.I)          # '12% CI [10-14]'
    s = re.sub(r"^(?:US\$|USD|EUR|[$€£])\s*", "", s, flags=re.I)                 # currency
    s = EMBEDDED_UNIT_RE.sub("", s).strip()                                        # '10 Mt'
    s = re.sub(r"(?<=\d)(?:t|kg|g|kt|mt|gt)(?:/.*)?$", "", s, flags=re.I).strip()  # '9.8t/...'
    s = re.sub(r"%\s*$", "", s).strip()
    s = QUALIFIER_RE.sub("", s).strip()
    s = re.sub(r"^\+", "", s).strip()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)                                        # '2,599,551'
    s = re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)                                      # '752 465'
    s = re.sub(r"^\[(-?[\d.]+),\s*(-?[\d.]+)\]$", lambda m: f"{m.group(1)}-{m.group(2)}", s)

    exp = 0                                                                        # power of ten
    m = re.search(r"\s*[x×*]\s*10\s*\^?\s*\(?\s*([-−]?\d{1,2})\s*\)?\s*$", s)
    if m:
        exp = int(m.group(1).replace("−", "-"))
        s = s[:m.start()].strip()
    s = re.sub(r"(?<=\d)\s+and\s+(?=-?\d)", "-", s)                                # 'A and B'

    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", s)                                      # '2/3'
    if m and int(m.group(2)):
        return (int(m.group(1)) / int(m.group(2)),) * 3

    mult = 1.0
    for word, factor in MULTIPLIERS.items():
        pat = re.compile(rf"(?<![a-z]){word}\b\.?", re.I)                          # also '17973.86million', 'thous.'
        if pat.search(s):
            s, mult = pat.sub("", s).strip(), factor
            break
    mult *= 10.0 ** exp

    s = re.sub(r"\s*\(.*?\)", "", s).strip()                                       # '12 (10-14)', '12 (+/- 2)'
    s = re.sub(r"\s*(?:±|\+\s*/\s*-|\+-)\s*\d+[\d.]*.*$", "", s).strip()          # '12 +/- 2' (not 'e+06')
    s = re.sub(r"^(-?[\d.]+)%-(-?[\d.]+)%?$", r"\1-\2", s)                         # '60.6%-89.3%'

    m = re.match(r"^(-?[\d.]+(?:e[+-]?\d+)?)\s*(?:–|-|to|~)\s*(-?[\d.]+(?:e[+-]?\d+)?)\s*$", s, re.I)
    if m:
        lo, hi = _float(m.group(1)), _float(m.group(2))
        if not (np.isnan(lo) or np.isnan(hi)):
            return (lo + hi) / 2 * mult, lo * mult, hi * mult
    if re.match(r"^-?[\d.]+(?:e[+-]?\d+)?$", s, re.I):
        v = _float(s)
        return (v * mult,) * 3 if not np.isnan(v) else NAN3
    # a number followed only by words: '3 kg', '25 percent', '23 reduction', '10-fold'
    m = re.match(r"^(-?[\d.]+)\s*-?\s*[a-z][a-z\s.()/%-]*$", s, re.I)
    # or a number followed by a phrase: '7.70 of total CH4 of entire Alberta' (unit % removed)
    m = m or re.match(r"^(-?[\d.]+)\s*%?\s+(?:of|in|for|to)\s+[a-z]", s, re.I)
    if m:
        v = _float(m.group(1))
        return (v * mult,) * 3 if not np.isnan(v) else NAN3
    return NAN3


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).translate(DASHES).lower()).strip()


def nums(s) -> list[str]:
    return NUM_RE.findall(EXP_RE.sub(" ", str(s).replace(",", "")))


def in_text(needle: str, hay: str) -> bool:
    """Normalized containment, else >= 90% of the needle's words present."""
    n, h = norm(needle), norm(hay)
    if not n or n in h:
        return True
    words = re.findall(r"[a-z0-9.]+", n)
    hay_words = set(re.findall(r"[a-z0-9.]+", h))
    return bool(words) and sum(w in hay_words for w in words) / len(words) >= 0.9


def numbers_in(value_text: str, hay: str) -> bool:
    plain = norm(hay).replace(",", "")
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d)", plain) for n in nums(value_text))


def span(start: int, end: int) -> str:
    if not (YEAR_MIN <= start <= end <= YEAR_MAX):
        return ""
    return str(start) if start == end else f"{start}-{end}"


def normalize_time(t) -> str:
    """'in 2019' / 'from 2000 to 2013' / '2009-10' / '2030s' -> 'YYYY' | 'YYYY-YYYY' | ''."""
    s = norm(t)
    m = RANGE_RE.search(s)
    if m:
        if m.group(4):                                     # 'between 2000 and 2013'
            return span(int(m.group(4)), int(m.group(5)))
        a, b = int(m.group(1)), m.group(2)
        end = a // 100 * 100 + int(b) if len(b) == 2 else int(b)
        if len(b) == 2 and end < a:                        # '1999-00'
            end += 100
        if m.group(3):                                     # '1970-1990s'
            end += 9
        return span(a, end)
    years = YEAR_RE.findall(s)
    if len(years) == 1:
        y, decade = int(years[0][0]), years[0][1]
        return span(y, y + 9 if decade else y)
    return ""
