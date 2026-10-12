# Greenhouse gas emission estimates from the literature

Code for *A literature-derived dataset of greenhouse gas emission estimates from national to city scales*.

Dataset: https://doi.org/10.6084/m9.figshare.34361370 (`emission_records.csv`, `papers.csv`; CC BY 4.0).

## Setup

```
pip install -r requirements.txt
```

Inputs that are not in this repository:

| Path | Content |
|---|---|
| `data/raw/*.xls` | Web of Science "Full Record" exports of the search in Supplementary Note 1 (not redistributable) |
| `data/external/owid-co2-data.csv` | Our World in Data CO2 dataset (github.com/owid/co2-data); Global Carbon Budget national CO2 |
| `data/external/primap-hist/PRIMAP-hist_v2.7_final_no_extrap.csv` | PRIMAP-hist v2.7 (doi:10.5281/zenodo.17090760) |
| `data/external/edgar2025/*.xlsx` | EDGAR_2025_GHG, totals by country (edgar.jrc.ec.europa.eu) |
| `data/external/geoBoundaries/geoBoundariesCGAZ_ADM{0,1}.shp` | geoBoundaries CGAZ boundaries (figures only) |

## Run

```
python pipeline/prepare.py                  # WoS exports -> output/prepared.csv
python pipeline/candidates.py               # Step 1: years and places mentioned in each text
python pipeline/extract.py --run full       # Step 2: LLM extraction
python pipeline/verify.py --run full        # Step 3: LLM review
python pipeline/build.py --run full         # Step 4: checks, normalisation, filters -> emission_records.csv
python pipeline/papers.py --run full        # papers.csv
```

Steps 2 and 3 resume where they stopped when re-run. `--uids-file eval/gold/uids.txt` runs Step 2 on the
reference papers only.

| File | Does |
|---|---|
| `pipeline/candidates.py` | Rules: years, periods, countries, subdivisions and cities named in the text (not shown to the model) |
| `pipeline/extract.py` | Step 2 prompt, schema and calls |
| `pipeline/verify.py` | Step 3 prompt, schema and calls |
| `pipeline/build.py` | Step 4: value parsing, geocoding, total flag, checks against the text, filters |
| `pipeline/places.py`, `values.py`, `units.py`, `sectors.py` | Geocoding, value and year parsing, unit conversion, sector groups |

## Validation

```
python eval/score.py --run full   # accuracy on the reference set (Table 4)
python eval/refcheck.py --run full               # national totals vs PRIMAP-hist, EDGAR, GCB (Table 5)
```

`eval/gold/` holds the reference set: `uids.txt` (100 randomly drawn papers) and `gold.psv` (one row per
emission value; alternatives separated by `;`). Matching and scoring rules are in Supplementary Table 2.

## Figures

```
python plots/plot_spatial.py     # Fig. 2
python plots/plot_temporal.py    # Fig. 3
python plots/plot_sector.py      # Fig. 4
```
