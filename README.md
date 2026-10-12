# Greenhouse gas emission estimates from the literature

Code for *A literature-derived dataset of greenhouse gas emission estimates from national to city scales*.

Dataset: https://doi.org/10.6084/m9.figshare.34361370

## Setup

```
pip install -r requirements.txt
```

Steps 2 and 3 call Gemini and read the key from `GEMINI_API_KEY`.

Not included: Web of Science exports (`data/raw/*.xls`) and the reference inventories in `data/external/`
(OWID CO2 data, PRIMAP-hist v2.7, EDGAR 2025, geoBoundaries CGAZ).

## Run

```
python pipeline/prepare.py              # WoS exports -> output/prepared.csv
python pipeline/candidates.py           # Step 1: years and places in each text
python pipeline/extract.py --run full   # Step 2: LLM extraction
python pipeline/verify.py --run full    # Step 3: LLM review
python pipeline/build.py --run full     # Step 4: checks and filters -> emission_records.csv
python pipeline/papers.py --run full    # papers.csv
```

## Validation

```
python eval/score.py --run full         # accuracy on the reference set in eval/gold/
python eval/refcheck.py --run full      # national totals vs PRIMAP-hist, EDGAR, GCB
```

## Figures

```
python plots/plot_spatial.py
python plots/plot_temporal.py
python plots/plot_sector.py
```
