# Greenhouse gas emission estimates from the literature

Code for *A literature-derived dataset of greenhouse gas emission estimates from national to city scales*.

Dataset: https://doi.org/10.6084/m9.figshare.34361370

## Setup

```
pip install -r requirements.txt
```


## Run

```
python pipeline/prepare.py              # WoS exports -> output/prepared.csv
python pipeline/candidates.py           # Step 1: years and places in each text
python eval/run_model.py --backend deepseek --model deepseek-flash --run full --workers 30   # Steps 2-3
python pipeline/build.py --run full     # Step 4 -> emission_records.csv
python pipeline/papers.py --run full    # papers.csv
```

## Validation

```
python eval/score.py --run full         # accuracy on the reference set (eval/gold/)
python eval/refcheck.py --run full      # national totals vs PRIMAP-hist, EDGAR, GCB
```


## Figures

```
python plots/plot_method.py
python plots/plot_spatial.py
python plots/plot_temporal.py
python plots/plot_sector.py
```
