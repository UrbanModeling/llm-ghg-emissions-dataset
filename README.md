# Greenhouse gas emission estimates from the literature

Code for *A literature-derived dataset of greenhouse gas emission estimates from national to city scales*.

Dataset: https://doi.org/10.6084/m9.figshare.34361370

## Setup

```
pip install -r requirements.txt
```

The dataset was built with DeepSeek V4.1 Flash through the DeepSeek API (`eval/run_model.py --backend deepseek`),
which reads the key from `DEEPSEEK_API_KEY` (environment or a `.env` file). `pipeline/extract.py` and
`pipeline/verify.py` hold the prompts and schemas and, run on their own, call Gemini 2.5 Flash Lite
(`GEMINI_API_KEY`), the model of the first version of the dataset (tag `gemini-2.5-flash-lite`).

Not included: Web of Science exports (`data/raw/*.xls`) and the reference inventories in `data/external/`
(OWID CO2 data, PRIMAP-hist v2.7, EDGAR 2025, geoBoundaries CGAZ).

## Run

```
python pipeline/prepare.py              # WoS exports -> output/prepared.csv
python pipeline/candidates.py           # Step 1: years and places in each text
python eval/run_model.py --backend deepseek --model deepseek-flash --run full --workers 30
                                        # Steps 2-3: LLM extraction and review
python pipeline/build.py --run full     # Step 4: checks and filters -> emission_records.csv
python pipeline/papers.py --run full    # papers.csv
```

## Validation

```
python eval/score.py --run full         # accuracy on the reference set in eval/gold/
python eval/refcheck.py --run full      # national totals vs PRIMAP-hist, EDGAR, GCB
```

Model comparison (Table 6): Steps 2-3 on the reference papers with other models, e.g.

```
python eval/run_model.py --backend ollama --model gemma4:31b --run b_gm31 --uids-file eval/gold/uids.txt
python pipeline/build.py --run b_gm31
python eval/score.py --run b_gm31
```

Backends: `deepseek`, `zhipu` (GLM), `mimo` (MiMo), `ollama` (Ollama Cloud), `openrouter` and `gemini`, with keys
`DEEPSEEK_API_KEY`, `ZHIPU_API_KEY`, `MIMO_API_KEY`, `OLLAMA_API_KEY`, `OPENROUTER_API_KEY` and `GEMINI_API_KEY`.

## Figures

```
python plots/plot_method.py
python plots/plot_spatial.py
python plots/plot_temporal.py
python plots/plot_sector.py
```
