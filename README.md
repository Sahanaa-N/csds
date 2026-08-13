# Case Studies in Data Science — Individual Task 1

Sahana Nataraja (s4164152) — RMIT University, 2026

Analysis code for Part 1.3 of Individual Task 1: applying two machine learning algorithms to two
public datasets relevant to a Data Scientist role in pricing and commercial analytics.

## Datasets

Neither dataset is committed to this repository. The notebook downloads both on first run.

| | Source | Size | Task |
|---|---|---|---|
| A | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 1,067,371 transactions, 8 attributes | Weekly product demand (regression) |
| B | [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | 12,330 sessions, 17 attributes | Session conversion (classification) |

## Models

Random Forest and histogram-based Gradient Boosting (scikit-learn), applied to each dataset.
Both differ from models used in previous coursework.

## Metrics

- **Dataset A** — MAE, RMSE, WAPE against a naive lag-1 baseline. MAPE was rejected because
  roughly a fifth of the panel is zero-demand, which makes it undefined or explosive.
- **Dataset B** — PR-AUC, ROC-AUC, F1. Accuracy is deliberately not reported: at a 15.5%
  conversion rate, always predicting "no purchase" scores about 85% and is commercially worthless.

## Results

Dataset A (held-out weeks, lower is better):

| Model | MAE | RMSE | WAPE |
|---|---|---|---|
| Naive (lag-1) | 129.10 | 1103.70 | 0.690 |
| Random Forest | 113.58 | 1094.83 | 0.607 |
| Gradient Boosting | **111.24** | 1095.15 | **0.595** |

Dataset B (held-out sessions, higher is better):

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | **0.721** | 0.922 | **0.655** |
| Gradient Boosting | 0.701 | 0.921 | 0.649 |
| Always "no purchase" | 0.155 | 0.500 | 0.000 |

## Notes on validity

- The Dataset A split is **temporal**, not random — a random split would leak future demand
  into the past.
- Zero-sale weeks are retained rather than dropped; a week with no sales is a real observation.
- The invoice count is **lagged**. An earlier version used the same-week count, which is
  contemporaneous with the target and leaked; it inflated results substantially before being caught.
- Weekly mean realised price is contemporaneous with the target. It is retained deliberately,
  since in a pricing context price is a decision variable set in advance. This is an assumption,
  not a neutral choice.
- In Dataset B, `PageValues` derives from pages preceding completed transactions and therefore
  partly encodes the target. It dominates permutation importance and a deployment-grade model
  would need re-evaluation without it.

## Running

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
jupyter notebook Part1_3_Analysis.ipynb
```

Run cells top to bottom. Outputs land in `figures/` and `results_dataset_*.csv`.

## Files

- `Part1_3_Analysis.ipynb` — annotated notebook (primary artefact)
- `analysis.py` — same pipeline as a standalone script
