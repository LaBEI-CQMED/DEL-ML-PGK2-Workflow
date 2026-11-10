# PGK2 DEL-ML Pipeline (Team SABIA)

Virtual-screening workflow for the **Target 2035 / CACHE DEL-ML Challenge** (target: PGK2),
submitted by Team SABIA and selected for the prospective phase. Given a compound library, it
produces a continuous binding score for every compound and selects a diverse shortlist for
experimental testing.

This repository is the open-source release of our proposed solution (submission 1).

## Method

The workflow keeps **DEL data load-bearing** (the challenge's eligibility requirement) while using
the released active-learning (AL) labels only to *select* and *rank* — never to relabel DEL rows.

1. **AL-filtered DEL core.** An ECFP4 LightGBM + LogisticRegression ensemble trained on the DEL
   rows that the AL labels endorse: a logistic AL model scores every DEL row, and the least
   AL-confident DEL positives and most AL-confident DEL negatives are dropped (`both`, 20th
   percentile). DEL negatives are sampled to match the chemistry of the library being scored.
2. **Ensemble.** The DEL core (rank-mean of the two learners) is rank-combined with an in-domain
   AL LogisticRegression, weighted `alpha = 0.2` toward the AL side. The DEL core carries the rest.
3. **Cluster-coverage selection.** The 39 AL hits are clustered at ECFP4 Tanimoto 0.4. Each library
   compound is assigned to its nearest hit-cluster (if within 0.4), and a greedy selector picks the
   `N` compounds that cover the most *distinct* clusters, breaking ties by ensemble score.

All models are fit on all available data; the method was validated out-of-sample with
leave-one-hit-cluster-out cross-validation (not included here).

## Layout

| File | Role |
|---|---|
| `config.py` | data paths (overridable via env vars) and the deployed hyperparameters |
| `features.py` | ECFP4 fingerprints and Tanimoto similarity |
| `data.py` | load the AL labels, the DEL table, and a library to score |
| `model.py` | DEL/AL learners, the AL row-selection filter, the rank-combination |
| `selection.py` | hit clustering and greedy cluster-coverage selection |
| `run.py` | end-to-end entry point |
| `tests/` | reproduction check against the committed submission 1 |

## Data

The challenge data is not redistributed here. Point the pipeline at the files (available from the
[challenge Synapse project](https://www.synapse.org/Synapse:syn75349604/wiki/)) either by placing
them in `./data/` or via environment variables:

| Variable | File | Contents |
|---|---|---|
| `DXC_DEL_TABLE` | `del_compounds.parquet` | DEL selection table, aggregated per SMILES, columns `smiles`, `label` |
| `DXC_ACTIVE_LEARNING` | `active_learning_data.csv` | 6,580 tested compounds, columns `CatalogID`, `RDKit_SMILES`, `Label` |
| `DXC_TEST_SPLIT` | `PGK2_Test_split.csv` | default library to score, columns `CatalogID`, `SMILES` |

## Usage

```bash
pip install -r requirements.txt

# Score a library and select 50 compounds.
DXC_JOBS=32 python run.py --library my_library.csv --out submission.csv
```

`--library` is any CSV with `CatalogID` and `SMILES` columns; compounds already present in the AL
set are excluded from scoring. `DXC_JOBS` caps worker threads (default 32).

## Reproducing submission 1

With the data configured, this retrains everything and asserts the output matches the committed
submission (`tests/submission_1_reference.csv`, 178,052 rows):

```bash
DXC_JOBS=32 python tests/reproduce_check.py
```
