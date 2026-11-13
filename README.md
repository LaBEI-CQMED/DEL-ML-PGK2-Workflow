# PGK2 DEL-ML Pipeline (Team SABIÁ)

Virtual-screening workflow for the **DREAMxCACHE / Target 2035 DEL-ML Challenge** (target: PGK2), submitted by Team SABIÁ and selected for the prospective phase. Given a compound library, it produces a continuous binding score for every compound and selects a diverse shortlist for experimental testing.

## Method

The workflow keeps **DEL data as the main component** (the challenge's eligibility requirement) while using the released active-learning (AL) labels only to *select* and *rank*, never to relabel DEL rows.

1. **AL-filtered DEL core.** An ECFP4 LightGBM + LogisticRegression ensemble trained on the DEL rows that the AL labels validate: a logistic AL model scores every DEL row, and the least AL-confident DEL positives and most AL-confident DEL negatives are dropped (`both`, 20thpercentile). DEL negatives are sampled to match the chemistry of the library being scored.
2. **Ensemble.** The DEL core (rank-mean of the two learners) is rank-combined with an in-domain AL LogisticRegression, weighted `alpha` toward the AL side. The DEL core carries the rest.
3. **Cluster-coverage selection.** The 39 AL hits are clustered at ECFP4 Tanimoto `t_cluster`. Each library compound is assigned to its nearest hit-cluster (if within `t_pool`), and a greedy selector picks the `N` compounds that cover the most *distinct* clusters, breaking ties by ensemble score.

All models are fit on all available data; the method was validated out-of-sample with leave-one-hit-cluster-out cross-validation (not included here).

### The two test submissions

Both submissions run the same engine; they differ only in the ranker weight and the coverage geometry:

| | `alpha` | `t_cluster` (hit-clustering) | `t_pool` (analogue cutoff) | novelty-as-cluster |
|---|---|---|---|---|
| **Submission 1** | `0.20` | `0.4` | `0.4` | off |
| **Submission 2** | `0.00` | `0.5` | `0.4` | on |

Submission 2 scores with the **pure DEL core** (`alpha = 0`, no AL weight) and clusters the hits more strictly (Tanimoto `0.5`).

## Layout

| File | Role |
|---|---|
| `src/config.py` | data paths (overridable via env vars) and the deployed hyperparameters |
| `src/features.py` | ECFP4 fingerprints and Tanimoto similarity |
| `src/data.py` | load the AL labels, the DEL table, and a library to score |
| `src/model.py` | DEL/AL learners, the AL row-selection filter, the rank-combination |
| `src/selection.py` | hit clustering and greedy cluster-coverage selection |
| `run.py` | end-to-end entry point |
| `tests/` | reproduction check against the committed submissions 1 and 2 |

## Data

The challenge data is not redistributed here. Point the pipeline at the files (available from the
[challenge Synapse project](https://www.synapse.org/Synapse:syn75349604/wiki/)) either by placing
them in `./data/` or via environment variables.

## Usage

```bash
$ pip install -r requirements.txt

# Score a library and select 50 compounds with the submission-1 configuration (default).
$ python run.py --submission 1 --library my_library.csv --out submission_1.csv

# The submission-2 configuration
$ python run.py --submission 2 --library my_library.csv --out submission_2.csv
```

## Reproducing the submissions

With the data configured, this retrains everything and asserts the output matches the committed submissions exactly.

```bash
$ python tests/reproduce_check.py        # both submissions
$ python tests/reproduce_check.py 2      # only submission 2
```

## Team SABIÁ - Structural and Artificial Intelligence Biology Alliance

We are a team of Brazilian researchers passionate about advancing the integration of machine learning (ML) and artificial intelligence (AI) into drug discovery. We see PGK2 as an ideal case study for exploring how computational strategies can improve the identification of novel molecular binders, particularly through the integration of experimental data, alternative molecular representations, and knowledge transfer across chemical spaces and biological targets.

**People**
- Arthur Cerveira
- João Pedro Aroucha
- Raquel Barbosa
- Renata Barros
- Matheus Dalmolin
- Marcelo Fernandes
- Philipe Fernandes
- Frederico Kremer
- Karina Machado
- Vinicius Maltarollo
- Giovanna Assunção Pereira Soares
- Gabriel Corrêa Veríssimo
- Adriano Velasque Werhli
- Rafael Junqueira Borges

**Institutions**
- Universidade Estadual de Campinas (UNICAMP)
- Universidade Federal do Rio Grande (FURG)
- Universidade Federal de Pelotas (UFPel)
- Universidade Federal do Rio Grande do Norte (UFRN)
- Universidade Federal de Minas Gerais (UFMG)
