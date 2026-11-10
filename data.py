"""Load the active-learning labels, the DEL training table, and a library to score."""
import pandas as pd

from config import ACTIVE_LEARNING, DEL_TABLE, NEG_RATIO, N_JOBS, SEED, TEST_SPLIT
from features import featurize, tanimoto_max


def load_active_learning():
    """The 6,580 experimentally tested active-learning compounds (39 hits)."""
    df = pd.read_csv(ACTIVE_LEARNING)
    return df.rename(columns={"CatalogID": "catalog_id", "RDKit_SMILES": "smiles",
                              "Label": "label"})[["catalog_id", "smiles", "label"]]


def load_library(path=TEST_SPLIT):
    """A library of compounds to score: any CSV with CatalogID and SMILES columns."""
    df = pd.read_csv(path)
    return df.rename(columns={"CatalogID": "catalog_id",
                              "SMILES": "smiles"})[["catalog_id", "smiles"]]


def load_del(reference_smiles, neg_ratio=NEG_RATIO, pool_size=1_000_000, n_ref=2000,
             seed=SEED, verbose=True):
    """All DEL positives plus negatives chosen most ECFP-similar to ``reference_smiles``.

    Negatives are sampled to match the chemistry we deploy on (the scored library) rather than
    the DEL library's own bulk; this is the ``similar`` sampler that won the baseline comparison.
    """
    df = pd.read_parquet(DEL_TABLE)
    positives = df[df.label == 1]
    negatives = df[df.label == 0]

    reference = reference_smiles.sample(n_ref, random_state=seed)
    X_ref, ok = featurize(reference, n_jobs=N_JOBS)
    X_ref = X_ref[ok]

    candidates = negatives.sample(min(pool_size, len(negatives)), random_state=seed)
    X_cand, ok_cand = featurize(candidates.smiles, n_jobs=N_JOBS, verbose=verbose)
    candidates = candidates[ok_cand]
    similarity = tanimoto_max(X_cand[ok_cand], X_ref)
    n_neg = min(int(neg_ratio * len(positives)), len(candidates))
    candidates = candidates.assign(sim=similarity).nlargest(n_neg, "sim")

    out = pd.concat([positives, candidates.drop(columns="sim")])[["smiles", "label"]]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
