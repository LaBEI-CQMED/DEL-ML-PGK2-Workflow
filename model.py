"""The DEL core learners, the AL row-selection filter, and the rank-combination.

The active-learning labels only *select* and *rank* here; they never relabel DEL rows, so the
DEL model stays a pure DEL model (the challenge's eligibility requirement).
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from config import N_JOBS, SEED


def logistic_regression(seed=SEED):
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                              solver="liblinear", random_state=seed)


def lightgbm(scale_pos_weight, seed=SEED, n_jobs=N_JOBS):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.4, scale_pos_weight=scale_pos_weight,
                          n_jobs=n_jobs, random_state=seed, verbose=-1)


def pos_weight(y):
    """Return the ratio of negative to positive rows parameter used by LightGBM"""
    y = np.asarray(y)
    n_pos = max(int(y.sum()), 1)
    return float((len(y) - n_pos) / n_pos)


def run_al_filter_on_del(X_al, y_al, X_del, seed=SEED):
    """AL-model P(active) for each DEL row, which will be used to filter the DEL dataset."""
    al_filter_model = logistic_regression(seed).fit(X_al, y_al)
    return al_filter_model.predict_proba(X_del)[:, 1]


def create_del_filtered_mask(confidence, y_del, p):
    """Keep every DEL row except the bottom-p positives and top-p negatives by AL confidence
    (the likely false positives the AL model rejects and false negatives it flags).
    Returns a boolean mask of the same length as y_del, True = keep, False = filter out."""
    y_del = np.asarray(y_del).astype(int)
    confidence = np.asarray(confidence, dtype=float)
    
    positives = np.where(y_del == 1)[0]
    negatives = np.where(y_del == 0)[0]
    
    filter_mask = np.ones(len(y_del), dtype=bool)
    inconsistent_positives = positives[confidence[positives] <= np.quantile(confidence[positives], p)]
    inconsistent_negatives = negatives[confidence[negatives] >= np.quantile(confidence[negatives], 1 - p)]
    inconsistent_rows = np.concatenate([inconsistent_positives, inconsistent_negatives])

    filter_mask[inconsistent_rows] = False
    return filter_mask


def rank_mean(*scores):
    """Mean of the per-column ranks; combines rankings on a common scale."""
    return np.mean([rankdata(s) for s in scores], axis=0)


def _rank(score, rng):
    """Rank per row, 0 = best. Ties are broken at random so that many identical scores do not
    fall back to input order."""
    score = np.asarray(score, dtype=float)
    order = np.lexsort((rng.random(len(score)), -score))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def rank_combine(del_score, al_score, alpha, seed=SEED):
    """Blend a DEL ranking and an in-domain AL ranking into one higher-is-better score;
    ``alpha`` weights the AL side."""
    rng = np.random.default_rng(seed)
    return -(alpha * _rank(al_score, rng) + (1 - alpha) * _rank(del_score, rng))
