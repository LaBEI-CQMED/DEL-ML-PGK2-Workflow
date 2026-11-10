"""Score a compound library for PGK2 binding and select N picks (submission-1 workflow).

Pipeline:
  1. Train an AL-filtered DEL core (ECFP4 LightGBM + LogisticRegression) on the DEL rows the
     active-learning labels select, with negatives matched to the library being scored.
  2. Rank-combine the DEL core with an in-domain active-learning model.
  3. Assign every compound to its nearest active-learning hit-cluster and greedily pick the N that
     cover the most distinct clusters.

Usage:
  DXC_JOBS=32 python run.py --library PGK2_Test_split.csv --out submission.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config as cfg
from data import load_active_learning, load_del, load_library
from features import featurize
from model import (al_filter_confidence, al_keep_mask, lightgbm, logistic_regression,
                   pos_weight, rank_combine, rank_mean)
from selection import assign_clusters, cluster_hits, greedy_coverage


def run(library, seed=cfg.SEED, n_pick=cfg.N_PICK, verbose=True):
    """Return a DataFrame with CatalogID, Score (0-1), and Sel_50 for the given library."""
    al = load_active_learning()
    X_al, ok = featurize(al.smiles, n_jobs=cfg.N_JOBS)
    al, X_al = al[ok].reset_index(drop=True), X_al[ok]
    y_al = al.label.values.astype(int)

    del_rows = load_del(library.smiles, seed=seed, verbose=verbose)
    X_del, ok = featurize(del_rows.smiles, n_jobs=cfg.N_JOBS, verbose=verbose)
    del_rows, X_del = del_rows[ok].reset_index(drop=True), X_del[ok]
    y_del = del_rows.label.values.astype(int)
    confidence = al_filter_confidence(X_al, y_al, X_del, seed=seed)
    keep = al_keep_mask(confidence, y_del, p=cfg.AL_FILTER_P)

    # Score every library compound except those already tested in the active-learning round.
    universe = library[~library.catalog_id.isin(al.catalog_id)].reset_index(drop=True)
    X_sub, ok = featurize(universe.smiles, n_jobs=cfg.N_JOBS, verbose=verbose)
    universe, X_sub = universe[ok].reset_index(drop=True), X_sub[ok]

    lgbm_p = lightgbm(pos_weight(y_del[keep]), seed=seed).fit(
        X_del[keep], y_del[keep]).predict_proba(X_sub)[:, 1]
    logreg_p = logistic_regression(seed).fit(X_del[keep], y_del[keep]).predict_proba(X_sub)[:, 1]
    al_p = logistic_regression(seed).fit(X_al, y_al).predict_proba(X_sub)[:, 1]
    score = rank_combine(rank_mean(lgbm_p, logreg_p), al_p, alpha=cfg.ALPHA, seed=seed)

    hit_fp = X_al[y_al == 1]
    hit_clusters, n_clusters = cluster_hits(hit_fp, cfg.T_CLUSTER)
    cluster = assign_clusters(X_sub, hit_fp, hit_clusters, cfg.T_CLUSTER)
    picks = greedy_coverage(cluster, score, cfg.PMAX, n_pick)
    if verbose:
        print(f"[run] {len(universe):,} scored | {n_clusters} hit-clusters | "
              f"{int((cluster >= 0).sum()):,} analogues | {len(picks)} selected", flush=True)

    # The challenge expects a 0-1 Score; rankdata/N is strictly monotonic in the ensemble score,
    # so AUC and the selection are unchanged, only the arbitrary rank-sum scale is removed.
    score01 = np.round(rankdata(score) / len(score), 6)
    out = pd.DataFrame({"CatalogID": universe.catalog_id.astype(str), "Score": score01})
    selected = set(universe.iloc[picks].catalog_id.astype(str))
    out["Sel_50"] = out.CatalogID.isin(selected).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default=cfg.TEST_SPLIT,
                        help="CSV of compounds to score (CatalogID, SMILES)")
    parser.add_argument("--out", default="submission.csv", help="output CSV path")
    parser.add_argument("--n-pick", type=int, default=cfg.N_PICK, help="number of compounds to select")
    args = parser.parse_args()

    out = run(load_library(args.library), n_pick=args.n_pick)
    out.to_csv(args.out, index=False)
    print(f"[write] {args.out}: {len(out):,} scored, {int(out.Sel_50.sum())} selected")


if __name__ == "__main__":
    main()
