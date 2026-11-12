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
from model import (run_al_filter_on_del, create_del_filtered_mask, lightgbm, logistic_regression,
                   pos_weight, rank_combine, rank_mean)
from selection import assign_clusters, cluster_hits, greedy_coverage


def run(library, seed=cfg.SEED, n_pick=cfg.N_PICK, verbose=True):
    """Return a DataFrame with CatalogID, Score (0-1), and Sel_50 for the given library."""
    # Load active learning dataset
    active_learning_full = load_active_learning()
    X_al_full, al_valid = featurize(active_learning_full.smiles, n_jobs=cfg.N_JOBS)
    active_learning = active_learning_full[al_valid].reset_index(drop=True)

    X_al = X_al_full[al_valid]
    y_al = active_learning.label.values.astype(int)

    # Load DEL dataset
    del_full = load_del(library.smiles, seed=seed, verbose=verbose)
    X_del, del_valid = featurize(del_full.smiles, n_jobs=cfg.N_JOBS, verbose=verbose)
    del_rows = del_full[del_valid].reset_index(drop=True)
    
    X_del = X_del[del_valid]
    y_del = del_rows.label.values.astype(int)

    # Filter the DEL dataset by the AL model's confidence
    del_confidence = run_al_filter_on_del(X_al, y_al, X_del, seed=seed)
    filter_mask = create_del_filtered_mask(del_confidence, y_del, p=cfg.AL_FILTER_P)

    # Score every library compound except those already tested in the active-learning round.
    universe = library[
        ~library.catalog_id.isin(active_learning.catalog_id)
    ].reset_index(drop=True)
    X_submission, submission_valid = featurize(universe.smiles, n_jobs=cfg.N_JOBS, verbose=verbose)
    universe = universe[submission_valid].reset_index(drop=True)
    X_submission = X_submission[submission_valid]

    # Train AL-filtered DEL core and active-learning model to rank submission compounds
    filtered_del_lgbm = lightgbm(pos_weight(y_del[filter_mask]), seed=seed).fit(
        X_del[filter_mask], y_del[filter_mask]
    )
    del_lgbm_p = filtered_del_lgbm.predict_proba(X_submission)[:, 1]

    filtered_del_logreg = logistic_regression(seed).fit(X_del[filter_mask], y_del[filter_mask])
    del_logreg_p = filtered_del_logreg.predict_proba(X_submission)[:, 1]

    al_logreg = logistic_regression(seed).fit(X_al, y_al)
    al_logreg_p = al_logreg.predict_proba(X_submission)[:, 1]

    score = rank_combine(
        rank_mean(del_lgbm_p, del_logreg_p), al_logreg_p, alpha=cfg.ALPHA, seed=seed
    )

    # Re-rank submission compounds to improve cluster diversity and coverage
    hit_fingerprints = X_al[y_al == 1]
    hit_clusters, n_clusters = cluster_hits(hit_fingerprints, cfg.T_CLUSTER)
    X_submission_cluster = assign_clusters(X_submission, hit_fingerprints, hit_clusters, cfg.T_CLUSTER)
    picks = greedy_coverage(X_submission_cluster, score, cfg.PMAX, n_pick)
    if verbose:
        print(f"[run] {len(universe):,} scored | {n_clusters} hit-clusters | "
              f"{int((X_submission_cluster >= 0).sum()):,} analogues | {len(picks)} selected", flush=True)

    # The challenge expects a 0-1 Score; rankdata/N is strictly monotonic in the ensemble score,
    # so AUC and the selection are unchanged, only the arbitrary rank-sum scale is removed.
    score_normalized = np.round(rankdata(score) / len(score), 6)
    scored_library = pd.DataFrame(
        {"CatalogID": universe.catalog_id.astype(str), "Score": score_normalized}
    )
    selected = set(universe.iloc[picks].catalog_id.astype(str))
    scored_library["Sel_50"] = scored_library.CatalogID.isin(selected).astype(int)
    return scored_library


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", default=cfg.TEST_SPLIT,
                        help="CSV of compounds to score (CatalogID, SMILES)")
    parser.add_argument("--out", default="submission.csv", help="output CSV path")
    parser.add_argument("--n-pick", type=int, default=cfg.N_PICK, help="number of compounds to select")
    args = parser.parse_args()

    scored_library = run(load_library(args.library), n_pick=args.n_pick)
    scored_library.to_csv(args.out, index=False)
    print(f"[write] {args.out}: {len(scored_library):,} scored, {int(scored_library.Sel_50.sum())} selected")


if __name__ == "__main__":
    main()
