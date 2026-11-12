"""Hit clustering and greedy cluster-coverage selection.

The oracle's primary metric rewards covering distinct hit series, so the selector maximises the
number of hit-clusters its picks fall into rather than raw score.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

from features import tanimoto_matrix, tanimoto_nearest


def cluster_hits(X_hits, t):
    """Connected components of the hits at Tanimoto >= t -> (cluster label per hit, count)."""
    adjacency = (tanimoto_matrix(X_hits, X_hits) >= t).astype(int)
    n_clusters, labels = connected_components(csr_matrix(adjacency), directed=False)
    return labels, n_clusters


def assign_clusters(X, X_hits, hit_clusters, t):
    """Assign each row of X to its nearest hit's cluster, or -1 if no hit within Tanimoto t."""
    similarity, nearest = tanimoto_nearest(X, X_hits)
    return np.where(similarity >= t, hit_clusters[nearest], -1)


def greedy_coverage(cluster, score, pmax, n_pick):
    """Greedily pick n_pick rows to maximise expected distinct-cluster coverage.

    Each candidate contributes p = pmax * score_percentile, discounted by how much of its cluster
    is still uncovered: the first pick in a cluster earns the full p, later picks earn less, so the
    objective rewards breadth across clusters before depth within any one.
    """
    cluster = np.asarray(cluster)
    p = pmax * (rankdata(np.asarray(score, dtype=float)) / len(score))
    survival = dict()  # cluster -> fraction of that cluster still uncovered
    chosen = np.zeros(len(p), dtype=bool)  # row index -> True if already picked
    picks = list()  # row indices of the greedy picks

    for _ in range(n_pick):
        # Compute the expected gain of picking each row, given how much of its cluster is still uncovered
        gain = np.where(cluster >= 0, p * np.array([survival.get(c, 1.0) for c in cluster]), 0.0)
        gain[chosen] = -1.0
        i = int(np.argmax(gain))
        picks.append(i)
        chosen[i] = True
        survival[int(cluster[i])] = survival.get(int(cluster[i]), 1.0) * (1.0 - p[i])

    return picks
