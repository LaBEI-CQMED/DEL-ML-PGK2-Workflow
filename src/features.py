"""ECFP4 fingerprints and Tanimoto similarity."""
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from .config import FP_BITS, FP_RADIUS

RDLogger.DisableLog("rdApp.*")
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def _fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:   # RDKit parses "" into a valid zero-atom molecule
        return None
    return _generator.GetFingerprintAsNumPy(mol).nonzero()[0].astype(np.int32)


def _fingerprint_chunk(chunk):
    return [_fingerprint(s) for s in chunk]


def _parallel_fingerprint(smiles, n_jobs, chunk):
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    with Pool(n_jobs) as pool:
        rows = list()
        for result in pool.imap(_fingerprint_chunk, chunks):
            rows.extend(result)

    return rows


def featurize(smiles, n_jobs=1, chunk=20_000, verbose=False):
    """SMILES iterable -> (csr matrix [n, FP_BITS], boolean mask of parsable rows).

    Unparsable rows become all-zero rows marked False in the mask; callers drop them.
    """
    smiles = list(smiles)

    if n_jobs > 1 and len(smiles) > chunk:
        rows = _parallel_fingerprint(smiles, n_jobs=n_jobs, chunk=chunk)
    else:
        rows = [_fingerprint(s) for s in smiles]

    valid = np.array([bits is not None for bits in rows])
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, bits in enumerate(rows):
        indptr[i + 1] = indptr[i] + (0 if bits is None else len(bits))

    indices = np.zeros(0, dtype=np.int32)
    if valid.any():
        indices = np.concatenate([b for b in rows if b is not None]) 

    data = np.ones(len(indices), dtype=np.float32)
    X = sparse.csr_matrix(
        (data, indices, indptr), 
        shape=(len(rows), FP_BITS), 
        dtype=np.float32
    )

    if verbose:
        print(f"[featurize] {len(rows):,} SMILES -> {X.shape[0]:,} x {X.shape[1]:,} ECFP4", flush=True)
        print(f"[featurize] {valid.sum():,} parsable | {len(rows) - valid.sum():,} unparsable", flush=True)
    
    return X, valid


def tanimoto_nearest(X, X_ref, chunk=20_000):
    """For each row of X: its max Tanimoto to X_ref and the index of that nearest reference row."""
    ref = np.asarray(X_ref.todense(), dtype=np.float32)
    ref_pop = ref.sum(axis=1)
    maxsim = np.zeros(X.shape[0], dtype=np.float32)
    argmax = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(0, X.shape[0], chunk):
        block = X[i:i + chunk]
        inter = block @ ref.T
        pop = np.asarray(block.sum(axis=1)).ravel()
        union = pop[:, None] + ref_pop[None, :] - inter
        sim = np.where(union > 0, inter / union, 0.0)
        maxsim[i:i + chunk] = sim.max(axis=1)
        argmax[i:i + chunk] = sim.argmax(axis=1)
    return maxsim, argmax


def tanimoto_max(X, X_ref, chunk=20_000):
    """For each row of X, its maximum Tanimoto to any row of X_ref."""
    return tanimoto_nearest(X, X_ref, chunk)[0]


def tanimoto_matrix(X, X_ref):
    """Dense |X| x |X_ref| Tanimoto matrix; for small reference sets only."""
    a = np.asarray(X.todense(), dtype=np.float32)
    b = np.asarray(X_ref.todense(), dtype=np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)
