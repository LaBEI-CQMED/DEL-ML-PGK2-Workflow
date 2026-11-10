"""Data paths and the deployed hyperparameters.

Paths default to a local ``data/`` directory and can be overridden with environment
variables so the challenge data never has to live inside the repository.
"""
import os

_env = os.environ.get
_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATA_DIR = _env("DXC_DATA", _DEFAULT_DATA)
DEL_TABLE = _env("DXC_DEL_TABLE", os.path.join(DATA_DIR, "del_compounds.parquet"))
ACTIVE_LEARNING = _env("DXC_ACTIVE_LEARNING", os.path.join(DATA_DIR, "active_learning_data.csv"))
TEST_SPLIT = _env("DXC_TEST_SPLIT", os.path.join(DATA_DIR, "PGK2_Test_split.csv"))

SEED = 0
# Shared machine: oversubscribing also slows LightGBM badly here, so cap workers well below nproc.
N_JOBS = int(_env("DXC_JOBS", "32"))

# ECFP4
FP_RADIUS = 2
FP_BITS = 2048

# DEL negative sampling: 10 negatives per positive, chosen most similar to the scored library.
NEG_RATIO = 10

# AL row-selection filter: drop the least AL-confident DEL positives and most AL-confident negatives.
AL_FILTER_P = 0.20

# Ensemble: weight on the in-domain AL ranking; the DEL core carries the remaining 1 - ALPHA.
ALPHA = 0.20

# Cluster-coverage selection.
T_CLUSTER = 0.4   # ECFP4 Tanimoto for both hit-clustering and the analogue cutoff
PMAX = 0.50       # coverage depth knob: spreads picks across clusters before deepening any one
N_PICK = 50
