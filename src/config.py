"""Data paths and the deployed hyperparameters.

Paths default to a local ``data/`` directory and can be overridden with environment
variables so the challenge data never has to live inside the repository.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file at the project root if present
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_env = os.environ.get
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data"

DATA_DIR = Path(_env("DXC_DATA", _DEFAULT_DATA))
DEL_TABLE = Path(_env("DXC_DEL_TABLE", DATA_DIR / "del_compounds.parquet"))
ACTIVE_LEARNING = Path(_env("DXC_ACTIVE_LEARNING", DATA_DIR / "active_learning_data.csv"))
TEST_SPLIT = Path(_env("DXC_TEST_SPLIT", DATA_DIR / "PGK2_Test_split.csv"))

SEED = 0
N_JOBS = int(_env("DXC_JOBS", "32"))

# ECFP4
FP_RADIUS = 2
FP_BITS = 2048

# DEL negative sampling: 10 negatives per positive
NEG_RATIO = 10

# AL row-selection filter: drop the least AL-confident DEL positives and most AL-confident negatives
AL_FILTER_P = 0.20

# Ensemble: weight on the in-domain AL ranking; the DEL core weight is 1 - ALPHA.
# alpha = 0 makes the score the pure DEL core (no AL weight).
ALPHA = 0.20

# Cluster-coverage selection.
T_CLUSTER = 0.4   # ECFP4 Tanimoto for hit-clustering
T_POOL = 0.4      # ECFP4 Tanimoto for the analogue pool
NOVELTY_AS_CLUSTER = False  # treat the novel pool (no hit within T_POOL)
PMAX = 0.50       # coverage depth knob
N_PICK = 50

# The two deployed submissions differ only in the ranker weight and the coverage geometry.
SUBMISSIONS = {
    1: dict(alpha=ALPHA, t_cluster=T_CLUSTER, t_pool=T_POOL, novelty_as_cluster=NOVELTY_AS_CLUSTER),
    2: dict(alpha=0.00, t_cluster=0.5, t_pool=0.4, novelty_as_cluster=True),
}
