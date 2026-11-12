"""Reproduction check: regenerate submission 1 and assert it matches the committed artifact.

Full run (retrains every model); needs the challenge data configured via the DXC_* environment
variables (see README). Run:

  DXC_JOBS=32 python tests/reproduce_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, Path(__file__).parent.parent)
import config as cfg          # noqa: E402
from data import load_library  # noqa: E402
from run import run            # noqa: E402

REFERENCE = Path(__file__).parent / "submission_1_reference.csv"


def main():
    out = run(load_library(cfg.TEST_SPLIT), verbose=False)
    ref = pd.read_csv(REFERENCE)
    assert len(out) == len(ref) == 178_052, f"row count {len(out)} vs {len(ref)}"

    merged = out.merge(ref, on="CatalogID", suffixes=("", "_ref"))
    assert len(merged) == len(ref), "CatalogID sets differ"
    assert (merged.Sel_50 == merged.Sel_50_ref).all(), "selection differs"
    max_dscore = merged.Score.sub(merged.Score_ref).abs().max()
    assert max_dscore < 1e-6, f"score differs by {max_dscore}"

    print(f"PASS: reproduces submission 1 ({len(out):,} rows, "
          f"{int(out.Sel_50.sum())} picks, max |dScore| {max_dscore:.2e})")


if __name__ == "__main__":
    main()
