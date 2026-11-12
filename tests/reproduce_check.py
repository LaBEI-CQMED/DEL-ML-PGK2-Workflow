"""Reproduction check: regenerate the committed submissions and assert they match the artifacts.

Full run (retrains every model for each submission); needs the challenge data configured via the
DXC_* environment variables (see README). Run:

  DXC_JOBS=32 python tests/reproduce_check.py            # both submissions
  DXC_JOBS=32 python tests/reproduce_check.py 2          # only submission 2
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg          # noqa: E402
from data import load_library  # noqa: E402
from run import run            # noqa: E402

HERE = Path(__file__).parent
REFERENCES = {
    1: HERE / "submission_1_reference.csv",
    2: HERE / "submission_2_reference.csv",
}


def check(submission, library):
    ref = pd.read_csv(REFERENCES[submission])
    out = run(library, verbose=False, **cfg.SUBMISSIONS[submission])
    assert len(out) == len(ref) == 178_052, f"row count {len(out)} vs {len(ref)}"

    merged = out.merge(ref, on="CatalogID", suffixes=("", "_ref"))
    assert len(merged) == len(ref), "CatalogID sets differ"
    assert (merged.Sel_50 == merged.Sel_50_ref).all(), "selection differs"
    max_dscore = merged.Score.sub(merged.Score_ref).abs().max()
    assert max_dscore < 1e-6, f"score differs by {max_dscore}"

    print(f"PASS: reproduces submission {submission} ({len(out):,} rows, "
          f"{int(out.Sel_50.sum())} picks, max |dScore| {max_dscore:.2e})")


def main():
    which = [int(sys.argv[1])] if len(sys.argv) > 1 else sorted(cfg.SUBMISSIONS)
    library = load_library(cfg.TEST_SPLIT)
    for submission in which:
        check(submission, library)


if __name__ == "__main__":
    main()
