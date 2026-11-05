"""Run the full analysis: Step 1 ARMA-EGARCH -> Step 2 CCC/DCC -> Step 3 TVP-VAR.

Usage (from the repository root):

    python run_all.py
    python run_all.py --quick    # skip the Step 1 model-candidate searches
"""
from __future__ import annotations

import argparse
import sys

import step1_arma_egarch
import step2_dcc_garch
import step3_connectedness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="skip ARMA grid and EGARCH candidate searches in Step 1")
    args = parser.parse_args()

    sys.argv = [sys.argv[0]] + (["--quick"] if args.quick else [])
    step1_arma_egarch.main()
    sys.argv = sys.argv[:1]
    step2_dcc_garch.main()
    step3_connectedness.main()


if __name__ == "__main__":
    main()
