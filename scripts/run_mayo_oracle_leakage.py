#!/usr/bin/env python3
"""Run aggregate-only Mayo target-in-note/tabular falsification diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.mayo_oracle_leakage import run_mayo_oracle_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=Path("data/derived/mayo"))
    parser.add_argument("--n-patients", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(run_mayo_oracle_leakage(args.derived, n_patients=args.n_patients), indent=2))


if __name__ == "__main__":
    main()
