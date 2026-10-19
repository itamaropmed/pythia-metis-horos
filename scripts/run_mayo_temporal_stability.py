#!/usr/bin/env python3
"""Print only aggregate Mayo temporal paired-bound diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.mayo_temporal_stability import run_mayo_temporal_stability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=Path("data/derived/mayo"))
    parser.add_argument("--periods", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_mayo_temporal_stability(
        args.derived, n_periods=args.periods
    ), indent=2))


if __name__ == "__main__":
    main()
