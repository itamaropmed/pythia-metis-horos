#!/usr/bin/env python3
"""Run the ten-generator clipped paired-bound comparison."""
from __future__ import annotations

import argparse
import json

from horos.benchmark.clipped_replicate_comparison import run_clipped_replicate_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--n-entities", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run_clipped_replicate_comparison(
        repetitions=args.repetitions, n_entities=args.n_entities
    ), indent=2))


if __name__ == "__main__":
    main()
