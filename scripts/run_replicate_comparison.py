#!/usr/bin/env python3
"""Regenerate the model-free near-replicate benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.replicate_comparison import run_replicate_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/replicate_comparison_20260915.json"),
    )
    args = parser.parse_args()
    result = run_replicate_comparison(repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
