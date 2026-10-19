#!/usr/bin/env python3
"""Regenerate the variable-size patient bounded-range benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.patient_range_comparison import run_patient_range_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/patient_range_comparison_20260914.json"),
    )
    args = parser.parse_args()
    result = run_patient_range_comparison(repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
