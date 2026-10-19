#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.heavy_tail_comparison import run_heavy_tail_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_heavy_tail_comparison(
        seed=args.seed, repetitions=args.repetitions,
    ), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
