#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.information_simulation import run_information_simulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--n", type=int, default=40_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_information_simulation(seed=args.seed, n=args.n)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
