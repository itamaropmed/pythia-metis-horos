#!/usr/bin/env python3
"""Run the known-support sparse metric sensitivity control."""
from __future__ import annotations

import argparse
import json

from horos.benchmark.sparse_metric_audit import run_sparse_metric_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run_sparse_metric_audit(repetitions=args.repetitions), indent=2))


if __name__ == "__main__":
    main()
