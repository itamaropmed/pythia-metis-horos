#!/usr/bin/env python3
"""Run the aggregate-only retrospective Mayo replicate diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.mayo_replicate import run_mayo_replicate_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=Path("data/derived/mayo"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/mayo_replicate.json"))
    args = parser.parse_args()
    result = run_mayo_replicate_diagnostic(args.derived)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
