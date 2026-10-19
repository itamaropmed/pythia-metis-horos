#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.benchmark.replicate_adversarial import run_replicate_adversarial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("/tmp/replicate_adversarial.json"))
    args = parser.parse_args()
    result = run_replicate_adversarial(repetitions=args.repetitions)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
