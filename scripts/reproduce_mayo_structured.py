#!/usr/bin/env python3
"""Train the historical Mayo candidate in Horos and emit frozen OOF predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from horos.models.mayo import reproduce_tuned_structured_oof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = reproduce_tuned_structured_oof(
        args.cases, args.manifest, args.output, seed=args.seed,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
