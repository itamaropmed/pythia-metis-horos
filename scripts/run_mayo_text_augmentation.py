"""Print aggregate-only train-free Mayo note-augmentation diagnostics."""
from __future__ import annotations

import argparse
import json

from horos.benchmark.mayo_text_augmentation import run_mayo_text_augmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", default="data/derived/mayo")
    parser.add_argument("--n-entities", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(
        run_mayo_text_augmentation(args.derived, n_entities=args.n_entities),
        indent=2, sort_keys=True,
    ))
