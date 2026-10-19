#!/usr/bin/env python3
import argparse
import json

from pythia.core.guarded import load_guard
from pythia.suite.mae_dependence_runner import run_mae_shared_bias_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard-dir", default="data/derived/mayo")
    parser.add_argument("--n-designs", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run_mae_shared_bias_suite(
        load_guard(args.guard_dir), n_designs=args.n_designs,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
