#!/usr/bin/env python3
import argparse
import json

from pythia.suite.dependence_runner import run_dependence_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--n", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(run_dependence_suite(seed=args.seed, n=args.n), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
