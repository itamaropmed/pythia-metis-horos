import argparse
import json
from pathlib import Path

from pythia.core.guarded import load_guard
from pythia.suite.guarded_runner import run_guarded_suite


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=10_000)
    parser.add_argument("--shared-bias", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    guard = load_guard(ROOT / "data" / "derived" / "mayo")
    result = run_guarded_suite(guard, args.n, args.seed_offset, args.shared_bias)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()

