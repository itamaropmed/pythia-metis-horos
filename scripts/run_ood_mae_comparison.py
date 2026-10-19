"""Reproduce the ten-family OOD known-noise MAE comparison."""
import json

from horos.benchmark.ood_mae_comparison import run_ood_mae_comparison


if __name__ == "__main__":
    print(json.dumps(run_ood_mae_comparison(), indent=2, sort_keys=True))
