"""Reproduce the ten-design quantile/window-miss known-risk comparison."""
import json

from horos.benchmark.decision_loss_comparison import run_decision_loss_comparison


if __name__ == "__main__":
    print(json.dumps(run_decision_loss_comparison(), indent=2, sort_keys=True))
