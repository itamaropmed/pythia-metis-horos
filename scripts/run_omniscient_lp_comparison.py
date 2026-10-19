"""Run a small ten-design MAE comparison against an omniscient LP heuristic."""
import json

from horos.benchmark.omniscient_lp_comparison import run_omniscient_lp_comparison


if __name__ == "__main__":
    print(json.dumps(run_omniscient_lp_comparison(), indent=2, sort_keys=True))
