"""Print aggregate retrospective Mayo pinball/W30 sensitivity JSON."""
import json

from horos.benchmark.mayo_decision_losses import run_mayo_decision_losses


if __name__ == "__main__":
    print(json.dumps(run_mayo_decision_losses("data/derived/mayo"), indent=2, sort_keys=True))
