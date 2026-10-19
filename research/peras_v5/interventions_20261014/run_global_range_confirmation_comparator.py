"""Matched global-range comparator for the registered cell-range confirmation.

This script does not alter the registered cell-local method.  It evaluates the
pre-existing global-range certificate on the same fresh seeds so that the
strict-improvement endpoint is paired rather than compared across seed sets.
"""
from __future__ import annotations

import json
import sys

import run_intervention_battery as base


if __name__ == "__main__":
    output = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 62001
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in base.MECHANISMS:
            for seed in range(start, start + count):
                for row in base.run_one(mechanism, seed)[:8]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(mechanism, flush=True)
