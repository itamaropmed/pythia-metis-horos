"""Run the M0 certification suite and write the acceptance report.

    python scripts/run_m0_suite.py            # full 100 designs (~3 min)
    python scripts/run_m0_suite.py --quick    # 14 designs
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from horos.benchmark.bregman_runner import run_suite  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m0_suite.json"))
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    run_suite(n_reg=8 if a.quick else 60,
              n_clf=6 if a.quick else 40,
              out=a.out)
