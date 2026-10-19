from pathlib import Path

import pytest

from pythia.core.guarded import load_guard
from pythia.suite.mae_dependence_runner import run_mae_shared_bias_suite


DERIVED = Path(__file__).parents[1] / "data" / "derived" / "mayo"


@pytest.mark.skipif(not (DERIVED / "env2.npy").exists(), reason="private guard unavailable")
def test_shared_bias_adjustment_repairs_adversarial_suite():
    result = run_mae_shared_bias_suite(load_guard(DERIVED), n_designs=20, seed_offset=37000)
    for row in result["by_rung"].values():
        assert row["adjusted_valid"] == row["total"]
