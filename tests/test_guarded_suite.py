from pythia.suite.guarded_runner import _wilson_lower


def test_wilson_gate_is_conservative():
    assert _wilson_lower(500, 500) > 0.99
    assert _wilson_lower(475, 500) < 0.95

