import numpy as np

from research.conditional_risk_ood_suite import (
    DESIGNS, _mixture_truth, generate,
)


def test_known_normal_mixture_risk_truth_is_finite_and_analytical():
    scale = np.array([10., 10.])
    weights = np.ones((2, 1))
    mae, miss = _mixture_truth(np.zeros(2), scale, weights.T, np.array([0.]), np.array([1.]))
    assert np.allclose(mae, 10 * np.sqrt(2 / np.pi))
    assert np.all((miss >= 0) & (miss <= 1))


def test_all_ten_designs_have_patient_disjoint_temporal_support_and_known_truth():
    assert len(DESIGNS) == 10
    for design in DESIGNS:
        data = generate(design, seed=31, patients=300)
        assert np.intersect1d(data["entity"][data["train"]],
                              data["entity"][data["test"]]).size == 0
        assert data["times"][data["train"]].max() < data["times"][data["test"]].min()
        for risk in data["risk_truth"].values():
            assert np.isfinite(risk).all() and (risk >= 0).all()
        for probability in data["miss_truth"].values():
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()


def test_text_concept_changes_known_noise_risk_and_noise_shift_is_invisible_preoutcome():
    data = generate("note_concept_risk", seed=32, patients=600)
    concept = data["notes"][:, 0]
    risk = data["risk_truth"]["frozen_model_a"]
    assert risk[concept == 1].mean() > risk[concept == 0].mean() + 5.

    base = generate("homoskedastic_normal", seed=33, patients=600, noise_shift=1.)
    shifted = generate("homoskedastic_normal", seed=33, patients=600, noise_shift=2.)
    te = base["test"]
    tr = base["train"]
    assert np.array_equal(base["features"], shifted["features"])
    assert np.array_equal(base["y"][tr], shifted["y"][tr])
    base_risk = base["risk_truth"]["frozen_model_a"][te]
    shifted_risk = shifted["risk_truth"]["frozen_model_a"][te]
    assert np.allclose(shifted_risk, 2 * base_risk)


def test_patient_shared_noise_really_has_within_patient_residual_dependence():
    data = generate("patient_shared_noise", seed=34, patients=1200)
    residual = (data["y"] - data["mu"]).reshape(-1, 2)
    assert np.corrcoef(residual[:, 0], residual[:, 1])[0, 1] > .35

    base = generate("patient_shared_noise", seed=35, patients=600, noise_shift=1.)
    shifted = generate("patient_shared_noise", seed=35, patients=600, noise_shift=2.)
    test = base["test"]
    assert np.allclose(
        shifted["risk_truth"]["frozen_model_a"][test],
        2 * base["risk_truth"]["frozen_model_a"][test],
    )
    assert np.allclose(
        (shifted["y"] - shifted["mu"])[test],
        2 * (base["y"] - base["mu"])[test],
    )
