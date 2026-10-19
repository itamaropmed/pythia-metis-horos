"""Monte Carlo check of selection-safe representation-graph composition."""
from __future__ import annotations

import json
import numpy as np

from peras.certificate_graph import GapEvidence, LowerEvidence, compile_certificate_graph


ALPHA = .05
REPETITIONS = 5000
N = 400
RISKS = {"tabular": .60, "concepts": .45, "raw": .35}
PROBLEM = "synthetic-binary-risk-v1"
EVENTS = 5
PER_EVENT = ALPHA / EVENTS
RADIUS = np.sqrt(np.log(1 / PER_EVENT) / (2 * N))


def lower(evidence_id, node, sample_mean, event):
    return LowerEvidence(
        evidence_id, PROBLEM, node, max(0., sample_mean - RADIUS), "conditional",
        "Hoeffding Bernoulli lower endpoint", {"n": N}, event, PER_EVENT,
        ("registered bounded independent calibration outcomes",))


def gap(evidence_id, poor, rich, sample_mean, event):
    return GapEvidence(
        evidence_id, PROBLEM, poor, rich, min(1., sample_mean + RADIUS),
        "conditional", "statistical_upper", "Hoeffding Bernoulli upper endpoint",
        {"n": N}, event, PER_EVENT,
        ("registered bounded independent gap-calibration outcomes",))


rng = np.random.default_rng(20261012)
violations = positives = selected_concept_base = 0
for _ in range(REPETITIONS):
    base_tab = rng.binomial(1, RISKS["tabular"], N).mean()
    base_concept = rng.binomial(1, RISKS["concepts"], N).mean()
    g_tc = rng.binomial(1, RISKS["tabular"] - RISKS["concepts"], N).mean()
    g_cr = rng.binomial(1, RISKS["concepts"] - RISKS["raw"], N).mean()
    g_tr = rng.binomial(1, RISKS["tabular"] - RISKS["raw"], N).mean()
    result = compile_certificate_graph(
        target_node="raw", family_alpha=ALPHA,
        lower_evidence=(lower("b-tab", "tabular", base_tab, "b-tab-event"),
                        lower("b-concept", "concepts", base_concept, "b-concept-event")),
        gap_evidence=(gap("g-tc", "tabular", "concepts", g_tc, "g-tc-event"),
                      gap("g-cr", "concepts", "raw", g_cr, "g-cr-event"),
                      gap("g-tr", "tabular", "raw", g_tr, "g-tr-event")))
    assert result.status == "complete"
    violations += result.bound > RISKS["raw"] + 1e-12
    positives += result.bound > 0
    selected_concept_base += result.selected_path[0] == "concepts"

overspend = compile_certificate_graph(
    target_node="raw", family_alpha=ALPHA,
    lower_evidence=(lower("b1", "raw", .4, "one"), lower("b2", "raw", .4, "two"),
                    lower("b3", "raw", .4, "three"), lower("b4", "raw", .4, "four"),
                    lower("b5", "raw", .4, "five"), lower("b6", "raw", .4, "six")),
    gap_evidence=())
assert overspend.status == "refused"

print(json.dumps({
    "repetitions": REPETITIONS,
    "family_alpha": ALPHA,
    "known_raw_risk": RISKS["raw"],
    "compiled_bound_exceedances": int(violations),
    "empirical_exceedance_rate": violations / REPETITIONS,
    "positive_bounds": int(positives),
    "concept_base_selected": int(selected_concept_base),
    "uncorrected_overspend_refused": True,
}, indent=2))
