"""Bound-guided development of the Metis risk labeler with note readers (LLM via LiteLLM, OpenMed, rules, Jev).

The risk labeler assigns each case to a risk cell (prediction bin x family x disagreement, ...) and Pythia
certifies the frozen model's miss rate per cell.  A note reader can make the labeler sharper only if the notes
carry information about the model's *error* beyond the current cells.  That is exactly an instrumented-
information quantity, with the outcome replaced by the miss indicator:

    risk information gain of reader R  =  V(cells; P(miss | cells, T))   certified via R  (Pythia P1/P2)

Development loop (each step is a certificate, not an estimate):
  1. GATE      certify the risk-information gain of every candidate reader (portfolio, alpha / J).
               Readers with a zero certified gain are not added to the labeler.
  2. REFINE    add the accepted reader's label-free bins as a new, finest hierarchy level; rebuild the
               outcome-blind support partition and the simultaneous local-risk intervals (v2 route).
  3. ACCOUNT   count certified-higher / certified-lower cases before vs after (resolution gained), and
               certify how much of the risk information the reader captures vs leaves (Pythia P3 ledger).
  4. GUARD     on a reader/model upgrade, run the regression test (P4) before swapping it in.

Splits: FIT entities train the deployed model and every reader's miss model; CAL entities are only used for
certificates.  Nothing here reads CAL outcomes to choose bins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pythia.core.instrumented import _groups, loo_centre_groups, projection_bound
from pythia.core.local_model_risk import build_local_model_risk_registry, build_support_partition


@dataclass
class GateResult:
    reader: str
    certified_gain: float
    accepted: bool


def miss_probabilities(X_fit, miss_fit, X_cal, cells_fit, cells_cal, C: float = 1.0):
    """Reader's miss model: logistic regression on (cell one-hot, reader features), trained on FIT only."""
    from sklearn.linear_model import LogisticRegression
    keys = {k: j for j, k in enumerate(sorted(set(cells_fit) | set(cells_cal)))}

    def design(X, cells):
        oh = np.zeros((len(cells), len(keys))); oh[np.arange(len(cells)), [keys[c] for c in cells]] = 1
        return np.column_stack([oh, X])
    m = LogisticRegression(C=C, max_iter=3000).fit(design(X_fit, cells_fit), miss_fit)
    return m.predict_proba(design(X_cal, cells_cal)), m.predict_proba(design(X_fit, cells_fit))


def gate_readers(cells_cal, miss_cal, reader_probs: dict, alpha: float = .05) -> list[GateResult]:
    """Step 1: certified risk-information gain per reader, one alpha ledger over the portfolio."""
    g = _groups(cells_cal); J = len(reader_probs); out = []
    for name, Q in reader_probs.items():
        b = projection_bound(loo_centre_groups(np.asarray(Q, float), g), np.asarray(miss_cal, int),
                             alpha / J, groups=g).bound
        out.append(GateResult(name, b, b > 0))
    return sorted(out, key=lambda r: -r.certified_gain)


def reader_bins(p_fit: np.ndarray, p_cal: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Label-free bins of the reader's predicted miss probability; edges from FIT predictions only."""
    edges = np.unique(np.quantile(p_fit, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(p_cal, edges)


def refine_and_certify(base_levels: dict, extra_bins, entity_cal, loss_cal: np.ndarray, *, model: str = "model",
                       min_entities: int = 100, alpha: float = .05) -> dict:
    """Step 2-3: local-risk v2 certificate with and without the reader level; returns relation counts.

    base_levels: {level_name: labels} finest first (outcome-blind), WITHOUT 'global'.
    extra_bins:  reader bins for the same CAL rows, or None for the baseline labeler."""
    levels = dict(base_levels)
    if extra_bins is not None:                     # nested chain: reader x each base level (finest first), then reader
        rb = np.char.add("r", np.asarray(extra_bins).astype(str))
        levels = {f"reader|{k}": np.char.add(np.char.add(rb, "|"), np.asarray(v).astype(str))
                  for k, v in base_levels.items()}
        levels["reader"] = rb
    levels["global"] = np.repeat("all", len(loss_cal))
    part = build_support_partition(levels, entity_cal, min_entities)
    hc = {"region": part.labels(levels), "global": levels["global"]}
    reg = build_local_model_risk_registry({model: loss_cal}, hc, entity_cal, loss_cap=1., alpha=alpha,
                                          min_entities=min_entities, outcome_blind_hierarchy=True,
                                          independent_entities=True, loss_cap_attested=True,
                                          alpha_ledger="eligible", relation_test="contrast")
    rel = [reg.route(model, {"region": hc["region"][i], "global": "all"}).local_relation for i in range(len(loss_cal))]
    cert = [r.startswith("certified") for r in rel]
    return {"regions": len(part.regions), "certified_share": float(np.mean(cert)),
            "relation_counts": {r: rel.count(r) for r in sorted(set(rel))}, "registry": reg, "region_of_row": hc["region"]}
