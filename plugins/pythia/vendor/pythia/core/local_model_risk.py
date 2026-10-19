"""Simultaneous support-aware local certificates for frozen-model loss.

This module does not train a model or a risk forecaster.  Given bounded losses,
an outcome-blind finite hierarchy, and independent entity clusters, it builds
simultaneous intervals for case-weighted expected loss in every hierarchy node.
An arriving case may then choose the finest sufficiently supported node using
covariates only without invalidating coverage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .concentration import heterogeneous_chernoff_lower_sum


@dataclass(frozen=True)
class LocalRiskNode:
    model: str
    level: str
    key: str
    n_rows: int
    n_entities: int
    estimate: float
    lower: float
    upper: float
    contrast_estimate: float | None = None
    contrast_lower: float | None = None
    contrast_upper: float | None = None
    eligible: bool = True


@dataclass(frozen=True)
class RoutedLocalRiskCertificate:
    status: str
    claim_tier: str
    model: str
    level: str | None
    key: str | None
    n_rows: int
    n_entities: int
    estimate: float | None
    lower: float | None
    upper: float | None
    global_estimate: float | None
    global_lower: float | None
    global_upper: float | None
    local_relation: str
    loss_cap: float
    alpha_family: float
    alpha_per_tail_node_model: float | None
    estimand: str
    assumptions: tuple[str, ...]
    reason: str | None = None
    contrast_lower: float | None = None
    contrast_upper: float | None = None
    relation_test: str = "overlap"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LocalRiskRegistry:
    status: str
    claim_tier: str
    loss_cap: float
    alpha_family: float
    alpha_per_tail_node_model: float | None
    min_entities: int
    level_order: tuple[str, ...]
    nodes: Mapping[str, LocalRiskNode]
    assumptions: tuple[str, ...]
    reason: str | None = None
    alpha_ledger: str = "all_nodes"
    relation_test: str = "overlap"
    n_charged_tests: int | None = None

    def route(self, model: str, keys: Mapping[str, object]) -> RoutedLocalRiskCertificate:
        if self.status == "refused":
            return RoutedLocalRiskCertificate(
                status="refused", claim_tier="refused", model=model,
                level=None, key=None, n_rows=0, n_entities=0, estimate=None,
                lower=None, upper=None, loss_cap=self.loss_cap,
                global_estimate=None, global_lower=None, global_upper=None,
                local_relation="refused",
                alpha_family=self.alpha_family,
                alpha_per_tail_node_model=self.alpha_per_tail_node_model,
                estimand="case-weighted expected frozen-model loss in routed cell",
                assumptions=self.assumptions, reason=self.reason,
            )
        for level in self.level_order:
            if level not in keys:
                continue
            key = str(keys[level])
            node = self.nodes.get(f"{model}|{level}|{key}")
            if node is not None and node.n_entities >= self.min_entities:
                global_node = None
                if self.level_order:
                    global_level = self.level_order[-1]
                    if global_level in keys:
                        global_node = self.nodes.get(
                            f"{model}|{global_level}|{str(keys[global_level])}"
                        )
                if global_node is None or level == self.level_order[-1]:
                    relation = "global_only"
                elif self.relation_test == "contrast" and node.contrast_lower is not None:
                    # direct simultaneous interval for (local risk - global risk)
                    if node.contrast_lower > 0:
                        relation = "certified_higher_than_global"
                    elif node.contrast_upper < 0:
                        relation = "certified_lower_than_global"
                    else:
                        relation = "local_shift_unresolved"
                elif node.lower > global_node.upper:
                    relation = "certified_higher_than_global"
                elif node.upper < global_node.lower:
                    relation = "certified_lower_than_global"
                else:
                    relation = "local_shift_unresolved"
                return RoutedLocalRiskCertificate(
                    status="ok", claim_tier=self.claim_tier, model=model,
                    level=level, key=key, n_rows=node.n_rows,
                    n_entities=node.n_entities, estimate=node.estimate,
                    lower=node.lower, upper=node.upper, loss_cap=self.loss_cap,
                    global_estimate=(None if global_node is None else global_node.estimate),
                    global_lower=(None if global_node is None else global_node.lower),
                    global_upper=(None if global_node is None else global_node.upper),
                    local_relation=relation,
                    alpha_family=self.alpha_family,
                    alpha_per_tail_node_model=self.alpha_per_tail_node_model,
                    estimand="case-weighted expected frozen-model loss in routed cell",
                    assumptions=self.assumptions,
                    contrast_lower=node.contrast_lower, contrast_upper=node.contrast_upper,
                    relation_test=self.relation_test,
                )
        return RoutedLocalRiskCertificate(
            status="refused", claim_tier="refused", model=model,
            level=None, key=None, n_rows=0, n_entities=0, estimate=None,
            lower=None, upper=None, loss_cap=self.loss_cap,
            global_estimate=None, global_lower=None, global_upper=None,
            local_relation="refused",
            alpha_family=self.alpha_family,
            alpha_per_tail_node_model=self.alpha_per_tail_node_model,
            estimand="case-weighted expected frozen-model loss in routed cell",
            assumptions=self.assumptions,
            reason="no hierarchy node reaches the declared independent-entity support",
        )


def _node_interval(loss: np.ndarray, groups: np.ndarray, mask: np.ndarray,
                   cap: float, alpha_tail: float) -> tuple[int, int, float, float, float]:
    selected_loss = loss[mask]
    selected_groups = groups[mask]
    entities, inverse = np.unique(selected_groups, return_inverse=True)
    n_rows = len(selected_loss)
    sums = np.bincount(inverse, weights=selected_loss, minlength=len(entities)) / n_rows
    counts = np.bincount(inverse, minlength=len(entities))
    caps = counts.astype(float) * cap / n_rows
    if len(entities) < 2:
        return n_rows, len(entities), float(sums.sum()), 0.0, float(cap)
    lower = heterogeneous_chernoff_lower_sum(sums, caps, alpha_tail)
    upper = float(caps.sum() - heterogeneous_chernoff_lower_sum(
        caps - sums, caps, alpha_tail,
    ))
    return n_rows, len(entities), float(sums.sum()), float(lower), upper


def _contrast_interval(loss: np.ndarray, groups: np.ndarray, mask: np.ndarray,
                       cap: float, alpha_tail: float) -> tuple[float, float, float]:
    """Simultaneous-ready interval for D = (case-weighted local risk) - (case-weighted global risk).

    Per entity g, d_g = sum_i L_i (1{i in c}/n_c - 1/n).  Entities are independent and d_g lies in the
    outcome-blind range [-(n_g - n_gc) cap / n, n_gc (1/n_c - 1/n) cap], so the heterogeneous-support
    Chernoff bound applies to the shifted contributions.  This tests the relation directly instead of
    requiring two separate intervals to be disjoint.
    """
    entities, inverse = np.unique(groups, return_inverse=True)
    n, n_c = len(loss), int(mask.sum())
    coef = np.where(mask, 1.0 / n_c - 1.0 / n, -1.0 / n)
    d = np.bincount(inverse, weights=loss * coef, minlength=len(entities))
    n_g = np.bincount(inverse, minlength=len(entities)).astype(float)
    n_gc = np.bincount(inverse, weights=mask.astype(float), minlength=len(entities))
    lo_r = -(n_g - n_gc) * cap / n
    hi_r = n_gc * (1.0 / n_c - 1.0 / n) * cap
    width = hi_r - lo_r
    shifted = np.clip(d - lo_r, 0, width)
    est = float(d.sum())
    lower = heterogeneous_chernoff_lower_sum(shifted, width, alpha_tail) + float(lo_r.sum())
    upper = float(hi_r.sum() - heterogeneous_chernoff_lower_sum(width - shifted, width, alpha_tail))
    return est, float(lower), upper


def build_local_model_risk_registry(
    losses: Mapping[str, Sequence[float]],
    hierarchy: Mapping[str, Sequence[object]],
    entity_ids: Sequence[object],
    *,
    loss_cap: float,
    alpha: float = .05,
    min_entities: int = 100,
    outcome_blind_hierarchy: bool,
    independent_entities: bool,
    loss_cap_attested: bool,
    alpha_ledger: str = "all_nodes",
    relation_test: str = "overlap",
) -> LocalRiskRegistry:
    """Build a selection-safe hierarchy of local frozen-model risk intervals.

    alpha_ledger="all_nodes" (default, original) charges every hierarchy node.  "eligible" charges only the
    nodes whose independent-entity count reaches ``min_entities`` -- an outcome-blind property, so the
    family is fixed before outcomes are read; ineligible nodes get the trivial interval [0, cap] and can
    never be routed to.  relation_test="contrast" adds a direct simultaneous interval for
    (local - global) risk in every eligible non-global node and uses it for the relation label.
    """
    assumptions = (
        "hierarchy labels are fixed without outcomes",
        "entity clusters are mutually independent",
        "loss is almost surely within [0, loss_cap]",
        "new-case routing reads no calibration outcome",
    )
    if not (outcome_blind_hierarchy and independent_entities and loss_cap_attested):
        missing = []
        if not outcome_blind_hierarchy:
            missing.append("outcome-blind hierarchy")
        if not independent_entities:
            missing.append("independent entities")
        if not loss_cap_attested:
            missing.append("hard loss cap")
        return LocalRiskRegistry(
            status="refused", claim_tier="refused", loss_cap=float(loss_cap),
            alpha_family=float(alpha), alpha_per_tail_node_model=None,
            min_entities=int(min_entities), level_order=tuple(hierarchy), nodes={},
            assumptions=assumptions, reason="missing attestation: " + ", ".join(missing),
        )
    cap = float(loss_cap)
    if not np.isfinite(cap) or cap <= 0 or not 0 < alpha < 1 or min_entities < 2:
        raise ValueError("positive cap, valid alpha, and min_entities >= 2 required")
    groups = np.asarray(entity_ids).reshape(-1)
    model_loss = {name: np.asarray(value, float).reshape(-1) for name, value in losses.items()}
    levels = {name: np.asarray(value).astype(str).reshape(-1) for name, value in hierarchy.items()}
    n = len(groups)
    if not model_loss or not levels or n < 2:
        raise ValueError("nonempty aligned losses, hierarchy, and entities required")
    if any(len(value) != n for value in [*model_loss.values(), *levels.values()]):
        raise ValueError("losses, hierarchy, and entities must align")
    if any(not np.isfinite(value).all() or np.any(value < 0) or np.any(value > cap)
           for value in model_loss.values()):
        raise ValueError("observed loss lies outside the attested range")
    if alpha_ledger not in ("all_nodes", "eligible") or relation_test not in ("overlap", "contrast"):
        raise ValueError("unknown alpha_ledger or relation_test")
    node_specs = [(level, key, labels == key)
                  for level, labels in levels.items() for key in np.unique(labels)]
    global_level = tuple(levels)[-1]
    n_ent = {(level, str(key)): len(np.unique(groups[mask])) for level, key, mask in node_specs}
    eligible = {k: (v >= min_entities) for k, v in n_ent.items()}
    charged = node_specs if alpha_ledger == "all_nodes" else [
        sp for sp in node_specs if eligible[(sp[0], str(sp[1]))]]
    n_contrasts = sum(1 for sp in charged if sp[0] != global_level and eligible[(sp[0], str(sp[1]))]) \
        if relation_test == "contrast" else 0
    n_tests = len(charged) + n_contrasts
    alpha_tail = float(alpha / (2 * len(model_loss) * max(n_tests, 1)))
    nodes: dict[str, LocalRiskNode] = {}
    for model, loss in model_loss.items():
        for level, key, mask in node_specs:
            ok = eligible[(level, str(key))]
            if alpha_ledger == "eligible" and not ok:
                n_rows = int(mask.sum())
                nodes[f"{model}|{level}|{key}"] = LocalRiskNode(
                    model=model, level=level, key=str(key), n_rows=n_rows,
                    n_entities=n_ent[(level, str(key))], estimate=float(loss[mask].mean()) if n_rows else 0.0,
                    lower=0.0, upper=cap, eligible=False)
                continue
            n_rows, n_entities, estimate, lower, upper = _node_interval(
                loss, groups, mask, cap, alpha_tail,
            )
            c_est = c_lo = c_hi = None
            if relation_test == "contrast" and ok and level != global_level:
                c_est, c_lo, c_hi = _contrast_interval(loss, groups, mask, cap, alpha_tail)
            nodes[f"{model}|{level}|{key}"] = LocalRiskNode(
                model=model, level=level, key=str(key), n_rows=n_rows,
                n_entities=n_entities, estimate=estimate, lower=lower, upper=upper,
                contrast_estimate=c_est, contrast_lower=c_lo, contrast_upper=c_hi, eligible=ok,
            )
    return LocalRiskRegistry(
        status="ok", claim_tier="conditional", loss_cap=cap,
        alpha_family=float(alpha), alpha_per_tail_node_model=alpha_tail,
        min_entities=int(min_entities), level_order=tuple(levels), nodes=nodes,
        assumptions=assumptions, alpha_ledger=alpha_ledger, relation_test=relation_test,
        n_charged_tests=n_tests,
    )


@dataclass(frozen=True)
class SupportPartition:
    """Outcome-blind backoff partition: the region a case is routed to is exactly the set the
    certificate describes.

    Plain hierarchy routing certifies the *whole* node a case backs off to, although the cases that
    back off are precisely those whose finer cells were too sparse -- a subpopulation that can differ
    from the node average.  The partition removes that mismatch.  Levels are processed from finest to
    coarsest; a node becomes a region when its *residual* rows (rows not already claimed by a finer
    region) contain at least ``min_entities`` independent entities.  Remaining rows fall to the global
    residual region.  Only hierarchy labels and entity ids are read, never outcomes.
    """
    level_order: tuple[str, ...]
    regions: frozenset
    min_entities: int

    def region_of(self, keys: Mapping[str, object]) -> str:
        for level in self.level_order:
            if level in keys and (level, str(keys[level])) in self.regions:
                return f"{level}|{keys[level]}"
        return f"{self.level_order[-1]}|{keys[self.level_order[-1]]}"

    def labels(self, hierarchy: Mapping[str, Sequence[object]]) -> np.ndarray:
        levels = {k: np.asarray(v).astype(str) for k, v in hierarchy.items()}
        n = len(next(iter(levels.values())))
        out = np.empty(n, dtype=object); done = np.zeros(n, bool)
        for level in self.level_order:
            if level not in levels:
                continue
            keys = levels[level]
            for key in np.unique(keys[~done]):
                if (level, key) in self.regions or level == self.level_order[-1]:
                    m = (~done) & (keys == key); out[m] = f"{level}|{key}"; done |= m
        return out.astype(str)


def build_support_partition(hierarchy: Mapping[str, Sequence[object]], entity_ids: Sequence[object],
                            min_entities: int = 100) -> SupportPartition:
    """Construct the outcome-blind backoff partition (finest-first residual support rule)."""
    order = tuple(hierarchy)
    levels = {k: np.asarray(v).astype(str) for k, v in hierarchy.items()}
    groups = np.asarray(entity_ids).reshape(-1)
    done = np.zeros(len(groups), bool); regions = set()
    for level in order[:-1]:
        keys = levels[level]
        for key in np.unique(keys[~done]):
            m = (~done) & (keys == key)
            if len(np.unique(groups[m])) >= min_entities:
                regions.add((level, key)); done |= m
    return SupportPartition(level_order=order, regions=frozenset(regions), min_entities=int(min_entities))
