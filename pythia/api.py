"""Public fail-closed certification entry point."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import (
    Certificate, ClaimTier, DependenceGuardSpec, FoldMatrix, GuardSpec, PairMatrix,
    RangeDependenceGuardSpec, ReplicateGuardSpec, Representation, TaskCard,
)
from .core.bregman import bregman_bound
from .core.dependence_bounds import (
    catoni_cluster_pairwise_bound,
    empirical_bernstein_patient_pairwise_bound,
    sharp_pairwise_population_bound,
)
from .core.guarded import cluster_bootstrap_mae, guarded_mae, mae_shared_bias_sensitivity
from .core.replicates import paired_replicate_clipped_envelope, paired_replicate_lower_bound
from .core.router import route


def _refusal(task: TaskCard, representation: Representation, reason: str,
             referral: str | None = None) -> Certificate:
    return Certificate(
        status="refused", claim_tier=ClaimTier.REFUSED, task=task,
        representation=representation, bound=None, estimator=None,
        achieved_loss=None, gamma=None,
        refusal={"reason": reason, "referral": referral},
    )


def certify(task: TaskCard, fold_matrix: FoldMatrix | PairMatrix, representation: Representation,
            guard: GuardSpec | DependenceGuardSpec | RangeDependenceGuardSpec | ReplicateGuardSpec | None = None,
            slices: Mapping[str, np.ndarray] | None = None) -> Certificate:
    """Certify one task without silently changing its claim strength."""
    fold_matrix.validate()
    if (representation.provenance.get("uses_target") is True
            or representation.provenance.get("available_at_prediction_time") is False):
        return _refusal(
            task, representation,
            "representation contains target-derived or post-prediction-time information",
            "remove the leaked feature and rebuild the evaluation artifact",
        )
    if isinstance(fold_matrix, PairMatrix):
        if (representation.provenance.get("available_at_prediction_time") is not True
                or representation.provenance.get("outcome_blind") is not True
                or not representation.source_hash):
            return _refusal(
                task, representation,
                "paired-replicate certification requires an explicit pre-outcome representation audit and source hash",
                "attest timing and outcome blindness and supply the representation source hash",
            )
        if not isinstance(guard, ReplicateGuardSpec):
            return _refusal(
                task, representation, "paired-replicate certification requires ReplicateGuardSpec",
                "provide a versioned smoothness/range guard",
            )
        try:
            guard.validate()
        except ValueError as exc:
            return _refusal(task, representation, str(exc), "repair the replicate guard")
        outcome_blind = fold_matrix.meta.get("pairing_outcome_blind") is True
        randomized_blocks = fold_matrix.meta.get("iid_randomized_blocks") is True
        conditionally_independent = (
            fold_matrix.meta.get("conditionally_independent_entities") is True
        )
        independent_blocks = (
            fold_matrix.meta.get("independent_blocks") is True or randomized_blocks
        )
        selected_concentration = guard.concentration
        automatic_selection = selected_concentration == "auto"
        if automatic_selection:
            if conditionally_independent:
                selected_concentration = "conditional_empirical_bernstein"
            elif independent_blocks:
                selected_concentration = "conditional_block_empirical_bernstein"
            else:
                return _refusal(
                    task, representation,
                    "automatic replicate inference found neither audited entity independence nor independent blocks",
                    "audit one dependence contract or select no certificate",
                )
        conditional_method = selected_concentration.startswith("conditional_")
        conditional_block_method = selected_concentration.startswith("conditional_block_")
        design_valid = (
            outcome_blind and independent_blocks
            if conditional_block_method else
            outcome_blind and conditionally_independent
            if conditional_method else outcome_blind and randomized_blocks
        )
        if not design_valid:
            required = (
                "outcome-blind pairing and independent declared blocks"
                if conditional_block_method else
                "outcome-blind pairing and conditionally independent entity outcomes"
                if conditional_method else
                "outcome-blind pairing inside iid randomized blocks"
            )
            return _refusal(
                task, representation,
                f"replicate theorem requires {required}",
                "supply an audited pair artifact with the required design flags",
            )
        try:
            replicate_args = (
                fold_matrix.y_left, fold_matrix.y_right, fold_matrix.distances,
                fold_matrix.left_entities, fold_matrix.right_entities,
                fold_matrix.pair_blocks,
            )
            replicate_kwargs = dict(
                loss=task.loss, lipschitz_bound=guard.lipschitz_bound,
                distance_bound=guard.distance_bound,
                outcome_difference_bound=guard.outcome_difference_bound,
                alpha=guard.alpha, concentration=selected_concentration,
                quantile=task.quantile, window=task.window,
            )
            result = (
                paired_replicate_clipped_envelope(
                    *replicate_args, **replicate_kwargs,
                    outcome_term_clips=guard.outcome_term_clip_grid,
                )
                if guard.outcome_term_clip_grid is not None else
                paired_replicate_lower_bound(*replicate_args, **replicate_kwargs)
            )
        except ValueError as exc:
            return _refusal(task, representation, str(exc), "repair the pair artifact or guard")
        if result.get("status") != "complete":
            return _refusal(
                task, representation, result.get("reason", "replicate bound refused"),
                "repair the pair artifact or predeclared guard",
            )
        return Certificate(
            status="complete", claim_tier=ClaimTier.CONDITIONAL,
            task=task, representation=representation, bound=float(result["bound"]),
            estimator=float(result["estimate"]), achieved_loss=None,
            gamma=float(result["estimate"] - result["bound"]),
            assumptions=tuple(result["assumptions"]),
            diagnostics={
                **result, "automatic_concentration_selection": automatic_selection,
                "selected_concentration": selected_concentration,
                "attainable_within_window_success_upper_percent": (
                    float(100.0 * (1.0 - result["bound"]))
                    if task.loss.lower() in {"w30_miss", "window_miss"} else None
                ),
            },
            rungs={
                f"replicate_{selected_concentration}"
                f"{'_clipped_envelope' if guard.outcome_term_clip_grid is not None else ''}":
                float(result["bound"])
            },
            intervals={"confidence": 1.0 - guard.alpha,
                       "type": (
                           f"fixed-design block-level {selected_concentration.replace('_', ' ')}"
                           if conditional_block_method else
                           f"fixed-design pair-level {selected_concentration.replace('_', ' ')}"
                           if conditional_method else
                            f"block-level {selected_concentration.replace('_', ' ')}"
                       )},
            provenance={
                "pair_meta": dict(fold_matrix.meta), "guard_version": guard.version,
                "guard_source_hash": guard.source_hash,
                "guard_calibration_population": guard.calibration_population,
                "representation_hash": representation.source_hash,
            },
            warnings=(
                "Certified only conditionally on the metric, smoothness cap, declared range or clip, and sampling design.",
                "The estimand is exactly the one stated in diagnostics, not automatically the full deployment population.",
                "The paired estimator field is a corrected lower-bound center, not a universally accurate Bayes-risk point estimate.",
                "Clipping-grid selection is valid only when its guard version and grid were frozen before evaluation outcomes.",
            ),
        )
    decision = route(task.loss)
    if decision.regime == "refuse":
        return _refusal(task, representation, decision.reason, decision.referral)
    dependence_guard = isinstance(guard, (DependenceGuardSpec, RangeDependenceGuardSpec))
    if dependence_guard and decision.regime != "bregman":
        return _refusal(
            task, representation,
            "the Catoni dependence guard is currently proved only for squared loss",
            "use loss='mse' or a loss-specific proved guard",
        )

    if decision.regime == "bregman":
        if dependence_guard:
            try:
                guard.validate()
            except ValueError as exc:
                return _refusal(task, representation, str(exc), "repair the versioned dependence guard")
            if task.loss.lower() != "mse":
                return _refusal(
                    task, representation,
                    "the Catoni dependence guard is currently proved only for squared loss",
                    "use loss='mse' or a loss-specific proved guard",
                )
            role = fold_matrix.meta.get("evaluation_role")
            frozen = fold_matrix.meta.get("predictions_frozen_before_evaluation") is True
            if role not in {"external_holdout", "prospective"} or not frozen:
                return _refusal(
                    task, representation,
                    "the independent-patient theorem cannot be applied to overlapping OOF or unfrozen predictions",
                    "supply frozen predictions on an external_holdout or prospective cohort",
                )
            if fold_matrix.groups is None:
                return _refusal(
                    task, representation, "patient/entity groups are required by the dependence guard",
                    "supply the independent sampling-unit identifier",
                )
            try:
                if isinstance(guard, DependenceGuardSpec):
                    robust = catoni_cluster_pairwise_bound(
                        fold_matrix.y, fold_matrix.P_oof, fold_matrix.groups,
                        guard.lambda_max, guard.risk_variance_bound,
                        guard.disagreement_variance_bound, alpha=guard.alpha,
                    )
                    rung_name = "catoni_finite_variance"
                    interval_type = "simultaneous Catoni moment confidence bound"
                    moment_risks = robust.get("catoni_risks")
                else:
                    robust = empirical_bernstein_patient_pairwise_bound(
                        fold_matrix.y, fold_matrix.P_oof, fold_matrix.groups,
                        guard.lambda_max, guard.absolute_error_bound,
                        guard.prediction_difference_bound, alpha=guard.alpha,
                    )
                    rung_name = "empirical_bernstein_bounded_range"
                    interval_type = "simultaneous patient-level empirical-Bernstein confidence bound"
                    moment_risks = robust.get("empirical_risks")
            except ValueError as exc:
                return _refusal(task, representation, str(exc), "repair the evaluation artifact or guard")
            if robust.get("status") != "complete":
                return _refusal(
                    task, representation, robust.get("reason", "dependence bound refused"),
                    "repair the evaluation artifact or predeclared guard",
                )
            y = np.asarray(fold_matrix.y, float)
            p = np.asarray(fold_matrix.P_oof, float)
            groups = np.asarray(fold_matrix.groups).astype(str)
            unique, inverse = np.unique(groups, return_inverse=True)
            counts = np.bincount(inverse)
            group_risks = np.asarray([
                np.bincount(inverse, weights=(row - y) ** 2) / counts for row in p
            ])
            risks = np.mean(group_risks, axis=1)
            group_disagreements = np.asarray([
                [np.bincount(inverse, weights=(p[i] - p[j]) ** 2) / counts
                 for j in range(p.shape[0])]
                for i in range(p.shape[0])
            ])
            disagreement = np.mean(group_disagreements, axis=2)
            plugin = sharp_pairwise_population_bound(risks, disagreement, guard.lambda_max)
            L = float(np.min(moment_risks))
            plugin_value = robust["bound"] if plugin["status"] != "complete" else float(plugin["bound"])
            estimate = float(np.clip(plugin_value, robust["bound"], L))
            return Certificate(
                status="complete", claim_tier=ClaimTier.CONDITIONAL,
                task=task, representation=representation,
                bound=float(robust["bound"]), estimator=estimate,
                achieved_loss=L, gamma=estimate - float(robust["bound"]),
                assumptions=tuple(robust["assumptions"]),
                diagnostics={
                    "robust_moments": robust,
                    "sharp_plugin_estimator": plugin,
                    "evaluation_role": role,
                    "n_independent_groups": int(unique.size),
                },
                rungs={rung_name: float(robust["bound"])},
                intervals={
                    "confidence": 1.0 - guard.alpha,
                    "type": interval_type,
                },
                provenance={
                    "fold_meta": dict(fold_matrix.meta),
                    "dependence_guard_version": guard.version,
                    "dependence_guard_source_hash": guard.source_hash,
                    "dependence_guard_calibration_population": guard.calibration_population,
                    "representation_hash": representation.source_hash,
                },
                warnings=(
                    "Certified only conditionally on the versioned correlation and nuisance caps.",
                ),
            )
        result = bregman_bound(
            fold_matrix.y, fold_matrix.P_oof, task.loss,
            families=list(fold_matrix.families), predictor_names=list(fold_matrix.names),
        )
        return Certificate(
            status="complete", claim_tier=ClaimTier.CONDITIONAL,
            task=task, representation=representation,
            bound=result["bound"], estimator=result["estimator"],
            achieved_loss=result["L"], gamma=result["gamma"],
            assumptions=tuple(result["assumptions"]),
            diagnostics=result["diagnostics"],
            rungs={"orthogonality": result["bound"]},
            provenance={"fold_meta": dict(fold_matrix.meta)},
            warnings=tuple(result["warnings"]),
        )

    if task.loss.lower() != "mae":
        return _refusal(
            task, representation,
            "The guarded pinball implementation has not yet passed its per-alpha calibration gate.",
            "Use MAE or contribute a frozen, independently validated alpha-specific envelope.",
        )
    if guard is None:
        return _refusal(
            task, representation, "MAE requires an explicit versioned guard artifact.",
            "Provide GuardSpec; no default guard is selected silently.",
        )

    labels = None
    slice_name = None
    if slices:
        if len(slices) != 1:
            raise ValueError("v0.1 accepts one predeclared stratification per certificate")
        slice_name, labels = next(iter(slices.items()))
    try:
        result = guarded_mae(fold_matrix.y, fold_matrix.P_oof, guard, labels)
    except ValueError as exc:
        return _refusal(task, representation, str(exc), "Use a proved rung or expand validation.")

    global_result = result["global"]
    primary = result["stratified"] or global_result
    bound = float(primary["rungs"]["calibrated_2d"])
    estimator = float(global_result["estimator"])
    L = float(global_result["L"])
    if not (0.0 <= bound <= estimator <= L + 1e-10):
        raise RuntimeError(f"hard certificate invariant failed: B={bound}, E={estimator}, L={L}")
    bootstrap_reps = int(fold_matrix.meta.get("bootstrap_reps", 0))
    intervals = {}
    if bootstrap_reps:
        if fold_matrix.groups is None:
            return _refusal(
                task, representation,
                "Clustered bootstrap requested but entity group IDs are absent.",
                "Supply the patient/entity grouping used by the fold contract.",
            )
        intervals = {
            "sampling_uncertainty": cluster_bootstrap_mae(
                fold_matrix.y, fold_matrix.P_oof, fold_matrix.groups, guard,
                labels, reps=bootstrap_reps,
                seed=int(fold_matrix.meta.get("bootstrap_seed", 0)),
            )
        }
    fold_warning = ()
    if np.unique(np.asarray(fold_matrix.fold_id)).size < 2:
        fold_warning = ("Per-row fold assignments are unavailable; OOF status relies on the frozen artifact provenance.",)
    supplied_bias_caps = fold_matrix.meta.get("common_bias_mad_caps")
    if supplied_bias_caps is None:
        supplied_bias_caps = [0.0, 0.05 * L, 0.10 * L, 0.20 * L, 0.50 * L]
        bias_cap_source = "default sensitivity grid as fractions of achieved loss; no cap selected"
    else:
        bias_cap_source = "externally supplied in fold artifact metadata"
    bias_sensitivity = mae_shared_bias_sensitivity(
        result, np.asarray(supplied_bias_caps, float),
        use_stratified=result["stratified"] is not None,
    )
    return Certificate(
        status="complete", claim_tier=ClaimTier.CALIBRATED,
        task=task, representation=representation, bound=bound,
        estimator=estimator, achieved_loss=L, gamma=estimator - bound,
        assumptions=(
            "A-quad: calibrated MAE quadrature envelope",
            "A-indep: residual/model-error combination represented by the frozen library",
            "A-fold: fold-error exchangeability",
        ),
        diagnostics={
            "rho": global_result["rho"], "tau": global_result["tau"],
            "m_plus": global_result["m_plus"], "m_bar": global_result["m_bar"],
            "slice": slice_name, "stratified": result["stratified"],
            "uncertainty_separation": (
                "intervals quantify patient-cluster sampling variation; rungs quantify guard/theory sensitivity"
            ),
            "rung_claim_tiers": global_result["rung_claim_tiers"],
            "qcap_claim_note": (
                "Qcap is a proved guard inequality; the complete rung remains conditional on "
                "the predictor-error/dependence assumptions."
            ),
            "shared_bias_sensitivity": bias_sensitivity,
            "shared_bias_cap_source": bias_cap_source,
        },
        rungs=primary["rungs"],
        intervals=intervals,
        provenance={
            "fold_meta": dict(fold_matrix.meta), "guard_version": guard.version,
            "guard_kind": guard.kind, "guard_source_hash": guard.source_hash,
            "guard_calibration_population": guard.calibration_population,
            "representation_hash": representation.source_hash,
        },
        warnings=(
            "The primary MAE bound is simulation-calibrated, not theorem-proved.",
            "Model disagreement cannot identify predictor bias shared by the pool; no nonzero shared-bias cap is selected automatically.",
        ) + fold_warning,
    )
