"""Direct fixed-design Bayes-risk certificates from repeated environments."""
from __future__ import annotations

import numpy as np

from peras.core.concentration import bentkus_lower_mean


LOSSES = ("MAE", "MSE", "Q10", "Q50", "Q90")
METHODS = ("Hoeffding", "empirical_Bernstein", "Bentkus")
PUBLISHED_METHODS = METHODS + ("dKelly_betting",)


def dkelly_betting_lower_mean(values, cap, alpha, d=10, iterations=60):
    """One-sided diversified-Kelly lower bound for a bounded average mean.

    This is the positive-capital, fixed-time specialization of the dKelly
    strategy in Waudby-Smith and Ramdas' ``confseq`` reference code.  For each
    ``c=j/(d+1)``, the capital at candidate normalized mean ``m`` is

        product_i (1 - c + c * x_i / m).

    Their equally weighted mixture is an e-value.  The same proof covers
    independent non-identical observations when the null constrains their
    *average* mean: AM--GM bounds the product of expected factors by one.
    """
    x = np.asarray(values, float).ravel()
    cap = float(cap)
    if (len(x) < 1 or not np.isfinite(x).all() or not np.isfinite(cap)
            or cap <= 0 or np.any(x < 0) or np.any(x > cap)
            or not 0 < alpha < 1 or not isinstance(d, (int, np.integer))
            or d < 1 or iterations < 20):
        raise ValueError("bounded finite scores, valid alpha, and d required")
    z = x / cap
    upper = float(np.mean(z))
    if upper <= 0:
        return 0.
    c = np.arange(1, d + 1, dtype=float) / (d + 1)
    threshold = np.log(1 / alpha)

    def log_evalue(m):
        # At positive m all factors are strictly positive because c < 1.
        logs = np.sum(np.log((1 - c[:, None]) + c[:, None] * z[None, :] / m),
                      axis=1)
        top = float(np.max(logs))
        return top + np.log(np.mean(np.exp(logs - top)))

    # The capital is nonincreasing in m.  The sample mean is always accepted:
    # AM--GM gives each component capital <= 1 there.
    tiny = np.finfo(float).tiny
    if log_evalue(max(tiny, upper * 1e-15)) < threshold:
        return 0.
    lower, accepted = 0., upper
    for _ in range(iterations):
        middle = (lower + accepted) / 2
        if log_evalue(max(tiny, middle)) >= threshold:
            lower = middle
        else:
            accepted = middle
    # Return the accepted edge, conservatively nudged toward zero against
    # floating-point root error.
    return float(max(0., np.nextafter(accepted * cap, 0.)))


def _loss_range(loss, clip):
    lo, hi = map(float, clip)
    width = hi - lo
    if loss == "MAE":
        return width
    if loss == "MSE":
        return width ** 2
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        return max(q, 1 - q) * width
    if loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        return 1.
    raise ValueError("unsupported loss")


def environment_saa_unit_scores(replicates, *, loss, k, clip=(0., 300.)):
    """One empirical optimal-risk score per environment block and unit."""
    y = np.asarray(replicates, float)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 2 * k or y.shape[1] < 1
            or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi or not isinstance(k, (int, np.integer)) or k < 2):
        raise ValueError("finite environment matrix, valid clip, and >=2 blocks required")
    m = y.shape[0] // k
    block = np.clip(y[:m * k], lo, hi).reshape(m, k, y.shape[1])
    if loss == "MSE":
        per_unit = np.var(block, axis=1)
    elif loss == "MAE":
        action = np.median(block, axis=1)
        per_unit = np.mean(np.abs(block - action[:, None, :]), axis=1)
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        action = np.quantile(block, q, axis=1, method="inverted_cdf")
        u = block - action[:, None, :]
        per_unit = np.mean(np.maximum(q * u, (q - 1) * u), axis=1)
    elif loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        ordered = np.sort(block, axis=1)
        best = np.zeros((m, y.shape[1]), dtype=np.int16)
        unresolved = np.ones_like(best, dtype=bool)
        for length in range(k, 0, -1):
            spans = ordered[:, length - 1:, :] - ordered[:, :k - length + 1, :]
            hit = unresolved & np.any(spans <= 2 * h + 1e-12, axis=1)
            best[hit] = length
            unresolved[hit] = False
            if not np.any(unresolved):
                break
        per_unit = 1 - best / k
    else:
        raise ValueError("unsupported loss")
    return per_unit


def environment_saa_scores(replicates, *, loss, k, clip=(0., 300.)):
    """One average empirical optimal-risk score per environment block."""
    return np.mean(environment_saa_unit_scores(
        replicates, loss=loss, k=k, clip=clip), axis=1)


def environment_mse_unbiased_scores(replicates, *, k, clip=(0., 300.)):
    """Block scores whose expectation is the fixed-design MSE Bayes risk.

    For squared loss the Bayes risk is the conditional variance.  The usual
    within-coordinate sample variance (``ddof=1``) is unbiased, so averaging it
    over registered units removes the finite-k SAA optimism without fitting a
    predictor.  Blocks remain independent because they use disjoint
    environment rows.
    """
    y = np.asarray(replicates, float)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 2 * k or y.shape[1] < 1
            or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi or not isinstance(k, (int, np.integer)) or k < 2):
        raise ValueError("finite environment matrix, valid clip, and >=2 blocks required")
    m = y.shape[0] // k
    block = np.clip(y[:m * k], lo, hi).reshape(m, k, y.shape[1])
    return np.mean(np.var(block, axis=1, ddof=1), axis=1)


def environment_mse_unbiased_unit_scores(replicates, *, k, clip=(0., 300.)):
    """Unbiased MSE scores retaining the independent unit dimension."""
    y = np.asarray(replicates, float)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 2 * k or y.shape[1] < 1
            or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi or not isinstance(k, (int, np.integer)) or k < 2):
        raise ValueError("finite environment matrix, valid clip, and >=2 blocks required")
    m = y.shape[0] // k
    block = np.clip(y[:m * k], lo, hi).reshape(m, k, y.shape[1])
    return np.var(block, axis=1, ddof=1)


def _lower_mean(values, cap, alpha, method):
    values = np.asarray(values, float).ravel()
    n = len(values)
    if method == "Bentkus":
        return float(bentkus_lower_mean(values, cap, alpha))
    if method == "dKelly_betting":
        return dkelly_betting_lower_mean(values, cap, alpha)
    if method == "Hoeffding":
        radius = cap * np.sqrt(np.log(1 / alpha) / (2 * n))
    elif method == "empirical_Bernstein":
        log_term = np.log(2 / alpha)
        radius = (np.sqrt(2 * np.var(values, ddof=1) * log_term / n)
                  + 7 * cap * log_term / (3 * (n - 1)))
    else:
        raise ValueError("unsupported method")
    return float(max(0., np.mean(values) - radius))


def _upper_mean(values, cap, alpha, method):
    values = np.asarray(values, float).ravel()
    n = len(values)
    if method == "Bentkus":
        return float(cap - bentkus_lower_mean(cap - values, cap, alpha))
    if method == "Hoeffding":
        radius = cap * np.sqrt(np.log(1 / alpha) / (2 * n))
    elif method == "empirical_Bernstein":
        log_term = np.log(2 / alpha)
        radius = (np.sqrt(2 * np.var(values, ddof=1) * log_term / n)
                  + 7 * cap * log_term / (3 * (n - 1)))
    else:
        raise ValueError("unsupported method")
    return float(min(cap, np.mean(values) + radius))


def plugin_risk(replicates, *, loss, clip=(0., 300.)):
    """Exploratory full-environment plug-in for the fixed-design Bayes risk."""
    y = np.clip(np.asarray(replicates, float), *clip)
    if y.ndim != 2 or y.shape[0] < 2 or not np.isfinite(y).all():
        raise ValueError("finite replicate matrix required")
    if loss == "MSE":
        per_unit = np.var(y, axis=0)
    elif loss == "MAE":
        a = np.median(y, axis=0)
        per_unit = np.mean(np.abs(y - a[None, :]), axis=0)
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        a = np.quantile(y, q, axis=0, method="inverted_cdf")
        u = y - a[None, :]
        per_unit = np.mean(np.maximum(q * u, (q - 1) * u), axis=0)
    elif loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        per_unit = np.empty(y.shape[1])
        for j in range(y.shape[1]):
            ordered = np.sort(y[:, j])
            left = best = 0
            for right in range(len(ordered)):
                while ordered[right] - ordered[left] > 2 * h:
                    left += 1
                best = max(best, right - left + 1)
            per_unit[j] = 1 - best / len(ordered)
    else:
        raise ValueError("unsupported loss")
    return float(np.mean(per_unit))


def environment_saa_portfolio(
        replicates, *, loss, score_cap, ks=(2, 4, 8, 16, 32),
        clip=(0., 300.), alpha_family=.01,
        environment_provenance=None, methods=PUBLISHED_METHODS):
    """Selection-safe lower certificate over environment-block sizes/methods."""
    if environment_provenance is None or not str(environment_provenance).strip():
        return {"status": "refused", "bound": None,
                "reason": "repeated-environment provenance required"}
    methods = tuple(methods)
    if (not ks or len(set(ks)) != len(ks) or any(k < 2 for k in ks)
            or not methods or len(set(methods)) != len(methods)
            or any(method not in PUBLISHED_METHODS for method in methods)
            or not 0 < alpha_family < 1):
        raise ValueError("unique blocks/methods and valid family alpha required")
    loss_range = _loss_range(loss, clip)
    cap = min(float(score_cap), loss_range)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    witnesses = ("empirical_optimum", "unbiased_variance") if loss == "MSE" else ("empirical_optimum",)
    alpha = alpha_family / (len(ks) * len(methods) * len(witnesses))
    candidates = []
    for k in ks:
        for witness in witnesses:
            if witness == "unbiased_variance":
                raw = environment_mse_unbiased_scores(replicates, k=k, clip=clip)
                # Popoviciu plus the k/(k-1) Bessel correction bounds every
                # block score; the registered cap may safely truncate it.
                witness_range = (clip[1] - clip[0]) ** 2 * k / (4 * (k - 1))
                candidate_cap = min(cap, witness_range)
            else:
                raw = environment_saa_scores(replicates, loss=loss, k=k, clip=clip)
                candidate_cap = cap
            scores = np.minimum(raw, candidate_cap)
            for method in methods:
                candidates.append({
                    "k": int(k), "method": method, "witness": witness,
                    "bound": _lower_mean(scores, candidate_cap, alpha, method),
                    "score_mean": float(np.mean(scores)),
                    "raw_score_mean": float(np.mean(raw)),
                    "score_cap": float(candidate_cap),
                    "n_blocks": int(len(scores)),
                })
    chosen = max(candidates, key=lambda row: row["bound"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "loss": loss, "bound": chosen["bound"],
        "selected_k": chosen["k"], "selected_method": chosen["method"],
        "selected_witness": chosen["witness"],
        "selected_score_mean": chosen["score_mean"],
        "selected_raw_score_mean": chosen["raw_score_mean"],
        "selected_n_blocks": chosen["n_blocks"],
        "plugin_clipped": plugin_risk(replicates, loss=loss, clip=clip),
        "plugin_claim_tier": "exploratory point estimate",
        "candidate_ks": list(map(int, ks)), "methods": list(methods),
        "witnesses": list(witnesses),
        "score_cap": cap, "loss_range": float(loss_range),
        "alpha_family": float(alpha_family), "alpha_per_candidate": float(alpha),
        "clip": list(map(float, clip)),
        "environment_provenance": str(environment_provenance),
        "target": "fixed-design mean clipped-outcome Bayes risk conditional on each registered unit's full information",
        "assumptions": [
            "environment rows are independent and identically distributed",
            "each coordinate marginal is the registered target unit's conditional outcome law",
            "environment design, clip, score cap, block sizes, methods, and alpha allocation were frozen",
            "clipping cannot increase the supported location-loss Bayes risk",
        ],
        "candidates": candidates,
    }


def environment_coordinate_saa_portfolio(
        replicates, *, loss, score_cap, ks=(2, 4, 8, 16, 32),
        clip=(0., 300.), alpha_family=.01,
        environment_provenance=None,
        coordinate_independence_provenance=None):
    """Lower certificate exploiting registered independence across units.

    This route is stronger than :func:`environment_saa_portfolio` only under a
    stronger design contract.  Product independence across unit coordinates,
    together with independent environment rows, makes every disjoint
    block--unit score independent.  The flattened scores may be non-identically
    distributed; the registered target is their average expectation.
    """
    if environment_provenance is None or not str(environment_provenance).strip():
        return {"status": "refused", "bound": None,
                "reason": "repeated-environment provenance required"}
    if (coordinate_independence_provenance is None
            or not str(coordinate_independence_provenance).strip()):
        return {"status": "refused", "bound": None,
                "reason": "coordinate-independence provenance required"}
    if (not ks or len(set(ks)) != len(ks) or any(k < 2 for k in ks)
            or not 0 < alpha_family < 1):
        raise ValueError("unique block sizes >=2 and valid family alpha required")
    loss_range = _loss_range(loss, clip)
    cap = min(float(score_cap), loss_range)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("positive finite score cap required")
    witnesses = (("empirical_optimum", "unbiased_variance")
                 if loss == "MSE" else ("empirical_optimum",))
    alpha = alpha_family / (len(ks) * len(METHODS) * len(witnesses))
    candidates = []
    for k in ks:
        for witness in witnesses:
            if witness == "unbiased_variance":
                raw = environment_mse_unbiased_unit_scores(
                    replicates, k=k, clip=clip).ravel()
                witness_range = (clip[1] - clip[0]) ** 2 * k / (4 * (k - 1))
                candidate_cap = min(cap, witness_range)
            else:
                raw = environment_saa_unit_scores(
                    replicates, loss=loss, k=k, clip=clip).ravel()
                candidate_cap = cap
            scores = np.minimum(raw, candidate_cap)
            for method in METHODS:
                candidates.append({
                    "k": int(k), "method": method, "witness": witness,
                    "bound": _lower_mean(scores, candidate_cap, alpha, method),
                    "score_mean": float(np.mean(scores)),
                    "raw_score_mean": float(np.mean(raw)),
                    "score_cap": float(candidate_cap),
                    "n_independent_block_unit_scores": int(len(scores)),
                })
    chosen = max(candidates, key=lambda row: row["bound"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "loss": loss, "bound": chosen["bound"],
        "selected_k": chosen["k"], "selected_method": chosen["method"],
        "selected_witness": chosen["witness"],
        "selected_score_mean": chosen["score_mean"],
        "selected_raw_score_mean": chosen["raw_score_mean"],
        "selected_n_independent_scores": chosen["n_independent_block_unit_scores"],
        "plugin_clipped": plugin_risk(replicates, loss=loss, clip=clip),
        "plugin_claim_tier": "exploratory point estimate",
        "candidate_ks": list(map(int, ks)), "methods": list(METHODS),
        "witnesses": list(witnesses), "score_cap": cap,
        "loss_range": float(loss_range), "alpha_family": float(alpha_family),
        "alpha_per_candidate": float(alpha), "clip": list(map(float, clip)),
        "environment_provenance": str(environment_provenance),
        "coordinate_independence_provenance": str(
            coordinate_independence_provenance),
        "target": "fixed-design mean clipped-outcome Bayes risk conditional on each registered unit's full information",
        "assumptions": [
            "environment rows are independent and identically distributed",
            "all unit coordinates are mutually independent within each environment row",
            "each coordinate marginal is the registered target unit's conditional outcome law",
            "environment and coordinate independence, clip, score cap, block sizes, methods, and alpha allocation were frozen",
            "clipping cannot increase the supported loss Bayes risk",
        ],
        "candidates": candidates,
    }


def environment_dependence_aware_portfolio(
        replicates, *, loss, score_cap, ks=(2, 4, 8, 16, 32),
        clip=(0., 300.), alpha_family=.01,
        environment_provenance=None,
        coordinate_independence_provenance=None):
    """Select row or coordinate concentration under one valid alpha ledger."""
    if coordinate_independence_provenance is None:
        row = environment_saa_portfolio(
            replicates, loss=loss, score_cap=score_cap, ks=ks, clip=clip,
            alpha_family=alpha_family,
            environment_provenance=environment_provenance)
        return {
            "status": row["status"], "claim_tier": row.get("claim_tier"),
            "bound": row.get("bound"), "selected_route": "row_level",
            "route_alpha": {"row_level": float(alpha_family)},
            "routes": {"row_level": row},
            "reason": row.get("reason"),
        }
    route_alpha = alpha_family / 2
    row = environment_saa_portfolio(
        replicates, loss=loss, score_cap=score_cap, ks=ks, clip=clip,
        alpha_family=route_alpha,
        environment_provenance=environment_provenance)
    coordinate = environment_coordinate_saa_portfolio(
        replicates, loss=loss, score_cap=score_cap, ks=ks, clip=clip,
        alpha_family=route_alpha,
        environment_provenance=environment_provenance,
        coordinate_independence_provenance=coordinate_independence_provenance)
    valid = {"row_level": row, "coordinate_independent": coordinate}
    if any(result["status"] == "refused" for result in valid.values()):
        return {"status": "refused", "bound": None,
                "reason": "one or more registered dependence routes refused",
                "routes": valid}
    selected_route, selected = max(
        valid.items(), key=lambda item: item[1]["bound"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "loss": loss, "bound": selected["bound"],
        "selected_route": selected_route,
        "selected_k": selected["selected_k"],
        "selected_method": selected["selected_method"],
        "selected_witness": selected["selected_witness"],
        "route_alpha": {name: float(route_alpha) for name in valid},
        "family_alpha": float(alpha_family),
        "target": selected["target"],
        "routes": valid,
    }


def environment_holdout_upper(
        replicates, *, loss, clip=(0., 300.), alpha_family=.01,
        environment_provenance=None):
    """Upper endpoint from an action learned on disjoint environments."""
    if environment_provenance is None or not str(environment_provenance).strip():
        return {"status": "refused", "upper": None,
                "reason": "repeated-environment provenance required"}
    y = np.asarray(replicates, float)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 4 or y.shape[1] < 1
            or not np.isfinite(y).all() or lo >= hi or not 0 < alpha_family < 1):
        raise ValueError("finite repeated environments, valid clip, and alpha required")
    y = np.clip(y, lo, hi)
    learn, evaluate = np.array_split(y, 2, axis=0)
    if loss == "MSE":
        action = np.mean(learn, axis=0)
        losses = (evaluate - action[None, :]) ** 2
    elif loss == "MAE":
        action = np.median(learn, axis=0)
        losses = np.abs(evaluate - action[None, :])
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        action = np.quantile(learn, q, axis=0, method="inverted_cdf")
        u = evaluate - action[None, :]
        losses = np.maximum(q * u, (q - 1) * u)
    else:
        return {"status": "refused", "upper": None,
                "reason": "holdout upper currently supports location losses only"}
    scores = np.mean(losses, axis=1)
    loss_range = _loss_range(loss, clip)
    alpha = alpha_family / len(METHODS)
    candidates = [{"method": method,
                   "upper": _upper_mean(scores, loss_range, alpha, method)}
                  for method in METHODS]
    chosen = min(candidates, key=lambda row: row["upper"])
    return {
        "status": "conditional", "claim_tier": "conditional",
        "loss": loss, "upper": float(chosen["upper"]),
        "selected_method": chosen["method"],
        "evaluation_score_mean": float(np.mean(scores)),
        "n_action_learning_environments": int(len(learn)),
        "n_action_evaluation_environments": int(len(evaluate)),
        "loss_range": float(loss_range), "alpha_family": float(alpha_family),
        "alpha_per_method": float(alpha), "clip": [lo, hi],
        "environment_provenance": str(environment_provenance),
        "target": "fixed-design mean clipped-outcome Bayes risk conditional on each registered unit's full information",
        "assumptions": [
            "action-learning and evaluation environment halves are independent",
            "environment rows are iid and each coordinate has the target unit-conditional law",
            "loss, clip, split, methods, and alpha allocation were frozen",
        ],
        "candidates": candidates,
    }
