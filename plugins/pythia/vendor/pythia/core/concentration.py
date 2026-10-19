"""One-sided concentration utilities for independent bounded witnesses."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.stats import binom

from .truncated_saa import dkelly_lower_mean


def heterogeneous_chernoff_lower_sum(values: np.ndarray, caps: np.ndarray,
                                      alpha: float) -> float:
    """Lower endpoint for a sum of independent ``Xi in [0, cap_i]``.

    This inverts the Chernoff envelope after maximizing the Bernoulli-extremal
    mgf over all allocations of the null mean across the heterogeneous caps.
    Unlike replacing every cap by ``max(cap_i)``, it retains every row's
    outcome-blind support width.  The result targets ``sum E[Xi]``; dividing
    by ``n`` gives an average-mean endpoint.
    """
    x = np.asarray(values, float).ravel()
    r = np.asarray(caps, float).ravel()
    if (len(x) != len(r) or len(x) < 2 or not 0 < alpha < 1
            or not np.isfinite(x).all() or not np.isfinite(r).all()
            or np.any(r < 0) or np.any(x < -1e-12)
            or np.any(x > r + 1e-12)):
        raise ValueError("independent finite values and matching nonnegative caps required")
    keep = r > 1e-14
    x, r = np.clip(x[keep], 0, r[keep]), r[keep]
    observed = float(x.sum())
    total_cap = float(r.sum())
    if not len(x) or observed <= 0:
        return 0.0

    def max_log_mgf(mean_sum: float, lam: float) -> float:
        if mean_sum <= 0:
            return 0.0
        if mean_sum >= total_cap:
            return float(lam * total_cap)
        lr = np.minimum(lam * r, 700.0)
        a = np.expm1(lr)

        def allocation(log_eta: float) -> np.ndarray:
            eta = math.exp(log_eta)
            return np.clip(1.0 / (eta * r) - 1.0 / a, 0.0, 1.0)

        def balance(log_eta: float) -> float:
            return float(np.dot(r, allocation(log_eta)) - mean_sum)

        low, high = -60.0, 60.0
        while balance(low) < 0:
            low -= 20.0
        while balance(high) > 0:
            high += 20.0
        p = allocation(brentq(balance, low, high, xtol=1e-11))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_zero = np.log1p(-p)
            log_one = np.log(p, where=p > 0, out=np.full_like(p, -np.inf)) + lr
        return float(np.logaddexp(log_zero, log_one).sum())

    def log_p_bound(mean_sum: float) -> float:
        # The null M=0 must be able to drive the bound below alpha even when
        # the observed success sits on a very small-cap coordinate.
        upper_lam = max(1.0, 50.0 / float(r.min()))
        opt = minimize_scalar(
            lambda log_lam: -math.exp(log_lam) * observed
            + max_log_mgf(mean_sum, math.exp(log_lam)),
            bounds=(-14.0, math.log(upper_lam)), method="bounded",
            options={"xatol": 1e-7},
        )
        return min(0.0, float(opt.fun))

    target = math.log(alpha)
    if log_p_bound(observed) <= target:
        # Unusual but possible for highly dispersed supports; search above the
        # observation rather than silently clipping the endpoint.
        high = total_cap
    else:
        high = observed
    root = float(brentq(lambda m: log_p_bound(m) - target, 0.0, high,
                        xtol=1e-9, rtol=1e-10))
    # Guard the certified endpoint against root/KKT floating-point error.
    chernoff = max(0.0, root - 1e-8 * max(1.0, total_cap))
    # Hoeffding's lemma is a quadratic relaxation of this same extremal mgf,
    # so the exact Chernoff inversion theoretically weakly dominates it.  The
    # max protects that pointwise order from numerical optimizer tolerance; it
    # is not outcome-adaptive method selection and spends no extra alpha.
    hoeffding = max(
        observed - math.sqrt(float(np.dot(r, r)) * math.log(1 / alpha) / 2),
        0.0,
    )
    return float(max(chernoff, hoeffding))


def bentkus_lower_mean(values: np.ndarray, upper_bound: float, alpha: float) -> float:
    """Lower confidence bound on the *average* expectation of bounded scores.

    Inverts Bentkus (2004), Theorem 1.2: for independent, potentially
    nonidentical Zi in [0,1], the upper deviation tail is at most e times the
    log-linearly interpolated Binomial(n, mean(E Zi)) tail. This is not valid
    for overlapping patients or outcome-adaptive score construction.
    """
    samples = np.asarray(values, float).ravel()
    if (samples.size < 2 or not np.isfinite(samples).all()
            or not np.isfinite(upper_bound) or upper_bound <= 0
            or not 0 < alpha < 1):
        raise ValueError("independent finite scores, positive cap, and alpha required")
    if np.any(samples < -1e-12) or np.any(samples > upper_bound + 1e-12):
        raise ValueError("score outside declared [0, cap]")
    mean = float(np.mean(np.clip(samples / upper_bound, 0, 1)))
    if mean <= 0:
        return 0.0
    n = samples.size
    observed_sum = n * mean
    k = min(int(np.floor(observed_sum)), n)
    fraction = observed_sum - k

    def log_interpolated_tail(p: float) -> float:
        if p <= 0:
            return -np.inf
        if p >= 1:
            return 0.0
        low = float(binom.logsf(k - 1, n, p))
        if fraction < 1e-12 or k >= n:
            return low
        high = float(binom.logsf(k, n, p))
        return (1.0 - fraction) * low + fraction * high

    target = np.log(alpha) - 1.0  # log(alpha/e)
    root = brentq(
        lambda p: log_interpolated_tail(p) - target,
        0.0, 1.0, xtol=1e-13,
    )
    return float(upper_bound * root)


def bounded_upper_mean(values: np.ndarray, upper_bound: float, alpha: float,
                       method: str) -> float:
    """Upper confidence endpoint for an independent bounded-score mean."""
    x = np.asarray(values, dtype=float).ravel()
    upper = float(upper_bound)
    if (len(x) < 2 or not np.isfinite(x).all() or not math.isfinite(upper)
            or upper <= 0 or np.any(x < -1e-12) or np.any(x > upper + 1e-12)
            or not 0 < alpha < 1):
        raise ValueError("independent values in a finite [0, upper] range required")
    x = np.clip(x, 0, upper)
    if method == "Bentkus":
        return float(upper - bentkus_lower_mean(upper - x, upper, alpha))
    if method == "dKelly":
        return float(upper - dkelly_lower_mean(
            upper - x, cap=upper, alpha=alpha))
    if method == "Hoeffding":
        radius = upper * math.sqrt(math.log(1 / alpha) / (2 * len(x)))
    elif method == "empirical_Bernstein":
        log_term = math.log(2 / alpha)
        radius = (math.sqrt(2 * float(np.var(x, ddof=1)) * log_term / len(x))
                  + 7 * upper * log_term / (3 * (len(x) - 1)))
    else:
        raise ValueError("unsupported upper-mean method")
    return float(min(upper, float(np.mean(x)) + radius))
