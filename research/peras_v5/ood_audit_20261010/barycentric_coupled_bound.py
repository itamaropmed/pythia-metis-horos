"""Barycentric coupled-environment penalty: a sharper, loss-specific near-replicate correction.

Setting (same contract as coupled_environment_w1.py).  Globally disjoint, outcome-blind blocks t of k units.
Unit i has clipped (optionally shift-aligned) conditional law Q_i.  Evaluation outcomes are independent across
units.  Independent environment rows supply one coupled draw per unit with the correct marginals; the coupling
inside a row is arbitrary.

Theorem (barycentric penalty).  Let ell(u) be a convex location loss and, inside one block, let
(V_1..V_k) be ANY coupling of (Q_1..Q_k) and Vbar = mean_j V_j.  Let S = inf_a k^-1 sum_j ell(Y_j - a) be the
in-tuple SAA score on independent evaluation outcomes Y_j ~ Q_j, and T = k^-1 sum_j R(Q_j) the block's mean
Bayes risk.  Then
    (i)  MAE and pinball (multiplier c = 1 for MAE, 1/2 for every pinball level q): E S <= T + c * E[ k^-1 sum_j |V_j - Vbar| ]
    (ii) squared loss:                                               E S <= T + E[ k^-1 sum_j (V_j - Vbar)^2 ]
Proof.  E S <= inf_a k^-1 sum_j E ell(Y_j - a)  (SAA optimism; depends only on the marginals, so the Y_j may
be replaced by the coupled V_j).
 (i)  with d_j = V_j - Vbar, rho_q(V_j - a) <= rho_q(Vbar - a) + q d_j+ + (1-q) d_j-; since sum_j d_j = 0 the block
      means of d+ and d- are both half of mean|d|, giving c = 1/2 (MAE = 2 rho_0.5: c = 1); take a optimal for the law of Vbar: the first term becomes
      R(law Vbar).  With a_j* optimal for Q_j and abar = mean a_j*, convexity gives
      ell(Vbar - abar) = ell(mean_j (V_j - a_j*)) <= mean_j ell(V_j - a_j*), so R(law Vbar) <= T.
 (ii) k^-1 sum_j (V_j - a)^2 = (Vbar - a)^2 + k^-1 sum_j (V_j - Vbar)^2 exactly; with a = E Vbar the first term is
      Var(Vbar) <= (mean_j sd_j)^2 <= mean_j Var_j = T.                                                    QED
The penalty in (i) is never larger than the pairwise coupling cost (m k^2)^-1 sum_ij |V_i - V_j| used by the
generic route, because |V_j - Vbar| <= k^-1 sum_i |V_j - V_i|; it is strictly smaller for k >= 3 in general.
(ii) replaces the generic MSE penalty 2*D*W1 (D = clip width, e.g. 600 x W1 on a [0,300] clip) by a squared
coupled dispersion, which is the dominant improvement.  Both penalties are environment means of bounded
statistics (ranges D/2 and D^2/4 after clipping), so an upper confidence bound from independent environment rows
upper-bounds them; any shift fixed without the evaluation and cap outcomes leaves every R(Q_i) unchanged
(Bayes risk of a location loss is translation invariant).

Elementary ingredients (SAA optimism, Jensen, coupling) are classical; the contribution is the loss-specific
barycentric form of the near-replicate penalty and its outcome-only calibration.  No predictor is trained.
"""
from __future__ import annotations

import numpy as np

from core import erm_risk
from coupled_environment_w1 import _upper_mean, METHODS
from peras.core.concentration import bentkus_lower_mean

# effective multipliers: rho_q(x + d) <= rho_q(x) + q d+ + (1-q) d-, and inside a block sum_j (V_j - Vbar) = 0,
# so the mean positive and negative parts are equal: penalty = (q + (1-q))/2 * mean|d| = mean|d|/2 for ANY pinball
# level (0.5 for Q90 instead of the generic Lipschitz 0.9); MAE = 2 rho_0.5 gives multiplier 1.
LIPSCHITZ = {"MAE": 1.0, "Q90": 0.5}


def environment_bary_costs(replicates, blocks, *, clip, shift=None):
    """Per-environment barycentric dispersions: mean |V - Vbar| and mean (V - Vbar)^2 over all block members."""
    y = np.asarray(replicates, float); idx = np.asarray(blocks, int); lo, hi = map(float, clip)
    if y.ndim != 2 or y.shape[0] < 2 or idx.ndim != 2 or idx.shape[1] < 2 or np.unique(idx).size != idx.size:
        raise ValueError("replicate matrix and globally disjoint blocks required")
    if shift is not None:
        y = y - np.asarray(shift, float)[None, :]
    y = np.clip(y, lo, hi)
    V = y[:, idx]                                   # (E, m, k)
    dev = V - V.mean(axis=2, keepdims=True)
    D = hi - lo
    return np.abs(dev).mean(axis=(1, 2)), (dev ** 2).mean(axis=(1, 2)), D / 2.0, D * D / 4.0


def calibrate_bary(replicates, blocks, *, clip, delta_family, coupling_provenance, shift=None, shift_provenance=None):
    if not coupling_provenance or (shift is not None and not shift_provenance):
        return {"status": "refused", "reason": "coupling and shift provenance required"}
    a, s, ra, rs = environment_bary_costs(replicates, blocks, clip=clip, shift=shift)
    alpha = delta_family / (2 * len(METHODS))      # two penalty families x concentration routes
    up = lambda v, r: min(_upper_mean(v, r, alpha, m) for m in METHODS)
    return {"status": "conditional", "abs_cap": up(a, ra), "sq_cap": up(s, rs),
            "abs_mean": float(a.mean()), "sq_mean": float(s.mean()), "n_environments": int(len(a)),
            "coupling_provenance": coupling_provenance, "shift_provenance": shift_provenance or "none"}


def lower_bound_bary(outcomes, blocks, *, loss, cap, score_clip, alpha, outcome_clip, shift=None):
    """T_full >= retained * max(0, LCB(E S) - penalty), penalty from calibrate_bary (Theorem above)."""
    y = np.asarray(outcomes, float); idx = np.asarray(blocks, int); lo, hi = map(float, outcome_clip)
    if shift is not None:
        y = y - np.asarray(shift, float)
    yy = np.clip(y, lo, hi)
    scores = np.minimum(erm_risk(loss, yy[idx]), score_clip)
    lcb = bentkus_lower_mean(scores, score_clip, alpha)
    pen = cap["sq_cap"] if loss == "MSE" else LIPSCHITZ[loss] * cap["abs_cap"]
    retained = idx.size / y.size
    return {"status": "conditional", "loss": loss, "bound": float(retained * max(0.0, lcb - pen)),
            "lower_score_mean": float(lcb), "penalty": float(pen), "score_clip": float(score_clip),
            "tuple_size": int(idx.shape[1]), "retained_fraction": float(retained)}
