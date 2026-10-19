"""PYTHIA-U: one simultaneous, assumption-graded certificate for table + text.

Target (Brier units, g = 1 - |p|^2): for a common row set S, conditional on the design (X_i, T_i)_{i in S}:
    G      = (1/|S|) sum_i (1 - |p(X_i, T_i)|^2)                     joint (table + text) floor
    V      = within-cell text information  sum_c (m_c/|S|) (m_c/(m_c-1)) mean_{i in c} |p_i - phat_c|^2   (= NLI)
    G + V  = table floor (exact: the within-cell all-pairs U-statistic has conditional mean G + V)
    G0     = marginal floor; G0 - (G + V) = table information I_X (in-sample).
Given the design, labels are independent with means p_i, so every certificate below is a bound for INDEPENDENT,
NON-IDENTICAL variables and all of them hold simultaneously by a union bound:
    lo_l <= G + D_l <= hi_l      text-neighbour matchings (pair route), every row once per level
    lo'_l <= G + F_l <= hi'_l    global text matchings
    tL <= G + V <= tU            table cells (McDiarmid on the within-cell U-statistic)
    V >= nliL                    NLI projection certificate (Theorem 5, cohort form, vector-valued, any K)
    G <= mU                      held-out model Brier (any predictor fitted on the other fold)
    D_l <= 2 V                   exact
    D_a <= t_ab^gamma D_b + eps  shape assumption A(gamma), relaxed by eps (eps = inf: assumption-free)
min/max G and min V over this LP give the certified joint-floor interval and text information; the table floor,
table information and loss transfers follow.  The NLI and model components are cross-fitted over two halves of the cohort.
Optional stratification: the same certificate inside each stratum of a coarse table variable (alpha split), and the
global floor bound is the weighted sum (the conditional, table-signal-aware form)."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog
from .lp_bounds import design_statistics
from .multiscale_pair_extrapolation import kl_bernoulli_upper, betting_upper
from .replicate_pair_certificates import kl_bernoulli_lower, f_K
from .concentration import (bentkus_lower_mean, bounded_upper_mean,
                            heterogeneous_chernoff_lower_sum)


# ------------------------------------------------------------------ components
def _upper_mean01(z, alpha):
    z = np.clip(np.asarray(z, float), 0, 1)
    return min(kl_bernoulli_upper(float(z.mean()), len(z), alpha / 2), betting_upper(z, alpha / 2))


def table_box(cells, y, K, alpha):
    """Within-cell all-pairs disagreement Q; E[Q | design] = G + V exactly (cells with m >= 2).  Changing one label
    moves Q by <= 2/n  ->  McDiarmid two-sided.  Rows in singleton cells are excluded by the caller (S has m >= 2)."""
    n = len(y); Q = 0.0
    for c in np.unique(cells):
        I = np.flatnonzero(cells == c); m = len(I); cnt = np.bincount(y[I], minlength=K).astype(float)
        Q += (m / n) * (1 - (cnt * (cnt - 1)).sum() / (m * (m - 1)))
    dev = math.sqrt(2 * math.log(2 / alpha) / n)
    return max(Q - dev, 0.0), min(Q + dev, 1.0), Q


def marginal_box(y, K, alpha):
    n = len(y); cnt = np.bincount(y, minlength=K).astype(float); Q = 1 - (cnt * (cnt - 1)).sum() / (n * (n - 1))
    dev = math.sqrt(2 * math.log(2 / alpha) / n); return max(Q - dev, 0.0), min(Q + dev, 1.0), Q


def loo_centre(cells, Qp):
    """h_i = q_i - mean_{j != i, same cell} q_j (vector).  Label-free; sums to 0 in every cell."""
    H = np.zeros_like(Qp)
    for c in np.unique(cells):
        I = np.flatnonzero(cells == c); m = len(I)
        if m >= 2:
            S = Qp[I].sum(0); H[I] = Qp[I] - (S[None] - Qp[I]) / (m - 1)
    return H


def _lower_mean_empirical_bernstein(values, upper_bound, alpha):
    """One-sided empirical-Bernstein lower endpoint on a bounded mean.

    ``bounded_upper_mean`` implements the repository's finite-sample empirical
    Bernstein endpoint.  Reflection around the declared cap turns that upper
    endpoint into a lower endpoint without changing the assumptions.
    """
    x = np.asarray(values, float)
    return float(upper_bound - bounded_upper_mean(
        upper_bound - x, upper_bound, alpha, "empirical_Bernstein"))


def nli_cohort_bound(cells, Qp, y, alpha, concentration="hoeffding"):
    """Vector NLI projection certificate (cohort form):  (1/m) sum |p_i - c_{cell}|^2 >= (C - dev)_+^2 / H for EVERY
    per-cell constant c (in particular the in-sample cell mean), hence V >= the same.  C = mean h_{i, y_i},
    H = mean |h_i|^2, dev from Hoeffding with per-row ranges (max_k h_ik - min_k h_ik)."""
    H = loo_centre(cells, Qp); m = len(y)
    z = H[np.arange(m), y]
    C = float(z.mean()); lo = H.min(1); w = H.max(1) - lo; HH = float((H ** 2).sum(1).mean())
    if HH <= 1e-10:                                   # constant score: no direction (avoids 0/0 round-off)
        return 0.0, dict(C=C, C_lower=0.0, H=HH, dev=0.0, rho=0.0,
                         concentration=concentration)

    allowed = {"hoeffding", "bentkus", "empirical_bernstein",
               "heterogeneous_chernoff", "stratified_bentkus", "safe_portfolio"}
    if concentration not in allowed:
        raise ValueError(f"concentration must be one of {sorted(allowed)}")

    # Each shifted observation x_i = z_i - min_k h_ik is nonnegative.  The
    # score matrix is fixed without the held-out labels, hence x_i are
    # independent conditional on the design even when their distributions and
    # row-specific ranges differ.
    x = z - lo
    cap = float(np.max(w))

    def endpoint(method, a):
        if method == "hoeffding":
            dev_ = math.sqrt((w ** 2).sum() * math.log(1 / a) / 2) / m
            return C - dev_, dev_
        if cap <= 1e-15:
            return C, 0.0
        if method == "heterogeneous_chernoff":
            lower = float(lo.mean() + heterogeneous_chernoff_lower_sum(x, w, a) / m)
            return lower, C - lower
        if method == "stratified_bentkus":
            # Quantile bins depend only on the frozen score ranges, never on
            # held-out outcomes.  Within each bin a local cap replaces the
            # often much larger cohort-wide cap.  All bin endpoints hold
            # simultaneously by an explicit alpha split and their weighted
            # sum targets the original cohort mean exactly.
            positive = np.flatnonzero(w > 1e-15)
            if not len(positive):
                return C, 0.0
            order = positive[np.argsort(w[positive], kind="stable")]
            groups = [g for g in np.array_split(order, min(4, len(order))) if len(g)]
            lower = float(lo.mean())
            for g in groups:
                local_cap = float(w[g].max())
                lower += len(g) / m * bentkus_lower_mean(x[g], local_cap, a / len(groups))
            # Zero-range rows have x_i == 0 and are already represented by
            # their deterministic lower offset in mean(lo).
            return lower, C - lower
        if method == "bentkus":
            lower_x = bentkus_lower_mean(x, cap, a)
        else:
            lower_x = _lower_mean_empirical_bernstein(x, cap, a)
        lower = float(lo.mean() + lower_x)
        return lower, C - lower

    if concentration == "safe_portfolio":
        # Outcome-adaptive selection is valid because all three endpoints hold
        # simultaneously after this explicit Bonferroni allocation.
        candidates = {}
        for method in ("hoeffding", "bentkus", "empirical_bernstein",
                       "heterogeneous_chernoff", "stratified_bentkus"):
            candidates[method] = endpoint(method, alpha / 5)
        selected = max(candidates, key=lambda name: candidates[name][0])
        C_lower, dev = candidates[selected]
    else:
        selected = concentration
        C_lower, dev = endpoint(selected, alpha)
        candidates = {selected: (C_lower, dev)}

    bound = max(C_lower, 0.0) ** 2 / HH
    return bound, dict(
        C=C, C_lower=float(C_lower), H=HH, dev=float(dev),
        rho=C / math.sqrt(HH), concentration=concentration,
        selected_concentration=selected, cap=cap,
        candidates={name: {"C_lower": float(v[0]), "dev": float(v[1])}
                    for name, v in candidates.items()},
    )


def model_ceiling(Qp, y, K, alpha):
    """Held-out model Brier: E[|q_i - e_{y_i}|^2 | design] = g_i + |q_i - p_i|^2 >= g_i.  Range [0, 2]."""
    z = ((Qp - np.eye(K)[y]) ** 2).sum(1) / 2
    return 2 * _upper_mean01(z, alpha), float(2 * z.mean())


# ------------------------------------------------------------------ the unified LP
def unified_lp(st, gamma=.5, eps=None, what="G", sense="min", use_global=True):
    """Variables x = [G, V, D_0..D_{L-1}, F_0..F_{M-1}].  Returns value or None (infeasible = A(gamma) refuted at eps)."""
    L = len(st["box"]); M = len(st["gbox"]) if use_global else 0; nv = 2 + L + M; A, b = [], []
    def vec(pairs):
        r = np.zeros(nv)
        for i, v in pairs:
            r[i] += v
        return r
    G, V = 0, 1; D = lambda l: 2 + l; F = lambda l: 2 + L + l
    for l, (lo, hi) in enumerate(st["box"]):
        A.append(vec([(G, 1), (D(l), 1)])); b.append(hi); A.append(vec([(G, -1), (D(l), -1)])); b.append(-lo)
        A.append(vec([(V, -2), (D(l), 1)])); b.append(0.0)
    for l in range(M):
        lo, hi = st["gbox"][l]
        A.append(vec([(G, 1), (F(l), 1)])); b.append(hi); A.append(vec([(G, -1), (F(l), -1)])); b.append(-lo)
    A.append(vec([(G, 1), (V, 1)])); b.append(st["gx"][1]); A.append(vec([(G, -1), (V, -1)])); b.append(-st["gx"][0])
    A.append(vec([(V, -1)])); b.append(-st.get("nliL", 0.0))                          # NLI certificate
    A.append(vec([(G, 1)])); b.append(st.get("mU", 1.0))                               # model ceiling
    if eps is not None and np.isfinite(eps):
        for fam, ss, idx in ((L, st["s"], D), (M, st.get("gs", []), F)):
            for a_ in range(fam):
                for b_ in range(a_ + 1, fam):
                    t = (ss[a_] / ss[b_]) ** gamma
                    if t < 1:
                        A.append(vec([(idx(a_), 1), (idx(b_), -t)])); b.append(eps)
    c = np.zeros(nv); c[G if what == "G" else V] = 1 if sense == "min" else -1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)] * nv, method="highs")
    return None if res.status != 0 else float(res.x[G if what == "G" else V])


def _bisect(fn, lo, hi, it=30):
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return hi


def unified_certificate(cells, E, F, y, K, alpha=.05, gamma=.5, levels=3, seed=0, model_factory=None, population=True,
                        models=None, nli_concentration="hoeffding", prediction_builders=None):
    """PYTHIA-U on all rows.  Cohort S = rows kept by the text-matching design (cells truncated to multiples of
    2^levels).  The NLI and model components are cross-fitted over two halves of S: the model for half f is trained
    on every row outside half f (labels independent of half f), so V >= sum_f w_f nli_f and G <= sum_f w_f mU_f.
    alpha split: design 1/2, table 1/10, marginal 1/20, NLI 1/10, model 1/10, population lift 1/10.
    models: optional portfolio [(F_j, factory_j), ...]; every model gets its share of the NLI and model alpha and the
    best bound per half is kept (union bound), so adding models never invalidates the certificate."""
    cells = np.asarray(cells); y = np.asarray(y); n = len(y)
    st = design_statistics(cells, E, y, K, levels, alpha * .5, seed)
    S = st["keep"]; cS, yS = cells[S], y[S]; N = len(S)
    tL, tU, tQ = table_box(cS, yS, K, alpha / 10); st["gx"] = (tL, tU)
    m0L, m0U, _ = marginal_box(yS, K, alpha / 20)
    half = np.random.default_rng(seed + 7).permutation(N) % 2; nli, mU, mpt, info = 0.0, 0.0, 0.0, []
    if prediction_builders is None:
        raise ValueError("Pythia requires frozen external prediction_builders; use the Horos adapter for model fitting")
    J = len(prediction_builders)
    for f in (0, 1):
        loc = np.flatnonzero(half == f); rows = S[loc]; tr = np.ones(n, bool); tr[rows] = False; w = len(rows) / N
        bs, us = [], []
        for builder in prediction_builders:
            Qp = np.asarray(builder(np.flatnonzero(tr), rows, K), float)
            if Qp.shape != (len(rows), K) or not np.isfinite(Qp).all():
                raise ValueError("prediction_builder returned an invalid frozen probability matrix")
            b, inf_ = nli_cohort_bound(
                cells[rows], Qp, y[rows], alpha / (20 * J),
                concentration=nli_concentration,
            ); bs.append(b); info.append(inf_)
            u, pt = model_ceiling(Qp, y[rows], K, alpha / (20 * J)); us.append((u, pt))
        nli += w * max(bs); u, pt = min(us); mU += w * u; mpt += w * pt
    st["nliL"], st["mU"] = nli, mU
    pair_U = min(h for _, h in st["box"])
    out = dict(rows=S, n_S=N, table=(tL, tU), table_point=tQ, marginal=(m0L, m0U), nli_lower=nli, nli_info=info,
               model_upper=mU, model_point=mpt, pair_upper=pair_U)
    out["G_upper"] = unified_lp(st, gamma, None, "G", "max")
    out["V_lower"] = unified_lp(st, gamma, None, "V", "min")
    out["inconsistent"] = out["G_upper"] is None      # an assumption-free certificate failed (probability <= alpha)
    if out["inconsistent"]:
        out["G_upper"], out["V_lower"] = min(pair_U, tU, mU), nli
    gl = unified_lp(st, gamma, 0.0, "G", "min"); out["refuted"] = gl is None
    eps0 = 0.0 if gl is not None else _bisect(lambda e: unified_lp(st, gamma, e, "G", "min") is None, 0.0, 1.0)
    out["eps_min_feasible"] = eps0
    out["G_lower_gamma"] = gl if gl is not None else (unified_lp(st, gamma, eps0, "G", "min") or 0.0)
    out["eps_tipping"] = _bisect(lambda e: (unified_lp(st, gamma, e, "G", "min") or 0.0) > 1e-9, eps0, 1.0) \
        if out["G_lower_gamma"] > 1e-9 else eps0
    out["binding_upper"] = min((("pair", pair_U), ("table-NLI", tU - nli), ("model", mU)), key=lambda t: t[1])[0]
    out["G"] = (out["G_lower_gamma"], out["G_upper"])
    out["table_information"] = (max(m0L - tU, 0.0), max(m0U - tL, 0.0))
    GL, GU = out["G"]
    out["losses"] = dict(brier_half=(GL / 2, GU / 2), zero_one=(float(f_K(GL, K)), GU), log_lower=GL,
                         text_info_brier_half=out["V_lower"] / 2, text_cond_MI_lower_nats=out["V_lower"] / 2)
    if population:
        out["G_population"] = (kl_bernoulli_lower(GL, N, alpha / 20), kl_bernoulli_upper(GU, N, alpha / 20))
    return out


def stratified_certificate(strata, cells, E, F, y, K, alpha=.05, gamma=.5, levels=3, seed=0, min_rows=128, model_factory=None,
                           models=None, nli_concentration="hoeffding", prediction_builders=None):
    """Table-signal-conditional form: the unified certificate inside each stratum s of a coarse table variable
    (alpha split proportional to stratum size); global joint floor bound = sum_s w_s [L_s, U_s].
    Strata with fewer than min_rows rows are left out of the cohort."""
    strata = np.asarray(strata); n = len(y); ids = np.unique(strata); big = [s for s in ids if (strata == s).sum() >= min_rows]
    nb = sum((strata == s).sum() for s in big); res = {}
    for s in big:
        I = np.flatnonzero(strata == s); a_s = alpha * len(I) / nb
        local_builders = None
        if prediction_builders is not None:
            local_builders = []
            for builder in prediction_builders:
                local_builders.append(lambda tr, te, k, b=builder, idx=I: b(idx[tr], idx[te], k))
        res[s] = unified_certificate(
            cells[I], E[I], F[I], y[I], K, a_s, gamma, levels, seed,
            model_factory, population=False, models=None,
            nli_concentration=nli_concentration,
            prediction_builders=local_builders,
        )
    NS = sum(r["n_S"] for r in res.values())          # estimand: mean g over the union of the strata cohorts
    L = sum(r["n_S"] / NS * r["G"][0] for r in res.values()); U = sum(r["n_S"] / NS * r["G"][1] for r in res.values())
    V = sum(r["n_S"] / NS * r["V_lower"] for r in res.values())
    rows = {s: np.flatnonzero(strata == s)[r["rows"]] for s, r in res.items()}
    return dict(G=(L, U), V_lower=V, strata=res, rows=rows, n_S=NS)
