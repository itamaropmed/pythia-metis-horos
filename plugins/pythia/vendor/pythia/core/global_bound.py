"""Three-gap certificate (PYTHIA-G3): loss-agnostic, model-free bounds on the best achievable risk from a table X and its
text T, the gaps around it, and per-sample risk labels.

For a loss l and rows independent given records:
  R*_l(X)    best risk of ANY rule using the table          two-sided, assumption-free (cells are exact replicates)
  R*_l(X,T)  best risk of ANY rule using table + text       upper: assumption-free (neighbour-label predictors);
                                                            lower: needs an assumption (no-free-lunch) -> PYTHIA-X gamma
  gaps:  distribution gap  = width of the certified intervals (finite n, unknown P)
         data/information gap = R*_l(X) - R*_l(X,T)          lower bound, assumption-free
         estimation gap   = R_l(f) - R*_l(X,T)              lower bound assumption-free (headroom any model could still gain
                                                            over f); upper bound needs the gamma lower floor
Key lemma (neighbour-label predictor, ANY loss).  If h is any function of OTHER rows' labels and of all records, then
h is independent of Y_i given the records, so E l(Y_i, h) >= R*_l(record_i).  Averaging leave-one-out losses inside
outcome-blind blocks therefore upper-bounds the Bayes floor for every loss, with no smoothness, no model and no
assumption on how similar the block members are.  Block means are independent, so Hoeffding/betting apply.
Lower bound inside table cells (ANY loss with an exact in-cell minimiser): S = (1/n) sum_c min_a sum_{i in c} l(Y_i,a)
<= (1/n) sum_i l(Y_i, a*_c) (fixed-action domination), and the right side concentrates around R*_l(X)."""
from __future__ import annotations
import math
import numpy as np
from .geometry import knn_query
from .multiscale_pair_extrapolation import kl_bernoulli_lower, kl_bernoulli_upper, betting_lower, betting_upper

# ------------------------------------------------------------------ losses: (pointwise loss, in-cell exact minimum, range)
def loss_value(name, y, pred, K=None, eps=1e-3, q=.9):
    if name == "01":
        return (y != pred).astype(float)
    if name == "brier":                    # pred: probability vectors
        return ((pred - np.eye(K)[y]) ** 2).sum(-1) / 2          # in [0,1]
    if name == "log":                      # pred: probability vectors, floored at eps (bounded loss)
        p = np.clip(pred[np.arange(len(y)), y], eps, 1); return -np.log(p)
    if name == "mse":
        return (y - pred) ** 2
    if name == "mae":
        return np.abs(y - pred)
    if name == "pinball":
        u = y - pred; return np.maximum(q * u, (q - 1) * u)
    raise ValueError(name)


def loss_range(name, K=None, eps=1e-3, y_range=None):
    return {"01": 1.0, "brier": 1.0, "log": -math.log(eps)}.get(name, None) or \
           {"mse": (y_range or 1) ** 2, "mae": (y_range or 1), "pinball": (y_range or 1)}[name]


def cell_min(name, yc, K=None, q=.9):
    """Exact min_a sum_i l(y_i, a) inside a cell (sum, not mean)."""
    if name == "01":
        return len(yc) - np.bincount(yc, minlength=K).max()
    if name == "brier":
        f = np.bincount(yc, minlength=K) / len(yc); return len(yc) * (1 - (f ** 2).sum()) / 2
    if name == "mse":
        return float(((yc - yc.mean()) ** 2).sum())
    if name == "mae":
        return float(np.abs(yc - np.median(yc)).sum())
    if name == "pinball":
        a = np.quantile(yc, q, method="inverted_cdf"); u = yc - a; return float(np.maximum(q * u, (q - 1) * u).sum())
    raise ValueError(name)


# ------------------------------------------------------------------ blocks
def text_blocks(cells, E, k, seed=0):
    """Outcome-blind blocks of size k inside table cells, greedily grown by text nearest neighbours (random tie-breaks)."""
    rng = np.random.default_rng(seed); blocks = []
    for c in np.unique(cells):
        I = np.flatnonzero(cells == c); I = I[rng.permutation(len(I))]
        if len(I) < k:
            continue
        G = E[I] + 1e-9 * rng.normal(size=E[I].shape)
        _, all_idx = knn_query(G, k=min(len(I), 4 * k)); free = np.ones(len(I), bool)
        for s in range(len(I)):
            if not free[s]:
                continue
            cand = [j for j in all_idx[s] if free[j]]
            if len(cand) < k:
                rest = np.flatnonzero(free); d = ((G[rest] - G[s]) ** 2).sum(1); cand = list(rest[np.argsort(d)])
            if len(cand) < k:
                break
            blk = cand[:k]; free[blk] = False; blocks.append(I[blk])
    return blocks


def loo_predict(name, ys, K=None, lam=.5, q=.9):
    """Leave-one-out aggregator from the other block members' labels (a valid predictor for each member)."""
    m = len(ys); out = []
    for i in range(m):
        o = np.delete(ys, i)
        if name == "01":
            out.append(np.bincount(o, minlength=K).argmax())
        elif name in ("brier", "log"):
            out.append((np.bincount(o, minlength=K) + lam) / (len(o) + lam * K))
        elif name == "mse":
            out.append(o.mean())
        elif name == "mae":
            out.append(np.median(o))
        else:
            out.append(np.quantile(o, q, method="inverted_cdf"))
    return np.array(out)


# ------------------------------------------------------------------ certificates
def _upper_mean(z, B, alpha):
    zz = np.clip(z / B, 0, 1); return B * min(betting_upper(zz, alpha / 2), kl_bernoulli_upper(float(zz.mean()), len(zz), alpha / 2))


def _lower_mean(z, B, alpha):
    zz = np.clip(z / B, 0, 1); return B * max(betting_lower(zz, alpha / 2), kl_bernoulli_lower(float(zz.mean()), len(zz), alpha / 2))


def joint_floor_upper(name, cells, E, y, K=None, ks=(2, 4, 8), alpha=.05, y_range=None, seed=0):
    """Assumption-free UPPER bound on R*_l(X,T) for ANY loss: LOO neighbour-label predictors in text-NN blocks inside
    table cells; block means independent; best block size, Bonferroni."""
    B = loss_range(name, K, y_range=y_range); best = (np.inf, None)
    for k in ks:
        blocks = text_blocks(cells, E, k, seed)
        if not blocks:
            continue
        bm = np.array([loss_value(name, y[b], loo_predict(name, y[b], K), K).mean() for b in blocks])
        nb = sum(len(b) for b in blocks); n = len(y)
        # rows left out of blocks (small cells, remainders) get the trivial bound B: the estimand stays ALL rows
        U = (nb / n) * _upper_mean(bm, B, alpha / len(ks)) + (1 - nb / n) * B
        if U < best[0]:
            best = (U, k)
    return dict(upper=best[0], k=best[1])


def table_floor_bounds(name, cells, y, K=None, alpha=.05, y_range=None, ks=(2, 4, 8), seed=0):
    """Two-sided, assumption-free bounds on R*_l(X) (table only).  Lower: exact in-cell SAA + domination (row-level
    concentration of the Bayes-action losses).  Upper: LOO in random within-cell blocks (any loss)."""
    B = loss_range(name, K, y_range=y_range); n = len(y)
    if name == "log":     # Shannon entropy (nats) >= 1 - |p|^2 = 2 x (Brier/2 floor): transfer the Brier lower bound
        L = 2 * table_floor_bounds("brier", cells, y, K, alpha, y_range, ks, seed)["lower"]; S = None
    else:
        S = sum(cell_min(name, y[cells == c], K) for c in np.unique(cells)) / n
        L = B * kl_bernoulli_lower(min(S / B, 1.0), n, alpha / 2)   # S <= mean of n independent [0,B] variables
    rng = np.random.default_rng(seed); Erand = rng.normal(size=(n, 1))   # random geometry = random blocks in cells
    U = joint_floor_upper(name, cells, Erand, y, K, ks, alpha / 2, y_range, seed)["upper"]
    return dict(lower=L, upper=U, saa=S)


def model_risk_bounds(name, y, pred, K=None, alpha=.05, y_range=None):
    B = loss_range(name, K, y_range=y_range); z = loss_value(name, y, pred, K)
    return dict(lower=_lower_mean(z, B, alpha), upper=_upper_mean(z, B, alpha), point=float(z.mean()))


def three_gap_certificate(name, cells, E, y, model_pred_holdout, K=None, alpha=.05, y_range=None, gamma_lower=None, seed=0):
    """Simultaneous (alpha/5 each) report.  model_pred_holdout: out-of-fold predictions of a model f on the SAME rows."""
    a = alpha / 5
    tab = table_floor_bounds(name, cells, y, K, a, y_range, seed=seed)
    up = joint_floor_upper(name, cells, E, y, K, alpha=a, y_range=y_range, seed=seed)
    mr = model_risk_bounds(name, y, model_pred_holdout, K, a, y_range)
    joint_U = min(up["upper"], tab["upper"])
    dI_L = 0.0
    if name in ("01", "brier", "log"):
        from .anchored_information import anchored_information
        ai = anchored_information(cells, E, y, K, a, seed=seed)       # Brier pair route (AFI): tighter for 01/brier
        G_U = ai["GXT_upper"]                                           # upper on mean(1-|p|^2)
        joint_U = min(joint_U, {"01": G_U, "brier": G_U / 2, "log": np.inf}[name])
        dI_L = ai["dI_lower"]                                           # Brier information beyond the table
    lowJ = gamma_lower if gamma_lower is not None else 0.0                          # assumption-dependent side
    return dict(loss=name, table_floor=(tab["lower"], tab["upper"]), joint_floor=(lowJ, joint_U),
                data_gap_lower=max(tab["lower"] - joint_U, {"brier": dI_L / 2, "log": dI_L / 2}.get(name, 0.0), 0.0),
                # brier: (Delta I_A)/2 in Brier/2 units; log: conditional MI >= Delta I_A / 2 (Pinsker + l2 <= l1)
                estimation_gap_lower=max(mr["lower"] - joint_U, 0.0),                   # R(f) - R*(X,T) >= ...
                estimation_gap_upper=mr["upper"] - lowJ, model_risk=(mr["lower"], mr["upper"]), block_k=up["k"])


# ------------------------------------------------------------------ per-sample risk labels
def local_irreducibility(cells_cal, E_cal, y_cal, cells_new, E_new, K, k=15):
    """Model-free per-sample irreducibility score: Gini disagreement among the k nearest calibration texts in the same
    table cell (an estimate of g(p(x,t)) + local drift; never uses the new sample's label)."""
    out = np.zeros(len(cells_new))
    for c in np.unique(cells_new):
        J = np.flatnonzero(cells_cal == c); I = np.flatnonzero(cells_new == c)
        if len(J) < 3:
            out[I] = np.nan; continue
        kk = min(k, len(J)); _, idx = knn_query(E_cal[J], E_new[I], k=kk)
        lab = y_cal[J][idx]
        for t, row in enumerate(lab):
            cnt = np.bincount(row, minlength=K).astype(float); m = cnt.sum()
            out[I[t]] = 1 - (cnt * (cnt - 1)).sum() / (m * (m - 1))
    return out


def certified_easy_threshold(score_cal, err_cal, eps, delta=.05, grid=None):
    """Learn-then-Test style: the largest score threshold s such that the error rate among calibration rows with
    score <= s is certified <= eps (binomial/KL upper bound at level delta, fixed-sequence testing from low scores, so
    no multiplicity correction is needed).  Guarantee: P(error rate of the 'easy' group > eps) <= delta."""
    grid = np.unique(np.quantile(score_cal, np.linspace(.02, 1, 50))) if grid is None else grid
    best = None
    for s in grid:
        sel = score_cal <= s
        if sel.sum() < 20:
            continue
        U = kl_bernoulli_upper(float(err_cal[sel].mean()), int(sel.sum()), delta)
        if U <= eps:
            best = s
        else:
            break
    return best
