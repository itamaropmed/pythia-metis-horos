"""Pipeline-level simulator for Pythia local W30 certificates (Mayo-shaped, known conditional law)."""
from __future__ import annotations
import sys, os, time
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pythia.core.local_model_risk import build_local_model_risk_registry

BASE = np.array([45, 60, 75, 90, 110, 130, 160, 200.])
FPROB = np.array([.18, .17, .15, .13, .12, .1, .09, .06])
THR = 30.0


def population(n_pat, rng, shift=None, pid0=0):
    """shift: dict with optional keys family_prob, sigma_mult (array per family), mean_shift (per family),
    z_mean."""
    shift = shift or {}
    k = 1 + np.minimum(rng.poisson(.45, n_pat), 3)
    pid = np.repeat(np.arange(pid0, pid0 + n_pat), k)
    n = len(pid)
    u = np.repeat(rng.normal(size=n_pat), k)
    F = rng.choice(8, n, p=shift.get("family_prob", FPROB))
    z = rng.normal(shift.get("z_mean", 0.0), 1, n)
    h = rng.normal(size=n)
    x3 = rng.normal(size=n)
    t = (h + rng.normal(size=n) > 0).astype(float)
    sig = (15 + 3 * F) * np.asarray(shift.get("sigma_mult", np.ones(8)))[F]
    mu = BASE[F] + 20 * z + 12 * h + np.asarray(shift.get("mean_shift", np.zeros(8)))[F]
    s = np.sqrt(100 + sig ** 2)                        # u and eps integrated out analytically
    y = mu + 10 * u + sig * rng.normal(size=n)
    XA = np.column_stack([F, z, x3]); XB = np.column_stack([F, z, x3, t])
    return dict(pid=pid, F=F, XA=XA, XB=XB, y=y, mu=mu, s=s)


def miss_prob(pred, mu, s):
    return norm.cdf((pred - THR - mu) / s) + norm.sf((pred + THR - mu) / s)


def gbm(seed):
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=150, learning_rate=.08,
                                         max_leaf_nodes=31, min_samples_leaf=40, random_state=seed)


def edges(v, bins):
    e = np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]); return np.unique(e)


def labels(F, pa, pb, pe, de):
    pbin = np.digitize((pa + pb) / 2, pe); dbin = np.digitize(np.abs(pa - pb), de)
    f = F.astype(int)
    return {"family_prediction_disagreement": np.char.add(np.char.add(f.astype(str), "|"), (pbin * 10 + dbin).astype(str)),
            "family_prediction": np.char.add(np.char.add(f.astype(str), "|"), pbin.astype(str)),
            "prediction": pbin.astype(str), "global": np.repeat("all", len(f))}


def route_vec(reg, model, lab):
    """Vectorised routing: (level, key) of the finest eligible node per row."""
    n = len(lab["global"]); lv = np.empty(n, object); ky = np.empty(n, object); done = np.zeros(n, bool)
    for level in reg.level_order:
        keys = lab[level]
        for key in np.unique(keys[~done]):
            node = reg.nodes.get(f"{model}|{level}|{key}")
            if node is not None and node.n_entities >= reg.min_entities:
                m = (~done) & (keys == key); lv[m] = level; ky[m] = key; done |= m
    return lv, ky


def relation_of(reg, model, level, key):
    gnode = reg.nodes[f"{model}|global|all"]; node = reg.nodes[f"{model}|{level}|{key}"]
    if level == "global":
        return "global_only"
    if reg.relation_test == "contrast" and node.contrast_lower is not None:
        return ("certified_higher_than_global" if node.contrast_lower > 0 else
                "certified_lower_than_global" if node.contrast_upper < 0 else "local_shift_unresolved")
    return ("certified_higher_than_global" if node.lower > gnode.upper else
            "certified_lower_than_global" if node.upper < gnode.lower else "local_shift_unresolved")


def evaluate(reg, models, lab_later, lab_fresh, p_fresh):
    """Truth = deployed model's population cell risk (fresh sample). Returns metrics dict."""
    out = dict(cover_all=True, false_dir=0, certified=0, unresolved=0, n=0, widths=[], cells_checked=0, miss_cells=0)
    for m in models:
        lv, ky = route_vec(reg, m, lab_later)
        flv, fky = route_vec(reg, m, lab_fresh)
        gtrue = p_fresh[m].mean()
        for (level, key) in set(zip(lv, ky)):
            if level is None:
                continue
            sel = (flv == level) & (fky == key)
            if sel.sum() < 200:
                continue                                # truth not estimable for this cell
            truth = p_fresh[m][sel].mean()                      # risk of the cases that ROUTE here
            fnode = lab_fresh[level] == key
            truth_node = p_fresh[m][fnode].mean()               # risk of the whole node (stated estimand)
            node = reg.nodes[f"{m}|{level}|{key}"]
            out["cells_checked"] += 1
            if not (node.lower - 1e-9 <= truth_node <= node.upper + 1e-9):
                out["cover_all"] = False; out["miss_cells"] += 1
            if not (node.lower - 1e-9 <= truth <= node.upper + 1e-9):
                out["routed_miss_cells"] = out.get("routed_miss_cells", 0) + 1
            rel = relation_of(reg, m, level, key); nrow = int(((lv == level) & (ky == key)).sum())
            if rel.startswith("certified_higher") and not truth > gtrue:
                out["false_dir"] += nrow
            if rel.startswith("certified_lower") and not truth < gtrue:
                out["false_dir"] += nrow
            if (rel.startswith("certified_higher") and not truth_node > gtrue) or (rel.startswith("certified_lower") and not truth_node < gtrue):
                out["false_dir_node"] = out.get("false_dir_node", 0) + nrow
            out["certified"] += nrow * rel.startswith("certified"); out["unresolved"] += nrow * (rel == "local_shift_unresolved")
            out["n"] += nrow; out["widths"] += [node.upper - node.lower] * nrow
    out["median_width"] = float(np.median(out.pop("widths"))) if out["n"] else None
    return out


def registry(losses, lab, pid, **kw):
    return build_local_model_risk_registry(losses, lab, pid, loss_cap=1., alpha=.05, min_entities=100,
                                           outcome_blind_hierarchy=True, independent_entities=True,
                                           loss_cap_attested=True, **kw)
