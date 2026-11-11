"""Known-truth simulations for the tighter scalar certificate (pythia/core/tight_bounds.py).

World: G table cells; true mean residual of the deployed table model m_i = (signal from note facts) ; duration noise
N(0, 20^2) or a heavy-tailed t3 (scaled to sd 20).  A reader sees the note facts through noise (quality rho).
Target: V = within-cell variance of E[clip(r, +-cap) | design] on the certification rows (Monte-Carlo exact).

Methods (all at one pre-declared alpha = 0.05; cap = 60 declared):
  split_mp      FIT -> CAL, Maurer-Pontil empirical Bernstein (previous default), taming tuned on FIT
  split_bet     FIT -> CAL, predictable-plug-in EB / betting with lambda from the FIT pilot
  xfit_bet      2-fold cross-certification on FIT u CAL, betting, alpha/2 per fold
  split_clt     FIT -> CAL normal approximation (reference only; not a finite-sample certificate)
Report: certified share of truth (mean), miss rate P(bound > truth) (must be <= alpha), null false-claim rate.
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from pythia.core.tight_bounds import scalar_cert, tune_cert, tame, crossfit_certificate, centre_within  # noqa

ALPHA, CAP, NOISE = .05, 60., 20.


def ridge_score(Xtr, rtr, gtr, Xev, gev, lam=10.):
    Xc = Xtr - Xtr.mean(0); sd = Xc.std(0) + 1e-9
    def cc(X, g):
        X = X.copy()
        for c in np.unique(g):
            I = g == c; X[I] -= X[I].mean(0)
        return X / sd
    A = cc(Xtr, gtr); rc = centre_within(rtr, gtr)
    b = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ rc)
    return cc(Xev, gev) @ b


def world(n, signal_sd, rho, heavy, rng, G=12, p=8, junk=30):
    g = rng.integers(0, G, n).astype(str)
    F = rng.random((n, p)) < rng.uniform(.05, .4, p)                    # binary "note facts"
    beta = rng.normal(0, 1, p); s = (F @ beta); s = (s - s.mean()) / (s.std() + 1e-9) * signal_sd if signal_sd > 0 else 0 * s
    cell_eff = rng.normal(0, 15, G)[g.astype(int)]
    mu = 90 + cell_eff + s
    eps = (rng.standard_t(3, n) / np.sqrt(3)) * NOISE if heavy else rng.normal(0, NOISE, n)
    y = mu + eps
    X = np.column_stack([F + rng.normal(0, rho, F.shape), rng.normal(0, 1, (n, junk))])   # noisy reader + junk
    return g, mu, y, X


def capped_truth(m_minus_f, heavy, rng, draws=1500):
    e = (rng.standard_t(3, (draws, 1)) / np.sqrt(3)) * NOISE if heavy else rng.normal(0, NOISE, (draws, 1))
    return np.clip(m_minus_f[None, :] + e, -CAP, CAP).mean(0)


def wvar(m, g):
    return float(np.mean((m - centre_within(m, g) - m) ** 2)) if False else float(np.mean(centre_within(m, g) ** 2))


def one(n, signal_sd, rho, heavy, seed):
    rng = np.random.default_rng(seed)
    g, mu, y, X = world(n, signal_sd, rho, heavy, rng)
    u = rng.random(n); fit, cal = u < .4, (u >= .4) & (u < .7)
    cm = {c: y[fit & (g == c)].mean() for c in np.unique(g)}
    f = np.array([cm.get(c, y[fit].mean()) for c in g]); r = y - f
    mc = capped_truth(mu - f, heavy, rng)
    out = {}
    # ---- split methods: train on FIT, tune on FIT out-of-fold, certify on CAL
    ifit = np.flatnonzero(fit); fo = rng.permutation(len(ifit)) % 2; oof = np.zeros(len(ifit))
    for k in (0, 1):
        a, b = ifit[fo != k], ifit[fo == k]
        oof[fo == k] = ridge_score(X[a], r[a], g[a], X[b], g[b])
    raw = ridge_score(X[fit], r[fit], g[fit], X[cal], g[cal]); V_cal = wvar(mc[cal], g[cal])
    for meth in ("mp", "betting", "clt"):
        cfg = tune_cert(oof, r[fit], g[fit], ALPHA, int(cal.sum()), caps=(CAP,), method="mp" if meth == "clt" else meth)
        h = tame(raw, g[cal], cfg["q"])
        ce = scalar_cert(h, r[cal], CAP, ALPHA, method=meth, lam=cfg["lam"] if meth == "betting" else None, groups=g[cal])
        out[f"split_{meth}"] = (ce.bound, V_cal)
    # ---- cross-certification on FIT u CAL
    lab = fit | cal; il = np.flatnonzero(lab)
    res = crossfit_certificate(ridge_score, X[il], r[il], g[il], ALPHA, K=2, seed=seed, method="betting", cap=CAP)
    # truth for the pooled target: sum_k n_k V_k / n with V_k = within-cell variance inside fold k
    rng2 = np.random.default_rng(seed); fold = rng2.permutation(len(il)) % 2
    Vp = sum((fold == k).sum() * wvar(mc[il][fold == k], g[il][fold == k]) for k in (0, 1)) / len(il)
    out["xfit_bet"] = (res["bound"], Vp)
    out["_plugin_split"] = (max(np.mean(tame(raw, g[cal], .02) * np.clip(r[cal], -CAP, CAP)), 0) ** 2 /
                            np.mean(tame(raw, g[cal], .02) ** 2), V_cal)
    return out


def main(reps=120):
    grid = [(n, s, rho, hv) for n in (1500, 3400, 8000) for s in (12., 0.) for rho in (.5,) for hv in (False, True)]
    grid += [(3400, 12., 1.5, False), (3400, 6., .5, False)]
    table = []; t0 = time.time()
    for (n, s, rho, hv) in grid:
        acc = {}
        for rep in range(reps):
            for k, (b, v) in one(n, s, rho, hv, 1000 * rep + int(n) + int(10 * s)).items():
                acc.setdefault(k, []).append((b, v))
        row = {"n": n, "signal_sd": s, "reader_noise": rho, "heavy_tail": hv}
        for k, L in acc.items():
            B = np.array(L)
            row[k] = {"share_of_truth": float(np.mean(B[:, 0] / np.maximum(B[:, 1], 1e-9))) if s > 0 else None,
                      "miss_rate": float(np.mean(B[:, 0] > B[:, 1] + 1e-9)),
                      "claim_rate": float(np.mean(B[:, 0] > 0)), "mean_bound": float(B[:, 0].mean()),
                      "mean_truth": float(B[:, 1].mean())}
        table.append(row)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                                                                 for kk, vv in v.items() if kk in ("share_of_truth", "miss_rate", "claim_rate")})
                          for k, v in row.items()}), f"[{time.time() - t0:.0f}s]", flush=True)
    (Path(__file__).parent / "bound_sims.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
