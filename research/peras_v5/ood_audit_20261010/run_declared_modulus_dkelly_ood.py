"""Frozen fresh audit of declared-modulus concentration and block-size routes.

Registration before opening results:
- existing ten synthetic tabular/text mechanisms, n=800.  The initial audit
  used seeds 113001--113005; the factorial attribution audit added after that
  result uses untouched seeds 114001--114005;
- external median shift fitted on an independent n=4000 draw from the same
  mechanism (this is not Peras training);
- residual clip [-150,150], outcome-blind greedy feature blocks;
- legacy route: k=(2,4,8,16), Bentkus, three fixed score caps;
- betting route: k=(2,4,8,16,32,64), dKelly, same caps;
- joint route: both Bentkus and dKelly on the extended k grid;
- alpha=.05 separately for each route/loss; every internal candidate is
  Bonferroni-accounted;
- oracle barycentric discrepancies are used only to test the theorem and
  sensitivity geometry, never as an operational estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from audit_ood import make
from declared_modulus_certificate import declared_modulus_portfolio
import gold
import peras5 as P
from run_coupled_environment_ood import ppf
from run_disjoint_transport_audit import greedy_feature_blocks


LOSSES = ("MAE", "MSE", "Q10", "Q50", "Q90")
CAPS = {
    "MAE": (40., 80., 160.), "MSE": (2000., 4000., 8000.),
    "Q10": (5., 10., 20.), "Q50": (20., 40., 80.),
    "Q90": (5., 10., 20.),
}
LEGACY_KS = (2, 4, 8, 16)
EXTENDED_KS = (2, 4, 8, 16, 32, 64)
U = (np.arange(2000) + .5) / 2000


def raw_features(d):
    return np.column_stack([d["tab"], d["llm"]])


def independent_shift(design: int, seed: int, d: dict) -> np.ndarray:
    train = make(design, 4000, seed + 600000)
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=.5, max_iter=300, learning_rate=.05,
        random_state=0).fit(raw_features(train), train["y"])
    return model.predict(raw_features(d))


def oracle_declarations(d: dict, blocks_by_k: dict, shift: np.ndarray,
                        clip=(-150., 150.)) -> tuple[dict, dict]:
    quantiles = np.empty((len(d["mu"]), len(U)))
    for family in np.unique(d["fam"]):
        mask = d["fam"] == family
        quantiles[mask] = (d["mu"][mask, None]
                           + d["s"][mask, None] * ppf(family, U)[None, :])
    quantiles = np.clip(quantiles - shift[:, None], *clip)
    absolute, squared = {}, {}
    for k, blocks in blocks_by_k.items():
        q = quantiles[blocks]
        deviation = q - q.mean(axis=1, keepdims=True)
        absolute[k] = float(np.mean(np.abs(deviation)))
        squared[k] = float(np.mean(deviation ** 2))
    return absolute, squared


def route(y, blocks, shift, loss, declarations, *, ks, methods, name):
    selected_blocks = {k: blocks[k] for k in ks}
    selected_declarations = {k: declarations[k] for k in ks}
    result = declared_modulus_portfolio(
        y, selected_blocks, loss=loss, score_caps=CAPS[loss],
        clip=(-150., 150.), declared_discrepancy=selected_declarations,
        alpha_family=.05, shift=shift,
        shift_provenance="independent n=4000 median model, registered seed offset",
        block_provenance="greedy disjoint blocks on frozen Peras representation",
        methods=methods)
    return {key: result[key] for key in (
        "bound", "selected_k", "selected_method", "selected_score_cap",
        "selected_lower_score_mean", "declared_discrepancy",
        "tipping_discrepancy", "retained_fraction", "alpha_per_candidate")}


def run(design: int, seed: int) -> dict:
    d = make(design, 800, seed)
    representation = P.represent(d)
    shift = independent_shift(design, seed, d)
    blocks = {k: greedy_feature_blocks(representation, k) for k in EXTENDED_KS}
    p_abs, p_sq = oracle_declarations(d, blocks, shift)
    pointwise = gold.pointwise(d["fam"], d["s"])
    pointwise["Q50"] = pointwise["MAE"] / 2
    row = {
        "design": design, "seed": seed,
        "gold": {loss: float(np.mean(pointwise[loss])) for loss in LOSSES},
        "routes": {},
    }
    for loss in LOSSES:
        declarations = p_sq if loss == "MSE" else p_abs
        row["routes"][loss] = {
            "legacy_bentkus": route(d["y"], blocks, shift, loss, declarations,
                                     ks=LEGACY_KS, methods=("Bentkus",), name="legacy"),
            "same_grid_dkelly": route(d["y"], blocks, shift, loss, declarations,
                                       ks=LEGACY_KS, methods=("dKelly_betting",), name="same_grid_betting"),
            "extended_bentkus": route(d["y"], blocks, shift, loss, declarations,
                                       ks=EXTENDED_KS, methods=("Bentkus",), name="extended_bentkus"),
            "extended_dkelly": route(d["y"], blocks, shift, loss, declarations,
                                      ks=EXTENDED_KS, methods=("dKelly_betting",), name="betting"),
            "joint": route(d["y"], blocks, shift, loss, declarations,
                           ks=EXTENDED_KS, methods=("Bentkus", "dKelly_betting"), name="joint"),
        }
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=113001)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(args.seed_start, args.seed_start + args.reps):
                result = run(design, seed)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(design, seed, {
                    loss: {name: round(value["bound"] / result["gold"][loss], 3)
                           for name, value in result["routes"][loss].items()}
                    for loss in LOSSES}, flush=True)
