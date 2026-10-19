"""Development comparison for capped overlapping Q90 witnesses."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from peras import overlapping_pinball_barycentric_portfolio
from peras.core.concentration import bentkus_lower_mean
from peras.core.truncated_saa import dkelly_lower_mean


KS = (8, 16, 32, 40, 80)
CAPS = (.005, .01, .02, .04, .08)
Q = .9


def quantile_function(design, u):
    u = np.asarray(u)
    if design == 1: return u
    if design == 2: return u**2
    if design == 3: return np.sqrt(u)
    if design == 4: return beta.ppf(u, 2, 5)
    if design == 5: return beta.ppf(u, .5, .5)
    if design == 6: return np.floor(5*u)/4
    if design == 7: return np.where(u < .98, .4*u/.98, .8+.2*(u-.98)/.02)
    if design == 8: return u + .1*np.sin(2*np.pi*u)
    if design == 9: return (np.exp(4*u)-1)/(np.exp(4)-1)
    if design == 10: return np.where(u < .7, .3*u/.7, .3+.7*(u-.7)/.3)
    raise ValueError("design must be 1--10")


@lru_cache(maxsize=None)
def gold(design):
    u = (np.arange(1_000_000)+.5)/1_000_000
    y = quantile_function(design, u)
    action = float(quantile_function(design, np.array([Q]))[0])
    residual = y-action
    return float(np.mean(np.maximum(Q*residual, (Q-1)*residual)))


def score(y, blocks):
    values = y[blocks]
    action = np.quantile(values, Q, axis=1, method="inverted_cdf")
    residual = values-action[:, None]
    return np.mean(np.maximum(Q*residual, (Q-1)*residual), axis=1)


def partitions(n, seed, repeats=20):
    rng = np.random.default_rng(seed+9_000_000)
    return {k: np.stack([rng.permutation(n).reshape(-1,k)
                         for _ in range(repeats)]) for k in KS}


def finite_range(k):
    counts=np.arange(k+1)
    factor=np.max(np.minimum(Q*(k-counts)/k,(1-Q)*counts/k))
    return float(factor)


def baselines(y, route):
    pair = route[8][0].reshape(-1)[:len(y)].reshape(-1,2)
    pair_score=score(y,pair)
    paired=float(bentkus_lower_mean(pair_score,finite_range(2),.05))
    candidates=[]
    for k in (2,4,8,16):
        rng=np.random.default_rng(77+k)
        blocks=rng.permutation(len(y)).reshape(-1,k)
        raw=score(y,blocks)
        for cap in CAPS:
            c=min(cap,finite_range(k))
            candidates.append(dkelly_lower_mean(
                np.minimum(raw,c),cap=c,alpha=.05/(4*len(CAPS))))
    return paired,float(max(candidates))


def run(design,seed):
    rng=np.random.default_rng(seed)
    y=quantile_function(design,rng.random(800))
    route=partitions(800,seed)
    multi=overlapping_pinball_barycentric_portfolio(
        y,route,quantile=Q,clip=(0,1),
        declared_discrepancy_by_k={k:0. for k in KS},score_caps=CAPS,
        alpha_family=.05,partition_provenance="20 frozen random balanced partitions",
        discrepancy_provenance="iid exact-law mechanism",
        portfolio_frozen_before_outcomes=True)
    paired,disjoint=baselines(y,route)
    return {"design":design,"seed":seed,"gold":gold(design),
            "multipoint":multi.bound,"selected_k":multi.selected_k,
            "selected_cap":multi.selected_score_cap,
            "paired_bentkus":paired,"disjoint_dkelly":disjoint}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1,11):
            for seed in range(136001,136101):
                handle.write(json.dumps(run(design,seed))+"\n");handle.flush()
        print(args.out)
