"""Development of geometry-weighted Q90 e-mixtures with nonzero penalties."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from peras import mixture_dkelly_pinball_portfolio
from run_multipoint_pinball_development import Q, gold as base_gold, quantile_function


K, CAP, N_PARTITIONS = 16, .16, 20
TAU_FACTORS = (.05, .1, .25, .5, 1.)


def scales(design, x):
    if design == 1: return .6+.8*x
    if design == 2: return .7+.6*x*x
    if design == 3: return .65+.7*(x>.5)
    if design == 4: return .6+.8*np.abs(x-.5)*2
    if design == 5: return .8+.5*np.sin(np.pi*x)**2
    if design == 6: return .65+.6*(np.arange(len(x))%5==0)
    if design == 7: return .6+.8*x**3
    if design == 8: return .7+.6*(np.sin(3*np.pi*x)**2)
    if design == 9: return .6+.8*(np.abs(x-.5)<.15)
    if design == 10: return .65+.7*(x>.8)
    raise ValueError("design must be 1--10")


@lru_cache(maxsize=None)
def base_mean(design):
    u=(np.arange(1_000_000)+.5)/1_000_000
    return float(np.mean(quantile_function(design,u)))


def registered_partitions(x, seed):
    rng=np.random.default_rng(seed+12_000_000)
    sigmas=np.linspace(0,.18,N_PARTITIONS)
    out=[]
    for sigma in sigmas:
        order=np.argsort(x+rng.normal(0,sigma,len(x)),kind="stable")
        out.append(order.reshape(-1,K))
    return np.stack(out)


def discrepancies(partitions, scale, mean_q):
    values=[]
    for partition in partitions:
        sb=scale[partition]
        values.append(float(mean_q*np.mean(np.abs(
            sb-sb.mean(axis=1,keepdims=True)))))
    return np.asarray(values)


def endpoint(y, partitions, penalty, weights=None, weight_name=None):
    kwargs={}
    if weights is not None:
        kwargs={"partition_weights_by_k":{K:weights},
                "weight_provenance":weight_name}
    return mixture_dkelly_pinball_portfolio(
        y,{K:partitions},quantile=Q,clip=(0,2),
        declared_partition_discrepancies={K:penalty},score_caps=(CAP,),
        alpha_family=.05,partition_provenance="frozen noisy-rank feature partitions",
        discrepancy_provenance="analytic comonotone scale-law discrepancy",
        portfolio_frozen_before_outcomes=True,**kwargs).bound


def run(design,seed):
    x=(np.arange(800)+.5)/800; scale=scales(design,x)
    y=scale*quantile_function(design,np.random.default_rng(seed).random(800))
    parts=registered_partitions(x,seed); p=discrepancies(parts,scale,base_mean(design))
    best=int(np.argmin(p))
    row={"design":design,"seed":seed,
         "gold":float(np.mean(scale)*base_gold(design)),
         "penalties":p.tolist(),"best_partition":best,
         "single_best_geometry":endpoint(y,parts[best:best+1],p[best:best+1]),
         "equal_mixture":endpoint(y,parts,p)}
    positive=p[p>1e-15]; reference=float(np.median(positive)) if len(positive) else 1.
    for factor in TAU_FACTORS:
        weights=np.exp(-(p-np.min(p))/max(factor*reference,1e-12))
        row[f"softmax_{factor}"]=endpoint(
            y,parts,p,weights,f"softmax declared penalty, tau={factor}*median")
    return row


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1,11):
            for seed in range(138001,138101):
                handle.write(json.dumps(run(design,seed))+"\n");handle.flush()
        print(args.out)
