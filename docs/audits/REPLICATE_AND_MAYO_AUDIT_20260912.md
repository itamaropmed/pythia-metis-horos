# Pythia replicate and Mayo leakage audit — 2026-09-12

Status: retrospective and synthetic research diagnostic. No new Mayo certificate
or general superiority claim is made. Pythia trained no predictor.

## Ten known-truth simulation designs

Reproduce with:

```sh
python3 scripts/run_replicate_adversarial.py --repetitions 20 --output /tmp/pythia_replicate_ten_designs.json
```

The ten generators are linear-uniform, nonlinear-sine, heteroskedastic,
asymmetric-mixture, five-dimensional nonlinear, bounded-heavy-tail,
quadratic, two-way interaction, sparse-ten-dimensional, and bimodal-noise.
Each is tested at 1,000, 4,000, and 10,000 entities for MAE and MSE with 20
independent repetitions: 1,200 runs. The known conditional Bayes risk and
the declared Lipschitz/range caps come from each generator. The comparison
keeps each target estimand fixed: global and blocked conditional bounds target
the same audited covariate population; population-block bounds do not.

| Method/control | Valid / 1,200 | Median usable tightness | Detail |
|---|---:|---:|---|
| Global nearest-pair conditional empirical Bernstein | 1,200 | 0.597 | Wilson lower 95% coverage 0.9968 |
| Same pairs, conditional Hoeffding | 1,200 | 0.285 | Empirical Bernstein was tighter in 957 runs |
| Blocked-pair conditional empirical Bernstein | 1,200 | 0 | Global matching was tighter in 920 runs |
| Half the true Lipschitz cap (invalid control) | 1,153 | 0 | Assumption violation; Wilson lower 0.9483 |
| No Lipschitz correction (invalid control) | 865 | 0.385 | Invalid despite appearing tighter |

The global corrected center had median relative absolute error 0.241 as a
point estimate, versus 0.825 for blocked pairing, and lower absolute error
in 945/1,200 paired runs. It is a lower-bound plug-in, **not** a generally
identified Bayes-MAE estimator. In the sparse-ten-dimensional design, its
median relative error was 1.0 and its median bound was zero. Therefore the
suite does not establish “best estimator” or “tightest bound” across all
designs. It does establish empirical improvement over the matched Hoeffding
bound on many, not all, cases under the same finite-sample contract.

An additional point-estimator check used 20 independent 2,000-entity datasets
from each of the ten designs. Horos computed a leave-one-out 20-neighbor
median (MAE) or mean (MSE) and evaluated residual risk; this trains an
outcome-based local predictor **outside** Pythia and is not a bound. Against
known Bayes risk, kNN had lower absolute error in 190/200 MAE runs (median
0.0121 versus Pythia center 0.0886). Pythia had lower error in 131/200 MSE
runs (median 0.00470 versus kNN 0.00617). This directly rejects a general
“best MAE floor estimator” claim for the current pair center. It does not
compare exact implementations of mutual-kNN or fused-lasso papers.
Reproduce with `python3 scripts/run_knn_floor_comparison.py --repetitions 20 --n-entities 2000`.

## Prespecified sparse-metric control

Reproduce with:

```sh
python3 scripts/run_sparse_metric_audit.py --repetitions 25
```

On the *same* 75 generated sparse datasets, the all-ten-coordinate metric
gave median zero MAE and MSE bounds. If the generator's relevant first
coordinate is declared before sampling outcomes, one-coordinate matching and
the corresponding valid smoothness cap gave median MAE bound 0.1496 against
true 0.25, and MSE bound 0.0421 against true 0.0833. All 75 bounds in each
arm/loss were below truth (Wilson lower 95% 0.9513). The median MSE plug-in
absolute error fell from 0.0833 to 0.00162. This is a *known-support control*:
it does not show that Pythia can learn which clinical variables matter. The
proof permits a predeclared anisotropic center-movement modulus; outcome-based
metric selection would require separate training data and selection control.

## Mayo current retrospective readout

`scripts/run_mayo_replicate.py` was rerun on the canonical derived files:
10,528 unique-patient rows used; best OOF MAE on those rows 32.8288 minutes.
The table gives global fixed-design conditional empirical-Bernstein lower
bound diagnostics, not released certificates:

| Declared smoothness cap `L` | Tabular metric | Tabular + note values/applicability |
|---:|---:|---:|
| 0 | 26.597 | 28.413 |
| 100 | 26.126 | 27.429 |
| 300 | 25.184 | 25.459 |
| 600 | 23.771 | 22.505 |

The metrics have different dimensions and no externally established Mayo
smoothness cap; equal numeric `L` is **not** an equal clinical assumption.
Patient-conditional independence, note timing, and transfer to future cases
also remain unaudited. The legacy 28.8496-minute Arm-B guarded MAE bound and
31.1186-minute `E2` floor estimate remain *calibrated* historical results;
the approximately 24.18-minute Q-cap rung is conservative under its guard,
not a new prospective Mayo result. The alleged 31.2623-minute incumbent is
not reproduced as a held-out final-duration MAE (see
`IMPLEMENTATION_STATUS.md`); no silent replacement is made.

## Deliberate outcome leakage in notes and tabular features

`scripts/run_mayo_oracle_leakage.py` selected 2,048 unique patients with
pre-cutoff note bundles. It appended a literal actual-duration sentence to
selected bundles **in memory only**, parsed that marker, and supplied the
same value as a tabular oracle feature. No modified note, clinical row, or
patient identifier was written. A full-cohort frozen OOF winner had MAE
32.1905 on this diagnostic subset. Substituting the target on 10%, 50%, and
100% of rows gave mixed MAE 29.0118, 16.8501, and 0.0 minutes. On the fully
injected oracle coordinate, the pairwise center with its correct `L=900`
was numerically zero and the bound was zero. The public `certify` API refused
the representation as target-derived and post-prediction-time. These numbers
measure leakage sensitivity, **not** an achievable clinical gain or a valid
Mayo floor.
An independent 4,096-patient diagnostic draw (seed 20260913) repeated the
pattern: baseline 32.2685, mixed MAE 28.9918 / 16.3180 / 0.0 minutes,
full-oracle pair bound zero, and public API refusal.

As a no-new-information control, the same script duplicated the pre-existing
scheduled-duration coordinate twenty times. On the same 2,048 patients, the
tabular metric's exploratory paired bound at an **unaudited** numeric `L=300`
moved from 22.2148 to 20.2881 minutes. The true Bayes risk cannot change
when a deterministic feature is copied. The numeric bound moved because the
metric and matching changed; the same `L=300` is not even the same smoothness
assumption under the two metrics. This is why Pythia cannot be claimed to be
agnostic to arbitrary feature engineering.

## Literature comparison boundary

The matched Hoeffding versus empirical-Bernstein comparison tests two
established finite-sample concentration methods under one estimand. The
empirical-Bernstein implementation follows [Maurer–Pontil, Theorem 11](https://arxiv.org/abs/0907.3740).
Graph-based variance estimation such as [Madrid Padilla (2024)](https://jmlr.org/papers/v25/23-1061.html)
targets variance point estimation under graph-signal assumptions, not a
one-sided MAE floor at a declared fixed clinical representation.
[Mutual nearest-neighbor regression](https://jmlr.org/papers/v14/guyader13a.html)
and [k-NN fused-lasso quantile regression](https://jmlr.org/papers/v22/20-1462.html)
train predictive/conditional-location models; they are relevant *external*
Horos comparators but were not implemented here. It would be inaccurate to
claim Pythia beat those papers or every current method. A fair next study
must match the risk, assumptions, information cutoff, sampling unit, and
validation split before ranking them.

## Engineering checks

The public paired-replicate API now requires an explicit pre-outcome,
outcome-blind representation attestation and source hash **separate from the
pair artifact hash**, and refuses
explicitly leaked representations. An attestation is not an independent
provenance verification. The global nearest-pair routine handles repeated
feature vectors without self-pairs,
and corrected estimates are clipped to nonnegative feasible risk while the
raw center remains in diagnostics. The full unit suite passed: 115 tests.
