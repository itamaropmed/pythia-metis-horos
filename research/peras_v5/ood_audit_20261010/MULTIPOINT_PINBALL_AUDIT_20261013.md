# Multi-point Q90 witnesses: theorem, negative result, and e-value mixture

## Question

Can overlapping multi-point blocks repair the weak extreme-pinball certificate
without fitting a predictor or pretending overlapping scores are independent?

## Route 1: capped fractional partitions with entity McDiarmid

For every frozen balanced partition, average the empirical-optimum pinball
score over its blocks.  Average those statistics over any number of
partitions.  Although blocks overlap between partitions, replacing one entity
changes the grand mean by at most

```
min(k*c, max(q,1-q)*D) / n,
```

where `k` is block size, `c` the score cap, and `D` the outcome width.
Entity-level McDiarmid therefore gives a valid endpoint.  SAA optimism and the
barycentric theorem subtract one-half of the registered average discrepancy.

On development seeds `136001--136100`, this route:

- had zero known-risk exceedances;
- beat paired Bentkus in 763/1,000 runs;
- had median bound/gold 0.303 versus 0.259 for paired Bentkus;
- lost to disjoint dKelly in 1,000/1,000 runs; dKelly median was 0.763.

This route is retained as a valid negative control, not adopted as the Q90
default.

Development raw SHA-256:
`1f478567feeeaa4f352a6dfcf9b6e15bf5a5ddb37fe5c839181847f7fd03ea3c`.

## Route 2: arithmetic mixture of partition e-values

Each balanced partition contains disjoint entities.  Its diversified-Kelly
statistic is therefore an e-value for the corresponding bounded average-score
null.  Partition e-values are dependent because they reuse entities, so their
product is invalid.  Their preweighted arithmetic average remains an e-value
under arbitrary dependence.  This is a classical e-merging fact; the Pythia
step is to compose it with partition-specific barycentric discrepancy caps
and invert it as an attainable-risk endpoint.

For target risk `T`, partition `r`, and declared penalty `P_r/2`, its e-value
tests the mean-score null `T + P_r/2`.  The implementation accepts either a
uniform discrepancy cap or one cap per partition.  It never multiplies across
overlapping partitions.

Primary references:

- [Waudby-Smith and Ramdas, bounded-mean betting](https://arxiv.org/abs/2010.09686)
- [Wang, the only admissible way of merging arbitrary e-values](https://academic.oup.com/biomet/article/112/2/asaf020/8086785)

## Untouched confirmation

Frozen before seeds `137001--137100`:

- ten bounded exact-law mechanisms;
- 800 entities and 20 balanced random partitions;
- Q90, clip `[0,1]`, `k=16`, score cap `.08`;
- exactly one candidate at alpha `.05`;
- matched single-partition dKelly and paired Bentkus comparators.

| Route | Median bound/gold | Positive | Known-risk exceedances |
| --- | ---: | ---: | ---: |
| Partition-mixture dKelly | **0.8213** | 1,000/1,000 | 0 |
| Single-partition dKelly | 0.8192 | 1,000/1,000 | 0 |
| Paired Bentkus | 0.2586 | 1,000/1,000 | 0 |

The mixture beat the single partition in 583/1,000 runs and in 8/10
mechanism medians.  It beat paired Bentkus in 1,000/1,000 and 10/10 mechanism
medians.  Eight of ten mechanism signs is not a conventional 5% sign-test win
(`p=0.0547`, one-sided), and the median gain over dKelly is small.  The honest
conclusion is **valid variance stabilization with a modest, not significant,
improvement over the strongest matched route**.

Confirmation raw SHA-256:
`745ebe13caeba82413065867c7d19627e015017d58a1a95912db681b3256b88e`.

## Novelty position

Arithmetic e-value merging and bounded-mean betting are established.  The
potential contribution is narrower: constructing overlapping, balanced,
loss-specific SAA witnesses; attaching a separate barycentric transfer cap to
each partition; and compiling them into one fail-closed attainable-risk
certificate.  The present exact-law evidence is not strong enough to claim a
new state of the art against disjoint betting.

## Next gate

The route becomes scientifically interesting if outcome-blind feature
geometry supplies several near-replicate partitions with different valid
penalty caps.  A frozen geometry-weighted arithmetic e-mixture can favor
partitions with smaller declared penalties without reading evaluation
outcomes.  It must be compared against the single best predeclared matched
partition on fresh nonzero-discrepancy mechanisms.  Until that gate, the
production Q90 route remains disjoint dKelly.

## Geometry-weighted near-replicate development

The proposed next gate was run on seeds `138001--138100`: ten scale-law
mechanisms, nonzero analytic partition penalties, 20 feature-only noisy-rank
partitions, `k=16`, cap `.16`, and five softmax weight temperatures.  All
routes had zero known-risk exceedances.

| Route | Median bound/gold | Wins over single best | Mechanism-median wins |
| --- | ---: | ---: | ---: |
| Single minimum-penalty partition | **0.7622** | — | — |
| Equal mixture | 0.7120 | 49/1,000 | 1/10 |
| Best soft weighting (`tau=.05`) | 0.7593 | 128/1,000 | 1/10 |

The outcome-blind weighting idea is valid but does not beat simply freezing
the minimum-declared-penalty partition.  No untouched confirmation was run,
because development did not satisfy the registered go criterion.  Weighted
mixtures remain useful only where randomization stability itself matters.

Development SHA-256:
`8a54e51db41aa687f97a68ea57581b64c2325d52be97f034365252c90efc92ca`.
