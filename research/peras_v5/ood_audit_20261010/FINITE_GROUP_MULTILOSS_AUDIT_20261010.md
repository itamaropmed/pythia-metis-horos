# Eight-loss grouped risk intervals: valid but deliberately scoped

Follow-up: `TARGETED_BENTKUS_PORTFOLIO_20261010.md` adds a jointly accounted randomized Bentkus route for the MSE upper endpoint. It improves median MSE interval width on fresh seeds but is not pointwise dominant and does not change the target scope below.

Tail follow-up: `ORIGINAL_WINDOW_TAIL_TRANSFER_20261010.md` adds a jointly budgeted tail-probability correction that upgrades W15/W30/W60 from clipped-outcome to original-outcome group-risk intervals. Magnitude-sensitive losses still require external tail or moment assumptions for an original-outcome upper endpoint.

## Result

`finite_group_multiloss_interval.py` now supplies training-free lower and upper bounds for the **clipped-outcome Bayes risk of a declared finite group representation** under MAE, MSE, q10/q50/q90 pinball, and W15/W30/W60 miss probability. W intervals also return within-window **percentage** endpoints `[100(1-U),100(1-L)]`. The grouped empirical optimum is a separate exploratory point estimate, not a confidence endpoint.

This closes a multi-loss implementation gap under strong, explicit repeated-scenario assumptions; it does **not** repair the v5.7 continuous-feature empirical bound, certify raw note information, or establish a novel concentration theorem.

## Conditional interval argument

For each declared group g, assume independent iid calibration outcomes from one group-reference law stable at the target date. Audit group weights `pi_g`, loss, clip, k=16, score caps, and the analysis were frozen without calibration outcomes. In each group, split calibration cases into disjoint k-tuples. Each tuple yields:

1. In-tuple sample optimal risk `S_gb`, clipped at score cap C. By `E inf <= inf E` and score clipping, its expectation is no greater than the group's clipped Bayes risk.
2. Loss on the tuple's first outcome of an action minimizing loss on the other k−1 outcomes, `H_gb`. The held-out outcome is independent of that action; its expected loss is no less than group Bayes risk. **Do not clip these upper scores below the full loss range.**

The tuples are independent conditional on the group design. Weighted Hoeffding gives lower and upper deviations with squared weight factor `sum_g pi_g²/m_g`, where `m_g=floor(n_g/k)`. Each endpoint spends `alpha=0.05/16` for eight losses × two sides; union coverage is at least 95% for all 16 endpoints **on one dataset**, conditional on the assumptions. Upper score ranges for a `[lo,hi]` clip of width D are D for MAE, D² for MSE, `max(q,1−q)D` for pinball, and 1 for window miss. Lower score caps may be smaller because reducing a nonnegative SAA witness preserves lower validity.

The interval target is **clipped** group Bayes risk. Monotone clipping contracts the supported location losses, so the lower endpoint is also below original-outcome *group* risk; the upper endpoint is **not** generally above original-outcome risk. Neither endpoint transfers automatically to richer raw tabular/text information. `refinement_gap_profile.py` continues to refuse that transfer without an independently justified information-gap cap. No predictive model is trained or stored; a tuple's empirical optimal action is only a risk statistic.

## Validation

`check_finite_group_multiloss.py` compared all eight intervals against analytic two-point Bayes risks across 200 seeds: **0/1,600 interval misses**. It also checks refusal for missing/undersized groups, and all-loss collapse to zero under a supplied known full-disclosure risk-gap cap. W1 transfer of window miss is refused.

`run_finite_group_multiloss_ood.py` froze ten existing exact-law noise mechanisms, 20,000 independent calibration cases, 800 audit cases, the sufficient synthetic `(code,note concept)` group key, outcome clip `[0,300]`, k=16, and loss-specific lower-score caps. Five untouched seeds 92001–92005 per mechanism were evaluated against **numeric clipped-law gold** from CDF integration, not the unbounded original gold. Results are 0/400 interval misses; all minimum lower/gold and upper/gold margins are positive on the evaluation grid. Zero observed misses do not replace the conditional proof.

| Loss | Positive lower / 50 | Median lower/gold | Median upper/gold | Exploratory plug-in median relative error |
| --- | ---: | ---: | ---: | ---: |
| MAE | 50 | 0.855 | 1.914 | 0.006 |
| MSE | 50 | 0.706 | 8.186 | 0.012 |
| Q10 | 50 | 0.782 | 4.643 | 0.008 |
| Q50 | 50 | 0.855 | 1.914 | 0.006 |
| Q90 | 50 | 0.767 | 4.258 | 0.013 |
| W15 miss | 50 | 0.655 | 1.220 | 0.052 |
| W30 miss | 44 | 0.396 | 1.477 | 0.080 |
| W60 miss | **1** | **0.000** | 2.804 | 0.150 |

The clipped-outcome W30 interval is logically useful on most designs; W60 is nearly vacuous because its floor is often near zero. MSE and pinball upper endpoints are loose because Hoeffding pays for the entire clipped outcome range. A bounded-score upper witness cannot be artificially winsorized. For instance, design 1 seed 92001 has known W30 miss floor 0.1336 and interval `[0.0286,0.2324]`, equivalent to within-30 percentage `[76.76%,97.14%]` for that clipped group task.

The group plug-in point estimate is accurate with 20,000 calibration cases and an oracle-sufficient group key, but that is **not** an estimator win against the earlier 800-case continuous/text Pythia setting or external fitted methods. Full rows: `finite_group_multiloss_results.jsonl`.

## Research interpretation

The lower/upper SAA principle is established stochastic-programming methodology; see [Mak–Morton–Wood](https://calhoun.nps.edu/bitstream/10945/38421/1/MakMortonWood.pdf). [Recent Bayes-sufficiency work](https://arxiv.org/abs/2606.04045) also makes loss-specific representation sufficiency explicit. Pythia's challenge is **not** to rename either principle. It is to make useful intervals when clinical tabular and text features produce nonidentical conditional laws without exact repeated scenarios; to justify the information-gap or transport caps independently; to obtain sharper, selection-safe upper bounds under realistic tails; and to preserve the loss-specific target through any augmentation. Until then, these intervals are a reference benchmark and a conditional router, not a Mayo certificate or literature SOTA claim.
