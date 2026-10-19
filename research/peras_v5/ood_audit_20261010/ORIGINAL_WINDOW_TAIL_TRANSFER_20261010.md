# From clipped intervals to original W15/W30/W60 percentages

## Result

The earlier eight-loss intervals target clipped outcomes. `tail_transfer.py` now converts the W15/W30/W60 intervals into valid conditional intervals for **original-outcome** miss probability and within-window percentage, without assuming bounded duration magnitude. On ten mechanisms × five untouched seeds, all 150 original-window intervals contained known gold. This does not transfer them from the declared group representation to richer raw features.

## Tail-transfer lemma

Let `C(y)` clip y to `[lo,hi]`, and let `R_C*` and `R*` be clipped- and original-outcome Bayes risks for the same representation. For the supported location losses, clipping contracts loss, so `R_C* <= R*`; any lower bound on clipped risk is also a lower bound on original risk.

For an action a optimal for the clipped law, choose a inside `[lo,hi]`. Then:

- MAE: `ell(Y,a)-ell(CY,a) <= |Y-CY|`.
- pinball q: the difference is at most `max(q,1-q)|Y-CY|`.
- MSE, writing `D=hi-lo` and `T=|Y-CY|`: the difference is at most `T²+2DT`.
- window miss `1{|Y-a|>h}`: the difference is at most `1{Y outside [lo,hi]}`.

Therefore `R* <= R_C* + E[penalty(Y)]`. The result is elementary loss stability, not a new robust-statistics theorem.

For window loss, the tail penalty is binary regardless of how far the outcome lies outside the clip. `tail_upper` estimates its audit-weighted group mean from calibration outcomes and applies weighted Hoeffding. For MAE, pinball, and MSE, tail magnitude is unbounded. Those upper transfers **refuse** unless the user supplies either:

- a valid almost-sure penalty cap, followed by weighted Hoeffding; or
- externally justified groupwise population variance caps, followed by a one-sided Cantelli bound.

Observed sample maxima and variances are not valid external caps. The variance route is intentionally conservative. More advanced Catoni-style estimators can improve heavy-tail mean inference under moment assumptions—see [Catoni](https://arxiv.org/abs/0909.5366), [Brownlees–Joly–Lugosi](https://arxiv.org/abs/1406.2462), and [Catoni-style confidence sequences](https://arxiv.org/abs/2202.01250)—but they cannot remove the need for a moment/tail assumption. Pythia should add such a route only with a separately audited theorem and fixed tuning.

## Checks and fresh audit

`check_tail_transfer.py` ran 4,000 deterministic empirical-distribution inequalities across Student-t3 outcomes and eight losses: zero contraction/upper-transfer violations. In 500 independent two-group calibrations, the window tail-probability cap had zero misses against a large Monte Carlo target for W15/W30/W60. These checks are diagnostics; Hoeffding and the deterministic lemma provide the claim.

`run_original_window_ood.py` freezes a single 19-event family: 16 endpoints for eight clipped losses plus three window tail caps, all at `alpha=0.05/19`. The group key, 20,000 calibration cases, 800 audit cases, clip `[0,300]`, k=16, caps, ten mechanisms, and seeds 96001–96005 were fixed before results. Joint coverage is at least 95% by a union bound, conditional on group-law stationarity, outcome independence, sufficient synthetic grouping, and frozen choices.

| Original target | Positive lower / 50 | Median lower/gold | Median upper/gold | Median interval width | Known-gold misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| W15 miss | 50 | 0.645 | 1.258 | 0.275 | 0 |
| W30 miss | 42 | 0.399 | 1.595 | 0.217 | 0 |
| W60 miss | 0 | 0.000 | 3.326 | 0.128 | 0 |

The corresponding median percentage widths are 27.5, 21.7, and 12.8 percentage points. Median empirical tail probability was 0.0026 and its 19-event upper cap 0.0157. The correction increased all 150 upper endpoints. The uncorrected clipped upper happened to remain above original gold in this conservative experiment, but it had **no theorem-level original-outcome interpretation**; observed nonfailure is not permission to omit the tail term. Raw rows: `original_window_ood_results.jsonl`.

## Limits

W60 remains effectively uninformative because its miss floor is close to zero; an upper interval can still be operationally useful, but its lower risk certificate is not. These results use a synthetic sufficient group key and independent stationary calibration. They do not establish Mayo group sufficiency, temporal stability, patient independence, or a raw-note certificate. For Mayo, the tail probability audit is feasible, but the representation-refinement gap remains the dominant unresolved assumption.
