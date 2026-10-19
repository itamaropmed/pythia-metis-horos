# Pythia v5.7: design-level significance, the weak spots, and leakage stress tests

All numbers below come from **fresh seeds 9200–9219** (200 draws, 10 designs). No configuration was chosen on these seeds.
- v5.4 and v5.7 were frozen (`FROZEN_v57.sha256`) before their confirmation outputs were inspected.
- The leak tests use seeds 12000–12003.

## 1. Are the wins significant at the *design* level, not only per draw?
The design is the honest unit, so each test treats one design as one observation:
- a Wilcoxon signed-rank test on the 10 per-design log error ratios;
- a sign test on designs won;
- a cluster bootstrap that resamples whole designs.

Pythia-v5.7 against the strongest accepted method:

| Loss | Rival (median error) | Pythia-v5.7 | Draws won | Designs won | Wilcoxon p (designs) | Design-bootstrap 95% CI of error ratio |
|---|---|---|---|---|---|---|
| MSE | Gamma test* (0.152) | **0.041** | 150/200 | 7/10 | **0.027** | [1.44, 8.78] |
| RMSE | Gamma test* (0.075) | **0.021** | 152/200 | 8/10 | **0.020** | [1.49, 8.65] |
| MAE | Cortes-LC (0.078) | **0.025** | 174/200 | **10/10** | **0.002** | [2.18, 4.42] |
| Q90 | Cortes-LC (0.119) | **0.037** | 169/200 | 9/10 | **0.006** | [1.83, 4.26] |
| W30 | Cortes-LC (0.097) | **0.073** | 126/200 (p = 3e-4) | 6/10 | 0.32 (n.s.) | [0.95, 2.12] |
| W60 | Cortes-LC (0.450) | **0.283** | 114/200 (p = 0.06) | 6/10 | 0.38 (n.s.) | [0.61, 4.77] |

**Answer.** For MSE, RMSE, MAE and Q90 the advantage is significant at the design level too, and the bootstrap intervals exclude 1. For the window losses it is significant only per draw (W30) or not at all (W60).

**Same LLM features given to every rival.** MSE, MAE and Q90 stay design-level significant:
- MSE: 9/10 designs, p = 0.020
- MAE: 10/10 designs, p = 0.002
- Q90: 9/10 designs, p = 0.004

**Against our own earlier trained RTX-LC.** Pythia-v5.7 wins per draw on MSE, MAE and Q90 (129, 124 and 136 of 200; p ≤ 9e-4) but not per design (Wilcoxon p ≥ 0.13). On W60 it loses per draw (93/200 wins) and the design-level difference is not significant. Note that RTX-LC trains GBMs, whereas Pythia trains nothing.

## 2. The weak spots: what changed

| Weak spot | Before (v5, seeds 9100–9119) | After (v5.7, fresh seeds 9200–9219) | What fixed it |
|---|---|---|---|
| Design 9 MAE (interaction + t3) | 0.092 vs rival 0.036 | **0.026 vs rival 0.026 (tie)** | Explicit curvature term (ρ⁴ extrapolation over tuple size) + design-balanced selection |
| Worst-design MAE | 0.092 | **0.043** | Same |
| W30 | 0.069; 5/10 designs | 0.073; 6/10 designs; draw-level p = 3e-4 | Partly; still not design-significant |
| W60 | 0.282; 5/10 designs | 0.283; 6/10 designs; n.s. | Not solved: true values are < 1%, so relative errors are noise-dominated |
| Exact leaks (y in a column/note) | estimate floor 0.18 × R* | **0.06 × R*** | Scale-free ridge + SIMEX guard |

**Trade-off, stated plainly.** Choosing the configuration for the worst designs rather than the pooled median made the *pooled* MAE median slightly worse (0.017 for v5 vs 0.025 for v5.7 on the same fresh seeds). In exchange, no design is now badly estimated, and MAE wins 10/10 designs. Every other loss is unchanged within noise.

**Without the LLM reader.**
- Errors rise to 0.14–0.25. Replacing the reader with unsupervised TF-IDF coordinates did not help (MAE 0.215 vs 0.183 with no text at all).
- The reason: bag-of-words cannot resolve negations and distractors in these notes. Without a reader that extracts the information, Pythia estimates the attainable risk of the *table alone*. That is a different, larger quantity, not an estimation failure (THEORY_v4 Thm 1′).
- So the reader is an essential part of the method, not an add-on.

## 3. Leakage stress tests (designs 1, 6, 8 × 4 seeds × 9 variants, exact gold)
Each cell is estimate ÷ the original attainable risk, median over draws. "Gold" is the true attainable risk after the leak. "+R" means the accepted method was also given the reader's output.

| Variant | Gold (MAE) | Pythia-v5.7 | GBM plug-in | GBM+R | Cortes-LC | Cortes-LC+R | RTX-LC+R |
|---|---|---|---|---|---|---|---|
| none | 1.000 | 0.989 | 1.076 | – | 0.940 | – | – |
| duration written in **every** note | 0.016 | **0.063** | 1.066 | 0.055 | 0.918 | 0.002 | 0.032 |
| in 50% of notes | 0.509 | **0.543** | 1.072 | 0.681 | 0.932 | 0.560 | 0.529 |
| in 10% of notes | 0.904 | **0.919** | 1.069 | 1.061 | 0.940 | 0.919 | 0.988 |
| column = round(y) | 0.016 | 0.063 | 0.055 | – | 0.002 | – | – |
| column = 3y + 7 | 0.000 | 0.061 | 0.054 | – | 0.007 | – | – |
| column = y + N(0,10²) | 0.447 | **0.448** | 0.496 | – | 0.408 | – | – |
| column = round(y/5) | 0.078 | 0.101 | 0.098 | – | 0.027 | – | – |
| **hidden**: number of letter "m" in a junk code = round(y/5) | 0.078 | **0.081** (with LLM audit) / 0.989 (without) | 1.074 | 0.098 | 0.929 | 0.027 | 0.072 |

**Mean |estimate − gold| ÷ original risk over all leak variants** (lower is better):

| Method | MSE | MAE | Q90 | W30 | W60 |
|---|---|---|---|---|---|
| **Pythia-v5.7** | 0.020 | **0.033** | **0.048** | **0.015** | **0.068** |
| Cortes-LC+R | 0.064 | 0.038 | 0.080 | 0.041 | 0.447 |
| RTX-LC+R | 0.057 | 0.038 | 0.062 | 0.044 | 0.290 |
| GBM+R | 0.136 | 0.111 | 0.260 | 0.128 | 0.776 |
| Accepted methods with TF-IDF only | 0.26–1.38 | 0.29–1.04 | 0.32–1.16 | 0.28–2.04 | 0.74–7.5 |

What the tests show:
1. **Leaks written in text are invisible to the accepted methods without a reader.** Even "actual in-room time 173 min" in *every* note leaves the GBM plug-in at 1.07 and Cortes-LC at 0.92 of the original risk, because TF-IDF does not read numbers. The LLM reader parsed 150/150 stated durations correctly, and Pythia then drops to about 0.06.
2. **Partial leaks are tracked proportionally.** At 50% and 10% leakage Pythia gives 0.54 and 0.92 (truth 0.51 and 0.90). At 50%, the GBM plug-in even with the reader overstates (0.68).
3. **A noisy proxy column is handled almost exactly**: 0.448 against a true 0.447.
4. **Hidden channel ("number of m's").** Without auditing, *every* method, Pythia included, reports the pre-leak risk (about 1.0). Every lower bound becomes invalid: Pythia 7%, RTX 10% valid.
   - This is the one-sided identifiability theorem in action: no method can certify a floor against information it cannot read.
   - **An outcome-blind LLM audit** of 40 unlabeled notes proposed six features, one of which was the m-count. With those features Pythia returns 0.081 (truth 0.078) and its bound is 100% valid.
5. **Remaining imperfection.** Under exact leaks Pythia's estimate and bound floor at about 0.06 of the original risk, against a truth of about 0.016 (a 1-minute rounding cell). The bound is therefore formally invalid there, though it has fallen from about 0.85 to about 0.06 of the original risk. Cortes-LC goes lower in the exact-leak cells (0.002–0.007). This floor is the slope-estimation noise of the local fits.

## 4. Lower bound on fresh seeds (v5.7)
Each cell is coverage / median tightness (bound ÷ truth):

| Bound | MSE | MAE | Q90 | W30 | W60 |
|---|---|---|---|---|---|
| Pythia-v5.7 | 0.99 / **0.82** | 1.00 / **0.84** | 0.995 / **0.85** | 0.995 / **0.63** | 0.995 / **0.20** |
| RTX-LCB (ours, earlier) | 1.00 / 0.70 | 1.00 / 0.76 | 1.00 / 0.73 | 1.00 / 0.55 | 1.00 / 0.07 |

The paired witness and Fano bounds are invalid on most draws, as in the v5 results. Pythia coverage is within the nominal one-sided 95% level, and the bound is markedly tighter.

## 5. Still open
- **Window losses.** No design-level significance yet.
  - W60 truths are below 1%, and its errors stay large in relative terms.
  - A principled next step is a smoothed-window (ramp) target with an exact sandwich (paper §6), which turns the non-regular loss into a regular one.
- **Exact-leak floor.** About 0.06 × R*. Local fits with a curvature-aware SIMEX on the bound, not only the estimate, should remove it.
- **Inference on 10 designs.** Ten designs give limited design-level power. More designs would tighten the W30 conclusion either way.
