# Training-free two-sided certificates: results (development, 2026-10-17)

Theory: `docs/theory/MODEL_FREE_SANDWICH_20261017.md`.

**Columns**
- Upper floor U (assumption-free).
- Information lower bound I_L (assumption-free).
- PYTHIA-X lower floor L (γ = ½).

**Accepted comparator.** For the upper side, the held-out Brier score of fitted classifiers, KL-certified:
- synthetic and tabular data: the better of logistic regression and gradient boosting;
- MTSamples: the best of tuned logistic regression, logistic regression with C = 4, and 15-NN.

## 1. Synthetic, known laws

n = 4000, 30 runs. Ratios are the bound divided by the true value. The "/k" suffix counts violations.

| Design | Brier floor Ḡ | **U (no model)** | Fitted-model U | L (γ = ½) | True info I | **I_L (no assumption)** | Bayes BSS\* | BSS\* lower bound |
|---|---|---|---|---|---|---|---|---|
| linear2 | 0.380 | **1.08/0** | 1.13/0 | 0.79/0 | 0.120 | 0.057/0 | 0.24 | 0.12/0 |
| smooth5 | 0.328 | **1.10/0** | 1.17/0 | 0.61/0 | 0.069 | 0.007/0 | 0.17 | 0.02/0 |
| sharp2 | 0.133 | **1.17/0** | 1.26/0 | 0.56/0 | 0.367 | 0.312/0 | 0.73 | 0.67/0 |
| sparse10 | 0.380 | 1.14/0 | 1.14/0 | 0.45/0 | 0.120 | 0.033/0 | 0.24 | 0.07/0 |
| K4_3d | 0.696 | **1.04/0** | 1.09/0 | 0.87/0 | 0.050 | 0.001/0 | 0.07 | 0.00/0 |
| checker_coarse | 0.420 | **1.13/0** | 1.32/0 | 0.81/0 | 0.080 | 0.002/0 | 0.16 | 0.00/0 |
| checker_fine (adversarial) | 0.420 | 1.26/**0** | 1.32/0 | 0.99/15 | 0.080 | 0.000/**0** | 0.16 | 0.00/0 |
| rowwise_random (adversarial) | 0.420 | 1.25/**0** | 1.32/0 | 1.00/15 | 0.080 | 0.000/**0** | 0.16 | 0.00/0 |
| null2 (no signal) | 0.480 | 1.06/1 | 1.11/0 | 0.83/0 | 0 | 0.000/0 | 0 | 0/0 |

**Reading**
- The assumption-free sides held everywhere, including both adversarial designs where PYTHIA-X's lower side fails.
- U had 1 miss in 270 runs, within the α/4 budget.
- **The training-free U is tighter than the fitted-model U in 8 of 9 designs.** In sparse10 it is marginally looser (1.140 vs 1.137).
- I_L is conservative: it is non-vacuous only when the signal is strong (sharp2) or moderate (linear2).

## 2. Real labels: MTSamples specialty from note text

There are 2,352 unique notes, so no duplicates are involved. There is no gold value, and none is needed for the assumption-free sides.

> **Two validity bugs found and fixed. Both only affect tie-heavy geometries.**
> 1. **Row order.** The CSV is sorted by specialty, so nearest-neighbour search order and tie-breaking leaked the label. Rows are now shuffled with a label-independent seed, and `perm_seed` randomizes tie-breaking.
> 2. **Self-pairs (2026-10-17).** With tied points, the neighbour list can contain the row itself at any rank, not only at rank 0. The greedy matcher then paired a row with itself, and a self-pair always "agrees", which biased Q downward. `_greedy_perfect` now excludes self explicitly and asserts that the matching is perfect. There is a unit test with massive ties.
>
> The word and char TF-IDF geometries, which are continuous and have no ties, were unaffected. The concept-dictionary, keyword and length geometries changed a lot (concept U went from 0.269 to 0.601). The earlier claim that the concept geometry "refutes A(½)" was an artifact of bug 2 and is **withdrawn**: its Q now rises with scale, as A predicts. PMLB results did not change, because continuous Age prevents ties. The numbers below come from runs after both fixes.

| Labels | Geometry | **U, Brier floor** | **I_L** | **BSS\* lower bound** | L (γ = ½) | Fitted-model U |
|---|---|---|---|---|---|---|
| Top 10 + other (K = 11) | word TF-IDF → SVD | **0.262** | **0.475** | **0.64** | 0 (vacuous) | 0.359 (tuned LR), 0.388 (15-NN), 0.405 (LR, C = 4) |
| | char-4gram TF-IDF → SVD | 0.278 | 0.458 | 0.62 | 0 | |
| | concept dictionary (30 terms) | 0.601 | 0.136 | 0.18 | 0.108 | |
| | log length only | 0.786 | 0.000 | — | 0.585 | |
| | random geometry (null control) | 0.812 | **0.000** | — | 0.604 | |
| | keywords field | 0.412 | 0.324 | 0.44 | 0 | |
| | model-guided pairing (tuned-LR / 15-NN scores, cross-fitted) | 0.347 / 0.361 | 0.37 / 0.35 | — | — | |
| All 40 specialties | word TF-IDF → SVD | **0.291** | **0.474** | **0.62** | 0 | 0.422 (15-NN), 0.456 (LR, C = 4), 0.608 (tuned LR) |
| | concept dictionary | 0.632 | 0.133 | 0.17 | 0.135 | |
| | model-guided (15-NN / tuned LR) | 0.387 / 0.405 | 0.36 / 0.34 | — | — | |

**Reading**
- **Training-free and model-free: a certified ceiling on the specialty floor from text.** The Brier floor is at most 0.262 (K = 11) and 0.291 (K = 40), from word TF-IDF pairing. That beats every fitted model's certified upper bound (0.359 and 0.422).
- **The text carries certified Brier information.** At least 0.47, so the Bayes rule's Brier skill is at least 0.62–0.64, with no shape assumption.
- **Controls.** The null geometry certifies 0 information, and so does length alone. The 30-term concept dictionary certifies only 0.13, which quantifies how much a hand-built dictionary loses relative to the full text.
- **Model-guided pairing** beats the guiding model's own certified Brier score (0.387 vs 0.422 for 15-NN; 0.405 vs 0.608 for tuned LR at K = 40). It is still weaker than plain text-geometry pairing here.
- **Lower side.** The text lower floor is vacuous at γ = ½.
- **Near-duplicates are a possible caveat for the independence assumption.** Only 29 of 2,357 notes have a nearest neighbour with cosine similarity ≥ 0.9. Label agreement among nearest neighbours is 77% even at cosine < 0.6.

## 3. Real labels: PMLB clinical tabular (n = 4096, 10 samples)

| Task | U (no model) | Best fitted-model U | I_L | L (γ = ½) |
|---|---|---|---|---|
| lowbwt LOW (birth weight excluded) | 0.387 | 0.386 | 0.011 | 0.223 |
| pharynx Status | 0.385 | 0.383 | 0.000 | 0.196 |
| pharynx Grade | **0.608** | 0.621 | 0.000 | 0.433 |

On low-signal tabular data the training-free U matches fitted gradient boosting or logistic regression. Certified information is near 0, consistent with weak signal.

## Net

The Pythia sandwich is now fully model-free on the upper side and on the information lower bound, with no shape assumption. Only the lower floor and the upper information bound still need A(γ), and the sandwich's own consistency check (L ≤ U and Q nondecreasing in scale) can refute A.
