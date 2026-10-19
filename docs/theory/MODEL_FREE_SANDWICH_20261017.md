# Training-free, two-sided certificates: floor sandwich and certified information (PYTHIA-S²)

Code: `pythia/core/multiscale_pair_extrapolation.py`:
- `sandwich`
- `all_pairs_disagreement`

Suite: `research/peras_v5/sandwich_20261017/`.

## Why this matters for Pythia

Until now, the upper side of the Pythia sandwich was a *fitted model's* held-out loss. That makes it model-dependent, it needs training, and it is loose whenever the model is miscalibrated.

The equal-floor identity gives an upper side that needs **no model, no training and no assumption beyond independent rows**. It also gives a model-free, assumption-free **lower bound on the information** that X carries about Y.

## Results (rows independent given X; matchings and geometry outcome-blind)

**U1. Upper bound on the floor (assumption-free).**

E Q(M) = Ḡ + D(M) ≥ Ḡ holds for *every* perfect matching. So

  Ḡ ≤ min_ℓ Q(M_ℓ)_U   (Chernoff–KL, Bonferroni over levels),

and R₀₁ = mean(1 − max p_i) ≤ Ḡ, because 1 − max p ≤ 1 − |p|².

This is a finite-sample, perfect-matching version of the Cover–Hart upper side. There is no asymptotics, and *any* outcome-blind geometry may be used.

**U2. Geometry portfolio.** The floor of the full information X is at most the floor of any function of X. So Ḡ(X) is at most the minimum over a declared set of geometries (text TF-IDF, character n-grams, concept vectors, …), with Bonferroni over the set.

**U3. Model-guided pairing (calibration-free).** The geometry may be the predicted probability vectors of a model trained on *other* rows. It is still outcome-blind for the rows being certified, so U1 holds unchanged.
- The model is used only to *find similar rows*, never as a predictor.
- Its miscalibration costs nothing: the certificate is Ḡ + D, where D is the label-law drift between rows the model groups together.
- By contrast, the model's own Brier score is Ḡ + E|p − q|², which also pays for miscalibration.

**I1. Information lower bound (assumption-free).**

Let I = (1/n) Σ |p_i − p̄|². This is the Brier information: the Brier-score improvement of the Bayes rule over the best constant, in absolute units. Let Q_all = 1 − Σ_k n_k(n_k − 1)/(n(n − 1)), the U-statistic over all pairs. Then

  E Q_all = Ḡ + (n/(n − 1)) · I.

Hoeffding's U-statistic argument applies to independent, non-identical rows: Q_all is an average over permutations of means of ⌊n/2⌋ independent pair terms, so the Chernoff–KL bound with N = ⌊n/2⌋ is valid. Hence

  I ≥ ((n − 1)/n) · (Q_all,L − Ḡ_U)   with **no shape assumption**.

The Bayes rule's Brier skill score is BSS\* = I/S with S = 1 − |p̄|². It is bounded below by 1 − Ḡ_U / (Q_all,L − 1/(n − 1)).

**L1. Lower bound on the floor / upper bound on information.** PYTHIA-X under A(γ), as in `MODULUS_FREE_EXTRAPOLATION_20261016.md`.

**Tiers**

| Side | Requires |
|---|---|
| U1, U2, U3, I1 | Independence of rows only. *Positive* dependence between rows that get paired (copied labels) would bias Q downward, so it must be audited. |
| L1, and the upper side of I | A(γ) |

## Relation to prior work

- **Cover–Hart.** R\* ≤ R_NN ≤ 2R\*(1 − R\*) asymptotically. The NN *error* also overshoots the floor when estimated: that is the upper side. U1 is its finite-sample, certified perfect-matching form, and adds the K-class Brier version.
- **Nearest-neighbour independence and dependence tests** detect whether information exists. I1 *quantifies* it with a finite-sample lower confidence bound.
- **Model-based upper bounds** (held-out loss of a classifier) are standard. U3 shows that a model can certify more as a *pairing device* than as a predictor whenever it is miscalibrated or overfit. We found no prior statement of this in the searches we ran.
