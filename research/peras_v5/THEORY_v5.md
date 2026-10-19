# Pythia v5: a training-free, loss-agnostic optimisation backbone for attainable risk

Status tags: [classical], [reframed], [new], [empirical].

## 0. Target and constraints
For inputs X = (tabular, free-text note) and any loss ℓ, the target is the attainable (Bayes) risk

    R*_ℓ = E min_a E[ℓ(Y−a) | X]

for MSE, RMSE, MAE, pinball Q90, and the window losses W30 and W60, all from one pass over the data.

Constraints:
- **No trained predictor.** No global model of Y is fitted, tuned or stored. Only closed-form local statistics are computed for each case.
- **Text via an LLM API.** The note is read by an LLM, used as an outcome-blind reader.

## 1. Reading text by LLM = exact discrete coordinates [empirical + reframed]
The LLM maps each note to concept codes (complexity, redo, access risk). On the hard notes (paraphrase, negation, typos, misleading distractors):
- Whole-note reading by the LLM agreed with its cached sentence-level reading on 300/300 notes.
- Both were 100% correct on every field.
- Keyword rules were 58%, 52% and 93% correct.

Because the reader is outcome-blind, R*(tab, LLM codes) ≥ R*(tab, note), with equality when the codes are sufficient (data processing; THEORY_v4 Thm 1). This holds here, and in general it costs only the information the reader drops.

## 2. Near-replicate tuples with leave-tuple-out local-linear detrending [new as a Bayes-risk device]
For case i, let T_i = {i} ∪ (k−1 nearest neighbours) in an outcome-blind metric:
- numeric columns are standardised;
- binary columns and LLM codes get a large weight (near-exact matching).

A ridge-stabilised local linear fit on the neighbourhood N_i \ T_i, which excludes the tuple, gives a slope β̂_i. Each member j is detrended to e_j = Y_j − β̂_iᵀ(z_j − z_i).

**Proposition 1 (structure of a detrended tuple).** Suppose Y|x = μ(x) + σ(x)ε with μ ∈ C², and σ, F_ε locally constant at scale ρ (the tuple radius). Then

    e_j = c_i + σ ε_j + ξ_iᵀ dz_j + ½ dz_jᵀ H dz_j + o(ρ²),

where:
- ξ_i = β − β̂_i is *independent of the tuple's noise* (leave-tuple-out), with covariance C_β ≈ σ̂²(AᵀA + R)⁻¹;
- c_i is a common constant, irrelevant to every location loss.

Two terms contaminate the tuple: a random common linear trend (slope noise) and a curvature term of order ρ². Without detrending, the location heterogeneity is O(‖∇μ‖ρ). On our designs that inflated MSE by +90% to +450%; after detrending the inflation was +3% to +20%.

## 3. SIMEX removes slope noise for every loss at once [reframed: Cook & Stefanski 1994]
Let g(λ) be the expectation of any tuple statistic (in-tuple ERM risk, leave-one-out risk, or a witness) after adding independent synthetic trends ξ̃ᵀdz with ξ̃ ~ N(0, λĈ_β).

**Proposition 2.** If ξ is approximately Gaussian, the contamination covariance is (1+λ)C_β. For Gaussian contamination, g is smooth in λ (heat-semigroup smoothing of a bounded-variation functional), so

    g(−1) = (uncontaminated value)

is recovered to first order by linear extrapolation over λ ∈ {0, ½, 1, 2}.

The correction is **loss-agnostic**: it operates on the tuple, not on the loss. Effect on dev (median |rel. err.|): MSE 0.133 → 0.062 and MAE 0.073 → 0.026. Extrapolating instead in the tuple radius over-corrects, which matches the O(ρ⁴) curvature order in Proposition 1. So we do not extrapolate in ρ.

## 4. Optimism–pessimism combination [classical core, reframed]
For i.i.d. tuples of size k, the in-tuple ERM risk is the Bernstein witness. It satisfies E r_k ≤ R* (optimism), while the leave-one-out risk satisfies E r̃_k ≥ R* (pessimism) (paper Thm 3/Prop 6).

For regular losses the two first-order terms are equal, so w r + (1−w) r̃ with w = ½ is unbiased to O(1/k). Residual heterogeneity inflates both r and r̃ equally, so a weight slightly above ½ cancels it. The window losses have a cube-root pessimism:optimism ratio [conjecture], so their weights differ.

The optimisation backbone selects, per loss, the weight w and the set of tuple sizes **once on development seeds** (validity-unconstrained, minimise median error), then freezes them:
- MSE w = 0.58, k ∈ {4, 6, 8}
- MAE 0.60, k ∈ {4, 6, 8, 12}
- Q90 0.50, k ∈ {4, 6, 8}
- W30 0.50, k ∈ {4, 6, 8}
- W60 0.55, k ∈ {4, 6, 8, 12}

## 5. The lower bound: a validity-constrained optimisation [new procedure, empirical validity]
The candidates are:
- in-tuple ERM witnesses (raw, or SIMEX-extrapolated) for k ∈ {3, …, 16};
- designed polynomial lower witnesses (paper §4; certified ψ_w ≤ λ) for MAE and Q90 at k ∈ {8, 12}.

Each has a one-sided CLT margin z·sd·√k/√n; the √k inflates for tuple overlap.

For each loss we solve

    maximise  median tightness (LCB/R*)  over candidates
    subject to  dev validity = 100%  and  max dev ratio LCB/R* ≤ 0.95  (a 5% safety margin).

**Frozen picks:**
- MSE: SIMEX-ERM at k = 12 (dev tightness 0.825)
- MAE: raw ERM at k = 4 (0.845)
- Q90: designed witness, SIMEX, k = 12 (0.849)
- W30: raw ERM at k = 12 (0.618)
- W60: raw ERM at k = 16 (0.210)

*Honest status.* A distribution-free lower bound on R*(X) is impossible for continuous X (Barber et al. 2021; THEORY_v4 Thm 1′). This bound is valid when residual heterogeneity is smaller than the finite-k optimism, which holds for smooth designs. Its validity is checked empirically on untouched seeds, not proved. On exact replicates (design 10 and the LLM strata), the ERM witness is valid in expectation without any assumption.

## 6. Model-free relevance screening [reframed: Gamma-test feature selection]
When there are more than 10 numeric columns, columns are added greedily while the detrended k = 4 tuple dispersion falls by at least 1%. This is a statistic, not a predictor. Unlike marginal distance correlation, it recovers interaction variables: on design 5 it kept {b, X0, X1, X2}, whereas dCor kept {b, X0, X1}.

## 7. What is and is not new
- **Classical:** SAA/LOO bracketing, Bernstein optimism, SIMEX, local-linear smoothing, Gamma-test-style selection.
- **New as a method:**
  1. Leave-tuple-out detrended near-replicates as a *loss-agnostic* device for Bayes risk. The only prior difference-based estimators are variance-only (Rice; Gasser–Sroka–Jennen-Steinmetz) or pair-based, and pairs cannot see non-quadratic risk (paper Prop. 1).
  2. SIMEX on the induced trend, which corrects every loss at once.
  3. Validity-constrained selection of tuple witnesses as the bound.
  4. LLM-read concepts as exact coordinates in a training-free pipeline.

---
## Addendum (v5.4 – v5.7): weak spots, leak behaviour, and design-level inference

### A. Design-balanced selection
The estimator's per-loss configuration is now chosen to minimise the **mean over designs of the per-design median error** on dev seeds, not the pooled median. That objective matches design-level inference, where designs rather than draws are the unit. It removed the global tuning that had masked curvature.

### B. Curvature-adaptive MAE (design 9)
- In v5 the MAE weight was w = 0.6. The extra weight on the optimistic statistic cancelled the average heterogeneity inflation on most designs, but under-corrected curved ones. Design 9, with a sin(X0·X1) interaction, had error 0.07–0.09.
- v5.4 instead uses w = 0.4 together with an **explicit curvature term**. After linear detrending the residual quadratic term inflates tuple statistics in proportion to ρ⁴(k) (Prop. 1), so the combination is extrapolated in ρ⁴ across k ∈ {6, 8, 12}.
- Dev result: worst-design MAE error fell from 0.073 to 0.042–0.05, and design 9 from 0.07 to about 0.03. The overall median stayed at 0.018.

### C. Scale-free ridge and SIMEX guard (v5.6/v5.7)
- **The problem.** An absolute ridge shrinks the local slope along any coordinate whose local spread is tiny. That is exactly what happens when a column, or an LLM-read value, is (nearly) a function of y. The result was a false floor of about 0.18 × MAE* on exact leaks.
- **The fix.** v5.6 penalises each slope by 1% of its local design variance, floored at 1% of a unit-variance coordinate. The fit is then invariant to the units of any coordinate, so an affine function of y is fitted almost exactly and the attainable risk goes to about 0.
- **The guard.** Low-variance directions have large slope variance. SIMEX is a small-contamination expansion, so v5.7 drops any tuple size whose induced contamination is not small (g(λ=1)/g(0) − 1 ≥ 0.3).

### D. Leak behaviour = one-sided identifiability in action (THEORY_v4 Thm 1′)
- **If the reader extracts the leaked quantity** (LLM-parsed durations, 150/150 correct; tabular columns), Pythia estimates the post-leak attainable risk. It gives about 0 for exact leaks, about (1−p)·R* when a fraction p of notes leak, and the posterior risk for a noisy proxy.
- **If the leak is hidden from the reader** (number of letter m's in a junk code = round(y/5)), *every* method estimates R* of what it can see. Every lower bound, ours included, then becomes invalid. This is the impossibility theorem, not a bug.
- **The remedy is outcome-blind reading.** From 40 unlabeled notes, an LLM auditor proposed six features; one was the m-count. With those features Pythia is back on gold, and its bound is valid.

### E. Design-level significance
Draws within a design share the data-generating process. The honest unit of inference is the design, using a Wilcoxon signed-rank test on per-design log error ratios and a design-cluster bootstrap. RESULTS_v5b.md reports both.
