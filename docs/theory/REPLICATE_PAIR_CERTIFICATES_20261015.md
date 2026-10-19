# Replicate certificates for 0-1 Bayes floors: nuisance quotients, unbiased minorants, second-order slack

Code: `pythia/core/replicate_pair_certificates.py`.
Tests:
- `research/peras_v5/realdata_20261015/text_quotient_plasmode.py`
- `research/peras_v5/realdata_20261015/tabular_replicate_ladder.py`
- `research/peras_v5/realdata_20261015/mtsamples_real_specialty.py`

## Where this sits in the literature (checked 2026-10-15)

| Line of work | What it gives | What it does not give |
|---|---|---|
| FeeBee (Renggli et al., NeurIPS D&B 2021) | Benchmark of kNN, KDE and divergence-based BER estimators on embeddings | No finite-sample guarantee, and the answer depends on the embedding |
| Cover–Hart / Fukunaga NN inversion | R* ≥ f_K(R_NN) | Asymptotic, needs a metric, and the NN error is estimated with bias |
| Hard-label averaging (Ishida et al. 2023; "Practical estimation … soft labels and calibration", arXiv 2505.20761) | Instance-free BER from repeated labels per instance, with a Hoeffding term | Plug-in min(p̂, 1−p̂) has O(m^-1/2) bias for small m; binary only (multi-class left open) |
| Pure-error / lack-of-fit with replicates (classical regression) | Unbiased σ² for squared loss | Not for 0-1 or quantile losses, and replicates must exist |

**Gaps closed here**
1. Finite-sample, multi-class lower confidence bounds that are unbiased for a minorant of the floor, with no plug-in optimism, even when cells have m = 2. These are Theorems A and C.
2. A way to manufacture replicates in raw text without embeddings: a declared nuisance quotient, plus a falsification audit for it.
3. A second-order penalty for near-replicates: δ² instead of 2δ.

These novelty claims are relative to the works above. A systematic review is still needed before publication.

## Definitions

**Information modulo a nuisance group.** Let T be raw text, and let c be a declared canonicalizer, i.e. a quotient by a set G of edits:
- header, name, MRN and date of service;
- dates;
- abbreviations;
- section order;
- whitespace and case;
- addenda, when an addendum slack is declared.

**Invariance assumption:** Y ⟂ T | c(T). Under it, R\*(T) = R\*(c(T)), and rows with equal c(T) are exact replicates. The assumption is about the data-generating process, not about a model. It can be falsified (see the audit below) but never proved.

**Sufficiency slack.** If instead TV(P(Y | t), P(Y | t′)) ≤ δ whenever c(t) = c(t′), the certificates below hold with explicit penalties.

## Theorems

Throughout:
- n_c is the vector of class counts in cell c;
- m_c is the cell size;
- g_c = 1 − Σ_k p_ck²;
- f_K(g) = (K−1)/K · (1 − sqrt(1 − K g/(K−1))), which is convex and increasing, and satisfies R_p ≥ f_K(g_p).

**Theorem A (pair disagreement).** The fraction of disagreeing pairs ĝ_c is unbiased for g_c for every m_c ≥ 2. By Jensen, R ≥ (n₂/n) · f_K(G_L), where G_L is a Poisson–Bennett lower confidence bound on Σ (m_c/n₂) ĝ_c. Validity of this concentration step rests on Bernoulli extremality of [0, w_c] variables.

**Theorem B (stratified Jensen).** Strata of cells are chosen outcome-blind on a development split, and each stratum is bounded at α/S. *Empirically this did not help: weak strata cost more in α than they recover from Jensen.*

**Theorem C (unbiased minorant, linear).** Let q be any polynomial with q(g) ≤ f_K(g) on [0, (K−1)/K]. Then q(g_c) has an unbiased U-statistic estimator from the counts once m_c ≥ 2·deg q:
- s = Σp² and s² are estimated by falling factorials;
- the estimate is linear across cells, so there is no Jensen loss.

Fixed-action domination says E[1 − max_k n_ck/m_c] ≤ R_c (the accepted min-count / hard-label route). The certificate is therefore Σ_c (m_c/n) e_c, where each cell's estimator e_c (min-count, q₁ or q₂) is chosen per size class on development data.
- q is fitted by an LP on development data, then shifted down by its exact maximum violation.
- Concentration is the better of weighted Hoeffding (exact per-cell ranges) and shifted Poisson–Bennett.

**Theorem D (second-order slack for pairs).** For two laws,

  1 − Σ p_i p_j = (g_i + g_j)/2 + ‖p_i − p_j‖₂²/2 ≤ (g_i + g_j)/2 + TV².

So with pairwise TV ≤ δ inside cells, Theorem A holds with G_L − δ², and the q₁ route with penalty c₁δ². The min-count route pays 2δ, which is first order. This is the text analogue of smooth domination's L²d² penalty.

**Assumption the U-statistic routes add: conditional independence within cells.** Theorems A and C need labels in a cell to be iid given the cell. The min-count route (fixed-action domination) needs only each row's marginal law.
- *Negative* dependence, e.g. listings sampled without replacement, biases ĝ upward and makes the bound invalid.
- *Positive* dependence, e.g. a copied-forward label, biases ĝ downward, so the bound stays conservative.
- `within_cell_independence_audit` tests a necessary condition: under H0 every cell has pair agreement ≥ 1/K. It uses a Chernoff–KL test over cells.

On real MTSamples specialty labels this assumption fails: only 2 of 3,105 duplicate pairs share a label. The audit rejects, and Pythia refuses the U-statistic routes there.

**Chunking.** For large cells, random outcome-blind chunks of size M are again independent exact replicate cells, which gives more concentration units. M is chosen on the development split.

**Audit (falsification of the quotient).** Within canonical cells, correlate pair label-disagreement with the distance between what the canonicalizer removed. Permuting the residuals within cells gives an exact p-value under invariance.

## Results

See `research/peras_v5/realdata_20261015/REPLICATE_RESULTS_20261015.md`.
