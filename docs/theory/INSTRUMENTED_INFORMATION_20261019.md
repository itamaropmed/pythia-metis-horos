# Instrumented information: certified accounting of what readers take from clinical text (2026-10-19)

- **Code:** `pythia/core/instrumented.py`.
- **Experiments:** `research/instrumented_20261019/`.
- **Status:** theory and simulation suite with a known law. MTSamples runs use a planted outcome; LLM readers run through LiteLLM on the Mac.

## 1. The idea in one paragraph

Today an LLM is used on clinical notes in one of two ways:

- as a **surrogate labeller**, corrected afterwards by PPI or DSL against gold labels;
- as a **feature extractor**, whose AUC gain is reported without any guarantee.

Both treat the LLM's output as the thing to be trusted. We propose the opposite stance: **the LLM is a measuring instrument (a *reader*), and the quantity we certify is the information the *text* carries about the outcome beyond the table.**

Reading can only lose information (data processing). So every reader, good or bad, honest or hallucinating, gives a valid *lower* bound — provided it is fixed without the evaluation labels.

This turns the multiplicity of prompts, models and decoding settings from a threat ("LLM hacking") into a portfolio. We call the resulting framework **instrumented information accounting**.

## 2. Setting

- **Data:** rows (X_i, T_i, Y_i), with table cells c(X) and K classes.
- **Truth:** p_i = P(Y | X_i, T_i). For finite-sample concentration, outcome vectors must be independent across declared sampling entities (e.g. patients); outcomes may be arbitrarily dependent within an entity. If IDs are omitted, the implementation explicitly assumes row independence.
- **Readers:** a reader ρ is any map (X, T) ↦ R fixed without the evaluation labels: a prompt + model + decoding config, an embedding or a rule.
- **Probabilities:** q_i are held-out probabilities from any model of (X, R), cross-fitted.
- **Spread:** for a label-free grouping G, V(G) = (1/m) Σ_i |p_i − p̄_{G(i)}|².
- **Text information:** NLI(T | X) = V(cells), the information the text holds beyond the table (in-sample, in Brier geometry).

## 3. Results

**P0 — Data processing.** In population form, E|P(Y|X,R) − P(Y|X)|² ≤ E|P(Y|X,T) − P(Y|X)|². This holds for any deterministic reader, and for any stochastic reader whose randomness is independent of Y given (X, T).

*Why:* P(Y|X,R) = E[P(Y|X,T) | X, R], and conditional expectation is a contraction in L².

**P1 — Projection certificate.** Let H be label-free and sum to zero inside every group of G. Then
V(G) ≥ (C_lower)₊² / mean|H|², where C = mean_i H[i, y_i].

*Why:*
1. E[C | design] = (1/m) Σ ⟨p_i, h_i⟩ = (1/m) Σ ⟨p_i − p̄_G, h_i⟩.
2. Cauchy–Schwarz bounds this by √(V(G) · mean|H|²).
3. For entity c, sum the row witnesses into X_c = Σ_{i∈c}(H[i,Y_i] − min_k H[i,k]). Conditional on the design, independent clusters give independent X_c ∈ [0, Σ_{i∈c} w_i]. C_lower is a heterogeneous-support Chernoff lower bound over these cluster sums, normalized by the number of rows.

Under the label-free-score premise, the bound depends on the reader only through the *alignment* cos²(h, p − p̄); an uninformative reader tends to give zero. The numerical API receives raw arrays and cannot authenticate that premise, so a positive returned value is conditional evidence unless score provenance is separately verified.

*Relation to prior work:*
- Floodgate (Zhang & Janson) targets the same squared-loss mMSE gap. Its original model-X route assumes a known P(T | X); the paper also gives a doubly-robust extension. Its inference is asymptotic. Do not characterize it as inapplicable to text without comparing that extension under the intended assumptions.
- VIM and LOCO are asymptotic and need consistent nuisance estimates.
- The GCM test gives no effect size.
- P1 is finite-sample, design-conditional and model-free, conditional on independent clusters and a valid outcome-blind score. It is not robust to dependence across declared clusters.

**P2 — Reader portfolio (selection adjustment, conditional on precommitment).** For a finite family of J readers fixed independently of evaluation outcomes, max_j L_j(α/J) is a valid lower bound. The current API does not enforce precommitment: a label-adaptive score matrix can yield a positive invalid value, so the implementation is not “hacking-proof” by itself.

The search cost is only the union bound. On a null text, a naive "pick any configuration that certifies" policy has false-claim rate 1 − (1 − α)^J; the portfolio keeps it at α or below (E2).

Contrast with surrogate-label pipelines. There, prompt and model choice create a non-vanishing error component (TEE) and flip conclusions in about 31% of hypotheses. Here, a bad reader can only make the bound smaller.

**P3 — Certified additive ledger.** Let reader A have label-free bins a(i) refining the cells. In-sample Pythagoras gives:

V(X) = V(X, a) + B(a), with B(a) = (1/m) Σ |p̄_{X,a(i)} − p̄_{X(i)}|².

- B(a) is bounded with P1, using a score H constant on (X, a) groups and centred inside X cells: the information A's categories capture.
- V(X, a) is bounded with P1 using any second reader centred on (X, a) groups: the information the text holds that A's categories miss.

At α/2 each, NLI ≥ L_B + L_V. The two terms add, and each is attributable to a reader.

**P4 — Reader regression test.** Use the OLD reader's scores, centred within the NEW reader's (X, bins) groups. A bound L > 0 certifies that the text holds information inside the new reader's categories that the old reader recovers: the upgrade lost it.

The same test is a **leakage audit**:
- old = full-text reader;
- new = the reader on time-masked text (notes before the cutoff only).

A positive bound means the full-text reader uses information that is not in the pre-outcome text.

**P5 — Certified selective reading.** A policy that runs an expensive reader only on cases chosen by a label-free rule (fixed on another fold) is itself a reader. P1–P4 therefore certify the information retained per token spent. This is Metis's role: a value-of-reading router with certified information per cost.

## 4. What the simulations show (`research/instrumented_20261019/results`)

See `docs/experiments/INSTRUMENTED_INFORMATION_SIMULATIONS_20261019.md`.

## 5. Open problems

1. **Pooled cross-fitting:** a valid bound that uses every row with every fold's model, without splitting α. A prequential (online) version could work, but the within-cell LOO centring breaks predictability.
2. **Tightness:** the bound is alignment² × V minus the deviation. Levers:
   - prompt ensembles (E8);
   - better nuisance models;
   - variance-adaptive inequalities.
3. **Upper bounds:** readers give only lower bounds. Upper bounds need the table floor (McDiarmid box) or a pair route.
4. **Non-binary and continuous outcomes:** the same projection works for bounded regression targets (scalar h); this still needs implementing.
5. **Reader-invariant information:** min over a class of readers, i.e. robust information under prompt perturbation.
6. **Cluster dependence:** the implementation supports arbitrary within-entity dependence only when the entity partition is correct; hospital-wide shocks or cross-patient dependence require larger independent blocks or a different concentration argument. Historical simulation tables below used independent rows and do not validate this extension.
