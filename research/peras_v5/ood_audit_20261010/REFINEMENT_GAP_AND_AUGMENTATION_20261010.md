# Representation-refinement gap: scope, sensitivity, and leakage controls

## Scope correction

The grouped calibration estimator/bounds now explicitly target **the declared group representation only** and return `raw_feature_certificate=False`. Independent calibration from `P(Y|G)` estimates a group-reference floor even if individuals have heterogeneous `P(Y|X)`. But it does not lower-bound the floor using richer X: coarsening can hide prediction-relevant information and raise Bayes risk. The earlier synthetic exact-law studies remain correct because their group key was sufficient by construction; their result must not transfer automatically to Mayo procedure/note groups.

`refinement_gap_profile.py` adds fail-closed transfer: `transfer_bound(result)` refuses a raw-information bound unless a valid gap cap and provenance are supplied. Provenance is recorded, **not validated**. `sensitivity_profile` exposes monotone curves without certifying any particular cap.

## Classical decision-theoretic correction

Let a group-reference bound satisfy `B_G <= R_G`. For a population refinement G=h(X), the value of additional information is the generalized-entropy gap `I_l = R_G-R_X >= 0`. This is classical Blackwell/Bayesian decision theory, already documented in the repository; it is not new Pythia information theory. If a justified `kappa >= R_G-R_X` is available, then

```
B_X(kappa) = max(0, B_G-kappa) <= R_X.
```

The same inequality holds for an audit-weighted group-reference risk and a finite audited full-information risk, provided kappa actually caps that difference; in that finite-design setting the reference-minus-target difference need not be nonnegative. The API requires a nonnegative upper cap, not an assumed population monotonicity claim.

Alternatively suppose eta upper-bounds the audit mean W1 between **clipped** individual full conditional laws and their clipped group-reference laws. Bayes-risk W1 stability yields

```
B_X(eta) = max(0, B_G-L*eta),
L = 1 for MAE; 0.9 for Q90; 2*(hi-lo) for clipped MSE.
```

Clipping contracts supported location-loss risk, so this bounds original-outcome risk. The risk-gap cap has loss units and can be much tighter than a generic W1 cap. Neither cap is identifiable here without substantive structure, external evidence, or separately controlled calibration. If a cap has its own statistical failure probability, it must be added to the group-bound budget. A supplied cap of zero is **not** evidence of representation sufficiency. Hard window losses remain unsupported by this W1 transfer.

## Known-noise tabular/text augmentation regression

`check_refinement_augmentation.py` uses two Bernoulli outcome groups with gold floors MAE=9.90, Q90=0.99, MSE=99.98; 800 independent calibration cases per group; and 400 fresh audit cases. Across 100 seeds, a target value is revealed for 0%, 25%, 50%, 75%, or 100% of randomly selected cases. Disclosure in a tabular value and the exact text grammar `duration outcome: VALUE minutes.` produce identical representations under a frozen deterministic parser. For each audited mask, the true full-information floor and exact information-gap/W1 caps are known from the DGP.

- 1,500 corrected loss cases: **0 risk-gap exceedances and 0 W1 exceedances**.
- Full disclosure: both corrected bounds are zero for all three losses.
- Deliberately false zero-gap sufficiency: **300/300 positive bounds at full disclosure**, despite a true zero floor. This is an assumption-negative-control, not a valid certificate.
- Missing cap: structured refusal.
- Sensitivity curves: monotone and zero beyond the breakpoint.
- Character-shuffled notes fail the exact parser, but **that does not establish loss of all raw-text information**. Character counts, digits, and signs can retain the target. The parser-derived representation loses the readable value; the raw-string estimand is different.

These are model-free synthetic checks, not fitted NLP performance or Mayo validation. The synthetic caps are oracle diagnostics and must never be presented as estimated, deployable Mayo caps.

## Current literature and limits of novelty

[Grünwald and Dawid (2004)](https://arxiv.org/abs/math/0410076) already connect generalized entropy to decision theory. The June 2026 preprint [Bayes-Sufficient Representations in Supervised Learning](https://arxiv.org/abs/2606.04045) distinguishes sufficiency for a fixed loss's Bayes action from preservation of the entire conditional law. This matters here: equality of full conditional laws is sufficient but stronger than necessary for a particular loss's floor. We do not claim the Bayes-quotient or loss-relative sufficiency idea.

[Shah and Peters (2020)](https://arxiv.org/abs/1804.07203) show an impossibility for uniformly valid nonparametric conditional-independence testing in broad continuous-conditioning settings; it is not a blanket impossibility for finite discrete groups or structured models. [He et al. (NeurIPS 2025)](https://arxiv.org/abs/2512.14000) further study practical kernel-test failures. Therefore a nonsignificant conditional-independence test cannot be converted into a proof that richer notes have no remaining information. The unresolved Pythia opportunity is a practically justified, selection-safe upper **information-gap** cap under declared structure—not a redefinition of information, a generic absence-of-signal test, or an assumption-free floor for every future representation.
