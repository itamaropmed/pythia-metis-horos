# Pythia backbone verdict and next gates — 2026-10-10

## Decision

**The general Pythia backbone is not complete.** A strong conditional core now exists, but the project does not yet justify a universal certificate for the best possible predictor under arbitrary tabular feature engineering and raw clinical notes. The ten-mechanism OOD evidence also does not establish broad statistically significant superiority over accepted methods.

The correct current headline is:

> Pythia provides model-training-free, finite-sample lower risk certificates for a registered bounded representation under explicit independence, stationarity, grouping/transport, and selection assumptions, with fail-closed sensitivity or refusal when those assumptions cannot be justified.

## Evidence used for the decision

### Frozen v5.7 point estimator

Ten new known-law outcome mechanisms, three seeds each:

| Loss | Strict mechanism wins / 10 | One-sided sign p | Pythia median relative error | Best-rival median relative error | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| MSE | 1 | 0.9990 | 0.0841 | 0.0639 | loses |
| MAE | 8 | 0.0547 | 0.0629 | 0.1057 | promising, not sign-test significant |
| Q90 pinball | 9 | 0.0107 | 0.0594 | 0.1799 | unadjusted significant only |
| W30 miss | 3 | 0.9453 | 0.1516 | 0.1761 | loses |
| W60 miss | 7 | 0.1719 | 0.2050 | 0.2707 | inconclusive |

None survives a Holm correction over the five loss questions. At matched `n=200`, the MAE point estimate was closer than the implemented Anderson--Bjarnadottir protocol on 9/10 mechanism medians (`p=0.0107`), but the Pythia bound still missed in 3/30 draws. Point-estimator accuracy cannot rescue an invalid certificate.

### Frozen v5.7 lower bound

The unseen OOD mechanisms produced truth exceedances in 7/30 MSE, 5/30 MAE, 1/30 Q90, and 4/30 W30 runs. Therefore SIMEX/overlap v5.7 is calibrated exploratory machinery, not a proved universal lower certificate.

### Replacement finite-group certificate

For a declared exact/sufficient group representation, the new SAA route has a finite-sample proof and obtained 0/400 clipped-law interval misses across ten mechanisms, five fresh seeds, and eight losses. Median lower/gold ratios were 0.855 MAE, 0.706 MSE, 0.782 Q10, 0.855 Q50, 0.767 Q90, 0.655 W15, 0.396 W30, and 0 W60. These are strong scoped results, not raw-feature certificates.

On a resource-matched exact-group comparison, grouped tuple SAA beat the transport route in 50/50 runs for MAE, Q90, and MSE. SAA optimism is an established principle; this result improves the implementation and router but is not by itself a novel theorem.

### Multi-point geometry result

Against a matched disjoint-pair Bentkus witness on ten hand-specified exact-law mechanisms, disjoint tuples were strictly tighter on 10/10 Q90 mechanisms, 6/10 MSE mechanisms, and 3/10 MAE mechanisms. The fresh jointly accounted selector preserved the Q90 gain. This is a credible narrow contribution: multi-point geometry helps skew-sensitive loss when exact conditional-law cells are externally justified. It is not evidence of dominance over all published Bayes-floor methods.

### Outcome-disclosure and text augmentation

The localized-refinement theorem was checked on ten mechanisms, five seeds, five nested disclosure levels, and eight losses: 0/2,000 lower endpoints exceeded known richer-information gold, and every certificate reached zero at complete disclosure. A deliberate false-sufficiency negative control failed in 1,400 cases. This demonstrates logical sensitivity and that the sufficiency assumption is indispensable.

## What is genuinely innovative enough to develop

The most defensible research contribution is not “we discovered a new definition of information.” It is an auditable certification system that combines:

1. model-training-free Bayes/attainable-risk witnesses;
2. loss-specific pair and multi-point geometry;
3. simultaneous finite-sample selection across routes;
4. representation, tail, dependence, and shift sensitivity objects;
5. structured refusal plus adversarial falsification tests.

The narrowest current new scientific lead is the Q90 multi-point witness/router under exact conditional-law groups. The broader systems novelty could be a certificate compiler that maps an estimand and assumption ledger to the strongest valid route while refusing unidentifiable claims. Metis should later search transformations only inside a frozen error-spending protocol; it must not be used to retroactively select successful simulations.

## Mandatory next gates

### Gate 1 — external comparator fidelity

- Reproduce the authors' Anderson--Bjarnadottir protocol from their public artifact, not only a local method-family reimplementation.
- Add any directly applicable published attainable-risk/Bayes-floor estimator with the same information and calibration budget.
- Separate point estimators from one-sided confidence bounds in every table.

### Gate 2 — representation ladder

- Pre-register nested deterministic representations from tabular fields through note concepts and text bundles.
- Develop simultaneous incremental risk-gap bounds between adjacent rungs using independent calibration outcomes.
- End in an explicit terminal sufficiency assumption, sensitivity curve, or refusal; never infer sufficiency from a weak learner's failure.

### Gate 3 — near-replicate heterogeneity

- Replace oracle exact-law keys with an externally auditable conditional-law distance cap.
- Test a fresh 10-mechanism near-replicate suite, including discontinuities, sparse interactions, heteroskedasticity, mixtures, and heavy tails.
- Require zero theorem-level violations under satisfied assumptions and report the nonvacuity/tightness cost.

### Gate 4 — fair superiority experiment

- Freeze mechanisms, sample sizes, seeds, representations, hyperparameters, and all method-selection rules before evaluation.
- Treat mechanism as the unit of inference and specify one primary loss.
- Give every method the same data, grouping knowledge, and tuning budget.
- Require a multiplicity-aware superiority result; do not redefine success after inspecting losses.

### Gate 5 — Mayo transfer

- Preserve the existing Mayo numerical anchors as `calibrated`/retrospective.
- Audit patient clustering, temporal stationarity, cutoff provenance, and the exact representation ladder.
- Run target disclosure and valid outcome-independent note placebos on sandboxed copies only.
- Do not promote the Mayo 28.8496-minute result to `proved` from synthetic success alone.

## Stop/go criterion

Proceed with the theory programme because the scoped certificate and Q90 multi-point result are real. Do **not** claim that Pythia already bounds every possible model or is broadly SOTA. A paper-ready general backbone requires Gates 1--4; a Mayo certificate additionally requires Gate 5.
