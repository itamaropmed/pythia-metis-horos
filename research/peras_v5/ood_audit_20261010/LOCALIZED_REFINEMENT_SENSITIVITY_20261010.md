# Localized representation-refinement sensitivity

## Honest result

Pythia can now express how a valid lower certificate for a declared finite group representation degrades when richer tabular or text information can reveal the target on only a bounded fraction of cases. The result is model-free and loss-routed. It is a conditional sensitivity theorem, not a procedure that discovers whether the representation is sufficient.

On ten frozen exact-law mechanisms, five untouched seeds, five disclosure levels, and eight losses, all 2,000 computed lower endpoints stayed below the known richer-information Bayes risk. The curves were monotone and reached zero under complete target disclosure. The exact text parser agreed with the equivalent tabular disclosure in every row; randomly permuted characters matched the exact grammar zero times. These are controlled implementation checks, not evidence that arbitrary shuffled text is information-free.

## Theorem: localized refinement transfer

Let `G=g(X)` be the declared group representation, let `X` be a richer information set, and let `R_G` and `R_X` be their Bayes risks for a fixed bounded-outcome loss. Suppose there is a predeclared exceptional event `E` with target probability at most `epsilon` such that, outside `E`, `G` is Bayes-sufficient for this loss: the conditional Bayes risks under `G` and `X` coincide. If `M_loss` uniformly upper-bounds conditional Bayes risk on the clipped outcome domain, then

```
0 <= R_G - R_X <= epsilon * M_loss.
```

Consequently, if `B_G <= R_G` is a simultaneous lower confidence bound, then

```
B_X(epsilon) = max(0, B_G - epsilon * M_loss)
```

is a lower confidence bound for `R_X`, subject to the same calibration assumptions and the declared localized-sufficiency assumption. The proof is the conditional risk decomposition over `E` and its complement. The gap is zero off `E`; on `E` it cannot exceed `M_loss`.

For a clipped interval of width `D`, the implemented distribution-free constants are:

| Loss | `M_loss` | Witness for the cap |
| --- | ---: | --- |
| MAE | `D/2` | midpoint action |
| MSE | `D^2/4` | midpoint action / variance bound |
| pinball `q` | `q(1-q)D` | fixed action `(1-q)lo + q hi` |
| window miss with half-width `h` | `0` if `2h >= D`, otherwise `1` | midpoint or loss range |

This statement is an elementary bounded-risk sensitivity result. In isolation it is not claimed as a new robust-statistics theorem. Its role in Pythia is to make a previously hidden sufficiency assumption quantitative, inspectable, and fail-closed.

## Frozen augmentation audit

`run_localized_refinement_ood.py` uses ten exact-law duration-like mechanisms, seeds 97001--97005, 20,000 independent calibration cases, and 800 audit cases. Disclosure masks are frozen per case and nested at nominal levels 0%, 10%, 25%, 50%, and 100%. When disclosed, the target appears both as a numeric tabular field and in the exact note sentence `duration outcome: <round-trip float> minutes.` No predictive model is fitted.

The table reports how many of the 50 mechanism-seed runs retained a strictly positive lower certificate at each nominal disclosure level. There were zero known-gold exceedances in every loss.

| Loss | 0% | 10% | 25% | 50% | 100% | Gold exceedances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAE | 50 | 25 | 0 | 0 | 0 | 0 |
| MSE | 50 | 0 | 0 | 0 | 0 | 0 |
| Q10 | 50 | 31 | 0 | 0 | 0 | 0 |
| Q50 | 50 | 25 | 0 | 0 | 0 | 0 |
| Q90 | 50 | 42 | 0 | 0 | 0 | 0 |
| W15 miss | 50 | 46 | 39 | 1 | 0 | 0 |
| W30 miss | 42 | 12 | 5 | 0 | 0 | 0 |
| W60 miss | 0 | 0 | 0 | 0 | 0 | 0 |

The fast collapse for magnitude losses is mathematically honest: a worst-case exceptional subset can carry a large share of the original irreducible risk. W15 survives longer because its loss range is only one. W60 was already near a zero floor and remains uninformative as a lower certificate.

An independent discrete stress check covered 8,000 loss/disclosure cases and 8,000 maximum-risk checks, with zero bound violations. Its deliberate negative control falsely declaring `epsilon=0` at complete disclosure returned a positive bound in 1,400 cases. That failure is expected and demonstrates that the assumption is load-bearing.

## What the character shuffle establishes—and does not

The shuffled note matched the registered exact grammar zero times, so the deterministic parser no longer extracts the target. It does **not** follow that shuffled text contains no target information: digits, sign, decimal characters, token counts, or a stronger decoder may retain it. Therefore Pythia must not automatically set `epsilon=0`, certify sufficiency, or label shuffled text as a no-information control. A valid no-information placebo should independently resample notes across outcome-independent units or generate text from an outcome-independent law while preserving the registered nuisance structure.

## Literature position

The ordering of experiments by information and the monotonicity of Bayes risk are classical decision-theoretic ideas. Generic Bayes-risk lower bounds based on information divergences are developed by Chen, Guntuboyina, and Zhang (JMLR 2016). Anderson and Bjarnadottir (PLOS ONE 2024) estimate attainable MAE through a constrained omniscient model, but their protocol depends on a chosen metric/linking rule and is not a finite-sample distribution-free certificate for arbitrary raw representations. Robust epsilon-contamination theory studies adversarial contamination of observations or distributions; Pythia's `epsilon` instead bounds the target mass on which a coarsened representation may cease to be Bayes-sufficient. Recent work on Bayes-sufficient representations makes clear that representation sufficiency and minimality are separate mathematical objects. Recent disagreement certificates bound target-model generalization or cross-model disagreement, not the attainable risk of all measurable predictors under a declared information set.

Accordingly, the defensible proposed novelty is the **combination** of:

1. a model-training-free, finite-sample certificate for the attainable risk of a registered representation;
2. loss-routed geometric witnesses and simultaneous concentration;
3. explicit refusal or sensitivity curves for representation refinement, tails, dependence, and selection;
4. falsification through outcome disclosure, placebo text, and known-law OOD mechanisms.

No single item above should be called novel without a complete theorem comparison and implementation of external baselines.

## Remaining theoretical gap

The full claim “bounds every possible model trained from arbitrary engineered tabular and clinical-note features” is not identified from finite observational data without an assumption that connects the declared representation to the raw information sigma-field. Non-significant residual tests, failed NLP models, or a shuffled-string parser cannot prove that connection. The current valid choices are:

- certify only the declared finite representation;
- supply and audit a structural sufficiency argument;
- provide an externally justified exceptional-mass or risk-gap cap and report the whole sensitivity curve; or
- refuse the full-information certificate.

The next theory target should therefore be an auditable **representation ladder**: pre-register nested deterministic maps, use fresh outcomes to bound the incremental decision-risk gap between adjacent maps, and spend a common error budget over the entire ladder. This may tighten the crude `epsilon * M_loss` penalty, but it will still require either a terminal sufficiency assumption or a refusal at the raw-information endpoint.

## Files and reproducibility

- Implementation: `refinement_gap_profile.py`
- Unit/property check: `check_localized_refinement.py`
- Frozen audit runner: `run_localized_refinement_ood.py`
- Raw rows: `localized_refinement_ood_results.jsonl`

The audit does not contain Mayo records. It uses exact-law synthetic mechanisms and stores only generated results.
