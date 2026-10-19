# Pythia representation-intervention research programme

## Objective

Turn target injection, text corruption, nuisance duplication, and neighborhood
rewiring into theorem-directed falsification tests rather than informal demos.
Every intervention must have a population-level risk relation fixed before
outcomes are inspected.  Synthetic mechanisms with known Bayes risk come
first; Mayo is a retrospective transfer only after the expected relation is
recovered on untouched synthetic data.

## Modern research gaps relevant to Pythia

1. **Dataset-level attainable risk with coverage.** Anderson--Bjarnadóttir
   estimates a model-agnostic MAE floor but explicitly describes its LP route
   as heuristic without finite-sample coverage. Classical Bayes-risk lower
   bounds usually concern priors/minimax estimation problems rather than the
   attainable prediction risk of one declared dataset representation.
2. **Information versus usable representation.** Predictive V-information
   measures information usable by a specified learner family. It can change
   under an invertible encoding because computational accessibility changes.
   Unrestricted Bayes risk cannot. Pythia needs to report these as different
   estimands rather than treating failed text extraction as absent information.
3. **Transformation invariance.** Blackwell order handles deterministic
   garblings and enrichments at the population decision level, but practical
   finite-sample floor witnesses can change when a lossless nuisance fragments
   neighborhoods. A valid but representation-unstable lower bound is not an
   information measure.
4. **Unverifiable no-information claims.** General conditional independence is
   untestable without restrictions. “This note field adds no information”
   therefore cannot be certified from an arbitrary observational table alone;
   it needs randomization, a registered finite alphabet, repeated environments,
   or an external structural cap.
5. **Extreme quantiles and tails.** The new curvature route materially repairs
   Q90 but still fails in high-SNR, heteroscedastic and skew mechanisms. Local
   curvature, richer pinball witnesses and tail-valid concentration remain
   open.
6. **Point estimation.** Training-free lower witnesses are not automatically
   accurate floor estimators. Pythia must keep an estimate tier separate and
   compare it with externally fitted kNN/graph/LP methods under matched
   information and splits.

## Population intervention contracts

For loss-specific Bayes risk `R*(Z)`:

| Intervention | Registered relation |
|---|---|
| bijective re-encoding of `Z` | exact equality |
| add `N` with `Y independent of N given Z` | exact equality |
| deterministic garbling `T(Z)` | `R*(T(Z)) >= R*(Z)` |
| enrich with any additional channel | new risk no larger |
| reveal the exact target with a visible marker | risk zero |
| independently reveal target on fraction `p` | `(1-p) R*(Z)` |
| encode target to a value with absolute error at most `h` | MAE at most `h` |

These are statements about the true decision problem. A numerical lower bound
need not be monotone under each transformation. The new falsification layer
tests whether reported risk intervals contradict the registered population
relation and distinguishes contradiction from insufficient resolution.

## Frozen experiment ladder

### Phase A — ten known-gold synthetic mechanisms

Use linear, Friedman, sparse, high-frequency, heteroscedastic, t3, lognormal,
discrete-cell, high-SNR and low-SNR mechanisms. Cross each with:

- literal target sentence at 0/10/25/50/100%;
- unary target code (`X` count), placed at fixed and random text positions;
- target rounded to 5 and 15 units;
- target plus registered noise levels;
- independent random `X` count and random identifiers;
- bijective token substitution and reversible cipher;
- token/character shuffle, span masking and note dropout;
- note permutation globally and within declared families;
- feature duplication and irrelevant dimensions.

Primary endpoints are MAE, Q90, W30 and CRPS where a valid witness exists.
Report true Bayes risk, lower endpoint, external point estimate, interval width,
refusal rate and whether the metamorphic contract is confirmed, unresolved or
contradicted. Mechanisms—not repeated seeds—are the significance unit.

### Phase B — family and neighborhood geometry

- multiscale neighborhood persistence rather than best-scale selection;
- same-family/similar-note, same-family/dissimilar-note,
  cross-family/similar-note and cross-family/dissimilar-note quadrants;
- within-family versus global note permutation;
- degree-preserving graph rewiring and community-label permutation;
- leave-one-family-out stability;
- rare-complication tail communities scored with Q90/W30/CRPS;
- bootstrap stability of neighbor edges and community membership.

All geometry is frozen without certification outcomes. Selection over scales,
partitions or losses receives an outer alpha allocation.

### Phase C — Mayo transfer

Run only aggregate, in-memory perturbations on the canonical cohort. Never
write modified notes or patient identifiers. Exact/partial target channels and
post-cutoff information are falsification controls and must be refused by the
public certificate. Mayo has no known Bayes gold, so it can establish logical
behavior, provenance enforcement and sensitivity—not universal tightness.

## Acceptance gates

1. Every intervention has a machine-readable contract and provenance.
2. Synthetic true risks satisfy the contract by construction.
3. A 95% interval may be called valid only after untouched coverage has a 95%
   Wilson lower bound of at least 0.95.
4. Harmless bijections/nuisance additions must not be described as information
   gain or loss when only the witness changes.
5. Target-derived channels collapse diagnostic risk but always trigger public
   refusal.
6. Family/neighborhood claims require at least 250 entities, multiplicity
   control and stability reporting.
7. Mayo anchors remain frozen until the synthetic gate and provenance audit
   pass.
