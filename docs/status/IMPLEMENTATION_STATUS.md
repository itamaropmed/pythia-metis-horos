# Implementation status

## Completed in the first advancement pass

- Git repository initialized; patient-derived and generated data ignored.
- A tracked manifest records 15 patient-data/guard hashes and seven matrix shapes without
  recording clinical contents; the smoke check now fails on drift.
- Package discovery fixed and development dependencies installed.
- Immutable public contracts and `certify(...)` API added.
- Structured refusal and five claim tiers added.
- Guarded MAE, 2-D/median envelopes, Qcap, sqrt(2), stratification, and clustered
  bootstrap ported into reusable code.
- Frozen Mayo Arm B reproduced exactly.
- Mayo A/B/C information-set ladder added.
- Exploratory squared-loss dependence profile added; it is not used for certification.
- Incumbent artifact audit added and configured to fail without OOF provenance.
- Clinical-note cutoff and patient-fold isolation audit added; it emits aggregate counts
  and hashes without reading or reporting note text.
- The shipped capped note bundle is explicitly rejected by the strict deployable cutoff
  audit; a deterministic re-cut command is available as an interim rung before the raw-corpus
  rebuild.
- Prospective 3,000-case freeze and analysis protocol drafted.
- New independent 500-design MAE gate: 491 evaluated, 491 valid, 9 correctly refused as
  outside the guard domain; Wilson lower 95% coverage bound 0.9922.
- Arm B now includes a deterministic 800-replicate patient-clustered sampling interval;
  guard/theory uncertainty remains visibly separated in the four-rung ladder.
- The exploratory dependence profile now reports feasible dependence/risk intervals, active
  constraints, and an analytic collapse point. A 24-design latent common-bias regression
  suite varies noise, SNR, error family, and common bias independently.
- The interim deployable re-cut retained 92,552 of 153,656 capped source notes across
  11,590 cases and re-audited with zero cutoff or alignment violations. Its artifact hash is
  `26bf4c6124d948d713a245cf8152ec7d285e249958626d40cb229db88d38d69f`.
- Canonical historical folds are now materialized: 10,882 development rows, 2,839 temporal
  holdout rows, and 474 earlier rows excluded because their patients enter holdout. There is
  zero patient overlap or multi-fold assignment. Manifest hash:
  `15508f9391e72c636e9b958039c4dfc5c2bb32faa1bc1fd820137c03493699a7`.
- The strict note re-cut passes both cutoff and patient-fold isolation against that manifest.
- Generic study YAML loading, grouped temporal preparation, and entity-safe cross-fitting are
  implemented. Mayo now exercises these interfaces through `mayo.study.yaml`.
- The tuned structured XGBoost candidate was regenerated through the generic crossfit engine:
  OOF MAE `33.2749753373`, 10,882/10,882 predictions, exact cohort/fold audit passed, artifact
  hash `da1a5003daffdacf3c6ea8da58aff647e75c1a655f43518d659275499125b232`.
- A fail-closed `DependenceGuardSpec` now exposes the finite-variance Catoni rung only for
  frozen predictors on external/prospective independent patients; unsupported contracts
  return structured refusals.
- A `RangeDependenceGuardSpec` and patient-level empirical-Bernstein rung remove the need
  for guessed variance caps when honest endpoint and prediction-difference ranges exist.
  Its sharp confidence-rectangle geometry preserves the full pairwise risk-difference
  information instead of using only the conservative excess-risk sum inequality.
- In 1,800 variable-size patient simulations the range rung was valid 1,800/1,800
  (Wilson lower 0.9979), was strictly tighter than matched patient Hoeffding in 1,797,
  and reached median tightness 0.950 at 10,000 patients. The legacy bound covered 66.3%.
- A second, predictor-independent near-replicate rung now supports both MAE and MSE.
  It pairs distinct entities only inside randomized blocks, concentrates across blocks,
  exposes a simultaneous Lipschitz sensitivity curve, and never assumes that global
  matching produces iid pair scores.
- Across 3,600 near-replicate simulations the corrected rung was valid 3,600/3,600
  (Wilson lower 0.9989), versus 78.7% for the uncorrected difference method and
  88.3% when the true Lipschitz constant was understated by half.
- The generic CLI now consumes hash-pinned NPZ fold/pair artifacts and versioned YAML
  guards for the finite-variance, bounded-range, and replicate paths. Artifact drift and
  unaudited pair designs fail closed.
- A retrospective Mayo replicate sensitivity uses 10,528 of 10,532 patient-level rows
  in 1,316 blocks. Its entire nine-point curve is simultaneous, remains exploratory,
  and explicitly refuses self-certification of iid sampling or outcome blindness.
- A fixed-design replicate rung now conditions on audited covariates and concentrates
  across all disjoint pairs. It supports Hoeffding plus the independent/non-identical
  empirical-Bernstein inequality of Maurer--Pontil Theorem 11, and explicitly changes
  the estimand from sampling-population risk to the audited covariate population.
- Deterministic global nearest pairing removes the randomized-block geometry penalty
  without reading outcomes. In 1,500 adversarial runs, global conditional empirical
  Bernstein was valid 1,500/1,500 (Wilson lower 0.9974), reached median tightness
  0.624 versus 0.250 for blocked conditional empirical Bernstein, and was tighter in
  1,356/1,500 runs. At 10,000 entities its median tightness was 0.752.
- On the retrospective Mayo diagnostic, the global fixed-design empirical-Bernstein
  curve is 26.61--23.78 minutes for tabular and 28.41--22.50 for tabular plus note
  values over `L=0..600`. These remain exploratory, not certificates.
- Conditional Lipschitz curves now use a single outcome-term confidence event for
  the entire `L` grid; only alternative pair designs consume multiplicity budget.
  This is an exact tightening, not a calibration discovered on Mayo outcomes.
- Conditional block Hoeffding and empirical Bernstein now provide a real-world
  downgrade when entity independence is implausible. A 500-run latent cluster-shock
  test gave 500/500 validity (Wilson lower 0.992); the invalid pair-independence
  shortcut covered only 93.4% (Wilson lower 0.909).
- The replicate guard now supports fail-closed `auto` selection: audited entity
  independence selects pair-level conditional empirical Bernstein; audited cluster
  independence selects the block rung; missing audits produce a refusal.
- A generic clustered nearest-pair constructor now pairs without outcomes inside
  declared operational clusters and records deterministic attrition. On Mayo, an
  exploratory 170-ISO-week analysis retained 10,444 patients; its tabular curve is
  12.50--2.99 minutes and its note-value curve is 12.92--0 over `L=0..600`.
- A ten-generator, 1,200-run known-risk audit found 1,200/1,200 valid global
  conditional empirical-Bernstein bounds and 957/1,200 strict improvements over
  matched Hoeffding, but also a zero-bound sparse 10D failure. A prespecified
  signal-coordinate metric recovered a positive bound on the same sparse DGP.
- An external leave-one-out kNN floor comparator beat the pair center on MAE
  point error in 190/200 synthetic runs. The pair center is now explicitly
  labeled a lower-bound plug-in, not a universally accurate MAE estimator.
- Mayo in-memory actual-duration injection into 10%, 50%, and 100% of note
  bundles yielded zero error on injected rows; the public API refused the
  leaked representation. Duplicating a legitimate tabular feature moved the
  exploratory numeric bound despite unchanged information content. See
  `REPLICATE_AND_MAYO_AUDIT_20260912.md` for exact results and assumptions.
- Paired-replicate certification now requires a separately hashed representation
  and explicit timing/outcome-blindness attestation. Duplicate feature vectors
  no longer cause self-pairing; corrected centers are clipped to nonnegative risk.

## Still gated

- A case-weighted, patient-cluster-aware finite-cell SAA endpoint and matching
  fixed-policy information-value bridge are implemented for MAE, pinball, and
  window-miss loss.  Their bounded-difference widths treat independent
  patients as sampling units while permitting arbitrary within-patient
  dependence.  A preregistered ten-law battery had 1,000/1,000 validity
  (Wilson lower 0.9962), 867/1,000 positive strict improvements over the
  matched cluster-valid global-range endpoint, and median bound/truth 0.3659.
  In a separate repeated-row falsification, the cluster route remained valid
  1,000/1,000 while an invalid row-IID shortcut covered only 638/1,000.  An
  aggregate-only Mayo transfer retains 11,356 cases from 9,300 patients and
  widens the five-node interval by 2.613 minutes relative to the row shortcut;
  it remains exploratory because cluster independence, temporal stability and
  the working support are not externally attested.  Variance-sensitive
  concentration for this nonlinear SAA functional is the next theory gap.
- A complementary support-free clustered block-SAA route now partitions
  patients outcome-blindly, computes independent finite-cell empirical optima,
  truncates witness scores rather than outcomes, and selects a finite
  route/cap portfolio under an explicit alpha ledger.  It performs no model
  training and needs no population outcome range.  On a frozen Mayo-scale
  confirmation with 4,096 patients, two perfectly dependent cases per patient,
  and ten unbounded noise laws, it was valid, positive, and strictly tighter
  than matched support-free disjoint-pair/dKelly in **500/500** runs.  Median
  bound/truth was 0.7865 versus 0.6728 and all ten mechanism medians won.  On
  retrospective Mayo it gives exploratory lower endpoints 77.398 (global),
  51.765 (family), 30.094 (procedure), 25.398 (procedure+metadata), and 23.876
  (procedure+hashed bag).  The deterioration under fine note cells exposes
  fragmentation rather than proving that notes remove information.

- A machine-readable representation-intervention auditor now covers bijective
  re-encoding, independent nuisance, Blackwell garbling/enrichment, exact and
  partial target reveal, and bounded MAE quantization. It distinguishes
  confirmed, compatible, unresolved and contradicted interval relations and
  refuses unattested transformation contracts.
- A new unequal-cell SAA endpoint gives a training-free MAE, pinball and
  window-miss lower bound for frozen finite text/concept representations. Its
  entity-level McDiarmid proof supports singletons and arbitrary cell sizes,
  requires a population hard-loss cap, iid rows and outcome-blind labels, and
  refuses literal or encoded target leakage.
- A frozen 5,000-row ten-mechanism intervention battery produced zero
  exceedances in 4,000 accepted rows and refused all 1,000 target-derived
  rows. Bijective encodings, canonical nuisance removal and bag-token shuffles
  were exactly invariant in 500/500 paired cases. Retaining independent random
  identifiers left true risk unchanged but reduced median bound/gold from
  0.525 to zero, exposing representation fragmentation.
- A cell-local bounded-difference refinement replaced the global outcome range
  by externally declared hard conditional ranges. On a preregistered fresh
  4,000-row confirmation it had zero known-floor exceedances, raised baseline
  median bound/gold from 0.5298 to 0.7045, strictly improved all 500 baseline
  comparisons and repaired all 50 high-SNR zero bounds. Baseline, bijective,
  canonical-nuisance and bag-token representations remained exactly invariant
  in 500/500 cases. All 10 mechanism-level baseline medians improved (two-sided
  sign-test p=0.001953125). The refinement is conditional on externally valid
  cellwise hard supports; current Mayo data do not yet supply that contract.
- A simultaneous finite representation-lattice endpoint now composes
  finite-cell bounds with deterministic Blackwell garbling and independently
  attested equal-risk contracts. It uses one explicit Bonferroni ledger,
  checks observed nesting, performs no model training, and refuses missing
  population/timing provenance. In a preregistered 500-run confirmation over
  ten noise mechanisms, all 2,500 raw and closed node endpoints stayed below
  known Bayes risk, all 500 lattices respected the information order, and an
  independent-nuisance equality repaired text fragmentation in 500/500 runs.
  Median nuisance bound/gold rose from 0.5197 to 0.7325, exactly matching the
  canonical note-neighborhood endpoint. Patient clustering, temporal shift
  and defensible Mayo support maps remain required before clinical use.
- An outcome-blind estimator router now selects the least-fragmented member of
  an independently attested equal-risk representation class. Empirical
  refinement can only decrease the in-sample SAA optimum, giving the route a
  weak expected-downward-bias justification without claiming universal error
  dominance. On a separately registered fresh 500-run confirmation, median
  absolute floor-estimation error fell from 0.1345 to 0.0130, with 496/500
  paired wins and improved medians in all ten mechanisms (two-sided sign-test
  p=0.001953125). The result applies only when information equivalence is
  established externally; it is not yet a general estimator SOTA result.
- A model-free finite-refinement permutation test now evaluates whether a
  frozen neighborhood adds decision-relevant signal beyond finite coarse
  strata. It supports MAE, pinball and window-miss loss, trains no predictor,
  and refuses target-derived or unattested production use. Known informative
  concepts were detected in 500/500 fresh runs. A separate 1,000-seed null
  audit rejected 5.6% at nominal 5% (95% Wilson 4.34%--7.20%). Without audited
  conditional exchangeability it returns an exploratory tail area, not a
  valid p-value or information-value bound.
- The first aggregate-only Mayo family/neighborhood lattice used 9,300 unique
  patients and is explicitly exploratory. Note metadata beyond procedure had
  an empirical SAA reduction of 0.9715 minutes versus a 0.8882 permutation
  99th percentile (diagnostic two-test tail 0.004); a generic hashed lexical
  bag added no signal beyond metadata (0.744). Token shuffle and parser-removed
  target-coded X were exactly invariant; the raw target-coded X channel was
  refused. No row-level clinical data were persisted.
- A fixed-policy information-value bridge now lower-bounds the magnitude
  `R_coarse-R_refined` by combining a simultaneous coarse-risk lower endpoint
  with an upper risk endpoint for an externally frozen policy measurable from
  the refined representation. Pythia performs no fitting and refuses policies
  selected on evaluation outcomes, using post-outcome inputs, or outside the
  declared information set. On a preregistered 500-run, ten-mechanism
  confirmation at 8,192 entities, it had zero gold violations, positive bounds
  in 500/500 runs and 500/500 strict wins over matched global-range/Hoeffding.
  Median bound/gold was 0.6429 versus 0.4546. Mayo still lacks a prospectively
  frozen note policy and independently audited evaluation cohort for this
  magnitude claim.
- The separate clinical-notes 31.5564 artifact is now located and its saved
  verifier passes exactly: MAE 31.556418073067334 on 2,839 later cases with
  zero recorded patient overlap, after OOF-only model selection. This confirms
  the retrospective number but not the unrelated 31.2623 claim. Applying the
  fixed-policy magnitude bridge with an explicit exploratory downgrade gives
  zero lower value beyond both procedure family and procedure: the family
  coarse lower endpoint is 30.2263 versus policy-risk upper 33.8533. The
  negative result is preserved rather than promoted into a note-value claim.

- A curvature-corrected block-SAA route now replaces the first-order
  MAE/pinball action-drift correction by the minimum of a Lipschitz penalty and
  a second-order Jensen-gap penalty.  The public API requires explicit action
  dispersion and curvature provenance, supports fail-closed local curvature
  domains, and labels the output conditional.  The underlying smoothness
  inequality is classical; the candidate contribution is its composition into
  the Pythia finite-sample certificate architecture.  In a frozen ten-mechanism
  confirmation with learning-half score caps, MAE median tightness rose from
  0.5018 to 0.8356 (learned key) and 0.5244 to 0.8403 (aligned), with 10/10
  mechanism-median wins for both and zero known-floor exceedances.  Q90 rose
  from zero to 0.5979/0.6285; the aligned route won 10/10 mechanism medians,
  while learned-key Q90 won 8/10 and was not significant at mechanism level.
  High-SNR, heteroscedastic, and skew Q90 remain important failures.
- Honest repeated environments now provide a distribution-free, non-oracle
  bridge for median/quantile action dispersion: simultaneous order-statistic
  intervals are converted to exact worst-case block dispersion by vertex
  enumeration.  It allows arbitrary within-environment cross-unit dependence,
  requires independent environment rows and hard support provenance, and
  refuses blocks larger than the exact enumerator's registered capacity.
  A frozen 1,000-row ten-mechanism battery covered the exact variance, MAD,
  maximum-displacement, and curvature-penalty targets in 1,000/1,000 rows
  (95% Wilson lower 0.9962). Median curvature cap/truth was 1.221 for median
  actions and 1.284 for Q90 actions; worst weak-signal ratios were 6.13 and
  8.55. Current Mayo data have one outcome per case and therefore do not
  qualify.

- A reusable declared-feature-modulus bridge now converts frozen block
  geometry into simultaneous MAE, pinball, and MSE sensitivity curves.  It
  requires metric and modulus provenance and performs no model training.  The
  supplied synthetic attachment reproduces its oracle-discrepancy claims, but
  no non-oracle Mayo modulus is available; the clinical route remains
  conditional rather than certified.
- Independent repeated environments can now calibrate the absolute and
  squared barycentric discrepancies without oracle laws.  A frozen
  dKelly/Bentkus router covered the analytic population penalty in 1,000/1,000
  fresh runs for each family and materially tightened the absolute endpoint.
  The squared endpoint remains highly range-limited, and the API flags that
  exact target replicates also permit the stronger direct-SAA comparator.
  Current Mayo data do not satisfy the repeated-environment contract.
- A hard-support refinement computes exact blockwise barycentric cost ranges
  by vertex enumeration.  On a second untouched 1,000-run confirmation it
  reduced median absolute and squared cap/truth ratios to 1.091 and 2.009,
  with zero misses in both families.  This requires externally justified
  unit-level hard supports; fitted intervals and current Mayo metadata do not
  qualify.
- Overlapping multi-point Q90 routes now include an entity-McDiarmid witness
  and an arithmetic mixture of partition-level dKelly e-values.  The former
  lost to disjoint dKelly; the latter confirmed a small valid improvement
  (median 0.8213 vs 0.8192, 8/10 mechanism medians, zero violations) that was
  not significant at the mechanism level.  Disjoint dKelly remains the
  production default while geometry-weighted nonzero-discrepancy tests remain
  open.

- The second ten-design MAE challenge (200 draws at each of 512 and 192 entities)
  validates the selection-safe paired/grouped envelope on all tested draws,
  but still exposes a zero-bound sparse 12D case and no uniformly superior
  train-free point estimator. See `THEORY_GAP_CLOSURE_20260913.md`.
- The public CI workflow and aggregate Mayo hash/value lock are authored but
  cannot run remotely until this uncommitted repository gets a Git remote and
  a pushed revision. A private runner is required to recompute clinical values.

- The alleged 31.2623 incumbent is not supported by the located artifact: the exact matching
  value is an OOF residual-corrector selection metric, not held-out duration MAE. A valid
  final-duration OOF artifact still needs to be identified or regenerated.
- The new dependence derivations still need an independent theorem audit before a paper
  claim, and retrospective OOF predictions remain outside the proved sampling contract.
- The original external 50-design library cannot yet be regenerated solely from this repo.
- RM-22 and full-corpus note arms must be rebuilt on the canonical cohort and fold contract.
- Prospective confirmation requires 3,000 newly arriving eligible cases.
- Metis autonomy and full Horos UI remain blocked on M2.
- No current Mayo result is established as the universally tightest valid MAE
  bound or the most accurate Bayes-floor estimator. The ten-design comparison
  is not an implementation of every literature method.
- The generic fold-matrix loader still permits a prediction-artifact hash to
  stand in for a representation source hash. That provenance contract needs
  the same separation added to paired-replicate requests before prospective
  certification.
- A train-free generalized pair/group decision envelope now covers q10/q90
  pinball and W15/W30/W60 miss risk, with outcome-blind family routing and a
  finite-group pinball score cap (the asymptotic cap is `tau*(1-tau)*R`).
  The public paired API accepts
  conditional Bentkus. See `docs/experiments/MULTILOSS_VALIDATION_20260913.md` for known-risk
  tests, accepted concentration comparators, external point-estimation
  baselines, negative results, and exploratory Mayo transfer. These results
  do not establish a universally best bound or estimator.
- The reusable multi-loss pair/group and geometry route are now exported from
  `pythia`. A separate, train-free, aggregate-only Mayo note perturbation
  diagnostic covers identity, partial/full dropout, character/token shuffle,
  span mask, segment dropout and duplication on one fixed patient cohort.
  It does not fit text features or predictors and does not overwrite frozen
  Mayo anchors. The numerical bound is not monotone under every perturbation;
  changing the text changes the metric and invalidates such a simple
  expectation. See `docs/status/PACKAGE_AND_AUGMENTATION_GUIDE_20260914.md`.

## 2026-10-14 (Claude session): fixed-action dominated SAA

- `pythia.core.dominated_saa`: full-sample, support-free, patient-cluster-valid lower endpoints for
  MAE, any pinball level and window-miss.  The exact truncated in-cell optimum is pathwise
  dominated by an unobserved additive sum of independent bounded cluster losses at the Bayes
  actions, so range-only Bentkus/Hoeffding invert it (no blocks, no outcome support, no model).
  Outcome-blind retention grid charged in the ledger.  Theory: `docs/theory/DOMINATED_SAA_20261014.md`.
- Untouched confirmation (500 draws, frozen support-free design): MAE valid 500/500, median
  bound/gold 0.900 vs block-SAA 0.787 (464/500 strict wins, 9/10 mechanisms; contaminated Gaussian
  ties); Q90 0.847 vs 0.687 (10/10); window-miss 0.934 vs 0.631 (10/10).  Stress: 0/2,400 violations.
- Mayo (exploratory, 14,195-case cohort): fragmented note representations hold ~36-37 min lower
  endpoints instead of 24-30; held-out fixed-policy information value beyond procedure family is
  now positive (10.76 min, joint ledger).  Procedure-level value remains 0.
- Remaining gaps: variance adaptivity for the nonlinear functional; global-node cluster-range
  inflation; small-G tightness; cohort reconciliation with the 11,356-case legacy cohort.
