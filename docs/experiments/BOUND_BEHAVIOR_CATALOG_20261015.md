# Pythia bound-behavior catalog

**Frozen documentation date:** 2026-10-15  
**Scope:** synthetic simulations, representation interventions, leakage and
augmentation tests, dependence falsifications, multi-loss checks, and Mayo
retrospective diagnostics implemented or preserved in this repository.

This document is the canonical map of what was tested and how the **current**
Pythia bounds behave. It deliberately preserves negative and superseded results.
It does not turn retrospective Mayo diagnostics into certificates, and it does
not claim that Pythia is universally the tightest lower bound or the best point
estimator.

## 1. What the number means

Pythia targets a lower endpoint for an **attainable Bayes risk** under a declared
information set, loss, sampling unit, and structural contract. A larger valid
lower endpoint is tighter: it says that even the best possible decision rule
using that information cannot attain loss below the endpoint. Pythia itself
does not train the predictor whose achieved loss is being compared with the
floor.

Three quantities must not be conflated:

1. **Gold risk**: known only in a simulator, or zero in a deliberately leaked
   oracle diagnostic.
2. **Point estimate**: an estimate of the gold floor; accuracy is not implied by
   lower-bound validity.
3. **Lower endpoint**: intended to stay at or below the gold floor with the
   registered confidence contract; tightness is endpoint divided by gold when
   gold is positive.

The repository uses five claim tiers:

- `proved`: the code instantiates an audited mathematical implication under
  explicitly checked premises;
- `conditional`: the implication is justified, but substantive assumptions
  such as support, independence, modulus, or provenance are externally supplied;
- `calibrated`: empirical coverage was established on a frozen design family,
  without a theorem broad enough for unconditional certification;
- `exploratory`: a diagnostic number without a complete certification contract;
- `refused`: a required premise failed or was not supplied.

Only `proved` and explicitly qualified `conditional` outputs may be described
as certified. Every Mayo result in this catalog is retrospective; most current
Mayo bound readouts are exploratory.

## 2. Final expected behavior under interventions

| Intervention | Population Bayes risk | Expected current Pythia behavior | What was observed |
|---|---|---|---|
| Bijective re-encoding | Exactly unchanged | Same endpoint when the registered geometry/cells are equivalent | Exact equality in 500/500 synthetic cases |
| Token-order shuffle under a bag parser | Exactly unchanged for that parser | Same representation and endpoint | Exact equality in 500/500 synthetic cases and exact Mayo invariance |
| Add independent nuisance then canonically remove it | Unchanged | Same endpoint | Exact equality in 500/500 cases |
| Retain an independent random identifier as a cell key | Unchanged | Remain valid but usually become weaker through fragmentation | Median bound/gold fell 0.525 to 0; equality-lattice closure restored it to 0.733 |
| Duplicate a deterministic feature | Unchanged | No automatic invariance unless the metric is canonicalized | Mayo pseudo-bound changed 22.2148 to 20.2881 |
| Garble/coarsen information | Cannot lower Bayes risk | Respect registered Blackwell order | No order failures in 500 finite lattices |
| Reveal the target on some cases | Risk decreases on those cases | Sensitivity curve decreases; production certificate refuses a target-derived representation | Synthetic curves were monotone; diagnostic mixed Mayo MAE fell 32.1905 to 29.0118, 16.8501, and 0 |
| Reveal the target on all cases | Gold risk is zero (up to declared quantization) | Diagnostic endpoint reaches zero; public certification refuses leakage | Full-oracle pair endpoint 0; finite-cell APIs refused all leaked rows |
| Encode target as unary `X` or a hidden character count | Potentially near-zero | Refuse if provenance identifies it; otherwise no method can guarantee detection | Raw X channel refused; an older unrecognized `m`-count channel invalidated every method |
| Character shuffle | Depends on the raw information and decoder | Report only what the registered parser consumes; never infer raw text is information-free | Exact duration grammar matched 0 shuffled strings, but digits/counts may still leak |
| Generic text dropout/mask/duplication | No universal ordering | Endpoint may rise or fall because cells, distances, score range, and estimand change | Mayo q90/W30 diagnostics were non-monotone |
| Duplicate rows from one patient | No new independent evidence | Cluster route widens or stays stable; row-IID shortcut must be refused | Cluster route 1000/1000 valid; row-IID shortcut only 638/1000 valid |
| Add genuinely informative pre-outcome concepts | Bayes risk cannot increase | Finer-representation floor may fall; value requires a gap test or frozen-policy bridge | Synthetic concept detected 500/500; Mayo metadata signal was exploratory, lexical bag was null |

The central lesson is that a valid lower bound is not required to move
monotonically under arbitrary string or feature edits. Information ordering is
a statement about sigma-fields and Bayes risks. A finite-sample witness also
depends on how the declared representation creates cells, neighborhoods,
distances, score caps, and effective independent units.

## 3. Current strongest confirmation results

These are the present theorem-backed or explicitly conditional synthetic
results that should be used when describing the final implementation.

| Route | Frozen design | Validity | Main tightness/result | Important limit |
|---|---:|---:|---:|---|
| Fixed-action dominated SAA | 500 draws, 10 unbounded laws, clustered cases | MAE/q90/window each 500/500; stress 0/2,400 violations | MAE median 0.8997 of gold; q90 0.8473; window 0.9341 | Not variance-adaptive; contaminated Gaussian limited by cap grid |
| Support-free clustered block SAA | 500 draws, 4,096 patients, 2 perfectly dependent cases/patient, 10 unbounded laws | 500/500 | Median 0.7865 vs matched pair 0.6728; 500/500 strict wins | Fine cells fragment blocks; MAE confirmation was primary |
| Clustered finite-cell local range | 1,000 draws, 10 bounded laws | 1,000/1,000 | Median 0.3659 vs global-range 0; 867 positive strict wins | Rare-tail floor unresolved; needs external cell supports |
| Cell-local finite SAA | 4,000 accepted intervention rows | 0 exceedances | Baseline median 0.7045 vs global 0.5298; 500/500 baseline wins | Cellwise hard supports are externally supplied |
| Representation lattice | 500 runs, 5 nodes/run | 0/2,500 raw or closed exceedances | 0 order failures; nuisance repaired 500/500 | Equality/order contracts must be external |
| Fixed-policy information value | 500 runs, 8,192 entities, 10 laws | 0 violations | Positive 500/500; median 0.6429 vs 0.4546 baseline | Requires a genuinely frozen refined-information policy |
| Curvature-corrected block SAA | 400 rows, 10 mechanisms, MAE/q90 | 0 exceedances | MAE 0.8356/0.8403 vs 0.5018/0.5244; q90 0.5979/0.6285 vs 0 | Simulator supplied action dispersion/curvature |
| Repeated-environment dispersion calibration | 1,000 rows | All four targets covered 1,000/1,000 | Median curvature cap/truth 1.221 MAE, 1.284 q90 | Mayo has no repeated outcome environments |
| Support-aware repeated modulus | 1,000 fresh rows per penalty | 0 misses | Cap/truth 1.091 absolute, 2.009 squared | Requires hard unit-level supports |
| Range dependence guard | 1,800 patient simulations | 1,800/1,800 | 1,797 Hoeffding wins; median tightness 0.950 at 10k patients | Frozen predictors/range and patient sampling contract |
| Corrected near-replicate route | 3,600 simulations | 3,600/3,600 | Invalid uncorrected route covered 78.7% | Requires outcome-blind pairs and valid action modulus |
| Global fixed-design replicate | 1,500 adversarial runs | 1,500/1,500 | Median 0.624 vs blocked 0.250; 1,356 wins | Conditional on audited covariates and entity independence |

The newest dominated-SAA route improves finite-cell concentration by avoiding
block-level SAA optimism: an observed truncated cell optimum is bounded by an
unobserved additive sum of independent cluster losses at fixed Bayes actions.
It is currently the strongest general finite-cell result in the repository,
but does not erase the need for outcome-blind cell labels, cluster independence,
an alpha ledger, or honest score truncation.

Primary reports:

- [dominated SAA](../../research/peras_v5/domination_20261014/DOMINATED_SAA_CONFIRMATION_RESULTS.md)
- [support-free cluster block SAA](../../research/peras_v5/interventions_20261014/SUPPORT_FREE_CLUSTER_BLOCK_RESULTS.md)
- [clustered finite-cell confirmation](../../research/peras_v5/interventions_20261014/CLUSTERED_CELL_CONFIRMATION_RESULTS.md)
- [cell-local confirmation](../../research/peras_v5/interventions_20261014/CELL_RANGE_CONFIRMATION_RESULTS.md)
- [family lattice](../../research/peras_v5/interventions_20261014/FAMILY_LATTICE_CONFIRMATION_RESULTS.md)
- [fixed-policy value](../../research/peras_v5/interventions_20261014/FIXED_POLICY_VALUE_CONFIRMATION_RESULTS.md)
- [curvature correction](../../research/peras_v5/diagnostics_20261014/CURVATURE_CONFIRMATION_RESULTS_20261014.md)

## 4. Target injection and leakage tests

### 4.1 Mayo actual-duration injection: current falsification

The canonical in-memory diagnostic selected 2,048 unique patients with
pre-cutoff note bundles. It appended a literal actual-duration sentence to a
selected fraction, parsed that marker, and also supplied the same value as a
tabular oracle coordinate. Nothing patient-derived was written back to disk.

| Injected fraction | Baseline/fused diagnostic MAE (minutes) | Expected behavior |
|---:|---:|---|
| 0% | 32.1905 | Ordinary frozen OOF readout on this subset |
| 10% | 29.0118 | Approximately 10% of residual loss removed |
| 50% | 16.8501 | Approximately half of residual loss removed |
| 100% | 0.0000 | Oracle predicts every target exactly |

On full injection, the pair center and lower endpoint were numerically zero
under the correct oracle-coordinate modulus `L=900`. The public `certify` API
returned a structured refusal because the representation is target-derived
and post-prediction-time. A separate 4,096-patient draw reproduced baseline
32.2685 and mixed MAEs 28.9918, 16.3180, and 0.0.

This is exactly the desired distinction:

- **diagnostic mode** confirms mathematical responsiveness to a perfect
  information channel;
- **production certification** rejects the channel because it is unavailable
  at decision time.

Source: [replicate and Mayo leakage audit](../audits/REPLICATE_AND_MAYO_AUDIT_20260912.md).

### 4.2 Synthetic localized disclosure: final theorem check

Ten exact-law duration mechanisms, five untouched seeds, five nested target
disclosure levels, and eight losses produced 2,000 endpoints. All endpoints
were at or below the known richer-information risk; every curve was monotone
and reached zero under full disclosure. Tabular disclosure and the exact text
sentence `duration outcome: VALUE minutes.` agreed in every row.

Positive endpoints out of 50 mechanism-seed runs:

| Loss | 0% | 10% | 25% | 50% | 100% | Exceedances |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 50 | 25 | 0 | 0 | 0 | 0 |
| MSE | 50 | 0 | 0 | 0 | 0 | 0 |
| q10 | 50 | 31 | 0 | 0 | 0 | 0 |
| q50 | 50 | 25 | 0 | 0 | 0 | 0 |
| q90 | 50 | 42 | 0 | 0 | 0 | 0 |
| W15 miss | 50 | 46 | 39 | 1 | 0 | 0 |
| W30 miss | 42 | 12 | 5 | 0 | 0 | 0 |
| W60 miss | 0 | 0 | 0 | 0 | 0 | 0 |

The fast collapse is not a defect. The theorem assumes that an exceptional
fraction `epsilon` can carry the maximum loss, so it subtracts the worst-case
gap `epsilon * M_loss`. It is intentionally conservative. A false declaration
of `epsilon=0` at full disclosure returned an invalid positive bound in 1,400
stress cases, demonstrating that the sufficiency assumption is load-bearing.

Source: [localized refinement sensitivity](../../research/peras_v5/ood_audit_20261010/LOCALIZED_REFINEMENT_SENSITIVITY_20261010.md).

### 4.3 Finite-text target interventions

In the 5,000-row intervention battery, the public finite-cell route accepted
4,000 outcome-blind rows with zero gold exceedances and refused all 1,000
target-derived rows:

- 500 literal target-sentence representations: refused;
- 500 unary `X` target-code representations: refused.

Their diagnostic Bayes risk was zero. Refusal is the correct product behavior;
returning a small number with a certification label would normalize leakage.

### 4.4 Exact/partial/proxy leaks in historical v5 research

The preserved v5.7 prototype was tested on exact text leaks, 10%/50% text
leaks, rounded targets, affine targets, noisy proxies, and a hidden unary
channel. It tracked partial disclosure reasonably (`0.543` estimate vs `0.509`
gold at 50%; `0.919` vs `0.904` at 10%) and a noisy target proxy almost exactly
(`0.448` vs `0.447`). But under exact disclosure it plateaued at roughly 6% of
the original risk while the gold was about 1.6%; its lower bound was therefore
formally invalid there.

This historical result is **superseded as certification evidence**. It is kept
because it motivated the current fail-closed intervention auditor and exact
localized-sensitivity route.

The most instructive negative control encoded `round(y/5)` as the number of
letter `m` characters in junk text. Without a reader that counted `m`, every
method missed the signal and the old Pythia bound was valid in only 7% of draws.
An outcome-blind audit proposed the count feature and restored the estimate to
0.081 against gold 0.078. The lesson is an identifiability limit: a bound over
raw information cannot be guaranteed when the registered representation does
not consume an available channel.

Source: [v5.7 historical results](../../research/peras_v5/RESULTS_v5b.md).

## 5. Text shuffling, dropout, masking, and duplication

### 5.1 Mayo fixed-cohort perturbation battery

The diagnostic used the first 256 chronological earliest-per-patient cases;
216 had nonempty strictly recut note bundles. A fixed 64-coordinate word/
unigram/bigram count hash was recomputed without outcomes for every perturbed
representation. `L=300`, `[0,900]`, and the alpha allocation were illustrative,
not clinically audited.

| Note control | Changed rows | q90 pinball lower endpoint | W30 miss lower endpoint |
|---|---:|---:|---:|
| Intact | 0 | 1.6756 | 0.2574 |
| 50% case dropout | 104 | 1.3774 | 0.2574 |
| 100% case dropout | 216 | 1.7813 | 0.2261 |
| Character shuffle | 216 | 1.4682 | 0.2574 |
| Token shuffle | 216 | 2.0465 | 0.2418 |
| 50% span mask | 216 | 1.2375 | 0.2457 |
| 50% segment dropout | 214 | 1.6496 | 0.2418 |
| 50% segment duplication | 216 | 1.7942 | 0.2183 |

The bound is intentionally **not monotone** in generic corruption. Token
shuffle raised q90; full dropout raised q90 while lowering W30; partial dropout
left W30 unchanged to four decimals. Altering text changed the hash geometry,
pairing, group membership, and interpretation of a fixed numeric modulus. No
information-theory theorem says this particular finite-sample statistic must
fall under every edit.

Source: [package and augmentation audit](../status/PACKAGE_AND_AUGMENTATION_GUIDE_20260914.md).

### 5.2 Exact invariances of declared parsers

The current representation tests avoid overinterpreting raw strings:

- bag-token shuffle: exact bound and gold equality in 500/500 synthetic cases;
- Mayo registered bag profile: token-order shuffle exactly invariant;
- appended target-proportional pure-X text: exactly invariant after the
  registered parser discarded pure-X noise;
- exposing the X count as a feature: refused as target-derived;
- character-shuffled exact-duration grammar: parser matched zero times, but
  the report explicitly does **not** call the raw text information-free.

Thus the object certified is the declared parsed representation, not every
possible future NLP interpretation of the bytes.

### 5.3 Older synthetic augmentation study

The historical RTX law-space study used three designs and three seeds:

| Augmentation | Gold ratio | RTX estimate ratio | kNN ratio |
|---|---:|---:|---:|
| Target in 50% of notes | 0.50 | 0.60--0.63 | 0.97--0.98 |
| Target in 100% of notes | 0.00 | 0.05 | 0.73--0.74 |
| Three exact copies of every column | 1.00 | 1.00 | 0.75--0.79 |
| 50 pure-noise columns | 1.00 | 1.00--1.02 | 1.24--1.31 |
| Character shuffle | rises only where text matters | 1.00 / 1.36 / 1.08 | 1.01 / 1.11 / 1.05 |

This is useful estimator behavior, but RTX used externally fitted nuisance
models and its lower endpoint was empirically calibrated rather than covered by
the current finite-cell theorem. It must not be merged with current proof-tier
claims. Source: [RTX results](../../research/rtx_law_space/RESULTS.md).

## 6. Feature engineering and representation geometry

### 6.1 No-new-information controls

Duplicating the Mayo scheduled-duration coordinate twenty times changed an
exploratory paired endpoint from 22.2148 to 20.2881 under the same numeric
`L=300`. The true information set is unchanged, but the normalized metric,
nearest neighbors, and meaning of `L` changed. Therefore the bound is agnostic
to the internals of a downstream predictor, but **not** to arbitrary changes in
the representation used to construct its witness.

In the synthetic intervention battery, retaining an independent identifier
created a median 706 cells and 45.2% singletons. Gold stayed unchanged, while
median endpoint/gold fell from 0.525 to zero and an external split-median
estimate's relative error worsened from 2.1% to 39.4%. Canonically removing the
nuisance restored the baseline exactly.

### 6.2 Equality-aware repair

The finite representation lattice combines deterministic garbling order with
externally attested equal-risk relations. Across 500 fresh runs:

- 0/2,500 raw and 0/2,500 closed endpoint exceedances;
- 0/500 information-order failures;
- nuisance fragmentation repaired in 500/500;
- raw nuisance median endpoint/gold 0.5197, closed 0.7325, exactly matching the
  canonical note-neighborhood node.

The equality-class point-estimator router then selected the least-fragmented
member using registry metadata only. On another 500 fresh runs, median absolute
error fell from 0.13446 to 0.01303, with 496/500 wins and all ten mechanism
medians improved. This is strong confirmation for **attested equality classes**,
not a universal estimator-SOTA result.

### 6.3 Garbling and misalignment

The intervention battery also checked:

- five-family garbling: Bayes risk never decreased;
- within-family reassignment: exact equality to family-only gold;
- global note misalignment: risk no smaller than family-only gold;
- bijective cipher: exact endpoint and gold equality.

Larger endpoint/gold ratios after garbling do not imply more information. The
gold denominator is also larger and coarse cells provide more repetitions per
action.

## 7. Sampling dependence and repeated patients

### 7.1 Synthetic repeated-row falsification

One center-heavy residual was copied to 100 rows per patient. Treating patients
as independent clusters versus treating rows as iid gave:

| Route | Valid draws | Median endpoint |
|---|---:|---:|
| Patient-cluster Pythia | 1,000/1,000 | 0.0840 |
| Invalid row-IID shortcut | 638/1,000 | 0.3684 |

Row duplication manufactured a much tighter-looking but invalid endpoint. A
separate latent cluster-shock study also gave 500/500 validity for the block
route while the pair-independence shortcut covered only 93.4%.

### 7.2 Mayo repeated-patient diagnostic

The matched legacy cohort retained 11,356 cases from 9,300 patients (2,056
repeat cases; maximum cluster size six; count-effective sample 7,835.6).

| Representation | Range-based cluster | Support-free block | Invalid row-IID |
|---|---:|---:|---:|
| Global | 68.005 | **77.398** | 70.618 |
| Procedure family | 46.349 | **51.765** | 48.961 |
| Procedure | 30.010 | **30.094** | 32.622 |
| Procedure + note metadata | **29.103** | 25.398 | 31.716 |
| Procedure + hashed bag | **28.403** | 23.876 | 31.016 |

The row-IID column is 2.613 minutes tighter only because it falsely counts
repeated cases as independent. Support-free blocks win at coarse nodes but lose
under fine note-cell fragmentation. The two valid families cannot be selected
post hoc without a joint alpha ledger. All values are exploratory because
patient-cluster independence, temporal stability, and support are not externally
attested.

Source: [Mayo clustered family](MAYO_CLUSTERED_FAMILY_20261014.md).

## 8. Ten-design and OOD simulation history

### 8.1 First ten-generator replicate audit

Linear, sine, heteroscedastic, mixture, five-dimensional nonlinear, bounded
heavy-tail, quadratic, interaction, sparse-10D, and bimodal mechanisms were
run at 1,000/4,000/10,000 entities, MAE and MSE, 20 repetitions: 1,200 results.

| Method | Valid | Median tightness |
|---|---:|---:|
| Global pair empirical Bernstein | 1,200/1,200 | 0.597 |
| Same pairs, Hoeffding | 1,200/1,200 | 0.285 |
| Blocked pair empirical Bernstein | 1,200/1,200 | 0 |
| Half true modulus (invalid control) | 1,153/1,200 | 0 |
| No modulus correction (invalid control) | 865/1,200 | 0.385 |

The sparse-10D endpoint was zero. A prespecified signal-coordinate metric
recovered median MAE endpoint 0.1496 against truth 0.25 and MSE endpoint 0.0421
against truth 0.0833, with all 75 runs per loss valid. This diagnosed metric
dilution; it did not demonstrate that Pythia can discover the signal coordinate
from evaluation outcomes.

For point estimation, external leave-one-out kNN beat the pair center on MAE
in 190/200 runs (median error 0.0121 vs 0.0886). The pair center is therefore a
bound plug-in, not a competitive universal MAE floor estimator.

### 8.2 Second ten-design MAE challenge

On 200 untouched 512-entity draws, the selection-safe pair/group envelope was
valid 200/200. It beat paired-only at the same total confidence in 175 draws,
lost 5, and tied 20 (`p` about `1e-45` within that suite). Combined median
tightness was 48.6%, versus 41.8% paired. The sparse-12D design was zero in all
20 cases until an oracle predeclared projection was supplied.

The estimator did not win: at 512 entities, group jackknife median absolute
error was 0.0234 versus kNN 0.0110. At 192 entities, LP jackknife 0.0157 versus
kNN 0.0222 yielded 99 vs 101 draw wins—no significant superiority.

### 8.3 Frozen v5.7 OOD falsification

A later ten-mechanism audit deliberately froze the older v5.7 estimator and
bound. Point estimation was promising for q90 (9/10 strict mechanism wins) and
MAE (8/10, sign `p=0.0547`), but the lower bound failed:

| Loss | Bound above known truth / 30 |
|---|---:|
| MSE | 7 |
| MAE | 5 |
| q90 | 1 |
| W30 | 4 |
| W60 | 0, but 7 nonpositive |

This falsified universal certification for the SIMEX/local-fit prototype and
directly motivated declared moduli, finite-cell SAA, score domination, and
fail-closed provenance. These failures are not failures of the present
dominated-SAA theorem.

Source: [OOD audit](../../research/peras_v5/ood_audit_20261010/OOD_AUDIT_20261010.md).

## 9. Multi-loss behavior

### 9.1 Pair/group q10, q90, and window validation

The fresh ten-design run contained five numeric and five deterministic
text-template designs, 20 draws/design, 512 entities, and five losses (1,000
loss cells). All methods were valid on all 1,000 cells.

| Bound | Median tightness on 800 positive-floor cells |
|---|---:|
| Adaptive pair/group | 0.1581 |
| Frozen loss-only route | **0.1843** |
| Paired Hoeffding | 0.0000 |
| Paired empirical Bernstein | 0.0717 |
| Paired Bentkus | 0.1818 |

Adaptive Pythia beat paired Bentkus in only 237/800 and lost 543/800. The frozen
loss-only route won 304, lost 76, tied 420; it tied paired Bentkus for q10/q90,
while improving window losses. Exact Lipschitz group optimization was valid in
250/250 cells and improved its radial group witness in 132, but still lost to
paired Bentkus in 134 cells and beat it in only 59.

This is the key negative result behind the current loss-routed architecture:
group geometry was not automatically stronger than sharp paired concentration.

Point estimation was also decisively weaker than external fitted estimators:

| Loss | Pythia witness-center error | kNN | Histogram | Kernel |
|---|---:|---:|---:|---:|
| q10 | 1.9729 | 0.1905 | 0.2726 | **0.1389** |
| q90 | 1.9643 | 0.1963 | 0.2419 | **0.1723** |
| W15 | 0.2857 | **0.0156** | 0.0192 | 0.0176 |
| W30 | 0.1242 | 0.0192 | 0.0191 | **0.0173** |

### 9.2 Later finite-group routing

A finite-group pinball cap and outcome-blind geometry gate improved the
quantile witness. On a later untouched five-loss seed, the route was valid in
500/500 cells and, against the numerical maximum of three unadjusted paired
comparators, won 339, tied 140, and lost 21. Median positive-floor tightness
was 0.2195 versus 0.1791; nine designs had higher medians and sparse-4D tied.
Because the comparator maximum was not itself multiplicity-adjusted and W60
was zero throughout, this is useful engineering evidence rather than a
universal literature-SOTA claim.

### 9.3 Current dominated multi-loss route

The newer fixed-action dominated-SAA confirmation is stronger and simpler for
finite representations: q90 was valid 500/500 with median 0.8473 of gold and
window miss was valid 500/500 with median 0.9341. It strictly beat registered
support-free block SAA in 495/500 and 500/500 respectively. This improvement
does not transfer automatically to arbitrary continuous raw representations;
it relies on frozen finite cells and clustered sampling.

Source: [multi-loss audit](MULTILOSS_VALIDATION_20260913.md).

## 10. Other structural simulations

### Declared modulus and dKelly concentration

With a DGP-supplied barycentric discrepancy, the frozen dKelly route improved
matched Bentkus on untouched confirmation seeds:

| Loss | Bentkus median bound/gold | dKelly | Strict wins/ties | Exceedances |
|---|---:|---:|---:|---:|
| MAE | 0.3849 | **0.5593** | 50/0 | 0 |
| MSE | 0.3998 | **0.6327** | 50/0 | 0 |
| q10 | 0.0087 | **0.1475** | 32/18 | 0 |
| q50 | 0.3849 | **0.5593** | 50/0 | 0 |
| q90 | 0.0345 | **0.1900** | 36/14 | 0 |

The discrepancy is declared, not identified from one observation per unit.
No Mayo modulus is currently justified.

### Repeated-environment modulus

Global-range calibration had 0/1,000 misses but was loose, especially for MSE
(median cap/truth 47.0). Hard support geometry reduced fresh-confirmation ratios
to 1.091 absolute and 2.009 squared, again with 0/1,000 misses. These tests show
that tighter bounds can come from stronger auditable resources rather than
post-hoc numeric tuning.

### Refinement signal test

A model-free permutation statistic detected a known informative concept in
500/500 runs. An independent 1,000-seed null audit rejected 56 times at nominal
5% (Wilson 4.34%--7.20%). The test establishes conditional signal under
exchangeability; it does not itself lower-bound the magnitude of information
value.

## 11. Mayo results: final interpretation

### 11.1 Frozen historical guarded MAE anchors

These remain unchanged and refer to the original guarded Arm-B construction:

| Quantity | Minutes | Tier |
|---|---:|---|
| Best historical OOF MAE | 32.8700117033 | achieved retrospective loss |
| Global attainable-floor estimate | 31.1186421201 | estimate |
| Stratified calibrated lower bound | 28.8495996043 | `calibrated` |
| Conservative Q-cap rung | 24.1867259 | conditional on its guard |

The 28.85 result is not upgraded to proof tier by newer simulations.

### 11.2 Verified Foundation model

The separate clinical-notes policy is reproducible at MAE
**31.556418073067334** on 2,839 later cases after 10,882 training cases, with
zero recorded patient overlap and strict `note_date < scheduled_in_room - 2
days`. It is a strong retrospective held-out result, not a new prospective
evaluation. The unrelated 31.2623 number was not validated as held-out final
duration MAE.

The first fixed-policy information-value bridge returned zero beyond both
procedure family and procedure because coarse lower endpoints did not exceed
the policy upper endpoint 33.8533. The newer dominated-SAA route on the same
held-out losses gave an exploratory joint-ledger value lower endpoint of
10.760 minutes beyond an 8-group procedure-family map, but still zero beyond
procedure. This later result is not confirmatory: the holdout has been viewed,
and the family map/cohort contracts remain retrospective.

### 11.3 Current finite-cell Mayo lattice

On a newly reconciled all-row cohort of 14,195 cases / 11,447 patients, the
dominated route produced:

| Representation | MAE | q90 pinball | W30 miss |
|---|---:|---:|---:|
| Global | 70.05 | 18.12 | 0.676 |
| Procedure family | 51.95 | 13.70 | 0.551 |
| Procedure | 37.45 | 9.93 | 0.408 |
| Procedure + note metadata | 36.71 | 9.64 | 0.395 |
| Procedure + hashed bag | 36.14 | 9.36 | 0.384 |

These are exploratory lower endpoints, not estimates of achieved model error.
They no longer collapse under fine note cells, but the decreasing empirical
SAA and endpoints alone do not prove that notes add information. Cohort
reconciliation with the earlier 11,356-case study remains open.

Source: [Mayo dominated SAA](MAYO_DOMINATED_20261014.md).

### 11.4 Mayo note-value diagnostics

On 9,300 earliest unique patients, note metadata beyond procedure reduced the
empirical SAA score by 0.9715 minutes versus a within-procedure permutation
99th percentile of 0.8882 (exploratory two-test tail 0.004). A hashed lexical
bag added 0.6057 beyond metadata with adjusted tail 0.744. Thus metadata had an
unusual retrospective signal, while the generic lexical sketch did not show
incremental signal. Neither result is a certificate of deployable note value.

### 11.5 Exploratory multi-loss Mayo geometry

Under illustrative `L=300` and `[0,900]` support, the later geometry route gave
tabular q10/q90 endpoints 2.9436/5.2664 loss-minutes and W15/W30/W60 miss-risk
endpoints 0.4220/0.2678/0.1049. W30 corresponds to an illustrative ceiling of
73.22% attainable within-30-minute accuracy, versus 62.60% achieved by the
frozen OOF model. These numbers were used during route development and lack
audited smoothness/support/timing assumptions; they remain exploratory.

## 12. What is established, falsified, and still unknown

### Established under declared contracts

- The current finite-cell and clustered routes can be valid without training a
  predictor.
- Patient clustering is necessary when cases repeat within patients.
- Target-derived representations can and should fail closed.
- Exact representation equivalences and deterministic information order can be
  used to repair fragmentation without inspecting evaluation outcomes.
- Cell-local support, fixed-policy bridges, score domination, curvature, and
  repeated environments can materially tighten bounds when their premises are
  genuinely available.
- MAE, pinball, and window-miss losses are supported by current finite-cell
  routes; MSE has conditional modulus/replicate routes but is more tail/range
  sensitive.

### Falsified

- The witness center is not a universally competitive point estimator.
- The bound is not invariant to arbitrary feature engineering.
- Generic note corruption does not force a monotone endpoint response.
- A parser failing to read shuffled text does not prove raw text lacks target
  information.
- Old v5.7 SIMEX bounds were not universally valid OOD.
- Adaptive grouping did not beat paired Bentkus across the original multi-loss
  suite.
- A positive endpoint in sparse/high-dimensional geometry cannot be guaranteed
  without repetition or stronger registered structure.

### Still unknown

- No simulation suite establishes universal superiority over every published
  Bayes-floor method.
- No train-free point estimator is uniformly best across losses and designs.
- Current Mayo data do not identify the raw-note Bayes floor for every possible
  future NLP representation.
- No current Mayo bound has prospective sampling confirmation.
- The full literature novelty claim still needs an independent theorem and
  comparator audit.

## 13. Reproduction map

The exact command depends on the route; the detailed registration beside each
result is authoritative. Core public checks include:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python scripts/run_mayo_text_augmentation.py --n-entities 256
.venv/bin/python scripts/run_mayo_oracle_leakage.py
python3 scripts/run_replicate_adversarial.py --repetitions 20 --output /tmp/pythia_replicate_ten_designs.json
python3 scripts/run_sparse_metric_audit.py --repetitions 25
```

Important frozen raw hashes:

| Artifact | SHA-256 |
|---|---|
| Intervention battery | `d538d20f2eaea40e499f700811eaa6e2246bde948ca80a9e3de33987e8c72e57` |
| Cell-local confirmation | `dbb5f4f3162d9d813273d603987d8720f3fec313652e9e7d673a3bfa30b7a20e` |
| Family lattice | `4756b94120fbaf61549ece1da8e2c893d5be9f8755798368eb4480ed6c05ddce` |
| Equality estimator | `6b030646aa42786cc8a79cb0794f7a7028b846b2ffd8767ba82ace1485755dd6` |
| Fixed-policy value | `f920b16e0adf28a2862b2213f238693e591c790fd5632a8281114af9ffe20149` |
| Clustered finite-cell | `6ec0025ece98962d68a5e5de8add74a1e414502b54accd8559d69d1326ebbb16` |
| Support-free cluster block | `4c0c0307305fa34b750e0a5c5a4d97c4b03d59ce17abd3510149e0ca11197c8b` |
| Dominated SAA | `cd96eada6ff5d8fe1bde199dccd0d3ba91abee36dd51019635a64844f0d73004` |
| Curvature confirmation | `560a5af465bf9869a738c565f138a647aaefffa1a930474baf8f365dafddb010` |
| Action-dispersion confirmation | `870d3275a5b34e160cd1c6cb6df858bebb001326e8ce3a66f1adc0b41ca2c431` |
| Mayo clustered aggregate | `ba744408fbe7a7fba23f1364cd7d61e54161f88d8aec9131c586db6f966dd0fc` |
| Mayo family-lattice aggregate | `550bcda8f2fd3b5609bc97ab82acdb66aeb7025ef4758fd9bd4ef0838e0dbeef` |

No patient-level values, note text, patient identifiers, or row-level Mayo
results are stored in this documentation.

## 14. Complete study ledger

This ledger prevents an older development result from being quoted without its
later audit. “Current” means usable under the stated contract; it does not mean
prospectively validated on Mayo.

| Study family | Scale | Principal finding | Present status |
|---|---:|---|---|
| Original guarded Mayo MAE | legacy 50-design guard + Mayo OOF | 28.8496 stratified calibrated endpoint; 24.1867 Q-cap rung | Preserved historical anchor; calibrated/conditional, not upgraded |
| Independent guarded-MAE gate | 500 designs | 491/491 in-domain valid; 9 out-of-domain refused; Wilson lower 0.9922 | Current guard regression |
| Squared-loss dependence profile | 24 latent common-bias designs | Feasible dependence/risk interval and collapse point respond to shared bias | Theorem-candidate/exploratory; no MAE proof transfer |
| Range dependence guard | 1,800 patient simulations | 1,800 valid; 1,797 tighter than Hoeffding | Current conditional route |
| Near-replicate conditional route | 3,600 simulations | Corrected 3,600 valid; uncorrected only 78.7% | Current conditional route |
| Fixed-design global pairs | 1,500 adversarial simulations | 1,500 valid; median 0.624; 1,356 blocked-pair wins | Current conditional route |
| Cluster-shock falsification | 500 runs | Cluster route 500 valid; pair-IID shortcut 93.4% | Current negative control |
| First ten-generator audit | 1,200 loss-runs | 1,200 valid; 957 EB-over-Hoeffding wins; sparse failure | Current diagnostic; not estimator SOTA |
| Sparse prespecified metric | 75 datasets/loss | Positive recovered endpoint, all valid | Oracle-support control only |
| External kNN estimator check | 200 MAE/MSE runs | kNN won MAE 190/200; Pythia won MSE 131/200 | Falsifies universal estimator claim |
| Second MAE challenge | 200 draws at 512 plus 192 entities | Pair/group valid; strong paired-bound win; estimator tie/loss | Current geometry evidence |
| Multi-loss main audit | 1,000 cells | All valid; paired Bentkus beat adaptive route on most pinball cells | Current negative/route-selection evidence |
| Exact group LP pilot | 250 cells | 250 valid; improved radial, lost to pair in 134 | Exploratory numerical route |
| Selection-safe window route | two fresh 500-cell suites | All valid; large W15 and smaller W30 gains | Current opt-in geometry evidence |
| Finite-group geometry route | 500 fresh cells | 339 wins/140 ties/21 losses vs unadjusted classic max | Current heuristic route; not universal SOTA |
| RTX law-space confirmation | 200 draws × 6 losses | Empirically 1,200/1,200 calibrated endpoints; strong estimators except W30 | Historical calibrated track; uses external fitted nuisances |
| Frozen v5.7 OOD audit | 10 new mechanisms × 3 seeds | Multiple bound violations, including 5/30 MAE | Superseded/falsified certificate |
| Synthetic leak battery v5.7 | 3 designs × 4 seeds × 9 variants | Partial leaks tracked; exact-leak lower floor invalid; hidden channel broke all methods | Historical falsification |
| Refinement-gap audit | 100 seeds × 5 disclosures × 3 losses | 1,500 corrected cases, zero exceedances; full disclosure zero | Current conditional sensitivity |
| Localized refinement audit | 10 mechanisms × 5 seeds × 5 disclosures × 8 losses | 2,000 valid, monotone, zero at complete disclosure | Current conditional sensitivity |
| Declared modulus confirmation | 50 fresh rows/loss | dKelly improved Bentkus; zero exceedances | Current conditional sensitivity |
| Repeated modulus calibration | 1,000 rows/stage | Valid; hard supports tightened absolute to 1.091 and squared to 2.009 cap/truth | Current repeated-environment route |
| Text intervention battery | 5,000 rows | 4,000 valid; 1,000 target rows refused; nuisance fragmentation exposed | Current representation regression |
| Cell-local range confirmation | 4,000 rows | Zero exceedances; all baseline comparisons improved | Current conditional route |
| Representation lattice | 500 lattices / 2,500 nodes | Zero exceedances/order failures; equality repaired 500 | Current conditional route |
| Equality-class estimator | 500 fresh runs | Error 0.1345 to 0.0130; 496 wins | Current, only for externally attested equality |
| Refinement signal/null | 500 power + 1,000 null | 500/500 power; 5.6% null rejection | Current test under exchangeability |
| Fixed-policy value | 500 runs | All valid/positive/wins; median 0.6429 | Current conditional magnitude bridge |
| Curvature correction | 400 rows | Zero exceedances; large MAE/q90 gains | Current conditional route |
| Action-dispersion calibration | 1,000 rows | All four targets covered in every row | Current repeated-environment bridge |
| Clustered finite-cell | 1,000 laws + 1,000 dependence stress | All cluster endpoints valid; row-IID stress 638 valid | Current cluster route |
| Support-free block SAA | 500 clustered unbounded-law draws | All valid/positive/wins; median 0.7865 | Current support-free route |
| Fixed-action dominated SAA | 500 × 3 losses + 2,400 stress | All primary valid; zero stress violations; median 0.900 MAE | Current strongest finite-cell route |
| Mayo note perturbations | 8 controls on 256-case cohort | Non-monotone q90/W30 responses | Exploratory falsification only |
| Mayo target injection | 2,048 and 4,096 patients | Mixed MAE collapses with disclosure; full endpoint zero; API refuses | Exploratory oracle falsification |
| Mayo unique-patient lattice | 9,300 patients | Metadata tail 0.004; lexical increment 0.744; exact parser invariances | Exploratory |
| Mayo repeated-patient lattice | 11,356 cases / 9,300 patients | Cluster correction wider; route tradeoff under fragmentation | Exploratory |
| Mayo dominated lattice | 14,195 cases / 11,447 patients | Stable finite-cell endpoints across MAE/q90/W30 | Exploratory; cohort reconciliation open |
| Foundation policy verification | 10,882 train / 2,839 later cases | Held-out MAE 31.556418, zero recorded patient overlap | Verified retrospective achieved risk |

## 15. One-sentence final reading

The final Pythia bounds behave logically **relative to a declared representation
and sampling contract**: they collapse or refuse under target revelation,
remain invariant under registered equivalences, weaken under irrelevant
fragmentation, widen under patient dependence, improve when honest local
support or repeated-environment resources are supplied, and refuse to infer
that arbitrary raw text has no information merely because one parser or model
cannot read it.
