# Modern research gap map for Pythia

## The estimand must come first

For information set `X` and loss `ell`, Pythia targets

```
R*(X,ell) = E[ inf_a E[ell(Y,a) | X] ],
```

the Bayes/attainable risk over every measurable decision rule using exactly the registered information. This is not the test risk of a fitted predictor, a PAC/PAC-Bayes generalization gap, conformal coverage, model disagreement, or a minimax risk over a parametric family. Many superficially similar papers answer one of those other questions.

## Research landscape

| Line of work | Primary object | What it already establishes | Remaining distinction for Pythia |
| --- | --- | --- | --- |
| Blackwell--Sherman--Stein and Le Cam comparison of experiments | ordering/deficiency of information structures | more information cannot worsen Bayes risk; approximate experiment comparison controls decision risk | does not supply a tight finite-sample certificate from one heterogeneous observational dataset |
| [Reid & Williamson, JMLR 2011](https://www.jmlr.org/papers/volume12/reid11a/reid11a.pdf) | information, divergence, risk for binary experiments | deep equivalences between Bayes risk, statistical information, proper losses, and divergences | Pythia must not claim that connecting risk and information is new |
| [Williamson & Cranko, JMLR 2024](https://jmlr.org/papers/v25/22-0988.html) | information--risk bridge for multiple distributions | modern general geometry joining experiment information and Bayes risk | raises the novelty bar: Pythia needs a finite-data certification mechanism, not a renamed information functional |
| [Chen, Guntuboyina & Zhang, JMLR 2016](https://www.jmlr.org/beta/papers/v17/16-185.html) | theoretical Bayes-risk lower bounds via f-informativity | general loss/prior inequalities and classical minimax consequences | usually needs a specified statistical model/prior or informativity bound; not an automatic arbitrary-dataset floor |
| [Anderson & Bjarnadottir, PLOS ONE 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296904) | attainable MAE heuristic | constrained omniscient-model estimate using similarity/linking assumptions | nearest direct comparator; its metric/link choices and lack of a universal finite-sample coverage theorem differ from Pythia |
| [Origins of unpredictability, PNAS 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11181083/) | decomposition of prediction error | separates irreducible and learning error conceptually and empirically | does not itself solve finite-sample lower certification for arbitrary continuous/text information |
| [Conformal Risk Control, ICLR 2024](https://research.google/pubs/conformal-risk-control/) | risk control for a calibrated predictive procedure | finite-sample control of monotone expected losses | controls the output/procedure's risk, not the best achievable risk over all future predictors |
| [Bound to Disagree, UAI 2026](https://proceedings.mlr.press/v337/bazinet26a.html) | target-model generalization through a certifiable surrogate | disagreement-based risk transfer without modifying the target model | bounds a given predictor, not the representation's Bayes floor |
| [Model Agreement via Anchoring, COLT 2026](https://proceedings.mlr.press/v336/eaton26a.html) | independent-training model disagreement | algorithm-specific stability/disagreement bounds | does not identify irreducible outcome risk; shared error remains possible |
| [Bayes-Sufficient Representations, 2026 preprint](https://arxiv.org/abs/2606.04045) | loss-dependent representation sufficiency/minimality | formalizes the Bayes quotient and loss-specific sufficient representations | directly relevant; Pythia must certify or assume sufficiency rather than declare a reader universally adequate |
| [Shah & Peters, Annals of Statistics 2020](https://doi.org/10.1214/19-AOS1857) | hardness of conditional-independence testing | unrestricted conditional independence cannot be given a uniformly powerful valid test | blocks the idea that residual testing can generally prove `Y independent of raw X given representation` |
| Huber contamination and robust mean/risk inference | stability to corrupted/heavy-tailed samples | minimax estimation under explicit contamination/moment classes | different from representation-refinement uncertainty; useful concentration tools but not a sufficiency solution |
| [Certify or Refuse, 2026 preprint](https://arxiv.org/html/2608.10893v1) | selective risk control under covariate shift | fail-closed selective certification with coverage floors | different estimand, but close framing; Pythia must distinguish attainable-risk certification and document priority carefully |

## Gaps that are actually open for this project

### 1. Representation-to-raw-information transfer

The decisive unresolved problem is an upper bound on

```
R*(registered representation) - R*(all allowed raw information).
```

Without a structural class, repeated environments, randomized feature acquisition, or an external discrepancy/information cap, this gap is not identifiable. A weak NLP model, a failed conditional-independence test, or character shuffling does not close it.

Current Pythia progress:

- exact finite groups: direct full or partial-cell SAA certificate;
- finite pre-outcome refinement: conditional-information sensitivity;
- localized failure: exceptional-mass sensitivity;
- arbitrary raw notes: refuse unless one of those bridges has provenance.

Finite decision-information follow-up (2026-10-12): for fully declared finite
outcome, action, coarse, and refinement alphabets, Pythia now produces a
statistical loss-relative gap edge.  One multinomial L1 event gives
`TV(P,P_hat)<=delta`; each Bayes-risk functional is loss-range-Lipschitz, hence
`Delta_loss(P)<=Delta_loss(P_hat)+2*A*delta`.  On untouched ten-law confirmation
this direct cap beat the matched CMI/Pinsker cap in 50/50 runs and 10/10
mechanisms at each of `n=3k,30k,300k`, with zero known-gap exceedances.  A
second untouched `n=3k` battery produced 50/50 wins for each of MAE, MSE, Q10,
Q50, Q90, and exact-window miss, again with zero exceedances.  This is a useful
bridge for small registered concept ladders, not raw text: the full-alphabet
radius saturates for high-cardinality outcomes/features, and outcome binning
cannot certify information absent from the bins.

Outcome-alphabet-free follow-up: uniform one-sided concentration over all
registered coarse and refined finite policies replaces the joint multinomial
event.  Its complexity is `|G| log|A|` and `|G||Z| log|A|`, so continuous
outcomes are allowed; the finite action class must contain every relevant
Bayes action.  On new untouched seeds it beat the joint-L1 direct cap in 50/50
runs and 10/10 mechanisms for each of MAE, MSE, Q10, Q50, Q90, and exact-window
miss, with zero known-gap exceedances.  This narrows the high-cardinality gap
but remains range-limited for Mayo's 0--900 minute action domain.

Variance-adaptive follow-up: simultaneous empirical-Bernstein rectangles for
every cell/action coordinate were weaker for MAE, MSE, Q50, and window miss but
stronger for extreme pinball losses.  A router frozen on development evidence
and confirmed on new seeds uses coordinate Bernstein only for Q10/Q90.  It had
zero misses in 300 confirmation cells, beat policy Hoeffding in all 50 Q10 and
50 Q90 runs, and refuses untested quantile levels.  This supports loss-specific
concentration routing rather than one claimed-universal inequality.

### 2. Near-replicate heterogeneity

Exact conditional-law cells are powerful but uncommon. Continuous covariates create near rather than exact repeats. A useful theorem needs a valid upper cap on average conditional-law displacement—not merely geometric closeness in standardized feature space. Wasserstein/Lipschitz transfer is mathematically straightforward; obtaining a non-oracle cap is the research problem.

Promising designs:

- repeated environments or time blocks with independently observed conditional laws;
- randomized acquisition or masking of information;
- predeclared smoothness/transport classes with falsification on held-out environments;
- partial exact-cell contributions plus zero/transport-corrected residual mass.

Implemented follow-up: `coupled_environment_w1.py` turns independently repeated **joint** outcome environments into an outcome-only upper confidence cap on average within-block W1. A common-quantile synthetic design produced positive MAE certificates in 46/50 fresh near-replicate runs with zero known-gold exceedances; independent product coupling produced 0/50. This closes the oracle-cap gap only when the repeated-environment marginal/coupling contract is real. It is not available from ordinary Mayo rows and did not recover Q90 or MSE.

Stronger implemented follow-up: `repeated_environment_saa.py` uses the repeated coordinates directly. The blockwise empirical optimal risk has expectation no greater than the fixed-design Bayes risk (`E inf <= inf E`), so bounded-mean concentration gives a direct lower certificate without a transport intermediary. Across ten fresh mechanisms and five seeds, product-coupled environments yielded positive bounds in 50/50 runs for each of MAE, MSE, Q10, Q50, and Q90; median bound/known-floor ratios ranged from 0.794 to 0.859 with zero observed exceedances. W15 and W30 were positive in 50/50, while W60 was positive in 40/50. Fresh two-sided audits had zero lower misses, upper misses, or contradictions. This is a strong theorem-backed result for a strong repeated-measurement design, not a route available in ordinary Mayo data.

Loss-specific MSE follow-up: because squared-loss Bayes risk is conditional variance, the Bessel-corrected within-unit variance is an unbiased repeated-environment witness. A fresh ten-mechanism audit improved median bound/known-floor from 0.8204 to 0.9701, won all 10/10 mechanisms and 50/50 runs against the former empirical-optimum witness, and had zero known-floor exceedances. The selection-safe two-witness router retained 0.9686 after multiplicity accounting. This closes most repeated-environment MSE bias but does not repair the unrelated one-outcome A-fold orthogonality problem.

Dependence-typed follow-up: under a registered product environment, each disjoint block--unit witness is independent, so concentration can use the coordinate sampling unit rather than treating an entire environment block as one observation. An untouched row/coordinate router reached median bound/floor 0.972 (MAE), 0.966 (MSE), 0.943 (Q10), 0.972 (Q50), and 0.921 (Q90), with zero gold exceedances. Fresh window audits reached 0.840 (W15), 0.773 (W30), and 0.497 (W60), beating row-level concentration on all 10/10 mechanisms for each window. This is conditional on a real product-independence design; ordinary Mayo rows do not qualify.

Declared-modulus follow-up (2026-10-12): for ordinary one-draw settings, the barycentric residual-law discrepancy cannot be recovered from the evaluation rows and is now represented explicitly as a declared sensitivity profile `P(k)`/`P2(k)`. The pinball penalty is one-half the barycentric W1 dispersion for every quantile because positive and negative deviations cancel inside each block. A fixed-time diversified-Kelly lower-mean route, retaining the original `k=(2,4,8,16)` grid, was selected in a factorial audit and confirmed on untouched seeds: median bound/gold improved from 0.385 to 0.559 (MAE), 0.400 to 0.633 (MSE), and 0.035 to 0.190 (Q90), with 0 known-gold exceedances. Extra block sizes and a joint method selector were valid but weaker after multiplicity. These are oracle-discrepancy theorem checks; Mayo still receives a conditional curve, not an identified certificate.

Feature-modulus attachment audit (2026-10-13): the supplied `Lambda G` / `Lambda_2^2 H` construction is accepted as a sufficient-condition adapter to the more general direct-`P(k)` profile. Its supplied rows reproduce the reported 50/50 positive MAE and MSE results after an independently fitted location shift, but the evaluation uses the true synthetic discrepancy and therefore remains an oracle theorem check. The public API now computes `G/H` and compiles a provenance-bearing conditional curve without fitting the shift or estimating `Lambda`. “Feature-only” is reserved for the geometry, not the certificate. See `DECLARED_MODULUS_ATTACHMENT_AUDIT_20261013.md`.

Non-oracle calibration follow-up (2026-10-13): independent repeated environments now produce simultaneous upper endpoints for the absolute and squared barycentric penalties, plus effective aggregate modulus caps. A frozen dKelly/Bentkus loss router had zero analytic-truth misses in 1,000 fresh runs per penalty (Wilson lower validity 0.9962). Median absolute cap/truth improved from 2.808 to 1.996, while squared cost remained very loose at 47.020 because its global range is quadratic. The API records whether exact target replicates were supplied and warns that direct repeated-environment SAA is then a stronger resource-matched comparator. Ordinary Mayo rows do not satisfy this contract. See `REPEATED_MODULUS_CALIBRATION_AUDIT_20261013.md`.

Support-geometry follow-up: convexity places the maximum absolute or squared barycentric cost over a frozen per-unit support box at a vertex. Exact blockwise vertex ranges reduced median cap/truth on another untouched 1,000-run confirmation to 1.091 (absolute) and 2.009 (squared), with zero misses in each family. This largely closes the synthetic MSE range failure when genuine hard supports exist. It is not available from fitted prediction intervals and does not currently tighten Mayo.

Extreme-pinball follow-up: averaging capped scores over overlapping balanced partitions admits an entity-McDiarmid endpoint, but it lost to disjoint dKelly in all 1,000 development runs. A second route instead averages valid partition-level dKelly e-values, never multiplying dependent e-values. On untouched exact-law Q90 data it had zero violations and median bound/gold 0.8213 versus 0.8192 for a matched single partition, winning 8/10 mechanism medians (`p=0.0547` one-sided). This is a valid compositional method but not a significant SOTA win; disjoint dKelly remains the default pending a nonzero-discrepancy, geometry-weighted audit.

The nonzero-discrepancy weighting audit was negative: the minimum-penalty single partition reached median 0.7622, versus 0.7593 for the best soft weighting and 0.7120 for equal weights; weighted routes won only 1/10 mechanism medians. No confirmation was opened. This closes geometry-weighted partition mixing as the immediate Q90 route and preserves the simpler disjoint dKelly default.

### 3. Tight multi-loss concentration

The group witness is strongest when geometry and concentration are matched. Bentkus dominates generic Hoeffding in several pinball/window settings; multi-point SAA helps Q90 under exact laws. MSE upper endpoints remain loose because squared-loss range is large. The open engineering/theory task is a selection-safe router over exact concentration, empirical Bernstein, Bentkus, bounded-moment robust means, and loss-specific witnesses—without tuning on evaluation outcomes.

The repeated-environment audits now show that choosing the valid sampling unit can matter more than choosing among inequalities. Product-coordinate concentration raised W30 tightness from 0.356 to 0.773, while MSE sometimes preferred averaging coordinates before clipping. The next fair concentration comparison should implement bounded-mean betting intervals and one-sided empirical U-statistic intervals as named published routes, then validate them on untouched mechanisms under the dependence assumptions each theorem actually supports.

Original-scale follow-up (2026-10-12): outcome clipping is unnecessary once a
nonnegative original-outcome SAA score exists.  Truncating the *score* at a
predeclared cap preserves a lower target, and dKelly concentration plus an
alpha-accounted cap/block portfolio yields an original-loss certificate with
no moment assumption.  On a fresh ten-noise-law confirmation it reached median
bound/gold 0.924 (MAE), 0.845 (Q90), and 0.862 (MSE), with zero known-gold
exceedances.  A heterogeneous-mean Catoni second-moment endpoint was valid but
lost all supported confirmation comparisons (0.846, 0.516, 0.805); Cantelli was
weaker still.  The production router should therefore prefer bounded-score
dKelly and retain declared-moment routes as explicit fallbacks or diagnostics.
This improves concentration once valid independent SAA scores exist; it does
not solve the one-row near-replicate or representation-sufficiency gap.

### 4. Search and representation selection

Metis may eventually search transformations, concepts, groupings, clips, and witnesses. Every searched option creates selection debt. A valid system needs one of:

- pre-registration on a development split;
- simultaneous confidence sequences/e-processes;
- reusable holdout/privacy-style control;
- a finite alpha ledger over a declared search graph.

The theoretical novelty could lie in compiling a search/action graph into a certificate and refusal trace. Metis currently remains far behind Pythia implementation and should not be described as experimentally validated.

Implemented composition layer (2026-10-12): `pythia.certificate_graph` represents base lower endpoints and richer-representation risk-gap upper endpoints as an acyclic evidence graph. Every stochastic evidence event has a unique alpha-ledger identifier; reuse of a genuinely simultaneous event is charged once, while conflicting declarations, overspending, unsupported tiers, cycles, and missing target paths fail closed. A path telescopes base lower risk minus upper information gaps, and the compiler selects the maximum only after making all candidate paths simultaneous. `pythia.evidence_adapters` adds a content-addressed decision-problem identity, preventing accidental composition across populations, losses, scales, clips, or decision times. A 5,000-repetition three-rung audit with two base witnesses and three gap routes had zero known-target exceedances; an uncorrected six-event registry was refused. A second 1,500-case tabular/text target-disclosure audit had zero truth exceedances and collapsed to zero under complete disclosure, while a false zero-gap control failed in all 300 cases. This closes finite-graph selection/task-identity accounting, not the harder problem of producing a justified raw-note gap edge. Metis search is still unimplemented.

### 5. Dependence, shift, and clinical sampling

Patient repeats, temporal drift, schedule-derived variables, and note provenance violate naive iid assumptions. Cluster concentration/bootstrap addresses sampling uncertainty but not conditional-law stationarity. Pythia needs separate objects for entity dependence, environment shift, shared predictor error, and representation refinement; combining them into one empirical correction obscures which assumption carries the claim.

## Candidate paper contributions, ranked honestly

### Strongest current contribution

**Loss-routed finite-group attainable-risk certification with selection-safe multi-point witnesses, partial sparse-cell recovery, and structured refusal.** This is implemented, theorem-backed under stated conditions, and repeatedly checked on known-law mechanisms.

### Credible narrow novelty lead

**Independent multi-point geometry for skew-sensitive pinball risk.** The Q90 tuple route beat the matched pair/Bentkus route on all ten exact-law mechanisms and survived a jointly accounted selector. It needs broader external baselines and a theorem explaining when `k>2` is optimally tighter.

### Credible systems novelty lead

**A certificate compiler over nested information structures.** Input: estimand, loss, representation graph, provenance, dependence structure, and calibration resources. Output: strongest valid certificate, complete alpha/assumption ledger, sensitivity curves, falsification results, or refusal. The individual inequalities are mostly classical; the contribution would be compositional validity and practical automation.

### Not currently supported

- a new universal definition of information;
- a distribution-free tight bound for arbitrary raw text or feature engineering;
- broad SOTA estimator performance across losses;
- a theorem that failed learners prove no remaining signal;
- confirmatory Mayo evidence;
- a working autonomous Metis/Horos system.

## Development programme

1. **Freeze the estimand registry.** Every result names loss, information set, outcome clip/original scale, target population, and claim tier.
2. **Finish external comparator fidelity.** Run author artifacts where available, especially Anderson--Bjarnadottir, and never compare a point heuristic to a confidence endpoint as if they were identical objects.
3. **Derive the Q90 multi-point regime.** Characterize finite-sample bias and concentration versus pair witnesses by distribution shape, tuple size, and cap.
4. **Finish producer integration and freeze a runnable Mayo manifest.** Generic typed adapters now cover lower-route dictionaries, standalone statistical/declared caps, localized/information sensitivity profiles, and structural-zero edges. Strict content-hashed schema-version-1 manifests now compile graphs and refuse drafts/unresolved evidence. The aggregate-only Mayo draft intentionally refuses while preserving legacy anchors and enumerating missing cohort, OOF, dependence, base-bound, and terminal raw-note-gap evidence. Dedicated transport/dependence adapters and a fully evidenced frozen Mayo manifest remain; Metis search must consume, never bypass, the manifest.
5. **Create a second untouched benchmark registry.** Include discontinuity, sparse interactions, heavy tails, rare informative text concepts, irrelevant high-cardinality features, target leakage, character/token perturbations, and environment shift.
6. **Require one primary predeclared superiority question.** Mechanism is the inference unit; all secondary losses are multiplicity controlled and negative findings remain visible.
7. **Only then run Mayo.** Reproduce cohort/predictions, audit patient/time dependence and representation provenance, and retain retrospective/calibrated labels.

## Go/no-go rule

Continue development: the scoped certificate and sparse-cell/multi-point results justify it. Do not submit a broad universal claim until external comparator fidelity, near-replicate cap provenance, and representation-ladder certification are complete. A narrower paper on exact/partial finite representations and Q90 multi-point geometry may become defensible sooner.
