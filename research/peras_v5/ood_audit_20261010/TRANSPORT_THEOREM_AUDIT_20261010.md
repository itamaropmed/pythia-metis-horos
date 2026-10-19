# Pythia: conditional tuple-bound theorem audit and OOD stress test (2026-10-10)

## Decision

The frozen v5.7 near-replicate *estimator* is a plausible research lead, particularly for MAE and Q90, but its empirical lower-bound correction is not a general certificate: the independent ten-mechanism audit records 5/30 MAE and 7/30 MSE bounds above known Bayes risk. The new transport-corrected construction below restores a one-sided conditional guarantee under explicit assumptions, but is almost always zero at `n=800`. Neither line currently establishes a generally tight, model-agnostic bound. Do not use “state of the art” or an unqualified “certified” claim.

## Estimand and information boundary

Freeze an audited representation `X=(tabular columns, outcome-blind note readings)`, its `n` observed feature rows, and a loss `ell`. The finite-design target is

`R*_X = n^{-1} sum_i inf_a E[ell(Y_i,a) | X_i]`.

This is not the risk floor conditional on *all information in the raw text* unless the reader is sufficient. It is not a guarantee for every representation a future agent could invent. A new feature that reveals the target changes the conditional law and the estimand. A positive finite-sample lower bound without regularity, exact replicates, or independently verified distribution-shift caps is unidentifiable: the unobserved conditional laws can be changed at observed feature points while keeping the observed sample possible.

## Conditional tuple inequality

Let `T_t=(i_t,j_t2,...,j_tk)` be `m` outcome-blind tuples, each with distinct entity indices. Conditional on all feature rows, assume the `Y_i` are independent. Let `C` clip to `[lo,hi]`, width `D`, and assume externally justified constants `w_tj >= W1(P_i_t,P_j)` for the *original* conditional outcome laws. For loss `ell` on clipped responses, let `L` be a uniform Lipschitz constant in the response argument, and let `r_t = inf_a k^{-1} sum_j ell(C(Y_j),a)`. Define `S=m^{-1} sum_t r_t`, `c_i` the number of appearances of entity `i`, and

`A = (L/(mk)) sum_tj w_tj`,

`H_alpha = (L D/(mk)) sqrt((log(1/alpha)/2) sum_i c_i^2)`.

Then, with conditional probability at least `1-alpha`,

`m^{-1} sum_t R*(P_i_t) >= max(0, S-A-H_alpha)`.

Here `R*(P_i_t)` is the Bayes risk for the *unclipped* response under the chosen loss. This targets the tuple-anchor multiset; it equals the finite-design mean when each row anchors exactly once.

Proof sketch: (1) For each tuple, `E[r_t] <= inf_a k^{-1} sum_j E[ell(C(Y_j),a)] <= R*(C#P_i_t) + (L/k) sum_j W1(C#P_i_t,C#P_j) <= R*(P_i_t) + (L/k) sum_j w_tj`. The first inequality is `E inf <= inf E`; the second is Kantorovich–Rubinstein for each fixed action; clipping contracts W1 and the supported location-loss Bayes risks. (2) Changing one clipped response changes `S` by at most `L D c_i/(mk)`. McDiarmid's one-sided inequality gives `E[S] >= S-H_alpha`. Combine. This proof uses conditional independence and a *valid externally established upper cap*; entering a provenance string is not evidence that the cap is valid.

Implemented losses and constants: MAE `L=1`, q=.9 pinball `L=.9`, squared loss on `[lo,hi]` `L=2D`. Hard window misses are excluded: they are not W1-Lipschitz. One can study a ramp surrogate or TV caps, but that is a separate theorem and estimand check.

The inequality is not novel as a collection of standard ingredients: tuple SAA optimism, Wasserstein stability, and bounded differences are established concepts. The repository's `research/rtx_law_space/THEORY.md` already sketches Wasserstein tuple stability. The contribution, if any, would need to be an independently audited *sharper, useful* instance with representation/provenance contracts and matched-method evidence.

### Average-law rather than edgewise transport sensitivity

The proof above only uses the **average** of the W1 distances over the selected tuple edges. Thus an externally justified bound `eta >= (mk)^{-1} sum_tj W1(P_anchor(t),P_member(t,j))` suffices; separate valid upper bounds for every edge are stronger than needed. For a fixed tuple design, loss, clip, and one-sided concentration event, write `I=S-H_alpha` and `L` for the loss's outcome Lipschitz constant. Every `eta >= 0` is covered by the *same* event, yielding the conditional sensitivity curve `B(eta)=max(0,I-L eta)`. It becomes useful only if an externally supportable `eta` is below the breakpoint `eta_break=max(0,I/L)`. The curve is not itself evidence for a particular eta. This differs from the already documented *conditional-center* Lipschitz curve in `docs/theory/REPLICATE_BOUND.md`; that older theorem bounds movement of a mean/median, whereas this route restricts the entire conditional law in W1. Neither is assumption-free or a claimed new concentration theorem.

`transport_tuple_bound.average_w1_sensitivity` implements this diagnostic and refuses W30 under W1. `check_average_w1_profile.py` checks monotonicity, exact equality with a uniform-cap version of the original bound, the breakpoint identity, malformed grids, and refusal. A predeclared clip/tuple/loss portfolio still needs a union allocation over the distinct random statistics, but **not** over the displayed eta values because each curve shares one concentration event.

### Disjoint near-replicate tuples with bounded scores

The above overlapping-tuples route applies worst-case bounded differences to every outcome, which is particularly costly for pinball and squared loss. `disjoint_transport_bound.py` implements a second conditional theorem using outcome-blind, globally disjoint blocks. For each block `t` of size `k`, optionally subtract an externally frozen shift `s_i` from each `Y_i`, then compute its in-block empirical Bayes risk `r_t` and bounded score `g_t=min(r_t,C)`. For MSE only, clip residual outcomes to a declared finite interval *before* computing `r_t`; this contracts W1 and can only reduce the original squared-loss Bayes risk. MAE and Q90 need no raw-outcome clip because their loss is globally W1-Lipschitz and **the score itself** is clipped.

Let `Q_i=Law(Y_i-s_i|X_i)`, `m` be the number of disjoint blocks, `p=mk/n` the retained fraction, and `eta` an externally valid upper cap on the mean **ordered within-block** transport distance `(mk²)^{-1} sum_t sum_{i,j in t} W1(Q_i,Q_j)`. If `LCB_alpha(g_1,...,g_m)` is a valid lower confidence bound on the mean expectation of the independent bounded scores, then with conditional probability at least `1-alpha`,

`R*_full >= p * max(0, LCB_alpha(g)-L eta)`.

For any block and any member `j`, `E[r_t] <= inf_a k^{-1} sum_i E ell(Z_i,a) <= R*(Q_j)+(L/k)sum_i W1(Q_i,Q_j)`. Average over `j`, then over blocks. Clipping the nonnegative score from above preserves `E[g_t] <= E[r_t]`. Conditional entity independence makes disjoint block scores independent but not necessarily identically distributed, permitting the existing [Bentkus (2004)](https://arxiv.org/abs/math/0410159) lower-mean inversion. Omitted cases contribute a conservative zero. Per-case translation leaves Bayes MAE, pinball risk, and MSE unchanged, so an externally frozen `s_i` changes witness geometry but **not** the target information set. This is an application of translation alignment, not invention of translation-invariant Wasserstein geometry; see [Relative-Translation Invariant Wasserstein Distance (2024)](https://arxiv.org/abs/2409.02416).

The result is conditional on the **aligned-law** cap, not on observed residual similarity. Neither a declared cap provenance string nor a small empirical residual difference proves it. If the shift was fit using audit outcomes—even cross-fitted shifts from mutually overlapping audit folds—the score-independence argument needs a new dependence proof or an independent development/audit split. A frozen external predictor could supply the shift without any model training *inside Pythia*, but would still need timing, provenance, and cap audits. `check_disjoint_transport_theorem.py` found 0 violations in 500 exact-replicate Bernoulli runs for each of MAE/Q90/MSE, with positive bounds in every run; this is a code falsification check, not a proof. A deterministic-law negative control has true Bayes risk zero yet produces a positive bound when a deliberately false zero-W1 cap is asserted; the correct mean cap reduces it to zero.

## Frozen new experiment

`transport_tuple_bound.py` implements the inequality. `run_transport_audit.py` uses analytic **oracle** W1 upper caps from the known synthetic location-scale laws: a shared-noise coupling for same-family pairs and an independent-noise coupling otherwise, with Cauchy–Schwarz bounds on `E|Z|`. Oracle caps make this a mathematical falsification test, not a deployable protocol. `run_transport_portfolio.py` predeclares tuple sizes `4,8,12`, clips `[40,280]`, `[70,230]`, `[90,210]`, and three losses; it divides alpha .05 by all 27 candidates, picks the maximum after seeing outcomes, and uses *new* seeds 28001–28010 on each of the ten frozen OOD mechanisms (`n=800`). Thus the selection remains within a simultaneous 95% conditional statement, provided all caps and independence assumptions hold.

| Loss | Positive bound / 100 | Bound above known gold / 100 | Mean bound / gold |
|---|---:|---:|---:|
| MAE | 10 | 0 | 0.0162 |
| Q90 | 0 | 0 | 0 |
| MSE | 0 | 0 | 0 |

All ten positive MAE runs were from the Friedman-1 Gaussian variant; its median bound was 2.766 minutes. The other nine mechanisms had median zero. These 100 observations do not prove 95% coverage (that follows from the assumptions and derivation), and zero observed violations do not validate caps on real data. A separate 500-seed exact-replicate Bernoulli ±10 simulation found 0 violations for MAE, Q90, and MSE; it gave positive MAE and MSE bounds in 500/500, but Q90 remained zero. This checks an implementation corner case, not general utility.

An additional exact-replicate route uses the *synthetic DGP's* equality key `(code, true note complexity)` for mechanism 10, with tuple sizes `2,4,6`, the same three clips, and Bonferroni alpha across all 27 within-route choices. Anchors without enough exact twins are omitted and the positive subset bound is multiplied by its anchor fraction, using zero as a lower bound for omitted anchors. The first ten runs, on seeds 28001–28010 already used by the nearest-neighbor portfolio, were **post-hoc exploratory**: positive MAE in 10/10 (median 1.71 minutes), zero Q90/MSE. After freezing this route, fresh seeds 29001–29010 again gave positive MAE in 10/10 (median 1.91 minutes versus median gold 15.67), no gold violations, and zero Q90/MSE. The ordinary nearest-neighbor route had zero on mechanism 10. This isolates a real matching-geometry gain but remains weak and **oracle-only**: the equality of conditional laws and true note complexity came from the generator, not a validated reader or Mayo evidence. See `run_exact_replicate_route.py`, `exact_replicate_results.jsonl`, and `exact_replicate_fresh_results.jsonl`.

### Published concentration baseline on exact twins

The tuple route was not the strongest available way to use genuine independent replicates. Before opening seeds 30001–30010, `run_exact_pair_concentration.py` fixed an outcome-blind disjoint matching within each synthetic equality cell, three pair witnesses, five clips per loss, and Hoeffding, Maurer–Pontil empirical Bernstein, and Bentkus lower-mean concentration. Its 45 candidate choices each receive alpha `0.05/45`; the strongest lower bound is selected. For equal-law independent pairs, `E[|Y-Y'|/2] <= MAE*`, `E[min(q,1-q)|Y-Y'|/2] <= pinball_q*`, and `E[(Y-Y')²/2] = MSE*`. Clipping each nonnegative witness from above preserves the inequality while giving a known range. Pairs are disjoint, so their clipped scores are independent conditional on features, though not identically distributed. Unpaired rows contribute a valid zero lower bound. The existing `pythia.core.concentration.bentkus_lower_mean` implements [Bentkus (2004)](https://arxiv.org/abs/math/0410159); the empirical-Bernstein comparator follows [Maurer and Pontil (2009), Theorem 11](https://arxiv.org/abs/0907.3740).

The previously frozen exact-tuple route was also run on those same seeds, with no retuning, for a paired comparison:

| Loss | Median known gold | Median disjoint-pair bound | Median overlapping-tuple bound | Pair strictly tighter |
|---|---:|---:|---:|---:|
| MAE | 15.751 | 7.230 | 1.750 | 10/10 |
| Q90 | 4.117 | 0.703 | 0 | 10/10 |
| MSE | 482.050 | 180.668 | 0 | 10/10 |

Bentkus won the within-pair concentration selection on every run. There were zero pair-bound exceedances of known gold in these 30 loss results, but ten seeds are not an empirical coverage proof. This is a **negative result for the proposed overlapping-tuple route** and a concrete reason to make exact-law independent matching the first branch when justified. It is still not deployable on Mayo: the equality key and conditional-law identity are generator facts. Nor is it a Pythia novelty claim; the pair witness and Bentkus concentration are established ingredients already present in the repository. Exact results are `exact_pair_fresh_results.jsonl` and `exact_tuple_comparator_results.jsonl`.

### Disjoint tuples recover a loss-specific advantage

An overlapping-tuple concentration penalty is not the only way to test the tuple witness. `run_disjoint_tuple_concentration.py` partitions each synthetic exact-law cell outcome-blindly into independent blocks of size `2,3,4,6`. For MAE and Q90, the in-block empirical Bayes risk has expectation no greater than the population Bayes risk (`E inf <= inf E`); for MSE the `k/(k-1)`-corrected sample variance is unbiased. Clipping these nonnegative block scores from above remains conservative. Hoeffding, Maurer–Pontil, and Bentkus are each offered five cap choices per loss. All 180 candidate bounds receive alpha `0.05/180`, so within-route selection is simultaneous conditional on exact-law grouping and entity independence. Omitted cases contribute zero. The pair route remained frozen and was rerun on the same *fresh* seeds 32001–32010.

| Loss | Median gold | Median disjoint-tuple bound | Median disjoint-pair bound | Tuple strictly tighter |
|---|---:|---:|---:|---:|
| MAE | 15.753 | 7.050 | 7.244 | 1/10 |
| Q90 | 4.150 | 1.081 | 0.710 | 10/10 |
| MSE | 486.727 | 192.755 | 183.012 | 6/10 |

No disjoint-tuple bound exceeded gold in these 30 results; that observation is not a coverage proof. Bentkus was selected in every tuple run. Q90's improvement is an **exact-replicate, single-mechanism finding**, not broad OOD superiority or a new concentration theorem. It suggests a defensible router: independent pairs for MAE, larger independent tuples for skew-sensitive pinball risk, and compare both for MSE with a simultaneous selection allocation. Whether the Q90 advantage persists over other law families and group sizes is the next untouched test. Data: `disjoint_tuple_fresh_results.jsonl` and `disjoint_pair_comparator_results.jsonl`.

### Fresh ten-mechanism noise-law battery

The next untouched test, `run_exact_law_battery.py`, froze the above candidate grids and used seeds 33001–33005 on ten **new exact-replicate generators**: normal, Laplace, Gamma(2), lognormal, t3, t5, low/high complication mixtures, mixed normal/Laplace, and mixed Gamma/lognormal. Cell counts vary from 10 to 80 and some mechanisms are heteroscedastic. The case's numeric `code` and a simple pre-outcome note concept form an exact-law key *by construction*; the two routes see that same oracle key. This battery stresses noise shape and group density, **not** NLP reading or arbitrary continuous covariates. Each mechanism, not each of its five draws, is the comparison unit.

| Loss | Tuple strictly tighter / 50 runs | Mechanisms won / 10 | Median tuple bound / gold | Median pair bound / gold | Gold exceedances, either route |
|---|---:|---:|---:|---:|---:|
| MAE | 11 | 3 | 0.485 | 0.492 | 0 |
| Q90 | **50** | **10** | **0.276** | 0.189 | 0 |
| MSE | 31 | 6 | 0.329 | 0.327 | 0 |

The Q90 design-level 10/10 sign count would have one-sided p=0.00098 under a hypothetical independent, exchangeable random sampling of mechanisms. These ten hand-specified mechanisms are **not** such a sample; the p-value is only a descriptive reference, not evidence of universal or literature-wide dominance. No multiplicity-adjusted general win claim is made. The gain is over one accepted exact-pair witness with matched Bentkus concentration, not over all published quantile-risk methods. It does support further development of independent multi-point witnesses when conditional-law equality is defensible. Full rows are in `exact_law_battery_results.jsonl`.

### A valid joint pair-or-tuple selector

The head-to-head routes above each spent their own 5% error budget, so taking their maximum would not be a 95% certificate. `run_exact_law_portfolio.py` fixes a **single** alpha `0.05/225` for 3 losses × 5 score caps × 3 concentration methods × (the pair route plus 4 disjoint tuple sizes), then takes the largest bound for each loss. The route and all grids were frozen before a new seed block 34001–34005 was opened on the same ten hand-specified law mechanisms. Under the stated oracle exact-law equality and conditional entity independence, the union bound gives joint 95% conditional coverage over all reported loss/route choices on a single generated dataset.

| Loss | Positive selected / 50 | Tuple route selected / 50 | Median selected bound / gold | Median pair-only bound / gold | Gold exceedances |
|---|---:|---:|---:|---:|---:|
| MAE | 50 | 13 | 0.478 | 0.471 | 0 |
| Q90 | 50 | **50** | **0.267** | 0.180 | 0 |
| MSE | 50 | 34 | 0.323 | 0.319 | 0 |

This closes the *method-selection accounting* gap for the synthetic exact-law route; it does not close the *cap/grouping provenance* gap. The pair-only column uses the same more stringent `0.05/225` per-candidate allocation, so the ratio comparison is within the joint portfolio, not against an independently spending 5% baseline. One 95% statement applies per generated dataset, not jointly across all 50 datasets. The selected Q90 improvement over the pair witness persisted under the corrected allocation. Full rows: `exact_law_portfolio_results.jsonl`.

The portfolio is not a win over accepted bounds: it has no positive Q90/MSE bound on these designs, and it uses DGP knowledge unavailable on Mayo. The point-estimator ranking from `OOD_AUDIT_20261010.md` remains separate and unchanged.

On a further fresh block (seeds 35001–35005), `run_transport_sensitivity.py` reused the nine predeclared clip/tuple choices per loss with alpha `0.05/27`. Known-DGP analytic coupling supplied a conservative upper cap on the **mean** W1; that is still oracle information, not a real-data estimate. Across 50 runs, MAE was positive in 5/50, all from one mechanism; Q90 and MSE were zero in 50/50. The 450 candidate evaluations per loss separate the causes:

| Loss | Concentration intercept nonpositive | Intercept positive but oracle cap beyond breakpoint | Positive at oracle cap |
|---|---:|---:|---:|
| MAE | 58 | 362 | 30 |
| Q90 | 450 | 0 | 0 |
| MSE | 450 | 0 | 0 |

No selected synthetic bound exceeded gold in these 150 loss results. For MAE, the local-law shift penalty is the dominant obstacle. For Q90/MSE, the bounded-difference radius defeats the witness *before* transport enters. This is a concrete rationale for testing disjoint, independently concentrated near-replicate tuples with score clipping; it does not prove that such a method will work. `transport_sensitivity_results.jsonl` contains every chosen breakpoint and cause count. The analytic coupling upper cap may be loose, so its excess over a breakpoint does not imply the *true* W1 exceeds that breakpoint.

The proposed disjoint/score-clipped route was then frozen and run on *new* seeds 36001–36005 for all ten OOD outcome mechanisms (`n=800`), using the same outcome-blind representation, greedy disjoint feature grouping, tuple sizes `2,4,8`, three score clips per loss, and alpha `0.05/27`. Its transport cap is an analytic upper coupling computed from the synthetic DGP; this is deliberately oracle information. The old overlap-aware route was rerun without retuning on the **same seeds** for an aligned comparison. Each route spends its own 5% budget, so these are head-to-head comparisons, not a jointly valid post-hoc maximum.

| Loss | Disjoint positive / 50 | Overlap positive / 50 | Disjoint strictly tighter / 50 | Median disjoint bound / gold | Disjoint gold exceedances |
|---|---:|---:|---:|---:|---:|
| MAE | **44** | 5 | **44** | 0.158 | 0 |
| Q90 | 0 | 0 | 0 | 0 | 0 |
| MSE | 0 | 0 | 0 | 0 | 0 |

MAE design medians improved on nine mechanisms and tied at zero on the sparse 30-D mechanism. The available caps remain synthetic oracle upper bounds, not deployable. Importantly, all 450 Q90 candidates and all 450 MSE candidates now have a *positive* bounded-score lower endpoint; all are subsequently eliminated by transport. This isolates a different bottleneck than the previous bounded-difference route. The conservative coupling caps themselves may be loose. Raw rows are `disjoint_transport_results.jsonl` and `overlap_comparator_36001.jsonl`.

Translation alignment was tested on another fresh block, seeds 37001–37005. The two candidate shifts available before outcome inspection were zero and the booked-slot duration; they share a single alpha `0.05/54` across shifts, losses, tuple sizes, and score clips. A third, **synthetic-only oracle** shift equal to the known conditional mean was evaluated separately at alpha `0.05/27` as an attainable ceiling, not included in the available selector. Every cap was recomputed for the corresponding residual conditional laws.

| Loss | Zero-shift positive / 50 | Booked-shift positive / 50 | Available selector positive / 50 | Oracle-mean positive / 50 | Oracle median bound / gold |
|---|---:|---:|---:|---:|---:|
| MAE | 45 | 25 | 45 | 50 | 0.552 |
| Q90 | 0 | 0 | 0 | 30 | 0.250 |
| MSE | 0 | 0 | 0 | 25 | 0.113 |

No bound exceeded known gold in these 150 loss outcomes for any shift. Booked-slot alignment was selected for MAE only once and did not recover Q90/MSE. The oracle result says that location alignment could make those losses certifiable under a strong external shift and valid residual-law cap; it does **not** demonstrate a practical way to obtain either on Mayo. Results: `aligned_transport_results.jsonl`.

## Literature gaps that actually matter

- [Anderson and Bjarnadóttir (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0296904) is a direct attainable-MAE heuristic comparator. The existing matched implementation must remain in the benchmark; a point-estimation advantage is not a lower-confidence-bound advantage.
- [Chen, Guntuboyina, and Zhang (JMLR 2016)](https://www.jmlr.org/papers/v17/16-185.html) lower-bound *Bayes risk under a prior over statistical parameters* using f-informativity. That is mathematically different from estimating irreducible prediction risk `E_X inf_a E[ell(Y,a)|X]` from a labeled sample. It is relevant conceptual literature, not a matched numerical baseline without a shared estimand.
- The [Gamma test](https://citeseerx.ist.psu.edu/document?doi=d6c2bce609e2c8a90fed1433276ede9bf6c644df&repid=rep1&type=pdf) and [graph-based variance estimators](https://jmlr.org/papers/v25/23-1061.html) estimate noise under smoothness; they expose the core question of which smoothness conditions are checkable or externally credible.
- [Learning-curve asymptotics](https://research.google/pubs/learning-curves-asymptotic-values-and-rate-of-convergence/) estimate a limit of a model-learning process, not an assumption-free Bayes floor.
- [Bound to Disagree](https://proceedings.mlr.press/v337/bazinet26a.html) concerns relative model risk, a different estimand. Do not imply it is beaten without a common target and protocol.
- [Conditional-inference impossibility results](https://arxiv.org/abs/1903.04684) reinforce the need for restricted assumptions, though their exact target differs from this finite-design Bayes-risk problem.
- [Low (1997)](https://doi.org/10.1214/aos/1030741084) and later honest adaptive nonparametric-inference results caution against learning an optimistically small regularity cap from the same outcomes while retaining uniform coverage. Their formal estimand is different; they motivate an independent cap-audit design, not a direct impossibility theorem for this exact Pythia statistic.

## Next theory/implementation gates

1. **Nonvacuous cap provenance.** Specify an externally justified conditional-law modulus, exact-replicate rule, or independent calibration sample. Never estimate `w_tj` from the same outcomes and present the result as proved without a simultaneous upper confidence correction. Distinguish an oracle-cap simulation from deployability.
2. **Sharper concentration and loss routing.** The exact-twin tests put disjoint-pair Bentkus ahead of *overlapping* tuples, while disjoint tuples recover a Q90 advantage across ten oracle exact-law noise mechanisms. On genuinely near-replicate OOD mechanisms, disjoint score-clipped tuples improve MAE nonvacuity from 5/50 to 44/50, but Q90/MSE still need much smaller *valid* residual-law caps. A joint pair/tuple selector now allocates one alpha budget for exact laws; near-law head-to-head results do not yet form a jointly valid max. Test externally frozen alignment and cap audits rather than borrowing the synthetic oracle mean. Use any exact-law branch *only when its grouping is externally verified*; otherwise keep transport-corrected claims conditional/refused.
3. **Loss routing.** For pinball, seek a one-sided order-statistic/binary-tail bound. For window misses, either derive a valid total-variation bound or a *lower* Lipschitz surrogate with quantified approximation gap. Do not transfer the MAE proof by analogy.
4. **Point-estimator fairness.** Compare on new outcome mechanisms and new note semantics, allow baseline feature screening on a development split, and report mechanism-level paired errors and multiplicity-adjusted uncertainty. Keep predictive training baselines outside Pythia's no-training core.
5. **Mayo transfer gate.** Only after a nonvacuous independently validated method exists, audit patient-grouped sampling, pre-cutoff notes, representation sufficiency, and cap provenance. Until then Mayo outputs may be retrospective calibrated estimates, not theorem-level Bayes certificates.

## Reproduction and code status

Run `python3 check_transport_theorem.py`, `python3 check_average_w1_profile.py`, `PYTHONPATH=REPOSITORY_ROOT python3 check_disjoint_transport_theorem.py`, `python3 run_transport_audit.py --out NEW_PATH`, `python3 run_transport_portfolio.py --out NEW_PATH`, `python3 run_transport_sensitivity.py --out NEW_PATH`, `PYTHONPATH=REPOSITORY_ROOT python3 run_disjoint_transport_audit.py --out NEW_PATH`, `PYTHONPATH=REPOSITORY_ROOT python3 run_aligned_transport_audit.py --out NEW_PATH`, `python3 run_exact_replicate_route.py --out NEW_PATH`, `PYTHONPATH=REPOSITORY_ROOT python3 run_exact_pair_concentration.py --out NEW_PATH`, `PYTHONPATH=REPOSITORY_ROOT python3 run_disjoint_tuple_concentration.py --out NEW_PATH`, `PYTHONPATH=REPOSITORY_ROOT python3 run_exact_law_battery.py --out NEW_PATH`, and `PYTHONPATH=REPOSITORY_ROOT python3 run_exact_law_portfolio.py --out NEW_PATH` from this audit directory. The output path must not exist. This research code is separate from the Pythia public API, has not been certified on Mayo, and should not be merged into the release path until cap provenance and nonvacuity gates pass.
