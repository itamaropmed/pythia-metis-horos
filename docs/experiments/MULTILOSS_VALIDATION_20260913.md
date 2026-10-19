# Multi-loss Pythia audit (2026-09-13)

## Scope

The reusable train-free `decision_pair_group_envelope` now covers pinball
(q10/q90) and within-window miss (W15/W30/W60). The predeclared `family_policy`
may be `paired`, `grouped`, or `adaptive`. Paired-only and grouped-only spend no
alpha on the unused family; adaptive selection covers both families. The public
paired certificate API now accepts conditional Bentkus as well as Hoeffding and
empirical Bernstein. Pythia fits no prediction model; kNN, histogram, and kernel
plug-ins live in Horos and are only point-estimation/achieved-risk comparators.

The theorem target is **finite-design conditional Bayes risk**, averaged over
the audited covariates. It requires outcome-blind matching/grouping and metric,
a predeclared and externally justified Lipschitz cap on a selected Bayes action,
independent entity outcomes given covariates, and a population-valid score cap
or predeclared clipping for unbounded losses. Bonferroni covers methods, clips,
families, and the five losses. Reusing outcomes to select any item of that
registry is not covered. W60 has a known zero Bayes miss risk in the synthetic
library; a bound/truth tightness ratio is undefined there, not zero or perfect.

### Sharper pinball concentration range

For a declared outcome support span `R` and pinball level `tau`, choose the
constant action `a = min_support + tau*R`. Every observation then has pinball
loss at most `tau*(1-tau)*R`. The group's minimized *relaxed* pinball score
cannot exceed this value. This improves the prior range
`min(tau,1-tau)*R`, especially near the median, without changing the witness
or adding an assumption. Equality can occur for a two-atom law with upper-atom
mass `1-tau`, so the cap is sharp over bounded distributions. An exact q50/MAE
scaling test and exhaustive two-point witness tests now guard the derivation.

For a predeclared simultaneous window grid, Bayes miss risk is nonincreasing
in the window width. The deterministic closure `B(w) <- max_{v >= w} B(v)`
can therefore only tighten a valid lower bound at smaller `w`; it spends no
additional alpha and cannot repair invalid assumptions. It is applied
equally to every synthetic bound comparator and to the exploratory Mayo
window grid.

## Fresh known-risk comparison

Seed `20261006`, fixed after the seven-design seed-`20261005` pilot and the
sharp cap change: ten duration-like designs (five numeric and five
deterministic pre-outcome text templates), 20 draws/design, 512 independent
entities/draw. The new designs stress irrelevant numeric dimensions, extreme
heteroskedasticity and a rare note cue. Analytic noise laws supply all five
Bayes risks. Five losses share each draw: 200 independent draws and 1,000
loss results. Each loss receives alpha=0.01 for joint 95% coverage per draw,
conditional on the theorem assumptions. The text is templated, not clinical
natural language, and its extractor is frozen and literal.

| Method | Valid / 1,000 | Median usable tightness on 800 positive-floor cells |
| --- | ---: | ---: |
| Adaptive Pythia pair/group envelope | 1,000 | 0.1581 |
| Frozen loss-only Bentkus route | 1,000 | 0.1843 |
| Paired conditional Hoeffding | 1,000 | 0.0000 |
| Paired conditional empirical Bernstein | 1,000 | 0.0717 |
| Paired conditional Bentkus | 1,000 | 0.1818 |

The frozen route is paired Bentkus for pinball and grouped Bentkus for windows.
It is *not* an adaptive win on the held-out outcomes. W15 route median tightness
is 0.2678 versus paired Bentkus 0.2175; W30 is 0.0747 versus 0.0580. For
q10/q90 the route **ties** the paired Bentkus baseline at 0.1862. On positive
floors, adaptive Pythia beats paired Bentkus in 237/800 and loses in 543/800;
the route wins 304, loses 76, and ties 420. The paired Hoeffding baseline also
beats the adaptive envelope in two cells. All five losses were jointly below
truth in 200/200 draws; the pooled 95% Wilson lower limit is 0.9812. Each
individual design has only 20 draws, so this is not evidence for uniform
coverage over text distributions. All W60 true floors and bounds are zero.
The sparse 4D design has particularly low median tightness (adaptive 0.0234,
routed 0.0311), exposing a metric/dimension weakness even when validity holds.
The simultaneous monotone-window closure was applied equally to all bound
methods; it did not alter these particular draws.

Per-loss median absolute point-estimation errors for the Pythia witness center
versus cross-fitted kNN / histogram / kernel, respectively:

| Loss | Witness center | kNN | Histogram | Kernel |
| --- | ---: | ---: | ---: | ---: |
| q10 pinball | 1.9729 | 0.1905 | 0.2726 | 0.1389 |
| q90 pinball | 1.9643 | 0.1963 | 0.2419 | 0.1723 |
| W15 miss | 0.2857 | 0.0156 | 0.0192 | 0.0176 |
| W30 miss | 0.1242 | 0.0192 | 0.0191 | 0.0173 |
| W60 miss | 0 | 0 | 0 | 0.0010 |

This negative control is decisive: the witness center is a lower-bound plug-in,
not a competitive general Bayes-floor point estimator. Risks with different
units should not be pooled to compare point-estimation error. Three established
concentration inequalities were compared on the same paired witness, **not**
three independent literature Bayes-error methods. No universal best-bound or
best-estimator claim is supported.

## Theory checks and Mayo transfer

Exact enumeration of all 16 outcome vectors from a two-point conditional law
confirmed the *population witness inequality* for q10, q90 and W10, for both
paired and grouped statistics. Numeric and text-like synthetic risk formulas
were independently checked by 100,000-row Monte Carlo draws. These tests do
not replace an independent proof audit.

The separate retrospective Mayo audit uses 10,528 earliest-per-patient cases
in four chronological panels, two existing representations and five losses.
The weighted tabular results after the sharp group pinball cap are q10 bound
2.3704, q90 bound 4.5177 loss-minutes,
W15 miss bound 0.3381, W30 0.2019 and W60 0.0752. W30 implies an illustrative
attainable within-30-minute percentage ceiling of 79.81%, versus 62.60%
achieved by the frozen MAE-selected OOF predictor. The tabular-plus-note-value
W30 bound is 0.1882; its lower numerical value is **not** evidence that notes
reduce true information. Both representations' window grids yield exploratory
strict absolute-error quantile floors p75 > 15 minutes and p90 > 30 minutes.

All Mayo multi-loss numbers remain **exploratory**: L=300 and a [0,900] minute
outcome cap are illustrative rather than externally validated; quantile and
coverage-action smoothness, patient independence, feature timing, and
retrospective re-use are not independently audited. The historical Arm-B
guarded MAE floor 28.8495996043 remains **calibrated** and is unchanged. The
alleged 31.2623 held-out duration incumbent is not reproducible from the
identified artifact and is not used as a comparator.

## Outcome-blind center pilot (new, not a release claim)

The grouped witness can now use `center_rule="coordinate_median"` in place of
its default coordinate mean. The rule depends only on audited features, never
outcomes, and fits no predictive model. The original group-witness inequality
continues to hold for any such feature-measurable center; changing the center
changes witness strength, not the target or confidence allocation. The default
remains `mean`, so frozen Mayo anchors are not silently changed.

On a fresh seed (`20261007`), ten designs x ten draws x q10/q90 gave 200
loss-results. The group-only Bentkus lower bounds were below their analytic
Bayes risks in all 200 results for both centers; the paired-only Bentkus bound
was also valid in all 200. The coordinate-median center improved the numeric
bound over the mean in 180/200 results and worsened it in none, but median
bound/truth tightness moved only from 0.18814 to 0.18866. It beat the paired
bound in 108/200 results. On the sparse 4D design, both group centers had
median tightness 0.01130 versus 0.03334 for the paired bound. Thus this is a
valid but small improvement, **not** the missing broad advantage. No point
estimator was improved or evaluated in this pilot.

The next theory target is to strengthen the witness itself under sparse or
high-dimensional geometry, with a predeclared neighborhood/graph design and a
finite-sample correction that accounts for overlapping outcome pairs. It
must preserve outcome-blind selection and be tested on a new untouched seed.
Concentration-only changes cannot close the witness-center point-estimation
gap shown above. Any point estimator compared with trained competitors needs
a separately defined train-free estimand/estimation method or must remain an
external Horos comparator; the certificate witness should not be advertised
as that estimator.

## Reproduction

### Exact group Lipschitz witness (2026-09-14 pilot)

An opt-in `witness_rule="exact_lipschitz"` now replaces the radial group
relaxation with the minimum empirical loss over group action vectors obeying
all pairwise declared Lipschitz constraints. For MAE and pinball, this is a
small linear program. For a within-window miss, a subset of cases can all be
covered exactly when every pair satisfies
`|y_i-y_j| <= 2*window + L*d(x_i,x_j)`; real-valued Lipschitz selection of
intervals makes the pairwise condition sufficient. The score is one minus
the maximum compatible-clique fraction, with a fail-closed group-size cap of
24. This is a loss-specific witness calculation, **not** predictor training.

Conditional on the externally justified Lipschitz action cap, the true Bayes
action vector is feasible, so the exact score is at most its observed loss.
Taking conditional expectations makes its mean no greater than the finite-
design Bayes risk. Its feasible set is contained in the radial relaxation,
so its raw score is never smaller than the radial score. Disjoint groups and
independent entity outcomes permit the same conditional concentration step.
The pinball/MAE score range remains bounded by a constant common action under
the declared outcome span; window-miss remains bounded by `1 - 1/group_size`.
Exact LP arithmetic has only numerical, not formal interval, verification;
its MAE/pinball code path is therefore labeled `exploratory` pending an
independent numerical solver audit. The exact window clique path remains
`conditional` under its explicit premises. This pilot is **not** a
machine-checked proof or a novelty claim. The default
remains radial and all earlier Mayo anchors remain unchanged.

On untouched seed `20261008`, ten numeric/text-like designs x five draws x
five losses, 256 entities per draw, the fixed exact-group Bentkus route was
below analytic truth in all 250 loss-results. It exceeded the radial group
bound in 132/250, tied in 118, and lost in none. Across 200 positive-floor
cells, median bound/truth tightness was 0.08952 exact versus 0.08639 radial.
But paired Bentkus reached 0.12400; exact groups beat it only 59/250, tied
57, and lost 134. In the sparse 4D design, radial and exact group medians
were both 0.00131, far below paired 0.01110. This confirms that the exact
optimization repairs relaxation slack but **not** the sparse-neighbor gap.
These are three witness geometries with the same Bentkus concentration, not
three independent literature floor-estimation methods. No point-estimator
comparison was performed in this pilot.

The opt-in Mayo retrospective rerun, on the same four panels and illustrative
`L=300`/[0,900] cap, moved weighted tabular q10/q90 bounds from 2.3704/4.5177
to 2.5986/4.7877 loss-minutes, and W30 miss from 0.2019 to 0.2190. The
tabular-plus-note-values W30 miss bound moved from 0.1882 to 0.2143. Those
figures are still **exploratory**, not certified: action smoothness, cap,
timing and retrospective selection remain unaudited. A richer
representation's smaller bound is not evidence of less information.

The remaining theory problem is consequential: without close repeats or a
stronger externally defensible structural assumption, distinct high-
dimensional feature points permit low-noise action functions that interpolate
the outcomes under a loose Lipschitz cap. Sharper concentration or group
optimization cannot by itself manufacture a positive floor. Future work
should test a predeclared metric/structural assumption on a new seed, and
separately define a train-free point estimator if Pythia is to compete on
Bayes-floor estimation accuracy.

### Entity-level bounded differences and selection-safe routing

The grouped window score, whether radial or exact, is the minimum of an
average of zero/one per-case losses. Altering one entity's outcome changes its
group score by at most `1/group_size`, and the full cohort mean by at most
`1/n_entities`. McDiarmid's one-sided inequality therefore gives radius
`sqrt(log(1/alpha)/(2*n_entities))` under conditional entity independence.
For MAE/pinball with a population-valid outcome span `R`, the analogous
radius uses per-case sensitivity `R` or `max(tau,1-tau)*R`; clipped-only
unbounded pinball refuses this route. This concentration result is standard;
its use here is a sharper accounting of the optimized witness's per-entity
sensitivity, **not** a new concentration theorem.

Seed `20261009` was development for this method: exact-group entity bounds
gave W15 median tightness 0.4222 versus exact-group Bentkus 0.2028, but zero
median tightness for pinball and W30 at 256 entities. A loss-only route was
then frozen before seed `20261010`: paired Bentkus for q10/q90, exact-group
entity bounds for W15, and exact-group Bentkus for W30/W60. On that new
ten-design, ten-draw, 512-entity run (500 loss-results), all route bounds
remained below analytic truth. Against paired Bentkus the route won 163,
tied 309, and lost 28; its median positive-floor tightness was 0.1833 versus
0.1798 paired. W15 was much stronger (0.4967 versus 0.2118), while sparse
4D remained weak.

A further predeclared, **selection-safe** route spends alpha across Bentkus
and entity-McDiarmid within each grouped window envelope and takes the
larger valid bound. On untouched seed `20261011` (same 500-cell design), it
was valid in all 500 cells. Against paired Bentkus it won 153, tied 310, and
lost 37. Median positive-floor tightness was 0.1863 versus 0.1804 paired;
W15 was 0.4852 versus 0.2151 and W30 0.0940 versus 0.0578. Its multiplicity
cost made W15 slightly weaker than the fixed entity route; it did not beat
paired on the sparse 4D design. These per-loss alpha allocations imply joint
95% coverage **per independent synthetic draw under the stated assumptions**,
not a proved uniform 95% statement across all data-generating processes.
Point-estimation accuracy was not improved here.

The corresponding opt-in Mayo `safe_group` route, with exact quantile groups,
moved the exploratory weighted tabular W15/W30/W60 miss-risk bounds to
0.4220/0.2678/0.1049, from the frozen radial legacy values
0.3381/0.2019/0.0752. The W30 result implies an illustrative attainable
within-30-minute ceiling of 73.22%, versus 62.60% achieved by the frozen OOF
model. For tabular-plus-note-values, W30 moved from 0.1882 to 0.2626.
These are **retrospective exploratory** numbers under an illustrative `L=300`
and [0,900] outcome cap, not a certified Mayo improvement. The old MAE
28.8495996043 calibrated anchor and default multi-loss path are unchanged.

An interpolation barrier explains the remaining gap: if a fixed-design
outcome vector at distinct feature points itself satisfies
`|y_i-y_j| <= L*d(x_i,x_j)` for all pairs, a deterministic Lipschitz
conditional action can pass through every observation and has zero Bayes
risk. A positive uniformly valid lower bound on that exact sample would fail
under this deterministic data-generating process. Thus close replication or
an independently defendable stronger structure is necessary to overcome
some sparse/high-dimensional cases; no optimization trick can guarantee a
positive floor there under the current assumptions alone. This is a
limitation argument, not a claimed novel impossibility theorem.

### Finite-group quantile cap and ten-design outcome-blind route

The previous pinball score cap `tau*(1-tau)*R` is sharp only when the group
size can realize the optimal low/high-atom proportion. For a group of fixed
size `k`, the exact worst-case empirical constant-action cap is
`R * max_{m=0,...,k} min(tau*(k-m)/k, (1-tau)*m/k)`. Partial minimization of
the jointly convex pinball objective is convex in the outcome vector, so its
maximum over a bounded outcome cube occurs at endpoint vectors; `m` counts
the lower endpoints. At q10 the factors are 0.05 for pairs, 0.075 for four,
and 0.09 for ten. This is a deterministic finite-group sharpening, not a
new assumption. It is **opt-in** (`pinball_range_rule="finite_group"`) so
earlier Mayo defaults and frozen results remain unchanged.

Using this cap, exact grouped Bentkus was numerically tighter than paired
Bentkus in 175/200 fresh q10/q90 cells on seed `20261012`; its median bound
was higher on nine of ten registered designs, but it lost all 20 cells on
the sparse 4D design. A predeclared geometry gate then used only features,
the declared action cap, and the outcome span: choose exact groups for
pinball when median paired action variation divided by the outcome span is
at most 0.01, otherwise retain paired Bentkus. On seed `20261013`, it beat
paired Bentkus in 180/200 quantile cells and tied the 20 sparse cells; all
200 were below the known Bayes risk. The gate is an outcome-blind heuristic,
not a theorem that these thresholds are optimal. It is exposed through the
reusable train-free `geometry_routed_decision_envelope`; the Mayo audit calls
that same route, while threshold provenance must be audited separately for
any certificate.

The five-loss gate uses selection-safe exact-group Bentkus/entity-McDiarmid
for windows when median action variation is at most half the window, paired
Bentkus otherwise. The half-window threshold was revised after inspecting
retrospective Mayo results; Mayo is therefore development evidence. On a
**new synthetic seed `20261015`**, ten registered designs x ten draws x five
losses (500 cells, 512 entities/draw), the route was below analytic truth in
500/500 cells. Against the numerical maximum of paired Hoeffding, empirical
Bernstein, and Bentkus it won 339 cells, tied 140, and lost 21. Its median
tightness over 400 positive-floor cells was 0.2195 versus 0.1791 for the
paired-classic maximum. Nine designs had a higher median; sparse 4D tied.
The comparator maximum is **not** a multiplicity-adjusted 95% certificate;
the individual paired methods each received the same alpha and Bentkus was
the numerical maximum on this seed. W60's true floor was zero in every draw
and contributes ties, not informative wins. This is nine strict design wins
and one tie, **not ten strict wins**, and no point-estimator improvement.

On the same frozen four-panel Mayo cohort, the opt-in geometry route gave
exploratory weighted tabular q10/q90 lower bounds 2.9436/5.2664 loss-minutes
and W15/W30/W60 miss-risk lower bounds 0.4220/0.2678/0.1049. For tabular
plus note values, the corresponding values were 2.9048/4.9176 and
0.4084/0.2626/0.1007. These retain or exceed the earlier opt-in exact/safe
results on the reported cells and the W30 historical anchors, but they are
**not independent Mayo validation**: the gate was revised after viewing
retrospective Mayo bounds, and the Lipschitz cap, outcome support and feature
timing remain unaudited. The default multi-loss path and calibrated Arm-B
MAE 28.8495996043 were not changed. Exact LP quantile outputs remain
`exploratory` until a numerical solver audit.

A separate outcome-blind 2-opt matching pilot reduced mean pair distance in
sparse 4D from 0.0688 to 0.0654 on seed `20261016`, but its median q10/q90
bound fell from 0.1000 to 0.0960; each quantile won six and lost four draws.
It remains an optional negative control, never an automatic route. Lower
geometric distance alone does not guarantee a stronger observed witness.

`python3 -m horos.benchmark.exact_group_comparison`

`pytest -q -p no:cacheprovider tests/test_multiloss_exact_witness.py
tests/test_exact_group_comparison.py`

### Earlier multi-loss reproduction

`pytest -q -p no:cacheprovider tests/test_multiloss_synthetic.py
tests/test_multiloss_exact_witness.py tests/test_decision_family_envelope.py
tests/test_multiloss_comparison.py tests/test_mayo_extended_losses.py`

The public synthetic tests require no Mayo data. The private Mayo test skips
when the local derived cohort is unavailable and prints only aggregates.
