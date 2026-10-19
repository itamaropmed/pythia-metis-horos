# Pythia package and note-augmentation audit (2026-09-14)

## What the package does

Pythia estimates the *attainable loss floor* for a declared information set
and decision loss. It is not a model trainer. The original `certify` API uses
honest, frozen out-of-fold predictions and a versioned guard for the
calibrated Mayo MAE ladder. The newer train-free replicate and pair/group
witness APIs use outcomes and an outcome-blind feature geometry, without
fitting a predictor. These are different estimands and contracts; neither
silently replaces the other.

For a fixed cohort with covariates `x_i`, let `a*(x_i)` be a conditional Bayes
action for loss `ell`. If externally justified
`|a*(x_i)-a*(x_j)| <= L d(x_i,x_j)`, then the observed loss minimized over
all action vectors obeying that cap is pointwise no greater than the loss of
the true Bayes actions. Its conditional expectation is therefore no greater
than the finite-cohort mean Bayes risk. Disjoint, outcome-blind groups of
conditionally independent entities allow a one-sided concentration bound.
Predeclared pair/group, method, clip and loss choices consume multiplicity
budget. Window-miss lower bounds imply upper bounds on attainable within-W
percentages; pinball lower bounds address quantile decisions. The grouped
exact MAE/pinball LP remains `exploratory` until numerical verification is
independently audited. The window clique path is `conditional` only under its
stated structural and sampling assumptions. The witness center is **not** a
competitive or certified point estimator of the Bayes floor.

The geometry router selects paired or grouped witnesses using only the
representation, declared `L`, outcome span/window and fixed thresholds. It
never reads outcomes to route. Those thresholds are empirical development
choices, not a new theorem; the W threshold was adjusted after retrospective
Mayo inspection. The original guarded Arm-B MAE result (28.8495996043 min,
`calibrated`) and default multi-loss route remain frozen.

## Using the generic train-free route

`from pythia import geometry_routed_decision_envelope` accepts an `n x d`
matrix in `[0,1]`, aligned outcomes, unique entity IDs, a disjoint partition
of row indices into pairs, and outcome-blind disjoint groupings. For q90,
pass `loss="pinball", quantile=.9, outcome_range_bound=R`; for W30 miss,
pass `loss="window_miss", window=30`. Always declare and audit
`lipschitz_bound=L`; `alpha` is a budget **for one registered call** unless
allocated across every loss, representation, slice and period. Pair and group
constructors live in `horos.represent`, not Pythia. The route returns a
structured refusal for unsupported losses and carries claim tier, family,
bound and diagnostic provenance. Calling it does not itself prove its
assumptions. Do not use a retrospective numeric output as a clinical
certificate without a prospective protocol and external audits.

Reproduction:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_mayo_text_augmentation.py tests/test_mayo_oracle_leakage.py
.venv/bin/python scripts/run_mayo_text_augmentation.py --n-entities 256
.venv/bin/python scripts/run_mayo_oracle_leakage.py
```

The first script reads the private `data/derived/mayo` artifacts in memory
and prints aggregate JSON only. It takes the first 256 chronological,
earliest-per-patient cases; 216 have nonempty recut pre-cutoff note bundles.
It reuses identical outcomes/entities for all controls and makes a stateless,
fixed 64-coordinate normalized word/unigram-bigram *count* hash of each note
(binary presence saturated on long notes in a development smoke test).
Tabular coordinates
are fixed; pair/group designs are recomputed *without outcomes* for each
changed representation. No TF-IDF/SVD vocabulary, predictor, embedding or
language model is trained. `L=300` and `[0,900]` are illustrative, not audited
population premises. The nominal 0.05 alpha is split across eight controls
and two losses; this does not fix retrospective selection or assumptions.

| Note control | Changed rows | q90 pinball lower bound (loss-min) | W30 miss lower bound |
| --- | ---: | ---: | ---: |
| Intact | 0 | 1.6756 | 0.2574 |
| 50% case dropout | 104 | 1.3774 | 0.2574 |
| 100% case dropout | 216 | 1.7813 | 0.2261 |
| Character shuffle | 216 | 1.4682 | 0.2574 |
| Token shuffle | 216 | 2.0465 | 0.2418 |
| 50% span mask | 216 | 1.2375 | 0.2457 |
| 50% segment dropout | 214 | 1.6496 | 0.2418 |
| 50% segment duplication | 216 | 1.7942 | 0.2183 |

The intact numerical bound is **not** largest. Token shuffle raises the q90
bound, full dropout raises q90 but lowers W30, and 50% dropout leaves W30
unchanged to four decimals. This is a useful falsification of any claimed
monotone bound response to generic text corruption, not an
information-theory violation: hashed, altered text changes pair distances,
group membership and the meaning of a fixed `L`, while these are lower-bound
statistics, not true Bayes risks. Token shuffle preserves unigrams but not
bigrams, and segment duplication can preserve lexical information while
changing counts and geometry. Neither a higher bound nor a fitted model's
lower MAE alone establishes the value of notes. Mayo Bayes gold is unknown.

The independent in-memory oracle test injects the **actual** duration into
10%, 50%, and 100% of note bundles on 2,048 different unique patients. Error
on injected rows is zero; full-oracle MAE and the full-oracle pair bound are
zero under its stated metric. The public certificate refuses the target-leaked
representation. It also confirms that duplicating a tabular schedule feature
20 times changes the numeric pseudo-bound despite no change in mathematical
information. See `REPLICATE_AND_MAYO_AUDIT_20260912.md` for exact values.

## What is new, and what is not claimed

Decision-specific information here is Bayes-risk reduction
`R*_ell(X)-R*_ell(X,Z)`, an established decision-theoretic/Blackwell notion,
not an invented Shannon information measure. The engineering contribution is
a general train-free risk-floor witness across MAE, quantile and within-window
decisions, with loss-specific finite-group range accounting, outcome-blind
geometry routing, explicit multiplicity and fail-closed claim tiers. Whether
this combination is a publishable novel theorem remains an open literature
and independent-proof question. It is distinct in target from recent
[Bound to Disagree](https://proceedings.mlr.press/v337/bazinet26a.html),
which bounds a predictor's generalization error with surrogates/disagreement,
and from [Model Agreement via Anchoring](https://proceedings.mlr.press/v336/eaton26a.html),
which controls model disagreement; Pythia targets an attainable loss floor.
The concentration components are established methods, including
[Bentkus's binomial-tail inequality](https://arxiv.org/abs/math/0410159),
not new inequalities.

The ten known-risk numeric/text-like synthetic designs are the honest place
to compare tightness: the frozen route was below analytic truth on 500/500
loss cells and had a higher median bound on nine designs, tying one, than the
numerical max of three paired concentration comparators. This does **not**
show superiority to every literature floor estimator, and the comparator
maximum is not itself multiplicity-adjusted. External fitted point
estimators beat the Pythia witness center; a best-estimator claim is not
supported. Details and failure modes are in
`MULTILOSS_VALIDATION_20260913.md` and `THEORY_GAP_CLOSURE_20260913.md`.

## Release gates and next development

1. Independently audit the exact LP numerical envelope and action-smoothness
   assumptions; refuse proof-level claims until that passes.
2. Freeze the feature metric and router on development data *before* a new
   temporal/prospective Mayo evaluation. Add a legitimate all-notes and
   no-notes comparison with fold-contained text transforms and matched OOF
   predictors in Horos; Pythia remains train-free.
3. Audit recut note provenance, schedule leakage, entity independence and
   support/range. Restrict any clinical claim to tested, input-defined slices.
4. Freeze endpoints, models, representations, thresholds and analysis; use
   3,000 new eligible cases for paired patient-clustered confirmation. Until
   then all multi-loss Mayo numbers remain exploratory.
5. Keep Metis behind the M2 gate. It may propose representations and witnesses
   and run falsification checks, but cannot promote a tier or tune a guard on
   the evaluation outcomes. Horos owns model fitting, UI and reports.
