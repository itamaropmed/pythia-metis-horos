# Build plan

## Layers

| layer | what it is | ships as |
|---|---|---|
| **pythia** | theory + pure computation of B and Ê; no training | library + paper |
| **metis** | agent searching for tighter certificates under a gate | service + paper |
| **horos** | application: ingest → certificate → exploration | skill / app |

## Runtime pipeline

```
HOROS/EXTERNAL: S0 INGEST -> S2 REPRESENT -> S3 TRAIN/CROSSFIT -> FROZEN OOF ARTIFACT
                                                                    |
PYTHIA:                   S1 LOSS ROUTER -> S4 CERTIFY/REFUSE <-------+
                                                |
                                                v
                         S5 OBSERVE -> S7 EXPLORE -> S8 DECLARE
```

## S1 loss router

| loss | R* | regime | guard |
|---|---|---|---|
| MSE, RMSE, Poisson/Gamma/Tweedie deviance, log loss, cross-entropy | E[Var], E[D_l], H(Y\|X) | Bregman - exact | none, q = 1 |
| MAE | E[MAD(Y\|X)] | guarded | envelope q(rho, tau) |
| pinball at alpha | E[rho_alpha(Y\|X)] | guarded, per-alpha | alpha-envelope |
| 0-1, AUC, F1, NDCG | - | refuse | route to Ishida / Fukunaga-Hummels / Devroye-Wagner |

The Bregman decomposition is algebraically exact, but a finite-sample MSE
certificate is not unconditional while A-fold orthogonality remains untested.
The public API labels or refuses accordingly.

## 2026-09-11 certification checkpoint

Pythia now has two complementary, model-free real-world paths in addition to
the frozen predictor-disagreement route:

```text
frozen representation + outcome-blind disjoint pairs
  -> audited independent entities
       -> fixed-design conditional empirical Bernstein
  -> audited independent blocks only
       -> conditional block empirical Bernstein
  -> neither audit
       -> refusal
```

`concentration: auto` implements this hierarchy without fitting a model. Global
nearest pairing and the single-event Lipschitz curve materially tighten the
independent-entity path. A 1,500-run adversarial suite achieved 1,500/1,500
validity and median tightness 0.624; a separate clustered-shock suite showed
that ignoring block dependence fails the coverage gate.

The immediate development stage is no longer “invent another estimator.” It
is to turn the conditional inputs into auditable study contracts:

1. independent review of `FIXED_DESIGN_THEOREM_AUDIT.md`;
2. freeze the representation metric without outcome access;
3. justify a smoothness curve from external/pilot information, not the Mayo
   evaluation outcomes;
4. declare the dependence unit prospectively (patient, week, site/week, or a
   justified dependency graph); and
5. run the frozen pair construction on untouched cases.

The retrospective Mayo global curves remain exploratory. The 170-week stress
curve is deliberately much wider and shows why dependence auditing, rather
than optimistic automation, is the next binding constraint.

## S4 core

```
L   = min_m loss(y, P_oof[:,m])
rho = MAD(r)/sd(r) ;  tau = (M+/sqrt2)/MAD(r)
q   = envelope(rho, tau)          # = 1 on the Bregman branch
B   = sqrt((MAD(r)/q)^2 - M+^2)
E2  = sqrt((L/q50(rho,tau))^2 - (Mbar/sqrt2)^2)
Gamma = E2 - B
```

Always report the full rung ladder. Stratification R* = sum_s w_s R*_s licenses S7.
Cluster bootstrap on the entity key, never case-level.

## S6 Metis gate - all three or discard

1. **Validity** - bound valid on every design in the certification suite
2. **Non-inferiority** - pooled usable tightness `rho_B * 1{rho_B <= 1}` does not decrease
3. **Provenance** - re-parameterisation of a theorem, or re-calibration of the synthetic
   library. No constant fitted to target data.

Stop: S1 Gamma < eps*L | S2 no gated action for k rounds | S3 budget.
Hard ceilings B <= L/q and q >= 1 => B <= L; violation is a bug, halt.

## Milestones

| M | scope | acceptance | weeks |
|---|---|---|---|
| M0 | Bregman core (MSE + cross-entropy), CLI in / certificate out | B <= R* in 100/100 synthetic draws; \|E-R*\|/R* < 5%; leak screen fires on a planted summand | 3-4 |
| M1 | Guarded branch (MAE, pinball), 2-D envelope, rung ladder, cluster bootstrap; port Mayo | reproduces 28.85 / 31.12 bit-identically from stored OOF | 3-4 |
| M2 | Certification suite as infrastructure; regrow seeded generators; certify() as a service; real-data half on OpenML/AMLB | 50 designs regenerate bit-identically from seeds; certify() < 10 min; versioned registry | 3 |
| M3 | Exploration layer: partitions, clusters, Louvain, discovery, per-slice certificates, comparator table, unit translation | stratification identity < 1e-6; min-n gate + multiplicity control; reproduces the 38/42 comparator table | 4 |
| M4 | Metis loop: actions, gate, stop rule, trace (A2/A3/A4 first, A1 last) | over 100 randomised action sequences validity never drops below 50/50; trace replays the Mayo history | 5-6 |
| M5 | NL lane + retrieval: LLM concept extraction, bi/cross-encoder neighbourhoods, case base, method index | unaided, agent reproduces arm-B > arm-C and rejects broad text SVD with a recorded reason | 5-6 |
| M6 | Coverage guarantee + declaration product + re-certification cadence | empirical coverage >= 1-alpha on held-out library designs | 4-6 |

**M2 blocks M4.** An agent whose gate cannot re-validate is an agent with no gate.

## Research track (parallel)

| R | question | unblocks | risk |
|---|---|---|---|
| R1 | estimate lambda (fold-error correlation) from observables, paired with lambda*(q) = 1 - 1/(2q^2) | automatic rung; kills the 4.67-min Mayo ambiguity | low |
| R2 | theorem bounding q(rho, tau) - proved majorant replacing the measured envelope | makes R3 a statement about a proved object | HIGH - plan the fallback |
| R3 | finite-sample distribution-free coverage P(B <= R*) >= 1-alpha, inside conformal risk control | the headline claim | medium |
| R4 | Neyman orthogonality for the bound functional | answers "your bound depends on which models you fit" | medium |
| R5 | Metis no-invalidation theorem + regret bound | the agent claim | medium |

R2 acceptable outcomes: full majorant | majorant on a stated subclass (Mayo sits at
rho in [0.60, 0.79], tau in [0.21, 0.30] in every cell) | a local bound. Gate at week 6.

Early R1 experiment: measure lambda for TabPFN folds vs independently-trained GBM folds.
Prior-fitted "folds" may be far more correlated - a publishable caution if so.

## Validation

- **L1 synthetic ground truth** - validity rate must be N/N; usable tightness; estimator bias.
  Current baselines: 50/50 valid, usable tightness 0.821; estimator pooled \|rho-1\| 0.0601,
  41/50 paired better, p = 5.7e-5.
- **L2 comparator harness** - matched pairs, Rice/HKT difference-based variance, KNIFE-P /
  LMC-P, Ishida, augmented van Trees, CV label-noise. Win/loss per slice.
- **L3 real-data regression** - Mayo (frozen OOF) + OpenML/AMLB tasks.
- **L4 prospective** - frozen configuration, pre-registered declaration, untouched window.

Statistical hygiene enforced in code: exact one-sided Wilcoxon with `method='exact'`
(never the deprecated `mode=`), exact sign test alongside, ties dropped, entity-clustered
bootstrap.

## Client track (starts week 1)

C0 freeze and pre-register -> C1 evidenced information-set statement (includes the schedule
ablation: +0.39 min tabular+notes, +0.22 min full text; slot diagnostics clean) ->
C2 per-family declaration with unit translation -> C3 decision rule in OR hours and dollars ->
C4 re-certification cadence.

## Risks

| risk | trigger | response |
|---|---|---|
| R2 does not close | week-6 gate | fallback tier; R3 proceeds with the measured envelope and says so |
| lambda demotes Mayo to Qcap | R1 completes | floor drops to ~24.18, headroom widens to ~7 min - decide the narrative in advance |
| prospective window disagrees | C0 | halt and debug; the cheap failure, wanted early |
| base learners are the binding constraint | L >> deployed model | R4 orthogonality; also a legitimate finding |
| agent finds nothing | M4 | expected and acceptable - the guarantee is that it cannot invalidate |
| suite cannot be regrown | M2 | project-stopping for M4; surface immediately |

## Shortest paths

- **credible demo**: M0 alone - the Bregman branch is exact and needs none of the open theory (~4 weeks)
- **better Mayo declaration**: M1 -> R1 -> R3 -> C2, skipping R2 via the fallback (~3 months)
