# Local risk certificate, v2: split calibration, support partition, contrast relations, transport and monitoring

This note extends `LOCAL_MODEL_RISK_CERTIFICATES_20261018.md`.

- **Code:** `pythia/core/local_model_risk.py` (new options `alpha_ledger`, `relation_test`, and `build_support_partition`) and `pythia/core/local_transport.py`.
- **Tests:** `tests/test_local_risk_extensions.py`.
- **Simulations:** `research/pythia_local_20261018/`.
- **Literature:** `docs/literature/PYTHIA_LOCAL_RISK_LITERATURE_20261018.md`.

## 1. Split calibration (closes the out-of-fold gap)

The fixed-design theorem needs three things:

- (a) losses that are independent across patients given the design;
- (b) a hierarchy fixed without the certified outcomes;
- (c) the certified model to be the deployed model.

Five-fold out-of-fold (OOF) losses violate all three:

- Each fold model is trained on other patients' outcomes. Those patients' own losses come from models trained partly on the first patient.
- Prediction-bin edges are quantiles of OOF predictions.
- The deployed model is a refit on all the data.

As Bates, Hastie and Tibshirani (2024) show, OOF error estimates the average error of models fitted on other training sets, not the error of the deployed model.

**Split protocol.**
1. Partition development patients into a training set and a calibration set. A temporal split is preferred.
2. Fit and freeze the deployed model f on the training set.
3. On the calibration set, compute f's predictions, then the hierarchy and its bin edges from those predictions alone, then the losses L_i = ℓ(Y_i, f(X_i)).
4. Given the training data and the calibration design, the calibration labels are independent across patients. So (a), (b) and (c) hold exactly and the fixed-design theorem applies verbatim.

The cost is fewer certifying patients and a model trained on less data. In the simulation, later MAE rose by about 0.2–0.3 minutes, roughly 1%.

## 2. Ledger over eligible nodes only

Eligibility (`n_entities ≥ min_entities`) is computed from entity identifiers and hierarchy labels only. The tested family is therefore fixed before any outcome is read.

Ineligible nodes are never routed to, so they need no α. They receive the trivial interval [0, M]. Charging only the eligible nodes cut the family from about 115 to about 26 tests per model in the simulation. Coverage is unchanged, by Bonferroni over the eligible family.

## 3. Contrast relations

The overlap rule declares "higher" only if the local lower endpoint exceeds the global upper endpoint. That needs two separate intervals to be disjoint. Instead, Pythia certifies the contrast directly:

  D_c = r_c − r_global = Σ_g E d_g,  d_g = Σ_{i∈g} L_i (1{i∈c}/n_c − 1/n).

Patient clusters are independent. Each d_g lies in the outcome-blind range

  [−(n_g − n_gc)M/n,  n_gc(1/n_c − 1/n)M],

so the heterogeneous-support Chernoff bound applies to the shifted contributions. The direction is certified iff the D_c interval excludes 0. Contrasts are added to the same α ledger, so cell intervals and contrasts remain simultaneous.

## 4. Support partition: the estimand matches the routed cases

Hierarchy routing certifies the *whole* node a case backs off to. The cases that back off, however, are exactly those whose finer cells were sparse, and they can differ from the node average. In simulation this made the routed population's true risk fall outside the node interval in 2–7% of routed cells. The consequence is wrong directions for those cases even when every node interval was valid.

**Partition rule (outcome-blind).**
1. Process the levels from finest to coarsest.
2. A node becomes a *region* when its residual rows (rows not claimed by a finer region) contain at least `min_entities` independent entities.
3. Rows left over go to the global residual.

Certifying the regions makes the certified estimand identical to the population of cases routed there. The regions form a partition, so the family is smaller still.

## 5. Transport under a declared shift budget Λ

Assume a binary loss, and let the later population of a region differ from development by a density ratio w with sup w / inf w ≤ Λ. Then

  odds(r′) ∈ [odds(r)/Λ, Λ·odds(r)].

The bound is sharp: put the high weight on L = 1. This is the marginal sensitivity model of Zhao, Small and Bhattacharya (2019) applied to bounded-loss transport.

**Outputs.**
- `odds_transport`: the transported interval, conditional on Λ.
- `tipping_lambda`: the largest Λ at which a certified direction survives. Report it beside every certified direction.

## 6. Anytime-valid harmful-shift monitor

`BettingRiskMonitor` mixes fixed bets λ_j ∈ (0, 1/u). Its value

  E_t = mean_j Π(1 + λ_j(L_s − u))

is a nonnegative supermartingale under H0: "the conditional mean of every later loss is ≤ u". Ville's inequality then gives P(alarm ever) ≤ α. This follows Podkopaev and Ramdas (2022): it alarms only on *harmful* shift.

**Requirement:** losses that are conditionally independent given the past. Use one case per patient, or patient-level means.

## 7. Claim tiers

| Statement | Tier | Holds when |
|---|---|---|
| Interval and direction on the calibration population (split, partition, contrast) | `conditional` | patient independence and an attested loss cap |
| The same for the later population | `conditional on Λ` | the declared within-region shift budget |
| Monitor alarm | anytime-valid test | conditional independence of later losses given the past |

**Not provided:**
- an individual-case outcome guarantee;
- a Λ estimated from later outcomes, which would be circular;
- transport for non-binary losses. That needs a separate bound, because the sharp odds map is specific to {0,1} losses.
