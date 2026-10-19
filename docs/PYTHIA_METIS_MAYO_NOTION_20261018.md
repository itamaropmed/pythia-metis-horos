# Pythia and Metis on Mayo HRS

## First development step, proof of concept, and research record

**Documentation date:** 18 October 2026  
**Development stage:** first integrated proof of concept  
**Owner:** Opmed.ai  
**By:** Itamar Zernitsky — Opmed's shadow intern, like Robin to Batman or Green Lantern to Superman

---

## Executive summary

This document records the first integrated development step of Pythia and
Metis on the Mayo HRS procedure-duration problem.

The system is divided into three layers:

- **Pythia** measures attainable risk and produces training-free global and
  local certificates from frozen evidence.
- **Metis** assigns a numerical and categorical risk assessment to each new
  case for each frozen prediction model.
- **Horos** performs data preparation, model training, cross-fitting, and
  generation of the frozen evidence consumed by Pythia and Metis.

The main Mayo proof of concept compared a basic tabular XGBoost duration model
with the same design using 11 LLM-derived clinical-note concepts. The temporal
replay used 9,966 development cases from 8,231 patients and 2,599 later cases
from 2,301 new patients, with zero patient overlap. Every later case was scored
before its actual duration was revealed.

The first Metis risk map achieved Spearman correlations of 0.4385 and 0.4273
between assigned risk and later absolute error. The second development version
improved these to 0.4673 and 0.4618. The risk tiers separated cases from roughly
14 minutes of observed MAE in the low-risk group to roughly 68--70 minutes in
the very-high-risk group.

Pythia was then extended from a global information-floor system to a local
frozen-model W30-risk certificate. It assigned a certified higher- or
lower-than-global risk direction to 1,150 basic-model cases and 1,120 +11-LLM
cases. Remaining cases were labeled `local_shift_unresolved`.

This is the first practical Pythia–Metis POC: global information measurement,
per-case risk prediction, support-aware local certification, patient-clustered
validation, target-injection checks, note perturbations, and multi-loss Mayo
research are connected in one code path.

---

## 1. Mayo HRS objective

The Mayo HRS task predicts procedure duration in minutes before the procedure
begins. The target is `case_actual_duration_time`, and the primary operational
loss is `MAE = mean(|actual duration - predicted duration|)`.

The first system development step addresses four questions:

1. What attainable-error floor is supported by the declared information set?
2. How much room remains between a frozen model and that floor?
3. How difficult does an arriving case appear for each frozen model?
4. Is the local W30 error risk higher or lower than the model's global risk?

---

## 2. System design

### 2.1 Pythia

Pythia is the mathematical measurement and certification layer. It does not
train prediction models. Its inputs include a declared information set and
loss, honest frozen predictions or witnesses, patient identifiers, a finite
outcome-blind representation or hierarchy, and timing, provenance, support,
and leakage declarations.

Pythia can return an attainable-risk bound and estimate, achieved loss and
headroom, global or local risk intervals, a local relation to global risk,
provenance, and a structured refusal. It uses the tiers `proved`,
`conditional`, `calibrated`, `exploratory`, and `refused`.

### 2.2 Metis

For every frozen prediction model, Metis returns the duration prediction,
expected absolute error, W30-miss probability, a `low`/`moderate`/`high`/
`very_high` tier, routing support, the connected Pythia information floor, and
the local W30 relation when available.

Metis v1 uses a frozen hierarchical OOF risk map. Metis v2 uses a
development-selected Horos risk forecaster with development-only cross-fitting
and calibration.

### 2.3 Horos

Horos owns dataset assembly, pre-cutoff feature construction, patient-grouped
cross-fitting, model training, held-out prediction, and construction of frozen
inputs for Pythia and Metis.

### 2.4 Information flow

```text
tabular data + eligible pre-cutoff clinical-note information
                              |
                              v
                  Horos duration models
                              |
             development OOF predictions and losses
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
       Pythia information floor      Metis risk forecaster
       and local W30 intervals       and categorical tiers
                 |                          |
                 +------------+-------------+
                              |
                              v
              new-case model and risk card
```

---

## 3. Mayo information sets and reference results

### 3.1 Arm A: tabular

Arm A contains 296 encoded variables derived from permitted patient,
procedure, history, comorbidity, case, and scheduled-time fields. Its preserved
calibrated Pythia MAE lower bound is **29.29 minutes**.

### 3.2 Arm B: tabular plus 11 LLM concepts

Arm B adds 11 concepts extracted from eligible preoperative clinical notes.

| Quantity | Result |
|---|---:|
| Best historical OOF MAE | 32.8700117033 min |
| Global attainable-floor estimate | 31.1186421201 min |
| Stratified calibrated lower bound | **28.8495996043 min** |
| Conservative Q-cap rung | 24.1867259 min |

### 3.3 Foundation clinical-note model

The separate Foundation/PLS/meta-gate system uses tabular data and strict
pre-cutoff note representations without the old 11-feature table.

| Development cases | Later cases | Patient overlap | Held-out MAE |
|---:|---:|---:|---:|
| 10,882 | 2,839 | 0 | **31.556418** |

Its note timing rule is `note_date < scheduled_in_room - 2 days`.

### 3.4 Historical 31.2623 reference

The repository audit traced 31.2623 to an OOF residual-corrector component. The
verified final held-out note-model reference in this document is 31.556418.

---

## 4. Temporal held-out simulation

The 12,993-case A/B cohort was ordered by scheduled procedure time. The outer
boundary was the 80th percentile of scheduled time, and every later-cohort
patient was removed from development.

| Cohort | Cases | Patients |
|---|---:|---:|
| Development | 9,966 | 8,231 |
| Later held-out | 2,599 | 2,301 |
| Patient overlap | 0 | 0 |

The matched models were `basic_xgboost` on Arm A and
`xgboost_plus_11_llm` on Arm B. Both used an absolute-error objective, 400
trees, depth 6, learning rate 0.05, row and column subsampling of 0.8, minimum
child weight 8, and L2 regularization 1.0.

Within development, five patient-grouped folds generated honest OOF
predictions. For every later case, Horos generated predictions, Metis v1 and v2
generated risk cards, Pythia generated a local W30 card, and all outputs were
frozen before the actual duration was revealed.

---

## 5. Duration prediction results

| Model | Pythia information floor | Later MAE | Error over 30 min |
|---|---:|---:|---:|
| Basic XGBoost | 29.29 calibrated | 32.2312 | 38.17% |
| XGBoost + 11 LLM concepts | 28.85 calibrated | **31.9184** | 37.86% |

The +11 model improved later MAE by 0.3128 minutes and produced the smaller
individual error on 50.83% of cases.

---

## 6. Metis v1: hierarchical case-risk map

Metis v1 used procedure family, prediction bin, and disagreement bin. Cells
with fewer than 100 independent development patients backed off through
family-plus-prediction, prediction, and global levels.

| Model | Risk/error Spearman | Mean predicted MAE | Observed MAE | Predicted W30 | Observed W30 |
|---|---:|---:|---:|---:|---:|
| Basic | 0.4385 | 32.4865 | 32.2312 | 37.14% | 38.17% |
| +11 LLM | 0.4273 | 32.0485 | 31.9184 | 36.54% | 37.86% |

| Model | Tier | Cases | Predicted MAE | Observed MAE | Observed W30 |
|---|---|---:|---:|---:|---:|
| Basic | Low | 551 | 15.29 | 14.97 | 12.3% |
| Basic | Moderate | 1,101 | 27.31 | 26.65 | 33.4% |
| Basic | High | 719 | 43.43 | 45.12 | 56.3% |
| Basic | Very high | 228 | 64.54 | 60.29 | 66.2% |
| +11 LLM | Low | 572 | 15.32 | 15.56 | 12.1% |
| +11 LLM | Moderate | 1,190 | 27.90 | 27.50 | 34.8% |
| +11 LLM | High | 603 | 44.11 | 44.87 | 55.9% |
| +11 LLM | Very high | 234 | 62.94 | 60.99 | 70.1% |

---

## 7. Metis v2: development-selected risk forecasting

Metis v2 evaluated three candidate families using development OOF errors.
Selection and isotonic calibration were completed before later outcomes were
read. The risk representation used 34 arrival-time features: model predictions,
disagreement, safe tabular context, 11 LLM concepts, and fixed procedure-family
encoding.

| Model | Spearman | Risk MAE | W30 Brier | Mean risk bias |
|---|---:|---:|---:|---:|
| Basic | **0.4673** | 20.7125 | 0.19347 | 0.3979 min |
| +11 LLM | **0.4618** | 20.5118 | 0.19374 | 0.3847 min |

| Model | Tier | Cases | Predicted MAE | Observed MAE | Observed W30 |
|---|---|---:|---:|---:|---:|
| Basic | Low | 571 | 14.80 | 13.88 | 10.9% |
| Basic | Moderate | 1,022 | 26.82 | 26.53 | 32.9% |
| Basic | High | 749 | 42.28 | 41.83 | 54.6% |
| Basic | Very high | 257 | 67.23 | 67.67 | 72.0% |
| +11 LLM | Low | 555 | 14.63 | 14.02 | 9.9% |
| +11 LLM | Moderate | 992 | 26.21 | 25.44 | 31.5% |
| +11 LLM | High | 848 | 43.03 | 42.16 | 54.7% |
| +11 LLM | Very high | 204 | 65.40 | 69.55 | 75.0% |

### Patient-clustered paired comparison with v1

| Model | Measure | V2 − V1 | Patient-bootstrap 95% interval |
|---|---|---:|---:|
| Basic | Spearman | +0.0288 | [0.0147, 0.0440] |
| Basic | Risk MAE | -0.4804 | [-0.7641, -0.2238] |
| Basic | W30 Brier | -0.00630 | [-0.00884, -0.00374] |
| +11 LLM | Spearman | +0.0347 | [0.0200, 0.0492] |
| +11 LLM | Risk MAE | -0.5032 | [-0.7401, -0.2618] |
| +11 LLM | W30 Brier | -0.00498 | [-0.00760, -0.00240] |

All three changes improved in each of the 1,000 patient-clustered bootstrap
comparisons for both models.

---

## 8. Pythia local frozen-model W30 certificate

The local route uses the naturally bounded loss
`1{|actual duration - prediction| > 30 minutes}`. It constructs simultaneous
patient-clustered intervals for every frozen model and cell of an outcome-blind
hierarchy. A new case is routed to the finest cell with at least 100 independent
development patients.

Its categorical outputs are `certified_higher_than_global`,
`certified_lower_than_global`, `local_shift_unresolved`, `global_only`, and
`refused`.

| Measure | Basic | +11 LLM |
|---|---:|---:|
| Later observed W30 miss | 38.17% | 37.86% |
| Certified higher/lower cases | 1,150 | 1,120 |
| Unresolved local-shift cases | 1,449 | 1,479 |
| Median interval width | 0.2391 | 0.2354 |
| Finest-cell routes | 1,986 | 1,986 |
| Global-only routes | 0 | 0 |

Among certified-direction cells with at least 20 later cases, 20/21 directions
agreed with the later-vs-global ordering for each model. Among 20 cells with at
least 50 later cases, the later rate was inside the development interval in
19 basic-model cells and 18 +11-LLM cells.

---

## 9. Mayo research and experiment ledger

### 9.1 Bound, dependence, and estimator research

| Run | Scale | Result |
|---|---:|---|
| Historical guarded Arm B | Mayo OOF + 50-design guard | 28.8496 bound; 31.1186 estimate; 24.1867 conservative rung |
| Independent guarded-MAE gate | 500 designs | 491/491 in-domain valid; 9 out-of-domain refused; Wilson lower 0.9922 |
| Range-dependence guard | 1,800 simulations | 1,800/1,800 valid; 1,797 tighter than Hoeffding |
| Corrected near-replicate route | 3,600 simulations | 3,600/3,600 valid |
| Global fixed-design pairs | 1,500 runs | 1,500/1,500 valid; 1,356 blocked-pair wins |

### 9.2 Target-injection tests

The actual duration was injected in memory into a note sentence and a tabular
oracle channel. No modified patient-derived file was written.

| Injected cases | Diagnostic MAE |
|---:|---:|
| 0% | 32.1905 min |
| 10% | 29.0118 min |
| 50% | 16.8501 min |
| 100% | 0.0000 min |

A second 4,096-patient draw reproduced 32.2685, 28.9918, 16.3180, and 0.0000
minutes. Full injection drove the pair center and lower endpoint to zero, and
the public API identified the target-derived representation and refused it.

The synthetic localized-disclosure run covered 10 duration mechanisms, five
fresh seeds, five disclosure levels, and eight losses. All 2,000 endpoints were
valid, every curve was monotone, and complete disclosure reached zero.

### 9.3 Clinical-note perturbation battery

The fixed Mayo cohort contained 256 earliest-per-patient cases, including 216
with nonempty strictly recut note bundles.

| Note control | Changed rows | q90 endpoint | W30 endpoint |
|---|---:|---:|---:|
| Intact | 0 | 1.6756 | 0.2574 |
| 50% case dropout | 104 | 1.3774 | 0.2574 |
| 100% case dropout | 216 | 1.7813 | 0.2261 |
| Character shuffle | 216 | 1.4682 | 0.2574 |
| Token shuffle | 216 | 2.0465 | 0.2418 |
| 50% span mask | 216 | 1.2375 | 0.2457 |
| 50% segment dropout | 214 | 1.6496 | 0.2418 |
| 50% segment duplication | 216 | 1.7942 | 0.2183 |

The registered Mayo bag parser was exactly invariant to token-order shuffling;
the matching synthetic invariance test passed in 500/500 cases.

### 9.4 Feature and representation tests

- Duplicating scheduled duration 20 times moved the uncanonicalized geometry
  readout from 22.2148 to 20.2881.
- Equality-class closure restored no-new-information invariance in 500/500
  synthetic tests.
- An independent random identifier drove median bound/gold from 0.525 to zero;
  equality-lattice closure restored it to 0.733.
- The finite-text route accepted 4,000 outcome-blind rows with zero gold
  exceedances and refused all 1,000 target-derived rows.

### 9.5 Patient dependence and family neighborhoods

| Mayo run | Cohort | Result |
|---|---:|---|
| Unique-patient lattice | 9,300 patients | Note-metadata permutation tail 0.004; lexical increment tail 0.744 |
| Repeated-patient lattice | 11,356 cases / 9,300 patients | Cluster correction and hierarchy backoff exercised |
| Dominated finite-cell lattice | 14,195 cases / 11,447 patients | Stable endpoints across MAE, q90, and W30 |

| Representation | MAE | q90 pinball | W30 miss |
|---|---:|---:|---:|
| Global | 70.05 | 18.12 | 0.676 |
| Procedure family | 51.95 | 13.70 | 0.551 |
| Procedure | 37.45 | 9.93 | 0.408 |
| Procedure + note metadata | 36.71 | 9.64 | 0.395 |
| Procedure + hashed bag | 36.14 | 9.36 | 0.384 |

### 9.6 Multi-loss Mayo runs

The router exercised MAE, MSE, q10, q50, q90, and W15/W30/W60 losses.

| Loss | Lower endpoint |
|---|---:|
| q10 pinball | 2.9436 loss-minutes |
| q90 pinball | 5.2664 loss-minutes |
| W15 miss | 0.4220 |
| W30 miss | 0.2678 |
| W60 miss | 0.1049 |

The W30 readout represented a 73.22% attainable within-30-minute ceiling,
compared with 62.60% achieved by the frozen OOF model in that run.

### 9.7 Fixed-policy note-value bridge

The Foundation system was evaluated as a frozen policy. The dominated-SAA
family-map route produced a joint-ledger value lower endpoint of 10.760 minutes
beyond an eight-group procedure-family map. The procedure-level bridge returned
zero. This connected an attainable-information certificate with a frozen note
policy.

---

## 10. Synthetic and software validation

### 10.1 Local-risk simulation

The local W30 certificate used 400 independent entities, two perfectly
dependent rows per entity, two frozen models, and known cell risks.

| Result | Value |
|---|---:|
| Monte Carlo repetitions | 100 |
| Simultaneous family coverage | 100/100 |
| False certified directions | 0 |
| Certified higher routes | 103 |
| Certified lower routes | 150 |
| Unresolved routes | 147 |
| Median interval width | 0.1782 |

### 10.2 Broader theorem tests

| Test family | Scale | Result |
|---|---:|---|
| Fixed-action dominated SAA | 500 × 3 losses + 2,400 stress | All primary valid; zero stress violations; MAE median tightness 0.900 |
| Support-free cluster block SAA | 500 clustered draws | 500/500 valid and positive; median tightness 0.7865 |
| Clustered finite-cell | 1,000 laws + 1,000 dependence checks | All cluster endpoints valid |
| Representation lattice | 500 lattices / 2,500 nodes | Zero exceedances and order failures |
| Fixed-policy value | 500 runs | 500/500 valid, positive, and baseline wins |
| Curvature correction | 400 rows | Zero exceedances; improved MAE and q90 |
| Action-dispersion calibration | 1,000 rows | All four targets covered in every row |
| Heterogeneous-support Chernoff | 10,000 untouched draws | 99.71% coverage; 9,979 Hoeffding wins |

### 10.3 Software tests

The latest focused suite passes **14/14** tests covering local routing,
simultaneous relation labels, repeated-patient supports, singleton backoff,
loss-cap checks, refusal paths, threshold-free resolution diagnostics, Horos
risk forecasting, Metis output contracts, and architecture separation.

An earlier integrated run recorded 40 targeted passing tests across the Pythia
API, table/text certificates, residual information, heterogeneous
concentration, Metis case risk, and architecture boundaries.

---

## 11. POC output contract

```json
{
  "anonymous_case_id": "salted-local-pseudonym",
  "models": [
    {
      "model": "xgboost_plus_11_llm",
      "duration_prediction_minutes": 180.0,
      "expected_absolute_error_minutes": 32.3,
      "probability_error_over_30": 0.37,
      "metis_risk_tier": "moderate",
      "support_level": "family_prediction_disagreement",
      "support_entities": 214,
      "pythia_information_floor_minutes": 28.85,
      "pythia_local_w30_estimate": 0.35,
      "pythia_local_w30_interval": [0.24, 0.47],
      "pythia_local_relation": "local_shift_unresolved"
    }
  ]
}
```

The values illustrate the schema and are not a patient record.

---

## 12. Privacy and artifacts

- Patient-derived rows and note text remain outside Git.
- Tracked reports contain aggregate values only.
- Example case IDs use salted SHA-256 pseudonyms.
- The salt remains outside the repository.

Local result directories:

- `/Users/itamarzernitsky/Documents/clinical notes/output/metis_mayo_case_risk_v1/`
- `/Users/itamarzernitsky/Documents/clinical notes/output/metis_mayo_case_risk_v2/`
- `/Users/itamarzernitsky/Documents/clinical notes/output/pythia_mayo_local_w30/`

---

## 13. Next development steps

1. **Freeze POC v1 artifacts.** Record environment, cohort, feature, model,
   registry, hierarchy, and result hashes in one immutable manifest.
2. **Unify the case card.** Return Metis expected error, W30 probability, risk
   tier, Pythia global floor, local interval, relation, support, and provenance
   through one public API.
3. **Develop hierarchical pooling.** Add predeclared neighboring-cell pooling
   and an explicit variation penalty while preserving the alpha ledger.
4. **Add support and shift scoring.** Measure procedure support, note
   availability, feature-range support, temporal drift, and representation
   drift for each arriving case.
5. **Extend certified losses.** Apply the local hierarchy to W15, W60, pinball
   losses, and bounded or truncated scheduling costs.
6. **Connect raw-note residual information.** Condition text-derived
   information on the tabular representation and measure its additional signal.
7. **Expand arrival-time risk evidence.** Evaluate diverse honest model
   predictions, note communities, disagreement, and representation support.
8. **Build the Metis registry.** Version risk maps, Pythia registries, model
   cards, alpha ledgers, guards, and pseudonymization policy.
9. **Add the Metis action loop.** Let Metis propose an update, Horos generate
   frozen OOF evidence, and Pythia accept, downgrade, or refuse its claim.
10. **Freeze the next temporal evaluation.** Lock models, representations,
    endpoints, and analysis before scoring the next Mayo cohort.
11. **Run prospective confirmation.** Evaluate MAE, W30, tier ordering, local
    certificate consistency, family performance, calibration, drift, and OR
    hours.
12. **Prepare the POC paper package.** Present the system separation, local
    theorem, synthetic coverage, Mayo replay, target injection, perturbations,
    and prospective protocol.

---

## 14. Reproduction map

Core code:

- `pythia/core/local_model_risk.py`
- `pythia/core/risk_resolution.py`
- `metis/case_risk.py`
- `horos/models/case_risk.py`

Mayo runs:

- `scripts/run_mayo_metis_case_risk.py`
- `scripts/run_mayo_metis_case_risk_v2.py`
- `scripts/run_mayo_pythia_local_w30.py`
- `scripts/run_mayo_oracle_leakage.py`
- `scripts/run_mayo_text_augmentation.py`
- `scripts/run_mayo_clustered_family.py`
- `scripts/run_mayo_family_lattice.py`
- `scripts/run_mayo_fixed_policy_value.py`

Tests:

- `scripts/run_pythia_local_risk_validation.py`
- `tests/test_local_model_risk.py`
- `tests/test_risk_resolution.py`
- `tests/test_metis_case_risk.py`
- `tests/test_horos_case_risk.py`
- `tests/test_architecture.py`

Reports:

- `docs/experiments/METIS_MAYO_CASE_RISK_20261018.md`
- `docs/experiments/PYTHIA_LOCAL_RISK_VALIDATION_20261018.md`
- `docs/experiments/BOUND_BEHAVIOR_CATALOG_20261015.md`
- `docs/theory/LOCAL_MODEL_RISK_CERTIFICATES_20261018.md`
- `docs/audits/FOUNDATION_31_5564_POLICY_AUDIT_20261014.md`
- `docs/audits/INCUMBENT_AUDIT.md`
- `docs/results/mayo_frozen_ci_anchor.json`

---

## 15. POC conclusion

The first integrated development step connects three questions:

- Pythia measures what the declared information supports globally.
- Metis estimates how difficult an arriving case is for each frozen model.
- Pythia certifies whether supported local W30 risk is detectably higher or
  lower than the model's global W30 risk.

On Mayo, the system preserves the historical 28.85-minute Arm B floor,
reproduces the two-model temporal replay, improves held-out risk association to
approximately 0.46--0.47, separates low- and very-high-risk groups by roughly
54--56 minutes of observed MAE, and produces local certified risk directions
for approximately 43%--44% of later cases.

This document is the baseline POC record for the next Pythia and Metis
development cycle.
