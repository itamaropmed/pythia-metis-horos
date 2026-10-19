# Local risk certificate v2: pipeline-level simulations (2026-10-18)

- **Code:** `research/pythia_local_20261018/` (`sim_lib.py`, `run_pipeline_validity.py`, `run_shift.py`, `run_min_entities.py`, `run_monitor_power.py`).
- **Theory:** `docs/theory/LOCAL_RISK_V2_SPLIT_PARTITION_TRANSPORT_20261018.md`.
- **Literature:** `docs/literature/PYTHIA_LOCAL_RISK_LITERATURE_20261018.md`.
- **Status:** development suite with a known law. It is not preregistered and not confirmatory.

## Simulator (Mayo-shaped, known conditional law)

- **Patients and cases:** 6,000 development patients with 1–4 cases each (about 8,600 cases). A shared patient frailty makes repeated cases dependent.
- **Covariates:** 8 procedure families, a complexity score, a nuisance variable, and a binary "note concept" correlated with a hidden duration component.
- **Duration:** the family baseline, plus 20·complexity, plus 12·hidden, plus 10·frailty, plus heteroscedastic noise. Every case's true W30-miss probability is therefore known analytically.
- **Models:** two arms, each a gradient-boosted absolute-error model (arm A without the note concept, arm B with it).
- **Hierarchy:** the Mayo hierarchy, family × prediction bin × disagreement bin → family × prediction bin → prediction bin → global, with a 100-patient support threshold.
- **Later cohort and truth:** a later cohort of 1,500 patients. The truth is the deployed model's *population* risk, computed on a fresh sample of 60,000 patients.

## 1. Validity of the pipeline (19 replicates, 2 models each)

| Pipeline variant | Node intervals cover the truth | Truth of the cases actually routed outside the node interval | **Wrong certified directions** (rows; replicates affected) | Certified share of later cases | Charged tests per model |
|---|---|---|---|---|---|
| **Current Mayo replay**: 5-fold OOF, all-node ledger, overlap rule | 19/19 | 160/2,046 cells | **468 of 8,362 (5.6%); 19/19 replicates** | 0.102 | ~115 |
| OOF, eligible-node ledger, contrast rule | 19/19 | 160/2,046 | 753 of 12,650; 19/19 | 0.154 | — |
| Split calibration, all-node ledger, overlap rule | 19/19 | 65/912 | 0 of 2,167 | 0.026 | 114 |
| Split, eligible-node ledger, overlap rule | 19/19 | 78/912 | 0 of 4,819 | 0.059 | 26 |
| Split, eligible-node ledger, contrast rule | 19/19 | 70/912 | 367 of 10,526; 5/19 replicates | 0.128 | 51 |
| Split, **support partition**, overlap rule | 19/19 | 8/828 (global fallback only) | 0 of 2,709 | 0.033 | 22 |
| **Split, support partition, contrast rule (recommended)** | **19/19** | **7/828 (global fallback only)** | **0 of 6,025; 0/19** | **0.073** | 43 |

Later MAE was 23.0 / 22.3 minutes with the OOF refit and 23.2 / 22.6 minutes with the split model. The split costs about 0.2 minutes, roughly 1%.

**Reading**

1. **The node intervals themselves were never wrong**, in any variant.
2. **The routed cases were often wrong.** A case that backs off to a coarse node gets that node's interval, but the cases that back off are exactly the sparse subpopulations. Their true risk fell outside the node interval in 8% of routed cells under the current pipeline.
3. **The current Mayo replay produced wrong certified directions in every replicate**, about 5.6% of certified rows. Two separate errors combine here:
   - the routed-population mismatch in point 2;
   - certifying OOF fold models while deploying a refit, with hierarchy edges taken from OOF predictions.
4. **Split calibration removes the model mismatch, and the support partition removes the backoff mismatch.** Together they give **zero wrong directions**.
5. **Combined with the eligible-node ledger and the direct contrast test, they keep 2.8× the certified share of a naive split** (0.073 against 0.026).

## 2. Resolution vs the support threshold (recommended route, 12 replicates)

| min_entities | Regions | Node misses | Wrong directions | Certified share | Median width |
|---|---|---|---|---|---|
| 50 | 42 | 0/1,034 | 0 | 0.007 | 0.396 |
| 100 | 21 | 0/540 | 0 | 0.064 | 0.275 |
| 200 | 11 | 0/282 | 0 | 0.169 | 0.206 |
| 400 | 5 | 0/122 | 0 | **0.458** | 0.134 |

Resolution trades against specificity, and the support threshold is the main lever. It is outcome-blind, so it can be **planned before outcomes are seen**: choose the smallest threshold whose worst-case interval half-width (a function of the threshold, α and the number of regions only) is at most a declared tolerance.

## 3. Temporal shift (30 replicates, recommended route)

| Later scenario | Later true risk inside the development interval: Λ = 1 | Λ = 1.25 | Λ = 1.5 | Wrong certified directions | Harmful cells (true risk above the certified upper bound) |
|---|---|---|---|---|---|
| None | 1.000 | 1.000 | 1.000 | 0 | 0 |
| Case-mix shift between families | 1.000 | 1.000 | 1.000 | 0 | 0 |
| Noise +15% | 0.999 | 1.000 | 1.000 | 0 | 2 |
| **Noise +40%** | **0.883** | 0.993 | 1.000 | 0 | 156 |
| Long cases +30 minutes | 0.974 | 0.997 | 0.999 | 0 | 35 |
| Complexity drift (+1 SD) | 0.995 | 1.000 | 1.000 | 0 | 0 |
| All cases +15 minutes | 0.972 | 0.999 | 1.000 | 0 | 38 |

**Shift findings**
- Shift *between* cells is harmless, as expected.
- Shift *within* cells breaks development coverage (down to 88%). A declared Λ = 1.25–1.5 restores it.
- Directions were never wrong, because local and global drifted together.
- **Certified directions are fragile under localized drift.** The median tipping Λ\* was about 1.09: a direction survives only about 9% localized odds drift. **Report Λ\* with every direction.**

**Anytime-valid harmful-shift monitor** (1,000 streams per cell, certified upper bound 0.40)

| True risk − certified upper bound | N = 100 | N = 500 | N = 2,000 |
|---|---|---|---|
| 0 (null boundary) | 0.031 | 0.035 | 0.031 |
| −0.05 | 0.010 | 0.016 | 0.011 |
| +0.05 | 0.106 | 0.318 | 0.917 |
| +0.10 | 0.302 | 0.941 | 1.000 |

- False alarms stayed at or below 3.5% at any time, consistent with Ville's bound.
- Detection is reliable for a harm of about 0.10 within roughly 500 patients per region.
- Within a single later cohort (30–100 patients per region) it caught only 6 of 156 harmful cells. It is a deployment tool, not a one-shot check.

## Recommendations for Pythia

1. **Use split calibration for local-risk claims.** The current OOF-based Mayo replay should be relabelled exploratory or re-run.
2. **Make the support partition the default routing.** The certified estimand must be the population the case is routed into.
3. **Use the eligible-node ledger and contrast relations.** They are valid, and together give about 2–3× more resolved cases.
4. **Plan the support threshold before outcomes,** from a width tolerance.
5. **Ship Λ-transport and Λ\* in the case card,** and run one monitor per region in deployment.
6. **Open:**
   - a pooled or average-coverage tier, as in Armstrong, Kolesár and Plagborg-Møller;
   - top-down hierarchical testing, as in Meinshausen;
   - a variance-adaptive Chernoff bound for small risks;
   - transport for non-binary losses.
