# Real tabular and clinical-text tests (2026-10-15, development/exploratory)

Free sources (no credentialing): MTSamples medical transcriptions (4,966 notes, 40 specialties; GitHub copy of
the public mtsamples.com corpus) and PMLB clinical Bayesian-network datasets (breastTumor, pharynx, lowbwt;
iid samples from networks fitted to real clinical data).  HuggingFace/OpenML/UCI are blocked from this sandbox.

## 1. Real labels, exact textual replicates (MTSamples, 0-1 loss, specialty)
- 96% of rows belong to a group of byte-identical transcriptions (each note is listed under several specialties).
- Exact floor of "specialty given transcription text" on this corpus: **0.525** — no function of the text can be
  right on more than 47.5% of rows (finite-population fact, no assumptions).  Majority-class error 0.781.
- Leakage: adding the `keywords` field drops the floor to **0.124**; the label string appears in 77% of keyword
  fields -> flagged as target-derived.  Description and sample name do not change the floor (0.525).

## 2. Plasmode on real notes (known floor): y = g(text) + noise
| Route | Signal only from dictionary concepts | Signal includes an out-of-dictionary channel ('smok') |
|---|---|---|
| Generic char-4gram Jaccard metric (Pythia or NN pairs) | 0 (no near-replicates; even oracle penalty gives 0.36 of floor) | 0 |
| Concept-dictionary metric, declared per-concept moduli: PYTHIA metric domination | **0.67-0.70** of floor (5/5) | 0 |
| same metric, accepted NN-pair bound | 0.63-0.65 | 0 |
| **Exact textual replicates (finite cells, no metric, no smoothness)** | — | **0.60-0.66** of floor, 0 violations (10/10) |

## 3. Real clinical tabular (PMLB BN samples, 10k certification rows, MAE)
| Dataset | Floor: 1 / 2 / 4 / all categorical features | Model (all features incl. continuous) held-out MAE / upper | Certified value of all info beyond 1st feature |
|---|---|---|---|
| pharynx (survival) | 293.4 / 291.3 / 283.4 / 99.3 | 216.9 / 223.6 | **69.8** |
| lowbwt (birth weight) | 379.6 / 371.8 / 350.2 / 322.9 | 357.8 / 367.1 | **12.5** |
| breastTumor (size) | 7.66 / 7.25 / 6.77 / 4.83 | 7.23 / 7.44 | 0.22 |

## Reading
- Generic embedding-free text metrics do not create near-replicates; every metric-based route returns 0.
  Raw text needs either exact replicates or a registered representation whose sufficiency is declared.
- Exact or near-exact duplication (templating, copy-forward, multi-listing) is a natural repeated-measurement
  resource in clinical text and gives assumption-light raw-text floors.
- Proposed definition to develop next: information relative to a declared nuisance group G (canonicalisations
  that cannot change the outcome law): R*(T) = R*(T/G), orbits are exact replicates.
