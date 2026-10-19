# Pythia v5: confirmed results on 10 known-noise tabular + free-text designs

## Bottom line
- **Training-free Pythia-v5 estimator.** Text is read by an LLM, followed by detrended near-replicate tuples, SIMEX and the optimism–pessimism combination. It is the most accurate estimator of attainable risk for **every loss** against the accepted literature methods, on 200 fresh draws:
  - MSE 0.047 vs 0.159 (Gamma test);
  - MAE 0.018 vs 0.076 (learning curve);
  - Q90 0.037 vs 0.110;
  - W30 0.069 vs 0.100;
  - W60 0.28 vs 0.47.
- **Same LLM features given to the rivals.** It still wins on every loss (all p ≤ 0.0014). So the gain comes from the backbone, not only from the text reader.
- **Against our own earlier trained RTX-LC.** It wins on MSE, MAE, Q90 and W30, and ties on W60.
- **Pythia-v5 lower bound.** It is markedly tighter than every earlier bound:
  - MSE 0.84 vs 0.71;
  - MAE 0.85 vs 0.76;
  - Q90 0.85 vs 0.73;
  - W30 0.64 vs 0.55;
  - W60 0.22 vs 0.06.

  Coverage was 98.5–100% at a nominal one-sided 95% level. The earlier RTX-LCB covered 100% but was looser.

## Protocol
- **Designs.** 10 designs, n = 2,000 each. Signal and noise laws are known exactly (Laplace/Gamma pair twins, lognormal, t3, t5, mixtures, sparse 30-D, text-only signal, text-driven heteroscedasticity, discrete replicates).
- **Notes.** The notes are new and hard: paraphrases, negations, typos, and distractors such as "complex medication list" or "redo of labs". Every concept is stated in each note, so gold is exact.
- **LLM reader.** Claude subagents read the notes, outcome-blind:
  - whole-note reading matched the cached sentence-level reading on 300/300 notes, with 100% accuracy;
  - keyword rules scored 58% (complexity), 52% (redo) and 93% (access).
- **Freeze, then confirm.** Development used seeds 100–103 (40 draws). The code was then frozen (`FROZEN_v5.sha256`) and confirmed on untouched seeds 9100–9119 (200 draws). The frozen estimator reproduces the stored confirmation values bit-for-bit.
- **Rivals.** Starred rivals (*) received an oracle choice among three text representations, made draw by draw against gold. That is generous to them.
- **Independent check.** A separate subagent recomputed every headline number from the raw file with fresh code; all matched.

## 1. Estimators: median |relative error|, 200 draws
| Method | MSE | RMSE | MAE | Q90 | W30 | W60 |
|---|---|---|---|---|---|---|
| **Pythia-v5 (new, training-free)** | **0.047** | **0.024** | **0.018** | **0.037** | **0.069** | 0.282 |
| RTX-LC (ours, earlier, trained GBMs) | 0.078 | 0.038 | 0.032 | 0.058 | 0.107 | **0.269** |
| Gamma test* (+LLM features) | 0.159 (0.106) | 0.079 (0.053) | – | – | – | – |
| Cortes learning curve (+LLM) | 0.256 (0.238) | 0.137 (0.127) | 0.076 (0.071) | 0.110 (0.118) | 0.100 (0.117) | 0.468 (0.438) |
| Cross-fitted GBM plug-in (+LLM) | 0.185 (0.156) | 0.088 (0.075) | 0.093 (0.071) | 0.187 (0.169) | 0.295 (0.233) | 0.81 (0.85) |
| Rice / 1-NN* | 0.975 | 0.405 | – | – | – | – |
| kNN-LOO* | 1.013 | 0.419 | 0.535 | 0.566 | 1.74 | 6.90 |
| Anderson–Bjarnadóttir LP* | – | – | 0.795 | – | – | – |
| Pythia-v5 without the text reader | 0.247 | 0.117 | 0.143 | 0.136 | 0.191 | 0.413 |

### Head-to-head against the strongest accepted rival for each loss
| Loss | Rival | Pythia-v5 wins (draws) | Sign test p | Designs won vs best-of-all rivals |
|---|---|---|---|---|
| MSE / RMSE | Gamma test* | 146/200 | 6e-11 | 7/10 |
| MAE | Cortes-LC | 160/200 | 3e-18 | 9/10 |
| Q90 | Cortes-LC | 165/200 | 2e-21 | 9/10 |
| W30 | Cortes-LC | 125/200 | 5e-4 | 5/10 |
| W60 | Cortes-LC | 127/200 | 2e-4 | 5/10 |

With LLM features given to every rival, Pythia-v5 wins 148, 160, 170, 125 and 123 of 200 draws (all p ≤ 0.0014). Against everything, including our earlier trained RTX-LC, it wins MSE, MAE, Q90 and W30 (p ≤ 5e-4) and ties W60 (102/200).

Caveat: the 200 draws are 20 seeds × 10 designs, so the draw-level p-values overstate independence. The design-level column ("designs won vs best-of-all") is the conservative view: it compares Pythia-v5 with the best rival chosen separately for each design.

## 2. Lower bounds: coverage / median tightness (bound ÷ gold), 200 draws
| Bound | MSE | RMSE | MAE | Q90 | W30 | W60 |
|---|---|---|---|---|---|---|
| **Pythia-v5-LCB (new)** | 0.995 / **0.84** | 0.995 / **0.91** | 0.995 / **0.85** | 0.985 / **0.85** | 1.000 / **0.64** | 0.995 / **0.22** |
| RTX-LCB (ours, earlier) | 1.000 / 0.71 | 1.000 / 0.84 | 1.000 / 0.76 | 1.000 / 0.73 | 1.000 / 0.55 | 1.000 / 0.06 |
| Paired witness* | 0.46 | 0.46 | 0.25 | 1.000 / 0.59 | 0.35 | 0.44 |
| Fano entropy bound* | 0.00 | 0.00 | 0.00 | – | – | – |

Of the 1,000 bound checks, 6 misses fell in designs 9, 5, 10 and 2. Their worst ratio was 1.05, which is within the nominal 5% one-sided error rate.

## 3. Where it is weak
- **Design 9** (heavy t3 noise with a sin(X0·X1) interaction). The MAE error is 0.092 vs 0.036 for the best rival: linear detrending leaves interaction curvature in the tuple. Local-quadratic detrending (tested on dev only) did not help overall.
- **Window losses.** Pythia-v5 wins only 5/10 designs on W30 and W60. W60 gold values are tiny (often below 1%), so relative errors are noisy. RTX-LC ties on W60.
- **Text reader.** Without it, Pythia-v5 degrades to 0.14–0.25 error. The LLM reading is essential on the text designs.
- **Bound validity is empirical.** A distribution-free lower bound is impossible for continuous inputs (THEORY_v4 Thm 1′). Coverage is shown on untouched seeds, not proved.

## 4. What changed in the backbone (details in THEORY_v5.md)
1. **LLM reader turns notes into exact discrete coordinates.** It is outcome-blind, so data processing guarantees no leakage.
2. **Leave-tuple-out local-linear detrending of near-replicate tuples.** This removes first-order location heterogeneity. Raw tuple inflation was +90% to +450%; after detrending it is +3% to +20%.
3. **SIMEX on the induced slope noise.** It is loss-agnostic, so one correction applies to every loss. MSE error fell from 0.133 to 0.062 and MAE from 0.073 to 0.026 on dev.
4. **Optimism–pessimism combination** of in-tuple ERM and leave-one-out risks, with a per-loss weight frozen on dev.
5. **Validity-constrained bound selection.** Among ERM and designed-polynomial witnesses across tuple sizes, maximise tightness subject to 100% dev validity and a 5% safety margin.
6. **Model-free relevance screening.** Columns are added only when they reduce the detrended tuple dispersion. On the sparse 30-D design it recovers the interaction variable that marginal dCor misses.

## Files
- **Code:** `pythia5.py` (estimator and bound), `data5.py` and `notes.py` (designs, hard notes, LLM reader cache).
- **Benchmark:** `bench5.py`, `analyze5.py`.
- **Results:** `confirm5.jsonl` (raw, 200 draws), `confirm5_summary.json`.
- **Write-up:** `THEORY_v5.md`.
- **LLM validation:** `val_llm.jsonl`, `llm_bank.json`.
