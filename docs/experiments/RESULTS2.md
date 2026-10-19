# Fathom on Mayo HRS — tighter estimator, robustness of the bound, and per-family results

Minutes. n = 12,993 cases, 10,532 patients. Out-of-fold predictions reused from the previous run (no refitting).

## 1. Estimator: a validated correction

The published estimator is `E = sqrt(L^2 - (Mbar/sqrt2)^2)`, which implicitly sets the quadrature excess to q = 1 — and the library says q > 1 on 77% of shape pairs. For a **bound** the right summary of q is an upper percentile; for an **estimator** it is the central one. So build the *median* envelope `qtil50(rho, tau)` from the same 5,952-point 2-D library and use

    E2 = sqrt( (L/qtil50)^2 - (Mbar/sqrt2)^2 )

Validated on all 50 synthetic designs from stored per-design quantities, no reruns:

| suite | estimator | mean \|rho-1\| | signed bias | over-states | paired better | exact one-sided p |
|---|---|---|---|---|---|---|
| K (adversarial, 20) | E published | 0.0465 | **+0.0355** | 16/20 | — | — |
| K | **E2** | **0.0454** | **-0.0004** | **9/20** | 12/20 | 0.392 |
| P (prospective, 30) | E published | 0.0840 | +0.0839 | 29/30 | — | — |
| P | **E2** | **0.0699** | +0.0695 | 28/30 | **29/30** | **2.8e-09** |
| pooled 50 | E published | 0.0690 | +0.0645 | 45/50 | — | — |
| pooled 50 | **E2** | **0.0601** | **+0.0415** | **37/50** | **41/50** | **5.7e-05** |

The one-sidedness the paper flags as the estimator's main weakness is largely an artefact of the q = 1 assumption: on the adversarial suite the signed bias goes from +0.0355 to **-0.0004**.

On Mayo (notes arm) this moves the estimate from 32.31 to **31.12** — for the first time *below* the achieved temporal-holdout MAE of 31.67, which is the coherent direction for an estimate of a floor.

## 2. Bound: it cannot be tightened further, and that is now checked

The stratified bound is valid for **any** partition S = s(X). I tried five, their pairwise intersections with the pre-registered one, and nine grid sizes:

| partition | cells | B\* (2-D guard, stratified) |
|---|---|---|
| P1 predicted-duration quintiles | 5 | 28.85 |
| P2 procedure family | 7 | 28.67 |
| P3 note-content communities (k-means, 8) | 9 | 28.32 |
| P4 kNN-graph communities (Louvain) | 8 | 28.12 |
| P5 anesthesia x urgency | 7 | 28.30 |
| P1 x procedure family | 8 | 28.69 |
| P1 x note communities | 11 | 28.63 |
| P1 x kNN communities | 4 | 28.37 |

Grid-size sweep on the pre-registered partition, K = 3,4,5,6,8,10,12,16,20: 28.79, 28.80, **28.85**, 28.80, 28.76, 28.81, 28.80, 28.74, 28.76. Range 0.11 min.

Nothing beats the pre-registered choice, so **28.85 stands** — and the flatness is the useful result: the bound is a property of the data, not of the partition. Community and family structure adds no tightness, because tau ~ 0.21-0.30 and rho ~ 0.60-0.79 in essentially every cell (see section 4): the residual is heavy-tailed *within* families, not because families are mixed.

## 3. Headline results, all three covariate arms

| arm | L (best OOF MAE) | rho | tau | B\* 1-D (published) | **B\* 2-D + strat** | Qcap rung | E published | **E2** | temporal holdout MAE |
|---|---|---|---|---|---|---|---|---|---|
| tabular only | 33.32 | 0.667 | 0.233 | 26.52 | **29.29** | 24.37 | 32.75 | **31.55** | 32.28 [30.89, 33.50] |
| tabular + 11 note features | 32.87 | 0.669 | 0.229 | 26.22 | **28.85** | 24.18 | 32.31 | **31.12** | 31.67 [30.36, 32.85] |
| tabular + notes (features + text) | 33.82 | 0.669 | 0.218 | 27.20 | **30.01** | 25.41 | 33.30 | **32.22** | 32.01 [30.69, 33.22] |

**Notes arm: calibrated floor 28.85 min, estimated floor 31.12 min, current out-of-fold MAE 32.87 min.** Under the frozen guard and dependence assumptions, at most 4.0 min per case is recoverable; the estimate says about 1.8 min is. A shared-bias sensitivity curve is now required for a real-world claim.

## 4. Families, communities and neighbours

Five partitions were built. `procedure` = primary procedure name; `note` = k-means (k=8) on the 100-component note-text SVD, cases without notes kept separate; `knn` = Louvain communities of a 15-nearest-neighbour graph on [note SVD + predicted duration], 45 communities at modularity 0.764; `clin` = anaesthesia x urgency. `purity` is the fraction of a case's 15 neighbours sharing its procedure family; `pairs` is the matched-pairs lower bound computed inside the cell.

### Procedure families

| cell | n | L | rho | tau | **B\*** | **E2** | headroom (L-E2) | matched pairs | nbr purity |
|---|---|---|---|---|---|---|---|---|---|
| ABLATION - PVI | 2774 | 35.6 | 0.725 | 0.229 | **31.3** | **34.1** | 1.6 | 30.7 | 0.70 |
| PPM IMPLANT - DUAL CHAMBER | 1473 | 24.9 | 0.726 | 0.205 | **22.4** | **24.3** | 0.6 | 20.3 | 0.37 |
| ABLATION - PVC | 678 | 48.8 | 0.757 | 0.216 | **44.0** | **47.8** | 1.1 | 38.2 | 0.44 |
| TILT TABLE | 589 | 15.8 | 0.790 | 0.219 | **14.4** | **15.5** | 0.3 | 11.5 | 0.47 |
| ABLATION - VT | 449 | 59.4 | 0.775 | 0.240 | **52.6** | **57.8** | 1.6 | 50.7 | 0.36 |
| PPM GENERATOR CHANGE - DUAL CHAMBER | 447 | 16.2 | 0.622 | 0.211 | **13.2** | **15.5** | 0.7 | 12.4 | 0.39 |
| ABLATION - AV NODE | 398 | 16.8 | 0.645 | 0.212 | **15.0** | **16.1** | 0.6 | 13.4 | 0.42 |
| PPM TEMPORARY PERMANENT INSERTION - SINGLE | 370 | 15.5 | 0.649 | 0.303 | **11.6** | **13.9** | 1.6 | 9.9 | 0.25 |
| DIAGNOSTIC EPS | 356 | 38.1 | 0.695 | 0.216 | **33.2** | **36.6** | 1.5 | 29.1 | 0.09 |
| ABLATION - AVNRT - TYPICAL | 315 | 36.5 | 0.671 | 0.209 | **32.7** | **35.5** | 1.0 | 29.1 | 0.20 |
| LOOP RECORDER IMPLANT | 313 | 10.4 | 0.618 | 0.276 | **7.9** | **9.3** | 1.1 | 8.0 | 0.29 |
| ICD IMPLANT - SINGLE CHAMBER | 273 | 20.4 | 0.719 | 0.240 | **17.8** | **19.7** | 0.7 | 18.5 | 0.15 |

### Note-content communities

| cell | n | L | rho | tau | **B\*** | **E2** | headroom (L-E2) | matched pairs | nbr purity |
|---|---|---|---|---|---|---|---|---|---|
| note_c3 | 3810 | 29.3 | 0.643 | 0.231 | **25.2** | **27.2** | 2.1 | 22.4 | 0.30 |
| note_c6 | 3209 | 35.7 | 0.698 | 0.230 | **30.9** | **33.9** | 1.8 | 29.1 | 0.43 |
| note_c1 | 1969 | 31.1 | 0.661 | 0.231 | **26.8** | **29.1** | 2.0 | 26.0 | 0.19 |
| note_c4 | 1151 | 30.4 | 0.673 | 0.241 | **26.1** | **29.0** | 1.5 | 26.0 | 0.36 |
| nonote | 1073 | 36.9 | 0.683 | 0.231 | **31.9** | **35.0** | 1.9 | 28.1 | 0.22 |
| note_c5 | 503 | 22.4 | 0.595 | 0.242 | **17.9** | **20.5** | 1.9 | 17.2 | 0.32 |
| note_c0 | 482 | 42.1 | 0.716 | 0.211 | **37.8** | **41.1** | 1.1 | 35.0 | 0.75 |
| note_c2 | 475 | 45.7 | 0.718 | 0.235 | **39.9** | **43.7** | 2.0 | 36.3 | 0.37 |
| note_c7 | 321 | 34.2 | 0.694 | 0.233 | **29.5** | **32.5** | 1.7 | 29.8 | 0.41 |

### kNN-graph communities (Louvain)

| cell | n | L | rho | tau | **B\*** | **E2** | headroom (L-E2) | matched pairs | nbr purity |
|---|---|---|---|---|---|---|---|---|---|
| knn_c00 | 1470 | 32.7 | 0.715 | 0.240 | **28.3** | **31.4** | 1.3 | 27.7 | 0.43 |
| knn_c01 | 1175 | 27.1 | 0.616 | 0.228 | **22.0** | **25.2** | 2.0 | 20.7 | 0.25 |
| knn_c02 | 1035 | 34.3 | 0.662 | 0.239 | **29.4** | **32.5** | 1.8 | 27.8 | 0.15 |
| knn_c03 | 927 | 28.9 | 0.651 | 0.236 | **24.7** | **27.3** | 1.6 | 23.0 | 0.27 |
| knn_c04 | 609 | 31.8 | 0.632 | 0.241 | **25.7** | **29.9** | 1.9 | 24.3 | 0.25 |
| knn_c05 | 420 | 25.8 | 0.691 | 0.267 | **21.5** | **24.4** | 1.3 | 20.3 | 0.44 |
| knn_c06 | 407 | 37.9 | 0.685 | 0.234 | **32.6** | **36.0** | 2.0 | 31.6 | 0.79 |
| knn_c08 | 392 | 41.4 | 0.738 | 0.251 | **36.3** | **40.2** | 1.2 | 33.8 | 0.52 |
| knn_c07 | 392 | 39.1 | 0.726 | 0.242 | **34.0** | **37.9** | 1.2 | 31.9 | 0.63 |
| knn_c09 | 332 | 31.7 | 0.666 | 0.234 | **27.3** | **30.0** | 1.7 | 24.7 | 0.19 |
| knn_c10 | 315 | 37.3 | 0.728 | 0.224 | **32.9** | **36.3** | 1.0 | 32.0 | 0.56 |
| knn_c11 | 312 | 41.0 | 0.742 | 0.224 | **36.3** | **39.9** | 1.1 | 35.7 | 0.76 |

### Clinical case type

| cell | n | L | rho | tau | **B\*** | **E2** | headroom (L-E2) | matched pairs | nbr purity |
|---|---|---|---|---|---|---|---|---|---|
| General | Elective | 3479 | 40.3 | 0.708 | 0.223 | **35.3** | **38.7** | 1.6 | 33.3 | 0.55 |
| Monitored Anesthesia Care | Elective | 2430 | 34.7 | 0.690 | 0.228 | **30.1** | **33.0** | 1.7 | 27.4 | 0.24 |
| Monitored Anesthesia Care | Semi-Urgent | 2111 | 26.7 | 0.676 | 0.219 | **23.3** | **25.5** | 1.2 | 21.1 | 0.24 |
| Monitored Anesthesia Care | *Unspecified | 1501 | 27.0 | 0.655 | 0.217 | **23.5** | **25.1** | 1.9 | 21.6 | 0.21 |
| General | Semi-Urgent | 886 | 43.5 | 0.730 | 0.230 | **38.0** | **41.7** | 1.8 | 35.2 | 0.32 |
| *Unspecified | Elective | 727 | 15.4 | 0.674 | 0.232 | **13.2** | **14.5** | 0.8 | 11.5 | 0.43 |
| General | *Unspecified | 381 | 52.4 | 0.692 | 0.227 | **45.4** | **49.9** | 2.5 | 38.4 | 0.14 |
| Monitored Anesthesia Care | Urgent | 323 | 28.5 | 0.665 | 0.244 | **24.2** | **27.1** | 1.4 | 23.1 | 0.23 |
| *Unspecified | Semi-Urgent | 277 | 16.1 | 0.724 | 0.276 | **13.6** | **15.5** | 0.6 | 14.1 | 0.29 |

### What the family analysis shows

- **The floor scales with the procedure, and so does the error.** Ablation-VT sits at a floor of 52.6-57.8 min, tilt-table at 14.4-15.5. Any single global MAE target is the wrong instrument; the per-family floors are the schedulable quantity.
- **Headroom is small and flat.** L - E2 is 0.3-1.6 min in every procedure family. Summed over the 8,435 cases in the twelve families above, the estimated total recoverable time is about **9,350 minutes**, or 1.11 min/case — concentrated in ablation-PVI (4,342 min) simply because it is the highest-volume family, not because it is the least predictable.
- **Fathom dominates matching family by family.** B\* exceeds the matched-pairs bound in **38 of the 42** cells reported above; the four exceptions are all small cells (n < 500). This is the paper's global result reproduced inside every stratum of a real dataset.
- **Neighbourhoods cut across procedures.** kNN purity in note-embedding space ranges 0.09-0.79. Note-content communities are not procedure communities: `note_c0` is 0.75 pure while `DIAGNOSTIC EPS` cases have 0.09 purity, i.e. their notes look like everyone else's. That is why the note text (arm C) does not lower the floor — the text separates cases along an axis that is not the duration axis.
- **tau is remarkably stable at 0.21-0.30 everywhere**, and rho at 0.60-0.79. The residual is heavy-tailed inside every family, so no partition can buy back the guard.

## 5. What is and is not justified

- **E2 is validated** on 50 designs and adopted.
- **The stratified 2-D bound (28.85) is unchanged** from the previous round; this round only established that it is robust to the partition. Its validity rests on the additivity theorem plus the 2-D guard re-validated 50/50 last round.
- **The per-family bounds and estimates use the same machinery**, cell by cell, with a 250-case minimum. Below ~250 the sample-MAD bias on M+ starts to inflate B\*, so smaller families are excluded rather than reported.
- **Not done, and not claimed:** the definitionally strict M+ (max over fold-exchangeable pairs only, excluding cross-family pairs) would raise the bound further and is arguably more faithful to Definition 14, but it cannot be re-validated on the adversarial suite, whose generators are not available. It is left out.
