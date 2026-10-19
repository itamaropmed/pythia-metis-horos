# Literature survey for Pythia's local-risk certificate (2026-10-18)

**Scope.** Work that bears directly on the open problems of `pythia/core/local_model_risk.py`:

1. what out-of-fold losses certify;
2. tighter bounds for bounded, clustered means;
3. many cells, selection and pooling;
4. group-conditional guarantees;
5. shift and transport;
6. monitoring after deployment.

Each entry says what Pythia takes from the paper, what it leaves, and which development work package (WP) acts on it.

## 1. What cross-validated losses estimate

| Work | Key fact | Consequence for Pythia |
|---|---|---|
| Bates, Hastie & Tibshirani, *Cross-validation: what does it estimate and how well does it do it?* (JASA 2024, [arXiv 2104.00673](https://arxiv.org/abs/2104.00673)) | CV estimates the *average* error of models fitted on other training sets, not the error of the model actually deployed. Naive CV intervals undercover because fold errors are correlated; their nested-CV variance estimate repairs this. | Out-of-fold W30 losses describe the fold models, not the final model. The Mayo replay certifies the wrong object, and its "independent entities" attestation is not exact. **WP1: split calibration.** |
| Bayle, Bayle, Janson & Mackey, *Cross-validation confidence intervals for test error* (NeurIPS 2020, [paper](https://proceedings.neurips.cc/paper/2020/hash/bce9abf229ffd7e570818476ee5d7dde-Abstract.html)) | Under algorithmic stability, k-fold CV obeys a CLT, giving asymptotically valid intervals for k-fold test error. | This is an asymptotic and stability-conditional route to cross-fitting. It is a possible later `calibrated` tier that keeps the efficiency of using all data; it is not a finite-sample certificate. |

## 2. Concentration for bounded means

| Work | Key fact | Consequence |
|---|---|---|
| Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting* (JRSSB 2024, [arXiv 2010.09686](https://arxiv.org/abs/2010.09686)) | Betting confidence sets and sequences are typically much tighter than Hoeffding or empirical Bernstein, and they are anytime-valid. | Their predictable-bet theory needs a common conditional mean. For independent, non-identical clusters only *fixed* bets remain valid, which Pythia already uses (`betting_lower`). The heterogeneous-support Chernoff bound is the right fixed-design object. Betting is the natural tool for **monitoring (WP4)**. |
| Hoeffding (1956); Bentkus (2004); Maurer & Pontil (2009) | Classical endpoint and Bernstein bounds. | Already composed in `concentration.py`. |

## 3. Many cells, selection and pooling

| Work | Key fact | Consequence |
|---|---|---|
| Benjamini & Yekutieli, *FDR-adjusted multiple confidence intervals for selected parameters* (JASA 2005, [summary](https://academicnewsletter.sufe.edu.cn/info/445750)) | For intervals reported only for *selected* parameters, the false coverage rate (FCR) can be controlled with narrower intervals than FWER. | Routing selects which cells are reported. Pythia's simultaneous (FWER) intervals imply FCR control. An FCR tier would be narrower but makes a weaker claim. *Optional future tier.* |
| Armstrong, Kolesár & Plagborg-Møller, *Robust empirical Bayes confidence intervals* (Econometrica 2022, [arXiv 2004.03448](https://arxiv.org/abs/2004.03448)) | Shrinkage intervals with valid **average** coverage across many groups, robust to the distribution of group effects. | This is the principled form of "hierarchical pooling". It buys much shorter intervals for small cells, at the cost of guaranteeing coverage on average across cells rather than per cell. *Candidate `average-coverage` tier (WP3b, future).* |
| Meinshausen, *Hierarchical testing of variable importance* (Biometrika 2008, [IDEAS](https://ideas.repec.org/a/oup/biomet/v95y2008i2p265-278.html)) | Top-down testing on a tree controls FWER while spending α only where coarser nodes were rejected. | The hierarchy could be tested top-down, testing fine cells only below a certified coarse direction, to save α. *Future.* |
| Wang & Ramdas, *False discovery rate control with e-values* (JRSSB 2022) | e-BH controls FDR under arbitrary dependence. | Cell directions overlap and are dependent, so an e-BH directional tier is possible. *Future.* |

## 4. Group-conditional guarantees

| Work | Key fact | Consequence |
|---|---|---|
| Gibbs, Cherian & Candès, *Conformal prediction with conditional guarantees* ([arXiv 2305.12616](https://arxiv.org/abs/2305.12616)) | Exact coverage simultaneously over a finite class of (overlapping) groups, or over a function class. | It is the closest analogue to Pythia's hierarchy, but it targets prediction sets, not certified risk levels. It supports reporting the groups a case belongs to rather than a single backoff cell. It motivates **WP3: certify exactly the routed region.** |
| Angelopoulos, Bates et al., *Learn then Test* ([arXiv 2110.01052](https://arxiv.org/abs/2110.01052)) and *Conformal risk control* ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)) | Risk control by multiple testing over a calibration set that is separate from training. | It confirms the split-calibration architecture. Pythia's directional labels are LTT-style hypothesis tests. |

## 5. Shift and transport

| Work | Key fact | Consequence |
|---|---|---|
| Tibshirani, Barber, Candès & Ramdas, *Conformal prediction under covariate shift* (NeurIPS 2019, [arXiv 1904.06019](https://arxiv.org/abs/1904.06019)) | With a known likelihood ratio, weighted exchangeability restores validity. | Case-mix shift *between* cells is harmless for cell-level claims. A global claim needs the later case mix (reweighting). |
| Zhao, Small & Bhattacharya, *Sensitivity analysis for IPW estimators via the percentile bootstrap* (JRSSB 2019, [arXiv 1711.11286](https://arxiv.org/abs/1711.11286)), the marginal sensitivity model | A bounded odds or density-ratio budget Λ gives sharp worst-case bounds. | **WP4: Λ-transport.** For a {0,1} loss, odds(r′) ∈ [odds(r)/Λ, Λ·odds(r)] is sharp, and each certified direction gets a tipping value Λ\*. |
| *Distributionally robust risk evaluation with an isotonic constraint* ([arXiv 2407.06867](https://arxiv.org/abs/2407.06867)) | Restricting the shift class with shape constraints tightens distributionally robust risk bounds. | Future: monotone drift along predicted duration instead of a flat Λ. |

## 6. Monitoring after deployment

| Work | Key fact | Consequence |
|---|---|---|
| Podkopaev & Ramdas, *Tracking the risk of a deployed model and detecting harmful distribution shifts* (ICLR 2022, [arXiv 2110.06177](https://arxiv.org/abs/2110.06177)) | Sequential tests that alarm only on *harmful* shift (risk above a tolerance), anytime-valid. | **WP4: `BettingRiskMonitor`.** One monitor per routed region, with the null "later risk ≤ certified upper bound". |
| Chugg, Cortes-Gomez, Peng, Chowdhury, Abbott & Ramdas, *Auditing fairness by betting* (NeurIPS 2023, [arXiv 2305.17570](https://arxiv.org/abs/2305.17570)) | Anytime-valid sequential comparison of group-level risks. | Extends to online local-vs-global contrasts. *Future.* |

## Where Pythia sits

No surveyed method combines all of the following:

1. a finite-sample, training-free, patient-clustered certificate of *frozen-model* risk;
2. simultaneity over a support-adaptive hierarchy, with outcome-blind backoff;
3. categorical relations that follow logically from the intervals;
4. fail-closed attestation;
5. a declared-Λ transport tier with tipping values and an anytime-valid harmful-shift monitor.

The individual ingredients are classical: Chernoff inversion, Bonferroni, LTT-style testing, the sensitivity model and betting martingales. The defensible contribution is the **composition**, and the guarantee that the certified estimand matches the population a case is routed into (WP3).
