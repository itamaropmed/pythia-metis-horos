# Nobody yet certifies the tabular-plus-text risk floor

The 2022-2026 literature has every component of the proposed task but has never put them together, so the task has real room. The best achievable risk of any model given tabular data plus its text, bounded two-sidedly, loss-agnostically and with finite-sample guarantees, with the gaps quantified and per-sample risk labels attached, is unclaimed. Bayes-error estimators now carry dimension-free Hoeffding bounds, but only for 0-1, FPR, balanced error and AUC, and only when soft labels reveal the posterior ([Ishida et al. 2023](https://arxiv.org/pdf/2202.00395); [Ushio et al. 2026](https://arxiv.org/abs/2609.02304)). Proper-scoring-rule theory already defines Bayes risk and "information" for any loss as generalized entropy and Bregman information ([Gottwald & Braun 2024](https://arxiv.org/html/2409.20331v1)), but nobody has turned this into a certified estimator. Clinical studies measure the value of notes only as single-split AUROC deltas between particular models, and those deltas run from zero to about +5 points and are often contaminated by leakage ([Katyal 2026](https://www.researchsquare.com/article/rs-11124716/v1); [Lyu et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10148371/)). Conformal methods certify group-level and selection-conditional error, never individual patients ([Gibbs, Cherian & Candès](https://arxiv.org/abs/2305.12616)). The obstacle that decides the design is an asymmetry. An *upper* bound on the best achievable risk is easy to certify for any distribution: it is the held-out risk of any fitted model plus a concentration term. A *lower* bound, the floor itself, is impossible without assumptions such as smoothness, discrete strata or repeated labels ([Shah & Peters 2020](https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-3/The-hardness-of-conditional-independence-testing-and-the-generalised-covariance/10.1214/19-AOS1857.pdf); [Barber et al.](https://arxiv.org/abs/1903.04684)). A credible version of the task therefore states its assumptions openly, certifies an interval [lower floor, upper achievable] instead of a point, and treats the text encoder as part of the definition of the floor. Done this way it would be the first framework that answers "how much better can anyone do, and is the text worth it?" with a certificate instead of an ablation.

## Bayes-error estimators certify a proxy, not the floor

Since 2022 the field has shifted away from density and divergence estimators on (x, y) pairs and toward **instance-free soft-label estimators**. Ishida et al. estimate the binary Bayes error as the mean of min(c_i, 1−c_i) over soft labels c_i = P(y=+1|x_i). With clean soft labels the estimator is unbiased and its error is at most **√(log(2/δ)/(8n)) with probability 1−δ, regardless of input dimension**. The authors used it to estimate a Bayes error of about **6.8-9.6% for ICLR accept/reject decisions**, one of the few such estimates for a text task ([Ishida et al. 2023](https://arxiv.org/pdf/2202.00395)). Follow-ups extended this to multiclass ([Jeong et al. NeurIPS 2023](https://nips.cc/virtual/2023/poster/70427)), to the Bayes classifier's FPR and FNR ([Jeong et al. 2024](https://arxiv.org/abs/2401.15500)), and to optimal balanced error rate and AUC under unknown priors and order-preserving label corruption, with finite-sample bounds ([Ushio et al. 2026](https://arxiv.org/abs/2609.02304)).

The 2025 Ushio et al. paper contains the two results a certified-floor project depends on most. First, when soft labels come from averaging m hard labels, the estimator's bias is **nonpositive**, lying in [−E min{L(η)/m, √(π/2m)}, 0]. It therefore systematically *underestimates* the Bayes error, so subtracting the Hoeffding width gives a conservative lower bound. Second, **calibration is not enough**: a constant base-rate predictor is perfectly calibrated yet implies a Bayes error of 0.5 when the truth is 0. Consistency holds only if the soft labels are correctly *ordered*, after which isotonic recalibration recovers them ([Ushio, Ishida & Sugiyama 2025](https://arxiv.org/html/2505.20761v4)). The certificate therefore applies to the Bayes error of the *annotation distribution*, not the deployment label distribution. The CIFAR-10H result shows the difference matters: ViT's test error fell *below* the estimated floor ([Ishida et al. 2023](https://arxiv.org/pdf/2202.00395)).

Estimators that work from (x, y) pairs alone, which is what EHR data has, perform much worse. A 2025 benchmark found kNN "overwhelmingly the more accurate" nonparametric estimator. Even so, a 95% bound range under 5% needed about **1,000, 2,000 and 2,500 samples per class for 2, 3 and 4 features**, and no estimator met the target in every scenario ([Wheat et al. 2025](https://arxiv.org/html/2506.03159v1)). The upper-bound direction is far easier. BOLT trains against an f-divergence upper bound on the Bayes error ([Tavasoli Naeini et al. 2025](https://arxiv.org/html/2501.07754v1)), and in general any trained model's held-out error is a valid upper bound.

### The excess-risk decomposition has never been estimated end to end

The textbook decomposition is R(f̂) − R* = approximation + estimation + optimization. Only parts of it have data-driven certificates. Krishnamurthy et al. give finite-sample (Hoeffding variant) high-probability upper bounds on a model's excess risk *relative to the best model in a class*, and return a set of potentially optimal models, without needing the class complexity as input ([Krishnamurthy et al. 2024/25](https://arxiv.org/abs/2405.04636)). Combined with a soft-label R*, this would bound approximation error as (best in-class held-out risk) − R*. No paper does this end to end, and estimation and optimization error cannot be separated without retraining, so only their sum is observable.

The **data gap** is in worse shape. On OpenML CC-18, **14.9% of learning curves are ill-behaved**, about double the earlier estimate. The rate depends heavily on the learner: TabNet 74.3% versus about 1.5% for CatBoost and TabPFN v2. Parametric curve fits often fail to beat simply picking the best last-observed learner ([LCDB 1.1](https://arxiv.org/html/2505.15657)). Power-law extrapolation misestimates how much data is needed to reach a target ([Mahmood et al. CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Mahmood_How_Much_More_Data_Do_I_Need_Estimating_Requirements_for_CVPR_2022_paper.html)). Bayesian extrapolation gives predictive distributions, not certificates ([LC-PFN](https://neurips.cc/virtual/2023/poster/69992)). A learning-curve asymptote also estimates approximation gap + R*, not R*, so it cannot stand in for the floor. For the missing-feature or missing-modality version of the data gap, PRIMO models the absent modality as a latent variable and measures per-instance impact, but it needs some data where the modality is observed and assumes Gaussian latents ([Madaan et al. 2026](https://arxiv.org/html/2602.16979)). I found no method that certifies R*(X) − R*(X, X′).

## Proper scoring rules unify the definitions but not the estimators

The theory needed for loss-agnosticism is mature. For any proper loss with generalized entropy G (the Bayes risk of a distribution), the pointwise risk of a prediction η̂ splits into **Bayes risk G(η(x)) plus excess risk D_G(η(x)‖η̂(x))**, a Bregman divergence. Averaged over a posterior or ensemble, the excess term becomes a **Bregman information**. It is mutual information (BALD) for log loss and variance for Brier ([Kotelevskii & Panov 2024](https://arxiv.org/html/2402.10727v2); [Gruber & Buettner AISTATS 2023](https://proceedings.mlr.press/v206/gruber23a.html)). The same template covers regression under CRPS, log, quadratic and squared-error scores ([Fishkov et al. 2025](https://arxiv.org/html/2509.26610v1)) and multivariate kernel scores ([Bülte et al. UAI 2026](https://proceedings.mlr.press/v337/bulte26a.html)). Every proper score also splits into a calibration error plus a refinement term, with consistent, asymptotically unbiased estimators for all of them ([Popordanoska et al. AISTATS 2024](https://arxiv.org/abs/2312.08589v1)).

For a modality question the key definition is decision-theoretic. Gottwald and Braun define the information that a σ-algebra carries as the *reduction in minimal expected loss*. This gives H(X) = I(X;σ) + H(X|σ) for arbitrary losses: Shannon quantities under log loss, Var(E[X|Y]) under squared error, and the Jensen gap (Bregman information) under Bregman losses. Information defined this way is non-negative for nested σ-algebras but **not symmetric** in general ([Gottwald & Braun 2024](https://arxiv.org/html/2409.20331v1)). The proposed "text-conditioned residual" is therefore already well defined for any loss ℓ, as I_ℓ(Y; Text | Tab) = R*_ℓ(Tab) − R*_ℓ(Tab, Text). V-usable information is the same quantity with the minimization restricted to a model family. That restriction is why it can be negative and is model-relative ([Xu et al. ICLR 2020](https://mlanthology.org/iclr/2020/xu2020iclr-theory/); [Ethayarajh et al. ICML 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html)).

### Transfer inequalities connect losses only at square-root rates

Bao's generalized Pinsker inequality bounds |η − η̂| by the inverse modulus of convexity of the entropy, evaluated at half the proper-loss regret. The bound then carries over to 0-1 and ranking regret. Brier gives |η − η̂| ≤ √(2R). Hinge loss gives nothing for errors below ½ ([Bao COLT 2023](https://proceedings.mlr.press/v195/bao23a/bao23a.pdf)). Bao and Takatsu prove that a **½-order rate is the best any strictly proper loss can achieve**, and that strongly proper losses such as log and Brier attain it ([JMLR 2025](https://jmlr.org/papers/v26/24-1112.html)). Transfer is lossy in practice too: ranking by the uncertainty of the *same* loss as the task minimizes expected loss-rejection area ([Hofman et al. 2025](https://arxiv.org/html/2505.22538v1)).

These results transfer *regret*. Transfer of *floors* is simpler and more useful for the proposed task, though the following is my own inference from standard inequalities, not a published result. In the binary case, η(1−η) ≤ min(η, 1−η) ≤ 2η(1−η) pointwise. Taking expectations sandwiches the 0-1 Bayes error between half and all of the Brier Bayes risk E[2η(1−η)]. A certified Brier floor interval therefore yields a certified 0-1 floor interval at a factor-of-two cost. Log loss is harder: it is unbounded, so certifying it requires clipping or tail assumptions.

The estimation side is still missing. Loss-agnostic estimators exist only as posterior or ensemble plug-ins, which are model-dependent. Model-free floors exist for exactly two losses: MSE, through the nearest-neighbor residual-variance estimator with a dimension-free variance bound and exponential concentration ([Devroye, Györfi, Lugosi & Walk 2018](https://projecteuclid.org/euclid.ejs/1528250442)), and 0-1 through soft labels. I found no 2022-2026 model-free estimator for MAE, pinball, CRPS or log-loss floors, and no single estimator valid across proper losses with finite-sample guarantees.

The bias structure of any loss-agnostic plug-in E_X G(P̂_{Y|X}) is my own synthesis, not a published result. Concave G means noise in P̂ biases the floor *downward*, by Jensen's inequality, consistent with Ushio's nonpositive-bias theorem. Neighborhood smoothing mixes distinct posteriors, and because G is concave this biases the floor *upward*. A certified lower bound must control the second effect, for example by a Lipschitz bound on η, and can treat the first as conservative slack.

## Clinical notes add between nothing and five AUROC points, measured badly

The empirical answer to "is the text worth it?" depends on task, observation window, note type, how strong the tabular baseline is, and fusion design. The table summarizes recent controlled comparisons.

| Study | Setting | Structured-only | Notes-only | Fusion | Takeaway |
|---|---|---|---|---|---|
| [Katyal 2026 (preprint)](https://www.researchsquare.com/article/rs-11124716/v1) | MIMIC-IV, first 24h, mortality, 3 seeds | AUROC 0.859 | 0.732 (frozen ClinicalBERT) | 0.857 (bilinear) | No added value; cross-attention worst (0.836) |
| [Lyu et al. AMIA 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10148371/) | MIMIC-III, first 48h | 0.827 | 0.851 (fine-tuned) | 0.877 | +5.0 AUROC, +6.5 AUPRC |
| [Ravichandran et al. 2026](https://arxiv.org/html/2607.15380v1) | MIMIC-III mortality; kidney graft failure | 0.89; 0.88 | 0.87; 0.78 | 0.93; 0.90 | Gain over gradient boosting (0.89) only ~1 pt on graft |
| [Sadanandan 2026](https://arxiv.org/html/2603.14719v1) | MIMIC-IV deterioration, radiology reports | 0.760 | 0.527 | 0.786 | +2.5 pts; calibration fit on test set |
| [ClinicRealm, npj Digit Med 2026](https://www.nature.com/articles/s41746-026-02539-z) | LLM prompts with serialized EHR + notes | — | — | Between single modalities | Fusion did not beat notes alone |

The measurement practice is the main problem, more than the numbers. Every study reviewed defines added value as fusion AUROC minus structured AUROC on one split. None reports a confidence interval on the delta, a Brier or calibration change, or a model-free estimate. The delta therefore mixes three different things: the information in the notes, how well each encoder extracts it, and optimization effects. In the proposed task's language, it confounds the text residual with the approximation and estimation gaps of two different models.

Leakage makes it worse. Zero-shot LLMs reach **AUROC 97.89% on MIMIC-IV mortality from discharge summaries**, which are written after the outcome ([ClinicRealm](https://www.nature.com/articles/s41746-026-02539-z)). Age, sex and ICD codes alone reach **0.97-0.98**, because codes such as brain death and DNR are finalized after discharge, and **40.2%** of the reviewed MIMIC outcome studies used ICD codes as features ([Ramadan et al. 2025](https://www.medrxiv.org/content/10.1101/2025.08.09.25333360v1)). Serializing tables to text can leak as well: one LLM-encoder paper retracted its numbers because current length of stay was included in the serialization ([Hegselmann et al.](https://arxiv.org/pdf/2502.17403v2)).

The work on information decomposition across modalities is closest in spirit to the proposed task but lacks certificates. Liang et al.'s partial information decomposition (redundancy, uniqueness, synergy) scales to high-dimensional data ([NeurIPS 2023](https://arxiv.org/abs/2302.12247)). Its ICLR 2024 follow-up derives **model-free lower and upper bounds on synergy** from unlabeled paired data plus unimodal labeled data, and proposes using them to guide data collection, but states no finite-sample guarantee ([Liang et al. ICLR 2024](https://arxiv.org/html/2306.04539v2)). Gaussian PID bias correction is heuristic ([Venkatesh et al. 2023](https://arxiv.org/abs/2307.10515)). Sample-level PID has no error bounds ([LSMI 2025](https://arxiv.org/html/2506.17248)). I found no verified application of PID to MIMIC structured data versus notes, and no clinical benchmark that weighs the value of acquiring or processing notes against its cost.

## Per-instance difficulty scores are model-relative and the aleatoric label is contested

Almost every popular hardness score is defined relative to a model family and a training run, including PVI, data maps, Data-IQ, AUM, EL2N, forgetting and Cleanlab. The 13-method benchmark of Seedat et al. validates them only against *injected* hardness (synthetic mislabels, out-of-distribution and atypical samples), never against true per-instance Bayes risk min(η(x), 1−η(x)) ([Seedat et al. ICLR 2024](https://arxiv.org/pdf/2403.04551)). The only model-free tabular scores are geometric ones such as kDN, the fraction of nearest neighbors with a disagreeing label, which is a crude local Bayes-error plug-in ([Paiva et al. MLJ 2022](https://mlanthology.org/mlj/2022/paiva2022mlj-relating)). The difficulty literature and the Bayes-error literature barely cite each other.

The aleatoric/epistemic split has been criticized from five directions, and the criticisms reinforce one another:

- The entropy/mutual-information decomposition violates natural axioms ([Wimmer et al. UAI 2023](https://arxiv.org/html/2209.03302v2)).
- No loss gives a second-order learner an incentive to report its epistemic uncertainty faithfully ([Bengs et al. ICML 2023](https://proceedings.mlr.press/v202/bengs23a.html)).
- Evidential methods collapse epistemic uncertainty or make it depend arbitrarily on a regularizer ([Jürgens et al. ICML 2024](https://arxiv.org/html/2402.09056v3)).
- "No existing approach provides pairs of disentangled uncertainty estimators in practice" on ImageNet ([Mucsányi et al. NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5afa9cb1e917b898ad418216dc726fbd-Abstract.html)).
- Epistemic estimates ignore bias, which inflates the aleatoric term ([Jiménez et al. 2025](https://arxiv.org/abs/2505.23506)).

Bickford Smith et al. add that expected conditional entropy is only a *biased finite-sample estimator* of irreducible uncertainty. They redefine reducible uncertainty against an infinite-data limit that is generally incomputable ([ICML 2025](https://arxiv.org/abs/2412.20892)).

Two constructive results define what can actually be done. Ahdritz et al. prove that a higher-order-calibrated predictor recovers true aleatoric uncertainty averaged over level sets **with no distributional assumptions**, but only from k ≥ 2 independent labels per input. With a single label per input, cases with different aleatoric uncertainty are provably indistinguishable ([ICLR 2025](https://arxiv.org/html/2412.18808v1)). Paplhám et al. evaluate epistemic scores against true per-instance *regret*, using dense CIFAR-10H annotations as the oracle. They find rankings invert across tasks: evidential networks have the best regret (AuReC 0.0673) but weak near-OOD AUROC (0.711) ([Paplhám et al. 2026](https://arxiv.org/html/2607.14817)). DEUP and RHO-LOSS operationalize "headroom" as observed loss minus an estimated irreducible loss, but their irreducible term comes from another model ([DEUP](https://arxiv.org/abs/2102.08501v4); [RHO-LOSS](https://proceedings.mlr.press/v162/mindermann22a.html)).

For the proposed task this settles the design. A per-sample split into "irreducibly hard" and "model-hard" can be certified only by borrowing structure: repeated labels, a smoothness class, or a coarsening into groups.

## Conformal guarantees stop at the group, and clinics already feel it

Finite-sample conformal guarantees come in four strengths:

- Marginal guarantees: split conformal, and conformal risk control for monotone bounded losses ([Angelopoulos et al. ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)).
- PAC guarantees from Learn-then-Test, which casts threshold selection as multiple testing ([Angelopoulos et al.](https://arxiv.org/abs/2110.01052)).
- Exact coverage over finitely many overlapping groups: multivalid and Mondrian methods ([Jung et al. ICLR 2023](https://iclr.cc/virtual/2023/poster/11605); [Lu et al. AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/21459)), and finite-dimensional covariate-shift classes ([Gibbs, Cherian & Candès](https://arxiv.org/abs/2305.12616)).
- Selection-conditional guarantees, the closest certified analogue of a risk *label*: FDR control among selected units ([Jin & Candès](https://arxiv.org/abs/2210.01408)), and exact or PAC control of risk among accepted samples. Naive "threshold then calibrate" pipelines break exchangeability, which is why these dedicated methods are needed ([Selective CRC 2025](https://arxiv.org/html/2512.12844v1)).

Exact coverage conditional on X = x is impossible in finite samples: any distribution-free method that achieves it must produce sets of infinite expected size ([Barber et al.](https://arxiv.org/abs/1903.04684)).

Clinical deployments show both the value and the failure modes. Conformal triage of a head-CT model at MGH cut **false positives from 233 (45%) to 8 (5%) while abstaining on 14%** of cases, using thresholds calibrated for NPV and PPV ([Angelopoulos et al. 2024 preprint](https://www.medrxiv.org/content/10.1101/2024.02.09.24302543v1)). StratCP's action arm holds FDR at 0.046 on IDH status, where standard conformal prediction reached **0.096 against a 0.05 budget** ([Jin, Moon & Zitnik 2026](https://zitniklab.hms.harvard.edu/projects/StratCP)). The failure modes are just as concrete:

- Under prevalence shift, marginal coverage and review rate can look healthy while event-positive patients are released unreviewed ([Li et al. COPA 2026](https://proceedings.mlr.press/v329/li26a.html)).
- GPT-4o conformal prediction on ICU notes hit 90.4% marginal coverage but **99.7% for death versus 81.1% for survival**, with only 32.9% singleton sets ([Shah-Mohammadi et al. JAMIA Open 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13283473/)).
- In vanilla conformal prediction, noise and model error jointly set the global threshold, so set size cannot say whether *the model* is uncertain or *the outcome* is inherently unpredictable ([Sale et al. COPA 2025](https://proceedings.mlr.press/v266/sale25a.html)).

I found no clinical paper that certifies a per-patient separation of Bayes risk from model risk.

## Nearest-neighbor dependence estimators are consistent but almost never finite-sample

The pair-based and graph-based family looks like the natural engine for a model-free floor, but its guarantees are nearly all asymptotic:

- Chatterjee's ξ has an **n^{-1/4} detection boundary** for independence and a minimax n^{-1/2} boundary for non-trivial dependence ([Auddy, Deb & Nandy](https://arxiv.org/abs/2104.15140v2)). Its bootstrap is only asymptotically consistent ([Dette & Kroll](https://arxiv.org/abs/2308.01027v1)).
- The multivariate Azadkia-Chatterjee coefficient converges at n^{-1/max(d_Z,2)}, and its nominal 5% test rejects **15.68% of the time at n = 50** ([Huang, Li & Wang 2026](https://arxiv.org/pdf/2512.07443v2)).
- In infinite-dimensional or embedding spaces, nearest-neighbor graph degree can diverge and break the CLT ([Hörmann & Strenger](https://arxiv.org/pdf/2405.07732v3)).

The structural point (my synthesis): the Azadkia-Chatterjee T(Y, Z | X) is a normalized gap between E Var(P(Y ≥ t | X)) and E Var(P(Y ≥ t | X, Z)). That is a Brier-type Bayes-risk reduction integrated over thresholds, so these coefficients already *are* conditional information under a proper loss.

The closest competitor in this family is **Györfi, Linder and Walk (TEST 2025)**. They test H0: L*(S) = L*, meaning "the feature subset S loses no Bayes risk", for both classification and regression. The statistic is a nearest-neighbor estimate of the excess Bayes risk with threshold log n/√n, and they prove consistency under mild conditions, but **not finite-sample size control** ([Györfi, Linder & Walk](https://link.springer.com/article/10.1007/s11749-024-00958-2)). Applied with S = Tab and the full set = Tab ∪ Text, this is almost exactly a test of whether the text residual is zero. It is still asymptotic, and it is limited to MSE and 0-1 loss.

Finite-sample validity exists only for *tests*, and only under restrictions. Within-stratum permutation is exactly valid when the conditioning variable is discrete ([Kim et al. AoS 2022](https://arxiv.org/abs/2112.11666)). PairSwap is valid under isotonicity ([Hore et al. 2025](https://arxiv.org/pdf/2501.06133v2)). A test that rejects certifies that the dependence is positive but says nothing about its size. Shah and Peters show that no CI test can be uniformly valid and also have power, over continuous conditioning variables ([AoS 2020](https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-3/The-hardness-of-conditional-independence-testing-and-the-generalised-covariance/10.1214/19-AOS1857.pdf)). Practical Type-I inflation also traces to errors in estimating conditional mean embeddings ([He et al. NeurIPS 2025](https://arxiv.org/abs/2512.14000)).

The notes also cite a classical lower bound for Bayes-error estimation: no estimator converges uniformly at any rate (Antos, Devroye & Györfi, 1999). It was not re-verified for this report. I found no application of ξ, Azadkia-Chatterjee/FOCI, KPC or nearest-neighbor residual variance to text or mixed tabular-text data.

## Where the proposed task fits and what it must concede

### The task fills five verified gaps at once

Each component of the task matches a gap the notes found independently:

1. **No model-free certified floor exists beyond 0-1 and MSE.** Nothing covers log loss, Brier, pinball or general proper scores, and no single estimator spans losses.
2. **No end-to-end decomposition has been estimated.** No paper estimates the irreducible, approximation, estimation and finite-sample terms together for a real dataset.
3. **No certified value of a modality.** The value of notes beyond tables is measured only as model-specific, uncalibrated, CI-free deltas. PID bounds have no finite-sample guarantees and have not been applied to EHR.
4. **No certified per-sample split.** Nothing separates a patient's irreducible risk from model risk, and no hardness score has been validated against true local Bayes risk.
5. **No dependence or floor estimator for text.** None has been applied to text or mixed data with guarantees.

A framework that outputs a certified interval for R*_ℓ(Tab), a certified interval for the text residual I_ℓ(Y; Text | Tab), and their combination into a global floor, then derives selection-conditional risk labels from it, addresses all five at once. That breadth is the novelty argument.

### Closest competitors

| Work | What it certifies | What it lacks relative to the task |
|---|---|---|
| [Ishida 2023](https://arxiv.org/pdf/2202.00395) / [Ushio 2025](https://arxiv.org/html/2505.20761v4) / [Ushio 2026](https://arxiv.org/abs/2609.02304) | Finite-sample Bayes error, BER and AUC from soft labels | Needs soft or repeated labels; 0-1 family only; no modality split |
| [Györfi, Linder & Walk 2025](https://link.springer.com/article/10.1007/s11749-024-00958-2) | Consistent test that a feature subset loses no Bayes risk | Asymptotic; MSE/0-1 only; no magnitude interval; no text |
| [Devroye et al. 2018](https://projecteuclid.org/euclid.ejs/1528250442) | Nearest-neighbor residual variance with concentration | MSE only; smoothness needed for bias control |
| [Liang et al. ICLR 2024](https://arxiv.org/html/2306.04539v2) | Model-free population bounds on multimodal synergy | No finite-sample guarantee; no Bayes-risk floor |
| [Krishnamurthy et al. 2025](https://arxiv.org/abs/2405.04636) | High-probability excess risk relative to class optimum | Class-relative; no R* |
| [Ahdritz et al. ICLR 2025](https://arxiv.org/html/2412.18808v1) | Provable aleatoric uncertainty on level sets | Needs k ≥ 2 labels per input |
| [Paplhám et al. 2026](https://arxiv.org/html/2607.14817) | Evaluation of per-instance regret against a dense-annotation oracle | Evaluation protocol only, no certificate |
| [StratCP](https://zitniklab.hms.harvard.edu/projects/StratCP) / [Conformal triage](https://www.medrxiv.org/content/10.1101/2024.02.09.24302543v1) | FDR/NPV among patients assigned to a category | Model-relative; no floor; cannot say whether residual risk is irreducible |

### No-free-lunch dictates a two-sided, assumption-explicit design

The design the evidence supports separates the two directions explicitly. The table below is my synthesis.

| Quantity | Distribution-free in finite samples? | Route |
|---|---|---|
| Upper bound on R*(Tab) or R*(Tab, Text) | Yes | Held-out risk of any fitted model plus Hoeffding or Bernstein (bounded loss); uniform over many candidates via LTT-style testing |
| Lower bound on R* (the floor) | No ([Barber et al.](https://arxiv.org/abs/1903.04684); [Shah & Peters](https://projecteuclid.org/journals/annals-of-statistics/volume-48/issue-3/The-hardness-of-conditional-independence-testing-and-the-generalised-covariance/10.1214/19-AOS1857.pdf)) | Requires one of: Lipschitz η plus nearest-neighbor pairs; discrete or coarsened tabular strata; repeated labels or clinician multi-reads; order-preserving soft labels |
| Text residual is positive | Yes, as a test, under discrete strata | Within-stratum permutation ([Kim et al.](https://arxiv.org/abs/2112.11666)) on fixed-embedding statistics |
| Text residual is at least c | Only via [lower bound on R*(Tab)] − [upper bound on R*(Tab, Text)] | The hard term is the tabular floor; inherits its assumptions |
| Per-patient floor r*(x) | No ([Ahdritz et al.](https://arxiv.org/html/2412.18808v1)) | Group or level-set averages only; per-sample *labels* via selection-conditional FDR/NPV |

The useful property of this design is that, for the headline claim "the text is worth at least c", the hard direction lands on the tabular side. That claim needs only a lower bound on the tabular floor L_Tab and an upper bound U_Tab+Text on the joint achievable risk, so the certified residual is c = L_Tab − U_Tab+Text. Structured EHR features can be coarsened into low-dimensional or discrete strata, where nearest-neighbor and permutation methods behave, while the high-dimensional text side only needs an *upper* bound, which any strong fusion model or LLM supplies.

The *global* floor is harder. Its interval [L_Tab+φ(Text), U_Tab+Text] needs a lower bound on the joint floor itself. That requires assumptions on the joint space, such as low intrinsic dimension of a fixed embedding or repeated labels, and it will usually be loose. The bound carries three gaps:

- **Finite-sample gap**: the confidence widths.
- **Estimation gap**: the deployed model's held-out risk minus U_Tab+Text.
- **Approximation or representation gap**: since a fixed text encoder φ satisfies R*(Tab, φ(Text)) ≥ R*(Tab, Text) by data processing, any floor computed through φ is a floor *for that encoder*.

The task should say this explicitly, not claim a floor on raw text. This framing is my inference from the notes' data-processing argument and the dimension results in [Wheat et al.](https://arxiv.org/html/2506.03159v1).

Per-sample labels follow the same logic. Categorical labels such as "irreducibly hard", "model-hard", "low-risk" and "defer" can carry certified selection-conditional guarantees: the error rate among patients given a label is at most ε, with Mondrian versions for pre-specified subgroups ([Jin & Candès](https://arxiv.org/abs/2210.01408); [Selective CRC](https://arxiv.org/html/2512.12844v1)). Continuous labels, meaning a local floor estimate and a local excess-risk estimate, can only be certified on average over level sets or under smoothness. They should be presented as calibrated scores with a group-level certificate, not as individual guarantees.

### Remaining obstacles are empirical as much as theoretical

Five obstacles go beyond impossibility theorems:

- **Labels.** EHR outcomes have one label per patient, so the soft-label route needs clinician multi-reads or annotator-disagreement datasets for validation.
- **Leakage.** A floor estimate that comes out suspiciously low is itself a leakage alarm, and this is arguably the framework's most immediate industrial use; the ICD-code AUROC of 0.97 is the canonical case ([Ramadan et al.](https://www.medrxiv.org/content/10.1101/2025.08.09.25333360v1)).
- **Temporal validity.** Notes need chart times, and splits must respect time.
- **Unbounded losses.** Log loss needs clipping, and the ½-order transfer barrier limits how tight cross-loss statements can be ([Bao & Takatsu](https://jmlr.org/papers/v26/24-1112.html)).
- **Site and prevalence shift and label scarcity**, which the COPA 2026 audit shows can silently break release-side guarantees ([Li et al.](https://proceedings.mlr.press/v329/li26a.html)).

Validation has to use synthetic data with known η(x) and dense-annotation sets such as CIFAR-10H, following [Paplhám et al.](https://arxiv.org/html/2607.14817). No clinical ground truth exists.

### What makes it novel and what makes it sell

Academically, the contribution is the first estimator of the floor-interval *for arbitrary proper losses* with finite-sample guarantees, together with a certified, loss-indexed conditional information I_ℓ(Y; Text | Tab). In effect it would be a finite-sample, multi-loss, multimodal version of Györfi-Linder-Walk, linked to selection-conditional conformal labels. A strong paper would include five things:

1. A theorem pairing a distribution-free upper bound with an assumption-indexed lower bound whose assumption is testable or at least auditable, such as a Lipschitz constant or a stratification.
2. The floor-transfer sandwich across losses.
3. An explicit treatment of the encoder as part of the estimand.
4. A synthetic benchmark with known η where the bound's coverage is measured.
5. A MIMIC-IV case study on a leakage-audited, time-respecting cohort.

Industrially, the outputs answer questions buyers already ask:

- Is investing in an NLP pipeline worth it, given a certified lower bound on what text can add?
- When to stop tuning a model, given its certified headroom to the floor.
- Whether a dataset is leaking, given an implausibly low floor.
- Which patients to route to clinicians, given certified-error triage labels.

## Conclusion

The literature has converged on the right *definitions*: Bayes risk as generalized entropy, the value of a modality as Bregman information, and epistemic uncertainty as regret to the Bayes predictor. It has the right *certification machinery* in selection-conditional conformal and LTT-style multiple testing. It has the right *estimator family* in nearest-neighbor pair statistics. These were never put together because each community certifies a different object. Bayes-error researchers certify an annotation-distribution proxy. Conformal researchers certify a fixed model's group error. Dependence researchers certify asymptotic consistency. The proposed task's real contribution is to make one estimand, a loss-indexed interval for the best achievable risk with and without text, the shared target of all three.

The no-free-lunch results do not block the task. They tell it where to spend its assumptions: on the low-dimensional tabular floor, never on the text, and on groups, never on individual patients. One reframing changes the value proposition. The most immediately useful output is not a tight floor but a *certified upper bound on remaining headroom* plus a *flag when the floor looks implausibly low*. The first tells a team when to stop building models; the second catches leakage that, as the MIMIC ICD and discharge-summary results show, currently passes peer review.
