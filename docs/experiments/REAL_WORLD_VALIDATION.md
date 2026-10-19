# Real-world validation path

Pythia can dominate a comparator only on a matched estimand and assumption set.
A larger invalid lower bound is a loss, not a win. The real-world product will
therefore expose a rung ladder rather than silently selecting its strongest
number.

## Rungs

1. **Population sensitivity**: exact moment algebra over a grid of declared
   shared-error caps. This shows assumption fragility but is not a finite-sample
   certificate when empirical moments are substituted.
2. **Bounded independent evaluation**: empirical-Bernstein lower confidence
   bound for frozen predictors on untouched rows.
3. **Bounded independent clusters**: case-weighted weighted-Hoeffding bound for
   frozen predictors on independent patients. This is rigorous but can be
   vacuous when the operational duration range is large.
4. **Finite-variance independent patients**: Catoni patient-level bound. This
   tolerates unbounded/heavy-tailed losses and targets the expected mean case
   loss for a randomly sampled patient.
5. **Bounded-range patient empirical Bernstein**: variance-adaptive
   patient-level concentration using endpoint and predictor-difference ranges,
   with no externally guessed variance cap. The sharp version minimizes the
   exact pairwise dependence formula over simultaneous moment rectangles.
6. **Median-of-means fallback**: simpler finite-variance proof with weaker
   constants.
7. **OOF retrospective diagnostic**: calibrated/bootstrap only until a theorem
   handles overlapping training sets. Never relabel it proved.

## MAE shared-bias sensitivity

The Qcap expression proves a guard on combining centered components; it does
not prove that model disagreement sees predictor error shared by every model.
Accordingly, the complete Qcap rung is now labelled conditional even though the
guard inequality itself is proved.

Let a base MAE rung lower-bound `MAD(epsilon-b)`, where `b` is the component of
predictor error common to the pool. Since MAD is subadditive,

```text
MAD(epsilon-b) <= MAD(epsilon) + MAD(b).
```

If an external audit supplies `MAD(b) <= C`, then

```text
R*_MAE >= max(base_bound-C, 0).
```

The new `mae_shared_bias_sensitivity` reports this complete curve and never
infers `C` from disagreement. In 2,989 evaluated adversarial designs spanning
zero through extreme shared bias, the unadjusted calibrated rung was valid only
982 times and the unadjusted Qcap rung 1,489 times. Supplying the simulation-known
common-bias MAD cap restored 2,989/2,989 validity for every adjusted rung (Wilson
lower 0.9987). This validates the sensitivity correction, not a procedure for
estimating the cap.

The misspecification experiment is decisive: pooling all rungs and bias levels,
using no cap was valid in 44.4% of 8,967 cases and using half the true cap in
83.3%. The correct or a 50%-conservative cap was valid in 100%. Therefore cap
governance is part of the theorem, not an optional reporting detail.

For Mayo Arm B the sensitivity is directly interpretable in minutes:

| common-bias MAD cap C | calibrated floor | conditional Qcap floor |
|---:|---:|---:|
| 0 | 28.85 | 24.19 |
| 2 | 26.85 | 22.19 |
| 5 | 23.85 | 19.19 |
| 10 | 18.85 | 14.19 |

These are simultaneous algebraic sensitivity values, not a selection of a
plausible C. Without external information, the curve must be shown through its
collapse point rather than choosing zero shared bias.

## Nuisance-cap governance

Positive lower bounds are impossible without restricting invisible shared
model bias. Pythia must therefore accept a correlation cap as an assumption and
report the entire sensitivity curve. It must not estimate a convenient upper
cap from the same risks and disagreements: those moments generally identify a
minimum feasible correlation, not a valid maximum.

Variance caps for Catoni concentration must also be predeclared. The acceptable
workflow is:

1. use historical development data to propose caps with a safety factor;
2. freeze cap values, estimator code, endpoint, and patient-level estimand;
3. validate cap exceedance rules on a separate calibration window;
4. score already-frozen predictors on untouched/prospective patients;
5. refuse the corresponding rung if the frozen cap audit fails;
6. report sensitivity to larger caps without selecting after seeing outcomes.

Caps estimated and evaluated on the same Mayo cohort are exploratory only.

## Fail-closed public API

The finite-variance dependence rung is exposed through a versioned
`DependenceGuardSpec`. The public `certify(...)` path accepts it only for MSE,
requires patient/entity identifiers, and requires artifact metadata declaring
both an `external_holdout` or `prospective` evaluation role and predictions
frozen before evaluation. Retrospective OOF artifacts, unfrozen predictors,
unsupported losses, invalid caps, and incomplete guard provenance return a
structured refusal. A successful result is `conditional`, never `proved`,
because the correlation and variance caps remain declared nuisance assumptions.

## Current evidence

### Bounded simulations

Across 1,800 runs, the empirical-Bernstein rung covered 1,800/1,800, with a
95% Wilson lower bound of 0.9979. At 10,000 cases its median tightness was 0.887,
versus 0.256 for the Hoeffding rung. The sharp plug-in estimator reduced median
relative error from 9.54% to 1.77%.

### Heavy-tail simulations

Across 900 Student-t runs with known finite-variance and correlation caps,
Catoni covered 900/900 (Wilson lower 0.9957). At 10,000 patients its median
tightness was 0.821, compared with 0.781 for the legacy empirical bound and
0.375 for median-of-means. At smaller sample sizes Catoni was valid but more
conservative.

### Variable-size patient simulations

Across 1,800 untouched simulations with 500, 2,000, or 10,000 independent
patients and one to four cases per patient, the bounded-range patient empirical-
Bernstein rung covered 1,800/1,800 (Wilson lower 0.9979). Its median usable
tightness was 0.846, versus 0.561 for matched patient Hoeffding, and it was
strictly tighter in 1,797/1,800 runs. At 10,000 patients its tightness reached
0.950, exceeding the 0.931 of Catoni supplied with oracle variance caps. The
sharp moment-rectangle geometry improved the older conservative dependence
formula in 1,197/1,800 runs. The uncorrected legacy method covered only 66.3%.

### Mayo diagnostic

Mayo has 12,993 cases, 10,532 patients, and a case-weighted effective cluster
count of about 8,783. A bounded cluster theorem with a 0–900 minute endpoint is
vacuous because its squared-loss risk confidence radius is about 15,020, far
larger than the observed MSE near 2,451. This is why variance-adaptive robust
means are necessary.

The new patient empirical-Bernstein rung removes the guessed variance cap, but
the broad 0--900 minute endpoint/prediction range remains costly. On the current
retrospective Mayo artifacts its diagnostic RMSE floor is 19.75 at
`lambda_max=0`, 13.18 at `lambda_max=0.25`, and zero by `lambda_max=0.5`.
These values are not certificates because the predictions are overlapping OOF;
they quantify exactly what the range-only theorem would deliver on a future
frozen evaluation cohort with the same moments.

Using variance values measured on the same Mayo cohort only as a diagnostic,
the Catoni patient-level curve remains nonzero even after multiplying both
variance values by four. At that fourfold sensitivity level, conditional RMSE
floors range from 41.95 at `lambda=0.75` to 31.26 at `lambda=0.95`. These are not
certificates: the predictors are retrospective OOF artifacts and both nuisance
caps were inspected on the evaluation cohort.

## Remaining proof and validation work

- Establish a valid upper-correlation-cap protocol or retain the full curve.
- Add theory for overlapping OOF training, likely requiring algorithmic
  stability or a dedicated untouched evaluation split.
- Validate Catoni coverage with variable-size, correlated patient clusters,
  skew, mixtures, heteroskedasticity, and cap misspecification.
- Develop a separate MAE dependence theorem; squared-loss results do not imply
  the operational Mayo MAE result.
- Freeze exact KNIFE-P/LMC-P and other matched-estimand comparators.

Catoni source: Olivier Catoni, “High confidence estimates of the mean of
heavy-tailed real random variables,” https://arxiv.org/abs/0909.5366. A modern
review and sharper multivariate context is provided by Gupta, Hopkins, and
Price, “Beyond Catoni,” https://proceedings.mlr.press/v247/gupta24a.html.
