# Declared-modulus certificates: audit and betting-strengthened route

## Decision

The supplied `pythia_declared_modulus_bundle.tgz` contains a valid and useful
conditional theorem, with two essential qualifications:

1. the barycentric discrepancy is **declared**, not estimated from one outcome
   per unit;
2. the independently fitted location shift is an **external frozen alignment
   resource**, not model training inside Pythia.

The theorem is now implemented as `declared_modulus_certificate.py`.  It
returns a sensitivity certificate and refuses missing block/shift provenance.
It supports MAE, MSE, and arbitrary declared Qxx pinball losses.  Selection
over block size, score cap, and concentration route is alpha-accounted.

## Audited theorem

For fixed, outcome-blind, globally disjoint blocks, let `Q_j` be the clipped
law of the residual `Y_j-s(z_j)` and let the comonotone barycenter have quantile
function `Fbar^-1 = mean_j F_j^-1`.  Define

- `P = mean |F_j^-1-Fbar^-1|` for MAE/pinball;
- `P2 = mean (F_j^-1-Fbar^-1)^2` for MSE.

If `S` is the within-block empirical optimum, SAA optimism and the barycentric
coupling imply

    E[S] <= average Bayes risk + c_loss P,

with `c_MAE=1` and `c_Qq=1/2` for every `q`.  The latter coefficient is not a
generic Lipschitz constant: it follows because the barycentric deviations sum
to zero, so their mean positive and negative parts are both half their mean
absolute value.  For MSE the exact variance decomposition gives penalty `P2`.

Consequently, on a single lower-confidence event,

    full risk >= retained * max(0, LCB(E[S]) - declared penalty)

simultaneously over every declared discrepancy at least as large as the true
one.  A declaration may be scalar or a predeclared profile indexed by block
size.  This is a conditional sensitivity statement, not an identified
one-draw estimate of `P`.

### Clipping and shifting

A unit-specific frozen translation preserves that unit's location-loss Bayes
risk.  Projection to a fixed residual interval cannot increase MAE, pinball,
or squared-error Bayes risk, so the clipped residual target is a valid lower
target for the original outcome risk.  The implementation states both steps
explicitly; it does not say that clipping itself is risk-invariant.

### No-training boundary

The synthetic audit's HistGradientBoosting median map is trained on a separate
`n=4000` sample.  It is an external calibration/alignment resource passed into
Pythia after freezing.  Pythia fits no predictor in the certification sample.
An operational certificate must report the shift's data source and provenance.

## What the supplied evidence does and does not show

The supplied 50-row audit reproduces its summary: at the oracle aggregate
discrepancy, the independent-shift route has median bound/gold `0.4362` for
MAE, `0.4685` for MSE, and `0.0850` for Q90, with 0 known-gold exceedances.
These are theorem/geometry checks because the DGP supplies the true
discrepancy.  They are not operational estimates of that discrepancy and do
not create a Mayo certificate.

The uniform pairwise Lipschitz declaration is usually vacuous.  The useful
object is the weaker aggregate barycentric discrepancy profile.  That profile
must come from a defensible external resource, prospective repeats, or an
explicit sensitivity declaration.

## Concentration improvement

The supplied route used Bentkus.  We replaced only the lower-mean engine with
the fixed-time diversified-Kelly betting construction of Waudby-Smith and
Ramdas, keeping the declared-modulus theorem unchanged.  For independent,
possibly non-identical block scores with average mean at most `m`, each Kelly
component has expected capital

    product_i (1-c+c*mu_i/m)
      <= (1-c+c*mean_i(mu_i)/m)^n <= 1,

by AM--GM.  Its equal mixture is therefore an e-value for the average-mean
null.  Inverting it gives the lower endpoint used here.  The implementation was
also checked against the authors' public `confseq` grid convention.

Primary references:

- Waudby-Smith and Ramdas, *Estimating means of bounded random variables by
  betting*, 2020/2024: <https://arxiv.org/abs/2010.09686>
- Authors' reference implementation: <https://github.com/gostevehoward/confseq>
- Bentkus, empirical-Bernstein, and Hoeffding remain registered comparators;
  betting is not described as a new concentration theorem.

## Three-stage frozen evidence

All studies use the existing ten tabular/text mechanisms, `n=800`, an
independent `n=4000` shift sample, residual clip `[-150,150]`, three fixed score
caps, and oracle `P/P2` only as a theorem diagnostic.  Every result below is a
valid 95% route on its own; the studies are not pooled into one confidence
statement.

### Development audit, seeds 113001--113005

The extended dKelly route beat the supplied Bentkus route in 50/50 MAE,
50/50 MSE, 50/50 Q50, 32/50 Q10, and 36/50 Q90 runs, never losing and never
exceeding known gold.  Because it changed both concentration and the block
grid, it was not used as final attribution evidence.

### Factorial audit, untouched seeds 114001--114005

| Loss | Bentkus median bound/gold | dKelly, same grid | Strict wins / ties | Mechanism wins | Gold exceedances |
|---|---:|---:|---:|---:|---:|
| MAE | 0.3981 | **0.5756** | 50 / 0 | 10/10 | 0 |
| MSE | 0.4180 | **0.6493** | 50 / 0 | 10/10 | 0 |
| Q10 | 0.0177 | **0.1549** | 33 / 17 | 6/10 | 0 |
| Q50 | 0.3981 | **0.5756** | 50 / 0 | 10/10 | 0 |
| Q90 | 0.0577 | **0.1993** | 37 / 13 | 7/10 | 0 |

Adding `k=32,64` was worse in every positive run for both Bentkus and dKelly:
the extra multiplicity cost exceeded any witness gain.  The frozen production
candidate therefore keeps `k=(2,4,8,16)` and uses dKelly only.  A joint
Bentkus/dKelly selector is valid but unnecessarily weaker on this suite.

### Confirmation audit, untouched seeds 115001--115005

| Loss | Bentkus median bound/gold | Frozen dKelly route | Strict wins / ties | Mechanism wins | Positive / 50 | Gold exceedances |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 0.3849 | **0.5593** | 50 / 0 | 10/10 | 50 | 0 |
| MSE | 0.3998 | **0.6327** | 50 / 0 | 10/10 | 50 | 0 |
| Q10 | 0.0087 | **0.1475** | 32 / 18 | 6/10 | 32 | 0 |
| Q50 | 0.3849 | **0.5593** | 50 / 0 | 10/10 | 50 | 0 |
| Q90 | 0.0345 | **0.1900** | 36 / 14 | 7/10 | 36 | 0 |

The confirmation reproduces the gain.  Tail-pinball remains zero on some
lognormal, heavy-tail, and heterogeneous mechanisms.  This is expected honest
vacuity: concentration cannot repair a declared discrepancy larger than the
witness.  The result supports a sharper conditional route, not universal
dominance and not identification of raw-data information.

## Mayo status

No Mayo headline number changes.  The supplied Mayo script is a reasonable
pre-registration skeleton, but it has not supplied a clinically justified
upper bound on `P` or `P2`.  It may report `B(P)` and the tipping point after
the shift, cohort, split, features, and blocks are frozen.  A twin-implied
discrepancy is calibrated/model-based, not proved.  Until an external modulus
contract is justified, the Mayo output must remain a conditional sensitivity
curve rather than a certified 28.9-minute claim.

## Files and reproducibility

- `declared_modulus_certificate.py`: audited conditional API.
- `check_declared_modulus_certificate.py`: refusal, alpha-ledger, multi-loss,
  and monotonicity checks.
- `run_declared_modulus_dkelly_ood.py`: registered simulation runner.
- `declared_modulus_dkelly_ood_results.jsonl`: development rows.
- `declared_modulus_factorial_ood_results.jsonl`: attribution rows.
- `declared_modulus_confirmation_ood_results.jsonl`: confirmation rows.

The source bundle was audited from SHA-256
`43166b4056868fd0b36e58ade39af84e67da76b31ff78113ac72844053a4eb95`;
individual file
hashes were recorded during the audit.  No patient-derived data are contained
in these experiments or reports.
