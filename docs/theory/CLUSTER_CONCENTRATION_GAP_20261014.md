# Cluster concentration gap and development map

## Established Pythia statement

Let independent patient objects be `Z_1,...,Z_G`; each object may contain an
arbitrary number of mutually dependent cases.  Conditional on a frozen cell
map and fixed cluster memberships, let `S(Z_1,...,Z_G)` be the case-weighted
empirical finite-cell optimum.  Empirical minimization gives

```text
E[S] <= R_cell.
```

If replacement of patient `g` can change `S` by at most `c_g`, bounded
differences gives, with probability at least `1-alpha`,

```text
R_cell >= E[S] >= S - sqrt(log(1/alpha) * sum_g c_g^2 / 2).
```

For a cell-local hard loss cap `M_i`, Pythia uses
`c_g = sum_{i in g} M_i / n`.  This is the implemented theorem.  Its novelty
claim must be limited to the certificate composition and its representation
contracts; McDiarmid's inequality itself is classical.

## What cannot be imported automatically

1. **Bentkus.** Bentkus gives sharper comparisons for sums and martingales of
   bounded increments.  The existing Pythia Bentkus routine is an inversion for
   an additive vector of independent scores.  The finite-cell optimum is a
   nonlinear function of all patients, so reusing that routine would be an
   unjustified theorem transfer.
2. **Empirical Bernstein.** Maurer--Pontil applies directly to independent
   bounded observations.  A sample variance of patient rows is not a variance
   estimator for the optimized SAA functional.
3. **Bounded-mean betting.** Modern betting intervals are extremely sharp for
   independent bounded means, but the same obstacle remains: one observed SAA
   optimum is not an additive score sequence.
4. **Row bootstrap or row IID.** Repeated rows do not add independent
   information.  The frozen stress experiment obtained only 638/1,000 coverage
   after copying every patient outcome to 100 rows.

## Promising theorem routes

### A. Weak-interaction variance certificate

Maurer's variance estimator for functions of independent variables can turn a
Bernstein inequality for a nonlinear function into an empirical bound when
first- and second-order interactions are controlled.  Pythia would need:

- an exact first-order sensitivity for unequal patient sizes;
- a useful mixed second-difference bound for the cellwise MAE/pinball optimum;
- an extension or conditioning argument for non-identically distributed
  patient objects and unequal cluster sizes;
- an efficient replacement/deletion computation rather than an unexamined
  quadratic algorithm.

This is the highest-value theoretical route, but it remains a theorem
candidate—not production code.

### B. Independent block-SAA portfolio

Partition patients outcome-blindly, compute one empirical optimum per disjoint
block, and apply bounded-mean betting to the resulting independent scores.
Expectation remains below Bayes risk.  Development experiments show a genuine
tradeoff: block scores repair the rare-tail zero but lose substantial
tightness on broad-noise laws because small blocks increase optimization bias.
It should remain a simultaneous complementary rung, not replace the current
endpoint.

### C. CDF-envelope integration for ordered losses

For a univariate outcome with CDF `F`, the optimal pinball risk has the exact
representation

```text
R_tau = integral min((1-tau) F(t), tau (1-F(t))) dt.
```

MAE is the corresponding `integral min(F(t),1-F(t)) dt`.  A simultaneous CDF
band can therefore be mapped to a lower risk endpoint by minimizing the tent
integrand pointwise over the band.  The construction is attractive for
MAE/quantiles, but ordinary DKW assumes IID scalar outcomes.  A useful Mayo
route requires a weighted/cluster empirical-process theorem rather than
silently collapsing multiple cases per patient.

## Modern concentration references

- Vidmantas Bentkus, *On Hoeffding's Inequalities*, Annals of Probability 32
  (2004), DOI `10.1214/009117904000000360`.
- Andreas Maurer and Massimiliano Pontil, *Empirical Bernstein Bounds and
  Sample Variance Penalization*, COLT 2009.
- Andreas Maurer, *A Bernstein-type inequality for functions of bounded
  interaction*, 2017.
- Andreas Maurer, *A bound on the deviation probability for functions with
  weak interactions*, PMLR 75, 2018.
- Ian Waudby-Smith and Aaditya Ramdas, *Estimating means of bounded random
  variables by betting*, JRSS-B 86 (2024), DOI `10.1093/jrsssb/qkad009`.

The research gap is consequently precise: build an observable,
variance-sensitive lower confidence endpoint for a nonlinear conditional
Bayes-risk witness under independent clusters, unequal cluster sizes, and
arbitrary within-cluster dependence.  That would be a substantive theorem;
relabeling a standard bounded-mean inequality would not.
