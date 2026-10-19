# Support-free clustered block-SAA confirmation registration

Frozen on 2026-10-14 after development seeds in the 300,000--500,000 ranges
and before opening confirmation seeds `1,200,001`--`1,200,500`.

## Claim under test

For each outcome-blind partition into disjoint patient blocks, the blockwise
finite-cell empirical optimum is a nonnegative random score whose expectation
does not exceed the corresponding fixed-design Bayes risk.  After scaling each
block by its contribution to the full case-weighted target, the average score
expectation does not exceed the full target risk.  Truncating the score at a
frozen cap can only decrease that expectation.  A bounded-mean dKelly lower
endpoint therefore lower-bounds the original, untruncated Bayes risk without a
population outcome range.

Routes may reuse patients across candidate block sizes, but every route uses
disjoint blocks internally.  Bonferroni protects selection across the complete
route-by-cap menu.

## Frozen design

- Ten symmetric, unbounded noise laws with analytic median absolute error:
  Gaussian, Laplace, logistic, Student-t3, Student-t5, Gaussian scale mixture,
  symmetric Gaussian location mixture, signed lognormal, signed exponential,
  and contaminated Gaussian scale mixture.
- 50 fresh seeds per law; 500 draws total.
- 4,096 independent patients, each copied to two cases.  Within-patient
  dependence is therefore maximal while the case-weighted and patient-weighted
  target distributions coincide.
- Eight frozen representation cells with deterministic cell-dependent scale
  and location.
- Registered block sizes: 16, 32, 64, 128, 256 patients.
- Registered score caps: 0.25, 0.5, 1, 2, 4, 8 original MAE units.
- Family alpha 0.05.  The block method spends alpha over all 30 candidates.

## Comparator

The matched support-free comparator forms outcome-blind disjoint pairs within
each cell and uses the classical `|Y-Y'|/2 <= Bayes MAE` witness.  It uses the
same six cap candidates and its own 0.05 family alpha.  Both routes use dKelly;
the comparison is witness geometry, not a concentration mismatch.

## Primary gates

1. zero block-SAA endpoints above analytic gold;
2. Wilson 95% lower validity at least 0.95;
3. positive block endpoints in all 500 runs;
4. block-SAA strict wins over the pair comparator in a majority of runs and a
   majority of the ten mechanism medians;
5. report all losses, zeros, cap selections, and median bound/truth ratios.

This study tests a support-free MAE construction at Mayo-scale patient count.
It does not establish temporal exchangeability on Mayo or transfer the theorem
to arbitrary learned representations.
