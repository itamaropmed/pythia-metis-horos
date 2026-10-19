# Table-anchored text information: plasmode results (development, 2026-10-17)

Theory: `docs/theory/TABLE_ANCHORED_TEXT_INFORMATION_20261017.md`.

## Setup

- **Text:** 2,357 unique real MTSamples notes.
- **Table:** a synthetic record per note with 20 cells.
  - Age band, from the note's "NN-year-old" when present, else random.
  - Sex, from pronouns when present, else random.
  - Lab flag, made more likely by the note's diabetes mention.
- **Labels:** binary, from a known law p(X, T). 30 runs per regime.
- **Certificate:** the assumption-free lower bound on ΔI(T | X), the text information beyond the table.
- **Accepted comparator:** the model-based "added value", i.e. the Brier score of a cross-fitted logistic regression on the table alone minus the same model on table + TF-IDF text. This is an estimate, not a bound.

## Results

"Pos" counts runs with a certified bound above 0; "viol" counts runs where the bound exceeded the true ΔI.

| Regime | True ΔI | Text-NN pairing: point estimate | Certified ΔI, mean (pos / viol) | Model-guided pairing: certified (pos) | Model added value, mean (runs < 0) |
|---|---|---|---|---|---|
| S0 table only (text adds nothing) | 0.000 | 0.004 | 0.000 (0 / 0) | 0.000 (0) | −0.011 (30) |
| S1 text redundant with the table | 0.005 | 0.002 | 0.000 (0 / 0) | 0.000 (0) | −0.008 (29) |
| S2 text-unique signal (smoking, doses) | 0.030 | 0.014 | 0.000 (0 / 0) | 0.000 (0) | +0.005 (3) |
| S3 synergy (sex × pain, age × length) | 0.136 | 0.051 | 0.001 (4 / 0) | 0.000 (0) | **−0.005 (27)** |
| S4 strong text-unique signal | 0.280 | 0.118 | **0.045 (30 / 0)** | 0.000 (3) | +0.153 (0) |

**Also assumption-free:**
- The two-sided interval for the table floor Ḡ_X covered the truth in 150/150 runs.
- The upper bound on the joint floor Ḡ_XT had 0/150 violations.

## Reading

1. **Valid everywhere.**
   - There were no false certificates when the text adds nothing (S0) or only repeats the table (S1).
   - The point estimate is conservative in every regime: the within-cell drift D_txt is always ≥ 0.
2. **Power is the limit at n ≈ 2,350 binary labels.**
   - Certification succeeds only for strong text-unique signal (S4, 30/30).
   - Synergy (S3) is certified in only 4/30 runs.
   - Model-guided pairing lost power because it splits the data in half.
   - The bound needs larger corpora or multi-class or continuous targets. It scales as 1/√n.
3. **The model-based "added value" misses synergy.** In S3 the text truly adds 0.136, yet the logistic model reports a *negative* added value in 27/30 runs, because it cannot represent the interactions.

   The pairing statistic is model-free. Its point estimate sees half of the synergy (0.051), and it never claims information that is not there.

## Bug found during this work, which also affected the sandwich results

With tied points, the nearest-neighbour list can contain the row itself at a rank above 0. The greedy matchers then created self-pairs, which always agree, so Q was biased downward.

**Fixed in:**
- `multiscale_pair_extrapolation._greedy_perfect`
- `anchored_information._greedy_pairs`

Both now exclude self explicitly and assert the matching is perfect. A unit test with massive ties covers this. All numbers above come from runs after the fix. The affected tie-heavy geometries in `SANDWICH_RESULTS_20261017.md` were rerun and corrected.
