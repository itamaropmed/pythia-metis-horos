# Registered finite-text intervention battery

Written before generating results.

## Data

Ten bounded known-law mechanisms: Gaussian-like, uniform, triangular, clipped
t3, clipped lognormal, bimodal, heteroscedastic, high-SNR, low-SNR and
discrete-mixture noise. Each has 20 latent text/tabular cells and a known
cell-conditional Bayes MAE. Seeds 61001--61050 draw 2,048 iid entities, split
equally into an external action-learning half and a certification half.

## Representations

1. baseline cell text;
2. bijective cipher of the cell token;
3. independent nuisance token ignored by a canonical parser;
4. independent nuisance token retained by a naive parser;
5. deterministic five-family garbling;
6. within-family note reassignment;
7. globally misaligned note cell;
8. token shuffle interpreted by a bag-of-token parser;
9. literal exact-target sentence;
10. unary target encoding by target-proportional `X` count.

The first eight use `finite_cell_saa_lower_bound`. Exact-target and unary-target
representations must be refused because their labels are outcome-derived; their
diagnostic gold and external estimator are zero.

## Endpoints

- true MAE floor under the transformed finite representation;
- 95% finite-cell SAA lower endpoint;
- external split cell-median risk estimate (not a Peras certificate);
- positivity, singleton fraction and effective cell size;
- metamorphic contract result for the known gold risks.

Each transformation is a separate 95% procedure. Transformations are compared
for diagnosis, not adaptively selected into a 95% maximum. Report zero-floor
cases separately when calculating tightness. Mechanisms, not the 50 repeated
seeds, are the unit for win/significance summaries.
