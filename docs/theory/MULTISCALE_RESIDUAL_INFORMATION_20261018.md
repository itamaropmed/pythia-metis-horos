# Multiscale residual text information

## Purpose

Pythia's table-anchored witness estimates information in clinical text beyond
the declared tabular cells. A single text geometry can be uninformative even
when another frozen representation resolves the same notes well. The
multiscale route registers several geometries before outcomes are inspected,
evaluates each at `alpha / J`, and returns the largest lower endpoint.

The code is `pythia/core/residual_information.py` and the public entry points
are `pythia.multiscale_residual_information` and
`pythia.hierarchical_residual_information`.

## Guarantee

For registered geometries (E_1,ldots,E_J), let (B_j) be the anchored
lower endpoint computed at (alpha/J). If every geometry is outcome-blind and
the anchored witness assumptions hold, then


\[
  \Pr\{\max_j B_j \leq V(T\mid X)\}\geq 1-\alpha.
\]


The result is simply a Bonferroni composition of valid lower bounds. It is not
a new concentration inequality and cannot be selected after looking at the
evaluation outcomes without paying additional error budget.

The optional `resolution` alpha policy allocates the ledger using only cell
occupancy, pair support, singleton fraction, and geometry dimension. It never
reads outcomes. A parent-cell route can pool sparse child cells, but it must
receive an externally declared upper bound on the Brier-scale information
attributable to the child-table refinement. The returned lower bound is
`max(parent_lower - variation_penalty, 0)` and is therefore conditional rather
than silently optimistic.

The API records a `conditional` tier because it cannot independently verify
that a supplied text encoder was frozen before outcomes or that the entities
are conditionally independent. Missing provenance, malformed geometries, and
empty ladders return structured refusal.

## Fresh proof-of-concept results

### Frequent text signal

Ten fresh seeds (`93000`--`93009`) were generated with six tabular cells and a
binary text concept changing the class probability from 0.1 to 0.9. A noisy
four-dimensional geometry and the concept geometry were registered in advance.

- The informative concept geometry was selected in **10/10** runs.
- The multiscale lower bound was positive in **10/10** runs.
- Mean single-concept lower endpoint: **0.2342**.
- Mean multiscale endpoint: **0.2271**.

The small decrease is expected: the multiscale certificate spends half of
alpha on each candidate. The gain is selection safety and robustness to a bad
geometry, not a guaranteed numerical increase over a geometry chosen in
advance.

### Fresh generated OOD table-plus-text families

On four new 6,000-row runs (`918273`--`918276`), the unified Pythia
certificate satisfied every checked inequality: joint ceiling, table box,
NLI lower bound, population interval, and no false assumption refutation.

- A numbers-in-words family had exact residual text information around
  `0.058`; Pythia lower endpoints were `0.0213` and `0.0279`.
- A rare-token family had exact residual information around `0.022`--`0.0235`,
  but the multiscale lower endpoint was zero in both runs.

The zero rare-token result is correct behavior under the present resolution:
the signal is too sparse for reliable within-cell pair evidence. Multiscale
selection cannot create information that the finite sample does not resolve.

### Resolution and hierarchy POC

On a fresh 2,400-row frequent-signal run, resolution weighting selected the
registered concept geometry with alpha allocation `[0.01306, 0.03694]`; the
noise geometry was marked `insufficient_resolution` and the concept geometry
`positive_certificate`. The multiscale lower endpoint was `0.2244`.

With a predeclared parent-cell variation penalty of `0.015`, the hierarchical
route returned a parent endpoint `0.2207` and a child-table residual endpoint
`0.2057`, marked `positive_certificate`. The penalty came from the known
synthetic contract; it was not estimated from the evaluation outcomes.

On a fresh rare-token run, the endpoint was zero but the status was explicitly
`insufficient_resolution` (2,492 usable pairs), not `compatible_with_zero`.
On a numbers-in-words run, the endpoint was `0.00175` and the status was
`positive_certificate`. This distinction prevents a zero lower bound from
being misreported as proof that notes contain no information.

### Registered-support rare-signal route

Pythia now has a separate `support_residual_information` route for a finite
text support registry frozen before outcomes. It contrasts each non-baseline
support against the table within an orthonormal Helmert basis of outcome
space. Both signs and every registered contrast are in the alpha ledger; the
orthogonal outcome-axis projections may then be added. This is model-free and
does not ask a generic embedding to discover a rare concept.

On 20 fresh O6 rare-token runs at `n=6000`, the route was positive in **20/20**
signal runs, positive in **0/20** matched null runs, and had **0 known-law
violations**. Mean true residual Brier information was `0.022534`; mean lower
endpoint was `0.002611`, with median captured fraction `0.1110`. The Helmert
upgrade doubled the earlier single-coordinate endpoint exactly as predicted
for binary Brier geometry. This closes the qualitative zero-bound failure but
is still quantitatively conservative.

A ten-design phase diagram then varied prevalence from 1% to 20% and logit
signal from 1.5 to 3.0. Across five fresh seeds per design, registered support
beat generic concept matching in 49/50 runs with 0/50 truth violations; see
`docs/experiments/RARE_SIGNAL_PHASE_DIAGRAM_20261018.md`.

## Next improvements

1. Add outcome-blind, predeclared alpha weights based on geometry resolution
   (cell occupancy, duplicate rate, and distance concentration), rather than
   equal splitting. The weights must be chosen without outcomes.
2. Calibrate the hierarchical penalty from independent environments or
   externally justified table smoothness; the current synthetic penalty is not
   transferable to Mayo.
3. Add a hierarchical rare-signal route that pools registered neighboring
   table cells while charging a declared cell-variation/modulus penalty.
4. Add a support/refusal threshold so a zero endpoint is distinguished from a
   genuine null text effect.
5. Predictor and learned-geometry fitting now live in
   `horos/models/pythia_adapters.py`; keep the passing architecture guard as a
   permanent boundary test.
6. Validate the weighted and pooled variants on fresh null, frequent-signal,
   rare-signal, negation, interaction, shuffled-text, and target-leakage
   generators before any Mayo transfer.

These steps preserve the central principle: tighter bounds must come from
declared information, geometry, or support resources, never from silently
spending validity after observing the outcomes.
