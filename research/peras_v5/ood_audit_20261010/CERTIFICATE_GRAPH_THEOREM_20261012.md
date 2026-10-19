# A certificate graph for nested information structures

## Why this layer is needed

Pythia now has several valid but differently scoped witnesses: exact groups,
partial groups, repeated environments, declared transport/modulus profiles,
dependence routes, and structural-zero refinement.  Selecting whichever number
looks strongest after evaluation is invalid unless every candidate endpoint
is simultaneous.  Metis will eventually enlarge this search space further.

`pythia/certificate_graph.py` implements the missing composition layer.  It is
not a new concentration inequality and it does not infer sufficiency.  Its
candidate systems contribution is compiling heterogeneous evidence into one
auditable certificate/refusal trace.

`pythia/evidence_adapters.py` now provides the typed producer boundary.  A
content-addressed `EvidenceContext` hashes target, population, loss, outcome
scale, clip, and decision timing.  Exact/partial-group and declared-modulus
lower outputs, standalone gap caps, sensitivity profiles, and structural-zero
assertions can enter the graph only under that same problem identifier.
Cross-loss, cross-population, or cross-scale composition is refused.

`pythia/certificate_manifest.py` adds the frozen automation boundary.  It
accepts strict schema-version-1 aggregate JSON, derives a canonical SHA-256,
reconstructs the problem identity and evidence graph, and embeds both hashes
in the output diagnostics.  Draft or unresolved manifests refuse; unknown
top-level or evidence-record fields and non-finite JSON are rejected.  This is
the intended Metis contract: search may propose a new manifest, but changing a
route, cap, alpha event, or assumption changes the hash and requires a new
Pythia compilation.

## Pathwise transfer theorem

Let each node `v` denote an information set and `R_v` its Bayes risk for one
fixed population, outcome, loss, clip/scale, and decision time.  A directed
edge `u -> v` means `v` is richer and supplies evidence that

    R_u - R_v <= U_(u,v).

For any path `v0 -> ... -> vm`, if `L_v0 <= R_v0`, then telescoping gives

    L_v0 - sum_j U_(v(j-1),vj) <= R_vm.

This remains true after truncation at zero.  If all registered base and edge
events hold simultaneously, the maximum over every predeclared path is also a
valid lower endpoint.  A union bound over distinct evidence-event identifiers
provides that simultaneous event.  Reuse of one genuinely simultaneous event
is charged once; conflicting alpha declarations for the same identifier are
rejected.

This theorem is elementary.  The implementation contribution is the typed,
fail-closed ledger and complete selection trace.

## Supported edge semantics

- `structural_zero`: a provenance/design assertion of zero refinement value;
- `statistical_upper`: an independently calibrated upper risk-gap endpoint;
- `declared_sensitivity`: a conditional sensitivity value, including the new
  barycentric modulus profile.

The compiler requires nonnegative gaps, an acyclic graph, evidence provenance,
an allowed claim tier, and an alpha ledger within budget.  It refuses missing
paths, overspending, calibrated/exploratory evidence in a proved/conditional
compilation, and malformed event reuse.  It never interprets a failed learner
or nonsignificant conditional-independence test as a zero edge.

## Relation to existing theory

Blackwell comparison and generalized entropy already establish monotonicity of
Bayes risk under richer information.  Decision-theoretic information/risk
bridges such as Reid--Williamson and Williamson--Cranko likewise predate Pythia.
The graph does not rename those results.  Its narrower role is simultaneous
finite-data composition across representations and evidence resources.

The alpha ledger is ordinary familywise error control.  Betting/e-process and
reusable-holdout methods may later supply event nodes with more flexible search
semantics, but they do not justify unregistered adaptive paths automatically.

## Validation

`tests/test_certificate_graph.py` covers multiple paths/bases, shared events,
structural-zero transfer, alpha overspending, missing paths, disallowed tiers,
cycles, missing provenance, and monotonicity under adding an already-accounted
candidate.

`check_certificate_graph_coverage.py` simulates two base lower endpoints and
three independently calibrated gap upper endpoints for a three-rung ladder.
All five events spend `0.01`, the compiler selects among three paths, and the
known raw-information risk is checked over 5,000 repetitions.  A deliberately
uncorrected six-event registry is refused before compilation.

`check_compiled_augmentation_ladder.py` exercises the typed adapters on nested
target disclosure.  Across 100 repetitions, three losses, and five disclosure
levels (1,500 compiled cases), it observed zero known-risk exceedances, zero
nonmonotone disclosure paths, and zero nonzero bounds at complete disclosure.
Tabular target disclosure and an exactly equivalent frozen text parser share a
structural-zero edge.  A deliberately false zero-gap edge remained positive in
all 300 complete-disclosure negative controls, showing that the compiler
faithfully exposes rather than silently tests a substantive assumption.

## Remaining gap

The compiler and typed adapters make selection and task identity honest; they
cannot manufacture the edge evidence.
For Mayo the key missing edge is still the risk reduction from the richest
registered note representation to all allowed pre-cutoff raw information.
That edge needs prospective randomized acquisition/masking, repeated scenarios,
a clinically justified external modulus, or an explicit sensitivity curve.
Without one, the correct graph output is refusal at the raw-information node.

`research/peras_mayo5/mayo_certificate_manifest_draft.json` demonstrates that
state.  It contains no patient rows, preserves the legacy calibrated anchors,
and refuses with explicit unresolved items for cohort/prediction provenance,
patient/time assumptions, base lower evidence, and the terminal raw-note gap.
