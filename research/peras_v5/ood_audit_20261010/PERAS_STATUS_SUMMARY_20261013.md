# Pythia / Metis / Horos — where we stand (2026-10-13)

This is a code audit of `~/PycharmProjects/peras-poros-horos`. The test suite was run on the device. It is meant to be honest, not promotional.

## 0. Bottom line

- **Pythia (the theorem/certificate engine)** is real, tested and moving fast. 279 tests pass; 8 are skipped because they need private Mayo artifacts.
  - Between Oct 10 and Oct 13 it gained the following:
    - a certificate-graph compiler;
    - finite-policy information-gap edges;
    - original-scale truncated-SAA certificates;
    - non-oracle repeated-environment modulus calibration;
    - hard-support range refinement;
    - a closed-out Q90 multi-point line.
- **No Mayo number is certified.** The Mayo manifest is a *draft that refuses by design*. The anchors (28.85 guarded, 31.12 floor estimate, 32.87 best OOF) remain `calibrated_retrospective`. The 31.2623 incumbent is still **not reproduced**.
- **The central open problem is unchanged.** We cannot yet bound R*(registered representation) − R*(all allowed raw notes), and the near-replicate discrepancy, from ordinary one-outcome-per-case data. Every new route either:
  1. needs a resource Mayo does not have (repeated environments, hard supports); or
  2. turns the gap into a *declared* sensitivity curve.
- **Metis is empty.** All its files are 0-line `__init__.py` stubs. **Horos** has about 4.9k lines (benchmark, ingest, report, API skeleton) and is blocked on the same M2 gate.
- **Repository risk.** Git has **one commit** (`85af644`). Everything since Sep 14 is uncommitted or untracked: about 10 new core modules, 10 new test files and all of `research/`. There is **no remote**, so the CI workflow has never run. That much unbacked work on one laptop is the single largest practical risk right now.

## 1. What is established (theorem plus confirmed audit)

| Component | Status | Key evidence |
|---|---|---|
| Finite-group / partial-cell SAA lower certificate | Proved under iid + exact cells | Earlier audits; the strongest current paper contribution |
| Repeated-environment direct SAA (`E inf ≤ inf E`) | Proved; needs real repeated target environments | Median bound/floor 0.79–0.86, coordinate-unit router up to 0.97; 0 exceedances |
| Barycentric coupled penalty (MAE ×1, every pinball ×½, MSE squared) | Proved; verified by known-law checks | 0/360 violations; makes MSE certifiable |
| Declared aggregate discrepancy curve B(P) + tipping point P\* | Proved as a *conditional* curve | Oracle-discrepancy checks only; it does not identify P |
| Feature-modulus adapter (P ≤ ΛG, P₂ ≤ Λ₂²H) | Accepted as a sufficient condition | API computes G and H; never fits the shift, never estimates Λ |
| Non-oracle modulus calibration from repeated environments | Confirmed | 0/1,000 misses per penalty; abs cap/truth 1.996; squared 47.0 (loose) |
| Hard-support vertex range | Confirmed | Cap/truth 1.091 (abs), 2.009 (squared). Needs **genuine** per-unit hard supports |
| Original-scale score truncation + dKelly | Confirmed | Median bound/gold 0.924 MAE, 0.845 Q90, 0.862 MSE; Catoni and Cantelli lose |
| Finite-policy information-gap edge (no outcome alphabet) | Proved (classical uniform convergence); confirmed | Beat joint-L1 and CMI/Pinsker in 300/300 cells; Q10/Q90 router with coordinate Bernstein |
| Certificate graph + manifest + typed evidence adapters | Implemented and tested | 5,000-repetition audit 0 exceedances; uncorrected registry refused; false zero-gap control fails 300/300 |

**The honest novelty position, as the repo's own gap map states it:**
- The inequalities are mostly classical.
- The defensible contributions are:
  1. loss-routed finite-group certification with structured refusal;
  2. loss-specific barycentric penalties;
  3. the **certificate compiler**: compositional validity over nested information structures, with one alpha ledger and fail-closed provenance.

## 2. Negative results (kept visible, which is correct)

- Uniform Lipschitz modulus declarations are vacuous (0–4/50 positive). Discrepancy must be stated in **aggregate** form.
- Location-scale alignment loses to location-only alignment (0/50 tighter).
- Q90 multi-point routes:
  - **Entity-McDiarmid overlap** loses to disjoint dKelly 1,000/1,000.
  - **Partition e-value mixture** gives 0.8213 vs 0.8192; it wins 8/10 mechanisms, p = 0.055, so it is not significant.
  - **Geometry-weighted mixture** loses to the single minimum-penalty partition on 9/10.
  - The line is closed. **Disjoint dKelly stays the Q90 default.**
- Squared-penalty calibration without hard supports is range-limited (cap/truth ≈ 47).
- **Finite gap edge on Mayo MAE: refused.** Even the improved finite-policy route pays at least 44.2 min of uncertainty for a single binary concept. The clinically relevant floor range is only 25–32 min.

## 3. What Mayo can and cannot use today

| Route | Usable on current Mayo? | Why |
|---|---|---|
| Repeated-environment SAA / calibration | No | One outcome per case |
| Hard-support range | No | Only a broad 0–900 min operational domain; fitted intervals are not hard supports |
| Declared-modulus curve | Yes, but **conditional only** | Produces B(P) and P\*; no identified P |
| Finite-policy gap, MAE | No | Uncertainty ≥ 44 min (vacuous) |
| Finite-policy gap, **binary endpoint** | **Plausible** | Uncertainty 0.046–0.073 probability with 8 families and one concept |
| Twin-implied P | Calibrated anchor only | Model-based, not a proof |

Housekeeping blockers before *any* Mayo computation:
- the cohort hash is unresolved;
- the honest OOF incumbent has not been regenerated;
- the loader hard-codes the wrong `~/mnt/...` path (it should come from configuration);
- the patient/time sampling contract is unfrozen;
- the fold-loader still allows a prediction hash to stand in for a representation hash.

## 4. Recommended next step

### Step A — housekeeping (immediately, about an hour)

1. Commit the Oct 10–13 work: new `pythia/` modules, tests, and `research/` text and code. Keep the large `.jsonl` results in an artifact store, or record their SHA-256 hashes only.
2. Add a private remote so the CI workflow runs.
3. Fix the Mayo data path to come from an environment variable or configuration.

### Step B — the next Pythia theorem: a soft-support (bounded-moment) barycentric range

This is item 1.5 of the plan, and the natural next theorem.
- **The gap it fills.** The hard-support vertex theorem turned the squared calibration from 47× to 2× truth, but it needs hard per-unit supports, which real data never has.
- **The target.** Replace hard supports with a *declared, auditable* tail condition. For example, for every unit, E|V − s|^p ≤ m_p, or a sub-Gaussian / Bernstein envelope after the frozen shift.
- **The proof idea**, mirroring the original-scale truncation result:
  1. Truncate the barycentric cost at a predeclared level τ.
  2. Charge the excess by the moment bound: E[(C − τ)₊] ≤ m_p / ((p − 1)·τ^{p−1}) (Markov-type tail integral).
  3. Choose τ outcome-blind from (k, p, m_p, n, α).
  4. Concentrate the truncated cost with dKelly under the same alpha ledger.
- **Deliverable.** A typed lower endpoint `soft_support_barycentric_lower_bound` that enters the certificate graph, plus a frozen development audit, then an untouched confirmation, on heavy-tail and skew mechanisms.
- **Comparators:**
  - the global-range route (the 47× squared cap);
  - the hard-support route (the 2× oracle optimum);
  - the moment-free original-scale truncation route.
- **Go criterion.** Zero exceedances, and a median squared cap/truth well below the global range on at least 8/10 mechanisms.
- **Honesty check.** This still needs the *calibration environments* resource. It improves what repeated environments can certify; it does **not** by itself certify Mayo.

### Step C — in parallel, the first *real* compiled certificate

The finite-policy edge is the only gap route whose uncertainty is already clinically small on Mayo-sized data, and only for a binary endpoint. So the fastest path to a genuine, non-draft end-to-end certificate is:

1. Pre-register one binary operational endpoint, for example "case overruns its scheduled duration by more than 30 min". Freeze it before looking at any outcomes.
2. Declare complete alphabets from design files, never from outcomes:
   - 8 input-defined procedure families;
   - one binary note concept.
3. Run `finite_policy_gap_resolution` outcome-blind. Proceed only if the preflight uncertainty is at most 0.05.
4. Freeze the cohort hash, the patient/time cluster contract and the OOF provenance.
5. Compile the manifest with `status: frozen` → Pythia certificate graph → certificate or structured refusal.

This produces the first `proved`-tier clinical certificate on a narrow, honest question. It also exercises the whole manifest → graph → refusal machinery that Metis will later drive.

### Deliberately deprioritized

- Further Q90 partition-mixture work: closed.
- Any broad "universal raw-text floor" claim.
- Metis autonomy, until a frozen manifest exists for it to consume.

## 5. Paper options (most defensible first)

1. **Narrow, near-ready paper:** finite-group / partial-cell certification plus the finite-policy gap edge plus the certificate compiler, with refusal semantics. Optionally add the binary Mayo endpoint from Step C as the application.
2. **Medium paper:** add barycentric penalties with repeated-environment and soft-support calibration (needs Step B).
3. **Not yet defensible:** a Mayo MAE floor certificate, a raw-note sufficiency claim, or SOTA estimator claims.
