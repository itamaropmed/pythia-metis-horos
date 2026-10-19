# Dependency-graph domination: Bayes-floor certificates without iid, independent clusters or stationarity

## Gaps it closes

`DOMINATED_SAA_20261014.md` needed three assumptions that Mayo-like data violate:

1. **Independent patients.** In reality there are surgeon-day, room-day and calendar-day shocks.
2. **A stable population over the period.**
3. **Equal-range clusters.** Bentkus needs a common range.

## Theorem 1: realized-period estimand (removes stationarity)

**Setting**
- Each case i has its own law Q_i, which may drift arbitrarily in time and case mix.
- The cells c(i) are fixed in advance.

**Definition.** The target is

  R = (1/n) Σ_c min_a Σ_{i ∈ c} E_{Q_i} ℓ(Y_i − a).

This is the smallest expected loss, over the realized cases of the period, of any rule that maps the registered information to an action. If time windows are part of the cells, the rule may adapt in time. This is exactly the floor relevant to a deployed scheduling rule.

**Claim.** The fixed-action domination of `DOMINATED_SAA` applies verbatim, with a_c the pooled minimizers:

  S ≤ W̄ and E W̄ ≤ R.

No stationarity of P(X) or P(Y | X) is used.

## Theorem 2: graph-Chernoff domination (removes cluster independence)

**Assumption (dependency graph, Janson 2004).** Cases are dependent only if they share a declared unit value: patient, surgeon-day, day, week, and so on. Equivalently, Y_i = f_i(ξ_{units(i)}, ε_i), where the unit latents ξ and the case noises ε are independent.

**Statement.** Let a proper colouring of the graph use K classes. Then for every λ > 0,

  E exp(λ n W̄) ≤ (1 + p̄ (e^{λ K κ} − 1))^{n/K}, where p̄ = E W̄ / κ.

Hence

  P(S ≥ s) ≤ exp(−(n/K) · KL(s/κ ‖ p̄)),

and inverting at the observed S lower-bounds R.

**Proof.**
1. **Generalized Hölder over the K classes:** E Π_k Z_k ≤ Π_k (E Z_k^K)^{1/K}. No dependence between classes is assumed.
2. **Independence inside a class.**
3. **Bernoulli extremality** of [0, κ] variables (Hoeffding 1963): E e^{sW} ≤ 1 + p(e^{sκ} − 1).
4. **Concavity of p ↦ log(1 + p x)**, used twice: within classes and across classes of unequal size.
5. **Chernoff.** ∎

**Properties**
- K = 1 recovers the independent-row Chernoff bound.
- A patient-only graph gives K = the maximum number of cases per patient.
- Classes need not be equal in size, which removes the common-range problem.
- The greedy colouring gives K ≥ the maximum clique ω. When K = ω, the colouring is optimal; this held in every run.

**Retention.** Keeping the first m cases of each unit (outcome-blind) shrinks the cliques. Dropped cases contribute zero, which is safe. Every retention choice is charged in the alpha ledger.

## Algorithm (PYTHIA-G)

1. **Temporal split.** Use the past as the development set and the most recent period as the certification set.
2. **Dependence audit (development only).** For each candidate unit:
   - centre outcomes at their cell medians;
   - permute within cells;
   - test whether the between-unit sum of squares exceeds the permutation null.

   Units that reject are declared in the graph, and units that don't are still allowed to be declared. This is falsification, not proof.
3. **Freeze** the cells, unit list, cap grid, retention grid and α.
4. **Certify (certification data).** Compute the exact truncated cellwise SAA, colour the graph, invert the Chernoff–KL bound, and take the maximum over the charged candidates.
5. **Sensitivity.** `dependence_sensitivity_curve` reports the bound B(K) against an assumed inflation K and the tipping K\*. This guards against undeclared dependence: the bound stays valid for any K ≥ the true fractional chromatic number.
6. **Compose** with the lattice, the fixed-policy value bridge and the curvature penalty, as before.

## Development evidence (synthetic, known realized-period floor)

**Simulated surgical schedules (100 reps each, n ≈ 3,100–3,500).** Each simulation has surgeon-day shocks, patient effects, repeat patients and cell means drifting with time.

| Scenario | Graph valid | Graph bound/gold | Patient-Bentkus bound/gold | Row-iid bound/gold |
|---|---|---|---|---|
| No shock | 100/100 | 0.780 | 0.793 | 0.883 |
| Drift only | 100/100 | 0.825 | 0.834 | 0.920 |
| Shock (moderate) | 100/100 | 0.823 | 0.829 | 0.913 |
| Shock (strong) | 100/100 | 0.824 | 0.826 | 0.907 |
| Shock (dominant), big days | 100/100 | 0.726 | 0.827 | 0.915 (99/100 valid) |

In these scenarios the slack from truncation and optimism usually hides the dependence. The falsification test below removes that slack.

**Day-level shocks shared by every case of a day (200 reps, n ≈ 2,800, K = 44):**

| Loss | Graph valid | Row-iid valid | Graph bound/gold | Row-iid bound/gold |
|---|---|---|---|---|
| Window-miss | **200/200** | **176/200 (88%)** | 0.722 | 0.930 |
| MAE | **200/200** | **181/200 (90.5%)** | 0.574 | 0.908 |

The row-iid and patient-cluster routes are invalid under shared shocks. The graph route is valid and pays √K in width. This is the price of honesty, and the dependence audit plus the sensitivity curve make that price visible.

## What remains assumption-bound (no free lunch)

- **Dependence outside the declared units.** It is not identifiable in general. The audit can falsify a missing unit but cannot prove completeness. B(K) and its tipping point turn this into an explicit dial.
- **Near-replicate merging of continuous information** still needs a declared second-order dispersion V (curvature theorem).
  - Considered and rejected: certifying V from neighbour brackets under comonotonicity of a learned score. Validity would hinge on a *supremum* ranking error, which a finite learner essentially never satisfies.
- **Within-cell optimism** cannot be removed without shape assumptions: the worst-case k = 2 MAE ratio is 1/2. It only makes the bound conservative.
