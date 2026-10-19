# PYTHIA-X: strengthening the four weak points (development, 2026-10-16)

**Status: development.** Nothing here is pre-registered.
- Theory: `docs/theory/PYTHIA_X_STRENGTHENING_20261016.md`.
- Runs: 30 per design.
- Classification: n = 4000.
- Deep levels: n = 4096.

**Reading the cells.** "a/b" means the mean bound divided by the floor (a), and the number of runs above the floor (b).

**One bug found and fixed during the run.** `certificate_v2` did not skip level pairs where the "coarse" level had the smaller lag (t ≥ 1). This produced spurious values in the first deep-level run. The guard is now in place; the earlier `extrapolation_certificate` already had it. All numbers below come from runs after the fix.

## W3: is validity only slack? (checked first, since it frames the rest)

The expectation-level overshoot (E Q_a − t·E Q_b)/(1 − t) − Ḡ is computed from the true laws, with no sampling and no Jensen step.

| | linear2 | smooth5 | sharp2 | wiggle2 | sparse10 | checker_coarse | checker_fine | rowwise_random |
|---|---|---|---|---|---|---|---|---|
| Overshoot, γ = 1 | −0.0000 | +0.0003 | +0.0001 | +0.0002 | −0.0003 | **+0.014** | +0.079 | +0.080 |
| Overshoot, γ = ½ | −0.0008 | −0.0061 | −0.0021 | −0.0053 | −0.027 | −0.0008 | +0.078 | +0.080 |
| Brier-floor certificate, γ = 1 (no Jensen) | 0.88/0 | 0.83/0 | 0.78/0 | 0.87/0 | 0.77/0 | 0.94/**3** | 1.09/30 | 1.08/30 |
| Brier-floor certificate, γ = ½ | 0.80/0 | 0.66/0 | 0.62/0 | 0.78/0 | 0.50/0 | 0.84/0 | 1.02/20 | 1.01/20 |

**Answer**
- At γ = ½ the assumption holds *in expectation* in every resolvable design. It still holds on the boundary design, with a small margin. So validity there does not rest on slack.
- γ = 1 sits exactly on the edge for smooth laws, as Proposition 2 predicts (D ∝ s gives equality). It fails in expectation for boundary laws.
- The Brier-floor certificate needs no Jensen step and is a new target in its own right. It had 0 violations at γ = ½ in all resolvable designs.

## W2: choosing γ

**Deep levels, 0-1 loss** (7 levels; the 4-level version is shown after the slash).

| | linear2 | smooth5 | sharp2 | wiggle2 | sparse10 | K4_3d | checker_coarse |
|---|---|---|---|---|---|---|---|
| γ = 1 | 0.76/0 (0.73/0) | 0.71/0 | 0.63/0 | 0.80/0 | 0.64/0 | 0.83/0 | **1.03/20** (0.89/0) |
| γ = ½ | 0.71/0 (0.65/0) | 0.59/0 | 0.53/0 | 0.72/0 | 0.41/0 | 0.78/0 | **0.87/0** (0.73/0) |

**What we learned**
- **γ = ½ is the safe default** (Proposition 2: a law that is smooth with jumps has D = c₁s + c₂s^{1/2}). It had 0 violations in every resolvable design, including the jump law at deep levels, where γ = 1 failed in 20/30 runs.
- **Deep levels tighten every bound** (+0.03 to +0.14). They also expose γ = 1 on boundaries faster, because the wider lag window adds saturation.
- **The increments test is a negative result.** It made 0 rejections anywhere, even where A(1) fails. It stays as a diagnostic only.

## W1: adversarial non-identifiability. Test-retest substudy (m rows re-labelled)

**Falsification power** at γ = 1 and γ = ½, out of 30 runs. On the 6 resolvable designs, 1 of 1,620 tests (6 designs × 3 substudy sizes × 30 runs × 3 γ values) falsely rejected: sparse10 at m = 400, γ = 1.

| Substudy size m | checker_fine | rowwise_random | Substudy-only bound (assumption-free, 0 violations everywhere) |
|---|---|---|---|
| 400 | 18 / 12 | 21 / 14 | 0.31–0.66 |
| 800 | **27 / 19** | **23 / 18** | 0.43–0.74 |
| 1600 | 22 / 15 | 26 / 16 | 0.54–0.83 |

**Pipeline** (the largest γ ≤ ½ not falsified, at α/2, combined with the substudy bound at α/2):
- **Resolvable designs:** 0 violations at every m. Tightness is 0.46–0.82, with γ chosen by the data.
- **Adversarial designs:** violations fell from 30/30 to 2–7/30 at m = 800 (9–10/30 at m = 400, 7–11/30 at m = 1600).
- **The assumption-free tolerance holds.** Whenever γ is *not* rejected, the overshoot on the Brier scale stayed below the computed τ in every run. τ is 0.12–0.16 at m = 800 against a realized overshoot of 0.08. The guarantee is real but loose.
- **Split size.** m ≈ 20% of rows gave the best power. Larger m weakens the extrapolation half.

## W4: squared loss

Fixed-bet capital bounds versus Chernoff–KL and Bernstein.

| | mse_uniform | mse_gauss | mse_hetero |
|---|---|---|---|
| KL, γ = 1 (old) | 0.43/0 | 0.14/0 | 0.19/0 |
| Bernstein, γ = 1 | 0.61/0 | 0.15/0 | 0.22/0 |
| **Betting, γ = 1** | **0.86/0** | **0.78/0** | **0.76/0** |
| Betting, γ = ½ | 0.71/0 | 0.48/0 | 0.50/0 |
| Betting, declared range ×4, 4 levels | 0.46/0 | 0.03/0 | 0.06/0 |
| **Betting, declared range ×4, 7 levels** | — | **0.64/0** | **0.66/0** |
| Betting, declared range ×1, 7 levels, γ = 1 / γ = ½ | — | 0.86 / 0.69 | 0.82 / 0.67 |

The range now enters only the t-weighted far level. With deep levels (t ≈ 0.01–0.03), a range declared 4× too wide still gives 0.64–0.66 instead of 0.03–0.06. The lower side is range-free.

## Net status of the four weak points

| Weak point | Before | Now |
|---|---|---|
| Adversarial | 30/30 violations, no recourse | Proposition 1 proves no assumption-free fix exists without replicates. With a 20% retest substudy: an assumption-free bound (0.43–0.74), falsification power 60–90% at γ = 1, adversarial violations down to 2–10/30, and a computable tolerance τ that held in every run |
| γ | Rule of thumb | Proposition 2: γ = ½ is provably safe for smooth-plus-jump laws; 0 violations, deep levels included. γ = 1 is on the edge for smooth laws and fails on boundaries. The increments test failed |
| Slack | Validity partly from Jensen | The expectation-level check shows A(½) holds without slack in all resolvable designs. The Brier-floor target removes Jensen entirely |
| Squared loss | 0.36 | 0.76–0.86 at γ = 1 and 0.48–0.71 at γ = ½. Robust to over-declared range with deep levels |
