# PYTHIA-X known-signal suite (development, 2026-10-16)

**Status: development.** Nothing here is pre-registered. Next step: freeze the code and registration, then confirm on untouched seeds.

**Setup**
- Synthetic designs: n = 4000, X ~ U[0,1]^d, 30 runs each.
- MTSamples: 2,352 unique notes in TF-IDF→SVD(100) geometry, planted law, 30 runs per variant.
- PMLB pharynx: real covariates (one-hot categorical plus Age), planted law, n = 4000, 20 runs.

**Reading the cells.** Each cell is "mean bound/floor ÷ number of runs above the floor". For the bound columns, a run above the floor is a violation. For the two estimate columns (1-NN Cover–Hart, Gamma-test intercept), it is an overshoot, since those are not bounds. "Oracle L" is the true maximum realized TV/distance on the level-1 pairs. That is generous to the accepted route, which in practice must guess L. The ×0.5 and ×2 columns misdeclare it.

| Design | Floor | Oracle L | **PYTHIA-X** (γ ∈ {1, ½}) | γ = ½ only | γ = ¼ | Accepted, L ×0.5 | Accepted, oracle L | Accepted, L ×2 | 1-NN estimate | Gamma-test estimate |
|---|---|---|---|---|---|---|---|---|---|---|
| linear2 | 0.291 | 1.06 | 0.74 / 0 | 0.67 / 0 | 0.51 / 0 | 0.80 / 0 | 0.79 / 0 | 0.79 / 0 | 0.88 / 0 | 0.88 / 0 |
| smooth5 | 0.239 | 1.28 | 0.67 / 0 | 0.52 / 0 | 0.23 / 0 | 0.75 / 0 | 0.64 / 0 | 0.26 / 0 | 0.88 / 0 | 0.87 / 0 |
| sharp2 (boundary) | 0.093 | 3.7 | 0.57 / 0 | 0.44 / 0 | 0.16 / 0 | 0.64 / 0 | 0.60 / 0 | 0.42 / 0 | 0.77 / 0 | 0.77 / 0 |
| wiggle2 | 0.229 | 6.0 | 0.75 / 0 | 0.65 / 0 | 0.44 / 0 | 0.80 / 0 | 0.74 / 0 | 0.50 / 0 | 0.92 / 0 | 0.90 / 0 |
| sparse10 | 0.291 | 0.84 | **0.61 / 0** | 0.37 / 0 | 0.02 / 0 | 0.72 / 0 | 0.40 / 0 | 0.00 / 0 | 0.95 / **2** | 0.88 / **1** |
| K4_3d | 0.589 | 0.57 | 0.80 / 0 | 0.74 / 0 | 0.61 / 0 | 0.86 / 0 | 0.86 / 0 | 0.85 / 0 | 0.94 / **1** | 0.94 / 0 |
| mse2 (squared loss) | 0.083 | — | 0.36 / 0 | 0.03 / 0 | 0 / 0 | — | — | — | 1.01 / **15** | 1.00 / **14** |
| **notes, signal from text** | 0.231 | 4×10⁸ | **0.53 / 0** | 0.16 / 0 | 0 / 0 | 0 | 0 | 0 | 1.08 / **28** | 0.99 / **13** |
| **notes, + note-level noise** | 0.221 | 2×10⁹ | **0.70 / 0** | 0.33 / 0 | 0.01 / 0 | 0 | 0 | 0 | 1.28 / **30** | 1.24 / **30** |
| **pharynx real covariates** | 0.174 | 0.52 | **0.55 / 0** | 0.15 / 0 | 0 / 0 | 0.46 / 0 | 0.00 / 0 | 0.00 / 0 | 0.98 / **8** | 0.83 / 0 |

**Adversarial stress.** These laws are Jensen-tight: |p − ½| is constant, so the inversion has no slack.

| Design | Floor | PYTHIA-X | γ = ½ | γ = ¼ | Accepted (any L) | 1-NN estimate | Gamma-test estimate |
|---|---|---|---|---|---|---|---|
| checker_coarse (jumps resolvable, side 0.1) | 0.3 | 0.88 / 1 | **0.74 / 0** | 0.48 / 0 | 0 | 1.10 / 28 | 1.08 / 27 |
| checker_fine (side 0.008, below the NN spacing) | 0.3 | 1.15 / **30** | 1.03 / 18 | 0.80 / 2 | 0 | 1.55 / 30 | 1.55 / 30 |
| rowwise_random (no resolvable structure) | 0.3 | 1.15 / **30** | 1.03 / 18 | 0.80 / 2 | 0 | 1.54 / 30 | 1.57 / 30 |

## Reading

1. **Without any modulus, PYTHIA-X tracks the accepted route run with the true modulus.**
   - It beats the accepted route at oracle L in 27/30 runs (smooth5), 22/30 (wiggle2), 30/30 (sparse10, both notes variants) and 20/20 (pharynx).
   - It is close but lower in the 2-D designs, where nearest-neighbour pairs are so close that drift hardly matters.
   - A practitioner does not know L. With a 2× conservative L the accepted route falls to 0–0.50 in 7 of the 9 designs where it runs; the exceptions are linear2 and K4_3d. PYTHIA-X needs no L at all.
2. **Real text geometry.** Every modulus route is vacuous: the realized L is about 10⁸–10⁹. PYTHIA-X gives 0.53–0.70 of the floor with 0 violations.

   The accepted *estimators* (1-NN Cover–Hart, Gamma test) overshoot the floor in 13–30 of 30 runs on notes. They are not safe even as lower-bound proxies.
3. **The dial.** Choose γ = ½ for laws with boundaries. On `checker_coarse`, γ = 1 had 1–2 violations and γ = ½ had 0, as the theory predicts. γ = ¼ is safer and looser.
4. **The identification boundary is real.** When the law varies below the sampling resolution and Jensen has no slack, *every* non-vacuous method fails, estimators included. The accepted route survives only by being vacuous.

   No certificate can avoid this without replicates (Antos, Devroye & Györfi 1999). The γ assumption is where that cost is stated.
5. **Honest note on the realized assumption.** A(γ = 1) failed on the *realized* matchings in many runs (the realized-assumption checks), yet the bound stayed valid in all non-adversarial designs.
   - The cause is slack from Jensen and from concentration.
   - In the adversarial designs that slack was removed by construction, and the bound failed.
   - Validity in the non-adversarial rows is therefore partly conservatism, not proof that the assumption held.
6. **Weak spot: squared loss** (0.36). The declared outcome range enters the concentration. That range is a support declaration, which the domination routes avoid.
