# Fixed-action dominated SAA: confirmation results

Date: 2026-10-14.  Untouched seeds `2,200,001`–`2,200,500`, frozen design and routes exactly as in
`DOMINATED_SAA_CONFIRMATION_REGISTRATION.md`.  Raw JSONL SHA-256:
`cd96eada6ff5d8fe1bde199dccd0d3ba91abee36dd51019635a64844f0d73004`.

## Primary (MAE): all gates passed
- Validity **500/500** (Wilson lower 0.9924); positive 500/500.
- Strict wins over the registered support-free block-SAA: **464/500**; mechanism-median wins **9/10**.
- Median bound/gold: **0.8997** (block-SAA 0.7868; disjoint pairs 0.6728).
- The one non-win mechanism is the contaminated Gaussian (5% scale-10 contamination): 1.0152 vs
  1.0203 (14/50 wins), as anticipated in the registration; the cap grid tops out at 8.

## Secondary losses (separate alpha .05 families)
| Loss | Valid | Strict wins vs block | Mechanism wins | Median dominated/gold | Median block/gold |
|---|---:|---:|---:|---:|---:|
| Pinball q=.9 | 500/500 | 495/500 | 10/10 | **0.8473** | 0.6873 |
| Window miss (half-window 1) | 500/500 | 500/500 | 10/10 | **0.9341** | 0.6307 |

## Additional validity stress (development, seeds 2,500,001+)
2,400 certificates: G in {60,120,250} patients, 5–40 cells, cluster sizes 1–6 with a shared
within-cluster effect, cells varying inside clusters, retention none or 3: **0 violations**
(max bound/gold 0.863).  Small-G tightness is modest (median 0.33–0.61 for MAE by G).

## Interpretation
The gain comes from removing block optimism and concentrating over all clusters through a
pathwise-dominating additive sum.  Inequalities are classical (Bentkus 2004; SAA optimism as in
Mak–Morton–Wood 1999); the candidate contribution is the support-free, cluster-valid,
multi-loss composition for representation-cell Bayes floors.  Not variance-adaptive; heavy
contamination needs larger registered caps.
