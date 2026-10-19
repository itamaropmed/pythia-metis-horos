# Support-free clustered block-SAA confirmation results

Date: 2026-10-14

The untouched 500-draw confirmation passed every registered primary gate.

- Block-SAA validity: **500/500**; Wilson lower 95% limit **0.9924**.
- Matched pair validity: **500/500**.
- Positive block-SAA endpoints: **500/500**.
- Strict block-SAA wins over disjoint-pair/dKelly: **500/500**.
- Mechanism-median wins: **10/10**.
- Median block bound / analytic truth: **0.78654**.
- Median pair bound / analytic truth: **0.67281**.

| Unbounded noise | Median truth | Block-SAA | Matched pair | Block wins |
|---|---:|---:|---:|---:|
| Gaussian | 0.8936 | **0.7001** | 0.5885 | 50/50 |
| Laplace | 1.1196 | **0.8933** | 0.7747 | 50/50 |
| Logistic | 1.5527 | **1.2231** | 1.0458 | 50/50 |
| Student-t3 | 1.2346 | **0.9756** | 0.8393 | 50/50 |
| Student-t5 | 1.0628 | **0.8357** | 0.7161 | 50/50 |
| Gaussian scale mixture | 1.6081 | **1.2903** | 1.1304 | 50/50 |
| Gaussian location mixture | 2.2396 | **1.5401** | 1.1864 | 50/50 |
| Signed lognormal | 1.2713 | **0.9247** | 0.7544 | 50/50 |
| Signed exponential | 1.1195 | **0.8978** | 0.7768 | 50/50 |
| Contaminated Gaussian | 1.2947 | **1.0266** | 0.8544 | 50/50 |

The selected block route was 64 patients in every run; the selected score cap
was 2 in 347 runs, 4 in 140, and 1 in 13.  Route and cap selection were covered
by the frozen 30-candidate alpha ledger.

Both methods use the same dKelly concentration and require no population
outcome range.  The improvement is therefore attributable to witness geometry:
the block optimum exploits one-dimensional common-action structure that a pair
cannot see.  The claim remains limited to the registered MAE setting with
outcome-blind finite cells and independent patient clusters.

Raw JSONL SHA-256:
`4c0c0307305fa34b750e0a5c5a4d97c4b03d59ce17abd3510149e0ca11197c8b`
