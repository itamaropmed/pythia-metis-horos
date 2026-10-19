# Family/neighborhood lattice confirmation results

Date: 2026-10-14

The frozen protocol in `FAMILY_LATTICE_CONFIRMATION_REGISTRATION.md` was run on
50 untouched seeds for each of ten bounded noise mechanisms.  Each of the 500
runs simultaneously certified global, family, procedure, note-neighborhood
and nuisance-fragmented representations.

## Primary endpoints

- Raw known-risk exceedances: **0 / 2,500 node endpoints**.
- Closed known-risk exceedances: **0 / 2,500 node endpoints**.
- Information-order failures: **0 / 500 lattices**.
- Independent-nuisance repairs: **500 / 500**.
- Nuisance raw endpoint strictly weaker than canonical neighborhood:
  **500 / 500**.
- Positive canonical-neighborhood bounds: **500 / 500**, covering all ten
  mechanisms.
- Incorrect alpha spend: **0 / 500**; every lattice spent exactly 0.05.
- Model training calls: **0 / 500**.

| Node | Median raw bound / gold | Median closed bound / gold |
|---|---:|---:|
| global | 0.746727 | 0.748277 |
| family | 0.763564 | 0.764080 |
| procedure | 0.767273 | 0.767273 |
| note neighborhood | 0.732514 | 0.732514 |
| note neighborhood + random nuisance | 0.519664 | **0.732514** |

The equality repair is not inferred from the numerical similarity of two
representations.  It is licensed only by the registered synthetic generation
contract: the random token is independent of the outcome conditional on the
canonical neighborhood.  Real clinical text requires external provenance for
an analogous statement.

## Mechanism-level neighborhood tightness

| Mechanism | Median closed bound / gold |
|---|---:|
| bimodal | 0.77220 |
| discrete mixture | 0.77143 |
| Gaussian | 0.72925 |
| heteroscedastic | 0.70340 |
| high SNR | 0.72925 |
| lognormal | 0.60238 |
| low SNR | 0.72925 |
| t3 | 0.55487 |
| triangular | 0.78373 |
| uniform | 0.84336 |

Raw artifact:
`family_lattice_confirmation_results.jsonl`

SHA-256:
`4756b94120fbaf61549ece1da8e2c893d5be9f8755798368eb4480ed6c05ddce`

These results confirm the registered finite-case construction.  They do not
establish universal SOTA, clinical sampling validity, or a positive value of
notes on Mayo.
