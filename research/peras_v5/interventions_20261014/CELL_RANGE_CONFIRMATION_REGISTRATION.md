# Registered cell-specific range confirmation

Written after the 61001--61010 development run and before opening confirmation
seeds.

## Frozen protocol

- Same ten known-law mechanisms and first eight outcome-blind transformations
  as `INTERVENTION_BATTERY_REGISTRATION.md`.
- Fresh seeds 62001--62050; 4,000 rows total.
- Same finite-cell SAA witness and alpha 0.05.
- Replace the global loss range in McDiarmid by analytic conditional hard
  ranges registered for every observed cell. For an entity in cell `c`, the
  bounded-difference sensitivity is `D_c/n`.
- Global misalignment receives the global conditional range because every
  observed label has the full outcome mixture. No oracle row-specific range is
  permitted.

## Primary endpoints

1. zero known-floor exceedances;
2. baseline median bound/gold greater than the frozen global-range value 0.525;
3. positive high-SNR baseline bounds in at least 9/10 mechanism-seed deciles;
4. strict improvement over the global-range endpoint on a majority of rows;
5. exact invariance across baseline, bijective, canonical-nuisance and
   bag-token representations.

Mechanisms are the scientific comparison unit. Repeated seeds quantify
finite-sample stability and are not treated as independent mechanisms for a
paper significance claim.
