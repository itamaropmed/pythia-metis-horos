# Geometry upgrades: development results (NOT a confirmation)

Seeds 6,000,001+ and 6,100,001+ (development only), regime A design, n=1024 (certify 512), 10 seeds x 11
designs x 2 losses.  Variants: base = frozen PYTHIA-S regime A; bal = + balanced OT blocks; krr = alignment
chosen by smallest declared residual modulus among {none, linear, kernel-ridge}; both = bal + krr.

Validity: 0 violations in all 880 endpoints.

| Design | MAE base -> best | Q90 base -> best | What helped |
|---|---|---|---|
| quad2 | 0.653 -> 0.734 | 0.356 -> 0.576 | alignment chosen by declared modulus |
| interact2 | 0.710 -> 0.752 | 0.495 -> 0.622 | same |
| smooth2 | 0.750 -> 0.761 | 0.603 -> 0.642 | kernel-ridge alignment |
| sparse10 | 0.670 -> 0.716 | 0.377 -> 0.553 | balanced OT blocks (+ alignment) |
| sine1 | 0.757 -> 0.763 | 0.647 -> 0.650 | kernel-ridge alignment |
| linear1, hetero1, linear5, bimodal1, t3_2, skew1 | ~0 to -0.01 | ~0 to -0.015 | nothing; balanced only adds ledger cost |

Balanced blocks reduce mean within-block d^2 by 10-25% in d >= 2 (outcome-blind check) but cost ~1% via the
alpha ledger where geometry is already easy.  Proposed frozen rule for the next registration:
alignment = argmin declared residual modulus over {none, linear, kernel-ridge};
geometry = kd if TwoNN effective dimension (on the declared-metric-scaled X) < 3, else matching + balanced.
Next: register this rule and confirm on untouched seeds against the frozen regime-A PYTHIA-S and the accepted routes.
