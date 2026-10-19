# NLI and table floors

Units: g = 1 − |p|² (Brier/Gini). Half-Brier = g/2. Within table cells c with m_c ≥ 2:

- Table floor box: the within-cell all-pairs label disagreement Q has E[Q | design] = G + V exactly (G = joint
  table+text floor, V = within-cell text information = NLI). One label moves Q by ≤ 2/n → McDiarmid interval.
- NLI = E[d(P(Y|X,T), P(Y|X))] = R*(X) − R*(X,T) (operational theorem): text information beyond the table.
- Variational form: NLI = sup over E[h|X]=0 of (E⟨e_Y − p̄, h⟩)² / E|h|².
- Cohort certificate: h_i = leave-one-out cell-centred held-out probabilities (label-free, sums to 0 per cell).
  V ≥ (C_lower)₊² / H with C = mean h_{i,y_i}, H = mean |h_i|²; C_lower from a concentration bound
  (default `safe_portfolio`: best of five endpoints after a Bonferroni split — valid).
- Model ceiling: held-out Brier of any frozen model ≥ G, upper-bounded with KL/betting.
- Joint floor upper bound: G ≤ min(tableU − NLI_lower, modelU). Simultaneous at α (split 4 ways).

NLI is lower-only. It certifies that text helps; it cannot certify that text does not help.
The full PYTHIA-U LP (pair route, shape assumption A(γ), tipping ε, stratified map) lives in
`pythia.core.unified` in the repository; the plugin exposes the assumption-free components.
