# Input contract

All inputs are CSV with a header row. One row per case.

local-risk `--cal`:
- `--entity-col`: patient (or other independent unit) id. Never reported.
- `--levels`: hierarchy columns, finest first (e.g. `family|predbin|disbin`, `family|predbin`, `predbin`).
  Built from predictions/covariates only — bin edges from CAL predictions, never from outcomes.
- `--loss-cols`: one numeric column per model, values in [0, loss_cap]. W30 miss = 1{|y − f(x)| > 30}.
- Rows must come from entities **not** used to fit the model (split by patient, ideally by time).
`--route`: same level columns for new cases; no outcomes needed.

nli / table-floor `--data`:
- `--cell-col`: discretised table covariates (cells fixed without labels). Singleton cells are dropped.
- `--y-col`: class label (any strings; sorted order defines class index).
- nli: `p_<k>` columns (prefix `--prob-prefix`) in the same sorted class order, rows summing to 1,
  produced out-of-fold or by a frozen model that never saw these labels.
- One row per patient (or attest that rows are independent given the design).

Minimum: 100 rows; regions/cells reported only with ≥ 50 entities.
