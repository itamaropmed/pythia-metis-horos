# Pythia v5 guide: certified best-achievable risk from tables and clinical text

This is the entry point for the October 2026 development line, Pythia v5. It covers what the system certifies, where each piece of theory lives, which code implements it, and where it was tested.

- Planning: `docs/status/ROADMAP_V5_20261018.md`.
- Conversation record: `docs/status/DISCUSSION_LOG_V5_202610.md`.

## 1. The question

Records are (X, T, Y):
- X: a table row, such as structured fields or extracted concepts;
- T: free clinical text;
- Y: an outcome.

For a loss ℓ, Pythia v5 certifies, with no fitted model required and at a stated level α:

| Quantity | Meaning | Certified side |
|---|---|---|
| R\*(X) | best risk any rule can reach from the table | two-sided |
| R\*(X, T) | best risk any rule can reach from table + text | upper (assumption-free); lower (under the declared shape assumption A(γ)) |
| NLI = R\*(X) − R\*(X, T) | **Natural Language Information**: what the notes add beyond the table | lower (assumption-free); an upper bound is provably impossible without assumptions |
| R\*(∅) − R\*(X) | table information | two-sided |
| R(f) − R\*(X, T) | headroom of a model f (estimation gap) | lower |
| Per-patient labels | irreducible-risk score, model uncertainty, certified-easy groups | group-level (Learn-then-Test) |

**The three classical gaps** map onto these quantities:
- the *data/information* gap is NLI and table information;
- the *estimation* gap is model headroom;
- the *distribution/finite-sample* gap is the width of every interval.

## 2. Reading order

1. **Core identity and the pair route**
   - `theory/REPLICATE_PAIR_CERTIFICATES_20261015.md`
   - `theory/MODULUS_FREE_EXTRAPOLATION_20261016.md`
   - `theory/PYTHIA_X_STRENGTHENING_20261016.md`
2. **Model-free two-sided sandwich:** `theory/MODEL_FREE_SANDWICH_20261017.md`
3. **Text over a table**
   - `theory/TABLE_ANCHORED_TEXT_INFORMATION_20261017.md`
   - `theory/AGREEMENT_FIELD_INFORMATION_20261017.md`
   - `theory/AFI_TEXT_THEOREMS_20261017.md`
4. **LP layer and relaxation dial:** `theory/LP_RELAXATION_AND_TEXT_GEOMETRY_20261017.md`
5. **NLI definition, axioms, theorems:** `theory/NATURAL_LANGUAGE_INFORMATION_NLI_20261017.md`
6. **Unified certificate (PYTHIA-U):** `theory/PYTHIA_U_UNIFIED_CERTIFICATE_20261018.md`
7. **Condensed statements of all of the above:** `theory/PYTHIA_V5_COMPENDIUM.md`

All theory notes are indexed in `docs/theory/README.md`. The literature survey is `docs/literature/Certified risk bounds tabular text.md`.

## 3. Concept → theory → code → tests → experiments

| Concept | Theory | Code (`pythia/core/`) | Tests (`tests/`) | Experiments (`research/peras_v5/`) |
|---|---|---|---|---|
| Replicate pair certificates (0-1 floor from exact replicates) | REPLICATE_PAIR_CERTIFICATES | `replicate_pair_certificates.py` | `test_replicate_pair_certificates.py` | `realdata_20261015/` |
| Equal-floor identity, PYTHIA-X multiscale extrapolation, A(γ) | MODULUS_FREE_EXTRAPOLATION, PYTHIA_X_STRENGTHENING | `multiscale_pair_extrapolation.py` | `test_multiscale_pair_extrapolation.py` | `novelty_20261016/` |
| Model-free sandwich (U1 ceiling, I1 information) | MODEL_FREE_SANDWICH | `multiscale_pair_extrapolation.py` (`sandwich`) | same | `sandwich_20261017/` |
| Table-anchored text information | TABLE_ANCHORED_TEXT_INFORMATION | `anchored_information.py` | `test_anchored_information.py` | `anchored_20261017/` |
| Agreement-Field Information (AFI), spectrum, co-information | AGREEMENT_FIELD_INFORMATION, AFI_TEXT_THEOREMS | `agreement_field.py` | `test_agreement_field.py` | `afi_20261017/` |
| Dynamic / predictable geometry (martingale validity) | AFI_TEXT_THEOREMS, Theorem 5 | `dynamic_geometry.py` | `test_dynamic_geometry.py` | `dynamic_20261017/` |
| Three-gap certificate, any loss, per-sample labels (PYTHIA-G3) | G3 results doc | `global_bound.py` | `test_global_bound.py` | `g3_20261017/` |
| LP-1 design LP, LP-2 moment LP, conditional text map | LP_RELAXATION_AND_TEXT_GEOMETRY | `lp_bounds.py` | `test_lp_bounds.py` | `lp_20261017/` |
| **NLI**: projection certificate and exact test | NATURAL_LANGUAGE_INFORMATION_NLI | `research/peras_v5/nli_20261017/methods.py` (research code); vector form in `unified.py` | `test_nli.py` | `nli_20261017/` |
| **PYTHIA-U**: one simultaneous certificate | PYTHIA_U_UNIFIED_CERTIFICATE | `unified.py` | `test_unified.py` | `unified_20261018/` |
| Earlier v5 work: domination, smooth or graph routes | DOMINATED_SAA, DEPENDENCY_GRAPH_DOMINATION, SMOOTH_DOMINATION | `dominated_saa.py`, `graph_dominated_saa.py`, `smooth_dominated_saa.py`, `metric_dominated_saa.py` | `test_*dominated*` | `domination_20261014/`, `graph_20261014/`, `continuous_20261015/` |

## 4. Quick start: the unified certificate

```python
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pythia.core.unified import unified_certificate, stratified_certificate

# cells: integer table cell per row; notes: list of str; y: int labels 0..K-1
Xs = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(notes)
E = TruncatedSVD(64, random_state=0).fit_transform(Xs)
E /= np.linalg.norm(E, axis=1, keepdims=True)                  # outcome-blind text geometry
F = hstack([csr_matrix(np.eye(cells.max() + 1)[cells]), Xs]).tocsr()   # features for the NLI / model routes

u = unified_certificate(cells, E, F, y, K, alpha=.05, gamma=.5)
u["G"]                  # [lower (A(1/2)), upper (assumption-free)] on the joint Brier floor, cohort
u["table"]              # two-sided table floor
u["V_lower"]            # certified NLI (Brier units)
u["table_information"]  # two-sided
u["losses"]             # 0-1, Brier/2, log, conditional-MI transfers
u["G_population"]       # the same interval lifted to the population
u["refuted"], u["eps_tipping"]   # falsification of A(gamma) and the assumption budget

# A model portfolio, for example logistic regression plus gradient boosting, only spends alpha:
# unified_certificate(..., models=[(F, None), (F_dense, lambda: HistGradientBoostingClassifier())])

# Map of where in the table the notes add information:
# stratified_certificate(strata, cells, E, F, y, K)
```

## 5. Data and privacy rules (binding)

- No PHI is ever committed.
- Mayo analyses export aggregates only.
- Real clinical notes do not leave the device.
- Public data used here: MTSamples (`/home/claude/data/mtsamples.csv` in the cloud sandbox; not tracked) and PMLB.
- MIMIC runs are local and credentialed (`scripts/run_vtc_mimic.py`) and write aggregates only.
- Embeddings (`*.npy`) and data files are git-ignored.
