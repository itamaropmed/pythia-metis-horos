# research/peras_v5: experiment index

Each folder holds the run scripts, the raw logs (JSON lines) and a `*_RESULTS*.md` report for one question. The reports contain only aggregates from MTSamples, PMLB and generated data. Mayo results are aggregate exports.

| Folder | Question | Main scripts | Report |
|---|---|---|---|
| `ood_audit_20261010/`, `diagnostics_20261014/`, `interventions_20261014/` | Earlier v5 audits and representation interventions | — | see `../peras_v5/RESULTS_v5*.md` |
| `domination_20261014/` | Fixed-action domination floors: confirmation suite, Mayo aggregates | `run_dominated_confirmation.py` | `DOMINATED_SAA_CONFIRMATION_RESULTS.md`, `MAYO_DOMINATED_20261014.md` |
| `graph_20261014/` | Dependency-graph domination under day shocks | `run_graph_dev.py`, `run_graph_dayshock.py` | logs |
| `continuous_20261015/` | Smooth, geometry and local-field routes for continuous X: pre-registered confirmation | `run_continuous_suite.py`, `run_geo_confirm.py` | `*_REGISTRATION.md`, `*_RESULTS.md` |
| `realdata_20261015/` | Real text: replicate and quotient certificates on MTSamples | `mtsamples_*.py`, `tabular_replicate_ladder.py` | `REALDATA_RESULTS_20261015.md` |
| `novelty_20261016/` | PYTHIA-X vs accepted near-replicate and kNN routes; the four weak points | `run_extrapolation_suite.py`, `run_strengthening.py`, `run_deep_levels.py` | `EXTRAPOLATION_RESULTS_20261016.md`, `STRENGTHENING_RESULTS_20261016.md` |
| `sandwich_20261017/` | Model-free ceiling U1 and information I1 on synthetic data, MTSamples and PMLB | `run_sandwich_synthetic.py`, `run_sandwich_real.py` | `SANDWICH_RESULTS_20261017.md` |
| `anchored_20261017/` | Table-anchored text information | `run_anchored.py` | `ANCHORED_RESULTS_20261017.md` |
| `afi_20261017/` | AFI on 10 generated OOD families and MTSamples (extraction ladder). `generator.py` is the shared exact-law generator | `run_afi_generated.py`, `run_afi_learned.py`, `run_afi_real.py` | `AFI_RESULTS_20261017.md` |
| `dynamic_20261017/` | Predictable and learned geometries; e5-small-v2 embeddings | `run_dynamic_generated.py`, `run_dynamic_real.py`, `embed_mtsamples.py` | `DYNAMIC_RESULTS_20261017.md` |
| `g3_20261017/` | PYTHIA-G3: three gaps, any loss, per-sample labels | `run_g3.py`, `summ_g3.py` | `G3_RESULTS_20261017.md` |
| `lp_20261017/` | LP-1 and LP-2, relaxation path, conditional text map | `run_lp.py`, `summ_lp.py` | `LP_RESULTS_20261017.md` |
| `nli_20261017/` | **NLI** vs accepted methods on 6 unseen families and MTSamples; scaling | `run_nli.py`, `run_nli2.py`, `run_nli_real.py`, `run_nli_scale.py`, `summarize_nli.py`; `methods.py` (NLI and baselines), `generator_ood.py` | `NLI_RESULTS_20261017.md`, `summary_ood.txt`, `summary_real.txt` |
| `unified_20261018/` | **PYTHIA-U** unified certificate on generated data and MTSamples | `run_unified.py` (set `PORTFOLIO=1` for LR + gradient boosting), `run_unified_real.py`, `summ_unified.py` | `UNIFIED_RESULTS_20261018.md`, `summary_portfolio.txt`, `summary_lr_only.txt` |

## Reproducing

```bash
export OMP_NUM_THREADS=1                      # avoids BLAS oversubscription
cd research/peras_v5/unified_20261018
PORTFOLIO=1 python run_unified.py afi:D02_novel_findings,ood:O1_grammar_severity 5 > log.txt
python summ_unified.py log.txt
```

Paths are configurable through environment variables:

| Variable | Default | Points to |
|---|---|---|
| `PERAS_REPO` | the repository root, from the script location | the repository |
| `PERAS_DATA` | `/home/claude/data` | a folder with `mtsamples.csv` (the public MTSamples dump, columns `transcription`, `medical_specialty`) and the PMLB `*_BNG_*.tsv.gz` files |
| `PERAS_MODELS` | `/home/claude/models` | local Hugging Face model folders, e.g. `e5-small-v2` |

For example: `PERAS_DATA=~/data/public python run_unified_real.py`. Data files and embeddings are not tracked.
