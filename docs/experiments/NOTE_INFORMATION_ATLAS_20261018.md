# Note Information Atlas: what the clinical notes carry beyond the table (2026-10-18)

- **Code:** `research/note_atlas_20261018/`
  - `atlas.py`: the ten lenses;
  - `validate_mtsamples.py`: planted-truth validation on real public notes;
  - `run_mayo_atlas.py`: the staged local Mayo run, aggregates only.
- **Status:** exploratory or calibrated analysis. These are estimates with patient-cluster bootstrap intervals, not Pythia certificates.

**Information target.** r, the duration residual left after a cross-fitted XGBoost table model (patient-grouped, 5-fold). Anything the table already explains is excluded by construction. Information is measured in minutes² as

  I = E r² − E (r − g)²,

where g is a cross-fitted note model of r.

## The ten lenses

| Lens | Question | Method |
|---|---|---|
| A Structure | What are these notes physically? | Length, sections (header grammar), boilerplate (lines shared by ≥ 20 patients), within-patient copy-forward, numeric and negation density, reportable vocabulary |
| B Global information | How much do the notes add beyond the table? | I, R²\|table and ΔMAE of a TF-IDF (1–2-gram) ridge model of r; patient-cluster bootstrap CI; outcome-permutation null |
| C Per-case information | For each case: how far does reading the note move the prediction, and does it help? | Movement \|g_i\|; gain_i = r_i² − (r_i − g_i)²; concentration (Gini, top-10% share); by note length, by stratum, empty notes |
| D Section masking | Where in the note is the information? | Drop a section (reliance); keep only that section (sufficiency); retrain without it (unique information) |
| E Token-class masking | What kind of language carries it? | Mask numbers, dates and times, negation scopes, negation cues (a meaning flip), medication-like tokens, units |
| F Melting curve ("temperature") | How redundant is the information? | Random token dropout at rate p: retained information I(p)/I, heat capacity −dI/dp, half-melting point |
| G Metaheuristics | What is the smallest set of sections that carries the signal, globally and per case? | Beam search (width 3) over keep-only section sets; simulated annealing per case (Metropolis, geometric cooling) for minimal explanations |
| H Network diffusion | Is information organised by note similarity, and at what scale? | Patient-safe kNN case graph; leakage-free heat diffusion (training-fold residuals only, mass-normalised) giving I(t) over diffusion time t; Louvain note communities and their information |
| I Term network | Which families of terms move duration? | Co-occurrence PMI graph of reportable terms; personalised diffusion of residual association; Louvain term communities ranked by diffused minutes |
| J Topology | Is the residual smooth over note space? | MST of the embedding (H0 persistence); Dirichlet energy of r along cross-patient MST edges against a permutation null |

**Privacy.** Only aggregates leave the computation. A term is reported only if it appears for ≥ 50 distinct patients, is mostly lower-case (a proper-noun filter) and never follows a title. Section headers and document types need ≥ 200 patients.

## Validation: real MTSamples notes, planted outcome (known truth)

- **Setup:** 2,357 notes. The table is note type × age band × sex. Planted effects were "general" in the ANESTHESIA section (+25 min), the blood-loss number (+0.08 min/ml) and non-negated complications (+30 min). True information beyond the table is 111 min².
- **B: global information.** I = 33.3 [22.4, 43.3] min². A linear bag-of-words model recovers about 30% of the truth. The outcome-permutation null gave −1.2 and −0.6, and the negative control gave −0.1 [−3.1, 2.5].
- **D: sections.** **ANESTHESIA ranked first** on reliance, sufficiency and unique information. Keeping the ANESTHESIA section alone gave I = 56.5, more than the full note. A focused reader beats reading everything.
- **G: beam search** picked ANESTHESIA first. Annealing found a median of 1 section per case: 80% of a case's information lives in a single section.
- **I: term network.** Its top community contained "general, endotracheal, fascia, operating, loss", which is the planted signal.
- **C: per-case.** Spearman between movement and the true per-case effect was 0.36. Information is concentrated: the Gini of positive gain was 0.79, and the top 10% of cases carried 61% of it.
- **H: diffusion.** Information peaked at diffusion time t = 3 (I = 27).
- **J: topology.** The residual is smoother along the MST than chance (ratio 0.87, permutation p = 0.005).
- **Limit.** The planted complication effect was too rare to test (0.3% of notes).

## Mayo (local run in progress; aggregates only)

**Cohort.** 12,993 cases from 10,532 patients.
- Notes join for 96.6% of cases; 82.5% of cases have non-empty eligible notes.
- **The timing rule was re-verified.** Every dated note is at least 2.27 days before the case; there were 0 violations.
- The table model's out-of-fold MAE is 33.43 min for Arm A and 33.02 min for Arm B.

**A: structure.**
- Median 6,841 characters, about 1,050 tokens and 8 notes per case.
- The section grammar is dominated by echocardiography reports: ATRIA, RIGHT VENTRICLE, CARDIAC VALVES, OTHER ECHO FINDINGS, COMPARISON, LEFT VENTRICLE, and so on.
- 20.5% of lines are boilerplate, and within-patient copy-forward is 1.5%.
- There are 64 numbers and 11 negation cues per 1,000 tokens.

**B: global information.**
- I = **64.5 [49.4, 81.0] min²** beyond the table, which is 2.5% of the residual variance.
- **ΔMAE = −0.51 min.** The squared-loss model gains in MSE but loses in MAE, so the note signal sits in the tails. A median or Huber note model is needed for the operational loss.

**C: per-case.**
- The median case moves by 7.1 minutes and the top 1% by 20.6 minutes.
- **Information is highly concentrated.** The Gini of positive gain is 0.84, and the top 10% of cases carry 71% of it.
- **Cases without notes alone show I = 60.5.** Note presence or absence is itself informative, which motivates lens K below.

**E: token classes.**
- Masking medication-like tokens loses 0.74 min².
- Masking numbers, negation and units does not lose information, and slightly improves it.
- **The note signal is not in the numbers.**

**F: melting curve.**
- Information is extremely redundant or structural. **67% survives 97% random token dropout**, and the half-melting point was never reached.
- Together with the empty-note result, this says much of the signal is *which notes exist and how much is written*, not specific words.

**Pending:**
- D (sections), G (beam and annealing), H (diffusion and communities), I (term network), J (topology);
- K (meta): presence, volume and document type versus content;
- the beyond-11-concepts residual;
- permutation nulls and section-unique retrains.

The device disconnected mid-run. The stages resume with:

```bash
cd research/note_atlas_20261018
python3 run_mayo_atlas.py lens:D --derived <derived> --work ~/mayo_work/atlas --out mayo_local
```

Repeat with `lens:G`, `lens:H`, `lens:I`, `lens:J`, `meta`, `beyondB`, `null:0`, `null:1` and `unique:<k>`.
