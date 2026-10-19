# Prospective Mayo confirmation protocol

Status: draft pre-specification. This document does not authorize enrollment or expose
patient-derived data.

## Freeze before enrollment

Record immutable hashes for the cohort query, eligibility rules, feature builders, note
cutoff implementation, model weights, representation artifacts, guard library, endpoint
code, and analysis environment. The chosen incumbent and note arm must both pass the OOF,
cutoff, and patient-cluster audits before they may be frozen.

## Cohort and information timing

- Accrue 3,000 consecutive eligible future cases after the freeze timestamp.
- Apply exclusions mechanically from the frozen cohort query.
- Score the incumbent and chosen note arm on exactly the same cases.
- Admit a note only when `note_date < scheduled_in_room - 2 days`.
- Keep every patient in one analysis cluster and prevent patient overlap with development.
- Do not inspect outcome comparisons before the cohort reaches 3,000 eligible cases.

Operational monitoring may inspect only ingestion counts, missingness, schema validity,
and failed scoring jobs. It must not expose labels, errors, arm comparisons, or subgroup
performance.

## Endpoints

The primary endpoint is the paired patient-clustered difference in MAE between the frozen
note arm and incumbent. Report its point estimate and pre-specified confidence interval.

Secondary endpoints are W30, procedure-family MAE, error calibration by predicted-duration
band, certificate consistency, and operating-room hours implied by the paired MAE change.
Monetary conversion is optional and must use a separately versioned cost-per-minute input.

Only input-defined procedure families and note communities with at least 250 cases are
eligible for slice certificates. All slice comparisons require a frozen multiplicity
procedure.

## Required release manifest

The locked manifest must contain:

- freeze timestamp and software/environment hashes;
- cohort-query and exclusion hashes;
- incumbent and note-arm identifiers, weights, and OOF audit results;
- representation and feature-provenance hashes;
- note guard version and note-provenance audit;
- Pythia guard version and supported domain;
- primary/secondary endpoint implementations;
- patient-clustered resampling seed registry;
- slice definitions and multiplicity method;
- falsification checks and refusal conditions.

Retrospective Mayo findings remain `calibrated` or `exploratory` until this protocol is
completed. The prospective comparison does not upgrade a mathematical claim beyond its
proved or conditional assumptions.
