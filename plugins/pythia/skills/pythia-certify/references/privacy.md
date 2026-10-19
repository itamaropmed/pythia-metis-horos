# Privacy rules

- Clinical data stays where it is. Run the CLI on the machine that holds the data; do not upload PHI to a cloud
  workspace or paste it into chat.
- Share only aggregate outputs: `pythia_result.json` and `pythia_report.md` contain no identifiers or row values.
- No region, cell or term is reported with fewer than 50 distinct entities (patients).
- Never quote note text, names, MRNs, exact dates or free-text values in a response.
- `--row-cards` writes row-level output: keep it local, never commit it, never send it.
- Do not commit data files, derived parquet files or row cards to git.
