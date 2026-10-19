"""Value-of-Text Certification (VTC) on MIMIC-IV + MIMIC-IV-Note -- run LOCALLY by a credentialed user.

Privacy: reads the credentialed files on your machine, writes ONLY aggregate numbers (one JSON).  No note text, no
identifiers, no row-level values leave the process.  The geometry is unsupervised TF-IDF -> SVD (plus the optional
predictable posterior geometry); nothing is uploaded anywhere.

Usage:
  python scripts/run_vtc_mimic.py --mimic /path/to/mimic-iv-2.2 --note /path/to/mimic-iv-note-2.2 \
         --task readmit30|mortality_early_radiology --out vtc_mimic_result.json [--max_rows 40000]

Tasks (one admission per patient, chosen at random -> independent rows):
  readmit30                  text = discharge summary;  Y = readmission within 30 days (patients who died excluded)
  mortality_early_radiology  text = radiology reports charted within 24h of admission;  Y = in-hospital death
                             (discharge summaries are NOT used for mortality: they reveal the outcome)
Table (cells): anchor-age band (5) x gender (2) x admission type (4 groups) x insurance (3)."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pythia.core.agreement_field import afi_certificate
from horos.models.pythia_adapters import dynamic_certificate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def load(args):
    pat = pd.read_csv(os.path.join(args.mimic, "hosp", "patients.csv.gz"), usecols=["subject_id", "gender", "anchor_age"])
    adm = pd.read_csv(os.path.join(args.mimic, "hosp", "admissions.csv.gz"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "insurance", "hospital_expire_flag"],
                      parse_dates=["admittime", "dischtime"])
    adm = adm.sort_values(["subject_id", "admittime"])
    adm["next_admit"] = adm.groupby("subject_id").admittime.shift(-1)
    adm["readmit30"] = ((adm.next_admit - adm.dischtime).dt.days.between(0, 30)).astype(int)
    if args.task == "readmit30":
        notes = pd.read_csv(os.path.join(args.note, "note", "discharge.csv.gz"), usecols=["hadm_id", "text"])
        notes = notes.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).reset_index()
        df = adm[adm.hospital_expire_flag == 0].merge(notes, on="hadm_id"); df["y"] = df.readmit30
    else:
        rad = pd.read_csv(os.path.join(args.note, "note", "radiology.csv.gz"), usecols=["hadm_id", "charttime", "text"], parse_dates=["charttime"])
        rad = rad.merge(adm[["hadm_id", "admittime"]], on="hadm_id")
        rad = rad[(rad.charttime - rad.admittime).dt.total_seconds().between(0, 24 * 3600)]
        notes = rad.groupby("hadm_id").text.apply(lambda s: "\n".join(s)).reset_index()
        df = adm.merge(notes, on="hadm_id"); df["y"] = df.hospital_expire_flag
    df = df.merge(pat, on="subject_id")
    df = df.sample(frac=1, random_state=20261017).drop_duplicates("subject_id")          # one admission per patient
    if args.max_rows and len(df) > args.max_rows:
        df = df.iloc[: args.max_rows]
    return df.reset_index(drop=True)


def cells_of(df):
    age = np.digitize(df.anchor_age, [30, 45, 60, 75]); sex = (df.gender == "M").astype(int)
    at = df.admission_type.str.upper()
    atg = np.select([at.str.contains("EMER|EW"), at.str.contains("URGENT"), at.str.contains("ELECTIVE|SURGICAL SAME DAY")], [0, 1, 2], 3)
    ins = df.insurance.str.upper(); ing = np.select([ins.str.contains("MEDICARE"), ins.str.contains("MEDICAID")], [0, 1], 2)
    return ((age * 2 + sex) * 4 + atg) * 3 + ing


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mimic", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--task", default="readmit30"); ap.add_argument("--out", default="vtc_mimic_result.json")
    ap.add_argument("--max_rows", type=int, default=40000); args = ap.parse_args()
    df = load(args); y = df.y.to_numpy().astype(int); cell = cells_of(df)
    Xs = TfidfVectorizer(min_df=5, max_features=200000, sublinear_tf=True, ngram_range=(1, 2)).fit_transform(df.text)
    E = TruncatedSVD(128, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    del df["text"]
    c = afi_certificate(cell, E, y, 2, .025, levels=3, seed=0)
    dyn = dynamic_certificate(cell, E, y, 2, .025, B=3, learner="posterior", F=Xs, seed=0, fracs=(.2, .4, .4))
    res = dict(task=args.task, n=int(len(y)), base_rate=float(y.mean()), cells_used=c["cells_used"],
               table_information=c["I_X"], table_floor=c["G_X"], text_beyond_table_lower_static=c["dI_lower"],
               text_beyond_table_lower_dynamic=dyn["dI_lower"], text_beyond_table_lower=max(c["dI_lower"], dyn["dI_lower"]),
               joint_floor_upper=c["G_XT_upper"], text_information_lower=c["I_T_lower"],
               p_text_adds_nothing=c["p_text_adds_nothing"], resolution_spectrum=c["spectrum"],
               design_coinformation_point=c["design_coinfo_point"],
               note="aggregates only; alpha=.05 total (static .025 + dynamic .025 for the text-beyond-table statement)")
    json.dump(res, open(args.out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
