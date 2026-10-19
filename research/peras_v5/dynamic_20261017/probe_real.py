import runpy, json, numpy as np
L = runpy.run_path("run_dynamic_real.py", run_name="lib")
from horos.models.pythia_adapters import dynamic_certificate
for lname, y in L["LABELS"].items():
    K = int(y.max() + 1)
    for tname in ("T1_note_type", "T2_type_age_sex", "T3_type_age_sex_len"):
        cell = L["LADDER"][tname]; v = []
        for seed in range(3):
            v.append(dynamic_certificate(cell, L["E5"], y, K, .01, B=2, learner="posterior", F=L["F"], seed=seed, fracs=(.3, .7), certify_from=1)["dI_lower"])
        print(lname, tname, "posterior B=2 (30% warm-up, 70% certified):", np.round(v, 3), flush=True)
