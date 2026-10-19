"""Consistency: the projection certificate approaches the truth as n grows (O1 level 2 and O4 level 1)."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from generator_ood import make_ood
from methods import nli_projection
for fam, beta in (("O1_grammar_severity", .95), ("O4_copy_forward_template", .5)):
    for n in (2000, 4000, 8000, 16000, 32000):
        for rep in range(5):
            d = make_ood(fam, n, beta, 123_000 + 10 * n + rep)
            Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
            o = nli_projection(d["cell"], Xs, d["y"], .05, rep, B=19, p_true=d["P"][:, 1])
            print(json.dumps(dict(family=fam, n=n, rep=rep, true_nli=d["true_nli"], **o), default=float), flush=True)
