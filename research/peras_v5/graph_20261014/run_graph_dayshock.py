"""Falsification of row/patient independence: calendar-day shocks shared by all cases of a day."""
import sys, json, math
import numpy as np
sys.path.insert(0, ".")
from peras.core.graph_dominated_saa import graph_dominated_saa_lower_bound
from peras.core.dominated_saa import dominated_saa_lower_bound
reps = int(sys.argv[1]); base = int(sys.argv[2]); out = sys.argv[3]
rows = []
W = 1.0
for r in range(reps):
    rng = np.random.default_rng(base + r)
    days = 80; per_day = rng.integers(25, 45, days)
    day = np.repeat(np.arange(days), per_day); n = len(day)
    cell = rng.integers(0, 3, n)
    shock = rng.standard_normal(days) * 2.0
    y = 3.0 * cell + shock[day] + rng.standard_normal(n) * .5
    # case marginal: N(3c, 4.25): window-miss floor = 1 - P(|Z|<=W), MAE floor = sqrt(4.25)*sqrt(2/pi)
    sd = math.sqrt(4.25)
    from scipy.stats import norm
    gold = {"window_miss": 1 - (norm.cdf(W / sd) - norm.cdf(-W / sd)), "mae": sd * math.sqrt(2 / math.pi)}
    for loss, kw in (("window_miss", {"window": W}), ("mae", {})):
        common = dict(loss=loss, caps=(2., 4., 8.), alpha=.05, representation_provenance="sim",
                      outcome_blind_labels_attested=True, **kw)
        g = graph_dominated_saa_lower_bound(cell, y, {"day": day}, dependency_graph_attested=True, **common)
        gr = graph_dominated_saa_lower_bound(cell, y, {"row": np.arange(n)}, dependency_graph_attested=True, **common)
        rows.append(dict(rep=r, loss=loss, n=n, gold=gold[loss], graph=g.bound, K=g.diagnostics["selected_colours"],
                         row_iid=gr.bound))
open(out, "w").write("".join(json.dumps(x) + "\n" for x in rows))
for loss in ("window_miss", "mae"):
    R = [x for x in rows if x["loss"] == loss]
    print(loss, "valid graph", sum(x["graph"] <= x["gold"] for x in R), "/", len(R), "row-iid", sum(x["row_iid"] <= x["gold"] for x in R),
          "median graph/gold %.3f row/gold %.3f" % (np.median([x["graph"] / x["gold"] for x in R]), np.median([x["row_iid"] / x["gold"] for x in R])),
          "K", int(np.median([x["K"] for x in R])))
