"""Power and false-alarm rate of the anytime-valid harmful-shift monitor vs stream length and harm size."""
import json, sys, numpy as np
sys.path.insert(0, "../.."); from pythia.core.local_transport import BettingRiskMonitor
rng = np.random.default_rng(7); u = .40; out = []
for delta in (0.0, -0.05, 0.03, 0.05, 0.10, 0.15):
    for N in (50, 100, 200, 500, 1000, 2000):
        alarms, delays = 0, []
        for _ in range(1000):
            m = BettingRiskMonitor(u, .05)
            for x in rng.random(N) < u + delta:
                m.update(x)
                if m.alarm_at is not None:
                    break
            if m.alarm_at is not None:
                alarms += 1; delays.append(m.alarm_at)
        out.append(dict(delta=delta, N=N, alarm_rate=alarms / 1000, median_delay=float(np.median(delays)) if delays else None))
        print(json.dumps(out[-1]), flush=True)
