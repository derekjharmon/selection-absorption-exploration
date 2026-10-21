# Appendix A4, extension of loop_decomposition.py: does retraining ever add breadth once absorption is stronger?
# Selection with the three judges of proposal Figure 1, follow-ups written in place of researchers' own papers
# (displace), learning versus frozen at about 60 and 80 follow-ups per useful idea over 100 rounds. 20 runs per condition.
import json, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run

S = 20
SEL = {"regime": "convener", "q": 0.6, "V": 3, "rho": 0.2}
KEYS = ["eff_mechanisms", "ai_build_useful", "explore_useful"]

def one(a):
    p, s = a
    return np.array([[r[k] for k in KEYS] for r in run(p, s)])

def crossover(m):
    a = np.convolve(m[:, 1], np.ones(5) / 5, "same"); e = np.convolve(m[:, 2], np.ones(5) / 5, "same")
    c = np.flatnonzero(a[2:-2] > e[2:-2]); return int(c[0] + 3) if len(c) else None

if __name__ == "__main__":
    conds = []
    for f in (6.0, 8.0): conds += [("selection", f, 100)]
    rows = []
    for path, f, T in conds:
        base = SEL if path == "selection" else {"regime": "oracle"}
        res = {}
        for loop in (True, False):
            jobs = [({**base, "follow_rate": f, "loop": loop, "T": T}, s) for s in range(S)]
            with ProcessPoolExecutor() as ex: arr = np.array(list(ex.map(one, jobs)))
            m = arr.mean(0); sp = arr[:, :, 0]
            res[loop] = dict(mean=m, end=sp[:, -1].mean(), end_se=sp[:, -1].std() / np.sqrt(S),
                             r100=sp[:, 99].mean(), xo=crossover(m))
        L, F = res[True], res[False]
        rows.append(dict(path=path, follow_rate=f, T=T, learning=L["end"], frozen=F["end"], diff=L["end"] - F["end"],
                         se=(L["end_se"] ** 2 + F["end_se"] ** 2) ** 0.5, learning_r100=L["r100"], frozen_r100=F["r100"],
                         crossover_learning=L["xo"], crossover_frozen=F["xo"],
                         learning_series=L["mean"][:, 0].round(1).tolist(), frozen_series=F["mean"][:, 0].round(1).tolist()))
        r = rows[-1]
        print(f"{path:14s} f={f:.0f} (~{10*f:.0f} follow-ups) T={T}: learning {r['learning']:.1f} | frozen {r['frozen']:.1f} | "
              f"learning - frozen {r['diff']:+.1f} ± {r['se']:.1f} | at round 100 {r['learning_r100']:.1f} vs {r['frozen_r100']:.1f} | "
              f"crossover {r['crossover_learning']}", flush=True)
    json.dump(rows, open("results/loop_decomposition_ext.json", "w"))
