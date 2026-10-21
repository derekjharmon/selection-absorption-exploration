# Appendix A4 decomposition: how much of the narrowing and the widening runs through retraining (the loop),
# and how much through follow-ups themselves. Crosses the model relearning from the record each round (learning)
# against a model frozen at the start (frozen), with follow-ups written in place of researchers' building-on papers
# (displace) or added on top of them (extra), at about 0, 10, 20, and 40 follow-ups per useful idea, for relying on
# AI and for selection with the three judges of proposal Figure 1. 20 runs per condition.
import json, itertools, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run

S = 20
PATHS = {"relying_on_ai": {"regime": "oracle"},
         "selection": {"regime": "convener", "q": 0.6, "V": 3, "rho": 0.2}}
KEYS = ["eff_mechanisms", "useful_papers", "total", "ai_build_useful", "explore_useful", "useful_cells", "noise_cells"]

def one(a):
    p, s = a
    rs = run(p, s)
    return np.array([[r[k] for k in KEYS] for r in rs])          # rounds x keys

def crossover(m):
    a = np.convolve(m[:, KEYS.index("ai_build_useful")], np.ones(5) / 5, "same")
    e = np.convolve(m[:, KEYS.index("explore_useful")], np.ones(5) / 5, "same")
    c = np.flatnonzero(a[2:-2] > e[2:-2]); return int(c[0] + 3) if len(c) else None

if __name__ == "__main__":
    conds = [dict(path=pth, follow_rate=f, loop=lp, follow_mode=fm)
             for pth, f, lp, fm in itertools.product(PATHS, (0.0, 1.0, 2.0, 4.0), (True, False), ("displace", "extra"))]
    jobs = [({**PATHS[c["path"]], "follow_rate": c["follow_rate"], "loop": c["loop"], "follow_mode": c["follow_mode"]}, s)
            for c in conds for s in range(S)]
    with ProcessPoolExecutor() as ex: out = list(ex.map(one, jobs, chunksize=4))
    arr = np.array(out).reshape(len(conds), S, *out[0].shape)     # conds x seeds x rounds x keys
    mean = arr.mean(1); se = arr.std(1) / np.sqrt(S)
    np.savez("results/loop_decomposition.npz", mean=mean, se=se, final=arr[:, :, -1, :])
    json.dump({"keys": KEYS, "seeds": S, "conds": conds}, open("results/loop_decomposition.json", "w"), indent=1)

    start = mean[0, 0, KEYS.index("eff_mechanisms")]
    lines = [f"Loop decomposition, means of {S} runs. Spread at round 100 (start {start:.1f}); crossover round; follow-up papers (cumulative).",
             f"{'path':14s} {'f':>3} {'mode':8s} | {'learning':>15s} | {'frozen':>15s} | {'learning - frozen':>17s} | crossover L/F | follow-ups L/F"]
    for i, c in enumerate(conds):
        if not c["loop"]: continue
        j = next(k for k, d in enumerate(conds) if d == {**c, "loop": False})
        sl, sf = (mean[x, -1, KEYS.index("eff_mechanisms")] for x in (i, j))
        el, ef = (se[x, -1, KEYS.index("eff_mechanisms")] for x in (i, j))
        def fu(x):   # papers in useful cells not placed by the AI's top pick or by explorers: the follow-ups
            m = mean[x]; return m[-1, KEYS.index("useful_papers")] - m[:, KEYS.index("ai_build_useful")].sum() - m[:, KEYS.index("explore_useful")].sum()
        lines.append(f"{c['path']:14s} {c['follow_rate']:3.0f} {c['follow_mode']:8s} | {sl:7.1f} ± {el:4.1f} | {sf:7.1f} ± {ef:4.1f} | {sl - sf:+17.1f} | "
                     f"{str(crossover(mean[i])):>5s}/{str(crossover(mean[j])):<5s} | {fu(i):6.0f}/{fu(j):<6.0f}")
    open("results/loop_decomposition.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
