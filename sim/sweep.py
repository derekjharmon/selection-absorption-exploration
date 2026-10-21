# The sweep. Saves round-by-round means and standard errors for every condition so that the threshold, the timing
# and the noise race can all be read from the same runs.
#   A (accuracy dial): q x V x rho for the convener, plus oracle and tail_only, at follow-on 0 / 0.5 / 1.
#   B (field dials):   follow-on x exploration x reliance (x adoption for the convener), for oracle, tail_only and a
#                      good convener (q .6, V 5, rho .2). Asks whether the field ever broadens, and when.
import json, sys, itertools, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run
S = 20
KEYS = ["useful_cells", "noise_cells", "useful_papers", "noise_papers", "total", "ai_build", "ai_build_useful",
        "ai_build_noise", "explore_useful", "explore_noise", "eff_mechanisms", "eff_mechanisms_useful"]
def one(args):
    p, s = args; h = run(p, s)
    return np.array([[x[k] for k in KEYS] for x in h], dtype=np.float64)
def conds_A():
    out = []
    for f in (0.0, 0.5, 1.0):
        out += [{"regime": "oracle", "follow_rate": f}, {"regime": "tail_only", "follow_rate": f}]
        for q, V, rho in itertools.product((0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0), (1, 3, 5), (0.0, 0.2, 0.5)):
            out.append({"regime": "convener", "q": q, "V": V, "rho": rho, "follow_rate": f})
    return out
def conds_B():
    out = []
    for f, ex, rely in itertools.product((0.0, 0.5, 1.0, 2.0, 4.0), (0.02, 0.05, 0.10), (0.0, 0.5, 1.0)):
        base = {"follow_rate": f, "explore": ex, "rely": rely}
        out += [{**base, "regime": "oracle"}, {**base, "regime": "tail_only"}]
        for a in (0.25, 0.5, 1.0):
            out.append({**base, "regime": "convener", "q": 0.6, "V": 5, "rho": 0.2, "adopt": a})
    return out
if __name__ == "__main__":
    part = sys.argv[1]
    conds = conds_A() if part == "A" else conds_B()
    jobs = [({"follow_mode": "extra", **c}, s) for c in conds for s in range(S)]   # saved sweeps A and B used follow-ups added on top
    print(f"part {part}: {len(conds)} conditions x {S} seeds = {len(jobs)} runs", flush=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    arr = np.stack(res).reshape(len(conds), S, *res[0].shape)          # cond x seed x round x metric
    np.savez_compressed(f"results/sweep_{part}.npz", mean=arr.mean(1), se=arr.std(1) / np.sqrt(S), final=arr[:, :, -1, :])
    json.dump({"keys": KEYS, "seeds": S, "conds": conds}, open(f"results/sweep_{part}.json", "w"), indent=0)
    print(f"part {part} done", flush=True)
