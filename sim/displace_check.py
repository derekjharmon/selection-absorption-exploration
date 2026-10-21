# Robustness check: follow-on papers written by the researchers themselves (taking the place of build-on moves in the
# same row) instead of added on top. Compared with the matching part-B conditions (follow_mode "extra").
import json, itertools, numpy as np
from concurrent.futures import ProcessPoolExecutor
from sweep import one, KEYS, S
ix = {k: i for i, k in enumerate(KEYS)}
REG = [("oracle", {"regime": "oracle"}), ("tail_only", {"regime": "tail_only"}),
       ("convener", {"regime": "convener", "q": 0.6, "V": 5, "rho": 0.2, "adopt": 1.0})]
def cross(mk):
    a = np.convolve(mk[:, ix["ai_build_useful"]], np.ones(5) / 5, "same"); e = np.convolve(mk[:, ix["explore_useful"]], np.ones(5) / 5, "same")
    c = np.flatnonzero(a[2:-2] > e[2:-2]); return c[0] + 3 if len(c) else None
if __name__ == "__main__":
    conds = [{**p, "rely": r, "explore": ex, "follow_rate": f}
             for (_, p), r, ex, f in itertools.product(REG, (1.0, 0.0), (0.05, 0.10), (1.0, 2.0, 4.0))]
    jobs = [({**c, "follow_mode": "displace"}, s) for c in conds for s in range(S)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    arr = np.stack(res).reshape(len(conds), S, *res[0].shape); m = arr.mean(1)
    np.savez_compressed("results/displace_check.npz", mean=m, final=arr[:, :, -1, :])
    json.dump({"keys": KEYS, "conds": conds}, open("results/displace_check.json", "w"), indent=0)
    B = np.load("results/sweep_B.npz")["mean"]; CB = json.load(open("results/sweep_B.json"))["conds"]
    def matchB(c): return [k for k, b in enumerate(CB) if all(b.get(a) == c[a] for a in c)][0]
    print(f"{'regime':9s} {'rely':>4} {'ex':>5} {'f':>3} | spread100 extra -> displace | useful extra -> displace | noise | loop turns (extra -> displace)")
    for k, c in enumerate(conds):
        b = matchB(c)
        print(f"{c['regime']:9s} {c['rely']:4.1f} {c['explore']:5.2f} {c['follow_rate']:3.0f} | {B[b,-1,ix['eff_mechanisms']]:6.1f} -> {m[k,-1,ix['eff_mechanisms']]:6.1f} | "
              f"{B[b,-1,ix['useful_cells']]:5.0f} -> {m[k,-1,ix['useful_cells']]:5.0f} | {m[k,-1,ix['noise_cells']]:5.0f} | {cross(B[b])} -> {cross(m[k])}", flush=True)
