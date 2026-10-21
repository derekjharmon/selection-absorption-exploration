# Runs for the extra figures and tables (20 seeds, 5% exploration, follow-on written by the researchers themselves).
#   gate:   the two-gate map. Accuracy of each of three views (V 3, rho .2) x follow-on, plus oracle / random / no-AI
#           at each follow-on level.
#   robust: the main comparison (relying on AI, generation alone, selection, no AI) at ~20 follow-ups, under alternative settings.
#   grid:   one run per condition with the final grid saved, for the picture of the field at round 100.
import json, sys, itertools, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run
from sweep import one, KEYS, S
SPOT = {"regime": "convener", "q": 0.6, "V": 5, "rho": 0.2}
REGIMES = [("Everyone relies on AI (oracle)", {"regime": "oracle", "rely": 1.0}),
           ("Random untried ideas", {"regime": "tail_only", "rely": 1.0}),
           ("Spotting", {**SPOT, "rely": 1.0}),
           ("No AI", {"regime": "tail_only", "rely": 0.0})]
VARIANTS = [("Baseline", {}),
            ("Follow-ups added on top of researchers' papers", {"follow_mode": "extra"}),
            ("5% of untried cells useful", {"b": 0.05}),
            ("20% of untried cells useful", {"b": 0.20}),
            ("Useful ideas sit near popular mechanisms", {"useful_tracks_popularity": 1.0}),
            ("Field misjudges 5% of new ideas", {"absorb_err": 0.05}),
            ("Field misjudges 20% of new ideas", {"absorb_err": 0.20}),
            ("Exploration 2%", {"explore": 0.02}),
            ("Exploration 10%", {"explore": 0.10}),
            ("AI proposes fewer untried ideas (25%)", {"lam": 0.25}),
            ("AI proposes more untried ideas (75%)", {"lam": 0.75}),
            ("AI lists 5 ideas", {"k": 5}),
            ("AI lists 20 ideas", {"k": 20})]
FOLLOW = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
QS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
def conds_gate():
    out = []
    for f in FOLLOW:
        out += [{"regime": "oracle", "rely": 1.0, "follow_rate": f}, {"regime": "tail_only", "rely": 1.0, "follow_rate": f},
                {"regime": "tail_only", "rely": 0.0, "follow_rate": f}]
        out += [{"regime": "convener", "q": q, "V": 3, "rho": 0.2, "rely": 1.0, "follow_rate": f} for q in QS]
    return out
def conds_robust():
    return [{"follow_rate": 2.0, **p, **v, "_variant": vn, "_regime": rn} for vn, v in VARIANTS for rn, p in REGIMES]
def strip(c): return {k: v for k, v in c.items() if not k.startswith("_")}
def grid(args):
    p, s = args; h, n, useful, empty0 = run({**p, "return_grid": True}, s); return n, useful, empty0
if __name__ == "__main__":
    part = sys.argv[1]
    if part == "grid":
        specs = [("oracle", {"regime": "oracle", "rely": 1.0, "follow_rate": 2.0}), ("spotting", {**SPOT, "rely": 1.0, "follow_rate": 2.0}),
                 ("noai", {"regime": "tail_only", "rely": 0.0, "follow_rate": 2.0}), ("start", {"regime": "oracle", "T": 0})]
        out = {}
        for name, p in specs:
            n, u, e = grid((p, 0)); out[name] = n; out["useful"] = u; out["empty0"] = e
        np.savez_compressed("results/extras_grid.npz", **out); print("grid done"); sys.exit()
    conds = conds_gate() if part == "gate" else conds_robust()
    jobs = [(strip(c), s) for c in conds for s in range(S)]
    print(f"{part}: {len(conds)} conditions x {S} seeds", flush=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    arr = np.stack(res).reshape(len(conds), S, *res[0].shape)
    np.savez_compressed(f"results/extras_{part}.npz", mean=arr.mean(1), se=arr.std(1) / np.sqrt(S), final=arr[:, :, -1, :])
    json.dump({"keys": KEYS, "seeds": S, "conds": conds}, open(f"results/extras_{part}.json", "w"), indent=0)
    print(f"{part} done", flush=True)
