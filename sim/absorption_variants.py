# Appendix A4, red-team check that the two-gate result is not built in. In the main runs only useful cells draw
# follow-ups, so absorption tracks usefulness exactly. Here absorption is (a) blind to usefulness: every newly filled
# cell draws follow-ups with probability one half, useful or not (absorb_err 0.5), and (b) driven by fashion: a newly
# filled cell draws follow-ups with probability equal to its mechanism's popularity percentile in the field
# (absorb_rule "popularity"). Four paths of proposal Figure 1 at about 10, 20, and 40 follow-ups per absorbed idea,
# 20 runs per condition. Reports spread and spread from useful work only (duds' papers left out), as shares of the start.
import json, itertools, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run

S = 20
PATHS = {"relying_on_ai": {"regime": "oracle", "rely": 1.0},
         "generation_alone": {"regime": "tail_only", "rely": 1.0},
         "selection": {"regime": "convener", "q": 0.6, "V": 3, "rho": 0.2, "rely": 1.0},
         "no_ai": {"regime": "tail_only", "rely": 0.0}}
RULES = {"useful": {}, "blind": {"absorb_err": 0.5}, "fashion": {"absorb_rule": "popularity"}}
KEYS = ["eff_mechanisms", "eff_mechanisms_useful", "useful_cells", "noise_cells", "useful_papers", "noise_papers", "total"]

def one(a):
    p, s = a
    r = run(p, s)[-1]
    return np.array([r[k] for k in KEYS])

if __name__ == "__main__":
    def spread_of(n):
        col = n.sum(0); sh = col[col > 0] / col.sum(); return float(np.exp(-(sh * np.log(sh)).sum()))
    start = float(np.mean([spread_of(run({"T": 0, "return_grid": True}, s)[1]) for s in range(S)]))
    conds = [dict(path=pth, rule="none", follow_rate=0.0) for pth in PATHS]
    conds += [dict(path=pth, rule=r, follow_rate=f) for pth, r, f in itertools.product(PATHS, RULES, (1.0, 2.0, 4.0))]
    jobs = [({**PATHS[c["path"]], **RULES.get(c["rule"], {}), "follow_rate": c["follow_rate"]}, s) for c in conds for s in range(S)]
    with ProcessPoolExecutor() as ex: out = np.array(list(ex.map(one, jobs, chunksize=4))).reshape(len(conds), S, len(KEYS))
    mean = out.mean(1); se = out.std(1) / np.sqrt(S)
    rows = []
    for c, m, e in zip(conds, mean, se):
        d = dict(zip(KEYS, m))
        rows.append({**c, **{k: float(v) for k, v in d.items()}, "spread_se": float(e[0]), "useful_spread_se": float(e[1]),
                     "spread_pct": 100 * d["eff_mechanisms"] / start, "useful_spread_pct": 100 * d["eff_mechanisms_useful"] / start,
                     "useful_share_of_new_work": d["useful_papers"] / max(d["useful_papers"] + d["noise_papers"], 1)})
    json.dump({"start": start, "seeds": S, "rows": rows}, open("results/absorption_variants.json", "w"), indent=1)
    lines = [f"Absorption that does not track usefulness. Means of {S} runs at round 100; start spread {start:.1f}.",
             "Spread and useful-only spread as % of the start; useful cells and dud cells filled; useful share of papers in new cells.",
             f"{'rule':8s} {'f':>3} {'path':17s} | {'spread %':>8s} | {'useful-only %':>13s} | {'useful':>6s} {'duds':>6s} | {'useful share':>12s}"]
    for r in rows:
        lines.append(f"{r['rule']:8s} {r['follow_rate']:3.0f} {r['path']:17s} | {r['spread_pct']:8.0f} | {r['useful_spread_pct']:13.0f} | "
                     f"{r['useful_cells']:6.0f} {r['noise_cells']:6.0f} | {r['useful_share_of_new_work']:12.2f}")
    open("results/absorption_variants.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
