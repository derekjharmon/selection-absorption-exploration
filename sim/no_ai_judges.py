# Appendix A3, the field without AI as a ceiling. Compares selection with the three judges of proposal Figure 1 when
# researchers build on the AI's top pick and when they build on the record without it (rely = 0), with random exploration
# without AI for reference, at about 10, 20, and 40 follow-ups per useful idea. 20 runs per condition.
import json, numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run
J = {"regime": "convener", "q": 0.6, "V": 3, "rho": 0.2}
PATHS = {"selection_with_ai": {**J, "rely": 1.0}, "selection_without_ai": {**J, "rely": 0.0},
         "random_without_ai": {"regime": "tail_only", "rely": 0.0}}
def one(a):
    p, s = a; r = run(p, s)[-1]; return r["eff_mechanisms"], r["eff_mechanisms_useful"], r["useful_cells"], r["noise_cells"]
if __name__ == "__main__":
    def spread_of(n):
        col = n.sum(0); sh = col[col > 0] / col.sum(); return float(np.exp(-(sh * np.log(sh)).sum()))
    start = float(np.mean([spread_of(run({"T": 0, "return_grid": True}, s)[1]) for s in range(20)]))
    rows, lines = [], [f"Selection with and without AI, three judges each right 96% of the time. Means of 20 runs at round 100; start spread {start:.1f}."]
    for f in (1.0, 2.0, 4.0):
        for name, p in PATHS.items():
            with ProcessPoolExecutor() as ex: m = np.array(list(ex.map(one, [({**p, "follow_rate": f}, s) for s in range(20)]))).mean(0)
            rows.append(dict(path=name, follow_rate=f, spread=m[0], useful_spread=m[1], useful=m[2], duds=m[3]))
            lines.append(f"~{10*f:.0f} follow-ups | {name:21s} spread {m[0]:6.1f} ({100*m[0]/start:3.0f}%) | useful only {100*m[1]/start:3.0f}% | useful {m[2]:5.0f} duds {m[3]:5.0f}")
    json.dump({"start": start, "rows": rows}, open("results/no_ai_judges.json", "w"), indent=1)
    open("results/no_ai_judges.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
