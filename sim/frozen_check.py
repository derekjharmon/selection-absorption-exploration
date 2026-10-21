# Appendix A4 diagnostic: the loop link (the model relearning from the record) versus a model frozen at the start.
# Spread at round 100 and useful share of the record, relying on AI (oracle) and selection (stored as spotting), at about 20 follow-ups (displacement).
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from model import run
S = 20
def fin(a): p, s = a; return run(p, s)[-1]
def batch(p):
    with ProcessPoolExecutor() as ex: return list(ex.map(fin, [(p, s) for s in range(S)]))
if __name__ == "__main__":
    spot = {"regime": "convener", "q": 0.6, "V": 5, "rho": 0.2}
    for name, p in [("oracle", {"regime": "oracle"}), ("spotting", spot)]:
        for loop in (True, False):
            rs = batch({**p, "follow_rate": 2.0, "loop": loop})
            sp = np.array([r["eff_mechanisms"] for r in rs]); us = np.array([r["useful_papers"] / r["total"] for r in rs])
            print(f"{name:9s} {'learning' if loop else 'frozen  '}  spread {sp.mean():6.1f} ± {sp.std()/np.sqrt(S):.1f} | useful share {us.mean():.4f}", flush=True)
