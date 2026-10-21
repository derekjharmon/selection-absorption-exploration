# Checks before any result is trusted. Lines set in advance:
#  1. Placebo: a convener with no skill (q=0, rho=0) must not beat tail_only on useful cells reached at T
#     (difference within 2 standard errors over 20 seeds).
#  2. Loop (internal diagnostic only): with the model frozen at the start (loop=False), the convener's advantage over
#     the oracle in the useful share of the record at T should shrink relative to loop=True.
#  3. Direction: with real skill (q=0.6, V=5, rho=0.2) the convener reaches more useful cells than tail_only and oracle.
import numpy as np, json
from concurrent.futures import ProcessPoolExecutor
from model import run
S = 20
def final(p, s): h = run(p, s)[-1]; return h
def batch(p): 
    with ProcessPoolExecutor() as ex: return list(ex.map(final, [p] * S, range(S)))
def m(rs, key): v = np.array([r[key] for r in rs]); return v.mean(), v.std() / np.sqrt(len(v))
def share(rs): v = np.array([r["useful_papers"] / r["total"] for r in rs]); return v.mean(), v.std() / np.sqrt(len(v))
if __name__ == "__main__":
    out = {}
    tail = batch(dict(regime="tail_only")); plac = batch(dict(regime="convener", q=0.0, rho=0.0))
    a, b = m(tail, "useful_cells"), m(plac, "useful_cells"); se = np.hypot(a[1], b[1])
    out["placebo"] = dict(tail_only=a, convener_no_skill=b, pass_=abs(b[0] - a[0]) <= 2 * se)
    print(f"1. placebo: tail_only {a[0]:.1f}±{a[1]:.1f} vs no-skill convener {b[0]:.1f}±{b[1]:.1f} useful cells -> {'PASS' if out['placebo']['pass_'] else 'FAIL'}")
    good = dict(regime="convener", q=0.6, V=5, rho=0.2)
    orc = batch(dict(regime="oracle")); conv = batch(good)
    orc_f = batch(dict(regime="oracle", loop=False)); conv_f = batch({**good, "loop": False})
    adv, adv_f = share(conv)[0] - share(orc)[0], share(conv_f)[0] - share(orc_f)[0]
    out["loop"] = dict(advantage_loop=adv, advantage_frozen=adv_f, shrinks=adv_f < adv)
    print(f"2. loop: convener-minus-oracle useful share of record, loop {adv:.3f} vs frozen model {adv_f:.3f} -> {'shrinks (PASS)' if adv_f < adv else 'does not shrink (FAIL)'}")
    c3 = m(conv, "useful_cells")[0]; t3 = a[0]; o3 = m(orc, "useful_cells")[0]
    out["direction"] = dict(convener=c3, tail_only=t3, oracle=o3, pass_=c3 > t3 and c3 > o3)
    print(f"3. direction: useful cells reached at T: convener {c3:.1f} | tail_only {t3:.1f} | oracle {o3:.1f} -> {'PASS' if out['direction']['pass_'] else 'FAIL'}")
    for name, rs in [("oracle", orc), ("tail_only", tail), ("convener (q=.6,V=5,rho=.2)", conv)]:
        print(f"   {name:28s} useful cells {m(rs,'useful_cells')[0]:6.1f} | noise cells {m(rs,'noise_cells')[0]:6.1f} | useful share of record {share(rs)[0]:.3f} | noise share {np.mean([r['noise_papers']/r['total'] for r in rs]):.3f}")
    json.dump(out, open("results/checks.json", "w"), indent=1, default=float)
