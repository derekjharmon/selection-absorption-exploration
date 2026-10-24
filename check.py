#!/usr/bin/env python3
# Checks every number the proposal reports against the saved results in this repository.
#
#     python3 check.py
#
# Needs Python 3 and numpy only. Calls no model, needs no key or internet, and takes a few seconds.
# Each line gives the proposal's section, the claim as the proposal states it, the value recomputed here from the
# saved results, and PASS or FAIL. REVIEWER.md lists the same claims with the file each one comes from.
# A percentage passes when the recomputed value rounds to the number the proposal prints.
import json, os, sys, collections
from math import erf, sqrt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from model import run, DEFAULTS          # the simulation itself, used only to rebuild the field at round 0

rows = []
def claim(section, text, value, ok):
    rows.append((section, text, value, bool(ok)))

def results(name):
    d = np.load(os.path.join(ROOT, "sim", "results", f"{name}.npz"))
    meta = json.load(open(os.path.join(ROOT, "sim", "results", f"{name}.json")))
    return d["mean"], meta["conds"], {k: i for i, k in enumerate(meta["keys"])}, meta["seeds"]

def find(conds, **kw):
    hits = [k for k, c in enumerate(conds) if all(c.get(a) == b for a, b in kw.items())]
    assert len(hits) == 1, (kw, len(hits)); return hits[0]

# How often a judge of quality q picks the useful idea over a dud (Appendix A2, equation 4).
pair = lambda q: 100 * (0.5 * (1 + erf(q * (1 / 0.3) / sqrt(2 * (1 - q * q)) / sqrt(2)))) if q < 1 else 100.0

# Breadth today: the range of mechanisms in use at round 0, averaged over the same 20 seeds as every run.
def spread_of(n):
    cs = n.sum(0) / n.sum(); cs = cs[cs > 0]; return float(np.exp(-(cs * np.log(cs)).sum()))
starts = [run({"T": 0, "return_grid": True}, s) for s in range(20)]
TODAY = np.mean([spread_of(g[1]) for g in starts])
pct = lambda spread: 100 * spread / TODAY

# ---------- §2. The simulated field ----------
mG, CG, ixG, seeds = results("extras_gate")
claim("§2", "Researchers write one paper per round for 100 rounds", f"{mG.shape[1]} rounds", mG.shape[1] == 100)
claim("§2", "Each round, an AI proposes 10 ideas to each researcher", f"k = {DEFAULTS['k']}", DEFAULTS["k"] == 10)

# Share of each list of 10 that is untried at the start, sampled the way the model draws lists (Gumbel top-k).
rng = np.random.default_rng(0); shares = []
for _, n, _, _ in starts:
    base = (n + DEFAULTS["eps"]) ** DEFAULTS["alpha"]; freq = base / base.sum(1, keepdims=True)
    emp = n == 0; tail = emp / emp.sum(1, keepdims=True)
    prop = (1 - DEFAULTS["lam"]) * freq + DEFAULTS["lam"] * tail
    for i in range(n.shape[0]):
        for _ in range(200):
            cand = np.argpartition(-(rng.gumbel(size=n.shape[1]) + np.log(prop[i] + 1e-12)), 10)[:10]
            shares.append(emp[i, cand].mean())
claim("§2", "Most of the 10 ideas are tail ideas", f"{100 * np.mean(shares):.1f}% untried at the start", np.mean(shares) > 0.5)
claim("§2", "5% of researchers each round set out to explore", f"explore = {DEFAULTS['explore']:.2f}", DEFAULTS["explore"] == 0.05)
claim("§2", "A hidden 1 in 10 tail ideas is useful", f"b = {DEFAULTS['b']:.2f}", DEFAULTS["b"] == 0.10)
fmax = max(c["follow_rate"] for c in CG)
claim("§2", "Follow-ups on each useful idea run from none to about 40",
      f"0 to {fmax * DEFAULTS['follow_len']:.0f}", min(c["follow_rate"] for c in CG) == 0 and fmax * DEFAULTS["follow_len"] == 40)
claim("Fig. 1-3", "Every result is a mean of 20 runs", f"{seeds} seeds", seeds == 20)

# ---------- §2. Finding 1 and Figure 1 (three judges each right 96%, slight shared bias; about 10 and 40 follow-ups) ----------
def gate(f, **kw):
    k = find(CG, follow_rate=f, **kw); u = mG[k, -1, ixG["useful_cells"]]; z = mG[k, -1, ixG["noise_cells"]]
    return dict(today=pct(mG[k, -1, ixG["eff_mechanisms"]]), useful=u, duds=z, hit=100 * u / (u + z),
                useful_today=pct(mG[k, -1, ixG["eff_mechanisms_useful"]]))
SEL = dict(regime="convener", q=0.6, V=3, rho=0.2, rely=1.0)
rel10, gen10 = gate(1.0, regime="oracle", rely=1.0), gate(1.0, regime="tail_only", rely=1.0)
sel10, sel40 = gate(1.0, **SEL), gate(4.0, **SEL)
claim("§1, abstract", "Relying on AI narrows the field by more than half", f"{rel10['today']:.1f}% of today", rel10["today"] < 50)
claim("§2 F1", "Relying on the AI's recommendations narrows the field to 41% of today", f"{rel10['today']:.1f}%", round(rel10["today"]) == 41)
claim("§2 F1", "Explorers who pick at random find useful ideas at the 10% base rate", f"hit rate {gen10['hit']:.1f}%", round(gen10["hit"]) == 10)
claim("§2 F1", "Generation alone still narrows the field to 54%", f"{gen10['today']:.1f}%", round(gen10["today"]) == 54)
claim("§2 F1", "Three judges each right 96% of the time", f"{pair(0.6):.1f}%", round(pair(0.6)) == 96)
claim("§2 F1", "Selection finds about 3 times as many useful ideas, with fewer duds",
      f"{sel10['useful'] / gen10['useful']:.2f} times; duds {sel10['duds']:,.0f} vs {gen10['duds']:,.0f}",
      round(sel10["useful"] / gen10["useful"]) == 3 and sel10["duds"] < gen10["duds"])
claim("§2 F1", "With about 10 follow-ups, selection still narrows the field to 70%", f"{sel10['today']:.1f}%", round(sel10["today"]) == 70)
claim("§2 F1", "With about 40 follow-ups, selection widens the field to 144%", f"{sel40['today']:.1f}%", round(sel40["today"]) == 144)

# ---------- §2. Finding 2 and Figure 2 (no follow-ups) ----------
mA, CA, ixA, _ = results("sweep_A")
rand = mA[find(CA, regime="tail_only", follow_rate=0.0), -1, ixA["useful_cells"]]
LINES = {"five independent": dict(V=5, rho=0.0), "one judge": dict(V=1, rho=0.0), "five sharing a bias": dict(V=5, rho=0.5)}
curves = {}
for name, kw in LINES.items():
    ks = sorted([k for k, c in enumerate(CA) if c["regime"] == "convener" and c["follow_rate"] == 0.0
                 and all(c.get(a) == b for a, b in kw.items())], key=lambda k: CA[k]["q"])
    curves[name] = ([CA[k]["q"] for k in ks], [mA[k, -1, ixA["useful_cells"]] for k in ks])
ceiling = np.mean([max(y) for _, y in curves.values()]); target = rand + 0.9 * (ceiling - rand)
def needed(name):                       # judge quality where the line first reaches 90% of the gain, interpolated linearly in q
    qs, ys = curves[name]
    for (q0, y0), (q1, y1) in zip(zip(qs, ys), zip(qs[1:], ys[1:])):
        if y0 < target <= y1: return pair(q0 + (target - y0) / (y1 - y0) * (q1 - q0))
for name, label, stated in (("five independent", "five judges whose errors are independent each need", 81),
                            ("one judge", "a single judge needs", 97), ("five sharing a bias", "five judges who share a bias need", 94)):
    a = needed(name)
    claim("§2 F2", f"For 90% of the gain, {label} about {stated}%", f"{a:.1f}%", round(a) == stated)

# ---------- §2. Finding 3 and Figure 3 (three judges, slight shared bias, by accuracy and follow-ups) ----------
FOLLOW = sorted({c["follow_rate"] for c in CG}); QS = sorted({c["q"] for c in CG if c["regime"] == "convener"})
grid = np.array([[pct(mG[find(CG, regime="convener", q=q, V=3, rho=0.2, rely=1.0, follow_rate=f), -1, ixG["eff_mechanisms"]])
                  for q in QS] for f in FOLLOW])
grid_u = np.array([[pct(mG[find(CG, regime="convener", q=q, V=3, rho=0.2, rely=1.0, follow_rate=f), -1, ixG["eff_mechanisms_useful"]])
                    for q in QS] for f in FOLLOW])
acc = [pair(q) for q in QS]; row = {round(10 * f): i for i, f in enumerate(FOLLOW)}
def lowest_widening(g, follow):          # the lowest accuracy at which the field ends above today, at this many follow-ups
    above = [a for a, v in zip(acc, g[row[follow]]) if v > 100]
    return round(min(above)) if above else None
claim("§2 F3", "At about 30 follow-ups, the field widens with judges right at least 85%",
      f"lowest accuracy that widens: {lowest_widening(grid, 30)}%", lowest_widening(grid, 30) == 85)
claim("§2 F3", "At about 40 follow-ups, it widens with judges right at least 77%",
      f"lowest accuracy that widens: {lowest_widening(grid, 40)}%", lowest_widening(grid, 40) == 77)
top20 = grid[row[20]].max()
claim("§2 F3", "At about 20 follow-ups, even perfect judges only hold the field level", f"at most {top20:.1f}%", 95 <= top20 <= 100)
below = grid[:, [i for i, a in enumerate(acc) if round(a) < 77]].max()
claim("Fig. 3", "Below about 77%, no level of follow-ups simulated widens the field", f"highest {below:.1f}%", below < 100)
r0 = grid[row[0]]
claim("Fig. 3", "Without follow-ups, selection leaves the field near 46% at any accuracy", f"{r0.min():.1f}% to {r0.max():.1f}%", round(r0.min()) >= 45 and round(r0.max()) <= 47)
claim("Fig. 3", "With about 40 follow-ups but coin-flip judges, it reaches at most 67%", f"{grid[row[40], 0]:.1f}%", round(grid[row[40], 0]) == 67)
orac = [pct(mG[find(CG, regime="oracle", rely=1.0, follow_rate=f), -1, ixG["eff_mechanisms"]]) for f in FOLLOW]
claim("Fig. 3", "Relying on AI ends between 40% and 43% at every level of absorption", f"{min(orac):.1f}% to {max(orac):.1f}%", round(min(orac)) >= 40 and round(max(orac)) <= 43)
i96 = QS.index(0.6)
claim("Fig. 3", "Figure 1's selection paths are the 96% column at about 10 and 40 follow-ups",
      f"{grid[row[10], i96]:.1f}% and {grid[row[40], i96]:.1f}%", abs(grid[row[10], i96] - sel10["today"]) < 1e-9 and abs(grid[row[40], i96] - sel40["today"]) < 1e-9)

# ---------- §2. Breadth from useful work only ----------
order = [r["useful_today"] for r in (rel10, gen10, sel10)]
claim("§2", "Counting only useful work, Figure 1's paths rank the same", " < ".join(f"{v:.1f}%" for v in order) + f"; selection at about 40: {sel40['useful_today']:.1f}%",
      order == sorted(order) and sel40["useful_today"] > 100)
claim("§2", "Counting only useful work, Figure 3's thresholds are unchanged",
      f"lowest accuracy that widens: {lowest_widening(grid_u, 30)}% at about 30, {lowest_widening(grid_u, 40)}% at about 40",
      lowest_widening(grid_u, 30) == 85 and lowest_widening(grid_u, 40) == 77)

# ---------- §3. The vocabulary ----------
V = os.path.join(ROOT, "vocab")
sources = json.load(open(os.path.join(V, "sources_ids.json")))
claim("§3, abstract", "GPT-5.5 read the abstracts of 2,401 review articles", f"{len(sources):,} sources", len(sources) == 2401)
fields = collections.Counter(s["field"] for s in sources)
claim("§3", "Reviews from management and five other social sciences", f"{len(fields)} fields: " + ", ".join(sorted(fields)),
      "management" in fields and len(fields) == 6)
quotes = collections.Counter(); phen_fields = set()
for line in open(os.path.join(V, "extracted_noquotes.jsonl")):
    for r in json.loads(line)["records"]:
        if r.get("phenomena"): phen_fields.add(r["field"])
        for kind in ("phenomena", "mechanisms"):
            for x in r.get(kind, []): quotes[x.get("quote_ok")] += 1
claim("§3", "Phenomena are drawn from management reviews", "from " + ", ".join(sorted(phen_fields)), phen_fields == {"management"})
mech = json.load(open(os.path.join(V, "menu_mechanisms.json")))
by_field = collections.Counter()
for m in mech:
    for f, k in (m.get("fields") or {}).items():
        if f in fields: by_field[f] += k
other = 1 - by_field["management"] / sum(by_field.values())
claim("§3", "Mechanisms come mostly from reviews in the five other social sciences", f"{100 * other:.0f}% of mentions", other > 0.5)
total = sum(quotes.values())
claim("§3", "Every extracted term carries a quote checked word for word (failures dropped)",
      f"{quotes[True]:,} of {total:,} matched ({100 * quotes[True] / total:.1f}%)", quotes[True] == 6034 and total == 6041)
phen = json.load(open(os.path.join(V, "menu_phenomena.json")))
claim("§3", "The vocabulary holds 1,082 phenomena and 837 mechanisms", f"{len(phen):,} and {len(mech):,}", len(phen) == 1082 and len(mech) == 837)

# ---------- §3. Placement on pairings whose answer was known (Appendix A6) ----------
PV = os.path.join(ROOT, "placement", "validation")
def placement(version, cases_file, tag):
    cases = json.load(open(os.path.join(PV, cases_file)))["cases"]
    runs = [json.load(open(os.path.join(PV, f"results_{version}_{tag}run{i}.json")))["results"] for i in (1, 2, 3)]
    def call(r):                         # the same rule as score.py: the same concept on both sides is no call
        if r.get("cls") == "filled" and set(r.get("row_names") or []) & set(r.get("mech_names") or []): return "degenerate"
        return r["cls"]
    out = collections.Counter()
    for c in cases:
        if c["expected"] not in ("filled", "tail"): continue
        calls = [call(next(r for r in run_ if r["key"] == c["key"])) for run_ in runs]
        cl, k = collections.Counter(calls).most_common(1)[0]
        cl = cl if k >= 2 else "no call"
        out[(c["expected"], cl if cl in ("filled", "tail") else "no call")] += 1
    return out
held = placement("v5", "cases_heldout.json", "heldout_")
first = placement("v4", "cases.json", "")
n_filled = sum(v for (e, _), v in held.items() if e == "filled")
claim("§3", "The counts never mistook an empty cell for a filled one",
      f"empty called filled: {held[('tail', 'filled')]} of {sum(v for (e, _), v in held.items() if e == 'tail')} held out (v5), "
      f"{first[('tail', 'filled')]} of {sum(v for (e, _), v in first.items() if e == 'tail')} in the first set (v4)",
      held[("tail", "filled")] == 0 and first[("tail", "filled")] == 0)
claim("§3", "But they missed some well-studied pairings", f"filled called empty: {held[('filled', 'tail')]} of {n_filled} held out",
      held[("filled", "tail")] > 0)

# ---------- §4. Assumptions (Table A4, about 20 follow-ups, five judges) and absorption (Appendix A4) ----------
mR, CR, ixR, _ = results("extras_robust")
def robust(variant, regime):
    k = [k for k, c in enumerate(CR) if c["_variant"] == variant and c["_regime"] == regime][0]
    return pct(mR[k, -1, ixR["eff_mechanisms"]])
REG = ["Everyone relies on AI (oracle)", "Random untried ideas", "Spotting"]
half = robust("AI's ranking toward convention at half strength", REG[0])
claim("§4", "With the AI's ranking at half strength, the field narrows by about a fifth", f"{half:.1f}% of today ({100 - half:.0f}% narrower)", 75 <= half <= 85)
for variant in ("5% of untried cells useful", "20% of untried cells useful", "Half of build-on papers follow the AI"):
    s = [robust(variant, g) for g in REG]
    claim("§4", f"Main comparisons keep their direction: {variant[0].lower() + variant[1:]}",
          f"relying {s[0]:.0f}% < generation alone {s[1]:.0f}% < selection {s[2]:.0f}%", s[0] < s[1] < s[2])
AV = json.load(open(os.path.join(ROOT, "sim", "results", "absorption_variants.json")))["rows"]
def edge(rule, f):
    v = {x["path"]: x["useful_spread_pct"] for x in AV if x["rule"] == rule and x["follow_rate"] == f}
    return v["selection"] - v["generation_alone"]
shrinks = [(f, edge("useful", f), edge("fashion", f), edge("blind", f)) for f in (1.0, 2.0, 4.0)]
claim("§4", "Follow-ups that mix the useful with the fashionable shrink selection's advantage",
      "; ".join(f"about {10 * f:.0f}: {u:.0f} -> {fa:.0f} (fashion), {b:.0f} (blind) points" for f, u, fa, b in shrinks),
      all(0 < fa < u and 0 < b < u for _, u, fa, b in shrinks))

# ---------- report ----------
w = max(len(t) for _, t, _, _ in rows)
for sec, text, value, ok in rows:
    print(f"{'PASS' if ok else 'FAIL'}  {sec:<13} {text:<{w}}  {value}")
fails = sum(not ok for *_, ok in rows)
print(f"\n{len(rows) - fails} of {len(rows)} claims pass. Breadth today (round 0, 20 seeds) is {TODAY:.1f}.")
sys.exit(1 if fails else 0)
