# The three simulation figures, from saved results (no new runs).
#   1. The accuracy target (sweep A, follow-on 0).
#   2. The noise race (sweep A, follow-on 0).
#   3. Widening versus slower narrowing (displacement check: 5% exploration, follow-ups written in place of researchers' own papers).
import json, os, numpy as np
from math import erf, sqrt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
INK, GREY, LIGHT, ACCENT, ACCENT2 = "#1f2328", "#8a8f98", "#c9ccd1", "#b5482a", "#2f5d8a"
os.makedirs("figures", exist_ok=True)
def load(name):
    d = np.load(f"results/{name}.npz"); meta = json.load(open(f"results/{name}.json"))
    return d["mean"], meta["conds"], {k: i for i, k in enumerate(meta["keys"])}
def pick(C, **kw): return [k for k, c in enumerate(C) if all(c.get(a) == b for a, b in kw.items())]
def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(f"figures/{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

# pairwise accuracy: chance a judge ranks a useful idea above a dud (usefulness base rate 10%)
pair = lambda q: 100 * (0.5 * (1 + erf(q * (1 / 0.3) / sqrt(2 * (1 - q * q)) / sqrt(2)))) if q < 1 else 100.0
mA, CA, ix = load("sweep_A")

# ---------- Figure 1: the accuracy target ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
rand = mA[pick(CA, regime="tail_only", follow_rate=0.0)[0], -1, ix["useful_cells"]]
orac = mA[pick(CA, regime="oracle", follow_rate=0.0)[0], -1, ix["useful_cells"]]
lines = [("One judge", dict(V=1, rho=0.0), INK, "-"),
         ("Five judges with different blind spots", dict(V=5, rho=0.0), ACCENT, "-"),
         ("Five judges sharing a bias toward convention", dict(V=5, rho=0.5), ACCENT2, "--")]
top = []
for lab, kw, col, ls in lines:
    ks = sorted(pick(CA, regime="convener", follow_rate=0.0, **kw), key=lambda k: CA[k]["q"])
    x = [pair(CA[k]["q"]) for k in ks]; y = [mA[k, -1, ix["useful_cells"]] for k in ks]; top.append(max(y))
    ax.plot(x, y, ls, color=col, lw=1.6, marker="o", ms=3, label=lab)
ceil = np.mean(top); g90 = rand + 0.9 * (ceil - rand)
ax.axhline(rand, color=GREY, lw=0.8, ls=":"); ax.text(100.5, rand + 18, "Generation alone (a random untried idea)", color=GREY, fontsize=7.5, ha="right")
ax.axhline(orac, color=GREY, lw=0.8, ls=":"); ax.text(50.5, orac + 14, "Relying on AI (the AI's top pick)", color=GREY, fontsize=7.5)
ax.axhline(g90, color=LIGHT, lw=0.8); ax.text(50.5, g90 + 18, "90% of the possible gain", color=GREY, fontsize=7.5)
ax.set_xlim(49, 101); ax.set_ylim(0, 1050)
ax.set_xlabel("How often each judge picks the useful idea over a dud (%)")
ax.set_ylabel("Useful tail ideas found by round 100")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.15), fontsize=7.5)
save(fig, "figure2_accuracy_target")

# ---------- Figure 2: the noise race ----------
fig, ax = plt.subplots(figsize=(5.2, 3.4))
t = np.arange(1, mA.shape[1] + 1)
kr = pick(CA, regime="tail_only", follow_rate=0.0)[0]
ks = pick(CA, regime="convener", follow_rate=0.0, q=0.6, V=5, rho=0.2)[0]
for k, lab, col in [(kr, "Generation alone", GREY), (ks, "Selection", ACCENT)]:
    ax.plot(t, mA[k, :, ix["noise_cells"]], "--", color=col, lw=1.6)
    ax.plot(t, mA[k, :, ix["useful_cells"]], "-", color=col, lw=1.6)
    ax.text(101.5, mA[k, -1, ix["noise_cells"]], f"{lab}: duds", color=col, fontsize=7.5, va="center")
    ax.text(101.5, mA[k, -1, ix["useful_cells"]], f"{lab}: useful", color=col, fontsize=7.5, va="center")
ax.set_xlim(0, 100); ax.set_ylim(0, None)
ax.set_xlabel("Round"); ax.set_ylabel("New cells filled (cumulative)")
save(fig, "figureA1_noise_race")

# ---------- Figure 3 (appendix Figure A2): widening versus slower narrowing ----------
mD, CD, ixD = load("displace_check")
def series(**kw): return mD[pick(CD, explore=0.05, **kw)[0], :, ixD["eff_mechanisms"]]
def turn(**kw):
    m = mD[pick(CD, explore=0.05, **kw)[0]]
    a = np.convolve(m[:, ixD["ai_build_useful"]], np.ones(5) / 5, "same"); e = np.convolve(m[:, ixD["explore_useful"]], np.ones(5) / 5, "same")
    c = np.flatnonzero(a[2:-2] > e[2:-2]); return c[0] + 3 if len(c) else None
fig, ax = plt.subplots(figsize=(5.2, 3.6))
t = np.arange(1, mD.shape[1] + 1); start = series(regime="tail_only", rely=0.0, follow_rate=2.0)[0]
ax.axhline(start, color=LIGHT, lw=0.8); ax.text(62, start - 6, "Starting spread", color=GREY, fontsize=7.5, va="top")
specs = [("No AI (~40 follow-ups per useful idea)", dict(regime="tail_only", rely=0.0, follow_rate=4.0), GREY, "--", None),
         ("Relying on AI", dict(regime="oracle", rely=1.0, follow_rate=2.0), INK, "-", None),
         ("Selection, ~10 follow-ups per useful idea", dict(regime="convener", rely=1.0, follow_rate=1.0), "#e3a089", "-", dict(regime="convener", rely=1.0, follow_rate=1.0)),
         ("Selection, ~20 follow-ups", dict(regime="convener", rely=1.0, follow_rate=2.0), "#cc6b4c", "-", dict(regime="convener", rely=1.0, follow_rate=2.0)),
         ("Selection, ~40 follow-ups", dict(regime="convener", rely=1.0, follow_rate=4.0), ACCENT, "-", dict(regime="convener", rely=1.0, follow_rate=4.0))]
for lab, kw, col, ls, tk in specs:
    s = series(**kw); ax.plot(t, s, ls, color=col, lw=1.6, label=lab)
    r = turn(**tk) if tk else None
    if r: ax.plot(r, s[r - 1], "o", color=col, ms=5, mec="white", mew=0.8, zorder=5)
ax.plot([], [], "o", color=GREY, ms=5, mec="white", label="The AI now sends more researchers into useful new work than explorers do")
ax.set_xlim(0, 100); ax.set_ylim(0, None)
ax.set_xlabel("Round"); ax.set_ylabel("Spread of the field\n(effective number of mechanisms in use)")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=7.5, ncol=1)
save(fig, "figureA2_widening_vs_slower_narrowing")
print("ceiling", round(ceil), "random", round(rand), "oracle", round(orac),
      "| crossover round:", {f: turn(regime="convener", rely=1.0, follow_rate=f) for f in (1.0, 2.0, 4.0)})
