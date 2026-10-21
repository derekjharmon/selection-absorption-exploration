# Figures 4-5 and Tables 1 and 3 (appendix Tables A3 and A4) from the extras runs (no new simulation except the starting spread per seed).
import json, csv, numpy as np
from math import erf, sqrt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from model import run
from sweep import S

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "legend.frameon": False})
INK, GREY, ACCENT = "#1f2328", "#8a8f98", "#b5482a"
pair = lambda q: 100 * (0.5 * (1 + erf(q * (1 / 0.3) / sqrt(2 * (1 - q * q)) / sqrt(2)))) if q < 1 else 100.0
def spread_of(n):
    cs = n.sum(0) / n.sum(); cs = cs[cs > 0]; return float(np.exp(-(cs * np.log(cs)).sum()))
START = np.mean([spread_of(run({"T": 0, "return_grid": True}, s)[1]) for s in range(S)])
def load(name):
    d = np.load(f"results/{name}.npz"); meta = json.load(open(f"results/{name}.json"))
    return d["mean"], meta["conds"], {k: i for i, k in enumerate(meta["keys"])}
def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(f"figures/{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

# ---------- Figure 4: the two-gate map ----------
mG, CG, ix = load("extras_gate")
def g(**kw): return [k for k, c in enumerate(CG) if all(c.get(a) == b for a, b in kw.items())][0]
FOLLOW = sorted({c["follow_rate"] for c in CG}); QS = sorted({c["q"] for c in CG if c["regime"] == "convener"})
today = np.zeros((len(FOLLOW), len(QS)))          # spread at round 100 as % of the starting spread
for a, f in enumerate(FOLLOW):
    for b, q in enumerate(QS):
        today[a, b] = 100 * mG[g(regime="convener", q=q, follow_rate=f), -1, ix["eff_mechanisms"]] / START
cmap = LinearSegmentedColormap.from_list("gate", ["#555b66", "#c9ccd1", "#f6f3ee", "#e8b9a5", "#cc6b4c", "#8f2f17"])
norm = TwoSlopeNorm(vmin=40, vcenter=100, vmax=150)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
im = ax.imshow(today, origin="lower", cmap=cmap, norm=norm, aspect="auto")
for a in range(len(FOLLOW)):
    for b in range(len(QS)):
        v = today[a, b]; turned = v >= 100
        ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if (v > 125 or v < 55) else INK,
                fontweight="bold" if turned else "normal")
        if turned: ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
ax.set_xticks(range(len(QS))); ax.set_xticklabels([f"{pair(q):.0f}" if pair(q) < 99 else (f"{pair(q):.1f}" if pair(q) < 100 else "100") for q in QS])
ax.set_yticks(range(len(FOLLOW))); ax.set_yticklabels([f"~{10 * f:.0f}" for f in FOLLOW])
ax.set_xlabel("Selection: how often each of three judges picks the useful idea over a dud (%)")
ax.set_ylabel("Absorption: follow-ups per useful idea")
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Range of mechanisms in use at round 100, % of today", fontsize=8); cb.outline.set_visible(False)
cb.ax.axhline(100, color=INK, lw=0.8)
save(fig, "figure3_two_gates")
np.savetxt("tables/figure3_values.csv", today, delimiter=",", fmt="%.1f",
           header="spread at round 100 as % of start; rows: follow-ups ~" + " ".join(f"{10*f:.0f}" for f in FOLLOW) + "; cols: pairwise accuracy " + " ".join(f"{pair(q):.0f}" for q in QS))

# ---------- Tables 1 and 3: the summary and the robustness checks ----------
# Appendix Table A3: the four paths of proposal Figure 1 and the field without AI, at about 10 and about 40
# follow-ups per useful idea, three judges each right 96% of the time (q = 0.6, rho = 0.2), from the extras_gate runs.
def gstats(k):
    u = mG[k, -1, ix["useful_cells"]]; z = mG[k, -1, ix["noise_cells"]]; sp = mG[k, -1, ix["eff_mechanisms"]]
    return dict(useful=u, duds=z, hit=100 * u / max(u + z, 1), spread=sp, today=100 * sp / START,
                ai_useful=mG[k, -10:, ix["ai_build_useful"]].mean())
PATHS = [("Relying on AI", dict(regime="oracle", rely=1.0)), ("Generation alone", dict(regime="tail_only", rely=1.0)),
         ("Selection", dict(regime="convener", q=0.6, V=3, rho=0.2, rely=1.0)), ("No AI", dict(regime="tail_only", rely=0.0))]
rowsA3 = [(name, f, gstats(g(follow_rate=f, **kw))) for name, kw in PATHS for f in (1.0, 4.0)]
# spread from useful work only (duds' papers left out), from the main-rule rows of the absorption_variants runs
_AV = json.load(open("results/absorption_variants.json"))["rows"]
_AVP = dict(zip([n for n, _ in PATHS], ["relying_on_ai", "generation_alone", "selection", "no_ai"]))
for name, f, st in rowsA3:
    st["useful_spread"] = next(x["eff_mechanisms_useful"] for x in _AV if x["path"] == _AVP[name] and x["rule"] == "useful" and x["follow_rate"] == f)
    st["useful_today"] = 100 * st["useful_spread"] / START
with open("tables/tableA3_summary.md", "w") as fh:
    fh.write("# Table A3. The simulation in one table\n\n")
    fh.write(f"Round 100, means of {S} runs, 5% exploration. Selection uses three judges, each right 96% of the time, with a slight shared bias "
             f"toward convention. Spread is the effective number of mechanisms in use; it starts at about {START:.0f}.\n\n")
    fh.write("| Path | Follow-ups per useful idea | Useful ideas found | Duds | Hit rate | Spread at round 100 (% of today) | Spread from useful work only (% of today) | AI's recommendations into useful new work (papers per round, rounds 91–100) |\n|---|---|---|---|---|---|---|---|\n")
    for name, f, st in rowsA3:
        ai = "–" if name == "No AI" else f"{st['ai_useful']:.1f}"
        fh.write(f"| {name} | About {10 * f:.0f} | {st['useful']:,.0f} | {st['duds']:,.0f} | {st['hit']:.0f}% | {st['spread']:.0f} ({st['today']:.0f}%) | {st['useful_spread']:.0f} ({st['useful_today']:.0f}%) | {ai} |\n")
with open("tables/tableA3_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["path", "follow_ups", "useful", "duds", "hit_rate_pct", "spread_100", "pct_of_today", "useful_spread_100", "useful_pct_of_today", "ai_useful_per_round"])
    for name, f, st in rowsA3:
        w.writerow([name, round(10 * f), round(st["useful"]), round(st["duds"]), round(st["hit"], 1), round(st["spread"], 1),
                    round(st["today"], 1), round(st["useful_spread"], 1), round(st["useful_today"], 1), round(st["ai_useful"], 2)])

# Table 3 (appendix Table A4): robustness, from the extras_robust runs (five judges, about 20 follow-ups)
mR, CR, ixR = load("extras_robust")
def r(variant, regime): return [k for k, c in enumerate(CR) if c["_variant"] == variant and c["_regime"] == regime][0]
REG = ["Everyone relies on AI (oracle)", "Random untried ideas", "Spotting", "No AI"]   # keys stored in the results
def stats(k):
    u = mR[k, -1, ixR["useful_cells"]]; z = mR[k, -1, ixR["noise_cells"]]
    return dict(useful=u, duds=z, hit=100 * u / max(u + z, 1), spread=mR[k, -1, ixR["eff_mechanisms"]],
                ai_useful=mR[k, -10:, ixR["ai_build_useful"]].mean())
variants = list(dict.fromkeys(c["_variant"] for c in CR))
with open("tables/tableA4_robustness.md", "w") as fh, open("tables/tableA4_robustness.csv", "w", newline="") as fc:
    w = csv.writer(fc); w.writerow(["variant", "useful_generation_alone", "useful_selection", "hit_generation_alone", "hit_selection", "spread_relying_on_ai",
                                    "spread_generation_alone", "spread_selection", "spread_no_ai", "narrowing_undone_pct"])
    fh.write("# Table A4. Robustness checks\n\n")
    fh.write(f"Round 100, means of {S} runs, about 20 follow-ups per useful idea, five judges each right 96% of the time. Each row changes one setting from the baseline. "
             f"Spread starts at about {START:.0f}. Narrowing undone = (selection − relying on AI) / (start − relying on AI); it is not computed when the AI narrows the field by less than 100.\n\n")
    fh.write("| Setting changed | Useful ideas: generation alone → selection | Hit rate: generation alone / selection | Spread: relying on AI / generation alone / selection / no AI | Narrowing undone by selection |\n|---|---|---|---|---|\n")
    for v in variants:
        o, t, sp, na = (stats(r(v, rg)) for rg in REG)
        und = 100 * (sp["spread"] - o["spread"]) / (START - o["spread"])
        und_s = f"{und:.0f}%" if START - o["spread"] >= 100 else "— (AI narrows little)"
        fh.write(f"| {v} | {t['useful']:.0f} → {sp['useful']:.0f} | {t['hit']:.0f}% / {sp['hit']:.0f}% | "
                 f"{o['spread']:.0f} / {t['spread']:.0f} / {sp['spread']:.0f} / {na['spread']:.0f} | {und_s} |\n")
        w.writerow([v, round(t["useful"]), round(sp["useful"]), round(t["hit"], 1), round(sp["hit"], 1), round(o["spread"], 1),
                    round(t["spread"], 1), round(sp["spread"], 1), round(na["spread"], 1), round(und, 1)])

# ---------- Figure 5: the field at round 100 ----------
G = np.load("results/extras_grid.npz"); useful, empty0 = G["useful"], G["empty0"]
order = np.argsort(-G["start"].sum(0), kind="stable")
vmax = np.sqrt(max(G[k].max() for k in ("start", "oracle", "spotting", "noai")))
def rgb(n):
    img = np.ones(n.shape + (3,)) * np.array([0.975, 0.972, 0.965])
    lv = np.sqrt(n) / vmax
    est = (~empty0) & (n > 0); img[est] = (1 - (0.12 + 0.88 * lv[est]))[:, None] * np.array([0.92, 0.92, 0.94])
    nu = empty0 & useful & (n > 0); img[nu] = np.array([0.71, 0.28, 0.16])
    nd = empty0 & (~useful) & (n > 0); img[nd] = np.array([0.62, 0.71, 0.80])
    return img[:, order]
panels = [("Start", "start"), ("Round 100: relying on AI", "oracle"),
          ("Round 100: selection, ~20 follow-ups per useful idea", "spotting"), ("Round 100: no AI", "noai")]
fig, axes = plt.subplots(len(panels), 1, figsize=(6.6, 5.6), sharex=True, gridspec_kw={"hspace": 0.55})
for ax, (title, key) in zip(axes, panels):
    n = G[key]; ax.imshow(rgb(n), aspect="auto", interpolation="nearest")
    cs = np.sort(n.sum(0))[::-1]; top10 = 100 * cs[:10].sum() / cs.sum()
    ax.set_title(f"{title}   ·   spread {spread_of(n):.0f}   ·   top 10 mechanisms hold {top10:.0f}% of papers", fontsize=8, loc="left", color=INK)
    ax.set_yticks([]); ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
axes[-1].set_xticks([0, 799]); axes[-1].set_xticklabels(["most-studied mechanisms at the start", "unstudied at the start"], fontsize=7.5, color=GREY)
axes[-1].get_xticklabels()[0].set_ha("left"); axes[-1].get_xticklabels()[1].set_ha("right")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=(0.35, 0.35, 0.37), label="Studied at the start (darker = more papers)"),
                    Patch(color=(0.71, 0.28, 0.16), label="New pairing, useful"), Patch(color=(0.62, 0.71, 0.80), label="New pairing, dud")],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5)
fig.text(0.0, 0.5, "20 phenomena (rows) × 800 mechanisms (columns)", rotation=90, va="center", fontsize=7.5, color=GREY)
save(fig, "figureA3_field_at_round_100")
print("start spread", round(START, 1))
