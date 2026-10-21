# Proposal Figure 1: what the simulation shows. Today on the left, four paths stacked on the right, each with a bar
# for spread at round 100 as a share of the starting spread (means of 20 runs, 5% exploration, follow-ups displace
# researchers' own papers, from results/extras_gate, the runs behind Figure 3). The grids are illustrative.
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model import run
from sweep import S
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
AXIS = "#5b616b"
def spread_of(n):
    col = n.sum(0).astype(float); sh = col[col > 0] / col.sum(); return float(np.exp(-(sh * np.log(sh)).sum()))
START = np.mean([spread_of(run({"T": 0, "return_grid": True}, s)[1]) for s in range(S)])
_d = np.load("results/extras_gate.npz"); _m = json.load(open("results/extras_gate.json"))
_ix = _m["keys"].index("eff_mechanisms")
def final_spread(**kw):
    k = [i for i, c in enumerate(_m["conds"]) if all(c.get(a) == b for a, b in kw.items())]
    assert len(k) == 1, kw
    return float(_d["mean"][k[0], -1, _ix])
SP = {"rely": final_spread(regime="oracle", rely=1.0, follow_rate=1.0),
      "gen": final_spread(regime="tail_only", rely=1.0, follow_rate=1.0),
      "sel10": final_spread(regime="convener", q=0.6, V=3, rho=0.2, rely=1.0, follow_rate=1.0),
      "sel40": final_spread(regime="convener", q=0.6, V=3, rho=0.2, rely=1.0, follow_rate=4.0)}
R_ = {k: v / START for k, v in SP.items()}
def lab(k): return f"{'widens' if R_[k] >= 1 else 'narrows'} to {100 * R_[k]:.0f}% of today"
INK, GREY = "#1f2328", "#8a8f98"
WORK = {0: "#ffffff", 1: "#d9dbe0", 2: "#9ea3ad", 3: "#555b66"}
USEFUL, USEFUL_LIGHT, NOISE, EDGE = "#b5482a", "#ecc3b6", "#9db3c9", "#c9ccd1"

BASE = [[3,2,2,1,0,0,0,0,0,0],
        [2,3,1,2,1,0,0,0,0,0],
        [3,1,0,2,0,1,0,0,0,0],
        [2,3,2,0,1,0,0,0,1,0],
        [1,2,3,1,0,0,1,0,0,0],
        [3,2,1,0,2,0,0,0,0,0],
        [2,1,2,1,0,0,0,1,0,0]]
HIDDEN = {(0,5),(1,7),(2,4),(3,6),(5,3),(6,8)}
R, C = 7, 10
def base(r, c): return BASE[r][c]
def piled(r, c):
    v = BASE[r][c]
    return 3 if v >= 2 else (2 if v == 1 and c <= 2 else v)

def cell(ax, x, y, s, kind, level=0):
    g = 0.12 * s
    style = {"noise": (NOISE, NOISE, "-", 0.6), "useful": (USEFUL, USEFUL, "-", 0.6),
             "light": (USEFUL_LIGHT, USEFUL, "-", 0.6), "hidden": ("#ffffff", USEFUL, (0, (2, 1.5)), 0.8),
             "work": (WORK[level], EDGE, "-", 0.5)}[kind]
    fc, ec, ls, lw = style
    ax.add_patch(Rectangle((x + g / 2, y + g / 2), s - g, s - g, facecolor=fc, edgecolor=ec, linestyle=ls, linewidth=lw))

def draw_grid(ax, x0, y0, s, val, noise=(), filled=(), light=()):
    for r in range(R):
        for c in range(C):
            x, y = x0 + c * s, y0 + (R - 1 - r) * s
            if (r, c) in noise: cell(ax, x, y, s, "noise")
            elif (r, c) in filled: cell(ax, x, y, s, "useful")
            elif (r, c) in light: cell(ax, x, y, s, "light")
            elif (r, c) in HIDDEN: cell(ax, x, y, s, "hidden")
            else: cell(ax, x, y, s, "work", val(r, c))

def draw_bar(ax, x0, y, ratio, unit, label, today=False):
    h = 0.55
    color = GREY if today else (INK if ratio < 1 else USEFUL)
    ax.add_patch(Rectangle((x0, y), unit * ratio, h, facecolor=color, edgecolor="none", alpha=0.85, zorder=2))
    if not today:
        ax.add_patch(Rectangle((x0, y), unit, h, facecolor="none", edgecolor="#ffffff" if ratio >= 1 else GREY,
                               linewidth=0.7, linestyle=(0, (2, 1.5)), zorder=3))
    ax.text(x0 + max(unit, unit * ratio) + 0.3, y + h / 2, label, fontsize=7.5, color=INK, va="center")

OUTCOMES = [
    ("Relying on AI", "Researchers build on the AI's top picks,\nand the same cells fill.",
     piled, (), (), (), R_["rely"], lab("rely")),
    ("Generation alone", "Researchers try untried ideas at random,\nmostly duds, and convention still piles up.",
     piled, {(0,8),(1,9),(2,7),(3,9),(4,5),(5,8),(6,6)}, {(1,7)}, (), R_["gen"], lab("gen")),
    ("Selected, too little absorption", "Researchers select useful tail ideas, but\nothers write only about 10 follow-ups on each.",
     piled, {(4,9)}, (), {(0,5),(2,4),(3,6),(5,3),(6,8)}, R_["sel10"], lab("sel10")),
    ("Selected and absorbed", "Researchers select useful tail ideas, and\nothers write about 40 follow-ups on each.",
     base, {(4,9)}, {(0,5),(2,4),(3,6),(5,3),(6,8)}, (), R_["sel40"], lab("sel40")),
]

n = len(OUTCOMES)
S0, S1 = 1.0, 0.62
row_h = R * S1 + 1.9
H = n * row_h
W = 44
LEG = 7.7                                   # height reserved for the key
fig, ax = plt.subplots(figsize=(7.2, 7.2 * (H + LEG + 1.0) / W))
ax.set_xlim(-1.2, W); ax.set_ylim(-LEG, H + 0.6); ax.set_aspect("equal"); ax.axis("off")

# Today
ty = H / 2 - R * S0 / 2 + 1.0
draw_grid(ax, 0, ty, S0, base)
ax.text(0, ty + R * S0 + 0.5, "Today", fontsize=10, fontweight="bold", color=INK, va="bottom")
ax.text(-0.35, ty + R * S0 / 2, "phenomena \u2192", rotation=90, ha="right", va="center", fontsize=7.5, color=AXIS, style="italic")
ax.text(0, ty - 0.3, "mechanisms from any field \u2192", fontsize=7.5, color=AXIS, va="top", style="italic")
ax.text(0, ty - 1.75, "Work piles up in a few cells\nof each row. Empty cells are\nthe tail, and a few of them\nare useful.",
        fontsize=7.5, color=GREY, va="top")
ax.text(0, ty - 5.8, "Range of mechanisms in use", fontsize=7.5, color=INK, va="top", fontweight="bold")
unit = 5.0
draw_bar(ax, 0, ty - 7.15, 1.0, unit, "100%", today=True)

# Outcomes
gx = 15.0; tx = gx + C * S1 + 1.2
for i, (title, sub, val, noise, filled, light, ratio, lab) in enumerate(OUTCOMES):
    y0 = H - (i + 1) * row_h + 0.9
    draw_grid(ax, gx, y0, S1, val, noise, filled, light)
    ax.annotate("", xy=(gx - 0.3, y0 + R * S1 / 2), xytext=(C * S0 + 0.4, ty + R * S0 / 2),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=7))
    top = y0 + R * S1
    ax.text(tx, top, title, fontsize=9, fontweight="bold", color=INK, va="top")
    ax.text(tx, top - 0.95, sub, fontsize=7.5, color=GREY, va="top")
    draw_bar(ax, tx, y0 + 0.05, ratio, unit, lab)

# Key, in two groups
def key_item(x, y, kind, label, level=0):
    cell(ax, x, y - 0.5, 0.8, kind, level)
    ax.text(x + 1.15, y - 0.1, label, fontsize=7.2, color=INK, va="center")
ky = -2.2
ax.plot([0, 40], [-0.8, -0.8], color=EDGE, lw=0.7)                    # rule between figure and key
ax.text(0, ky, "Existing work", fontsize=7.5, fontweight="bold", color=INK, va="center")
key_item(0, ky - 1.2, "work", "Much work (convention)", 3)
key_item(0, ky - 2.4, "work", "Some work", 1)
key_item(0, ky - 3.6, "work", "Empty (the tail)", 0)
kx = 13.0
ax.text(kx, ky, "Tail ideas", fontsize=7.5, fontweight="bold", color=INK, va="center")
key_item(kx, ky - 1.2, "hidden", "Useful, not yet pursued")
key_item(kx, ky - 2.4, "light", "Useful, selected but not absorbed")
key_item(kx, ky - 3.6, "useful", "Useful, absorbed (others build on it)")
key_item(kx, ky - 4.8, "noise", "Dud (pursued, but not useful)")

for ext in ("png", "pdf"): fig.savefig(f"figures/figure1_paths.{ext}", dpi=220, bbox_inches="tight")
print("start", round(START, 1), {k: round(v, 1) for k, v in SP.items()}, {k: f"{100*v:.0f}%" for k, v in R_.items()})
