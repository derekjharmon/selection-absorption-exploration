"""Selection, Absorption, and Exploration When AI Makes Ideas Cheap: simulation of a field on the grid (phenomena x mechanisms).

Each period every researcher gets k ideas proposed by a model that learns the record by frequency (plus a share
lam of tail proposals, as with Verbalized Sampling), chooses one, and publishes it in that cell. A cell that was empty
at the start is either useful (others build on it: follow-on papers for L periods) or noise (it stays a lone paper,
which the next model still learns from). Three ways of choosing:

  oracle     : the model rates the ideas and the researcher takes its top pick; it favours filled cells and popular
               mechanisms (convention).
  tail_only  : tail ideas are proposed and chosen, but with no signal of usefulness (picks an empty-cell idea at random).
  convener   : picks the empty-cell idea that a panel of V views rates highest. Each view mixes a usefulness signal
               (quality q) with a pull toward convention (rho: 0 = independent of convention, 1 = all convention)
               and its own noise. The oracle is the limit V=1, q=0, rho=1.

Each period a researcher explores with probability `explore` (choosing as the regime dictates) and otherwise builds on
existing work. With probability `rely` the build-on move takes the model's top-rated proposal; otherwise the researcher
picks from the record directly (a filled cell in their own row, in proportion to its papers), with no model involved. Building on is where the loop acts: whatever has been absorbed,
useful or noise, is learned by the model, proposed again and built on. Follow-on work happens only through this channel
by default (follow_rate=0); automatic follow-on for useful cells is a robustness option.
Adoption a = share of researchers who use the convener when exploring; the rest let the oracle choose.
follow_mode "extra": follow-on papers are added on top of the researchers' papers. "displace": they are written by the
researchers themselves, taking the place of build-on moves in the same row (capacity-limited).
absorb_err = chance the field responds wrongly to a newly filled cell: a useful cell gets no follow-on, or a noise cell
gets follow-on as if it were useful (0 = the field builds on exactly the useful cells).
"""
import numpy as np

DEFAULTS = dict(rows=20, cols=800, init_filled=40, init_papers=1500, b=0.10, per_row=25, k=10, lam=0.5, alpha=1.0,
                eps=0.5, gamma=1.0, follow_rate=0.0, follow_len=10, T=100, regime="convener", q=0.5, V=3, rho=0.3,
                adopt=1.0, loop=True, useful_tracks_popularity=0.0, explore=0.05, rely=1.0, follow_mode="displace", absorb_err=0.0,
                absorb_rule="useful")

def run(p, seed):
    p = {**DEFAULTS, **p}; rng = np.random.default_rng(seed)
    P, C = p["rows"], p["cols"]
    n = np.zeros((P, C))
    w = 1 / np.arange(1, p["init_filled"] + 1)
    for i in range(P):
        cols = rng.choice(C, p["init_filled"], replace=False)
        n[i, cols] = np.maximum(1, np.round(p["init_papers"] * w / w.sum()))
    empty0 = n == 0
    colpop0 = np.log1p(n.sum(0)); colz = (colpop0 - colpop0.mean()) / (colpop0.std() + 1e-9)
    # usefulness: base rate b among cells empty at t0; optionally more likely for popular mechanisms (robustness)
    logit_b = np.log(p["b"] / (1 - p["b"])) + p["useful_tracks_popularity"] * colz[None, :]
    useful = (rng.random((P, C)) < 1 / (1 + np.exp(-logit_b))) & empty0
    zu = np.where(useful, 1.0, 0.0); zu = (zu - zu[empty0].mean()) / (zu[empty0].std() + 1e-9)
    follow = np.zeros((P, C)); n_model0 = n.copy()
    N = P * p["per_row"]; rows_of = np.repeat(np.arange(P), p["per_row"])
    adopters = rng.random(N) < p["adopt"]
    hist = []
    for t in range(p["T"]):
        nm = n if p["loop"] else n_model0                     # what the model has learned
        colpop = np.log1p(nm.sum(0)); conv = (colpop - colpop.mean()) / (colpop.std() + 1e-9)
        base = (nm + p["eps"]) ** p["alpha"]; freq = base / base.sum(1, keepdims=True)
        emp = (nm == 0); tail = emp / np.maximum(emp.sum(1, keepdims=True), 1)
        prop = (1 - p["lam"]) * freq + p["lam"] * tail
        ai_b = ai_bu = ai_bz = ex_u = ex_z = 0               # this round: build-on moves the model chose, and where explorers landed
        disp = p["follow_mode"] == "displace"
        if p["absorb_rule"] == "popularity":                 # absorption that follows fashion: a new cell draws follow-ups with
            fp = n.sum(0); srt = np.sort(fp)                    # probability equal to its mechanism's popularity percentile in the field
            pct = (np.searchsorted(srt, fp, "left") + np.searchsorted(srt, fp, "right")) / (2 * C)
        if disp:                                               # this round's follow-on demand, served by build-on moves
            D = np.zeros((P, C)); act = follow > 0; D[act] = rng.poisson(p["follow_rate"], act.sum())
        for r in range(N):
            i = rows_of[r]
            g = rng.gumbel(size=C) + np.log(prop[i] + 1e-12)
            cand = np.argpartition(-g, p["k"])[: p["k"]]       # k distinct proposals
            empty_c = cand[n[i, cand] == 0]
            exploring = rng.random() < p["explore"]             # otherwise: build on existing work (model's top pick)
            use_conv = exploring and p["regime"] == "convener" and adopters[r]
            if disp and not exploring and D[i].sum() > 0:         # follow-on paper in place of a build-on move
                js = np.flatnonzero(D[i]); j = rng.choice(js, p=D[i, js] / D[i, js].sum()); D[i, j] -= 1
            elif exploring and p["regime"] == "tail_only" and len(empty_c):
                j = rng.choice(empty_c)
            elif use_conv and len(empty_c):
                q, V, rho = p["q"], p["V"], p["rho"]
                noise = np.sqrt(1 - q ** 2) * (np.sqrt(rho) * conv[empty_c][None, :] + np.sqrt(1 - rho) * rng.standard_normal((V, len(empty_c))))
                score = (q * zu[i, empty_c][None, :] + noise).mean(0)
                j = empty_c[np.argmax(score)]
            elif not exploring and rng.random() >= p["rely"]:   # build on without the model
                filled = np.flatnonzero(n[i] > 0)
                j = rng.choice(filled, p=n[i, filled] / n[i, filled].sum())
            else:                                              # oracle
                score = p["gamma"] * np.log1p(nm[i, cand]) + conv[cand] + rng.gumbel(size=len(cand))
                j = cand[np.argmax(score)]
                if not exploring:
                    ai_b += 1; ai_bu += useful[i, j]; ai_bz += (empty0[i, j] and not useful[i, j])
            if exploring:
                ex_u += useful[i, j]; ex_z += (empty0[i, j] and not useful[i, j])
            if n[i, j] == 0 and empty0[i, j]:
                e = p["absorb_err"]
                if p["absorb_rule"] == "popularity":
                    if rng.random() < pct[j]: follow[i, j] = p["follow_len"]
                elif (useful[i, j] and (e == 0 or rng.random() >= e)) or (not useful[i, j] and e > 0 and rng.random() < e):
                    follow[i, j] = p["follow_len"]
            n[i, j] += 1
        act = follow > 0
        if not disp: n[act] += rng.poisson(p["follow_rate"], act.sum())
        follow[act] -= 1
        U = useful & (n > 0); Z = (~useful) & empty0 & (n > 0)
        cs = n.sum(0) / n.sum(); cs = cs[cs > 0]
        nu = np.where((~useful) & empty0, 0.0, n).sum(0); cu = nu / nu.sum(); cu = cu[cu > 0]   # breadth from useful work: duds' papers left out
        hist.append(dict(t=t + 1, useful_cells=int(U.sum()), noise_cells=int(Z.sum()),
                         useful_papers=float(n[useful].sum()), noise_papers=float(n[(~useful) & empty0].sum()),
                         mean_papers=float(n[~empty0].sum()), total=float(n.sum()), useful_available=int(useful.sum()),
                         ai_build=ai_b, ai_build_useful=int(ai_bu), ai_build_noise=int(ai_bz), explore_useful=int(ex_u),
                         explore_noise=int(ex_z), eff_mechanisms=float(np.exp(-(cs * np.log(cs)).sum())),
                         eff_mechanisms_useful=float(np.exp(-(cu * np.log(cu)).sum()))))
    return (hist, n, useful, empty0) if p.get("return_grid") else hist
