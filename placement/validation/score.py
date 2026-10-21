# Scores the validation runs by the rules fixed in placement_v4.py before any v4 run.
#   python3 score.py                       the first set (cases.json): v3 and v4
#   python3 score.py cases_heldout.json    the held-out set: v4 and v5
# A call whose phenomenon and mechanism share a name (the same concept on both sides) is listed as "degenerate" and
# counted as no call; the rules fixed in advance did not anticipate it, so the tally reports it separately.
import json, os, sys, collections
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CASES = sys.argv[1] if len(sys.argv) > 1 else "cases.json"; TAG = CASES.replace("cases", "").replace(".json", "").strip("_") or "main"
cases = json.load(open(CASES))["cases"]
def files(v):
    for i in (1, 2, 3):
        for f in (f"results_{v}_{TAG}_run{i}.json", f"results_{v}_run{i}.json" if TAG == "main" else None):
            if f and os.path.exists(f): yield f; break
def call(r):
    rn, mn = set(r.get("row_names") or []), set(r.get("mech_names") or [])
    return "degenerate" if r.get("cls") == "filled" and rn & mn else r["cls"]
def outcome(c, cls):
    if c["key"] == "K3": return "hit" if cls == "outside the social sciences" else "miss"
    hit, miss = ("filled", "tail") if c["expected"] == "filled" else ("tail", "filled")
    return "hit" if cls == hit else ("miss" if cls == miss else "no call")
lines = [f"Validation set {CASES}"]
for v in ("v3", "v4", "v5"):
    runs = [json.load(open(f))["results"] for f in files(v)]
    if not runs: continue
    lines.append(f"\n{v}: {len(runs)} runs")
    tally = collections.Counter(); agree = 0; per_run = collections.Counter(); degen = 0
    for c in cases:
        calls = [call(next(r for r in run if r["key"] == c["key"])) for run in runs]
        degen += calls.count("degenerate")
        cl, n = collections.Counter(calls).most_common(1)[0]
        o = outcome(c, cl) if n >= 2 else "no call"
        grp = "known-filled" if c["expected"] == "filled" else ("known-empty" if c["expected"] == "tail" else "outside")
        tally[(grp, o)] += 1; agree += len(set(calls)) == 1
        for x in calls: per_run[(grp, outcome(c, x))] += 1
        lines.append(f"  {c['key']:4s} {grp:12s} {' | '.join(calls):95s} -> {o}")
    for grp in ("known-filled", "known-empty", "outside"):
        if any(k[0] == grp for k in tally):
            lines.append(f"  {grp}: " + ", ".join(f"{o} {tally[(grp, o)]}" for o in ("hit", "miss", "no call")) +
                         "   (single runs: " + ", ".join(f"{o} {per_run[(grp, o)]}" for o in ("hit", "miss", "no call")) + ")")
    judged = sum(tally[(g, o)] for g in ("known-filled", "known-empty") for o in ("hit", "miss")); right = sum(tally[(g, "hit")] for g in ("known-filled", "known-empty"))
    lines.append(f"  two-sided cases judged: {judged} of {sum(1 for c in cases if c['key'] != 'K3')}, correct {right}; degenerate single-run calls {degen}; same call in every run: {agree} of {len(cases)}")
out = "\n".join(lines) + "\n"; open(f"score_{TAG}.txt", "w").write(out); print(out)
