# Part 1: phenomena rows from management reviews (classify, merge within level).
# Part 2: fill mechanism gaps from OpenAlex keywords for mechanisms named in earlier placement runs and the known cases.
# Checks set before running:
#   rows include: misconduct (organization level), organizational legitimacy, market categories or category emergence,
#     occupations or professions; broad words dropped: performance, behavior, outcomes, effectiveness
#   gaps: board interlocks (interlocking directorates) present in the menu after filling
import json, os, re, time, urllib.parse, urllib.request, urllib.error
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
MODEL = "gpt-5.5"; K = os.environ.get("OPENALEX_API_KEY", "")  # optional: OpenAlex works without a key at a lower rate limit
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'")).strip().lower().strip('."')
def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=1200) as r: return json.load(r)
        except urllib.error.HTTPError as e: print("HTTP", e.code, e.read().decode()[:120], flush=True); time.sleep(15 * (k + 1))
        except Exception as e: print("ERR", repr(e)[:120], flush=True); time.sleep(15 * (k + 1))
    raise RuntimeError("API failed")
def chat(p): return json.loads(post("https://api.openai.com/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p}], "response_format": {"type": "json_object"}})["choices"][0]["message"]["content"])
def embed(texts):
    out = []
    for i in range(0, len(texts), 100): out += [d["embedding"] for d in post("https://api.openai.com/v1/embeddings", {"model": "text-embedding-3-large", "input": texts[i:i+100]})["data"]]
    E = np.array(out); return E / np.linalg.norm(E, axis=1, keepdims=True)
def cached_batches(items, size, fn, path):
    done = json.load(open(path)) if os.path.exists(path) else {}
    todo = [items[i:i+size] for i in range(0, len(items), size)]; todo = [b for b in todo if not all(str(x) in done for x in b)]
    def run(b):
        d = fn(b); done.update(d); json.dump(done, open(path, "w")); return 1
    with ThreadPoolExecutor(4) as ex: list(ex.map(run, todo))
    return done
# ---------- Part 1: rows ----------
recs = [r for l in open("extracted.jsonl") for r in json.loads(l)["records"] if r["field"] == "management"]
P = [dict(name=norm(p["name"]), level=norm(p.get("level", "")), lit=r.get("literature", ""), rid=r["id"]) for r in recs for p in (r.get("phenomena") or []) if p.get("quote_ok") and p.get("name")]
pinfo = defaultdict(lambda: {"n": 0, "levels": Counter(), "lits": Counter()})
for p in P: d = pinfo[p["name"]]; d["n"] += 1; d["levels"][p["level"]] += 1; d["lits"][p["lit"]] += 1
pnames = sorted(pinfo); print("phenomenon mentions", len(P), "| distinct", len(pnames), flush=True)
CLS = """For each phenomenon name below (from management review articles), decide, for building a search vocabulary:
- "phenomenon": a specific outcome or process that a literature seeks to explain, whose name used as a search phrase would mostly find work on it (e.g., organizational misconduct, CEO succession, new venture legitimacy, employee voice, category emergence).
- "too broad": a general word that would find work on almost anything (e.g., performance, behavior, outcomes, effectiveness, change). If a specific form is clearly meant, give it in rename_to; otherwise leave it empty.
- "not a phenomenon": a theory, mechanism, method, measure or field label.
Return only JSON: {"items": [{"name": "", "verdict": "phenomenon|too broad|not a phenomenon", "rename_to": ""}]}

Names:
"""
def cls_fn(b):
    out = chat(CLS + "\n".join(b)).get("items", []); by = {norm(o.get("name", "")): o for o in out}
    return {n: by.get(n, {"verdict": "missing"}) for n in b}
PV = cached_batches(pnames, 150, cls_fn, "rows_verdicts.json")
print("row verdicts", dict(Counter(v.get("verdict") for v in PV.values())), flush=True)
kept = defaultdict(list)  # (name, level) -> source names
for n in pnames:
    v = PV.get(n, {}); lvl = pinfo[n]["levels"].most_common(1)[0][0]; r = norm(v.get("rename_to") or "")
    if v.get("verdict") == "phenomenon": kept[(n, lvl)].append(n)
    elif v.get("verdict") == "too broad" and r: kept[(r, lvl)].append(n)
KP = sorted(kept); print("kept rows before merge", len(KP), flush=True)
EP = embed([f"{n} ({l} level)" for n, l in KP]); S = EP @ EP.T
cand = [(KP[i], KP[j]) for i, j in zip(*np.triu_indices(len(KP), 1)) if S[i, j] >= 0.70 and KP[i][1] == KP[j][1]]
print("row candidate pairs (same level, cosine >= .70)", len(cand), flush=True)
VER = """For each pair of phenomenon names (same level of analysis), decide: "same" (synonyms or alternate names, so a search for either should include both), "related" (overlapping, or one is part of the other, but studied as distinct), or "different".
Return only JSON: {"pairs": [{"id": 0, "verdict": "same|related|different"}]}

Pairs:
"""
def ver_fn(b):
    out = chat(VER + "\n".join(f"{i}: {a[0]} | {c[0]}" for i, (a, c) in enumerate(b))).get("pairs", [])
    by = {int(o.get("id", -1)): o.get("verdict") for o in out if str(o.get("id", "")).lstrip("-").isdigit()}
    return {json.dumps([a, c]): by.get(i, "missing") for i, (a, c) in enumerate(b)}
cand_keys = [json.dumps([list(a), list(c)]) for a, c in cand]
PVP = cached_batches([[list(a), list(c)] for a, c in cand], 80, lambda b: {json.dumps(x): v for x, v in zip(b, ver_fn([(tuple(x[0]), tuple(x[1])) for x in b]).values())}, "rows_pairs.json") if cand else {}
par = {k: k for k in KP}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for k, v in PVP.items():
    if v == "same": a, c = json.loads(k); par[find(tuple(a))] = find(tuple(c))
cl = defaultdict(list)
for k in KP: cl[find(k)].append(k)
rows = []
for root, mem in cl.items():
    srcs = [s for m in mem for s in kept[m]]; lits = Counter(); [lits.update(pinfo[s]["lits"]) for s in srcs]
    canon = max(mem, key=lambda m: (sum(pinfo[s]["n"] for s in kept[m]), -len(m[0])))
    rows.append(dict(canonical=canon[0], level=canon[1], search_names=sorted({m[0] for m in mem}), provenance_names=sorted(set(srcs)),
                     mentions=sum(pinfo[s]["n"] for s in srcs), literatures=[l for l, _ in lits.most_common(5)]))
rows.sort(key=lambda r: -r["mentions"]); json.dump(rows, open("menu_phenomena.json", "w"), indent=1)
print("rows after merge", len(rows), "| levels", dict(Counter(r["level"] for r in rows)), flush=True)
allnames = {n for r in rows for n in r["search_names"]}
def has(rx): return [n for n in allnames if re.search(rx, n)][:6]
print("\nROW CHECKS")
for lab, rx in [("misconduct", r"misconduct|wrongdoing|fraud|corporate crime"), ("organizational legitimacy", r"legitima"),
                ("categories", r"categor"), ("occupations/professions", r"occupation|profession")]:
    print(f"  {lab}: {has(rx) or 'MISSING'}")
print("  broad words absent:", {w: w not in allnames for w in ["performance", "behavior", "outcomes", "effectiveness"]})
# ---------- Part 2: mechanism gaps ----------
menu = json.load(open("menu_mechanisms.json"))
need = ["moral licensing", "interlocking directorates", "legitimacy spillover"]
# The original run also took the mechanisms named in earlier placement runs on AI-generated ideas (Appendix A5). Those
# ideas are excluded from this repository for blind review, so a rerun checks the known cases only. The 13 entries the
# original run added are marked source = "openalex keywords" in menu_mechanisms.json.
if os.path.exists("placement_v2.json"):
    v2 = json.load(open("placement_v2.json"))["all_reads"]
    need += [norm(r["mechanism"]) for r in v2 if str(r.get("mechanism", "none")).lower() not in ("none", "") and r.get("mq_ok")]
need = sorted(set(need)); print("\nmechanisms needed:", len(need), flush=True)
mnames = [n for m in menu for n in m["search_names"]]; owner = {n: m["canonical"] for m in menu for n in m["search_names"]}
EM, EN = embed(mnames), embed(need); sim = EN @ EM.T
missing = [need[i] for i in range(len(need)) if sim[i].max() < 0.80]
print("already on the menu (cosine >= .80):", len(need) - len(missing), "| missing:", len(missing), flush=True)
def oa(path):
    for k in range(5):
        try:
            with urllib.request.urlopen(f"https://api.openalex.org/{path}{'&api_key=' + K if K else ''}", timeout=60) as r: return json.load(r)
        except Exception: time.sleep(3 * (k + 1))
    return {"results": []}
added = []
for n in missing:
    ks = [(x["display_name"], x["works_count"]) for x in oa(f"keywords?search={urllib.parse.quote(n)}&per-page=5")["results"]]
    if not ks: continue
    q = f"Mechanism: {n}\nCandidate keyword names: " + "; ".join(k for k, _ in ks) + '\nWhich candidates are names for this same mechanism (synonyms or variants usable as search phrases, not broader or unrelated terms)? Return only JSON: {"same": ["..."]}'
    same = [norm(x) for x in chat(q).get("same", []) if norm(x) in {norm(k) for k, _ in ks}]
    if same: added.append(dict(canonical=same[0], search_names=sorted(set(same)), provenance_names=[n], mentions=0, fields={"openalex keywords": 1}, source="openalex keywords"))
menu += added; json.dump(menu, open("menu_mechanisms.json", "w"), indent=1)
print("added from OpenAlex keywords:", len(added), [a["canonical"] for a in added][:20])
print("GAP CHECK board interlocks:", [m["canonical"] for m in menu if any("interlock" in s for s in m["search_names"])] or "MISSING")
