# Mechanism menu: drop generic labels, then merge names for the same mechanism.
# Checks set before running:
#   generic dropped: learning, selection, coordination, integration, cooperation
#   specific kept: moral licensing, adverse selection, agenda setting, social contagion, isomorphism (some form)
#   must stay apart: moral licensing/moral cleansing; adverse selection/moral hazard; agenda setting/framing
#   must merge (where both names exist): path dependence/path dependency; sensemaking/sense-making; signaling/signaling theory
import json, os, re, time, urllib.request, urllib.error
import numpy as np
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
MODEL = "gpt-5.5"
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'")).strip().lower().strip('."')
def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=1200) as r: return json.load(r)
        except urllib.error.HTTPError as e: print("HTTP", e.code, e.read().decode()[:150], flush=True); time.sleep(10 * (k + 1))
        except Exception as e: print("ERR", repr(e)[:150], flush=True); time.sleep(10 * (k + 1))
    raise RuntimeError("API failed")
def chat(prompt): return json.loads(post("https://api.openai.com/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}})["choices"][0]["message"]["content"])
def embed(texts):
    out = []
    for i in range(0, len(texts), 100): out += [d["embedding"] for d in post("https://api.openai.com/v1/embeddings", {"model": "text-embedding-3-large", "input": texts[i:i+100]})["data"]]
    E = np.array(out); return E / np.linalg.norm(E, axis=1, keepdims=True)
recs = [r for l in open("extracted.jsonl") for r in json.loads(l)["records"]]
M = [dict(name=norm(m["name"]), explains=m.get("explains", ""), field=r["field"], rid=r["id"]) for r in recs for m in (r.get("mechanisms") or []) if m.get("quote_ok") and m.get("name")]
info = defaultdict(lambda: {"n": 0, "fields": Counter(), "explains": Counter()})
for m in M:
    d = info[m["name"]]; d["n"] += 1; d["fields"][m["field"]] += 1
    if m["explains"]: d["explains"][norm(m["explains"])] += 1
names = sorted(info); print("mechanism mentions", len(M), "| distinct names", len(names), flush=True)
# 1. generic filter (cached)
if os.path.exists("generic.json"): verdicts = json.load(open("generic.json"))
else:
    GEN = """For each name below, decide what it is, for building a search vocabulary of social-science mechanisms.
- "specific": a named causal mechanism a scholar would recognize, whose name used as a search phrase would mostly find work on this mechanism (e.g., moral licensing, adverse selection, agenda setting, institutional isomorphism, social contagion, status signaling).
- "generic": a general process word or broad category that would find work on many unrelated things (e.g., learning, selection, coordination, integration, cooperation). If a specific established form exists that is clearly what is meant, give it in rename_to (e.g., "signaling" -> "signaling theory", "framing" -> "framing effects"); otherwise leave rename_to empty.
- "not a mechanism": a topic, outcome, variable, field or method.
Return only JSON: {"items": [{"name": "", "verdict": "specific|generic|not a mechanism", "rename_to": ""}]}

Names:
"""
    batches = [names[i:i+150] for i in range(0, len(names), 150)]
    def run(b):
        out = chat(GEN + "\n".join(b)).get("items", []); by = {norm(o.get("name", "")): o for o in out}
        return {n: by.get(n, {"verdict": "missing"}) for n in b}
    verdicts = json.load(open("generic_partial.json")) if os.path.exists("generic_partial.json") else {}
    todo = [b for b in batches if not all(n in verdicts for n in b)]
    def run_save(b):
        d = run(b); verdicts.update(d); json.dump(verdicts, open("generic_partial.json", "w")); return len(d)
    with ThreadPoolExecutor(4) as ex: list(ex.map(run_save, todo))
    json.dump(verdicts, open("generic.json", "w"), indent=1)
vc = Counter(v.get("verdict") for v in verdicts.values()); print("verdicts", dict(vc), flush=True)
kept = {}  # final name -> list of source names
for n in names:
    v = verdicts.get(n, {}); r = norm(v.get("rename_to") or "")
    if v.get("verdict") == "specific": kept.setdefault(n, []).append(n)
    elif v.get("verdict") == "generic" and r: kept.setdefault(r, []).append(n)
K = sorted(kept); print("kept names (after renames)", len(K), flush=True)
def ctx(k):
    ex = Counter();
    for src in kept[k]: ex.update(info[src]["explains"])
    return k + (": " + ex.most_common(1)[0][0] if ex else "")
E = embed([ctx(k) for k in K]); S = E @ E.T
iu = np.triu_indices(len(K), 1); cand = [(K[i], K[j]) for i, j in zip(*iu) if S[i, j] >= 0.75]
print("candidate pairs (cosine >= .75)", len(cand), flush=True)
# 2. verify candidate pairs (cached)
if os.path.exists("pairs.json"): pv = json.load(open("pairs.json"))
else:
    VER = """For each pair of mechanism names, decide: "same" (one is a synonym, spelling variant or alternate name of the other, so a search for either should include both), "related" (overlapping or one is a subtype, but scholars treat them as distinct), or "different".
Return only JSON: {"pairs": [{"a": "", "b": "", "verdict": "same|related|different"}]}

Pairs:
"""
    pb = [cand[i:i+80] for i in range(0, len(cand), 80)]
    def runp(b):
        out = chat(VER + "\n".join(f"{a} | {c}" for a, c in b)).get("pairs", [])
        return [[norm(o.get("a", "")), norm(o.get("b", "")), o.get("verdict")] for o in out]
    pv = []
    with ThreadPoolExecutor(8) as ex:
        for d in ex.map(runp, pb): pv += d
    json.dump(pv, open("pairs.json", "w"), indent=1)
par = {k: k for k in K}
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b, v in pv:
    if v == "same" and a in par and b in par: par[find(a)] = find(b)
cl = defaultdict(list)
for k in K: cl[find(k)].append(k)
menu = []
for root, mem in cl.items():
    srcs = [s for m in mem for s in kept[m]]; n = sum(info[s]["n"] for s in srcs)
    fields = Counter(); [fields.update(info[s]["fields"]) for s in srcs]
    canon = max(mem, key=lambda m: (sum(info[s]["n"] for s in kept[m]), -len(m)))
    menu.append(dict(canonical=canon, variants=sorted(set(mem) | set(srcs)), mentions=n, fields=dict(fields)))
menu.sort(key=lambda x: -x["mentions"]); json.dump(menu, open("menu_mechanisms.json", "w"), indent=1)
print("same pairs", sum(v == "same" for *_, v in pv), "| menu entries", len(menu), flush=True)
# checks
where = {v: m["canonical"] for m in menu for v in m["variants"]}
def present(x): return any(x in v for v in where)
gen_ok = all(not any(v == g for v in where) for g in ["learning", "selection", "coordination", "integration", "cooperation"])
spec_ok = {s: present(s) for s in ["moral licensing", "adverse selection", "agenda setting", "social contagion", "isomorphism"]}
apart = [("moral licensing", "moral cleansing"), ("adverse selection", "moral hazard"), ("agenda setting", "framing")]
def same(a, b):
    ca = {where[v] for v in where if v == a or v.startswith(a)}; cb = {where[v] for v in where if v == b or v.startswith(b)}
    return bool(ca & cb) if ca and cb else None
apart_res = {f"{a} / {b}": same(a, b) for a, b in apart}
together = [("path dependence", "path dependency"), ("sensemaking", "sense-making"), ("signaling theory", "signaling")]
tog_res = {f"{a} / {b}": same(a, b) for a, b in together}
print("\nCHECKS")
print("  generic examples dropped:", gen_ok)
print("  specific kept:", spec_ok)
print("  must stay apart (False = apart, None = one missing):", apart_res)
print("  must merge (True = merged, None = one missing):", tog_res)
print("\nlargest merged entries:")
for m in [x for x in menu if len(x["variants"]) > 1][:25]: print(f"  {m['canonical']} ({m['mentions']}): {', '.join(m['variants'][:8])}")
print("\nfield spread of menu:", dict(Counter(max(m['fields'], key=m['fields'].get) for m in menu)))
