# Placement v5 (Appendix A6). v5 is v4 plus two fixes, written after the v4 validation and tested on a held-out set:
#   4. Widening: every entry's names (from the menu, repaired, or added) are widened with up to six other names published
#      work uses for the same concept, including the standard name of the theory or construct when work on the concept uses
#      it; the widened list replaces the old one only if it passes the usability rule.
#   5. Distinct sides: if the phenomenon and the mechanism land on the same concept, the idea is read once more with the
#      instruction that they differ; if they still coincide, the case gets no call ("same concept on both sides").
# Held-out validation (cases_heldout.json), fixed before v5 was written: scored by the same rules as below.
#
# v4 (unchanged here) keeps v3's reading, choosing, counting, and rules, and adds three fixes:
#   1. Name repair: when a chosen entry's names fail the usability rule, GPT-5.5 proposes up to six other names published
#      research uses for the same concept; the entry's names plus these are rechecked by the same rule, then the new names
#      alone, and the first list that passes is used.
#   2. Entries on demand: when no vocabulary entry names the concept, GPT-5.5 gives its established name and search names,
#      which become an entry only if they pass the usability rule.
#   3. Stricter choosing: the choosing prompt answers "none" unless an entry names the same concept (which then goes to 2).
# Rules (unchanged from v3): tail if O <= 20 and O/min(R,C) < 1%; too small to judge if min(R,C) < 100;
#        names usable if they retrieve >= 50 works and >= 6 of the 10 most relevant are on target.
#
# Validation, scoring fixed before any v4 run (October 21, 2026), cases in cases.json:
#   known-filled (K1, K2, F1-F10): hit = "filled", miss = "tail", anything else = no call
#   known-empty (E1-E12):          hit = "tail",   miss = "filled", anything else = no call
#   K3:                            hit = "outside the social sciences"
#   Each case is placed three times; a case's call is the call made in at least two of three runs.
#
#   python3 placement_v5.py <v3|v4|v5> <run> [cases file]   writes results_<version>_<cases>_run<run>.json
import json, os, re, sys, time, urllib.parse, urllib.request, urllib.error
import numpy as np, threading
from concurrent.futures import ThreadPoolExecutor
LOCK = threading.Lock()
MODEL = "gpt-5.5"; SS = "12|14|18|20|32|33"
K = os.environ.get("OPENALEX_API_KEY") or (open(os.path.expanduser("~/.config/openalex_key")).read().strip() if os.path.exists(os.path.expanduser("~/.config/openalex_key")) else "")
HERE = os.path.dirname(os.path.abspath(__file__)); VOC = os.path.join(HERE, "..", "..", "vocab") + os.sep; os.chdir(HERE)
VERSION, RUN = sys.argv[1], sys.argv[2]; V4 = VERSION in ("v4", "v5"); V5 = VERSION == "v5"
CASES = sys.argv[3] if len(sys.argv) > 3 else "cases.json"; TAG = CASES.replace("cases", "").replace(".json", "").strip("_") or "main"
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower().strip('."')
def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=900) as r: return json.load(r)
        except urllib.error.HTTPError as e: print("HTTP", e.code, e.read().decode()[:120], flush=True); time.sleep(15 * (k + 1))
        except Exception as e: print("ERR", repr(e)[:120], flush=True); time.sleep(15 * (k + 1))
    raise RuntimeError("API failed")
def chat(p): return json.loads(post("https://api.openai.com/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": p}], "response_format": {"type": "json_object"}})["choices"][0]["message"]["content"])
def embed(texts):
    out = []
    for i in range(0, len(texts), 100): out += [d["embedding"] for d in post("https://api.openai.com/v1/embeddings", {"model": "text-embedding-3-large", "input": texts[i:i+100]})["data"]]
    E = np.array(out); return E / np.linalg.norm(E, axis=1, keepdims=True)
def oa(params):
    for k in range(5):
        try:
            with urllib.request.urlopen(f"https://api.openalex.org/works?{params}{'&api_key=' + K if K else ''}", timeout=60) as r: return json.load(r)
        except Exception: time.sleep(3 * (k + 1))
    return None
def orl(ns): return " OR ".join(f'"{n}"' if (" " in n or "-" in n) else n for n in ns)
def cnt(names):
    d = oa(f"filter=title_and_abstract.search:{urllib.parse.quote(orl(names))},primary_topic.field.id:{SS}&per-page=1&select=id"); return d["meta"]["count"] if d else None
mech = json.load(open(VOC + "menu_mechanisms.json")); rows = json.load(open(VOC + "menu_phenomena.json")); demo = json.load(open(VOC + "menu_demo.json"))
dm = {d["label"]: d["final_names"] for d in demo if d["kind"] == "mechanism"}; dr = {d["label"]: d["final_names"] for d in demo if d["kind"] == "phenomenon"}
M = [dict(name=m["canonical"], names=dm.get(m["canonical"], m["search_names"])) for m in mech]
R = [dict(name=r["canonical"], level=r["level"], names=dr.get(r["canonical"], r["search_names"])) for r in rows]
z = np.load("v3_emb.npz" if os.path.exists("v3_emb.npz") else os.path.join("..", "v3_emb.npz")); EM, ER = z["m"], z["r"]
CHK = json.load(open("namecheck_cache.json")) if os.path.exists("namecheck_cache.json") else {}
for d in demo: CHK[json.dumps(d["final_names"])] = dict(hits=d.get("fix_check", d)["hits"], on=d.get("fix_check", d)["on_target"])
def namecheck(label, kind, names):
    key = json.dumps(names)
    if key not in CHK:
        d = oa(f"filter=title_and_abstract.search:{urllib.parse.quote(orl(names))},primary_topic.field.id:{SS}&per-page=10&select=title&sort=relevance_score:desc")
        titles = [w["title"] for w in (d or {}).get("results", [])]
        j = chat(f"Concept ({kind}): {label}\nFor each title, is the work about this concept? Return only JSON: {{\"on\": [true, false]}}\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))) if titles else {"on": []}
        with LOCK:
            CHK[key] = dict(hits=(d or {}).get("meta", {}).get("count"), on=sum(bool(x) for x in j.get("on", [])[:len(titles)]))
            json.dump(CHK, open("namecheck_cache.json", "w"))
    c = CHK[key]; return c, (c["hits"] or 0) >= 50 and (c["on"] or 0) >= 6
def clean(ns):
    out = []
    for n in ns:
        n = norm(n)
        if n and len(n.split()) >= 2 and n not in out: out.append(n)
    return out
REPAIR = """Concept ({kind}{lvl}): {label}
Its search names {names} {why} in the titles and abstracts of social-science works.
Give up to six other names that published social-science research commonly uses for this same concept, each an exact phrase of two to five words. Keep the concept's meaning and level. Do not name a broader or a different concept, and do not give single words.
Return only JSON: {{"names": []}}"""
ADD = """An idea's {kind} reads: "{concept}"{lvl}.
Give the established name that social-science research uses for this {kind}, and up to six search names for it, each an exact phrase of two to five words that published work uses. Keep its meaning and level. Do not name a broader or a different concept.
Return only JSON: {{"name": "", "names": []}}"""
def repair(label, kind, level, names, chk):
    why = f"retrieve only {chk['hits']} works" if (chk["hits"] or 0) < 50 else "retrieve works mostly about other topics"
    new = clean(chat(REPAIR.format(kind=kind, lvl=f", {level} level" if level else "", label=label, names=names, why=why)).get("names", []))
    for cand in (clean(list(names) + new), new):
        if cand:
            c, ok = namecheck(label, kind, cand)
            if ok: return cand, c
    return None, None
def add(concept, kind, level):
    j = chat(ADD.format(kind=kind, concept=concept, lvl=f" ({level} level)" if level else ""))
    names = clean([j.get("name", "")] + j.get("names", []))
    if not names: return None
    c, ok = namecheck(j.get("name", concept), kind, names)
    return dict(name=norm(j.get("name", concept)), names=names, check=c) if ok else None
WIDEN = """Concept ({kind}{lvl}): {label}
Its search names are {names}. Published social-science work also refers to this same concept by other names, including the standard name of the theory or construct it belongs to when work on the concept uses that name.
Give up to six such names, each an exact phrase of two to five words. Do not name a different concept, and do not give single words.
Return only JSON: {{"names": []}}"""
def widen(label, kind, level, names):
    new = clean(chat(WIDEN.format(kind=kind, lvl=f", {level} level" if level else "", label=label, names=names)).get("names", []))
    cand = clean(list(names) + new)
    if cand == clean(names): return names, None
    c, ok = namecheck(label, kind, cand)
    return (cand, c) if ok else (names, None)
READ = """Read this research idea. Use only what its text states.
- phenomenon: what it seeks to explain (short phrase); phenomenon_quote: exact words from the idea; level: individual, group, organization, field or industry, or society.
- mechanism: the explanation it proposes, named by its established name if it has one (e.g., moral licensing, social contagion); mechanism_quote: exact words from the idea that state it. If the idea proposes no explanation (only a question, setting, or conditions), mechanism = "none".
- mechanism_home: "social sciences" if the mechanism is a social-science mechanism, "outside the social sciences" if it comes from the natural sciences, engineering or another non-social field (including analogies to physical processes).
Return only JSON: {"phenomenon": "", "phenomenon_quote": "", "level": "", "mechanism": "", "mechanism_quote": "", "mechanism_home": ""}

Idea: """
CHOOSE = """Idea: {idea}
Its phenomenon as read: {ph} ({lvl} level). Its mechanism as read: {me}.
Pick the vocabulary entry that names the SAME phenomenon (same concept and level, not merely related), or "none". Then the entry that names the SAME mechanism, or "none".{strict}
Phenomenon entries: {rows}
Mechanism entries: {mechs}
Return only JSON: {{"row": "<entry or none>", "mechanism": "<entry or none>"}}"""
STRICT = " If no entry names the same concept, answer \"none\"; do not pick the closest entry."
def finish(kind, name, names, src, level, r):
    if not V5: return name, names, src
    w, c = widen(name, kind, level, names)
    if c: r[f"{kind[:4]}_widened"] = dict(names=w, check=c); return name, w, src + "+widened"
    return name, names, src
def side(kind, entry, concept, level, r):
    """Return (name, names, source) for one side of the cell, or (None, None, reason)."""
    if entry is None:
        if not V4: return None, None, f"{kind} not on the menu"
        e = add(concept, kind, level)
        if not e: return None, None, f"{kind} not on the menu"
        r[f"{kind[:4]}_added"] = e; return finish(kind, e["name"], e["names"], "added", "", r)
    c, ok = namecheck(entry["name"], kind, entry["names"]); r[f"{kind[:4]}_check"] = c
    if ok: return finish(kind, entry["name"], entry["names"], "menu", entry.get("level", ""), r)
    if not V4: return None, None, "names unusable"
    names, c2 = repair(entry["name"], kind, entry.get("level", ""), entry["names"], c)
    if not names: return None, None, "names unusable"
    r[f"{kind[:4]}_repaired"] = dict(names=names, check=c2); return finish(kind, entry["name"], names, "repaired", entry.get("level", ""), r)
DISTINCT = "\nThe phenomenon and the mechanism must be different concepts. The mechanism is the explanation, not the thing explained.\n"
def place(case, retry=False):
    text = case["idea"]; r = dict(key=case["key"], idea=text, retry=retry)
    rd = chat(READ.replace("\n\nIdea: ", DISTINCT + "\nIdea: ") + text if retry else READ + text); r.update(rd); t = norm(text)
    mq = norm(rd.get("mechanism_quote", "")); r["mq_ok"] = bool(mq) and mq in t
    if norm(rd.get("mechanism", "none")) in ("none", "") or not r["mq_ok"]: r["cls"] = "no mechanism stated"; return r
    if rd.get("mechanism_home") == "outside the social sciences": r["cls"] = "outside the social sciences"; return r
    qe = embed([f"{rd['phenomenon']} ({rd.get('level','')} level)", rd["mechanism"]])
    ri = (ER @ qe[0]).argsort()[::-1][:8]; mi = (EM @ qe[1]).argsort()[::-1][:8]
    ch = chat(CHOOSE.format(idea=text, ph=rd["phenomenon"], lvl=rd.get("level", ""), me=rd["mechanism"], strict=STRICT if V4 else "",
                            rows="; ".join(f"{R[i]['name']} ({R[i]['level']})" for i in ri), mechs="; ".join(M[i]["name"] for i in mi)))
    rowm = next((R[i] for i in ri if norm(ch.get("row", "")).startswith(R[i]["name"])), None); mm = next((M[i] for i in mi if norm(ch.get("mechanism", "")) == M[i]["name"]), None)
    mname, mnames, msrc = side("mechanism", mm, rd["mechanism"], "", r)
    if not mnames: r["cls"] = msrc; return r
    rname, rnames, rsrc = side("phenomenon", rowm, rd["phenomenon"], rd.get("level", ""), r)
    if not rnames: r["cls"] = rsrc; return r
    r.update(row=rname, row_src=rsrc, row_names=rnames, mech=mname, mech_src=msrc, mech_names=mnames)
    if V5 and (set(rnames) & set(mnames) or norm(rname) == norm(mname)):
        if not retry: return place(case, retry=True)
        r["cls"] = "same concept on both sides"; return r
    Rn, Cn = cnt(rnames), cnt(mnames)
    q = urllib.parse.quote("(" + orl(rnames) + ") AND (" + orl(mnames) + ")")
    d = oa(f"filter=title_and_abstract.search:{q},primary_topic.field.id:{SS}&per-page=1&select=id"); On = d["meta"]["count"] if d else None
    r.update(R_=Rn, C_=Cn, O_=On); m = min(Rn or 0, Cn or 0); r["share"] = (On / m) if m and On is not None else None
    r["cls"] = "too small to judge" if m < 100 else ("tail" if On <= 20 and r["share"] < 0.01 else "filled")
    return r
if __name__ == "__main__":
    cases = json.load(open(CASES))["cases"]
    with ThreadPoolExecutor(4) as ex: res = list(ex.map(place, cases))
    json.dump({"version": VERSION, "run": RUN, "results": res}, open(f"results_{VERSION}_{TAG}_run{RUN}.json", "w"), indent=1)
    for c, r in zip(cases, res):
        print(f"{r['key']:4s} expected {c['expected']:28s} got {r['cls']:30s} | {r.get('row')} [{r.get('row_src')}] x {r.get('mech')} [{r.get('mech_src')}] | R {r.get('R_')} C {r.get('C_')} O {r.get('O_')}", flush=True)
