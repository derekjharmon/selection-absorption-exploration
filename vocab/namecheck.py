# Name-list check for the four literatures' rows and the mechanisms the ideas need. Flags set before running:
#   too narrow = names retrieve < 50 works; off target = < 6 of the 10 most relevant works are about the concept.
# Flagged lists get a suggested fix (better names, merge, or drop), and fixed names are re-checked the same way.
import json, os, re, time, urllib.parse, urllib.request, urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor
MODEL = "gpt-5.5"; K = os.environ.get("OPENALEX_API_KEY", ""); SS = "12|14|18|20|32|33"
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'")).strip().lower().strip('."')
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
def orl(ns): return " OR ".join(f'"{n}"' if " " in n or "-" in n else n for n in ns)
def check(kind, label, names, level=""):
    q = urllib.parse.quote(orl(names)); d = oa(f"filter=title_and_abstract.search:{q},primary_topic.field.id:{SS}&per-page=10&select=title,publication_year&sort=relevance_score:desc")
    if not d: return dict(hits=None, on_target=None, titles=[])
    titles = [f"{w['publication_year']} {w['title']}" for w in d["results"]]
    j = chat(f"Concept ({kind}{', ' + level + ' level' if level else ''}): {label}\nFor each title below, is the work about this concept? Return only JSON: {{\"on\": [true, false, ...]}}\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))) if titles else {"on": []}
    on = sum(bool(x) for x in j.get("on", [])[:len(titles)])
    return dict(hits=d["meta"]["count"], on_target=on, n_titles=len(titles), titles=titles)
def flag(r): return "too narrow" if (r["hits"] or 0) < 50 else ("off target" if (r["on_target"] or 0) < 6 else "ok")
rows = json.load(open("menu_phenomena.json")); menu = json.load(open("menu_mechanisms.json"))
LITS = [("Misconduct", r"misconduct|wrongdoing|fraud|corporate crime|scandal|corruption|unethical"), ("Legitimacy", r"legitima"),
        ("Categories and nascent markets", r"categor|nascent|emerging market|new market|market emergence|market creation"), ("Occupations and professions", r"occupation|profession")]
items = []
for lit, rx in LITS:
    for r in rows:
        if any(re.search(rx, n) for n in r["search_names"]): items.append(dict(kind="phenomenon", lit=lit, label=r["canonical"], level=r["level"], names=r["search_names"], others=None))
# The mechanisms to check were those named in one researcher's 40 AI-generated ideas (Appendix A7), matched to the
# vocabulary by embedding similarity. Those ideas are excluded from this repository for blind review, so the matched
# list is read back from the recorded output of the original run.
for c in sorted({x["label"] for x in json.load(open("namecheck.json")) if x["kind"] == "mechanism"}):
    m = next(x for x in menu if x["canonical"] == c); items.append(dict(kind="mechanism", lit="Mechanisms the ideas need", label=c, level="", names=m["search_names"], others=None))
print("lists to check:", len(items), flush=True)
def run(it):
    it.update(check(it["kind"], it["label"], it["names"], it["level"])); it["flag"] = flag(it)
    if it["flag"] != "ok":
        sib = [x["label"] for x in items if x["lit"] == it["lit"] and x["label"] != it["label"]]
        fx = chat(f"""A search-vocabulary entry failed a check ({it['flag']}: {it['hits']} works retrieved; {it['on_target']} of {it.get('n_titles', 0)} top results on target).
Entry ({it['kind']}{', ' + it['level'] + ' level' if it['level'] else ''}): {it['label']}; search names: {it['names']}.
Other entries in the same list: {sib[:25]}
Propose ONE fix: "rename" (2-5 common names scholars actually use for this concept, as search phrases), "merge" (it is the same as one of the other entries; name which), or "drop" (not a {it['kind']}, or too vague to search). Give a one-sentence reason.
Return only JSON: {{"action": "rename|merge|drop", "names": [], "merge_with": "", "reason": ""}}""")
        it["fix"] = fx
        if fx.get("action") == "rename" and fx.get("names"):
            r2 = check(it["kind"], it["label"], [norm(n) for n in fx["names"]], it["level"]); r2["flag"] = flag(r2); it["fix_check"] = r2
    return it
with ThreadPoolExecutor(4) as ex: items = list(ex.map(run, items))
json.dump(items, open("namecheck.json", "w"), indent=1)
from collections import Counter
print("flags:", dict(Counter((i["kind"], i["flag"]) for i in items)))
fixes = [i for i in items if i["flag"] != "ok"]
print("fix actions:", dict(Counter(i["fix"].get("action") for i in fixes)), "| renames that now pass:", sum(1 for i in fixes if i.get("fix_check", {}).get("flag") == "ok"), "of", sum(1 for i in fixes if i["fix"].get("action") == "rename"))
# review sheet
L = ["# Vocabulary review: four literatures", "", "Each entry is a row (a phenomenon) or a mechanism, with the names the tool searches for. **Hits** = works in OpenAlex's social sciences that mention any of the names; **On target** = how many of the 10 most relevant works are actually about the concept.",
     "Flags: *too narrow* (under 50 hits) or *off target* (under 6 of 10). Flagged entries show a suggested fix, re-checked the same way.", "",
     "**What to do:** for each entry, mark OK, or write what's wrong (wrong level, not a real phenomenon, should merge with another, missing a name). Add any phenomenon the literature studies that's missing.", ""]
for lit in [l for l, _ in LITS] + ["Mechanisms the ideas need"]:
    L += [f"## {lit}", "", "| Entry | Level | Names searched | Hits | On target | Flag | Suggested fix | Your call |", "|---|---|---|---|---|---|---|---|"]
    for i in [x for x in items if x["lit"] == lit]:
        fx = ""
        if i["flag"] != "ok":
            f = i["fix"]; a = f.get("action")
            if a == "rename": c = i.get("fix_check", {}); fx = f"rename to {', '.join(f.get('names', []))} → {c.get('hits')} hits, {c.get('on_target')}/10 ({c.get('flag')})"
            elif a == "merge": fx = f"merge with *{f.get('merge_with')}*"
            else: fx = "drop"
            fx += f". {f.get('reason', '')}"
        L.append(f"| {i['label']} | {i['level'] or '–'} | {', '.join(i['names'][:4])} | {i['hits'] if i['hits'] is not None else '?'} | {i['on_target']}/10 | {i['flag']} | {fx} | |")
    L.append("")
open("review_four_literatures.md", "w").write("\n".join(L)); print("wrote review_four_literatures.md")
