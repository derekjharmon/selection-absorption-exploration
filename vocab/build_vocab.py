# Vocabulary from review articles .
# Rows (phenomena) from management reviews: Academy of Management Annals (all) and Journal of Management reviews
# (title has review / meta-analysis / meta-analytic / integrative framework / research agenda), 2011-2025.
# Columns (mechanisms) from source-field reviews: Annual Review of Sociology, Psychology, Political Science, Economics;
# Annals of the International Communication Association. 2011-2025. Abstracts only (OpenAlex), IDs kept for reproduction.
# Step 1: pull abstracts -> sources.json. Step 2: extract with GPT-5.5 -> extracted.jsonl (one line per batch; resumable).
# Checks set in advance: >= 90% of quotes verbatim; the mechanism menu must contain moral licensing and board interlocks.
import json, os, re, time, urllib.parse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
K = os.environ.get("OPENALEX_API_KEY", ""); MODEL = "gpt-5.5"
def oa(path):
    for k in range(5):
        try:
            with urllib.request.urlopen(f"https://api.openalex.org/{path}{(('&' if '?' in path else '?') + 'api_key=' + K) if K else ''}", timeout=90) as r: return json.load(r)
        except Exception as e: time.sleep(3 * (k + 1))
    raise RuntimeError(path)
def source_id(name):
    r = oa(f"sources?search={urllib.parse.quote(name)}&per-page=5")["results"]
    return next(x for x in r if x["display_name"].lower() == name.lower())["id"].split("/")[-1]
def ab(inv):
    if not inv: return ""
    pos = {p: w for w, ps in inv.items() for p in ps}; return " ".join(pos[i] for i in sorted(pos))
SETS = [("management", "Academy of Management Annals", None),
        ("management", "Journal of Management", 'review OR "meta-analysis" OR "meta-analytic" OR "integrative framework" OR "research agenda"'),
        ("sociology", "Annual Review of Sociology", None), ("psychology", "Annual Review of Psychology", None),
        ("political science", "Annual Review of Political Science", None), ("economics", "Annual Review of Economics", None),
        ("communication", "Annals of the International Communication Association", None)]
if not os.path.exists("sources.json"):
    docs = []
    for field, journal, tq in SETS:
        sid = source_id(journal); flt = f"primary_location.source.id:{sid},publication_year:2011-2025,has_abstract:true"
        if tq: flt += ",title.search:" + urllib.parse.quote(tq)
        cur = "*"
        while cur:
            d = oa(f"works?filter={flt}&per-page=200&cursor={cur}&select=id,doi,title,publication_year,abstract_inverted_index,type")
            for w in d["results"]:
                a = ab(w.get("abstract_inverted_index"))
                if len(a.split()) >= 40 and w.get("type") not in ("erratum", "editorial", "paratext"):
                    docs.append(dict(id=w["id"].split("/")[-1], doi=w.get("doi"), title=w["title"], year=w["publication_year"], journal=journal, field=field, abstract=a))
            cur = d["meta"].get("next_cursor") if d["results"] else None
        print(journal, sum(x["journal"] == journal for x in docs), flush=True)
    json.dump(docs, open("sources.json", "w"), indent=1)
docs = json.load(open("sources.json")); print("total abstracts", len(docs), flush=True)
MGMT = """Below are abstracts of review articles in management. Use only what each abstract states.
For each review return:
- literature: the literature it reviews, as a short name (2-6 words).
- phenomena: up to 6 phenomena this literature studies, meaning what it seeks to explain (an outcome or a process), stated without the explanation. Each: name (2-6 words), level (individual, group, organization, field or industry, or society), quote (the exact words of the abstract that state it).
- mechanisms: up to 6 named causal processes the abstract mentions as explanations (established names, e.g. social contagion, institutional isomorphism, upper echelons). Settings, populations, variables, outcomes and methods are not mechanisms. Each: name (1-5 words), quote (exact words).
Return only JSON: {"reviews": [{"id": "W1", "literature": "", "phenomena": [{"name": "", "level": "", "quote": ""}], "mechanisms": [{"name": "", "quote": ""}]}]}

Abstracts:
"""
SRC = """Below are abstracts of review articles in {field}. Use only what each abstract states.
For each review return mechanisms: up to 6 named causal processes established in {field} that the review covers (e.g. moral licensing, adverse selection, agenda setting, social contagion, status signaling). A mechanism is a process that explains why something happens. Topics, settings, populations, variables, findings and methods are not mechanisms. Each: name (the established name, 1-5 words), explains (what it explains, 2-8 words), quote (the exact words of the abstract that state it). If the abstract names no mechanism, return an empty list.
Return only JSON: {{"reviews": [{{"id": "W1", "mechanisms": [{{"name": "", "explains": "", "quote": ""}}]}}]}}

Abstracts:
"""
def call(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=1200) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, e.read().decode()[:200], flush=True); time.sleep(10 * (k + 1))
        except Exception as e:
            print("ERR", repr(e)[:200], flush=True); time.sleep(10 * (k + 1))
    raise RuntimeError("API failed")
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip().lower().strip('."')
done = set()
if os.path.exists("extracted.jsonl"):
    for line in open("extracted.jsonl"): done.add(json.loads(line)["batch"])
batches = []
for field in ["management", "sociology", "psychology", "political science", "economics", "communication"]:
    fd = [d for d in docs if d["field"] == field]
    for i in range(0, len(fd), 15): batches.append((f"{field}-{i//15:03d}", field, fd[i:i+15]))
todo = [b for b in batches if b[0] not in done]; print("batches", len(batches), "to do", len(todo), flush=True)
def run(b):
    key, field, ds = b
    prompt = (MGMT if field == "management" else SRC.format(field=field)) + "\n\n".join(f"[{d['id']}] {d['title']}\n{d['abstract']}" for d in ds)
    out = json.loads(call(prompt)["choices"][0]["message"]["content"]).get("reviews", [])
    by = {o.get("id"): o for o in out}; recs = []
    for d in ds:
        o = by.get(d["id"], {}); a = norm(d["abstract"] + " " + d["title"])
        for lst in ("phenomena", "mechanisms"):
            for x in o.get(lst, []) or []: x["quote_ok"] = bool(norm(x.get("quote", ""))) and norm(x.get("quote", "")) in a
        recs.append(dict(id=d["id"], field=field, journal=d["journal"], title=d["title"], year=d["year"], **{k: v for k, v in o.items() if k != "id"}))
    with open("extracted.jsonl", "a") as f: f.write(json.dumps({"batch": key, "records": recs}) + "\n")
    return key
with ThreadPoolExecutor(8) as ex:
    for k in ex.map(run, todo): pass
recs = [r for line in open("extracted.jsonl") for r in json.loads(line)["records"]]
ph = [p for r in recs for p in r.get("phenomena", []) or []]; me = [m for r in recs for m in r.get("mechanisms", []) or []]
print(f"reviews {len(recs)} | phenomena {len(ph)} (quotes verbatim {sum(p['quote_ok'] for p in ph)/max(1,len(ph)):.0%}) | mechanisms {len(me)} (verbatim {sum(m['quote_ok'] for m in me)/max(1,len(me)):.0%})")
from collections import Counter
print("mechanisms by field:", dict(Counter(r["field"] for r in recs for m in (r.get("mechanisms") or []))))
names = Counter(norm(m["name"]) for m in me)
print("distinct mechanism names:", len(names)); print("most common:", names.most_common(25))
for probe in ["moral licensing", "interlock", "board interlock", "isomorphism", "social contagion", "agenda setting", "adverse selection"]:
    print(f"  menu has '{probe}':", [n for n in names if probe in n][:5])
