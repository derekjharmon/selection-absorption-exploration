# Placement v3 on the first five pairings a 2022-2025 article claims to study first (Appendix A6, Validating placement).
# Cut from placement_v3.py, with its three known cases replaced by the five pairings; run as: python3 placement_v3_first_five.py <run>.
# Original header of placement_v3.py follows.
# Placement v3 on the three known cases (Appendix A6, Table A7).
# Placement reads an idea, chooses its phenomenon and mechanism from the review-built vocabulary (no invented names),
# checks the chosen names on demand, and counts the cell in OpenAlex. Each known case looks like tail and is not:
#   K1 moral licensing x firms' CSR and misconduct   -> filled
#   K2 legitimacy x board interlocks                  -> filled
#   K3 fatigue crack growth x organizational misconduct -> outside the social sciences
# Rules: tail if O <= 20 and O/min(R,C) < 1%; too small to judge if min(R,C) < 100;
#        names usable if they retrieve >= 50 works and >= 6 of the 10 most relevant are on target.
#
#   python3 placement_v3.py          rerun the three cases (needs OPENAI_API_KEY; OPENALEX_API_KEY optional)
#   python3 placement_v3.py --dry    load the vocabulary and stop, with no API calls
#
# Cost: one-time embedding of the vocabulary (about 1,900 short names) plus about a dozen GPT-5.5 calls, well under US$1.
# Output: placement_v3_rerun.json, printed next to the recorded results in placement_v3_known.json (counts as of
# October 6, 2026). OpenAlex counts drift between queries, so a rerun can differ slightly from the record.
import json, os, re, time, urllib.parse, urllib.request, urllib.error, sys
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
MODEL = "gpt-5.5"; K = os.environ.get("OPENALEX_API_KEY", ""); SS = "12|14|18|20|32|33"
HERE = os.path.dirname(os.path.abspath(__file__)); VOC = os.path.join(HERE, "..", "..", "..", "vocab") + os.sep; os.chdir(HERE)
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
def cnt(names, extra=f",primary_topic.field.id:{SS}"):
    d = oa(f"filter=title_and_abstract.search:{urllib.parse.quote(orl(names))}{extra}&per-page=1&select=id"); return d["meta"]["count"] if d else None
# vocabulary: full menus, with checked names from menu_demo.json taking precedence
mech = json.load(open(VOC + "menu_mechanisms.json")); rows = json.load(open(VOC + "menu_phenomena.json")); demo = json.load(open(VOC + "menu_demo.json"))
dm = {d["label"]: d["final_names"] for d in demo if d["kind"] == "mechanism"}; dr = {d["label"]: d["final_names"] for d in demo if d["kind"] == "phenomenon"}
M = [dict(name=m["canonical"], names=dm.get(m["canonical"], m["search_names"])) for m in mech]
R = [dict(name=r["canonical"], level=r["level"], names=dr.get(r["canonical"], r["search_names"])) for r in rows]
if "--dry" in sys.argv: print(f"vocabulary loaded: {len(R)} phenomena, {len(M)} mechanisms; no API calls made"); sys.exit(0)
if os.path.exists(os.path.join("..", "..", "v3_emb.npz")): z = np.load(os.path.join("..", "..", "v3_emb.npz")); EM, ER = z["m"], z["r"]
else: EM, ER = embed([m["name"] for m in M]), embed([f"{r['name']} ({r['level']} level)" for r in R]); np.savez("v3_emb.npz", m=EM, r=ER)
CHK = json.load(open("v3_namecheck.json")) if os.path.exists("v3_namecheck.json") else {}
for d in demo: CHK[json.dumps(d["final_names"])] = dict(hits=d.get("fix_check", d)["hits"], on=d.get("fix_check", d)["on_target"])
def namecheck(label, kind, names):
    key = json.dumps(names)
    if key not in CHK:
        d = oa(f"filter=title_and_abstract.search:{urllib.parse.quote(orl(names))},primary_topic.field.id:{SS}&per-page=10&select=title&sort=relevance_score:desc")
        titles = [w["title"] for w in (d or {}).get("results", [])]
        j = chat(f"Concept ({kind}): {label}\nFor each title, is the work about this concept? Return only JSON: {{\"on\": [true, false]}}\n\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))) if titles else {"on": []}
        CHK[key] = dict(hits=(d or {}).get("meta", {}).get("count"), on=sum(bool(x) for x in j.get("on", [])[:len(titles)]))
        json.dump(CHK, open("v3_namecheck.json", "w"))
    c = CHK[key]; return c, (c["hits"] or 0) >= 50 and (c["on"] or 0) >= 6
READ = """Read this research idea. Use only what its text states.
- phenomenon: what it seeks to explain (short phrase); phenomenon_quote: exact words from the idea; level: individual, group, organization, field or industry, or society.
- mechanism: the explanation it proposes, named by its established name if it has one (e.g., moral licensing, social contagion); mechanism_quote: exact words from the idea that state it. If the idea proposes no explanation (only a question, setting, or conditions), mechanism = "none".
- mechanism_home: "social sciences" if the mechanism is a social-science mechanism, "outside the social sciences" if it comes from the natural sciences, engineering or another non-social field (including analogies to physical processes).
Return only JSON: {"phenomenon": "", "phenomenon_quote": "", "level": "", "mechanism": "", "mechanism_quote": "", "mechanism_home": ""}

Idea: """
CHOOSE = """Idea: {idea}
Its phenomenon as read: {ph} ({lvl} level). Its mechanism as read: {me}.
Pick the vocabulary entry that names the SAME phenomenon (same concept and level, not merely related), or "none". Then the entry that names the SAME mechanism, or "none".
Phenomenon entries: {rows}
Mechanism entries: {mechs}
Return only JSON: {{"row": "<entry or none>", "mechanism": "<entry or none>"}}"""
def ab(inv):
    if not inv: return ""
    pos = {p: w for w, ps in inv.items() for p in ps}; return " ".join(pos[i] for i in sorted(pos))
def place(item):
    key, text = item; r = dict(key=key, idea=text)
    rd = chat(READ + text); r.update(rd); t = norm(text)
    mq = norm(rd.get("mechanism_quote", "")); r["mq_ok"] = bool(mq) and mq in t
    if norm(rd.get("mechanism", "none")) in ("none", "") or not r["mq_ok"]: r["cls"] = "no mechanism stated"; return r
    if rd.get("mechanism_home") == "outside the social sciences": r["cls"] = "outside the social sciences"; return r
    qe = embed([f"{rd['phenomenon']} ({rd.get('level','')} level)", rd["mechanism"]])
    ri = (ER @ qe[0]).argsort()[::-1][:8]; mi = (EM @ qe[1]).argsort()[::-1][:8]
    ch = chat(CHOOSE.format(idea=text, ph=rd["phenomenon"], lvl=rd.get("level", ""), me=rd["mechanism"],
                            rows="; ".join(f"{R[i]['name']} ({R[i]['level']})" for i in ri), mechs="; ".join(M[i]["name"] for i in mi)))
    rowm = next((R[i] for i in ri if norm(ch.get("row", "")).startswith(R[i]["name"])), None); mm = next((M[i] for i in mi if norm(ch.get("mechanism", "")) == M[i]["name"]), None)
    r["row"], r["mech"] = (rowm or {}).get("name"), (mm or {}).get("name")
    if not mm: r["cls"] = "mechanism not on the menu"; return r
    if not rowm: r["cls"] = "phenomenon not on the menu"; return r
    cr, okr = namecheck(rowm["name"], "phenomenon", rowm["names"]); cm, okm = namecheck(mm["name"], "mechanism", mm["names"])
    r["row_check"], r["mech_check"] = cr, cm
    if not (okr and okm): r["cls"] = "names unusable"; return r
    Rn, Cn = cnt(rowm["names"]), cnt(mm["names"])
    q = urllib.parse.quote("(" + orl(rowm["names"]) + ") AND (" + orl(mm["names"]) + ")")
    d = oa(f"filter=title_and_abstract.search:{q},primary_topic.field.id:{SS}&per-page=1&select=id"); On = d["meta"]["count"] if d else None
    r.update(R_=Rn, C_=Cn, O_=On); m = min(Rn or 0, Cn or 0); r["share"] = (On / m) if m else None
    r["cls"] = "too small to judge" if m < 100 else ("tail" if On <= 20 and r["share"] < 0.01 else "filled")
    return r
EMPTY = [("E1", "Does the fraud triangle of pressure, opportunity, and rationalization explain why firms hold ISO 14001 environmental certification without implementing it?", "W2025 Business Strategy & Development"),
         ("E2", "When an organization's leader commits a public transgression, does the transgression threaten members' identity, so that they disengage from the organization and stop buying from it?", "W2024 European Journal of Marketing"),
         ("E3", "Does the theory of planned behavior explain a firm's green innovation, with subjective norms coming from its supply chain partners?", "W2022 Int. J. Operations & Production Management"),
         ("E4", "Does a leader's humor spur employees' creative work behavior by raising their prosocial motivation?", "W2024 Leadership & Organization Development Journal"),
         ("E5", "In hostile takeovers, do the target's managers shape how employees respond through sensemaking and sensegiving?", "W2022 Journal of Indian Business Research")]
import sys
run = sys.argv[1]
with ThreadPoolExecutor(3) as ex: res = list(ex.map(place, [(k, t) for k, t, _ in EMPTY]))
for r in res:
    print(f"  {r['key']}: {r['cls']} | row {r.get('row')} | mech {r.get('mech')} | R {r.get('R_')} C {r.get('C_')} O {r.get('O_')} | share {r.get('share')} | read: {r.get('phenomenon')} / {r.get('mechanism')}", flush=True)
json.dump({"empty": res}, open(f"empty_run_{run}.json", "w"), indent=1)
