# Re-verify the vocabulary's quotes without redistributing abstract text.
# Each extracted phenomenon or mechanism carries quote_span = [start, end], character offsets into the normalized text
# (abstract + " " + title) of its source review. This script refetches each source from OpenAlex by ID, rebuilds and
# normalizes the text exactly as build_vocab.py did, checks it against the stored SHA-1 (so a changed abstract is
# detected, not silently misread; OpenAlex re-assigns some IDs over time, so missing IDs are looked up by DOI), and prints each quote next to the name extracted from it.
#   python3 verify_quotes.py            check every source (about 50 OpenAlex requests)
#   python3 verify_quotes.py --show 20  also print the first 20 name/quote pairs
# Optional: OPENALEX_API_KEY in the environment (works without one at a lower rate limit).
import json, os, re, sys, time, hashlib, urllib.request, urllib.parse
K = os.environ.get("OPENALEX_API_KEY", "")
def norm(s): return re.sub(r"\s+", " ", str(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')).strip().lower().strip('."')
def ab(inv):
    if not inv: return ""
    pos = {p: w for w, ps in inv.items() for p in ps}; return " ".join(pos[i] for i in sorted(pos))
def fetch(ids):
    url = f"https://api.openalex.org/works?filter=openalex_id:{'|'.join(ids)}&per-page=50&select=id,title,abstract_inverted_index" + (f"&api_key={K}" if K else "")
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r: return json.load(r)["results"]
        except Exception: time.sleep(3 * (k + 1))
    raise RuntimeError("OpenAlex request failed")
here = os.path.dirname(os.path.abspath(__file__))
sources = {r["id"]: r for r in json.load(open(os.path.join(here, "sources_ids.json")))}
text, ids = {}, list(sources)
for i in range(0, len(ids), 50):
    for w in fetch(ids[i:i + 50]):
        wid = w["id"].split("/")[-1]; text[wid] = norm(ab(w.get("abstract_inverted_index")) + " " + (w.get("title") or ""))
for wid in [i for i in ids if i not in text]:          # OpenAlex re-assigns some IDs over time; fall back to the DOI
    doi = sources[wid].get("doi")
    if not doi: continue
    url = f"https://api.openalex.org/works/{urllib.parse.quote(doi, safe=':/')}?select=id,title,abstract_inverted_index" + (f"&api_key={K}" if K else "")
    try:
        with urllib.request.urlopen(url, timeout=90) as r: w = json.load(r)
        text[wid] = norm(ab(w.get("abstract_inverted_index")) + " " + (w.get("title") or ""))
    except Exception: pass
same = {i for i in text if hashlib.sha1(text[i].encode()).hexdigest() == sources[i]["text_sha1"]}
print(f"sources refetched {len(text)} of {len(ids)} | text identical to the extraction run {len(same)}")
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
n = ok = 0
for line in open(os.path.join(here, "extracted_noquotes.jsonl")):
    for rec in json.loads(line)["records"]:
        for kind in ("phenomena", "mechanisms"):
            for e in rec.get(kind, []):
                if "quote_span" not in e or rec["id"] not in same: continue
                s, t = e["quote_span"]; q = text[rec["id"]][s:t]; n += 1; ok += bool(q)
                if show > 0: print(f"  {('phenomenon' if kind == 'phenomena' else 'mechanism'):10s} {e['name']!r:45s} <- {q!r}"); show -= 1
print(f"quotes reconstructed from identical sources: {ok} of {n}")
