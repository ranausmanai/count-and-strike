"""Download StatsBomb open data for every competition-season with 360 frames: events + 360 frames, gzipped."""
import gzip, json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sb360")
def get(u):
    for i in range(4):
        try:
            return urllib.request.urlopen(u, timeout=90).read()
        except Exception:
            if i == 3: raise
comps = [c for c in json.loads(get(f"{BASE}/competitions.json")) if c.get("match_available_360")]
allm = []
for c in comps:
    ms = json.loads(get(f"{BASE}/matches/{c['competition_id']}/{c['season_id']}.json"))
    for m in ms:
        m["_comp"] = f"{c['competition_name']} {c['season_name']}"
    allm += ms
    print(c["competition_name"], c["season_name"], len(ms), flush=True)
json.dump(allm, open(os.path.join(OUT, "..", "sb360_matches.json"), "w"))
def one(m):
    mid = m["match_id"]
    for kind in ("events", "three-sixty"):
        p = os.path.join(OUT, f"{mid}_{kind}.json.gz")
        if os.path.exists(p):
            continue
        try:
            raw = get(f"{BASE}/{kind}/{mid}.json")
        except Exception:
            continue
        with gzip.open(p + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(p + ".tmp", p)
with ThreadPoolExecutor(16) as ex:
    for i, _ in enumerate(ex.map(one, allm)):
        if i % 50 == 0: print(i, "/", len(allm), flush=True)
print("done", len(allm))
