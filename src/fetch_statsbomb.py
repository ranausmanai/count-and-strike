"""Download StatsBomb open-data events for the full 2015/16 big-five league seasons.

Raw event JSON is stored gzipped (data/raw/sb/<match_id>.json.gz) so every later
step can be re-run from disk. Data: StatsBomb Open Data (https://github.com/statsbomb/open-data),
used under the StatsBomb public data user agreement (attribution required).
"""
import gzip, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sb")
COMPS = {2: "EPL", 11: "LaLiga", 9: "Bundesliga", 12: "SerieA", 7: "Ligue1"}
SEASON = 27  # 2015/2016


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa
            if i == tries - 1:
                raise
    return None


def fetch_match(mid):
    path = os.path.join(OUT, f"{mid}.json.gz")
    if os.path.exists(path):
        return mid, "cached"
    raw = get(f"{BASE}/events/{mid}.json")
    with gzip.open(path + ".tmp", "wb") as f:
        f.write(raw)
    os.replace(path + ".tmp", path)
    return mid, "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    all_matches = []
    for cid, name in COMPS.items():
        ms = json.loads(get(f"{BASE}/matches/{cid}/{SEASON}.json"))
        for m in ms:
            m["_league"] = name
        all_matches += ms
        print(name, len(ms), flush=True)
    with open(os.path.join(OUT, "..", "sb_matches.json"), "w") as f:
        json.dump(all_matches, f)
    ids = [m["match_id"] for m in all_matches]
    done = 0
    with ThreadPoolExecutor(16) as ex:
        for mid, st in ex.map(fetch_match, ids):
            done += 1
            if done % 100 == 0:
                print(done, "/", len(ids), flush=True)
    print("done", len(ids))


if __name__ == "__main__":
    main()
