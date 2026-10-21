#!/usr/bin/env python3
"""One backward + one forward snowballing iteration (Wohlin 2014) via the
Semantic Scholar Graph API.

  snowball.py --run main --seeds included_ids.txt --out snowball_it1.jsonl --against dedup_all.jsonl

Seeds: one identifier per line (arXiv:ID or DOI:...). For each seed we fetch its references
(backward) and its citations (forward), keep records inside the date window, drop anything already
screened, and write normalized records for screening with the same prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, http_get, norm_title, now_iso  # noqa: E402
from dedup import dedup  # noqa: E402
from search import START, norm_s2  # noqa: E402

FIELDS = "title,abstract,externalIds,year,venue,publicationDate,authors,publicationTypes"


def fetch(pid: str, direction: str) -> tuple[int, list[dict]]:
    key = "citedPaper" if direction == "references" else "citingPaper"
    out, offset = [], 0
    while True:
        st, body = http_get(f"https://api.semanticscholar.org/graph/v1/paper/{pid}/{direction}",
                            params={"fields": FIELDS, "limit": 1000, "offset": offset}, retries=12)
        if st != 200:
            return st, out
        d = json.loads(body)
        out += [x[key] for x in (d.get("data") or []) if x.get(key) and x[key].get("title")]
        if d.get("next") is None or offset + 1000 >= 9000:
            return 200, out
        offset = d["next"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--against", default="dedup_all.jsonl")
    a = ap.parse_args()
    run = REPO / "sok" / "slr" / "runs" / a.run
    seeds = [l.strip() for l in (run / a.seeds).open() if l.strip()]
    rawp = run / "raw" / a.out.replace(".jsonl", "_raw.jsonl")
    log = {"started": now_iso(), "seeds": len(seeds), "per_seed": {}}
    recs = []
    with open(rawp, "w") as raw:
        for sid in seeds:
            pid = sid.replace("doi:", "DOI:").replace("arxiv:", "arXiv:")
            for direction in ("references", "citations"):
                st, items = fetch(pid, direction)
                log["per_seed"].setdefault(sid, {})[direction] = {"status": st, "n": len(items)}
                for p in items:
                    raw.write(json.dumps({"seed": sid, "direction": direction, "paper": p}) + "\n")
                    r = norm_s2(p)
                    r["source"] = f"snowball-{'backward' if direction == 'references' else 'forward'}"
                    r["snowball_seed"] = sid
                    recs.append(r)
    n_all = len(recs)
    recs = [r for r in recs if (r.get("date") or f"{r.get('year') or 0}-12-31") >= START]
    n_window = len(recs)
    have = [json.loads(l) for l in (run / a.against).open()]
    ax = {r["arxiv_id"] for r in have if r.get("arxiv_id")}
    dois = {d for r in have for d in (r.get("dois") or [r.get("doi")]) if d}
    titles = {norm_title(r.get("title", "")) for r in have}
    new = [r for r in recs if not ((r.get("doi") and r["doi"] in dois) or (r.get("arxiv_id") and r["arxiv_id"] in ax)
                                   or norm_title(r["title"]) in titles)]
    out = dedup(new)
    for r in out:
        r["rid"] = "S" + r["rid"][1:]
    with open(run / a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.update(finished=now_iso(), candidates_total=n_all, in_window=n_window, not_already_screened=len(new),
               unique_new=len(out),
               failed_calls=sum(1 for v in log["per_seed"].values() for d in v.values() if d["status"] != 200))
    (run / a.out.replace(".jsonl", "_log.json")).write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "per_seed"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
