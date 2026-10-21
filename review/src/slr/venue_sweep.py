#!/usr/bin/env python3
"""Venue sweep recall check. DBLP's API is behind a bot challenge for scripted
clients, so the sweep runs on OpenAlex (--source openalex) or, while OpenAlex's no-key daily
budget is exhausted, on Semantic Scholar bulk search (--source s2): all F1' AND F2' records are
fetched and the title-only match and venue match are applied locally (logged deviation).

Lists all 2021-07-07..end papers whose TITLE matches F1' AND F2' and whose venue is one of the
venues; marks which ones the automated search already retrieved.

  venue_sweep.py --run main --against dedup_all.jsonl
Outputs: venue_sweep.jsonl (all sweep records, with `in_search` flag), venue_sweep_stats.json
"""
from __future__ import annotations

import os
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, http_get, norm_title, now_iso  # noqa: E402
from search import F1_R1, F2_R1, START, expand, norm_openalex, norm_s2  # noqa: E402


def _rx(t):
    t = t.strip('"').replace("*", r"\w*")
    return re.compile(r"\b" + t.replace(" ", r"[\s-]+") + r"\b", re.I)


F1RX = [_rx(t) for t in F1_R1]
F2RX = [_rx(t) for t in F2_R1]


def title_match(t: str) -> bool:
    return any(r.search(t) for r in F1RX) and any(r.search(t) for r in F2RX)


def sweep_s2(end):
    q = " + ".join("(" + " | ".join(f) + ")" for f in (F1_R1, F2_R1))
    params = {"query": q, "publicationDateOrYear": f"{START}:{end}",
              "fields": "title,abstract,externalIds,year,venue,publicationVenue,publicationDate,authors,publicationTypes"}
    out, n_title, n_all = [], 0, 0
    st, body = http_get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=params, cache=False, retries=12)
    while st == 200:
        d = json.loads(body)
        for p in d.get("data", []):
            n_all += 1
            if not title_match(p.get("title") or ""):
                continue
            n_title += 1
            names = [p.get("venue") or "", ((p.get("publicationVenue") or {}).get("name") or "")]
            names += (p.get("publicationVenue") or {}).get("alternate_names") or []
            v = next((k for k, pat in VENUES.items() if any(re.search(pat, n, re.I) for n in names if n)), "")
            if v:
                r = norm_s2(p)
                r["sweep_venue"] = v
                out.append(r)
        tok = d.get("token")
        if not tok:
            break
        st, body = http_get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                            params={**params, "token": tok}, cache=False, retries=12)
    return out, n_title, n_all, q

VENUES = {
    "ICSE": r"International Conference on Software Engineering(?!.*Workshop)",
    "FSE": r"Foundations of Software Engineering|Proceedings of the ACM on Software Engineering",
    "ASE": r"Automated Software Engineering",
    "ISSTA": r"Software Testing and Analysis",
    "MSR": r"Mining Software Repositories",
    "ICSME": r"Software Maintenance and Evolution",
    "SANER": r"Software Analysis,? Evolution,? and Reengineering",
    "TOSEM": r"Transactions on Software Engineering and Methodology",
    "TSE": r"IEEE Transactions on Software Engineering",
    "EMSE": r"^Empirical Software Engineering",
    "NeurIPS": r"Neural Information Processing Systems",
    "ICML": r"International Conference on Machine Learning",
    "ICLR": r"Learning Representations",
    "ACL": r"Association for Computational Linguistics",
    "EMNLP": r"Empirical Methods in Natural Language Processing",
    "NAACL": r"North American Chapter",
    "COLM": r"Conference on Language Modeling",
    "CCS": r"Computer and Communications Security",
    "USENIX Security": r"USENIX Security",
    "IEEE S&P": r"Symposium on Security and Privacy",
    "NDSS": r"Network and Distributed System Security",
}


def venue_of(w: dict) -> str:
    names = []
    for loc in [w.get("primary_location") or {}] + (w.get("locations") or []):
        s = loc.get("source") or {}
        names += [s.get("display_name") or "", loc.get("raw_source_name") or ""]
    for v, pat in VENUES.items():
        if any(re.search(pat, n, re.I) for n in names if n):
            return v
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--against", default="dedup_all.jsonl")
    ap.add_argument("--end", default=None)
    ap.add_argument("--source", default="s2", choices=["s2", "openalex"])
    a = ap.parse_args()
    run = REPO / "sok" / "slr" / "runs" / a.run
    end = a.end or a.run
    q = "(" + " OR ".join(expand(F1_R1)) + ") AND (" + " OR ".join(expand(F2_R1)) + ")"
    filt = f"title.search:{q},from_publication_date:{START},to_publication_date:{end}"
    base = {"filter": filt, "per-page": 200, "mailto": os.environ.get("CONTACT_EMAIL", "")}
    cursor, n_title, sweep = "*", 0, []
    n_all = None
    if a.source == "s2":
        sweep, n_title, n_all, filt = sweep_s2(end)
        cursor = None
    while cursor:
        st, body = http_get("https://api.openalex.org/works", params={**base, "cursor": cursor}, cache=False)
        d = json.loads(body)
        for w in d["results"]:
            n_title += 1
            v = venue_of(w)
            if v:
                r = norm_openalex(w)
                r["sweep_venue"] = v
                sweep.append(r)
        cursor = d["meta"].get("next_cursor")
        if not d["results"]:
            break
    have = [json.loads(l) for l in (run / a.against).open()]
    ax = {r["arxiv_id"] for r in have if r.get("arxiv_id")}
    dois = {d for r in have for d in (r.get("dois") or [r.get("doi")]) if d}
    titles = {norm_title(r.get("title", "")) for r in have}
    for r in sweep:
        r["in_search"] = bool((r.get("doi") and r["doi"] in dois) or (r.get("arxiv_id") and r["arxiv_id"] in ax)
                              or norm_title(r["title"]) in titles)
    with open(run / "venue_sweep.jsonl", "w") as fh:
        for r in sweep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in sweep:
        b = by.setdefault(r["sweep_venue"], {"sweep": 0, "in_search": 0})
        b["sweep"] += 1
        b["in_search"] += int(r["in_search"])
    stats = {"timestamp": now_iso(), "source": a.source, "query": filt, "records_scanned": n_all,
             "title_matches_all_venues": n_title,
             "in_target_venues": len(sweep), "already_in_search": sum(r["in_search"] for r in sweep),
             "missed_by_search": sum(not r["in_search"] for r in sweep), "by_venue": by,
             "note": "Missed records are screened with the same prompt; the recall estimate is computed on "
                     "records screened include/uncertain."}
    (run / "venue_sweep_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "query"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
