#!/usr/bin/env python3
"""ACM DL / IEEE Xplore coverage check: for each QGS paper whose
peer-reviewed version is published by ACM (10.1145) or IEEE (10.1109), report whether OpenAlex or
Semantic Scholar retrieved it.  Usage: acm_ieee_check.py --run main"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bibgen import resolve  # noqa: E402
from common import REPO, norm_title, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
run = REPO / "sok" / "slr" / "runs" / a.run
recs = {"openalex": [], "semanticscholar": []}
for f in ["records_S1_openalex.jsonl", "records_S1_s2.jsonl", "records_S1r1_s2.jsonl", "records_S2r1_s2.jsonl"]:
    for r in read_jsonl(run / f):
        recs[r["source"]].append(r)
idx = {s: ({r["arxiv_id"] for r in v if r.get("arxiv_id")}, {r["doi"] for r in v if r.get("doi")},
           {norm_title(r["title"]) for r in v}) for s, v in recs.items()}
rows = []
for q in csv.DictReader(open(REPO / "sok" / "slr" / "qgs.csv")):
    rec = resolve(q["doi_or_arxiv"])
    doi = (rec.get("doi") or "").lower()
    pub = "ACM" if doi.startswith("10.1145") else "IEEE" if doi.startswith("10.1109") else ""
    if not pub:
        continue
    ax = rec.get("arxiv_id", "")
    found = {s: bool((ax and ax in i[0]) or (doi in i[1]) or norm_title(rec["title"]) in i[2] or norm_title(q["title"]) in i[2])
             for s, i in idx.items()}
    rows.append({"qgs": q["doi_or_arxiv"], "title": rec["title"], "doi": doi, "publisher": pub,
                 "found_openalex": found["openalex"], "found_semanticscholar": found["semanticscholar"]})
with open(run / "acm_ieee_coverage.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["publisher"], r["found_openalex"], r["found_semanticscholar"], r["title"][:70])
print(f"{len(rows)} ACM/IEEE QGS papers; found by OpenAlex or S2: {sum(r['found_openalex'] or r['found_semanticscholar'] for r in rows)}")
