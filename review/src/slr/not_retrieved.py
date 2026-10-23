#!/usr/bin/env python3
"""Characterize the reports sought but not retrieved (no full text), for the threats to validity.

  not_retrieved.py --run main   -> sok/slr/runs/<run>/not_retrieved.csv

One row per report: year, source type, publisher, how it was found, open-access status and whether it is another
version of an included study. Metadata come from the search records; publisher, genre and open-access status from the
Unpaywall and Crossref responses cached during full-text retrieval (extract.py), so no network call is made. Reports
whose DOI has no cached response are `unknown`. synth.py turns the CSV into generated/tab_notretrieved.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE  # noqa: E402
from synth import SLR, retrieved  # noqa: E402

PREFIX = {"10.1109": "IEEE", "10.1145": "ACM", "10.1007": "Springer", "10.1016": "Elsevier", "10.2139": "SSRN",
          "10.5281": "Zenodo", "10.3390": "MDPI", "10.1002": "Wiley", "10.48550": "arXiv", "10.1080": "Taylor & Francis"}
PUBLISHER = {"Institute of Electrical and Electronics Engineers (IEEE)": "IEEE", "IEEE": "IEEE",
             "Association for Computing Machinery (ACM)": "ACM", "ACM": "ACM",
             "Springer Science and Business Media LLC": "Springer", "Springer Nature Switzerland": "Springer",
             "Springer Nature Singapore": "Springer", "Springer Berlin Heidelberg": "Springer",
             "Springer International Publishing": "Springer", "Springer Nature": "Springer",
             "Elsevier BV": "Elsevier", "Elsevier": "Elsevier", "Wiley": "Wiley", "MDPI AG": "MDPI",
             "Zenodo": "Zenodo", "Informa UK Limited": "Taylor & Francis"}
PREPRINT = {"SSRN", "Zenodo", "arXiv", "Research Square", "Preprints.org", "TechRxiv", "Authorea", "OSF"}


def run_dir(run: str) -> Path:
    return SLR / "runs" / run


def all_records(run: str) -> dict:
    """Every identified record by rid (same sources as extract.all_records, without its PDF dependencies)."""
    recs = {}
    d = run_dir(run)
    for f in sorted(d.glob("dedup_all*.jsonl")) + sorted(d.glob("snowball_*.jsonl")) + sorted(d.glob("other_sources*.jsonl")) + \
            sorted(d.glob("venue_sweep_new*.jsonl")) + sorted(d.glob("manual_new*.jsonl")):
        if not f.name.endswith("_raw.jsonl"):
            for line in f.open():
                r = json.loads(line)
                if "rid" in r:
                    recs[r["rid"]] = r
    return recs


def cached_api() -> tuple[dict, dict]:
    """Unpaywall and Crossref responses by lower-case DOI, from the on-disk HTTP cache."""
    up, cr = {}, {}
    for meta in (CACHE / "http").glob("*/*.meta"):
        m = json.loads(meta.read_text())
        if m.get("status") != 200:
            continue
        url = m["url"]
        for host, store, key in (("api.unpaywall.org/v2/", up, "/v2/"), ("api.crossref.org/works/", cr, "/works/")):
            if host in url:
                try:
                    body = json.loads(meta.with_suffix("").read_text())
                except (OSError, ValueError):
                    continue
                store[url.split(key, 1)[1].split("?")[0].lower()] = body.get("message", body) if store is cr else body
    return up, cr


def _list(x):
    if isinstance(x, str):
        try:
            return json.loads(x.replace("'", '"'))
        except ValueError:
            return []
    return x or []


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^\W*\(?\d{4}\)?\W*", "", (t or "").lower()))


def same_paper(t: str, pool: dict, self_rid: str) -> str:
    n = norm(t)
    if len(n) < 25:
        return ""
    for rid, tt in pool.items():
        m = norm(tt)
        if rid != self_rid and len(m) >= 25 and (n == m or (len(n) > 40 and (n in m or m in n))):
            return rid
    return ""


def classify(r: dict, u: dict | None, c: dict | None, dois: list[str]) -> tuple[str, str]:
    pub = PUBLISHER.get((u or {}).get("publisher") or "", (u or {}).get("publisher")) or \
        PUBLISHER.get((c or {}).get("publisher") or "", (c or {}).get("publisher")) or \
        (PREFIX.get(dois[0].split("/")[0]) if dois else None)
    ven, typ = (r.get("venue") or "").lower(), (r.get("type") or "").lower()
    genre = (u or {}).get("genre") or (c or {}).get("type") or ""
    if "ssrn" in ven:
        pub = "SSRN"
    if pub in PREPRINT or "zenodo" in ven or genre == "posted-content" or typ == "preprint" or "arxiv" in ven:
        kind = "Preprint or repository"
    elif genre == "dissertation" or "dissertation" in typ or "thesis" in ven or "diva" in ven:
        kind = "Thesis"
    elif genre == "proceedings-article" or "conference" in typ or any(
            w in ven for w in ("proceedings", "conference", "symposium", "workshop", "lecture notes")):
        kind = "Conference or workshop paper"
    elif genre == "journal-article" or typ in ("journalarticle", "article") or "journal" in ven or "transactions" in ven:
        kind = "Journal article"
    else:
        kind = "Other or unknown"
    return kind, pub or ("no DOI" if not dois else "other")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    rd = run_dir(a.run)
    recs = all_records(a.run)
    dec = list(csv.DictReader(open(rd / "decisions_ta.csv")))
    ok = retrieved(a.run)
    missing = sorted({d["rid"] for d in dec if d["forward_to_fulltext"] == "True"} - ok)
    included = {r["rid"]: r["title"] for r in csv.DictReader(open(SLR / "extraction.csv")) if r["rid"] in ok}
    up, cr = cached_api()
    rows = []
    for rid in missing:
        r = recs.get(rid, {})
        dois = [d.lower() for d in dict.fromkeys(([r["doi"]] if r.get("doi") else []) + _list(r.get("dois"))) if d]
        u = next((up[d] for d in dois if d in up), None)
        c = next((cr[d] for d in dois if d in cr), None)
        kind, pub = classify(r, u, c, dois)
        if rid[0] == "S":
            found = "snowballing"
        elif rid[0] == "V":
            found = "venue sweep"
        else:
            src = {"openalex" if "openalex" in s else "semanticscholar" if "semantic" in s else "arxiv"
                   for s in _list(r.get("sources"))}
            found = "openalex+semanticscholar" if {"openalex", "semanticscholar"} <= src else \
                "openalex" if "openalex" in src else "semanticscholar"
        if u is None:
            oa = "unknown"
        elif not u.get("is_oa"):
            oa = "closed"
        else:
            locs = [u.get("best_oa_location")] + (u.get("oa_locations") or [])
            oa = "open-pdf" if any((l or {}).get("url_for_pdf") for l in locs) else "open-landing"
        rows.append({"rid": rid, "title": r.get("title", ""), "year": r.get("year", ""), "doi": dois[0] if dois else "",
                     "source_type": kind, "publisher": pub, "found_by": found, "oa": oa,
                     "oa_status": (u or {}).get("oa_status") or "", "same_as_included": same_paper(r.get("title"), included, rid)})
    out = rd / "not_retrieved.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
