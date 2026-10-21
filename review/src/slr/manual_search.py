#!/usr/bin/env python3
"""Manual ACM Digital Library and IEEE Xplore search.

  manual_search.py strings                 # prints the revised strings in each library's syntax
  manual_search.py ingest --run main       # reads the exports, removes duplicates, writes manual_new.jsonl

Run each printed string in the library's advanced search (ACM DL: "Edit Search" > query syntax, Abstract and Title
fields; IEEE Xplore: Command Search, "All Metadata"), filter publication dates 2021-07-07 to 2026-10-14, and export
every result: ACM DL as BibTeX to runs/<run>/manual/acm_<string>.bib, IEEE Xplore as CSV to
runs/<run>/manual/ieee_<string>.csv. New records get ids M00001... and go through the same title/abstract screening,
full-text retrieval and extraction as the database records (screen.py, extract.py with --ids manual_forward_ids.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, norm_title  # noqa: E402
from search import expand, facets  # noqa: E402

STRINGS = ("S1r1", "S2r1")


def acm(string: str) -> str:
    return " AND ".join("(" + " OR ".join(f'Title:("{t}") OR Abstract:("{t}")' for t in expand(f)) + ")"
                        for f in facets(string))


def ieee(string: str) -> str:
    return " AND ".join("(" + " OR ".join(f'"All Metadata":"{t}"' for t in expand(f)) + ")" for f in facets(string))


def _bib(text: str) -> list[dict]:
    out = []
    for entry in re.split(r"\n@", text):
        f = dict(re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", entry))
        if f.get("title"):
            out.append({"title": re.sub(r"[{}]", "", f["title"]), "abstract": f.get("abstract", ""),
                        "year": int(f["year"]) if f.get("year", "").isdigit() else None, "doi": f.get("doi", "").lower(),
                        "venue": f.get("booktitle") or f.get("journal") or "", "publisher": "ACM"})
    return out


def _ieee_csv(path: Path) -> list[dict]:
    return [{"title": r.get("Document Title", ""), "abstract": r.get("Abstract", ""),
             "year": int(r["Publication Year"]) if (r.get("Publication Year") or "").isdigit() else None,
             "doi": (r.get("DOI") or "").lower(), "venue": r.get("Publication Title", ""), "publisher": "IEEE"}
            for r in csv.DictReader(open(path, encoding="utf-8-sig"))]


def ingest(run: str) -> None:
    rd = REPO / "sok" / "slr" / "runs" / run
    seen_doi, seen_title = set(), set()
    for f in ["dedup_all.jsonl", "snowball_it1.jsonl", "other_sources_sections.jsonl", "venue_sweep_new.jsonl"]:
        for l in open(rd / f):
            r = json.loads(l)
            for d in r.get("dois") or ([r["doi"]] if r.get("doi") else []):
                seen_doi.add(d.lower())
            seen_title.add(norm_title(r.get("title", "")))
    found, per_source = [], {}
    for p in sorted((rd / "manual").glob("*")):
        recs = _bib(p.read_text()) if p.suffix == ".bib" else _ieee_csv(p) if p.suffix == ".csv" else []
        per_source[p.name] = len(recs)
        found += recs
    new, keys = [], set()
    for r in found:
        k = r["doi"] or norm_title(r["title"])
        if k in keys or r["doi"] in seen_doi or norm_title(r["title"]) in seen_title:
            continue
        keys.add(k)
        new.append(r | {"rid": f"M{len(new) + 1:05d}", "source": f"manual-{r['publisher'].lower()}", "arxiv_id": ""})
    with open(rd / "manual_new.jsonl", "w") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")
    (rd / "manual_stats.json").write_text(json.dumps({"exports": per_source, "records": len(found), "new": len(new)}, indent=1))
    print(json.dumps({"exports": per_source, "records": len(found), "new_after_dedup": len(new)}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["strings", "ingest"])
    ap.add_argument("--run", default="main")
    a = ap.parse_args()
    if a.cmd == "strings":
        for s in STRINGS:
            print(f"== {s} ACM DL\n{acm(s)}\n\n== {s} IEEE Xplore\n{ieee(s)}\n")
    else:
        ingest(a.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
