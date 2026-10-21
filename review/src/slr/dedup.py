#!/usr/bin/env python3
"""Deduplicate search records by DOI, arXiv ID and normalized title.

  dedup.py --run main --inputs records_S1_openalex.jsonl,records_S1_s2.jsonl,records_S1_arxiv.jsonl --out dedup_S1.jsonl

Each output record carries every source it came from, all IDs, and the longest abstract.
Records outside the date window (before 2021-07-07) are dropped and counted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, norm_title, read_jsonl  # noqa: E402


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def dedup(records: list[dict]) -> list[dict]:
    uf = UF()
    keys = []
    for i, r in enumerate(records):
        ks = [f"rec:{i}"]
        if r.get("doi"):
            ks.append("doi:" + r["doi"].lower())
        if r.get("arxiv_id"):
            ks.append("ax:" + r["arxiv_id"])
        nt = norm_title(r.get("title", ""))
        if len(nt) > 15:
            ks.append("t:" + nt)
        for k in ks[1:]:
            uf.union(ks[0], k)
        keys.append(ks[0])
    groups: dict[str, list[dict]] = {}
    for i, r in enumerate(records):
        groups.setdefault(uf.find(keys[i]), []).append(r)
    out = []
    for g in groups.values():
        best = max(g, key=lambda r: (len(r.get("abstract") or ""), r["source"] == "openalex"))
        m = dict(best)
        m["sources"] = sorted({r["source"] for r in g})
        m["source_ids"] = sorted({f"{r['source']}:{r['source_id']}" for r in g})
        m["dois"] = sorted({r["doi"] for r in g if r.get("doi")})
        m["doi"] = m["dois"][0] if m["dois"] else ""
        axs = sorted({r["arxiv_id"] for r in g if r.get("arxiv_id")})
        m["arxiv_id"] = axs[0] if axs else ""
        m["venues"] = sorted({r["venue"] for r in g if r.get("venue")})
        m["n_merged"] = len(g)
        out.append(m)
    for i, r in enumerate(sorted(out, key=lambda r: (str(r.get("date") or ""), r.get("title") or ""))):
        r["rid"] = f"R{i + 1:05d}"
    return sorted(out, key=lambda r: r["rid"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2021-07-07")
    ap.add_argument("--against", default="", help="existing record files; matching records are dropped")
    ap.add_argument("--prefix", default="R")
    a = ap.parse_args()
    run = REPO / "sok" / "slr" / "runs" / a.run
    recs = []
    for f in a.inputs.split(","):
        recs += read_jsonl(run / f)
    n_in = len(recs)
    early = [r for r in recs if r.get("date") and r["date"] < a.start]
    recs = [r for r in recs if not (r.get("date") and r["date"] < a.start)]
    out = dedup(recs)
    known = 0
    if a.against:
        seen = set()
        for f in a.against.split(","):
            for r in read_jsonl(run / f):
                seen |= {"doi:" + d.lower() for d in (r.get("dois") or [r.get("doi") or ""]) if d}
                seen |= {"ax:" + r["arxiv_id"]} if r.get("arxiv_id") else set()
                seen.add("t:" + norm_title(r.get("title", "")))
        keep = [r for r in out if not ({"doi:" + d.lower() for d in r["dois"]} | ({"ax:" + r["arxiv_id"]} if r["arxiv_id"] else set())
                                       | {"t:" + norm_title(r.get("title", ""))}) & seen]
        known, out = len(out) - len(keep), keep
        for i, r in enumerate(out):
            r["rid"] = f"{a.prefix}{i + 1:05d}"
    with open(run / a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"records_in": n_in, "dropped_before_window": len(early), "already_screened": known, "unique_after_dedup": len(out),
             "duplicates_removed": n_in - len(early) - len(out),
             "by_source_combo": {}}
    for r in out:
        k = "+".join(r["sources"])
        stats["by_source_combo"][k] = stats["by_source_combo"].get(k, 0) + 1
    (run / (a.out.replace(".jsonl", "_stats.json"))).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
