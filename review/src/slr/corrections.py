#!/usr/bin/env python3
"""Consistency corrections to extraction.csv found after extraction, applied in place and logged in corrections.csv.

  corrections.py --run main [--audit scope_audit.csv]

1. Highlighted techniques: when a selected technique's study has no targeted dimension in extraction.csv, the
   target verified in the selection review (top/selected.csv, quote-checked) is written to dims_targeted.
2. Scope audit: studies that fail an inclusion or exclusion criterion move to side_lists.csv with the criterion and
   reason; a wrong year is corrected.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SLR = Path(__file__).resolve().parents[2] / "sok" / "slr"
DIM = {"R/M": "readability-maintainability", "EFF": "efficiency", "SEC": "security"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--audit", default="")
    a = ap.parse_args()
    ext = SLR / "extraction.csv"
    rows = list(csv.DictReader(open(ext)))
    fields = list(rows[0].keys())
    log = []
    sel = {r["tid"]: r for r in csv.DictReader(open(SLR / "top" / "selected.csv"))}
    seen: dict[str, int] = {}
    for r in rows:
        k = seen.get(r["rid"], 0)
        seen[r["rid"]] = k + 1
        s = sel.get(f"{r['rid']}#{k}")
        if s and not r["dims_targeted"].strip():
            dims = ";".join(DIM[d.strip()] for d in s["target_dim"].replace(",", ";").split(";") if d.strip() in DIM)
            if dims:
                r["dims_targeted"] = dims
                log.append({"rid": r["rid"], "action": "fill_target", "detail": dims, "reason": "selection-review target"})
    side_p = SLR / "runs" / a.run / "side_lists.csv"
    side = list(csv.DictReader(open(side_p)))
    sfields = list(side[0].keys())
    if a.audit:
        audit = list(csv.DictReader(open(a.audit)))
        drop = {x["rid"]: x for x in audit if x["action"] in ("exclude", "context")}
        years = {x["rid"]: x for x in audit if x["action"] == "fix_year"}
        for r in rows:
            if r["rid"] in years:
                y = "".join(ch for ch in years[r["rid"]]["reason"] if ch.isdigit())[:4]
                if y:
                    log.append({"rid": r["rid"], "action": "fix_year", "detail": f"{r['year']}->{y}", "reason": years[r["rid"]]["reason"]})
                    r["year"] = y
        for rid, x in drop.items():
            r0 = next((r for r in rows if r["rid"] == rid), None)
            if r0 is None:
                continue
            side.append({"rid": rid, "title": r0["title"], "year": r0["year"], "doi": r0["doi"], "arxiv_id": r0["arxiv_id"],
                         "side_list": "context" if x["action"] == "context" else "none",
                         "exclusion_criterion": x["criterion"] or "IC1", "reason": x["reason"], "confidence": "high",
                         "fulltext_source": r0["fulltext_source"], "identified_via": r0["identified_via"]})
            log.append({"rid": rid, "action": x["action"], "detail": x["criterion"], "reason": x["reason"]})
        rows = [r for r in rows if r["rid"] not in drop]
        with open(side_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
            w.writeheader()
            w.writerows(side)
    with open(ext, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lp = SLR / "corrections.csv"
    old = list(csv.DictReader(open(lp))) if lp.exists() else []
    with open(lp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rid", "action", "detail", "reason"])
        w.writeheader()
        w.writerows(old + log)
    print(f"{len(log)} corrections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
