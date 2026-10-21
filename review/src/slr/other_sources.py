#!/usr/bin/env python3
"""Collect primary-study candidates found by section-level searches (sok/paper/notes/candidates/*.csv)
as PRISMA 'other methods' records, drop those already in the database results, and write
normalized records (rid O#####) for screening with the same prompt.
  other_sources.py --run main --out other_sources_sections.jsonl"""
import argparse, csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bibgen import arxiv_meta, from_crossref  # noqa: E402
from common import NOTES, REPO, http_get, norm_title, read_jsonl  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--prefix", default="O"); ap.add_argument("--exclude", default="")
a = ap.parse_args()
run = REPO / "sok" / "slr" / "runs" / a.run
have = read_jsonl(run / "dedup_all.jsonl") + read_jsonl(run / "venue_sweep_new.jsonl") + [r for f in a.exclude.split(",") if f for r in read_jsonl(run / f)]
prev_other = run / a.out
ax = {r["arxiv_id"] for r in have if r.get("arxiv_id")}
dois = {d for r in have for d in (r.get("dois") or [r.get("doi")]) if d}
titles = {norm_title(r.get("title", "")) for r in have}
seen, out, dup = set(), [], 0
for f in sorted((NOTES / "candidates").glob("*.csv")):
    for c in csv.DictReader(open(f)):
        ident = (c.get("identifier") or "").strip()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        low = ident.lower()
        rec = {"source": "other-section-search", "source_id": ident, "candidate_section": f.stem,
               "candidate_note": c.get("note", ""), "candidate_family": c.get("family", "")}
        if low.startswith("arxiv:"):
            m = arxiv_meta(ident.split(":", 1)[1])
            if not m:
                continue
            st, body = http_get("http://export.arxiv.org/api/query", params={"id_list": m["arxiv_id"]})
            abst = re.sub(r"\s+", " ", re.search(r"<summary>(.*?)</summary>", body, re.S).group(1)) if st == 200 else ""
            rec.update(title=m["title"], abstract=abst, year=m["year"], date=m["published"], doi="",
                       arxiv_id=m["arxiv_id"], venue="arXiv", authors=m["authors"][:6])
        elif low.startswith("doi:"):
            d = ident.split(":", 1)[1].lower()
            cr = from_crossref(d)
            if not cr:
                continue
            st, body = http_get(f"https://api.crossref.org/works/{d}")
            abst = re.sub(r"<[^>]+>", " ", json.loads(body)["message"].get("abstract", "")) if st == 200 else ""
            rec.update(title=cr["title"], abstract=abst.strip(), year=cr["year"], date=f"{cr['year']}-01-01", doi=d,
                       arxiv_id="", venue=cr.get("journal") or cr.get("booktitle") or "", authors=cr["authors"][:6])
        else:
            continue
        if (rec["arxiv_id"] and rec["arxiv_id"] in ax) or (rec["doi"] and rec["doi"] in dois) or norm_title(rec["title"]) in titles:
            dup += 1
            continue
        out.append(rec)
with open(run / a.out, "w") as fh:
    for i, r in enumerate(out):
        r["rid"] = f"{a.prefix}{i + 1:05d}"; r["sources"] = ["other-section-search"]; r["dois"] = [r["doi"]] if r["doi"] else []
        r["venues"] = [r.get("venue", "")]
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"candidate identifiers: {len(seen)}; already in database results: {dup}; new other-method records: {len(out)}")
