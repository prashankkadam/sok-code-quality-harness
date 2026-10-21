#!/usr/bin/env python3
"""Phase 5 verification gates (DOC-001 kickoff §3, spec Phase 5).

  audit.py citations    # gate 1+2: re-verify EVERY bib entry from scratch (no cache) -> notes/citation_audit.csv
  audit.py evidence     # gate 3: merge evidence fragments -> notes/claims_evidence.csv, notes/numbers_audit.csv;
                        #         check every \\rep / \\lit occurrence has a row with a verbatim quote
  audit.py recheck      # gate 3: re-open sources (fresh download) for ALL \\lit rows and a random 25% of
                        #         \\rep and claim rows -> notes/recheck_log.csv
  audit.py spotcheck    # gate 4: sample 30 prose citations -> notes/spotcheck_sample.csv (judged by hand)
  audit.py isolation    # gate 6
  audit.py leftovers    # gate 7: counts of \\tbd, \\lit, \\unverified per section
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bibtexparser
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from bibgen import arxiv_meta, cited_keys_by_section, crossref  # noqa: E402
from common import CACHE, NOTES, PAPER, REPO, ascii_fold, http_get, norm_title, now_iso  # noqa: E402

SECTIONS = PAPER / "sections"


def load_bib():
    entries = {}
    dupkeys = []
    for f in sorted((PAPER / "bib").glob("*.bib")):
        parser = bibtexparser.bparser.BibTexParser(common_strings=True, ignore_nonstandard_types=False)
        db = bibtexparser.loads(f.read_text(), parser=parser)
        for e in db.entries:
            if e["ID"] in entries:
                dupkeys.append(e["ID"])
            e["_file"] = f.name
            entries[e["ID"]] = e
    return entries, dupkeys


def clean(s):
    return re.sub(r"[{}\\]", "", s or "").strip()


def last_name(a):
    a = clean(a).split(" and ")[0].strip()
    if "," in a:
        a = a.split(",")[0]
    else:
        a = a.split()[-1] if a.split() else a
    return re.sub(r"[^a-z]", "", ascii_fold(a).lower())


def verify_entry(e):
    """Re-query the authoritative source for one entry, without cache."""
    title, year, fa = clean(e.get("title")), e.get("year", ""), last_name(e.get("author", ""))
    res = {"key": e["ID"], "identifier": "", "source_type": "", "matched_title": "", "matched_first_author": "",
           "matched_year": "", "venue": clean(e.get("booktitle") or e.get("journal") or e.get("howpublished") or ""),
           "status": "not-found", "checked_at": now_iso(), "note": ""}
    doi, eprint, url = e.get("doi", ""), e.get("eprint", ""), e.get("url", "")
    if doi:
        res.update(source_type="crossref-doi", identifier=f"doi:{doi}")
        st, body = http_get(f"https://api.crossref.org/works/{doi}", cache=False)
        if st == 200:
            m = json.loads(body)["message"]
            import html as _h
            mt = _h.unescape(_h.unescape(" ".join(m.get("title") or [""])))
            mt = re.sub(r"<[^>]+>", "", mt)
            fam = (m.get("author") or [{}])[0].get("family") or (m.get("author") or [{}])[0].get("name", "")
            yr = ""
            for k in ("published-print", "published", "issued", "published-online"):
                if m.get(k, {}).get("date-parts") and m[k]["date-parts"][0][0]:
                    yr = str(m[k]["date-parts"][0][0])
                    break
            res.update(matched_title=mt, matched_first_author=fam, matched_year=yr)
            ok_t = fuzz.ratio(norm_title(mt), norm_title(title)) >= 90
            famn = re.sub(r"[^a-z]", "", ascii_fold(fam).lower())
            ok_a = (not fam) or famn == fa or (fa and fa in famn)  # compound surnames, e.g. "Muñoz Barón"
            ok_y = yr == year
            res["status"] = "verified" if (ok_t and ok_a and ok_y) else "mismatch"
            res["note"] = f"title_ok={ok_t} author_ok={ok_a} year_ok={ok_y}"
            return res
    if eprint:
        res.update(source_type="arxiv", identifier=f"arXiv:{eprint}")
        m = arxiv_meta(eprint, cache=False)
        for wait in (20, 60):  # the arXiv API returns empty pages under load; retry before falling through
            if m:
                break
            time.sleep(wait)
            m = arxiv_meta(eprint, cache=False)
        if m:
            res.update(matched_title=m["title"], matched_first_author=m["authors"][0] if m["authors"] else "",
                       matched_year=str(m["year"]))
            ok_t = fuzz.ratio(norm_title(m["title"]), norm_title(title)) >= 90
            ok_a = last_name(m["authors"][0]) == fa if m["authors"] else False
            ok_y = str(m["year"]) == year
            if not ok_y and e.get("ENTRYTYPE") == "inproceedings" and "openreview" in (url or "") or \
                    (not ok_y and e.get("ENTRYTYPE") == "inproceedings" and not doi):
                # Venue year (ICLR/ICML/NeurIPS via OpenReview) differs from the arXiv v1 year: check the venue record.
                rec = json.loads((NOTES / "resolved" / f"{e['ID']}.json").read_text()) if (NOTES / "resolved" / f"{e['ID']}.json").exists() else {}
                v = rec.get("openreview_venue", "")
                ok_y = year in v
                res["note"] = f"venue year from OpenReview record: {v}"
                res["source_type"] = "arxiv+openreview"
            res["status"] = "verified" if (ok_t and ok_a and ok_y) else "mismatch"
            res["note"] = (res["note"] + f" title_ok={ok_t} author_ok={ok_a} year_ok={ok_y}").strip()
            return res
    if url:
        res.update(source_type="vendor-page", identifier=f"url:{url}")
        st, raw = http_get(url, cache=False, binary=True)
        if raw[:4] == b"%PDF":
            from quotecheck import pdf_to_text
            body = pdf_to_text(raw)
        else:
            body = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
            import html as _h
            body = _h.unescape(re.sub(r"<[^>]+>", " ", body))
        from common import norm_text
        found = norm_text(title) in norm_text(body)
        res.update(matched_title=title if found else "", matched_first_author=clean(e.get("author")),
                   matched_year=year, status="verified" if (st == 200 and found) else "not-found",
                   note=f"HTTP {st}; exact title on page: {found}")
        return res
    res["note"] = "no doi, eprint or url"
    return res


def cmd_citations(a):
    entries, dupkeys = load_bib()
    by = cited_keys_by_section()
    cited = set().union(*by.values()) if by else set()
    rows = []
    for i, (k, e) in enumerate(sorted(entries.items())):
        r = None
        for attempt in range(4):  # network timeouts are retried, never counted as a verdict
            try:
                r = verify_entry(e)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{k}: {type(exc).__name__}, retry {attempt + 1}", flush=True)
                time.sleep(30 * (attempt + 1))
        if r is None:
            r = {"key": k, "status": "network-error", "checked_at": now_iso(), "note": "verification timed out 4x"}
        r["cited"] = k in cited
        rows.append(r)
        if i % 25 == 0:
            print(f"{i}/{len(entries)}", flush=True)
    # duplicate papers under different keys
    seen = defaultdict(list)
    for k, e in entries.items():
        # Titles identify papers only; web pages (no DOI or eprint) reuse generic titles such as "Changelog",
        # and a vendor release page is a different source from the paper about the same model.
        t = norm_title(clean(e.get("title"))) if (e.get("doi") or e.get("eprint")) else ""
        for ident in (e.get("doi"), e.get("eprint"), t):
            if ident:
                seen[ident].append(k)
    dup_papers = sorted({tuple(sorted(set(v))) for v in seen.values() if len(set(v)) > 1})
    fields = ["key", "source_type", "identifier", "matched_title", "matched_first_author", "matched_year", "venue",
              "status", "checked_at", "cited", "note"]
    with open(NOTES / "citation_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    missing = sorted(cited - set(entries))
    unverified_cited = sorted(r["key"] for r in rows if r["cited"] and r["status"] != "verified")
    orphans = sorted(r["key"] for r in rows if not r["cited"])
    summary = {"entries": len(rows), "status": dict(Counter(r["status"] for r in rows)),
               "cited_keys": len(cited), "cited_missing_from_bib": missing, "cited_not_verified": unverified_cited,
               "orphans": orphans, "duplicate_keys": dupkeys, "duplicate_papers": dup_papers, "at": now_iso()}
    (NOTES / "citation_audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if not (missing or unverified_cited or orphans or dupkeys or dup_papers) else 1


def braced(s, i):
    """Return (content, end) of the {...} group starting at s[i] == '{'."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def macro_values(text, name):
    out = []
    for m in re.finditer(r"\\" + name + r"\{", text):
        v, end = braced(text, m.end() - 1)
        key = ""
        if name == "lit" and end < len(text) and text[end] == "{":
            key, _ = braced(text, end)
        out.append((v, key))
    return out


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def cmd_evidence(a):
    frag = []
    for f in sorted((NOTES / "evidence").glob("*.csv")):
        frag += list(csv.DictReader(open(f)))
    with open(NOTES / "claims_evidence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "cite_key", "claim_or_value", "evidence_quote", "location", "checked_at"])
        w.writeheader()
        for r in frag:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    with open(NOTES / "numbers_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "kind", "value", "cite_key", "source_location", "original_setting"])
        w.writeheader()
        for r in frag:
            if r.get("kind") in ("rep", "lit"):
                w.writerow({"section": r["section"], "kind": r["kind"], "value": r["claim_or_value"], "cite_key": r["cite_key"],
                            "source_location": r["location"], "original_setting": r.get("setting", "")})
    problems = []
    per = {}
    for p in sorted(SECTIONS.glob("*.tex")):
        t = strip_comments(p.read_text())
        reps = [v for v, _ in macro_values(t, "rep")]
        lits = macro_values(t, "lit")
        rows = [r for r in frag if r["section"] == p.stem]
        have_rep = Counter(r["claim_or_value"].strip() for r in rows if r.get("kind") == "rep")
        have_lit = Counter((r["claim_or_value"].strip(), r["cite_key"]) for r in rows if r.get("kind") == "lit")
        need_rep, need_lit = Counter(v.strip() for v in reps), Counter((v.strip(), k.strip()) for v, k in lits)
        miss_rep = need_rep - have_rep
        miss_lit = need_lit - have_lit
        per[p.stem] = {"rep": sum(need_rep.values()), "lit": sum(need_lit.values()),
                       "rep_missing_rows": sum(miss_rep.values()), "lit_missing_rows": sum(miss_lit.values()),
                       "evidence_rows": len(rows), "empty_quotes": sum(1 for r in rows if not r.get("evidence_quote", "").strip())}
        for v, n in miss_rep.items():
            problems.append(f"{p.stem}: \\rep{{{v}}} x{n} has no evidence row")
        for (v, k), n in miss_lit.items():
            problems.append(f"{p.stem}: \\lit{{{v}}}{{{k}}} x{n} has no evidence row")
    out = {"per_section": per, "problems": problems, "rows_total": len(frag), "at": now_iso()}
    (NOTES / "evidence_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_section"}, indent=1)[:4000])
    for s, d in per.items():
        if d["rep"] or d["lit"] or d["evidence_rows"]:
            print(s, d)
    return 1 if problems else 0


def cmd_recheck(a):
    """Re-open sources: fresh download for every \\lit row and a seeded random 25% of rep and claim rows."""
    from quotecheck import fetch_text, match_level, value_in_quote
    frag = []
    for f in sorted((NOTES / "evidence").glob("*.csv")):
        frag += list(csv.DictReader(open(f)))
    rng = random.Random(20261014)
    lit = [r for r in frag if r.get("kind") == "lit"]
    others = [r for r in frag if r.get("kind") in ("rep", "claim")]
    sample = rng.sample(others, k=max(1, round(0.25 * len(others))))
    refreshed = set()
    out = []
    for r in lit + sample:
        k = r["cite_key"]
        path, prov = fetch_text(k, refresh=k not in refreshed)
        refreshed.add(k)
        lvl = match_level(r["evidence_quote"], path.read_text()) if path else "source-unavailable"
        vok = value_in_quote(r["claim_or_value"], r["evidence_quote"]) if r.get("kind") in ("rep", "lit") else ""
        out.append({"section": r["section"], "cite_key": k, "kind": r.get("kind"), "value_or_claim": r["claim_or_value"][:150],
                    "match": lvl, "value_in_quote": vok, "provenance": prov, "outcome": "ok" if lvl in ("exact", "near") and vok is not False else "DISCREPANCY",
                    "rechecked_at": now_iso()})
    with open(NOTES / "recheck_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    c = Counter(o["outcome"] for o in out)
    print(f"rechecked {len(out)} rows ({len(lit)} lit, {len(sample)} sampled of {len(others)} rep/claim): {dict(c)}")
    for o in out:
        if o["outcome"] != "ok":
            print("  ", o)
    return 0 if c.get("DISCREPANCY", 0) == 0 else 1


def cmd_spotcheck(a):
    rng = random.Random(1014)
    items = []
    for p in sorted(SECTIONS.glob("*.tex")):
        t = strip_comments(p.read_text())
        # prose only: drop table environments
        t = re.sub(r"\\begin\{(table|longtable|tabularx)\}.*?\\end\{\1\}", " ", t, flags=re.S)
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
            s = max(t.rfind(". ", 0, m.start()), t.rfind("\n\n", 0, m.start()))
            e = t.find(". ", m.end())
            sent = re.sub(r"\s+", " ", t[s + 1:(e if e > 0 else m.end() + 200) + 1]).strip()
            for k in m.group(1).split(","):
                items.append({"section": p.stem, "cite_key": k.strip(), "sentence": sent[:600]})
    sample = rng.sample(items, k=min(30, len(items)))
    ev = defaultdict(list)
    for f in sorted((NOTES / "evidence").glob("*.csv")):
        for r in csv.DictReader(open(f)):
            ev[(r["section"], r["cite_key"])].append(r["evidence_quote"])
    with open(NOTES / "spotcheck_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "section", "cite_key", "sentence", "evidence_quotes", "supports", "note"])
        w.writeheader()
        for i, s in enumerate(sample):
            w.writerow({"n": i + 1, **s, "evidence_quotes": " || ".join(q[:300] for q in ev.get((s["section"], s["cite_key"]), [])[:3]),
                        "supports": "", "note": ""})
    print(f"sampled {len(sample)} of {len(items)} prose citations -> {NOTES / 'spotcheck_sample.csv'}")
    return 0


def cmd_isolation(a):
    g = subprocess.run(["grep", "-rn", "benchmark/", str(PAPER)], capture_output=True, text=True)
    hits = [l for l in g.stdout.splitlines() if "/build/" not in l and ".git" not in l]
    log = subprocess.run(["git", "-C", str(REPO), "log", "--format=%H %s", "5f318a9..HEAD", "--name-only"],
                         capture_output=True, text=True).stdout
    touched = [l for l in log.splitlines() if l.startswith("benchmark/")]
    print(json.dumps({"grep_benchmark_in_sok_paper": hits, "commits_after_5f318a9_touching_benchmark": touched}, indent=1))
    return 0 if not hits and not touched else 1


def cmd_leftovers(a):
    out = {}
    for p in sorted(SECTIONS.glob("*.tex")):
        t = strip_comments(p.read_text())
        out[p.stem] = {"tbd": len(re.findall(r"\\tbd\{", t)), "cnt(provisional tbd)": len(re.findall(r"\\cnt\{", t)),
                       "lit": len(re.findall(r"\\lit\{", t)), "unverified": len(re.findall(r"\\unverified\{", t)),
                       "section_not_drafted": "section not yet drafted" in t}
    print(json.dumps(out, indent=1))
    (NOTES / "leftovers.json").write_text(json.dumps(out, indent=1))
    return 1 if any(v["section_not_drafted"] for v in out.values()) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("citations", cmd_citations), ("evidence", cmd_evidence), ("recheck", cmd_recheck),
                 ("spotcheck", cmd_spotcheck), ("isolation", cmd_isolation), ("leftovers", cmd_leftovers)):
        p = sub.add_parser(n)
        p.set_defaults(fn=f)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
