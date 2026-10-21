#!/usr/bin/env python3
"""Fetch the primary-source text of cited papers and check evidence quotes against it
(DOC-001 kickoff §1.D/§1.E and Phase 5 gate 3).

  quotecheck.py text <key|arXiv:ID|doi:...|url:...>   # dump source text to .cache/fulltext and print its path
  quotecheck.py grep <key> "<regex>" [-C 300]          # show matching passages (to copy quotes verbatim)
  quotecheck.py check [--file notes/evidence/X.csv]    # check every row; writes notes/quotecheck_report.csv

Source text: the arXiv PDF (pdftotext) plus the arXiv abstract for arXiv papers; otherwise the
open-access PDF or landing page found through OpenAlex/Crossref; for vendor pages, the page text.

Match levels (after ligature / quote / dash / hyphenation / whitespace normalization):
  exact   the quote occurs verbatim
  near    every quote token occurs in order within a short window (PDF tables and layout breaks)
  fuzzy   partial similarity >= 90 (needs a human look; not accepted as verified)
  fail    not found
For \\rep / \\lit rows the value string must also occur inside the quote.
"""
from __future__ import annotations

import os
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, NOTES, http_get, norm_text, now_iso  # noqa: E402

FT = CACHE / "fulltext"
EVID = NOTES / "evidence"


def resolved(key: str) -> dict | None:
    p = NOTES / "resolved" / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pdf_to_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
        fh.write(data)
        fh.flush()
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", fh.name, "-"], capture_output=True)
        return out.stdout.decode("utf-8", "replace")


def html_to_text(body: str) -> str:
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    body = re.sub(r"(?s)<[^>]+>", " ", body)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"'), ("&nbsp;", " ")):
        body = body.replace(a, b)
    return body


def fetch_text(target: str, refresh: bool = False) -> tuple[Path | None, str]:
    """Return (path, provenance). target is a cite key or an identifier."""
    rec = None
    if ":" not in target or target.startswith(("http",)) is False and not re.match(r"^(arxiv|doi|url):", target, re.I):
        rec = resolved(target)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", target)
    path = FT / f"{name}.txt"
    if path.exists() and not refresh:
        return path, "cache"
    FT.mkdir(parents=True, exist_ok=True)
    arxiv_id = doi = url = None
    if rec:
        arxiv_id, doi, url = rec.get("arxiv_id"), rec.get("doi"), rec.get("url")
        if rec.get("org"):
            arxiv_id = doi = None
    else:
        low = target.lower()
        if low.startswith("arxiv:"):
            arxiv_id = target.split(":", 1)[1]
        elif low.startswith("doi:"):
            doi = target.split(":", 1)[1]
        elif low.startswith("url:"):
            url = target.split(":", 1)[1]
    parts, prov = [], []
    if arxiv_id:
        st, body = http_get("http://export.arxiv.org/api/query", params={"id_list": arxiv_id})
        if st == 200:
            m = re.search(r"<summary>(.*?)</summary>", body, re.S)
            if m:
                parts.append("ABSTRACT: " + m.group(1))
                prov.append("arxiv-abstract")
        st, data = http_get(f"https://arxiv.org/pdf/{arxiv_id}", binary=True, timeout=120)
        if st == 200 and data[:4] == b"%PDF":
            parts.append(pdf_to_text(data))
            prov.append(f"arxiv-pdf:{arxiv_id}")
    if doi and not parts:
        st, body = http_get(f"https://api.openalex.org/works/doi:{doi}", params={"mailto": os.environ.get("CONTACT_EMAIL", "")})
        if st == 200:
            w = json.loads(body)
            ab = w.get("abstract_inverted_index")
            if ab:
                pos = sorted((i, word) for word, idx in ab.items() for i in idx)
                parts.append("ABSTRACT: " + " ".join(wd for _, wd in pos))
                prov.append("openalex-abstract")
            cands = [l.get("pdf_url") for l in w.get("locations", []) if l.get("pdf_url")]
            for cu in cands:
                st2, data = http_get(cu, binary=True, timeout=120)
                if st2 == 200 and data[:4] == b"%PDF":
                    parts.append(pdf_to_text(data))
                    prov.append(f"oa-pdf:{cu}")
                    break
        if len(prov) < 2:
            st, body = http_get(f"https://doi.org/{doi}", timeout=60)
            if st == 200 and len(body) > 2000:
                parts.append(html_to_text(body))
                prov.append(f"doi-landing:{doi}")
    if url and not any(p.startswith(("arxiv-pdf", "oa-pdf")) for p in prov):
        st, body = http_get(url, timeout=60, binary=True)
        if st == 200:
            if body[:4] == b"%PDF":
                parts.append(pdf_to_text(body))
            else:
                parts.append(html_to_text(body.decode("utf-8", "replace")))
            prov.append(f"url:{url}")
    if not parts:
        return None, "unavailable"
    path.write_text("\n\n".join(parts))
    (path.with_suffix(".prov")).write_text(" ".join(prov))
    return path, " ".join(prov)


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", s)


def match_level(quote: str, text: str) -> str:
    q, t = norm_text(quote), norm_text(text)
    if not q:
        return "fail"
    if q in t:
        return "exact"
    qt = tokens(q)
    if not qt:
        return "fail"
    tt = tokens(t)
    window = max(len(qt) * 4, 40)
    first = qt[0]
    for i, tok in enumerate(tt):
        if tok != first:
            continue
        j, k = i, 0
        while j < len(tt) and j - i <= window and k < len(qt):
            if tt[j] == qt[k]:
                k += 1
            j += 1
        if k == len(qt):
            return "near"
    if len(q) > 20 and fuzz.partial_ratio(q, t) >= 90:
        return "fuzzy"
    return "fail"


def value_in_quote(value: str, quote: str) -> bool:
    v = value.replace("\\%", "%").replace("$", "").replace("\\,", "").replace("{", "").replace("}", "").strip()
    v = re.sub(r"\\[a-zA-Z]+", "", v).strip()
    if not v:
        return False
    nq = norm_text(quote)
    nv = norm_text(v)
    if nv in nq:
        return True
    # allow "47.3%" vs "47.3 %" and "0.909 -> 0.878" written with separate tokens
    return all(tok in tokens(nq) for tok in tokens(nv))


def cmd_text(args) -> int:
    path, prov = fetch_text(args.target, refresh=args.refresh)
    print(json.dumps({"path": str(path) if path else None, "provenance": prov,
                      "chars": path.stat().st_size if path else 0}))
    return 0 if path else 2


def cmd_grep(args) -> int:
    path, _ = fetch_text(args.target)
    if not path:
        print("source text unavailable")
        return 2
    text = re.sub(r"\s+", " ", path.read_text())
    n = 0
    for m in re.finditer(args.pattern, text, re.I):
        s, e = max(0, m.start() - args.C), min(len(text), m.end() + args.C)
        print(f"--- @{m.start()}\n{text[s:e]}\n")
        n += 1
        if n >= args.max:
            break
    if n == 0:
        print("no match")
    return 0


def cmd_check(args) -> int:
    files = [Path(args.file)] if args.file else sorted(EVID.glob("*.csv"))
    out_rows = []
    for f in files:
        with open(f) as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                key = r.get("cite_key", "").strip()
                path, prov = fetch_text(key) if key else (None, "no-key")
                lvl = match_level(r.get("evidence_quote", ""), path.read_text()) if path else "source-unavailable"
                kind = r.get("kind", "claim")
                vok = value_in_quote(r.get("claim_or_value", ""), r.get("evidence_quote", "")) if kind in ("rep", "lit") else ""
                out_rows.append({"file": f.name, "row": i + 2, "section": r.get("section"), "cite_key": key,
                                 "kind": kind, "claim_or_value": r.get("claim_or_value", "")[:120],
                                 "match": lvl, "value_in_quote": vok, "provenance": prov, "checked_at": now_iso()})
    out = Path(args.out) if args.out else NOTES / "quotecheck_report.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows else ["file"])
        w.writeheader()
        w.writerows(out_rows)
    from collections import Counter
    c = Counter(r["match"] for r in out_rows)
    bad = [r for r in out_rows if r["match"] not in ("exact", "near") or r["value_in_quote"] is False]
    print(f"rows: {len(out_rows)}; match levels: {dict(c)}; problems: {len(bad)}; report: {out}")
    for r in bad[: args.show]:
        print(f"  {r['file']}:{r['row']} [{r['cite_key']}] match={r['match']} value_in_quote={r['value_in_quote']} :: {r['claim_or_value']}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("text")
    p.add_argument("target")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_text)
    p = sub.add_parser("grep")
    p.add_argument("target")
    p.add_argument("pattern")
    p.add_argument("-C", type=int, default=300)
    p.add_argument("--max", type=int, default=12)
    p.set_defaults(fn=cmd_grep)
    p = sub.add_parser("check")
    p.add_argument("--file")
    p.add_argument("--out")
    p.add_argument("--show", type=int, default=60)
    p.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
