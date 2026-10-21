#!/usr/bin/env python3
"""Run search string S1 on OpenAlex, Semantic Scholar (bulk) and arXiv.

  search.py --run main [--string S1|S2] [--count-only] [--sources openalex,s2,arxiv]

Writes, under sok/slr/runs/<run>/:
  queries_<string>.json       exact per-source query strings, timestamps, raw hit counts
  raw/<source>_<string>.jsonl raw API records (one per line, never overwritten once complete)
  records_<string>_<source>.jsonl normalized records
Dates: 2021-07-07 to the search date.
"""
from __future__ import annotations

import os
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, http_get, now_iso  # noqa: E402

START = "2021-07-07"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# facets, written once; each source gets its own syntax below.
F1 = ['"large language model*"', "LLM", "LLMs", '"language model*"', '"code model*"', '"foundation model*"',
      '"code LLM*"', "GPT*", "Codex", "ChatGPT", "Copilot"]
F2 = ['"code generation"', '"program synthesis"', '"generated code"', '"code completion"', '"code synthesis"',
      '"generating code"', '"AI-generated code"', '"LLM-generated code"', '"machine-generated code"',
      '"coding assistant*"']
F3 = ['"code quality"', "readab*", "maintainab*", '"code smell*"', '"technical debt"', "cyclomatic",
      '"cognitive complexity"', "idiomatic", '"coding style"', '"coding convention*"', '"style guide*"',
      '"non-functional"', "nonfunctional", '"code efficiency"', '"efficient code"', '"execution time"',
      '"runtime efficiency"', '"memory usage"', '"memory efficiency"', '"energy efficien*"',
      '"performance-improving"', '"code optimization"', '"secure code"', '"code security"', '"insecure code"',
      "vulnerab*", "CWE", '"security weakness*"']
F4 = ['"self-refin*"', '"self-repair"', '"self-debug*"', "reflexion", '"self-critique"', '"multi-agent"',
      '"static analy*"', "linter", '"compiler feedback"', '"retrieval-augmented"', '"constrained decoding"',
      "rerank*", '"best-of-n"', '"preference optimization"', "DPO", '"reinforcement learning"', '"fine-tun*"']

# Explicit expansions for sources without wildcard support (OpenAlex). Unquoted words are stemmed by OpenAlex.
EXPAND = {
    '"large language model*"': ['"large language model"', '"large language models"'],
    '"language model*"': ['"language model"', '"language models"'],
    '"code model*"': ['"code model"', '"code models"'],
    '"foundation model*"': ['"foundation model"', '"foundation models"'],
    '"code LLM*"': ['"code LLM"', '"code LLMs"'],
    "GPT*": ["GPT", "GPT-3", "GPT-3.5", "GPT-4", "GPT-4o", "GPT-5"],
    '"coding assistant*"': ['"coding assistant"', '"coding assistants"'],
    "readab*": ["readability", "readable"],
    "maintainab*": ["maintainability", "maintainable"],
    '"code smell*"': ['"code smell"', '"code smells"'],
    '"coding convention*"': ['"coding convention"', '"coding conventions"'],
    '"style guide*"': ['"style guide"', '"style guides"'],
    '"energy efficien*"': ['"energy efficient"', '"energy efficiency"'],
    "vulnerab*": ["vulnerability", "vulnerabilities", "vulnerable"],
    '"security weakness*"': ['"security weakness"', '"security weaknesses"'],
    '"self-refin*"': ['"self-refine"', '"self-refinement"', '"self-refining"'],
    '"self-debug*"': ['"self-debug"', '"self-debugging"'],
    '"static analy*"': ['"static analysis"', '"static analyzer"', '"static analyser"', '"static analyzers"'],
    "rerank*": ["rerank", "reranking", "reranker", "re-ranking"],
    '"code LM*"': ['"code LM"', '"code LMs"'],
    '"code edit*"': ['"code edit"', '"code edits"', '"code editing"'],
    '"programming task*"': ['"programming task"', '"programming tasks"'],
    '"coding task*"': ['"coding task"', '"coding tasks"'],
    '"code-related task*"': ['"code-related task"', '"code-related tasks"'],
    '"coding preference*"': ['"coding preference"', '"coding preferences"'],
    '"code revision*"': ['"code revision"', '"code revisions"'],
    "codebase*": ["codebase", "codebases"],
    '"code snippet*"': ['"code snippet"', '"code snippets"'],
    '"prompt technique*"': ['"prompt technique"', '"prompt techniques"'],
    '"prompting technique*"': ['"prompting technique"', '"prompting techniques"'],
    '"fine-tun*"': ['"fine-tuning"', '"fine-tuned"', '"fine-tune"'],
}


# Calibration iteration 1 (logged in sok/slr/runs/<run>/calibration_log.md):
# the task facet missed papers that edit, repair or optimize code without saying "code generation";
# the model facet missed "code LMs"; the technique facet had no prompting terms.
F1_R1 = F1 + ['"code LM*"', "LMs"]
F2_R1 = F2 + ['"source code"', '"code edit*"', '"code repair"', '"vulnerability repair"', '"code refinement"',
              '"program optimization"', '"code optimization"', '"programming task*"', '"coding task*"',
              '"code-related task*"', '"coding preference*"', '"code revision*"', "codebase*", '"code efficiency"',
              '"Python code"', '"code snippet*"']
F4_R1 = F4 + ['"prompt engineering"', '"prompt technique*"', '"prompting technique*"', '"chain-of-thought"',
              '"few-shot"', '"in-context learning"']


def facets(string: str):
    return {"S1": (F1, F2, F3), "S2": (F1, F2, F4),
            "S1r1": (F1_R1, F2_R1, F3), "S2r1": (F1_R1, F2_R1, F4_R1)}[string]


def expand(terms):
    out = []
    for t in terms:
        out.extend(EXPAND.get(t, [t]))
    return out


# ------------------------------------------------------------------ OpenAlex
def openalex_query(string: str) -> str:
    return " AND ".join("(" + " OR ".join(expand(f)) + ")" for f in facets(string))


def run_openalex(string, run_dir, end, count_only):
    q = openalex_query(string)
    filt = f"title_and_abstract.search:{q},from_publication_date:{START},to_publication_date:{end}"
    base = {"filter": filt, "per-page": 200, "mailto": os.environ.get("CONTACT_EMAIL", "")}
    st, body = http_get("https://api.openalex.org/works", params={**base, "per-page": 1}, cache=False)
    count = json.loads(body)["meta"]["count"]
    meta = {"source": "openalex", "endpoint": "https://api.openalex.org/works", "filter": filt,
            "query_string": q, "timestamp": now_iso(), "raw_hits": count}
    if count_only:
        return meta, []
    cursor, recs = "*", []
    raw = (run_dir / "raw" / f"openalex_{string}.jsonl").open("w")
    while cursor:
        st, body = http_get("https://api.openalex.org/works", params={**base, "cursor": cursor}, cache=False)
        d = json.loads(body)
        for w in d["results"]:
            raw.write(json.dumps(w) + "\n")
            recs.append(norm_openalex(w))
        cursor = d["meta"].get("next_cursor")
        if not d["results"]:
            break
    raw.close()
    meta["retrieved"] = len(recs)
    return meta, recs


def oa_abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = sorted((i, word) for word, idx in inv.items() for i in idx)
    return " ".join(wd for _, wd in pos)


def norm_openalex(w):
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
    arxiv = ""
    if doi.startswith("10.48550/arxiv."):
        arxiv = doi.split("arxiv.", 1)[1]
        doi = ""
    for loc in w.get("locations", []):
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", loc.get("landing_page_url") or "")
        if m:
            arxiv = arxiv or m.group(1)
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {"source": "openalex", "source_id": w["id"], "title": w.get("title") or "", "abstract": oa_abstract(w),
            "year": w.get("publication_year"), "date": w.get("publication_date"), "doi": doi, "arxiv_id": arxiv,
            "venue": src.get("display_name") or "", "venue_type": src.get("type") or "", "type": w.get("type"),
            "language": w.get("language"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:6] if a.get("author")]}


# ------------------------------------------------------------------ Semantic Scholar bulk
def s2_term(t: str) -> str:
    return t  # S2 bulk supports quotes and trailing * prefix wildcards


def s2_query(string: str) -> str:
    return " + ".join("(" + " | ".join(s2_term(t) for t in f) + ")" for f in facets(string))


def run_s2(string, run_dir, end, count_only):
    q = s2_query(string)
    params = {"query": q, "publicationDateOrYear": f"{START}:{end}",
              "fields": "title,abstract,externalIds,year,venue,publicationDate,authors,publicationTypes"}
    meta = {"source": "semanticscholar", "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
            "query_string": q, "params": {k: v for k, v in params.items() if k != "fields"},
            "timestamp": now_iso()}
    st, body = http_get("https://api.semanticscholar.org/graph/v1/paper/search/bulk", params=params, cache=False,
                        retries=12)
    if st != 200:
        meta["error"] = f"HTTP {st}"
        return meta, []
    d = json.loads(body)
    meta["raw_hits"] = d.get("total")
    if count_only:
        return meta, []
    recs = []
    raw = (run_dir / "raw" / f"s2_{string}.jsonl").open("w")
    while True:
        for p in d.get("data", []):
            raw.write(json.dumps(p) + "\n")
            recs.append(norm_s2(p))
        tok = d.get("token")
        if not tok:
            break
        st, body = http_get("https://api.semanticscholar.org/graph/v1/paper/search/bulk",
                            params={**params, "token": tok}, cache=False, retries=12)
        if st != 200:
            meta["error"] = f"pagination stopped at HTTP {st}"
            break
        d = json.loads(body)
    raw.close()
    meta["retrieved"] = len(recs)
    return meta, recs


def norm_s2(p):
    ext = p.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower()
    arxiv = ext.get("ArXiv") or ""
    if doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or doi.split("arxiv.", 1)[1]
        doi = ""
    return {"source": "semanticscholar", "source_id": p.get("paperId"), "title": p.get("title") or "",
            "abstract": p.get("abstract") or "", "year": p.get("year"), "date": p.get("publicationDate"),
            "doi": doi, "arxiv_id": arxiv, "venue": p.get("venue") or "", "venue_type": "",
            "type": ";".join(p.get("publicationTypes") or []), "language": "",
            "authors": [a.get("name") for a in (p.get("authors") or [])[:6]]}


# ------------------------------------------------------------------ arXiv
def arxiv_term(t: str) -> str:
    t = t.replace("*", "")
    return f"(ti:{t} OR abs:{t})"


def arxiv_queries(string: str, end: str) -> list[str]:
    """arXiv's API rejects very long queries, so the last facet is split into chunks; the union of
    chunk results equals the full boolean query."""
    f1, f2, f3 = facets(string)
    d = f"submittedDate:[{START.replace('-', '')}0000 TO {end.replace('-', '')}2359]"
    head = "(" + " OR ".join(arxiv_term(t) for t in f1) + ") AND (" + " OR ".join(arxiv_term(t) for t in f2) + ")"
    chunks = [f3[i:i + 7] for i in range(0, len(f3), 7)]
    return [f"{head} AND (" + " OR ".join(arxiv_term(t) for t in c) + f") AND {d}" for c in chunks]


def run_arxiv(string, run_dir, end, count_only):
    qs = arxiv_queries(string, end)
    meta = {"source": "arxiv", "endpoint": "https://export.arxiv.org/api/query", "query_strings": qs,
            "note": "trailing * removed from terms (arXiv applies stemming); last facet split into chunks",
            "timestamp": now_iso(), "raw_hits_per_chunk": []}
    recs = {}
    raw = None if count_only else (run_dir / "raw" / f"arxiv_{string}.jsonl").open("w")
    for q in qs:
        start, total = 0, None
        while True:
            st, body = http_get("https://export.arxiv.org/api/query",
                                params={"search_query": q, "start": start, "max_results": 1 if count_only else 1000,
                                        "sortBy": "submittedDate", "sortOrder": "ascending"}, cache=False, timeout=180)
            if st != 200:
                meta.setdefault("errors", []).append(f"HTTP {st} at start={start}")
                break
            root = ET.fromstring(body)
            if total is None:
                total = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
                meta["raw_hits_per_chunk"].append(total)
                if count_only:
                    break
            ents = root.findall(f"{ATOM}entry")
            if not ents:
                if start < total:  # arXiv sometimes returns an empty page; retry once
                    st, body = http_get("https://export.arxiv.org/api/query",
                                        params={"search_query": q, "start": start, "max_results": 1000,
                                                "sortBy": "submittedDate", "sortOrder": "ascending"},
                                        cache=False, timeout=180)
                    ents = ET.fromstring(body).findall(f"{ATOM}entry") if st == 200 else []
                    if not ents:
                        meta.setdefault("errors", []).append(f"empty page at start={start} of {total}")
                        break
                else:
                    break
            for e in ents:
                r = norm_arxiv(e)
                raw.write(json.dumps(r) + "\n")
                recs[r["arxiv_id"]] = r
            start += len(ents)
            if start >= total:
                break
    if raw:
        raw.close()
    meta["raw_hits"] = sum(meta["raw_hits_per_chunk"])
    meta["retrieved_unique"] = len(recs)
    return meta, list(recs.values())


def norm_arxiv(e):
    aid = e.find(f"{ATOM}id").text.rsplit("/abs/", 1)[1]
    aid = re.sub(r"v\d+$", "", aid)
    doi = e.find(f"{ARXIV_NS}doi")
    jr = e.find(f"{ARXIV_NS}journal_ref")
    return {"source": "arxiv", "source_id": aid, "title": re.sub(r"\s+", " ", e.find(f"{ATOM}title").text).strip(),
            "abstract": re.sub(r"\s+", " ", e.find(f"{ATOM}summary").text).strip(),
            "year": int(e.find(f"{ATOM}published").text[:4]), "date": e.find(f"{ATOM}published").text[:10],
            "doi": (doi.text.lower() if doi is not None else ""), "arxiv_id": aid,
            "venue": (jr.text if jr is not None else "arXiv"), "venue_type": "repository", "type": "preprint",
            "language": "en", "authors": [a.find(f"{ATOM}name").text for a in e.findall(f"{ATOM}author")[:6]]}


RUNNERS = {"openalex": run_openalex, "s2": run_s2, "arxiv": run_arxiv}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--string", default="S1", choices=["S1", "S2", "S1r1", "S2r1"])
    ap.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: run date)")
    ap.add_argument("--sources", default="openalex,s2,arxiv")
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    run_dir = REPO / "sok" / "slr" / "runs" / a.run
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    end = a.end or a.run
    qpath = run_dir / f"queries_{a.string}{'_count' if a.count_only else ''}.json"
    allmeta = json.loads(qpath.read_text()) if qpath.exists() else {}
    for s in a.sources.split(","):
        meta, recs = RUNNERS[s](a.string, run_dir, end, a.count_only)
        allmeta[s] = meta
        print(json.dumps({k: v for k, v in meta.items() if k not in ("query_strings",)})[:600])
        if not a.count_only:
            with open(run_dir / f"records_{a.string}_{s}.jsonl", "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        qpath.write_text(json.dumps(allmeta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
