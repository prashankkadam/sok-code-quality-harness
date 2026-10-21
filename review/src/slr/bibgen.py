#!/usr/bin/env python3
"""Generate every BibTeX entry of the SoK from authoritative metadata (DOC-001 kickoff §1.A).

No entry is ever typed by hand. A writer records an identifier; this tool resolves it:

  arXiv ID  -> arXiv API (title, authors, year, primary category), then the peer-reviewed version
               if one exists (Semantic Scholar externalIds -> DBLP record -> Crossref by DOI).
               The arXiv ID is kept in `eprint`.
  DOI       -> Crossref (works API).
  dblp key  -> DBLP record (covers ACL Anthology, PMLR, OpenReview, NeurIPS proceedings), then
               Crossref when the record has a DOI.
  url       -> @misc for model cards and vendor pages: exact title, organisation as author, url,
               note={Accessed <date>}. Verified only if the fetched page contains the exact title.

Usage
  bibgen.py lookup <id> [--section 07_prompting]      # resolve, print key, register for a section
      id forms: arXiv:2303.17651 | doi:10.1145/... | dblp:conf/icse/X24 |
                url:https://... --title "Exact Title" --org "Org" [--year 2026]
  bibgen.py find "<title words>"                       # search arXiv + DBLP for candidate IDs
  bibgen.py build                                      # write sok/paper/bib/*.bib from cited keys
  bibgen.py show <key>                                 # print the resolved record
"""
from __future__ import annotations

import os
import argparse
import html
import csv
import fcntl
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bibtexparser
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from common import NOTES, PAPER, ascii_fold, http_get, norm_text, norm_title, now_iso  # noqa: E402

REG_DIR = NOTES / "registry"
RES_DIR = NOTES / "resolved"
KEYINDEX = NOTES / "keyindex.csv"
LOCK = NOTES / ".keyindex.lock"
STOP_FIRST = {"a", "an", "the"}
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

SECTION_BIB = {
    "02_background": "02_background", "05_measurement": "05_measurement",
    "07_prompting": "07_prompting", "08_feedback_loops": "08_feedback_loops",
    "09_self_refinement": "09_self_refinement", "10_multi_agent": "10_multi_agent",
    "11_retrieval": "11_retrieval", "12_decoding": "12_decoding", "13_training": "13_training",
    "14_models": "14_models",
}
SECTION_ORDER = ["00_abstract", "01_introduction", "02_background", "03_related_surveys",
                 "04_methodology", "05_measurement", "06_taxonomy", "07_prompting",
                 "08_feedback_loops", "09_self_refinement", "10_multi_agent", "11_retrieval",
                 "12_decoding", "13_training", "14_models", "15_empirical", "16_discussion",
                 "17_agenda", "18_threats", "19_conclusion", "A_appendix"]


# --------------------------------------------------------------------------- sources
def arxiv_meta(aid: str, cache: bool = True) -> dict | None:
    aid = re.sub(r"v\d+$", "", aid.strip())
    st, body = http_get("http://export.arxiv.org/api/query", params={"id_list": aid}, cache=cache)
    if st != 200:
        return None
    root = ET.fromstring(body)
    ent = root.find(f"{ATOM}entry")
    if ent is None or ent.find(f"{ATOM}title") is None:
        return None
    eid = ent.find(f"{ATOM}id").text
    if "api/errors" in eid:
        return None
    title = re.sub(r"\s+", " ", ent.find(f"{ATOM}title").text).strip()
    authors = [re.sub(r"\s+", " ", a.find(f"{ATOM}name").text).strip()
               for a in ent.findall(f"{ATOM}author")]
    published = ent.find(f"{ATOM}published").text  # v1 date
    prim = ent.find(f"{ARXIV_NS}primary_category")
    doi = ent.find(f"{ARXIV_NS}doi")
    jref = ent.find(f"{ARXIV_NS}journal_ref")
    return {"arxiv_id": aid, "title": title, "authors": authors, "year": int(published[:4]),
            "published": published[:10], "primary_class": prim.get("term") if prim is not None else "",
            "arxiv_doi": doi.text if doi is not None else "",
            "journal_ref": jref.text if jref is not None else ""}


def s2_paper(pid: str, cache: bool = True) -> dict | None:
    fields = "title,year,venue,publicationVenue,externalIds,authors,publicationDate,journal,publicationTypes"
    st, body = http_get(f"https://api.semanticscholar.org/graph/v1/paper/{pid}",
                        params={"fields": fields}, cache=cache)
    if st != 200:
        return None
    return json.loads(body)


def crossref(doi: str, cache: bool = True) -> dict | None:
    st, body = http_get(f"https://api.crossref.org/works/{doi}", cache=cache)
    if st != 200:
        return None
    return json.loads(body)["message"]


def dblp_bib(key: str, cache: bool = True) -> dict | None:
    st, body = http_get(f"https://dblp.org/rec/{key}.bib", params={"param": "1"}, cache=cache)
    if st != 200 or "@" not in body:
        return None
    db = bibtexparser.loads(body)
    return db.entries[0] if db.entries else None


def dblp_search(title: str, cache: bool = True) -> list[dict]:
    q = re.sub(r"[^A-Za-z0-9 ]+", " ", ascii_fold(title))
    st, body = http_get("https://dblp.org/search/publ/api",
                        params={"q": q, "format": "json", "h": 30}, cache=cache)
    if st != 200:
        return []
    try:
        hits = json.loads(body)["result"]["hits"].get("hit", [])
    except (ValueError, KeyError):
        return []
    return [h["info"] for h in hits]


def clean_dblp_name(n: str) -> str:
    return re.sub(r"\s+\d{4}$", "", n.strip())


# --------------------------------------------------------------------------- resolution
def title_match(a: str, b: str) -> float:
    return fuzz.ratio(norm_title(a), norm_title(b))


OR_VENUES = {
    "ICLR": ("inproceedings", "International Conference on Learning Representations (ICLR)"),
    "ICML": ("inproceedings", "Proceedings of the International Conference on Machine Learning (ICML)"),
    "NeurIPS": ("inproceedings", "Advances in Neural Information Processing Systems (NeurIPS)"),
    "COLM": ("inproceedings", "Conference on Language Modeling (COLM)"),
    "TMLR": ("article", "Transactions on Machine Learning Research"),
}


def _first_last(authors: list[str]) -> str:
    return re.sub(r"[^a-z]", "", ascii_fold(authors[0].split()[-1]).lower()) if authors else ""


def openalex_work(path: str, params=None) -> dict | None:
    p = {"mailto": os.environ.get("CONTACT_EMAIL", "")}
    p.update(params or {})
    st, body = http_get(f"https://api.openalex.org/{path}", params=p)
    if st != 200:
        return None
    return json.loads(body)


def _oa_published_doi(w: dict) -> str:
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
    src = ((w.get("primary_location") or {}).get("source") or {})
    if doi and not doi.startswith("10.48550") and src.get("type") in ("journal", "conference", "book series", "ebook platform"):
        return doi
    for loc in w.get("locations", []):
        s = loc.get("source") or {}
        lp = (loc.get("landing_page_url") or "")
        if loc.get("version") == "publishedVersion" and s.get("type") in ("journal", "conference", "book series") \
                and "doi.org/" in lp and "10.48550" not in lp:
            return lp.split("doi.org/", 1)[1].lower()
    return ""


def peer_reviewed_doi(arxiv_id: str, title: str, authors: list[str]) -> str:
    """DOI of the peer-reviewed version of an arXiv paper, via OpenAlex then Crossref."""
    w = openalex_work(f"works/doi:10.48550/arXiv.{arxiv_id}")
    if w:
        d = _oa_published_doi(w)
        if d:
            return d
    q = re.sub(r"[^A-Za-z0-9 ]+", " ", ascii_fold(title))
    res = openalex_work("works", {"filter": f"title.search:{q}", "per-page": 10})
    fl = _first_last(authors)
    for w in (res or {}).get("results", []):
        if title_match(w.get("title") or "", title) < 95:
            continue
        auths = [a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")]
        if auths and fl and fl not in re.sub(r"[^a-z]", "", ascii_fold(auths[0]).lower()):
            continue
        d = _oa_published_doi(w)
        if d:
            return d
    st, body = http_get("https://api.crossref.org/works",
                        params={"query.bibliographic": title, "rows": 5,
                                "select": "DOI,title,type,author"})
    if st == 200:
        for it in json.loads(body)["message"]["items"]:
            if it.get("type") not in ("journal-article", "proceedings-article"):
                continue
            if title_match((it.get("title") or [""])[0], title) < 95:
                continue
            fam = re.sub(r"[^a-z]", "", ascii_fold((it.get("author") or [{}])[0].get("family", "")).lower())
            if fl and fam and fl != fam:
                continue
            return it["DOI"].lower()
    return ""


def openreview_venue(title: str) -> dict | None:
    """Official ICLR / ICML / NeurIPS / COLM / TMLR acceptance record on OpenReview (no DOI venues)."""
    for api in ("api2.openreview.net", "api.openreview.net"):
        st, body = http_get(f"https://{api}/notes/search",
                            params={"term": title, "type": "terms", "content": "title",
                                    "group": "all", "source": "forum", "limit": 10})
        if st != 200:
            continue
        try:
            notes = json.loads(body).get("notes", [])
        except ValueError:
            continue
        for n in notes:
            c = n.get("content", {})
            t = c.get("title"); t = t.get("value") if isinstance(t, dict) else t
            v = c.get("venue"); v = v.get("value") if isinstance(v, dict) else v
            if not t or not v or title_match(t, title) < 95:
                continue
            m = re.match(r"^(ICLR|ICML|NeurIPS|COLM|TMLR)\b.*?(\d{4})?", v)
            if not m or re.search(r"submission|withdrawn|reject|desk", v, re.I):
                continue
            yr = re.search(r"(20\d{2})", v)
            et, book = OR_VENUES[m.group(1)]
            workshop = "workshop" in v.lower()
            return {"entry_type": et, "venue_name": (v if workshop else book), "year": int(yr.group(1)) if yr else None,
                    "openreview_id": n["id"], "openreview_venue": v, "workshop": workshop,
                    "url": f"https://openreview.net/forum?id={n['id']}"}
    return None


def from_crossref(doi: str) -> dict | None:
    m = crossref(doi)
    if not m:
        return None
    title = html.unescape(html.unescape(" ".join(m.get("title") or [""]).strip()))
    title = re.sub(r"<[^>]+>", "", title)
    authors = []
    for a in m.get("author", []):
        if "family" in a:
            authors.append((a.get("given", "") + " " + a["family"]).strip())
        elif "name" in a:
            authors.append(a["name"])
    year = None
    for k in ("published-print", "published", "issued", "published-online"):
        if m.get(k, {}).get("date-parts") and m[k]["date-parts"][0][0]:
            year = int(m[k]["date-parts"][0][0])
            break
    ctype = m.get("type", "")
    container = html.unescape(html.unescape(" ".join(m.get("container-title") or [""]).strip()))
    rec = {"title": title, "authors": authors, "year": year, "doi": m.get("DOI", doi).lower(),
           "publisher": m.get("publisher", ""), "crossref_type": ctype}
    if ctype == "journal-article":
        rec.update(entry_type="article", journal=container, volume=m.get("volume", ""),
                   number=m.get("issue", ""), pages=m.get("page", "") or m.get("article-number", ""))
    elif ctype in ("proceedings-article", "book-chapter"):
        rec.update(entry_type="inproceedings", booktitle=container, pages=m.get("page", ""))
    else:
        rec.update(entry_type="misc", howpublished=container)
    return rec


def make_key(authors: list[str], year, title: str, family_name: str | None = None) -> str:
    last = family_name or (authors[0].split()[-1] if authors else "anon")
    last = re.sub(r"[^a-z]", "", ascii_fold(last).lower()) or "anon"
    words = [w for w in re.split(r"\s+", ascii_fold(title).strip()) if w]
    while words and re.sub(r"[^a-z0-9]", "", words[0].lower()) in STOP_FIRST:
        words = words[1:]
    first = re.sub(r"[^a-z0-9]", "", words[0].lower()) if words else "untitled"
    return f"{last}{year}{first}"


def resolve(ident: str, title: str | None = None, org: str | None = None, year: int | None = None,
            force_doi: str | None = None, attach_arxiv: str | None = None) -> dict:
    ident = ident.strip()
    rec: dict = {"identifier": ident, "resolved_at": now_iso(), "sources": []}
    low = ident.lower()
    if low.startswith("doi:10.48550/arxiv."):
        ident = "arXiv:" + ident.split("arxiv.", 1)[1]
        low = ident.lower()
    if low.startswith("arxiv:"):
        aid = ident.split(":", 1)[1]
        ax = arxiv_meta(aid)
        if not ax:
            return {**rec, "status": "not-found", "error": "arXiv API returned no entry"}
        rec["sources"].append("arxiv")
        rec.update(arxiv_id=ax["arxiv_id"], primary_class=ax["primary_class"], arxiv_title=ax["title"],
                   arxiv_authors=ax["authors"], arxiv_year=ax["year"])
        doi = force_doi or peer_reviewed_doi(ax["arxiv_id"], ax["title"], ax["authors"])
        cr = from_crossref(doi) if doi else None
        orv = None
        if cr and (force_doi or title_match(cr["title"], ax["title"]) >= 80):
            rec["sources"].append("crossref")
            rec.update(cr)
            rec["venue_source"] = "crossref"
        else:
            orv = openreview_venue(ax["title"])
        if orv and not cr:
            rec["sources"].append("openreview")
            rec.update(title=ax["title"], authors=ax["authors"], year=orv["year"] or ax["year"],
                       entry_type=orv["entry_type"], url=orv["url"], openreview_venue=orv["openreview_venue"],
                       venue_source="openreview")
            rec["journal" if orv["entry_type"] == "article" else "booktitle"] = orv["venue_name"]
        elif not cr:
            rec.update(entry_type="misc", title=ax["title"], authors=ax["authors"], year=ax["year"],
                       venue_source="arxiv-only")
        # Prefer the arXiv author list when the peer-reviewed metadata dropped authors.
        if len(rec.get("authors", [])) < len(ax["authors"]):
            rec["authors"] = ax["authors"]
        rec["status"] = "verified"
    elif low.startswith("doi:"):
        doi = ident.split(":", 1)[1].strip().lower()
        cr = from_crossref(doi)
        if not cr:
            return {**rec, "status": "not-found", "error": "Crossref returned no record"}
        rec["sources"].append("crossref")
        rec.update(cr)
        rec["venue_source"] = "crossref"
        if attach_arxiv:
            ax = arxiv_meta(attach_arxiv)
            if not ax or _first_last(ax["authors"]) != _first_last(cr["authors"]):
                return {**rec, "status": "mismatch", "error": "attached arXiv ID has a different first author"}
            rec["arxiv_id"] = ax["arxiv_id"]
            rec["primary_class"] = ax["primary_class"]
            rec["sources"].append("arxiv")
        rec["status"] = "verified"
    elif low.startswith("dblp:"):
        k = ident.split(":", 1)[1].strip()
        pr = dblp_bib(k)
        if not pr:
            return {**rec, "status": "not-found", "error": "DBLP returned no record"}
        rec["sources"].append("dblp")
        rec["dblp_key"] = k
        cr = from_crossref(pr["doi"]) if pr.get("doi") else None
        if cr:
            rec["sources"].append("crossref")
            rec.update(cr)
            rec["venue_source"] = "crossref"
        else:
            rec.update(title=pr["title"].replace("{", "").replace("}", ""),
                       authors=[clean_dblp_name(a) for a in re.split(r"\s+and\s+", pr["author"].replace("\n", " "))],
                       year=int(pr["year"]), venue_source="dblp")
            if pr.get("ENTRYTYPE") == "article":
                rec.update(entry_type="article", journal=pr.get("journal", ""), volume=pr.get("volume", ""),
                           number=pr.get("number", ""), pages=pr.get("pages", ""))
            else:
                rec.update(entry_type="inproceedings", booktitle=pr.get("booktitle", ""), pages=pr.get("pages", ""))
            if pr.get("url"):
                rec["url"] = pr["url"]
        if pr.get("eprint"):
            rec["arxiv_id"] = pr["eprint"]
        rec["status"] = "verified"
    elif low.startswith("url:"):
        url = ident.split(":", 1)[1].strip()
        if not (title and org):
            return {**rec, "status": "not-found", "error": "url identifiers need --title and --org"}
        st, raw = http_get(url, cache=True, binary=True)
        if raw[:4] == b"%PDF":
            from quotecheck import pdf_to_text
            body = pdf_to_text(raw)
        else:
            body = raw.decode("utf-8", "replace")
            body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
        text = norm_text(html.unescape(re.sub(r"<[^>]+>", " ", body or "")))
        text = re.sub(r"&amp;", "&", text)
        ok = st == 200 and norm_text(title) in text
        rec.update(entry_type="misc", title=title, authors=[org], org=org, url=url,
                   year=year or int(now_iso()[:4]), accessed=now_iso()[:10],
                   venue_source="vendor-page", http_status=st, title_found_on_page=ok,
                   status="verified" if ok else "not-found")
        rec["sources"].append("vendor-page")
        if not ok:
            rec["error"] = f"HTTP {st}; exact title found on page: {ok}"
    else:
        return {**rec, "status": "not-found", "error": "unknown identifier form"}
    fam = None
    rec["key"] = make_key(rec["authors"], rec["year"], rec["title"], fam) if rec.get("org") is None \
        else make_key([re.sub(r"[^A-Za-z]", "", org)], rec["year"], rec["title"])
    rec["first_author"] = rec["authors"][0] if rec.get("authors") else ""
    rec["venue"] = rec.get("booktitle") or rec.get("journal") or ("arXiv" if rec.get("venue_source") == "arxiv-only" else rec.get("howpublished", ""))
    return rec


# --------------------------------------------------------------------------- key index
def _read_index() -> dict[str, dict]:
    if not KEYINDEX.exists():
        return {}
    with open(KEYINDEX) as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def _write_index(idx: dict[str, dict]) -> None:
    with open(KEYINDEX, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "identifier", "norm_title", "doi", "arxiv_id"])
        w.writeheader()
        for k in sorted(idx):
            w.writerow(idx[k])


def same_paper(a: dict, rec: dict) -> bool:
    if a.get("doi") and rec.get("doi") and a["doi"] == rec["doi"]:
        return True
    if a.get("arxiv_id") and rec.get("arxiv_id") and a["arxiv_id"] == rec["arxiv_id"]:
        return True
    # Web pages (url: entries) are the same source only at the same address: vendors reuse generic
    # titles such as "Release notes" or "Models overview".
    a_url = str(a.get("identifier", "")).lower().startswith("url:")
    r_url = str(rec.get("identifier", "")).lower().startswith("url:")
    if a_url or r_url:
        return a_url and r_url and a["identifier"].strip() == rec["identifier"].strip()
    return a.get("norm_title") == norm_title(rec.get("title", ""))


def assign_key(rec: dict) -> str:
    """Give the paper its canonical key; one key per paper, collisions get a/b suffixes."""
    NOTES.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        idx = _read_index()
        for k, row in idx.items():
            if same_paper(row, rec):
                fcntl.flock(lk, fcntl.LOCK_UN)
                return k
        base = rec["key"]
        key = base
        n = 0
        while key in idx:
            n += 1
            key = base + "abcdefghij"[n]
        idx[key] = {"key": key, "identifier": rec["identifier"], "norm_title": norm_title(rec["title"]),
                    "doi": rec.get("doi", ""), "arxiv_id": rec.get("arxiv_id", "")}
        _write_index(idx)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return key


def register(section: str, key: str, ident: str) -> None:
    REG_DIR.mkdir(parents=True, exist_ok=True)
    path = REG_DIR / f"{section}.csv"
    rows = []
    if path.exists():
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    if any(r["key"] == key for r in rows):
        return
    rows.append({"key": key, "identifier": ident, "section": section, "added_at": now_iso()})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "identifier", "section", "added_at"])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- bibtex
def tex_escape(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}") if "$" not in s else s
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = re.sub(r"(?<!\\)" + re.escape(a), b.replace("\\", "\\\\"), s)
    return s


def to_bibtex(rec: dict) -> str:
    et = rec.get("entry_type", "misc")
    f = {"title": "{{" + tex_escape(rec["title"]) + "}}"}
    if rec.get("org"):
        f["author"] = "{" + tex_escape(rec["org"]) + "}"
    else:
        auths = rec["authors"]
        # WRITING_GUIDE allows "first authors followed by and others"; keeps the reference list in budget.
        shown = auths if len(auths) <= 6 else auths[:3] + ["others"]
        f["author"] = " and ".join(tex_escape(a) for a in shown)
    f["year"] = str(rec["year"])
    if et == "article":
        f["journal"] = tex_escape(rec.get("journal", ""))
        for k in ("volume", "number", "pages"):
            if rec.get(k):
                f[k] = str(rec[k])
    elif et == "inproceedings":
        f["booktitle"] = tex_escape(rec.get("booktitle", ""))
        if rec.get("pages"):
            f["pages"] = str(rec["pages"])
        if rec.get("publisher"):
            f["publisher"] = tex_escape(rec["publisher"])
    else:
        if rec.get("arxiv_id") and not rec.get("org"):
            f["howpublished"] = f"arXiv preprint arXiv:{rec['arxiv_id']}"
            f["note"] = f"arXiv:{rec['arxiv_id']}"
        elif rec.get("org"):
            f["howpublished"] = "Vendor documentation"
            f["note"] = f"Accessed {rec['accessed']}"
        elif rec.get("howpublished"):
            f["howpublished"] = tex_escape(rec["howpublished"])
    if rec.get("doi"):
        f["doi"] = rec["doi"]
    if rec.get("arxiv_id"):
        f["eprint"] = rec["arxiv_id"]
        f["archiveprefix"] = "arXiv"
    # Compact references (page budget): the arXiv category, the conference publisher and any URL that
    # duplicates the DOI or eprint are not printed. The resolved record keeps all of them.
    f.pop("publisher", None) if et == "inproceedings" else None
    if rec.get("url") and not rec.get("doi") and not rec.get("arxiv_id"):
        f["url"] = rec["url"]
    elif rec.get("org") and rec.get("url"):
        f["url"] = rec["url"]
    elif rec.get("openreview_venue") and rec.get("url") and not rec.get("doi"):
        f["url"] = rec["url"]
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in f.items())
    return f"@{et}{{{rec['key']},\n{body}\n}}\n"


def cited_keys_by_section() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    pat = re.compile(r"\\(?:cite[tp]?|citeauthor|citeyear|citealp|nocite)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
    litpat = re.compile(r"\\lit\{[^{}]*\}\{([^}]*)\}")
    for p in sorted((PAPER / "sections").glob("*.tex")):
        txt = re.sub(r"(?<!\\)%.*", "", p.read_text())
        ks = set()
        for m in list(pat.finditer(txt)) + list(litpat.finditer(txt)):
            ks.update(k.strip() for k in m.group(1).split(",") if k.strip())
        out[p.stem] = ks
    for p in sorted((PAPER / "generated").glob("*.tex")):
        txt = p.read_text()
        ks = set()
        for m in list(pat.finditer(txt)) + list(litpat.finditer(txt)):
            ks.update(k.strip() for k in m.group(1).split(",") if k.strip())
        out.setdefault("generated:" + p.stem, set()).update(ks)
    return out


def load_resolved(key: str) -> dict | None:
    p = RES_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cmd_build(_args) -> int:
    by_sec = cited_keys_by_section()
    usage: dict[str, list[str]] = {}
    for sec, ks in by_sec.items():
        for k in ks:
            usage.setdefault(k, []).append(sec)
    files: dict[str, list[str]] = {v: [] for v in set(SECTION_BIB.values()) | {"core"}}
    missing = []
    for k in sorted(usage):
        rec = load_resolved(k)
        if not rec or rec.get("status") != "verified":
            missing.append(k)
            continue
        secs = sorted({s.split(":")[0] if not s.startswith("generated:") else "core" for s in usage[k]},
                      key=lambda s: SECTION_ORDER.index(s) if s in SECTION_ORDER else 99)
        target = "core"
        if len(secs) < 3:
            target = next((SECTION_BIB[s] for s in secs if s in SECTION_BIB), "core")
        files[target].append(to_bibtex(rec))
    for name, entries in files.items():
        (PAPER / "bib" / f"{name}.bib").write_text(
            f"% {name}.bib: GENERATED by src/slr/bibgen.py build at {now_iso()}. Do not edit by hand.\n\n"
            + "\n".join(entries))
    print(f"cited keys: {len(usage)}; written: {sum(len(v) for v in files.values())}; "
          f"missing/unverified: {len(missing)}")
    for k in missing:
        print("  MISSING or UNVERIFIED:", k)
    return 1 if missing else 0


def cmd_lookup(args) -> int:
    rec = resolve(args.id, title=args.title, org=args.org, year=args.year, force_doi=args.doi, attach_arxiv=args.arxiv)
    if rec.get("status") != "verified":
        print(json.dumps({"status": rec.get("status"), "error": rec.get("error"), "identifier": args.id}))
        return 2
    rec["key"] = assign_key(rec)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / f"{rec['key']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    if args.section:
        register(args.section, rec["key"], args.id)
    auth = rec["authors"]
    print(json.dumps({"key": rec["key"], "status": "verified", "title": rec["title"],
                      "authors": auth[:3] + (["..."] if len(auth) > 3 else []), "year": rec["year"],
                      "venue": rec["venue"], "entry_type": rec.get("entry_type"),
                      "arxiv_id": rec.get("arxiv_id", ""), "doi": rec.get("doi", "")}, ensure_ascii=False))
    return 0


def cmd_find(args) -> int:
    q = args.query
    st, body = http_get("http://export.arxiv.org/api/query",
                        params={"search_query": "ti:" + " AND ti:".join(re.findall(r"[A-Za-z0-9]+", q)[:8]),
                                "max_results": 8})
    out = []
    if st == 200:
        root = ET.fromstring(body)
        for e in root.findall(f"{ATOM}entry"):
            out.append({"src": "arxiv", "id": "arXiv:" + e.find(f"{ATOM}id").text.rsplit("/", 1)[1].split("v")[0],
                        "title": re.sub(r"\s+", " ", e.find(f"{ATOM}title").text),
                        "year": e.find(f"{ATOM}published").text[:4],
                        "first_author": e.find(f"{ATOM}author/{ATOM}name").text})
    res = openalex_work("works", {"search": q, "per-page": 8})
    for w in (res or {}).get("results", []):
        auths = [x["author"]["display_name"] for x in w.get("authorships", []) if x.get("author")]
        ids = w.get("ids", {})
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        ident = ("arXiv:" + doi.split("arxiv.", 1)[1]) if "10.48550/arxiv." in doi.lower() else (("doi:" + doi) if doi else "")
        out.append({"src": "openalex", "id": ident, "title": w.get("title"), "year": w.get("publication_year"),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "type": w.get("type"), "first_author": auths[0] if auths else "", "cited_by": w.get("cited_by_count")})
    for o in out:
        print(json.dumps(o, ensure_ascii=False))
    return 0


def cmd_register_tex(args) -> int:
    """Register every key cited in sections/<section>.tex under that section (identifier from keyindex)."""
    idx = _read_index()
    keys = cited_keys_by_section().get(args.section, set())
    missing = []
    for k in sorted(keys):
        if k in idx:
            register(args.section, k, idx[k]["identifier"])
        else:
            missing.append(k)
    print(f"registered {len(keys) - len(missing)} keys for {args.section}; not in key index: {missing}")
    return 1 if missing else 0


def cmd_show(args) -> int:
    rec = load_resolved(args.key)
    print(json.dumps(rec, indent=1, ensure_ascii=False) if rec else "not resolved")
    if rec:
        print(to_bibtex(rec))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lookup")
    p.add_argument("id")
    p.add_argument("--section")
    p.add_argument("--title")
    p.add_argument("--org")
    p.add_argument("--year", type=int)
    p.add_argument("--doi", help="DOI of the peer-reviewed version of an arXiv paper (e.g. when the title changed)")
    p.add_argument("--arxiv", help="with a doi: identifier, attach this arXiv ID as eprint (same first author required)")
    p.set_defaults(fn=cmd_lookup)
    p = sub.add_parser("find")
    p.add_argument("query")
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("build")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("register-tex")
    p.add_argument("section")
    p.set_defaults(fn=cmd_register_tex)
    p = sub.add_parser("show")
    p.add_argument("key")
    p.set_defaults(fn=cmd_show)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
