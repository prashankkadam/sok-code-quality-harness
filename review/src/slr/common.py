"""Shared helpers for the DOC-001 literature scripts: polite HTTP with per-host rate limits
(safe across parallel processes), caching, and text normalization."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".cache"
PAPER = REPO / "sok" / "paper"
NOTES = PAPER / "notes"
UA = f"sok-doc001-literature-pipeline/1.0 (mailto:{os.environ.get('CONTACT_EMAIL', '')})"

# Minimum seconds between calls to each host, shared by every process through a lock file.
MIN_INTERVAL = {
    "export.arxiv.org": 3.1,
    "arxiv.org": 1.0,
    "dblp.org": 1.2,
    "api.semanticscholar.org": 1.1,
    "api.crossref.org": 0.3,
    "doi.org": 0.3,
    "api.openalex.org": 0.15,
}


def _host(url: str) -> str:
    return re.sub(r"^https?://([^/]+).*$", r"\1", url)


def _throttle(host: str) -> None:
    gap = MIN_INTERVAL.get(host, 0.5)
    lockdir = CACHE / "ratelimit"
    lockdir.mkdir(parents=True, exist_ok=True)
    lockfile = lockdir / f"{host}.lock"
    with open(lockfile, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        last = float(raw) if raw else 0.0
        wait = last + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def _mark_down(host: str) -> None:
    d = CACHE / "down"
    d.mkdir(parents=True, exist_ok=True)
    (d / host).write_text(str(time.time()))


def _is_down(host: str, ttl: float = 3600.0) -> bool:
    p = CACHE / "down" / host
    return p.exists() and time.time() - float(p.read_text() or 0) < ttl


def http_get(url: str, *, params=None, headers=None, cache: bool = True, timeout: int = 60,
             retries: int = 6, binary: bool = False):
    """GET with per-host throttling, retries on 429/5xx, and an on-disk cache.
    Returns (status_code, body) where body is str or bytes. 404 is returned, not raised."""
    api_hosts = ("api.", "export.arxiv.org", "doi.org")
    h = {"User-Agent": UA if any(a in _host(url) for a in api_hosts) else
         f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) sok-research (mailto:{os.environ.get('CONTACT_EMAIL', '')})"}
    if headers:
        h.update(headers)
    if "api.semanticscholar.org" in url and os.environ.get("S2_API_KEY"):
        h["x-api-key"] = os.environ["S2_API_KEY"]
    key = hashlib.sha256(json.dumps([url, params, headers], sort_keys=True).encode()).hexdigest()
    cpath = CACHE / "http" / key[:2] / key
    if cache and cpath.exists():
        meta = json.loads((cpath.with_suffix(".meta")).read_text())
        data = cpath.read_bytes()
        return meta["status"], (data if binary else data.decode("utf-8", "replace"))
    if _is_down(_host(url)):
        return 429, (b"" if binary else "")
    delay = 3.0
    last_exc = None
    for _ in range(retries):
        _throttle(_host(url))
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
        except requests.RequestException as exc:  # network hiccup
            last_exc = exc
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 429 and ("Insufficient budget" in r.text or "boolean operators" in r.text):
            # OpenAlex daily no-key budget exhausted: fail fast instead of retrying for minutes.
            _mark_down(_host(url))
            return 429, (r.content if binary else r.text)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if cache and r.status_code in (200, 404):
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_bytes(r.content)
            cpath.with_suffix(".meta").write_text(json.dumps({"status": r.status_code, "url": r.url}))
        return r.status_code, (r.content if binary else r.text)
    if last_exc:
        raise last_exc
    return 599, b"" if binary else ""


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_title(t: str) -> str:
    t = ascii_fold(t or "").lower()
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"[{}\\]", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
             "’": "'", "‘": "'", "“": '"', "”": '"', "−": "-",
             "–": "-", "—": "-", " ": " ", "‐": "-", "‑": "-",
             "×": "x", "→": "->", "≤": "<=", "≥": ">="}


def norm_text(s: str) -> str:
    """Normalization used to compare a verbatim quote with extracted source text:
    ligatures, typographic quotes and dashes, line-break hyphenation and whitespace."""
    for a, b in LIGATURES.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)  # hyphenation across line breaks
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path) -> list[dict]:
    """Read a .jsonl file, or its .jsonl.gz sibling if only the compressed copy exists."""
    import gzip
    p = Path(path)
    if not p.exists() and Path(str(p) + ".gz").exists():
        with gzip.open(str(p) + ".gz", "rt") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]
