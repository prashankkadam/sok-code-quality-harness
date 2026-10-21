"""T6 retrieval corpus: style-guide passages and secure exemplars, decontaminated against every evaluation task."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import httpx

from ..config import DATA
from ..tasks import Task

CORPUS = DATA / "rag_corpus.jsonl"

SOURCES = {
    "style": [
        ("PEP 8", "https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst"),
        ("PEP 20", "https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst"),
        ("Google Python Style Guide", "https://raw.githubusercontent.com/google/styleguide/gh-pages/pyguide.md"),
    ],
    "security": [
        (f"OWASP {n}", f"https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/{n}.md")
        for n in ["Injection_Prevention_Cheat_Sheet", "OS_Command_Injection_Defense_Cheat_Sheet",
                  "SQL_Injection_Prevention_Cheat_Sheet", "Deserialization_Cheat_Sheet", "Input_Validation_Cheat_Sheet",
                  "Cryptographic_Storage_Cheat_Sheet", "Password_Storage_Cheat_Sheet", "File_Upload_Cheat_Sheet",
                  "Secrets_Management_Cheat_Sheet", "XML_External_Entity_Prevention_Cheat_Sheet",
                  "Server_Side_Request_Forgery_Prevention_Cheat_Sheet", "Logging_Cheat_Sheet"]
    ],
}
# SecCoder's demonstration pool is the secure (after-fix) functions of SVEN's train split.
SECCODER_DEMOS = [f"https://raw.githubusercontent.com/eth-sri/sven/master/data_train_val/train/cwe-{c}.jsonl"
                  for c in ["022", "078", "079", "089", "125", "190", "416", "476", "787"]]

_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOK.findall(text)]


def _chunks(text: str, size: int = 220) -> list[str]:
    paras, out, cur = re.split(r"\n\s*\n", text), [], ""
    for p in paras:
        if len((cur + p).split()) > size and cur:
            out.append(cur.strip())
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c.split()) > 25]


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    t = tokens(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def build(eval_tasks: list[Task]) -> int:
    """Downloads the sources, chunks them, and drops any passage sharing a 13-gram with an evaluation prompt, test or reference."""
    banned: set[tuple[str, ...]] = set()
    for t in eval_tasks:
        for s in [t.prompt, t.reference or "", t.source or "", *(x.code + x.stdin for x in t.public_tests + t.hidden_tests)]:
            banned |= ngrams(s)
    rows = []
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        for kind, srcs in SOURCES.items():
            for name, url in srcs:
                for ch in _chunks(c.get(url).raise_for_status().text):
                    rows.append({"kind": kind, "source": name, "text": ch})
        for url in SECCODER_DEMOS:
            for ln in c.get(url).raise_for_status().text.splitlines():
                d = json.loads(ln)
                if d["file_name"].endswith(".py"):
                    rows.append({"kind": "secure exemplar", "source": f"SecCoder {d['vul_type'].upper()}",
                                 "text": d["func_src_after"]})
    rows = [r for r in rows if not (ngrams(r["text"]) & banned)]
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)


@lru_cache
def _index():
    from rank_bm25 import BM25Okapi
    docs = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    return docs, BM25Okapi([tokens(d["text"]) for d in docs])


def retrieve(task: Task, k: int = 3) -> list[dict]:
    if not CORPUS.exists():
        raise FileNotFoundError(f"{CORPUS}: run `harness rag-build` first")
    docs, bm25 = _index()
    want = {"SEC": {"security", "secure exemplar"}, "RM": {"style"}}.get(task.meta.get("target_dim", ""))
    scores = bm25.get_scores(tokens(task.prompt + " " + (task.source or "")))
    order = sorted(range(len(docs)), key=lambda i: -scores[i])
    picked = [docs[i] for i in order if not want or docs[i]["kind"] in want][:k]
    return picked
