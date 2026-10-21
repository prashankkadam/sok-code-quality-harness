"""E7 retrieval family (H37-H46). Corpora are built in prepare() from the authors' released data (or by their recipe when
not released), decontaminated against every E7 task with 13-gram overlap, and stored under ctx.dir(hid)."""
from __future__ import annotations

import ast
import builtins
import difflib
import io
import json
import keyword
import os
import random
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from .. import sandbox
from ..config import load
from ..llm import Message, scope
from ..tasks import Task
from ..techniques import common as C
from ..techniques.rag import ngrams, tokens
from ..training import fireworks as FW
from .base import Ctx, H2HTechnique, h2h

ENCODERS = {
    "codebert-python": ("neulab/codebert-python", "mean", 512),
    "graphcodebert": ("microsoft/graphcodebert-base", "cls", 512),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", 512),
    "qodo": ("Qodo/Qodo-Embed-1-1.5B", "last", 8192),
    "jina": ("jinaai/jina-embeddings-v3", "jina", 8192),
}
PIE_PYTHON = "1ec8eOWgnBrzy2HlNDlTX6iURwQcIxDXH"
RAS_MERCURY = "1CGGSdZsrgmNxFJx3j9xy8H6VtTVI8ahd"
CWE_CSV = "https://cwe.mitre.org/data/csv/1000.csv.zip"
GITHUB = "https://api.github.com"


# ---------- shared helpers (also used by prompting.py) ----------

def fetch(url: str, dest: Path, headers: dict | None = None) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", url, headers=headers, timeout=1200, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for b in r.iter_bytes():
                    f.write(b)
        tmp.rename(dest)
    return dest


def gdrive(file_id: str, dest: Path) -> Path:
    return fetch(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t", dest)


def github() -> httpx.Client:
    h = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        h["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    return httpx.Client(base_url=GITHUB, headers=h, timeout=120, follow_redirects=True)


def prep(hid: str) -> scope:
    """Charges nothing to a task: offline calls are logged under prepare/<hid>."""
    return scope(f"prepare/{hid}", hid)


@lru_cache(maxsize=1)
def banned() -> frozenset:
    from ..benchmarks.base import get
    out: set = set()
    for name in load("benchmarks")["h2h"]:
        for t in get(name).tasks():
            for s in (t.prompt, t.reference or "", t.source or "", *(x.code + x.stdin for x in t.public_tests + t.hidden_tests)):
                out |= ngrams(s)
    return frozenset(out)


def decontaminate(rows: list[dict], *keys: str) -> list[dict]:
    b = banned()
    return [r for r in rows if not any(ngrams(str(r.get(k) or "")) & b for k in keys)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@lru_cache(maxsize=4)
def _encoder(name: str, path: str | None = None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf, pool, _ = ENCODERS[name]
    src = path or hf
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(src, trust_remote_code=pool == "jina").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=pool == "jina")
    tok.padding_side = "right"
    return tok, model, dev


def embed(name: str, texts: list[str], path: str | None = None, batch: int = 16) -> np.ndarray:
    """L2-normalized sentence vectors from a local encoder."""
    import torch
    tok, model, dev = _encoder(name, path)
    _, pool, max_len = ENCODERS[name]
    if pool == "jina":
        v = np.asarray(model.encode(texts, task="text-matching", max_length=max_len), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)
    out = []
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**b).last_hidden_state
        m = b["attention_mask"]
        if pool == "cls":
            v = h[:, 0]
        elif pool == "last":
            v = h[torch.arange(h.size(0), device=dev), m.sum(1) - 1]
        else:
            v = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def save_index(d: Path, rows: list[dict], vecs: np.ndarray, name: str = "index") -> None:
    write_jsonl(d / f"{name}.jsonl", rows)
    np.save(d / f"{name}.npy", vecs.astype(np.float32))


@lru_cache(maxsize=16)
def load_index(d: Path, name: str = "index") -> tuple[list[dict], np.ndarray]:
    return read_jsonl(d / f"{name}.jsonl"), np.load(d / f"{name}.npy")


def top_k(vecs: np.ndarray, q: np.ndarray, k: int, exclude: set[int] | frozenset = frozenset()) -> list[int]:
    order = np.argsort(-(vecs @ q))
    return [int(i) for i in order if int(i) not in exclude][:k]


def pie_python(ctx: Ctx, split: str = "train") -> list[dict]:
    """PIE (Madaan et al.) Python slow/fast pairs from CodeNet: fields input (slow), target (fast), problem_id, cpu times."""
    z = gdrive(PIE_PYTHON, ctx.dir("pie-python") / "python_splits.zip")
    with zipfile.ZipFile(z) as f:
        return [json.loads(l) for l in f.read(f"python_splits/{split}.jsonl").decode().splitlines() if l.strip()]


def pie_hq(rows: list[dict], per_problem: int = 4) -> list[dict]:
    """PIE's high-quality subset: the highest measured speedups, at most four pairs per problem."""
    def speedup(r):
        return (r.get("measured_runtime_v0") or 0) / max(r.get("measured_runtime_v1") or 1e9, 1e-9)
    by: dict[str, list] = defaultdict(list)
    for r in sorted((r for r in rows if not r["code_same"] and speedup(r) > 1), key=lambda r: -speedup(r)):
        if len(by[r["problem_id"]]) < per_problem:
            by[r["problem_id"]].append(r)
    return [r for v in by.values() for r in v]


@lru_cache(maxsize=1)
def cwe_catalog(root: Path) -> dict[str, dict]:
    """MITRE CWE research view (CWE-1000): id -> name, description, mitigations."""
    import csv
    z = fetch(CWE_CSV, root / "cwe-1000.csv.zip")
    with zipfile.ZipFile(z) as f:
        text = f.read(f.namelist()[0]).decode("utf-8", "replace")
    out = {}
    for r in csv.DictReader(io.StringIO(text)):
        mit = [m.split("DESCRIPTION:", 1)[-1].split(":EFFECTIVENESS", 1)[0].strip(": ")
               for m in (r.get("Potential Mitigations") or "").split("::") if "DESCRIPTION:" in m]
        out[f"CWE-{r['CWE-ID']}"] = {"name": r["Name"], "description": r["Description"], "mitigations": mit}
    return out


def draft(task: Task, ctx: Ctx) -> str:
    """The program a code-optimization technique starts from: the given code for edit tasks, else one T0 generation."""
    if task.mode == "edit" and task.source:
        return ctx.keep(task.source)
    return ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task), role="draft")))


def cost(task: Task, code: str) -> float | None:
    """Public-example cost: user-space instructions (perf) or median wall time; None if any public example fails."""
    if not code.strip() or not sandbox.passes_public(task, code):
        return None
    total = 0.0
    for t in task.public_tests:
        n = sandbox.instructions(task, code, t)
        if n is None:
            w = sandbox.wallclock(task, code, t, repeats=3)
            if w is None:
                return None
            n = w[0]
        total += n
    return total


def fastest(task: Task, codes: list[str]) -> str | None:
    scored = [(c, cost(task, c)) for c in codes]
    ok = [(c, s) for c, s in scored if s is not None]
    return min(ok, key=lambda x: x[1])[0] if ok else None


def user(text: str, system: str | None = None) -> list[Message]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]


def json_of(text: str):
    blocks = re.findall(r"```json\s*(.*?)```", text or "", re.S)
    for cand in [*reversed(blocks), text or ""]:
        for open_, close in (("{", "}"), ("[", "]")):
            i, j = cand.find(open_), cand.rfind(close)
            if 0 <= i < j:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    continue
    return None


def functions(code: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def segment(code: str, node: ast.AST) -> str:
    lines = code.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[start - 1:node.end_lineno])


def single_function_change(before: str, after: str) -> tuple[str, str, str] | None:
    """(name, before function, after function) when the only change between two files is inside one function."""
    fb, fa = functions(before), functions(after)
    changed = [k for k in fa if k in fb and segment(before, fb[k]) != segment(after, fa[k])]
    if len(changed) != 1:
        return None
    k = changed[0]
    b, a = segment(before, fb[k]), segment(after, fa[k])
    return (k, b, a) if before.replace(b, a, 1) == after else None


def tuned(ctx: Ctx, hid: str):
    if hid not in ctx.tuned_models:
        ctx.tuned_models[hid] = (ctx.dir(hid) / "model.txt").read_text().strip()
    return ctx.tuned(hid)


def lora_sft(ctx: Ctx, hid: str, data: Path, *, rank: int, epochs: int, lr: float) -> str:
    """LoRA SFT on Fireworks with the paper's hyperparameters, context sized to the longest example; the adapter is served
    on the base deployment."""
    name = f"{hid.lower()}-{ctx.model_key}-{data.stem}".replace("_", "-")
    model = FW.sft(ctx.model_key, FW.upload_dataset(data, name), name, data,
                   {"loraRank": rank, "epochs": epochs, "learningRate": lr})
    served = FW.deploy(model, load("training")["deployments"][ctx.model_key])
    (ctx.dir(hid) / "model.txt").write_text(served)
    ctx.tuned_models[hid] = served
    return served


# ---------- H37 PIE dynamic retrieval-based few-shot prompting ----------

PIE_K, PIE_SAMPLES, PIE_TEMPERATURE = 4, 8, 0.7
PIE_SHOT = "### Program:\n{src}\n### Optimized Version:\n{tgt}\n\n"
PIE_QUERY = "### Program:\n{src}\n### Optimized Version:\n"


@h2h
class PIERetrieval(H2HTechnique):
    """K=4 nearest PIE training pairs by CodeBERTScore-Python embedding of the slow program; Best@8 at T=0.7, the fastest
    program passing the public examples is kept (PIE keeps the fastest correct of the 8)."""
    hid, name, family, target = "H37", "Dynamic retrieval-based few-shot prompting", "retrieval", ("EFF",)
    code_url = "https://github.com/LearningOpt/pie"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "index.npy").exists():
            return
        rows = decontaminate([{"src": r["input"], "tgt": r["target"], "problem_id": r["problem_id"]}
                              for r in pie_python(ctx) if not r["code_same"]], "src", "tgt")
        save_index(d, rows, embed("codebert-python", [r["src"] for r in rows]))

    def solve(self, task: Task, ctx: Ctx) -> str:
        p0 = draft(task, ctx)
        rows, vecs = load_index(ctx.dir(self.hid))
        shots = [rows[i] for i in reversed(top_k(vecs, embed("codebert-python", [p0])[0], PIE_K))]
        prompt = "".join(PIE_SHOT.format(**r) for r in shots) + PIE_QUERY.format(src=p0)
        outs = ctx.llm.sample(user(prompt, C.SYSTEM), PIE_SAMPLES, temperature=PIE_TEMPERATURE, role="optimize")
        return fastest(task, [C.code_of(task, o) for o in outs]) or p0


# ---------- H38 LAIL ----------

LAIL_BM25, LAIL_POS, LAIL_NEG = 50, 5, 5
LAIL_RANDOM_NEG, LAIL_LR, LAIL_BATCH = 63, 5e-5, 32
LAIL_EPOCHS, LAIL_TAU = 10, 0.05
LAIL_K, LAIL_TEMPERATURE = 3, 0.8


def _mbpp_text(r: dict) -> str:
    return f"{r['text']}\n{r['test_list'][0]}"


@h2h
class LAIL(H2HTechnique):
    """Candidates labelled by the model's own length-normalized log-likelihood of the reference program given one
    candidate example (white-box, 50 BM25 candidates, top/bottom 5); a GraphCodeBERT retriever is trained contrastively on
    those labels and selects 3 examples (most similar last), concatenated without instructions."""
    hid, name, family, target = "H38", "LAIL", "retrieval", ("RM",)
    whitebox = trains = True
    data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "retriever" / "config.json").exists():
            return
        from datasets import load_dataset
        from rank_bm25 import BM25Okapi
        pool = decontaminate([dict(r) for r in load_dataset("google-research-datasets/mbpp", "full", split="train")],
                             "text", "code")
        write_jsonl(d / "pool.jsonl", pool)
        bm25 = BM25Okapi([tokens(_mbpp_text(r)) for r in pool])
        lm, labels = ctx.lm(), []
        for q, r in enumerate(pool):
            sc = bm25.get_scores(tokens(_mbpp_text(r)))
            cands = [int(i) for i in np.argsort(-sc) if int(i) != q][:LAIL_BM25]
            ll = [self._loglik(lm, pool[c], r) for c in cands]
            order = [cands[i] for i in np.argsort(ll)[::-1]]
            labels.append({"q": q, "pos": order[:LAIL_POS], "neg": order[-LAIL_NEG:], "cands": cands})
        write_jsonl(d / "labels.jsonl", labels)
        self._train(d, pool, labels)

    @staticmethod
    def _loglik(lm, ex: dict, q: dict) -> float:
        import torch
        ctx_text = f"{_mbpp_text(ex)}\n```python\n{ex['code']}\n```\n\n{_mbpp_text(q)}"
        prefix = lm.encode_chat([{"role": "user", "content": ctx_text}])
        target = lm.encode(q["code"])
        ids = torch.cat([prefix, target], dim=1)
        logits = lm.forward(ids).logits[0, prefix.shape[1] - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, target[0].unsqueeze(-1)).squeeze(-1)
        return float(lp.mean())

    @staticmethod
    def _train(d: Path, pool: list[dict], labels: list[dict]) -> None:
        import torch
        tok, model, dev = _encoder("graphcodebert")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=LAIL_LR)
        texts = [_mbpp_text(r) for r in pool]
        rng = random.Random(0)

        def enc(xs):
            b = tok(xs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            return torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0], dim=-1)

        for _ in range(LAIL_EPOCHS):
            rng.shuffle(labels)
            for i in range(0, len(labels), LAIL_BATCH):
                opt.zero_grad()
                for lab in labels[i:i + LAIL_BATCH]:
                    outside = [j for j in range(len(pool)) if j != lab["q"] and j not in lab["cands"]]
                    negs = [rng.choice(lab["neg"])] + rng.sample(outside, min(LAIL_RANDOM_NEG, len(outside)))
                    e = enc([texts[lab["q"]], texts[rng.choice(lab["pos"])]] + [texts[j] for j in negs])
                    sims = (e[0:1] @ e[1:].T) / LAIL_TAU
                    loss = torch.nn.functional.cross_entropy(sims, torch.zeros(1, dtype=torch.long, device=dev))
                    (loss / LAIL_BATCH).backward()
                opt.step()
        model.eval()
        model.save_pretrained(d / "retriever")
        tok.save_pretrained(d / "retriever")
        with torch.no_grad():
            np.save(d / "pool.npy", np.concatenate([enc(texts[i:i + 32]).cpu().numpy() for i in range(0, len(texts), 32)]))

    def solve(self, task: Task, ctx: Ctx) -> str:
        d = ctx.dir(self.hid)
        pool, vecs = read_jsonl(d / "pool.jsonl"), np.load(d / "pool.npy")
        q = embed("graphcodebert", [task.prompt], path=str(d / "retriever"))[0]
        shots = [pool[i] for i in reversed(top_k(vecs, q, LAIL_K))]
        body = "".join(f"{_mbpp_text(r)}\n```python\n{r['code']}\n```\n\n" for r in shots) + C.task_text(task)
        return C.code_of(task, ctx.llm.ask(user(body), temperature=LAIL_TEMPERATURE))


# ---------- H39 KPC ----------

KPC_SYSTEM = "You are a helpful assistant."
KPC_LIST = ("List the fully qualified name of the Python standard library functions and methods used in the code above "
            "(module.qualname, for example json.loads or os.path.join; built-in functions without a module, for example "
            "int or open). Provide only the names, without parameters and with each name in backticks, no other text. "
            "Display in a list.")
KPC_CHECK = "Is the {exc} handled or caught for {api} in the code snippet? Answer in Yes or No only."
KPC_FIX = "Please check {cond} for {api} in the code snippet you provided, otherwise raise {exc}."
KPC_DOCS = "https://api.github.com/repos/python/cpython/contents/Doc/library?ref=3.12"
KPC_RAISE = re.compile(r":exc:`~?!?([\w.]+)`")


def _kpc_sentences(api: str, body: list[str], kb: dict) -> None:
    text = re.sub(r"\s+", " ", " ".join(body))
    for sent in re.split(r"(?<=[.;])\s+", text):
        if "raise" in sent.lower():
            for exc in KPC_RAISE.findall(sent):
                kb[api].append({"throw_exception": exc, "throw_condition": re.sub(r":\w+:`~?!?([^`]+)`", r"\1", sent).strip()})


def _kpc_kb() -> dict[str, list[dict]]:
    """Exception knowledge per API from the CPython 3.12 library reference: the sentences of each function, method or
    class entry that say an exception is raised (the JDK 'Throws:' sections in KPC)."""
    kb: dict[str, list[dict]] = defaultdict(list)
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for f in c.get(KPC_DOCS).raise_for_status().json():
            if not f["name"].endswith(".rst"):
                continue
            module, api, body = "", None, []
            for line in c.get(f["download_url"]).text.splitlines():
                s = line.strip()
                mod = re.match(r"\.\. (?:current)?module:: ([\w.]+)", s)
                entry = re.match(r"\.\. (?:function|method|classmethod|staticmethod|class|decorator):: ([\w.]+)", s)
                if entry or mod or (line[:1].strip() and api):
                    if api:
                        _kpc_sentences(api, body, kb)
                    api, body = None, []
                if mod:
                    module = mod.group(1)
                if entry:
                    name = entry.group(1)
                    api = name if not module or module == "builtins" or name.startswith(module + ".") else f"{module}.{name}"
                elif api is not None:
                    body.append(s)
            if api:
                _kpc_sentences(api, body, kb)
    return {k: v for k, v in kb.items() if v}


@h2h
class KPC(H2HTechnique):
    """Prompt chaining: generate; a checker lists the standard-library APIs used, asks per (API, documented exception)
    whether it is handled, and every unhandled pair becomes one 'check <condition> ... otherwise raise' line sent back to
    the generator; repeat until no pair is unhandled. Knowledge base: Python library reference instead of the JDK docs."""
    hid, name, family, target = "H39", "KPC", "retrieval", ("RM",)
    code_url = "https://github.com/goodchar123/KPC"
    ported = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        p = ctx.dir(self.hid) / "kb.json"
        if not p.exists():
            p.write_text(json.dumps(_kpc_kb()))

    def solve(self, task: Task, ctx: Ctx) -> str:
        kb = json.loads((ctx.dir(self.hid) / "kb.json").read_text())
        main = user(C.task_text(task), KPC_SYSTEM)
        resp = ctx.llm.ask(main, temperature=0, role="generate")
        ctx.keep(C.code_of(task, resp))
        while True:
            checker = user(f"{resp}\n{KPC_LIST}", KPC_SYSTEM)
            names = ctx.llm.ask(checker, temperature=0, role="checker")
            checker += [{"role": "assistant", "content": names}]
            apis = [m.replace(" ", "").split("(")[0] for m in re.findall(r"`(.+?)`", names)]
            todo = []
            for api in dict.fromkeys(a for a in apis if a in kb):
                for k in kb[api]:
                    q = KPC_CHECK.format(exc=k["throw_exception"], api=api)
                    ans = ctx.llm.ask(checker + [{"role": "user", "content": q}], temperature=0, role="checker")
                    checker += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
                    if ans.strip().lower().startswith("no"):
                        todo.append(KPC_FIX.format(cond=k["throw_condition"], api=api, exc=k["throw_exception"]))
            if not todo:
                return C.code_of(task, resp)
            main += [{"role": "assistant", "content": resp}, {"role": "user", "content": "\n".join(todo)}]
            resp = ctx.llm.ask(main, temperature=0, role="generate")
            ctx.keep(C.code_of(task, resp))


# ---------- H40 RAPGen ----------

RAPGEN_MIN_PROBLEMS, RAPGEN_TEMPERATURE = 2, 0.7
RAPGEN_PROMPT = """Complete the code below. Return the complete rewritten code in one ```{lang} block.

```{lang}
\"\"\"
{code}
\"\"\"
# {instruction}
{head}
```"""
_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d*)?|'[^']*'|\"[^\"]*\"|\S")


@lru_cache(maxsize=1)
def _common_identifiers() -> frozenset:
    import importlib
    s = set(keyword.kwlist) | set(dir(builtins)) | set(sys.stdlib_module_names)
    for t in (str, list, dict, set, tuple, int, float, bytes, frozenset):
        s |= set(dir(t))
    for m in ("sys", "math", "itertools", "collections", "heapq", "bisect", "functools", "re", "string", "operator", "io"):
        s |= set(dir(importlib.import_module(m)))
    return frozenset(x for x in s if not x.startswith("_"))


def _abstract(line: str) -> str:
    """RAPGen Algorithm 1 for Python: project identifiers and literals become placeholders; keywords and common APIs stay."""
    keep = _common_identifiers()
    out = []
    for t in _TOKEN.findall(line):
        if t[0].isalpha() or t[0] == "_":
            out.append(t if t in keep else "<N>")
        elif t[0].isdigit() or t[0] in "'\"":
            out.append("<N>")
        else:
            out.append(t)
    return " ".join(out)


def _rapgen_instruction(removed: list[str], added: list[str]) -> str | None:
    keep = _common_identifiers()
    ids = lambda ls: {t for l in ls for t in _TOKEN.findall(l) if t in keep and not keyword.iskeyword(t)}
    x, y = sorted(ids(added) - ids(removed)), sorted(ids(removed) - ids(added))
    if x and y:
        return f"PERF: Use {', '.join(x)} instead of {', '.join(y)} in the above method."
    if x:
        return f"PERF: Rewrite the above method with {', '.join(x)}."
    if y:
        return f"PERF: Rewrite the above method without {', '.join(y)}."
    return None


def _label(n: ast.AST) -> str:
    keep = _common_identifiers()
    name = getattr(n, "id", None) or getattr(n, "attr", None) or getattr(n, "arg", None)
    return type(n).__name__ + (f":{name}" if name in keep else ":#" if name else "")


def aroma(code: str) -> Counter:
    """Aroma structural features: token, parent-child and sibling features over the simplified parse tree."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    f: Counter = Counter()
    for node in ast.walk(tree):
        lab = _label(node)
        f[lab] += 1
        kids = list(ast.iter_child_nodes(node))
        for i, ch in enumerate(kids):
            f[f"{lab}>{i}>{_label(ch)}"] += 1
            if i:
                f[f"{_label(kids[i - 1])}~{_label(ch)}"] += 1
    return f


@h2h
class RAPGen(H2HTechnique):
    """Knowledge base of (abstracted buggy line -> before, after, instruction) mined from past performance fixes; the
    instruction of the single most Aroma-similar before-method whose pattern matches a line of the code is retrieved and
    the code is rewritten zero-shot. The C# fix commits are not released: the base is mined from PIE's Python pairs."""
    hid, name, family, target = "H40", "RAPGen", "retrieval", ("EFF",)
    ported = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        p = ctx.dir(self.hid) / "kb.jsonl"
        if p.exists():
            return
        rows = decontaminate([r for r in pie_python(ctx) if not r["code_same"] and r["improvement_frac"] > 0], "input", "target")
        entries = []
        for r in rows:
            a, b = r["input"].splitlines(), r["target"].splitlines()
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
                if op not in ("replace", "delete"):
                    continue
                ins = _rapgen_instruction(a[i1:i2], b[j1:j2])
                if ins:
                    for line in a[i1:i2]:
                        if line.strip():
                            entries.append({"key": _abstract(line.strip()), "instruction": ins, "before": r["input"],
                                            "after": r["target"], "problem_id": r["problem_id"]})
        spread: dict[str, set] = defaultdict(set)
        for e in entries:
            spread[e["key"]].add(e["problem_id"])
        write_jsonl(p, [e for e in entries if len(spread[e["key"]]) >= RAPGEN_MIN_PROBLEMS])

    @staticmethod
    @lru_cache(maxsize=2)
    def _kb(path: Path) -> dict[str, list[dict]]:
        kb: dict[str, list[dict]] = defaultdict(list)
        for e in read_jsonl(path):
            kb[e["key"]].append(e)
        return kb

    def solve(self, task: Task, ctx: Ctx) -> str:
        p0 = draft(task, ctx)
        kb = self._kb(ctx.dir(self.hid) / "kb.jsonl")
        cands = [e for k in dict.fromkeys(_abstract(l.strip()) for l in p0.splitlines() if l.strip()) for e in kb.get(k, [])]
        if not cands:
            return p0
        fq = aroma(p0)
        best = max(cands, key=lambda e: sum((fq & aroma(e["before"])).values()))
        fns = functions(p0)
        head = segment(p0, next(iter(fns.values()))).splitlines()[0] if task.mode == "function" and fns else ""
        prompt = RAPGEN_PROMPT.format(lang=task.language, code=p0, instruction=best["instruction"], head=head)
        out = C.code_of(task, ctx.llm.ask(user(prompt, C.SYSTEM), temperature=RAPGEN_TEMPERATURE, role="optimize"))
        return out or p0


# ---------- H41 CodeGuarder ----------

CODEGUARDER_ZIP = "https://zenodo.org/records/16957113/files/CodeGuarder_AE.zip?download=1"
CG_K_SUB, CG_K_PER, CG_DEFAULT_W = 5, 2, 0.01
CG_PREVALENCE = {w: cwes for w, cwes in [
    (0.4024, (391, 476, 690)), (0.2553, (120, 121, 122, 628, 676, 680, 787)), (0.1042, (822, 119)),
    (0.0886, (125, 129, 131, 193, 788)), (0.0621, (191, 20, 190, 192, 681)), (0.0503, (825, 401, 404, 459)),
    (0.0145, (369, 691))]}
CG_WEIGHT = {f"CWE-{c}": w for w, cs in CG_PREVALENCE.items() for c in cs}
CG_DECOMPOSE = """Please help me to break down a code generation query into smaller, detailed sub-tasks. For each sub-task, please utilize the natural language description of what the sub-tasks do, focusing on explaining the functionality of the sub-tasks.
# User's Query
{query}
# Output
Please output the following information in a JSON object format:
[{{"Description": "string"}}]"""
CG_PROMPT = """Your task is to generate secure code for the following functionality. Before you start writing the code, please review the security knowledge provided below, which includes potential vulnerabilities and how to fix them. Use this knowledge to avoid common security flaws and ensure the generated code is secure.
Besides, I would provide you with some reference examples, you can reference the examples as a reference if helpful.
### Security Knowledge:
{knowledge}
---

### Reference Code Examples:
**Example 1:**
 ```
{example}
```

### Task:
- Now, please generate the code for the following functionality:
{task}

### Notes:
- Ensure that the code you generate avoids the vulnerabilities described in the knowledge above.
- Pay attention to using secure patterns and avoiding insecure coding practices.
- If you are unsure about a security decision, refer to the fixing examples provided above for guidance.
- Only return the code, don't include any other information, such as a preamble or suffix."""


@h2h
class CodeGuarder(H2HTechnique):
    """Released ReposVul knowledge (functionality, vulnerable/fixing pattern) indexed by functionality with jina-v3; the
    query is decomposed into sub-tasks, k'=2 entries per sub-task, sub-tasks re-ranked by summed CWE prevalence and the
    top k=5 kept (paper defaults), plus the top-1 functional example (RACG setting)."""
    hid, name, family, target = "H41", "CodeGuarder", "retrieval", ("SEC",)
    code_url = "https://zenodo.org/records/16957113"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "knowledge.npy").exists():
            return
        z = fetch(CODEGUARDER_ZIP, d / "CodeGuarder_AE.zip")
        with zipfile.ZipFile(z) as f:
            vul = [json.loads(l) for l in f.read("CodeGuarder/dataset/ReposVul.jsonl").decode().splitlines() if l.strip()]
        know, code = [], []
        for r in vul:
            cwes = re.findall(r"CWE-\d+", str(r.get("cwe_id", "")))
            code.append({"code": r["fix"], "language": r["language"]})
            for e in json_of(r.get("root_cause") or "") or []:
                if isinstance(e, dict) and e.get("functionality"):
                    know.append({"cwe": cwes, "functionality": e["functionality"],
                                 "vulnerable_pattern": e.get("vulnerable_pattern", ""), "fixing_pattern": e.get("fixing_pattern", "")})
        know = decontaminate(know, "functionality", "vulnerable_pattern", "fixing_pattern")
        code = decontaminate(code, "code")
        save_index(d, know, embed("jina", [k["functionality"] for k in know]), "knowledge")
        save_index(d, code, embed("jina", [c["code"] for c in code]), "code")

    def solve(self, task: Task, ctx: Ctx) -> str:
        d = ctx.dir(self.hid)
        know, kv = load_index(d, "knowledge")
        code, cv = load_index(d, "code")
        query = C.task_text(task)
        subs = json_of(ctx.llm.ask(user(CG_DECOMPOSE.format(query=query)), temperature=0, role="decompose")) or []
        descs = [s.get("Description", "") for s in subs if isinstance(s, dict) and s.get("Description")] or [task.prompt]
        ranked = []
        for desc, q in zip(descs, embed("jina", descs)):
            hits = top_k(kv, q, CG_K_PER)
            w = sum(max([CG_WEIGHT.get(c, CG_DEFAULT_W) for c in know[i]["cwe"]] or [CG_DEFAULT_W]) for i in hits)
            ranked.append((w, hits))
        picked = list(dict.fromkeys(i for _, hits in sorted(ranked, key=lambda x: -x[0])[:CG_K_SUB] for i in hits))
        knowledge = "\n".join(json.dumps({k: know[i][k] for k in ("functionality", "vulnerable_pattern", "fixing_pattern")})
                              for i in picked)
        example = code[top_k(cv, embed("jina", [task.prompt])[0], 1)[0]]["code"]
        prompt = CG_PROMPT.format(knowledge=knowledge, example=example, task=query)
        return C.code_of(task, ctx.llm.ask(user(prompt), temperature=0, role="generate"))


# ---------- H42 Retrieval Augmented Search ----------

RAS_STEPS, RAS_K = 4, 8
RAS_DESCRIBE = ('You are an expert programmer who has been provided with a program solving a programming problem, called the '
                'source program. You need to identify the algorithm being used to solve the problem, and your goal is to '
                'generate a JSON object with the key "algorithm" which has the value as one sentence describing the '
                'algorithm used in the code snippet.')
RAS_OPTIMIZE = ('You are an expert programmer who needs to optimize a given program, called the source program. You are given '
                'one pair of fast and slow programs as an example, which are presented as a pair where "slower version" '
                'refers to the slow code and "optimized version" refers to the faster, more optimal version of the same '
                'program. The last program with the label "slower version" is the source program whose optimized version '
                'you need to generate. Rewrite the source program in a way that incorporates all of the optimizations in the '
                'example, and return a JSON-formatted string where the rewritten code is stored with the key '
                '"optimized_code". Do not output anything other than Python code.')


@h2h
class RAS(H2HTechnique):
    """Retrieval augmented search: each step the model describes the current program's algorithm, the 8 training pairs
    with the closest descriptions (not used before) each prompt one rewrite, and the fastest rewrite passing the public
    examples becomes the next program; 4 steps (36 calls published, so it runs at the ceiling). Corpus: the authors'
    Mercury pairs with their descriptions, re-embedded locally."""
    hid, name, family, target = "H42", "Retrieval Augmented Search", "retrieval", ("EFF",)
    code_url = "https://github.com/sagnikanupam/llmprogramoptimization"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "index.npy").exists():
            return
        z = gdrive(RAS_MERCURY, d / "mercury.zip")
        with zipfile.ZipFile(z) as f:
            tgt = json.loads(f.read("mercury_tgt_progs_pie.json"))
            rows = []
            for l in f.read("mercury_training_pairs_with_prog_descriptions_and_embeddings_pie.jsonl").decode().splitlines():
                r = json.loads(l) if l.strip() else {}
                if r.get("step_number") == 1 and r.get("program_key") in tgt:
                    rows.append({"src": r["src_prog"], "tgt": tgt[r["program_key"]], "description": r["src_program_description"]})
        rows = decontaminate(rows, "src", "tgt")
        save_index(d, rows, embed("qodo", [r["description"] for r in rows]))

    def solve(self, task: Task, ctx: Ctx) -> str:
        rows, vecs = load_index(ctx.dir(self.hid))
        p0 = draft(task, ctx)
        p, used = p0, set()
        for _ in range(RAS_STEPS):
            desc = json_of(ctx.llm.ask(user(p, RAS_DESCRIBE), role="describe")) or {}
            q = embed("qodo", [str(desc.get("algorithm", "")) if isinstance(desc, dict) else ""])[0]
            hits = top_k(vecs, q, RAS_K, frozenset(used))
            used |= set(hits)
            best, best_cost = None, None
            for i in hits:
                msg = (f"\n # slower version: \n{rows[i]['src']}\n# optimized version of the same code:\n{rows[i]['tgt']}"
                       f"\n # slower version: \n{p}\n# optimized version of the same code:\n")
                out = ctx.llm.ask(user(msg, RAS_OPTIMIZE), role="optimize")
                js = json_of(out)
                cand = js["optimized_code"] if isinstance(js, dict) and js.get("optimized_code") else C.code_of(task, out)
                c = cost(task, cand)
                if c is not None and (best_cost is None or c < best_cost):
                    best, best_cost = cand, c
                    ctx.keep(cand)
            if best is not None:
                p = best
        return p if cost(task, p) is not None else p0


# ---------- H43 SemOpt ----------

SEMOPT_KEYWORDS = (
    "performance", "optimize", "optimization", "speedup", "speed up", "speed-up", "fast", "faster", "fastest", "efficient",
    "efficiency", "throughput", "latency", "low-latency", "accelerate", "acceleration", "improve", "improvement", "enhance",
    "enhancement", "boost", "boosted", "tune", "tuning", "refactor for speed", "perf gain", "perf win", "reduce overhead",
    "reduce memory", "memory usage", "memory footprint", "memory consumption", "reduce allocation", "cache friendly",
    "cache efficiency", "cache hit", "cache miss", "inlining", "inline function", "loop unrolling", "vectorization",
    "vectorize", "simd", "parallelization", "parallelize", "multithreading", "multi-threading", "memoization",
    "lazy evaluation", "lock-free", "zero-copy", "hot path", "branch prediction", "prefetch", "bottleneck", "hotspot",
    "slow path", "critical path", "profile guided", "profiler", "complexity", "buffer reuse", "avoid copy",
    "avoid allocation", "reduce copy", "batch processing")
SEMOPT_KW = re.compile("|".join(re.escape(k) for k in SEMOPT_KEYWORDS), re.I)
SEMOPT_MAX_REPOS, SEMOPT_COMMITS_PER_REPO = 1000, 1000
SEMOPT_EPS, SEMOPT_MIN_CLUSTER = 0.85, 5
SEMOPT_COMMITS_PER_CLUSTER, SEMOPT_ATTEMPTS, SEMOPT_REPAIRS = 10, 5, 3
SEMOPT_TOP_PROPOSALS, SEMOPT_MAX_STRATEGIES = 25, 4
SEMGREP = ("semgrep", "scan", "--metrics=off", "--disable-version-check", "--json", "--quiet")
SEMOPT_FILTER_SYSTEM = """You are an expert in software performance optimization and Git commit analysis.
Your task is to determine whether a given Git commit meets ALL of the following criteria:
1. The primary purpose of the commit is performance optimization
   (specifically reducing runtime resource consumption such as execution time
   or memory usage)
2. The optimization technique used is relatively generic and transferable
   to other functions or even other codebases
3. The changes do not primarily focus on code readability, maintainability,
   or other non-performance-related improvements
Performance optimization must be the MAIN goal, and the optimization approach
should be something that could potentially be applied elsewhere (e.g., algorithmic
improvements, data structure optimizations, caching strategies, loop
optimizations, etc.).
You must answer strictly with "true" or "false":
- Answer "true" ONLY if the commit meets ALL three criteria above
- Answer "false" if the commit does not meet any of the criteria or if the
  optimization is too specific/context-dependent
Do not provide any explanation, reasoning, or additional text in your response.
Only return "true" or "false"."""
SEMOPT_FILTER_USER = """Here is the information for a Git commit:
Commit Message:
{message}
Git Diff:
{diff}
This commit only modifies one function in one file.
Based on the information provided above, does this commit meet all the following criteria:
1. Primary purpose is performance optimization (reducing runtime resource consumption)
2. Uses a relatively generic optimization technique that could be transferred
   to other functions/codebases
3. Not primarily focused on readability or maintainability improvements
Answer "true" only if ALL criteria are met, otherwise answer "false"."""
SEMOPT_SUMMARY = """Please analyze the following commit that implements performance optimization.
Your task is to provide a ONE-SENTENCE summary in English describing an
optimization strategy that can be applied to similar code patterns.
The summary must be exactly one sentence - no more, no less.
IMPORTANT: If the optimization matches one of the common patterns below,
use the provided template summary. Otherwise, create a summary following
the same style.
COMMON OPTIMIZATION PATTERNS AND THEIR SUMMARIES:
1. Changing 'container.size() == 0' to 'container.empty()' -->
  Replace size() == 0 checks with empty() method calls for better
  performance and readability.
2. Reordering conditions in if statements with multiple 'and'-connected
   sub-conditions -->
  Reorder sub-conditions in if statements with multiple 'and'-connected
  conditions to place simpler or less expensive checks before more complex
  ones for better short-circuit evaluation.
3. Caching repeated calculations -->
  Cache frequently computed values to avoid redundant calculations in loops
  or repeated calls.
GUIDELINES for creating optimization summaries:
1. Include the main code structure or pattern where the optimization applies
   (e.g., 'in for loops', 'when using containers', 'in recursive functions')
2. Describe the specific change made within that code context
3. Include relevant technical details that are important for applying
   the optimization
4. Make the description specific enough to identify similar code patterns
   but general enough to apply across different projects
EXAMPLE of including code context:
- If a commit changes from pass-by-value to pass-by-reference in for loop
  iteration: Use pass-by-reference instead of pass-by-value when iterating
  over containers in for loops to avoid unnecessary object copying.
- If a commit optimizes string concatenation in loops: Use string builders
  or pre-allocated buffers instead of repeated string concatenation operations
  within loops to reduce memory allocations.
- If a commit changes vector push_back to reserve: Pre-allocate container
  capacity using reserve() before adding multiple elements to avoid repeated
  memory reallocations during insertion.
This commit has already been confirmed as a performance optimization.
Analyze the code changes and create a summary that includes both the code
context and the specific optimization technique.
Repository: {repo}
Commit Hash: {sha}
Commit Message: {message}
Modified Files: {path}
Modified Functions: {func}
Code Changes:
{diff}
Please provide your response as exactly one sentence that describes both the
code context (where to apply) and the optimization strategy (what to change)."""
SEMOPT_RULE_SYSTEM = """You are an expert in Python program analysis and Semgrep rule creation.
Your task is to identify performance optimization patterns in code changes
and create Semgrep rules that can detect similar optimization opportunities
in other codebases.
When writing Semgrep rules for Python, please ensure that:
1. The rule follows proper Semgrep YAML format with correct metadata
   including 'id', 'message', 'languages', 'severity', and 'pattern' fields.
2. The rule must support Python with 'languages: [python]' in the metadata.
3. The rule should detect patterns where code could be optimized similar
   to the provided diff.
4. The rule should focus on performance improvements rather than
   security issues.
5. Include helpful messages that explain the performance impact and
   suggested improvements.
6. Ensure that the generated YAML code is enclosed within triple backticks
   with 'yaml' language specification (```yaml).
7. The pattern should be syntactically correct for Python and follow Semgrep
   pattern syntax.
8. Use appropriate Semgrep metavariables (like $VAR, $EXPR, $STMT) to make
   patterns generic.
9. Include severity level appropriate for performance optimizations
   (usually 'info' or 'warning')."""
SEMOPT_RULE_STEPS = (
    """What performance optimization pattern does the following diff implement?
Analyze the changes and identify the optimization technique being applied.
Do not write Semgrep rules at this time, just analyze the diff and explain
the performance improvement.
Diff content:
{diff}""",
    """Write a Semgrep rule for Python that can detect similar optimization opportunities
in other codebases. The rule should identify code patterns that could benefit from
the same optimization technique shown in the diff. Focus on creating a rule that
helps developers find performance improvement opportunities. Make sure the rule
follows proper Semgrep YAML format, uses correct Python pattern syntax, and supports
Python with 'languages: [python]'.""")
SEMOPT_FIX_EXEC = """There was an execution error when running the Semgrep rule:
Error: {error}
The YAML rule you provided:
{yaml}
Please fix the Semgrep rule to resolve this execution error.
Provide a complete, corrected YAML rule that can run successfully."""
SEMOPT_FIX_SEMANTIC = """The Semgrep rule execution completed but reported serious errors:
Full JSON result:
{result}
The YAML rule you provided:
{yaml}
Please analyze the errors and fix the Semgrep rule. Focus on:
1. Correct YAML syntax and structure
2. Valid Python pattern syntax for Semgrep
3. Proper use of metavariables
4. Correct rule metadata
Provide a complete, corrected YAML rule that addresses all reported errors."""
SEMOPT_OPTIMIZE = """Please optimize the following Python function based on the identified code
segment and reference optimization strategies.
FUNCTION WITH LINE NUMBERS:
```python
{numbered}
```
TARGET CODE SEGMENT TO OPTIMIZE:
Lines {start} to {end}
REFERENCE OPTIMIZATION STRATEGIES:
Note: The following strategies may contain duplicates or overlapping suggestions.
{strategies}
ORIGINAL FUNCTION TO OPTIMIZE:
```python
{func}
```
REQUIREMENTS:
1. First, identify the specific code segment that needs optimization based on
   the given line numbers ({start}-{end})
2. Carefully analyze and understand each reference optimization strategy provided
3. Determine if each strategy is applicable to the identified target code segment
4. Verify that applying each strategy will preserve the original semantics
   and correctness
5. Only apply strategies that are both applicable and semantically safe
6. Maintain exact functionality throughout the entire function
7. Your response must contain exactly ONE code block with the complete
   optimized function. You may provide explanations and analysis in your
   response, but ensure there is ONLY ONE code block containing the entire
   optimized function."""


def run_semgrep(config: Path, code: str) -> dict:
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "target.py"
        f.write_text(code)
        p = subprocess.run([*SEMGREP, "--config", str(config), str(f)], capture_output=True, text=True, timeout=600)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"fatal": (p.stderr or p.stdout)[-4000:], "results": [], "errors": []}


def dbscan(vecs: np.ndarray, eps: float, min_samples: int) -> list[int]:
    """DBSCAN over cosine similarity (neighbours: similarity >= eps); -1 marks noise."""
    sim = vecs @ vecs.T
    nb = [np.flatnonzero(sim[i] >= eps) for i in range(len(vecs))]
    labels, c = [-2] * len(vecs), -1
    for i in range(len(vecs)):
        if labels[i] != -2:
            continue
        if len(nb[i]) < min_samples:
            labels[i] = -1
            continue
        c += 1
        labels[i], stack = c, list(nb[i])
        while stack:
            j = int(stack.pop())
            if labels[j] == -1:
                labels[j] = c
            if labels[j] != -2:
                continue
            labels[j] = c
            if len(nb[j]) >= min_samples:
                stack.extend(nb[j])
    return labels


@h2h
class SemOpt(H2HTechnique):
    """Strategy library from single-function optimization commits (keyword match, LLM filter, three one-sentence summaries
    with the most central kept, DBSCAN clusters as strategies), LLM-written Semgrep rules per sampled commit (5 attempts,
    3 repair rounds, kept if they run without error); at test time every rule runs on the code, proposals are merged and
    ranked by count, and one call per target segment applies its top-4 matched strategies. The figshare library is not
    reachable: the Python library is rebuilt by the paper's recipe."""
    hid, name, family, target = "H43", "SemOpt", "retrieval", ("EFF",)
    code_url = "https://figshare.com/s/0391db49a3620ed53a1f"
    data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "rules.yaml").exists():
            return
        commits = self._mine(d)
        with prep(self.hid):
            kept = [c for c in commits if ctx.llm.ask(user(SEMOPT_FILTER_USER.format(message=c["message"], diff=c["diff"]),
                                                            SEMOPT_FILTER_SYSTEM), temperature=0).strip().lower().startswith("true")]
            for c in kept:
                sums = [x.strip() for x in ctx.llm.sample(user(SEMOPT_SUMMARY.format(**c), "You are a helpful assistant."), 3)]
                v = embed("minilm", sums)
                c["strategy"] = sums[int(np.argmax((v @ v.T).mean(1)))]
        kept = decontaminate(kept, "before", "after")
        labels = dbscan(embed("minilm", [c["strategy"] for c in kept]), SEMOPT_EPS, SEMOPT_MIN_CLUSTER)
        write_jsonl(d / "library.jsonl", [dict(c, cluster=l) for c, l in zip(kept, labels) if l >= 0])
        rules, rng = [], random.Random(0)
        with prep(self.hid):
            for cl in sorted({l for l in labels if l >= 0}):
                members = [c for c, l in zip(kept, labels) if l == cl]
                for c in rng.sample(members, min(SEMOPT_COMMITS_PER_CLUSTER, len(members))):
                    for _ in range(SEMOPT_ATTEMPTS):
                        rule = self._rule(ctx, d, c)
                        if rule:
                            rule.update(id=f"semopt-{len(rules)}", metadata={"repo": c["repo"], "sha": c["sha"], "cluster": int(cl)})
                            rules.append(rule)
        import yaml
        (d / "rules.yaml").write_text(yaml.safe_dump({"rules": rules}, sort_keys=False))

    @staticmethod
    def _mine(d: Path) -> list[dict]:
        p = d / "commits.jsonl"
        if p.exists():
            return read_jsonl(p)
        out = []
        with github() as gh:
            repos = []
            for page in range(1, SEMOPT_MAX_REPOS // 100 + 1):
                r = gh.get("/search/repositories", params={"q": "language:python stars:>=100", "sort": "stars", "per_page": 100,
                                                           "page": page})
                repos += [x["full_name"] for x in r.json().get("items", [])]
            for repo in repos:
                for page in range(1, SEMOPT_COMMITS_PER_REPO // 100 + 1):
                    cs = gh.get(f"/repos/{repo}/commits", params={"per_page": 100, "page": page}).json()
                    if not isinstance(cs, list) or not cs:
                        break
                    for c in cs:
                        msg = c["commit"]["message"]
                        if not SEMOPT_KW.search(msg):
                            continue
                        det = gh.get(f"/repos/{repo}/commits/{c['sha']}").json()
                        files = det.get("files") or []
                        if len(files) != 1 or files[0].get("status") != "modified" or not files[0]["filename"].endswith(".py") \
                                or len(det.get("parents") or []) != 1:
                            continue
                        path = files[0]["filename"]
                        raw = "https://raw.githubusercontent.com/{}/{}/{}"
                        before = gh.get(raw.format(repo, det["parents"][0]["sha"], path)).text
                        after = gh.get(raw.format(repo, c["sha"], path)).text
                        ch = single_function_change(before, after)
                        if ch:
                            out.append({"repo": repo, "sha": c["sha"], "message": msg, "path": path, "func": ch[0],
                                        "before": ch[1], "after": ch[2], "before_file": before,
                                        "diff": files[0].get("patch", "")})
        write_jsonl(p, out)
        return out

    @staticmethod
    def _rule(ctx: Ctx, d: Path, c: dict) -> dict | None:
        import yaml
        msgs = user(SEMOPT_RULE_STEPS[0].format(diff=c["diff"]), SEMOPT_RULE_SYSTEM)
        msgs += [{"role": "assistant", "content": ctx.llm.ask(msgs, temperature=0)},
                 {"role": "user", "content": SEMOPT_RULE_STEPS[1]}]
        for _ in range(SEMOPT_REPAIRS + 1):
            reply = ctx.llm.ask(msgs, temperature=0)
            text = (re.findall(r"```yaml\s*(.*?)```", reply, re.S) or [reply])[-1]
            path = d / "candidate.yaml"
            path.write_text(text)
            res = run_semgrep(path, c["before_file"])
            if not res.get("fatal") and not res.get("errors"):
                try:
                    return (yaml.safe_load(text) or {}).get("rules", [None])[0]
                except yaml.YAMLError:
                    return None
            fix = SEMOPT_FIX_EXEC.format(error=res["fatal"], yaml=text) if res.get("fatal") else \
                SEMOPT_FIX_SEMANTIC.format(result=json.dumps(res)[:20000], yaml=text)
            msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": fix}]
        return None

    def solve(self, task: Task, ctx: Ctx) -> str:
        p0 = draft(task, ctx)
        res = run_semgrep(ctx.dir(self.hid) / "rules.yaml", p0)
        counts: Counter = Counter()
        for r in res.get("results", []):
            meta = r.get("extra", {}).get("metadata", {})
            counts[(r["extra"].get("message", ""), meta.get("repo", ""), meta.get("sha", ""), r["start"]["line"], r["end"]["line"])] += 1
        segments: dict[tuple[int, int], list] = defaultdict(list)
        for (msg, repo, sha, s, e), n in counts.most_common(SEMOPT_TOP_PROPOSALS):
            segments[(s, e)].append((msg, repo, sha))
        best, best_cost = p0, cost(task, p0)
        for (s, e), strats in segments.items():
            func, offset = self._enclosing(p0, s, e)
            numbered = "\n".join(f"{offset + i + 1}: {l}" for i, l in enumerate(func.splitlines()))
            block = "\n".join(f"Strategy {i + 1}:\nOptimization suggestion: {m}\nSource: {r}:{h}"
                              for i, (m, r, h) in enumerate(strats[:SEMOPT_MAX_STRATEGIES]))
            out = C.code_of(task, ctx.llm.ask(user(SEMOPT_OPTIMIZE.format(numbered=numbered, start=s, end=e, strategies=block,
                                                                              func=func)), temperature=0, role="optimize"))
            cand = (p0.replace(func, out, 1) if func != p0 else out) if out else ""
            c = cost(task, cand)
            if c is not None and (best_cost is None or c < best_cost):
                best, best_cost = ctx.keep(cand), c
        return best

    @staticmethod
    def _enclosing(code: str, s: int, e: int) -> tuple[str, int]:
        inner = [n for n in functions(code).values() if n.lineno <= s and e <= n.end_lineno]
        if not inner:
            return code, 0
        n = min(inner, key=lambda n: n.end_lineno - n.lineno)
        start = min([n.lineno] + [x.lineno for x in n.decorator_list])
        return segment(code, n), start - 1


# ---------- H44 AutoPatch ----------

OSV_PYPI = "https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip"
AUTOPATCH_SINCE, AUTOPATCH_MAX_CVES = "2025-02-01", 75
AUTOPATCH_TAGS = "https://raw.githubusercontent.com/ai-llm-research/autopatch/main/src/rag_db/programming_tags.csv"
AUTOPATCH_WEIGHTS = (58.7924, 35.4870, 51.2777, 8.0147)
AUTOPATCH_MAP_THRESHOLD, AUTOPATCH_FUZZ, AUTOPATCH_RETRIES = 0.5, 0.80, 3
AP_JSON = 'in valid JSON format with the following schema, including the leading and trailing "```json" and "```"\n```json\n{schema}\n```'
AP_SEMANTIC = """You are an expert software engineer without any software security knowledge. Your goal is to analyze [Target Code] and provide a self-contained summary of its functionality. Perform the followings step by step and show the reasoning in each step. You are not aware of software security information so DO NOT deduce any security implication on any step. Start answering with "Let's think step-by-step."
1) Analyze the main functionality of [Target Code].
2) Explain the main functionality of [Target Code] in a self-contained low-level representation. The explanation must be general that it must not include any variable or function names.
3) Finally, provide the seld-contained explanation of [Target Code] """ + AP_JSON.format(schema='{\n        "result": string // the self-contained explanation of [Target Code] (Step 2)\n}')
AP_SEMANTIC_SHOT = ("[Supplementary Code]\n```python\ndef double_if_odd(x):\n    if x % 2 != 0:\n        return x * 2\n    return x\n```\n\n"
                    "[Target Code]\n```python\ndef process_array(arr):\n    result = 0\n    for v in arr:\n        result += double_if_odd(v)\n    return result\n```")
AP_SEMANTIC_ANSWER = """Let's think step-by-step.

Step 1: Analyze the main functionality of [Target Code].
- The function iterates through the list.
- For each element, it applies double_if_odd() and accumulates the result.
- Finally, it returns the sum of all processed elements.

Step 2: Explain the main functionality of [Target Code] in a self-contained low-level representation. The explanation must be general that it must not include external references.
- Iterate through a list of numbers.
- For each number, check if it is odd. If it is, double it; otherwise, leave it as is.
- Sum all the processed numbers and return the final total.

Step 3: Final result in JSON format.
```json
{"result": "Iterate through a list of numbers. For each number, check if it is odd. If it is, double it; otherwise, leave it as is. Sum all the processed numbers and return the final total."}
```"""
_AP_TAINT = """You are an expert software engineer. Your goal is to perform taint-analysis on each of the {kind}s referenced by the user-provided [Target Code] and provide entirely self-contained explanations of the {kind}s' functionalities in [Target Code]. Perform the followings step by step and show the reasoning in each step. Start answering with "Let's think step-by-step."
1) Extract all the referenced {kind}s within [Target Code].
- At least the {kind}s in [Data Flow] MUST be extracted.
2) For each extracted {kind}, trace its **data flow** within [Target Code].
- Use the [Data Flow] (format: "source variable/function" => destination variable/function list) to track {track}.
3) For each {kind}, generate a **low-level, self-contained explanation** of its functionality.
- The explanation MUST include:
{include}
- The explanation MUST NOT reference specific variable/function names. Instead, describe their roles in **abstract technical terms** (e.g., "a memory buffer holding intermediate graphical state," "a counter that governs iteration termination").
- Each explanation MUST be **self-contained** so it can be understood in isolation, without looking at [Target Code].
4) Finally, provide the self-contained {kind} functionality explanations """
AP_TAINT_VAR = _AP_TAINT.format(
    kind="variable", track="how the variable is initialized, transformed, passed to functions, or conditionally manipulated",
    include="    - The origin of the variable (input, derived from another variable, returned from a function, etc.).\n"
            "    - The operations performed on it (arithmetic, logical checks, memory management, iteration, dereferencing, etc.).\n"
            "    - How it interacts with other variables or functions (dependencies, propagation, transformations).\n"
            "    - The final role or outcome (what state it contributes to, what it enables, what risk it introduces).") + \
    AP_JSON.format(schema='{\n        "<variable_1_name>": string,  // the self-contained explanation of variable_1\n        ...\n'
                          '        "<variable_n_name>": string  // the self-contained explanation of variable_n\n}')
AP_TAINT_FUNC = _AP_TAINT.format(
    kind="function", track="how the function is used, how its output is propagated, and how it interacts with other variables "
                           "or functions.\n- If the function is in [Supplementary Code], you can use it to understand the data flow",
    include="    - The role of the function's inputs (where they originate and how they are validated or transformed).\n"
            "    - The internal operations (e.g., logical branching, arithmetic, memory management, data structure manipulation, iteration, synchronization).\n"
            "    - How the function interacts with external state or other components (e.g., modifies buffers, updates counters, signals errors).\n"
            "    - The function's final outcome (e.g., initializes a resource, validates conditions, propagates data, releases memory, introduces risks).") + \
    AP_JSON.format(schema='{\n        "<function_1_name>": string,  // the self-contained explanation of function_1\n        ...\n'
                          '        "<function_n_name>": string  // the self-contained explanation of function_n\n}')
AP_DB_TAINT = """You are an expert software engineer. Your goal is to perform taint-analysis on each of the {kind}s in [Target {Kind}s] and provide entirely self-contained explanations of the {kind}s' functionalities in [Target Code]. Use [Data Flow] (format: "source variable/function" => destination variable/function list) to trace each one. The explanation MUST NOT reference specific variable/function names, and MUST be self-contained. Start answering with "Let's think step-by-step." Finally, provide the explanations """
AP_DB_VERIFY = """You are an expert software security engineer. Your task is to provide the checklist to identify vulnerabilities similar to the user-provided [CVE] that occurred in [Vulnerable Code]. Perform the following steps in order and show the reasoning in each step. Start your answer with "Let's think step-by-step."
1) Identify the patched lines of [CVE] by comparing [Vulnerable Code] and [Patched Code]. Then, determine the root cause of [CVE].
2) Based on the root cause from Step 1, identify the variables and functions that are crucial for analyzing [CVE]. These variables/functions are the key vulnerability-related elements needed for the analysis.
3) Given that a user can match each identified variable and function from Step 2 to a counterpart in their own code snippet, provide the checklist to verify whether the snippet is vulnerable to the same [CVE].
    - Important: Present each checklist item in terms of the identified vulnerability-related variables and functions from Step 2 (e.g., "Check if function <function_name> is safely handling variable <variable_name>" or "Verify that variable <variable_name> is properly sanitized before it is passed to function <function_name>").
4) Using the checklist from Step 3, describe, step-by-step, why the vulnerability similar to [CVE] exists in the [Vulnerable Code].
5) Using the same checklist from Step 3, describe, step-by-step, why the vulnerability similar to [CVE] doesn't exist in the [Patched Code].
6) Finally, provide the results in the following schema, including the leading and trailing "```json" and "```"
```json
{
        "root_cause": string, // the root cause of the vulnerability of [CWE] wihtin [Target Code] (Step 1)
        "vulnerability_related_variables": array,  // the list of vulnerability-related variables (Step 2)
        "vulnerability_related_functions": array,  // the list of vulnerability-related functions (Step 2)
        "vulnerability_checklist": string, // the checklist to identify the vulnerabilities similar to [CVE] (Step 3)
        "cot": string, // the checklist checking process to determine why [Vulnerable Code] is vulnerable to [CVE] (Step 4)
        "safe_cot": string // the checklist checking process to determine why [Patched Code] is not vulnerable to [CVE] (Step 5)
}
```"""
AP_DB_PATCH = """You are an expert software security engineer. Your task is to provide the fix-list to patch vulnerabilities similar to the user-provided [CVE] that occurred in [Vulnerable Code]. Perform the following steps in order and show the reasoning in each step. Start your answer with "Let's think step-by-step."
1) Given the user-provided [Root Cause] and [Patched Code] of [CVE], analyze how the patched lines in [Patched Code] fixes [CVE].
2) Given that a user can match each of [Vulnerability-Related Variables] and [Vulnerability-Related Functions] to a counterpart in their own code snippet, which is vulnerable to the same [CVE], provide the fix-list to patch the snippet.
    - Important: Present each fix-list item in terms of [Vulnerability-Related Variables] and [Vulnerability-Related Functions] (e.g., "Check if function <function_name> is safely handling variable <variable_name>" or "Verify that variable <variable_name> is properly sanitized before it is passed to function <function_name>").
3) Using the fix-list from Step 2, describe, step-by-step, how to patch the vulnerability similar to [CVE] that exists in the [Vulnerable Code].
4) Finally, provide the results in the following schema, including the leading and trailing "```json" and "```"
```json
{
        "fix_list": string, // the fix-list to patch the snippet that is vulnerable to the same [CVE] (Step 2)
        "cot": string // the thinking process to patch [Vulnerable Code] (Step 3)
}
```"""
AP_VERIFY = """You are an expert software security engineer. Your goal is to analyse user-provided [Target Code] and verify if a vulnerability of {cwe} similar to {cve} exists. To verify if a vulnerability similar to {cve} exists, you will mainly focus on the following [Checklist for {cve}], where mapping from actual variables to each symbolic variables/functions in [Checklist for {cve}] should be provided by user with [Variable Mapping] and [Function Mapping].

[Checklist for {cve}]
{checklist}

Perform the followings step by step and show the reasoning in each step. Start answering with "Let's think step-by-step."
1) Based on [Checklist for {cve}] along with the user-provided [Variable Mapping] and [Function Mapping], verify if {cwe} similar to {cve} exists in [Target Code]
2) Based on the vulnerability verification process from Step 1, identify the root cause of {cwe} wihtin [Target Code].
3) Provide the results in the following schema, including the leading and trailing "```json" and "```"
```json
{{
        "result": boolean,  // the result of vulnerability verification (true = the vulnerability exists, false = the vulnerability does not exist)
        "cot" :  the thinking process for the vulnerability verification (Step 1)
        "root_cause": string // the root cause of the vulnerability of {cwe} wihtin [Target Code] (Step 2)
}}
```"""
AP_VERIFY_SHOT = """Now, I will analyse user-provided [Target Code] and verify if a vulnerability of {cwe} similar to {cve} exists. Then, I will provide the results.

Let's think step-by-step.

Step 1. Verify if [Target Code] has {cwe} similar to {cve}.
{cot}

Step 2. Identify the root cause of {cwe} wihtin [Target Code].
{root}

Step 3. Provide the results.
```json
{{"result": true, "cot": {cot_json}, "root_cause": {root_json}}}
```"""
AP_PATCH = """You are an expert software security engineer. Your goal is to patch user-provided [Target Code] having a vulnerabilitiy of {cwe} similar to {cve}. To patch a vulnerability similar to {cve}, you will mainly focus on the following [Fix-List for {cve}], where mapping from actual variables to each symbolic variables/functions in [Fix-List for {cve}] should be provided by user with [Variable Mapping] and [Function Mapping].

[Fix-List for {cve}]
{fix_list}

Perform the followings step by step and show the reasoning in each step. Start answering with "Let's think step-by-step."
1) Based on [Fix-List for {cve}] along with the user-provided [Variable Mapping] and [Function Mapping], describe how to patch [Target Code] for fixing {cwe} similar to {cve}.
2) Use the patch description from Step 1 to generate a patched code.
3) Provide the results in the following format, including the leading and trailing "```" and "```"
```
[CoT START]
<the thinking process for the vulnerability patching (Step 1)>
[CoT END]

[Patched Code START]
<the patched code>
[Patched Code END]
```"""
AP_PATCH_SHOT = """Now, I will patch user-provided [Target Code] having a vulnerabilitiy of {cwe} similar to {cve}. Then, I will provide the results.

Let's think step-by-step.

Step 1. Describe how to patch [Target Code] to fix {cwe} similar to {cve}.
{cot}

Step 2. Generate a patched code based on Step 1.
{patch}

Step 3. Provide the results.
```
[CoT START]
{cot}
[CoT END]

[Patched Code START]
{patch}
[Patched Code END]
```"""
AP_CONTEXT = "[Supplementary Code]\n{supp}\n\n[Variable Mapping]\n{vmap}\n\n[Function Mapping]\n{fmap}\n\n{extra}[Target Code]\n{code}"


def data_flow(code: str) -> str:
    """Source => destinations over assignments, calls and returns (the CodeQL data-flow query, on the Python AST)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    flow: dict[str, set] = defaultdict(set)
    names = lambda n: {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} | \
        {x.func.attr if isinstance(x.func, ast.Attribute) else getattr(x.func, "id", "") for x in ast.walk(n) if isinstance(x, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and n.value is not None:
            dst = {x.id for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) for x in ast.walk(t) if isinstance(x, ast.Name)}
            for s in names(n.value):
                flow[s] |= dst
        elif isinstance(n, ast.Call):
            f = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for a in n.args + [k.value for k in n.keywords]:
                for s in names(a):
                    flow[s].add(f)
        elif isinstance(n, ast.Return) and n.value is not None:
            for s in names(n.value):
                flow[s].add("<return>")
    return "\n".join(f'"{s}" => {sorted(d - {s})}' for s, d in flow.items() if s and d - {s})


def _supplementary(file_src: str, func: str) -> str:
    """Functions of the same file that the target calls (the relevant part of AutoPatch's supplementary code)."""
    fs = functions(file_src)
    called = {getattr(n.func, "attr", getattr(n.func, "id", "")) for n in ast.walk(ast.parse(func)) if isinstance(n, ast.Call)}
    return "\n\n".join(segment(file_src, fs[k]) for k in sorted(called & set(fs)))


class _AutoPatchStore:
    def __init__(self, ctx: Ctx, d: Path):
        self.ctx, self.d = ctx, d

    def ask_json(self, msgs: list[Message]) -> dict | None:
        for _ in range(AUTOPATCH_RETRIES):
            out = json_of(self.ctx.llm.ask(msgs, temperature=0, role="autopatch"))
            if isinstance(out, dict):
                return out
        return None


@lru_cache(maxsize=1)
def _programming_tags(path: Path) -> tuple[str, ...]:
    import csv
    return tuple(r["TagName"] for r in csv.DictReader(io.StringIO(fetch(AUTOPATCH_TAGS, path).read_text())))


def _keywords(text: str, tags: tuple[str, ...]) -> set[str]:
    words = {w for w in re.findall(r"[a-z][a-z+#.-]+", text.lower())}
    out = set()
    for w in words:
        m = difflib.get_close_matches(w, tags, n=1, cutoff=AUTOPATCH_FUZZ)
        if m:
            out.add(m[0])
    return out


def _fuzzy_jaccard(a: set[str], b: set[str]) -> float:
    def reduce(s):
        out: list[str] = []
        for w in sorted(s):
            if not any(difflib.SequenceMatcher(None, w, o).ratio() >= AUTOPATCH_FUZZ for o in out):
                out.append(w)
        return out
    ra, rb = reduce(a), reduce(b)
    inter = sum(1 for x in ra if any(difflib.SequenceMatcher(None, x, y).ratio() >= AUTOPATCH_FUZZ for y in rb))
    union = len(ra) + len(rb) - inter
    return inter / union if union else 0.0


def _mapping(user_vecs: dict[str, np.ndarray], rag_vecs: dict[str, list]) -> dict[str, tuple[str, float]]:
    out = {}
    for name, v in rag_vecs.items():
        best = max(((u, float(np.dot(np.asarray(v), uv))) for u, uv in user_vecs.items()), key=lambda x: x[1], default=None)
        if best and best[1] > AUTOPATCH_MAP_THRESHOLD:
            out[name] = best
    return out


@h2h
class AutoPatch(H2HTechnique):
    """Similarity analyzer (taint analysis of variables and functions, semantic summary, combined learned score over a CVE
    database), vulnerability verifier with the matched CVE's checklist as a one-shot, and patcher with its fix-list. The
    released database is 75 C/C++ CVEs: a Python database of post-cutoff PyPI CVEs (OSV, single-function GitHub fixes) is
    built by the authors' database prompts; data flow comes from the Python AST instead of CodeQL."""
    hid, name, family, target = "H44", "AutoPatch", "retrieval", ("SEC",)
    code_url = "https://github.com/ai-llm-research/autopatch"
    ported = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "db.jsonl").exists():
            return
        cwe = cwe_catalog(ctx.dir("cwe"))
        tags = _programming_tags(d / "programming_tags.csv")
        z = fetch(OSV_PYPI, d / "osv-pypi.zip")
        cases = []
        with zipfile.ZipFile(z) as f, github() as gh:
            for name in f.namelist():
                v = json.loads(f.read(name))
                cve = next((a for a in v.get("aliases", []) + [v["id"]] if a.startswith("CVE-")), None)
                cwes = (v.get("database_specific") or {}).get("cwe_ids") or []
                if not cve or not cwes or v.get("published", "") < AUTOPATCH_SINCE:
                    continue
                for ref in v.get("references", []):
                    m = re.match(r"https://github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{7,40})", ref.get("url", ""))
                    if not m:
                        continue
                    det = gh.get(f"/repos/{m.group(1)}/commits/{m.group(2)}").json()
                    pys = [x for x in det.get("files") or [] if x["filename"].endswith(".py") and x.get("status") == "modified"
                           and "test" not in x["filename"].lower()]
                    if len(pys) != 1 or len(det.get("parents") or []) != 1:
                        continue
                    raw = "https://raw.githubusercontent.com/{}/{}/{}"
                    before = gh.get(raw.format(m.group(1), det["parents"][0]["sha"], pys[0]["filename"])).text
                    after = gh.get(raw.format(m.group(1), det["sha"], pys[0]["filename"])).text
                    ch = single_function_change(before, after)
                    if ch:
                        cases.append({"cve_id": cve, "cwe_id": cwes[0], "cwe_type": cwe.get(cwes[0], {}).get("name", cwes[0]),
                                      "function_name": ch[0], "original_code": ch[1], "vuln_patch": ch[2],
                                      "supplementary_code": _supplementary(before, ch[1]) or "None"})
                        break
                if len(cases) >= AUTOPATCH_MAX_CVES * 2:
                    break
        cases = decontaminate(cases, "original_code", "vuln_patch")[:AUTOPATCH_MAX_CVES]
        st, db = _AutoPatchStore(ctx, d), []
        with prep(self.hid):
            for c in cases:
                body = f"[Supplementary Code]\n{c['supplementary_code']}\n\n[Target Code]\n{c['original_code']}"
                sem = st.ask_json(user(body, AP_SEMANTIC))
                ver = st.ask_json(user(f"[CVE]\n{c['cve_id']}\n\n[CWE]\n{c['cwe_type']}\n\n{body}\n\n[Patched Code]\n{c['vuln_patch']}",
                                       AP_DB_VERIFY))
                if not sem or not ver:
                    continue
                funcs = [f for f in ver.get("vulnerability_related_functions", []) if f != c["function_name"]]
                flow = data_flow(c["original_code"])
                tv = st.ask_json(user(f"{body}\n\n[Data Flow]\n{flow}\n\n[Target Variables]\n{ver.get('vulnerability_related_variables', [])}",
                                      AP_DB_TAINT.format(kind="variable", Kind="Variable") + AP_JSON.format(schema='{"<variable_name>": string}'))) \
                    if ver.get("vulnerability_related_variables") else {}
                tf = st.ask_json(user(f"{body}\n\n[Data Flow]\n{flow}\n\n[Target Functions]\n{funcs}",
                                      AP_DB_TAINT.format(kind="function", Kind="Function") + AP_JSON.format(schema='{"<function_name>": string}'))) \
                    if funcs else {}
                pat = st.ask_json(user(f"[CWE]\n{c['cwe_type']}\n\n{body}\n\n[Vulnerability Root Cause]\n{ver['root_cause']}\n\n"
                                       f"[Vulnerability-Related Functions]\n{funcs}\n\n[Vulnerability-Related Variables]\n"
                                       f"{ver.get('vulnerability_related_variables', [])}\n\n[Patched Code]\n{c['vuln_patch']}", AP_DB_PATCH))
                if not pat:
                    continue
                semantics = str(sem.get("result", ""))
                db.append(c | {"code_semantics": {"raw": semantics, "vec": embed("qodo", [semantics])[0].tolist(),
                                                  "ner": sorted(_keywords(semantics, tags))},
                               "verification_cot": str(ver.get("cot", "")), "vulnerability_checklist": str(ver.get("vulnerability_checklist", "")),
                               "root_cause": str(ver["root_cause"]), "patch_cot": str(pat.get("cot", "")), "fix_list": str(pat.get("fix_list", "")),
                               "vulnerability_related_variables": {k: {"raw": str(s), "vec": embed("qodo", [str(s)])[0].tolist()}
                                                                   for k, s in (tv or {}).items()},
                               "vulnerability_related_functions": {k: {"raw": str(s), "vec": embed("qodo", [str(s)])[0].tolist()}
                                                                   for k, s in (tf or {}).items()}})
        write_jsonl(d / "db.jsonl", db)

    def solve(self, task: Task, ctx: Ctx) -> str:
        d = ctx.dir(self.hid)
        db, st = read_jsonl(d / "db.jsonl"), _AutoPatchStore(ctx, d)
        tags = _programming_tags(d / "programming_tags.csv")
        code = draft(task, ctx)
        if not db:
            return code
        supp, flow = "None", data_flow(code)
        base = f"[Supplementary Code]\n{supp}\n\n[Target Code]\n{code}\n\n[Data Flow]\n{flow}"
        tv = st.ask_json(user(base, AP_TAINT_VAR)) or {}
        tf = st.ask_json(user(base, AP_TAINT_FUNC)) or {}
        sem = st.ask_json([{"role": "system", "content": AP_SEMANTIC}, {"role": "user", "content": AP_SEMANTIC_SHOT},
                           {"role": "assistant", "content": AP_SEMANTIC_ANSWER},
                           {"role": "user", "content": f"[Supplementary Code]\n{supp}\n\n[Target Code]\n{code}"}]) or {}
        semantics = str(sem.get("result", ""))
        q = embed("qodo", [semantics])[0]
        uv = {k: embed("qodo", [str(v)])[0] for k, v in tv.items()}
        uf = {k: embed("qodo", [str(v)])[0] for k, v in tf.items()}
        kw = _keywords(semantics, tags)

        scored = []
        for e in db:
            vm = _mapping(uv, {k: x["vec"] for k, x in e["vulnerability_related_variables"].items()})
            fm = _mapping(uf, {k: x["vec"] for k, x in e["vulnerability_related_functions"].items()})
            parts = (float(np.dot(q, np.asarray(e["code_semantics"]["vec"]))), _fuzzy_jaccard(kw, set(e["code_semantics"]["ner"])),
                     float(np.mean([v for _, v in vm.values()])) if vm else 0.0, float(np.mean([v for _, v in fm.values()])) if fm else 0.0)
            scored.append((sum(w * x for w, x in zip(AUTOPATCH_WEIGHTS, parts)), e, vm, fm))
        _, e, vm, fm = max(scored, key=lambda x: x[0])
        checklist, fix_list = e["vulnerability_checklist"], e["fix_list"]
        ex_v, ex_f, tg_v, tg_f = {}, {}, {}, {}
        for i, name in enumerate(e["vulnerability_related_variables"]):
            if name in vm:
                checklist, fix_list = checklist.replace(name, f"variable_{i}"), fix_list.replace(name, f"variable_{i}")
                ex_v[f"variable_{i}"], tg_v[f"variable_{i}"] = name, vm[name][0]
        for i, name in enumerate(e["vulnerability_related_functions"]):
            if name in fm:
                checklist, fix_list = checklist.replace(name, f"function_{i}"), fix_list.replace(name, f"function_{i}")
                ex_f[f"function_{i}"], tg_f[f"function_{i}"] = name, fm[name][0]
        cwe, cve = e["cwe_type"], e["cve_id"]
        verify = st.ask_json([
            {"role": "system", "content": AP_VERIFY.format(cwe=cwe, cve=cve, checklist=checklist)},
            {"role": "user", "content": AP_CONTEXT.format(supp="None", vmap=ex_v, fmap=ex_f, extra="", code=e["original_code"])},
            {"role": "assistant", "content": AP_VERIFY_SHOT.format(cwe=cwe, cve=cve, cot=e["verification_cot"], root=e["root_cause"],
                                                                   cot_json=json.dumps(e["verification_cot"]),
                                                                   root_json=json.dumps(e["root_cause"]))},
            {"role": "user", "content": AP_CONTEXT.format(supp=supp, vmap=tg_v, fmap=tg_f, extra="", code=code)}])
        if not verify or str(verify.get("result")).lower() != "true":
            return code
        out = ctx.llm.ask([
            {"role": "system", "content": AP_PATCH.format(cwe=cwe, cve=cve, fix_list=fix_list)},
            {"role": "user", "content": AP_CONTEXT.format(supp="None", vmap=ex_v, fmap=ex_f, extra=f"[Root Cause]\n{e['root_cause']}\n\n",
                                                          code=e["original_code"])},
            {"role": "assistant", "content": AP_PATCH_SHOT.format(cwe=cwe, cve=cve, cot=e["patch_cot"], patch=e["vuln_patch"])},
            {"role": "user", "content": AP_CONTEXT.format(supp=supp, vmap=tg_v, fmap=tg_f,
                                                          extra=f"[Root Cause]\n{verify.get('root_cause', '')}\n\n", code=code)}],
            temperature=0, role="patch")
        m = re.search(r"\[Patched Code START\]\s*(.*?)\s*\[Patched Code END\]", out, re.S)
        patched = C.code_of(task, m.group(1)) if m else ""
        return patched or code


# ---------- H45 PEACE ----------

PEACE_RANK, PEACE_EPOCHS, PEACE_LR = 32, 4, 1e-5
PEACE_MIN_IMPROVEMENT, PEACE_TRAIN, PEACE_ALTERNATIVES = 10.0, 3450, 2
PEACE_AGENT_ITERS, PEACE_FRAGMENT_LINES, PEACE_EXTERNAL, PEACE_INTERNAL = 10, 15, 2, 2
PEACE_OPTIMIZE = """Given the Python function below, improve its performance. Please only respond with the function code.

Function to optimize:
```python
{function}
```

Associated edits (previous changes related to this function):
```python
{edits}
```

If the message contains any information regarding the optimization goal, consider that as well. Otherwise, focus solely on improving execution speed, memory usage, and efficiency while keeping the functionality intact.

Message (may or may not be relevant to optimization goal):
```text
{message}
```

Optimized Version:
"""
PEACE_OPTIMIZER = ("Below are several versions of a codesnippet. You should analyze each version to determine their performance in "
                   "terms of execution speed. time efficiency.\nThe task is to create the most efficient version of the code in "
                   "terms of execution.\n### code snippet:\n{versions}\n### efficient version:")
PEACE_AGENT = """You are an expert Python programmer.
You are given the following code:
{code}
Your task is to find previous associated edits relevant to this code.
A valid associated edit is a previous change in the project that this code depends on or that should be propagated to it.
If no valid associated edits are found, return an empty string.

Available tool calls:
- get_top_k_fragments(k): Retrieve the top k edit fragments with the highest dependency score.
- get_fragments_in_range(l, r): Retrieve edit fragments ranked between l and r by dependency score.

Please respond in JSON format:
{{
    "isfinal_response": "",
    "usingtool": "",
    "response": ""
}}

- "isfinal_response": "1" if the response is final, "0" if more tool calls are needed.
- "usingtool": The tool call string (e.g., get_top_k_fragments(2)).
- "response": The final response containing relevant edit snippets."""
PEACE_INTEGRATE = """The function below is part of a program. An efficiency optimizer proposed an optimized version of it. Integrate the optimizer's edit into the function while preserving its correctness in the context of the whole program (same name, signature, behaviour and use by the rest of the program). Return only the integrated function in one ```python block.

Program:
```python
{program}
```

Function:
```python
{function}
```

Optimizer's version:
```python
{proposal}
```"""


def _call_order(code: str) -> list[str]:
    """Dependency-aware sequence: callees before callers over the program's own functions."""
    fs = functions(code)
    calls = {k: {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(v) if isinstance(n, ast.Call)} & set(fs) - {k}
             for k, v in fs.items()}
    order, seen = [], set()

    def visit(k, stack=()):
        if k in seen or k in stack:
            return
        for c in sorted(calls[k]):
            visit(c, stack + (k,))
        seen.add(k)
        order.append(k)
    for k in fs:
        visit(k)
    return order


@h2h
class PEACE(H2HTechnique):
    """Functions optimized in dependency order: an agent selects valid associated edits from the ranked history of edits
    made so far (tool calls, at most 10), the model writes an initial edit, a LoRA-tuned efficiency optimizer refines it with
    the most similar internal functions and external fast implementations, and the model integrates the result. Optimizer
    data rebuilt from PIE's Python pairs (the GitHub half and the external library are on an unreachable host; external
    fast functions are Mercury's fastest LeetCode solutions)."""
    hid, name, family, target = "H45", "PEACE", "retrieval", ("EFF",)
    code_url = "https://github.com/linrenmeng/Peace"
    trains = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if not (d / "external.npy").exists():
            from datasets import load_dataset
            ext = []
            for r in load_dataset("Elfsong/Mercury", split="train"):
                sols = [s for s in r.get("solutions") or [] if s.get("solution")]
                if sols:
                    best = min(sols, key=lambda s: float(re.sub(r"[^\d.]", "", str(s.get("runtime", "inf"))) or "inf"))
                    ext.append({"code": best["solution"], "slug": r.get("slug_name", "")})
            ext = decontaminate(ext, "code")
            save_index(d, ext, embed("qodo", [e["code"] for e in ext]), "external")
        if not (d / "model.txt").exists():
            rows = decontaminate([r for r in pie_python(ctx) if not r["code_same"]], "input", "target")
            by: dict[str, list] = defaultdict(list)
            for r in rows:
                by[r["problem_id"]].append(r)
            rng, data = random.Random(0), []
            for r in rng.sample(rows, len(rows)):
                if r["improvement_frac"] <= PEACE_MIN_IMPROVEMENT:
                    continue
                alts = [x["target"] for x in by[r["problem_id"]] if x["target"] not in (r["input"], r["target"])][:PEACE_ALTERNATIVES]
                versions = "\n".join(f"```python\n{v}\n```" for v in [r["input"], *alts])
                data.append({"messages": [{"role": "user", "content": PEACE_OPTIMIZER.format(versions=versions)},
                                          {"role": "assistant", "content": f"```python\n{r['target']}\n```"}]})
                if len(data) >= PEACE_TRAIN:
                    break
            write_jsonl(d / "optimizer.jsonl", data)
            lora_sft(ctx, self.hid, d / "optimizer.jsonl", rank=PEACE_RANK, epochs=PEACE_EPOCHS, lr=PEACE_LR)

    def solve(self, task: Task, ctx: Ctx) -> str:
        d = ctx.dir(self.hid)
        ext, ev = load_index(d, "external")
        optimizer = tuned(ctx, self.hid)
        program = draft(task, ctx)
        history: list[str] = []
        for name in _call_order(program) or [None]:
            fs = functions(program)
            func = segment(program, fs[name]) if name in fs else program
            edits = self._associated(ctx, func, history) if history else ""
            init = C.code_of(task, ctx.llm.ask(user(PEACE_OPTIMIZE.format(function=func, edits=edits, message="")),
                                               temperature=0, role="initial-edit")) or func
            internal = [segment(program, fs[k]) for k in fs if k != name]
            if internal:
                iv = embed("qodo", internal)
                internal = [internal[i] for i in top_k(iv, embed("qodo", [init])[0], PEACE_INTERNAL)]
            external = [ext[i]["code"] for i in top_k(ev, embed("qodo", [init])[0], PEACE_EXTERNAL)]
            versions = "\n".join(f"```python\n{v}\n```" for v in [init, *internal, *external])
            proposal = C.code_of(task, optimizer.ask(user(PEACE_OPTIMIZER.format(versions=versions)), temperature=0, role="optimizer"))
            final = C.code_of(task, ctx.llm.ask(user(PEACE_INTEGRATE.format(program=program, function=func, proposal=proposal or init)),
                                                temperature=0, role="integrate")) or init
            new = program.replace(func, final, 1) if func != program else final
            if cost(task, new) is not None or cost(task, program) is None:
                history.append("\n".join(difflib.unified_diff(func.splitlines(), final.splitlines(), lineterm="")))
                program = ctx.keep(new)
        return program

    @staticmethod
    def _associated(ctx: Ctx, code: str, history: list[str]) -> str:
        frags = [f for h in history for f in ("\n".join(h.splitlines()[i:i + PEACE_FRAGMENT_LINES])
                                              for i in range(0, len(h.splitlines()), PEACE_FRAGMENT_LINES))]
        fv, q = embed("qodo", frags), embed("qodo", [code])[0]
        ranked = [frags[i] for i in top_k(fv, q, len(frags))]
        prompt = PEACE_AGENT.format(code=code)
        for _ in range(PEACE_AGENT_ITERS):
            js = json_of(ctx.llm.ask(user(prompt), temperature=0, role="associated-edits"))
            js = js if isinstance(js, dict) else {}
            if str(js.get("isfinal_response", "1")) == "1":
                return str(js.get("response", ""))
            tool = str(js.get("usingtool", ""))
            res = [f"get_top_k_fragments({k}): {ranked[:int(k)]}" for k in re.findall(r"get_top_k_fragments\((\d+)\)", tool)]
            res += [f"get_fragments_in_range({l}, {r}): {ranked[int(l) - 1:int(r)]}"
                    for l, r in re.findall(r"get_fragments_in_range\((\d+),\s*(\d+)\)", tool)]
            prompt += f"\nUser:\n{chr(10).join(res)}\nresponse:\n"
        return ""


# ---------- H46 Mitigation-Aware Chain-of-Thought ----------

MACOT_KB = "https://raw.githubusercontent.com/mohsystem/paper3/main/cwes_ds/dataset/enriched_cwe_dataset_v6.json"
MACOT_TEMPERATURE = 1.0
MACOT_TAGGING = """You are a secure code classification assistant. Your task is to examine the given prompt description
(i.e., a programming task), then identify which of the following predefined tags are most relevant
based on the problem statement.

Note: The assigned tags will be used to link this prompt to corresponding
vulnerabilities or weaknesses (CWE entries) that might arise in the final code.
Therefore, it is crucial that you only select the tags that accurately reflect the
prompt's requirements or concerns.

Predefined Tags:
1. Memory & Resource Management
   - Dynamic allocation, stack vs. heap usage, pointer arithmetic, resource leaks.
2. Concurrency & Parallelism
   - Multi-threading, synchronization mechanisms, lock management, race conditions.
3. Networking & Communication
   - Socket programming, client-server architecture, protocols (TCP/UDP), web sockets.
4. Data safety and security
   - All sensitive data need to be encrypted in transition and at rest, storage, database, files.
5. Web Development, API & Web Services
   - Back-end frameworks, REST APIs, session management, front-end integration.
     Building/consuming RESTful APIs, microservices, GraphQL.
6. Database
   - SQL or No-SQL queries, ORM frameworks, transaction management.
7. Input Validation & Sanitization
   - Sanitizing user input, type checks, length checks, preventing injections.
8. File & I/O Handling
   - Reading, writing, directory traversal, file permissions, path handling.
9. Cryptography
   - Encryption/decryption, hashing, key management, SSL/TLS usage.
10. Authentication & Authorization
    - Password handling, login flows, session tokens, RBAC.
11. Error Handling
    - Exception capturing.
12. Code Injection
    - OS command injection, script injection.
13. Serialization & Deserialization
    - Safely (de)serializing JSON, XML, or binary data.
14. Hardcoding
    - Hardcoding credentials and secret data.
15. Language Basics
    - Language Basics always applicable for all languages and tasks.

Instructions:
1. Read the `prompt_description` below.
2. From the Predefined Tags list above, pick ONLY the tags that are relevant to the prompt.
3. Return these tags in a JSON array (of strings).
4. Do not include any explanations or additional commentary, and do not include tags that do not apply.
5. If multiple tags apply, list them all, e.g. ["Language Basics", "Concurrency & Parallelism", "File & I/O Handling"].
6. If no tags apply, return the default array ["Language Basics"].

Remember: These chosen tags will be used for further mapping to potential CWE vulnerabilities.

Prompt Description:
{task}

Required Output Format:
["TAG_NAME_1", "TAG_NAME_2", ...]
(Only the array of relevant tags, nothing else.)"""
MACOT_GENERATE = """Write the code for the following prompt description by applying the process below internally. Do not output the steps or analysis.
{task}

Chain-of-Through process in code generation:
1. Problem understanding: define the purpose of the code, required functionality, inputs, outputs, and operations.
2. Security requirements: the code must avoid vulnerabilities and security weaknesses.
   Provided CWE list, baseline security and quality rules:
{mitigations}
3. Secure coding generation: write the code step by step to prevent the weaknesses identified.
4. Code review: review the generated code line by line to confirm there are no weaknesses.
5. Secure code output: output the final corrected code only, inside the code block."""


def macot_mitigations(kb: list[dict], tags: list[str], lang: str) -> str:
    """The authors' get_related_cwe_list: records by category or additional tag, deduplicated, rendered per language."""
    recs, seen = [], set()
    for tag in [*tags, "Language Basics"]:
        for r in kb:
            if r.get("category") == tag or tag in [str(t) for t in r.get("Additional_tags", [])]:
                sig = json.dumps(r, sort_keys=True)
                if sig not in seen:
                    seen.add(sig)
                    recs.append(r)
    out, n = "", 0
    for r in recs:
        n += 1
        mit = r.get("mitigations", {})
        if mit.get("general_rules"):
            out += "## General Mitigation\n"
            for rule in mit["general_rules"]:
                out += f"[\"Rules#{n}: {rule.strip()}\"]\n"
                n += 1
            out += "\n"
        for l, data in (mit.get("language_specific") or {}).items():
            if l.lower() != lang.lower():
                continue
            out += f"## {l.upper()} Specific Mitigation\n"
            if data.get("guidance"):
                out += "### Guidance\n" + "".join(f"- {g.strip()}\n" for g in data["guidance"])
            if data.get("checklist"):
                out += "\n### Checklist\n" + "".join(f"- {c.strip()}\n" for c in data["checklist"])
            out += "\n"
        if r.get("code_review_checklist"):
            out += "## Code Review Checklist\n" + "".join(f"- {c.strip()}\n" for c in r["code_review_checklist"]) + "\n"
        exs = [e for e in r.get("finetune_examples", []) if (e.get("language") or "").strip().lower() == lang.lower()]
        if exs:
            out += "## Finetune Examples\n"
            for i, e in enumerate(exs, 1):
                out += f"### Example {i} ({(e.get('language') or 'generic').strip()})\n"
                for k, lab in (("instruction", "Instruction: "), ("input", "Input:\n```\n"), ("output", "Ideal Output:\n```\n")):
                    if (e.get(k) or "").strip():
                        out += lab + e[k].strip() + ("\n```\n" if k != "instruction" else "\n")
            out += "\n"
    return out


@h2h
class MACoT(H2HTechnique):
    """The model tags the task with security domains; the authors' mitigation dataset returns the matching records
    (general rules, Python guidance and checklist, review checklist, examples) inside a five-step secure CoT prompt."""
    hid, name, family, target = "H46", "Mitigation-Aware Chain-of-Thought", "retrieval", ("SEC",)
    code_url = "https://github.com/mohsystem/paper3"

    def prepare(self, ctx: Ctx) -> None:
        p = ctx.dir(self.hid) / "kb.json"
        if not p.exists():
            fetch(MACOT_KB, ctx.dir(self.hid) / "raw.json")
            p.write_text(json.dumps(decontaminate(json.loads((ctx.dir(self.hid) / "raw.json").read_text()), "finetune_examples")))

    def solve(self, task: Task, ctx: Ctx) -> str:
        kb = json.loads((ctx.dir(self.hid) / "kb.json").read_text())
        desc = C.task_text(task)
        tags = json_of(ctx.llm.ask(user(MACOT_TAGGING.format(task=desc)), temperature=MACOT_TEMPERATURE, role="tagging"))
        tags = [str(t) for t in tags] if isinstance(tags, list) else []
        prompt = MACOT_GENERATE.format(task=desc, mitigations=macot_mitigations(kb, tags, task.language))
        return C.code_of(task, ctx.llm.ask(user(prompt), temperature=MACOT_TEMPERATURE, role="generate"))
