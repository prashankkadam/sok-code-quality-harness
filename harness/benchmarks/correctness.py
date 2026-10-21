"""Correctness benchmarks: EvalPlus (HumanEval+, MBPP+), LiveCodeBench v6 and the LiveCodeBench Pro protocol."""
from __future__ import annotations

import ast
import base64
import doctest
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .. import config, sandbox
from ..measure.metrics import mean_pass_at_k
from ..sandbox import LANGS, Workdir, execute
from ..tasks import Output, Task, Test
from .base import Benchmark, register

WORKERS = int(os.environ.get("HARNESS_WORKERS", max(1, (os.cpu_count() or 2) // 2)))


def fetch(url: str, path: Path, headers: dict | None = None) -> Path:
    if not path.exists():
        import httpx
        r = httpx.get(url, headers=headers, follow_redirects=True, timeout=600)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def pmap(fn, *its) -> list:
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, *its))


def day(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).date().isoformat()


def pass_at_1(rows: list[dict]) -> float:
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        if "passed" in r:
            per[r["task_id"]][0] += 1
            per[r["task_id"]][1] += bool(r["passed"])
    return mean_pass_at_k([tuple(v) for v in per.values()])


def _assert(expr: str, want: str, prelude: str = "") -> Test | None:
    try:
        ast.parse(expr, mode="eval"), ast.parse(want, mode="eval")
    except SyntaxError:
        return None
    return Test("assert", f"{prelude}_got = {expr}\n_want = {want}\n"
                          "assert _got == _want or (isinstance(_want, float) and abs(_got - _want) <= 1e-6), _got\n")


def doc_examples(prompt: str, entry: str, prelude: str = "") -> list[Test]:
    """The examples a HumanEval prompt shows: doctest blocks, else `f(x) == y` / `f(x) ➞ y` lines."""
    try:
        pairs = [(e.source.strip(), e.want.strip()) for e in doctest.DocTestParser().get_examples(prompt) if e.want.strip()]
    except ValueError:
        pairs = []
    if not pairs:
        pairs = re.findall(rf"^\s*({re.escape(entry)}\(.*\))\s*(?:==|➞|=>|->)\s*(.+?)\.?\s*$", prompt, re.M)
    return [t for s, w in pairs if entry in s and (t := _assert(s, w, prelude))]


class _EvalPlus(Benchmark):
    dims = ("COR",)
    metrics = {"pass@1": True}
    dataset = ""

    def problems(self) -> dict:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        return get_human_eval_plus() if self.dataset == "humaneval" else get_mbpp_plus()

    def task(self, p: dict) -> Task:
        raise NotImplementedError

    def load(self) -> list[Task]:
        return [self.task(p) for p in self.problems().values()]

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        if not outs:
            return []
        from evalplus.sanitize import sanitize
        lines = []
        for o in outs:
            try:
                code = sanitize(o.code, tasks[o.task_id].entry_point)
            except Exception:
                code = o.code
            lines.append({"task_id": o.task_id, "solution": code})
        have = {o.task_id for o in outs}
        lines += [{"task_id": t, "solution": ""} for t in self.problems() if t not in have]
        with tempfile.TemporaryDirectory(prefix="evalplus_") as tmp:
            samples = Path(tmp, "samples.jsonl")
            samples.write_text("".join(json.dumps(x) + "\n" for x in lines))
            p = subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", self.dataset, "--samples", str(samples),
                                "--parallel", str(WORKERS)], capture_output=True, text=True)
            res = Path(tmp, "samples_eval_results.json")
            if not res.exists():
                raise RuntimeError(f"evalplus.evaluate failed: {p.stderr[-3000:]}")
            ev = json.loads(res.read_text())["eval"]
        seen, rows = Counter(), []
        for o in outs:
            r = ev[o.task_id][seen[o.task_id]]
            seen[o.task_id] += 1
            rows.append({"task_id": o.task_id, "base": r["base_status"] == "pass",
                         "passed": r["base_status"] == "pass" and r["plus_status"] == "pass"})
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"pass@1": pass_at_1(rows)}


@register
class HumanEvalPlus(_EvalPlus):
    name, dataset = "humaneval_plus", "humaneval"

    def task(self, p: dict) -> Task:
        ep = p["entry_point"]
        return Task(id=p["task_id"], benchmark=self.name, prompt=p["prompt"], entry_point=ep,
                    public_tests=doc_examples(p["prompt"], ep), hidden_tests=[Test("assert", f"{p['test']}\ncheck({ep})\n")],
                    reference=p["prompt"] + p["canonical_solution"])


@register
class MBPPPlus(_EvalPlus):
    name, dataset = "mbpp_plus", "mbpp"

    def task(self, p: dict) -> Task:
        shown = [Test("assert", ln.strip()) for ln in p["prompt"].splitlines() if ln.strip().startswith("assert ")]
        return Task(id=p["task_id"], benchmark=self.name, prompt=p["prompt"], entry_point=p["entry_point"],
                    public_tests=shown, hidden_tests=[Test("assert", p["assertion"])], reference=p["canonical_solution"])


LCB_REPO = "livecodebench/code_generation_lite"
LCB_FILES = {"release_v6": ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]}
LCB_IMPORTS = ("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\n"
               "from bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\n"
               "from itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\n"
               "from json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\n"
               "import collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\n"
               "import itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n")
LCB_STARTER = ("### Format: You will use the following starter code to write the solution to the problem and enclose your code "
               "within delimiters.\n```python\n{}\n```\n\n")
LCB_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the "
             "sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads "
             "the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
LCB_DRIVER = '''

import json as _lcb_json, signal as _lcb_signal, sys as _lcb_sys


def _lcb_alarm(*_):
    raise TimeoutError("timeoutexception")


_lcb_signal.signal(_lcb_signal.SIGALRM, _lcb_alarm)
with open("tests.json") as _lcb_f:
    _lcb = _lcb_json.load(_lcb_f)
_lcb_m = getattr(Solution(), _lcb["fn"]) if _lcb["cls"] else globals()[_lcb["fn"]]
for _lcb_args, _lcb_want in _lcb["cases"]:
    _lcb_signal.alarm(_lcb["timeout"])
    try:
        _lcb_got = _lcb_m(*_lcb_args)
    finally:
        _lcb_signal.alarm(0)
    if isinstance(_lcb_got, tuple):
        _lcb_got = list(_lcb_got)
    if _lcb_got != _lcb_want:
        _lcb_sys.exit(1)
print("__LCB_PASS__")
'''


def lcb_private(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def lcb_match(got: str, want: str) -> bool:
    a, b = ([ln.strip() for ln in s.strip().split("\n")] for s in (got, want))
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            if [Decimal(e) for e in x.split()] != [Decimal(e) for e in y.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


def lcb_call(code: str, fn: str, tests: list[Test], timeout: int, mem_mb: int) -> bool:
    spec = {"fn": fn, "cls": "class Solution" in code, "timeout": timeout,
            "cases": [[json.loads(t.code), t.expected] for t in tests]}
    with Workdir({"main.py": LCB_IMPORTS + "\n\n" + code + LCB_DRIVER, "tests.json": json.dumps(spec)}) as d:
        r = execute(["python3", "main.py"], d, "", (timeout + 1) * len(tests) + 5, mem_mb)
    return r.returncode == 0 and "__LCB_PASS__" in r.stdout


def lcb_stdio(code: str, tests: list[Test], timeout: int, mem_mb: int) -> bool:
    with Workdir({"main.py": LCB_IMPORTS + "\n" + code}) as d:
        for t in tests:
            r = execute(["python3", "main.py"], d, t.stdin, timeout, mem_mb)
            if r.timeout or r.returncode != 0 or not lcb_match(r.stdout, str(t.expected)):
                return False
    return True


@register
class LCBv6(Benchmark):
    name, dims, metrics = "lcb_v6", ("COR",), {"pass@1": True}
    release, timeout = "release_v6", 6

    def load(self) -> list[Task]:
        from huggingface_hub import hf_hub_download
        tasks = []
        for f in LCB_FILES[self.release]:
            with open(hf_hub_download(LCB_REPO, f, repo_type="dataset")) as fh:
                tasks += [self.task(json.loads(ln)) for ln in fh if ln.strip()]
        return tasks

    def task(self, p: dict) -> Task:
        fn = json.loads(p["metadata"]).get("func_name")

        def conv(t: dict) -> Test:
            if fn:
                return Test("call", json.dumps([json.loads(x) for x in t["input"].split("\n")]), expected=json.loads(t["output"]))
            return Test("io", stdin=t["input"], expected=t["output"])

        starter = p["starter_code"]
        prompt = f"### Question:\n{p['question_content']}\n\n" + (LCB_STARTER.format(starter) if starter else LCB_STDIN) \
            + "### Answer: (use the provided format with backticks)\n\n"
        return Task(id=p["question_id"], benchmark=self.name, prompt=prompt, mode="function" if fn else "stdio",
                    entry_point=(f"Solution().{fn}" if "class Solution" in starter else fn) if fn else None,
                    public_tests=[conv(t) for t in json.loads(p["public_test_cases"])],
                    hidden_tests=[conv(t) for t in lcb_private(p["private_test_cases"])],
                    date=p["contest_date"][:10], time_limit=self.timeout, memory_mb=4096,
                    meta={"platform": p["platform"], "difficulty": p["difficulty"], "contest_id": p["contest_id"],
                          "title": p["question_title"], "func_name": fn, "starter_code": starter, "prelude": LCB_IMPORTS})

    def score(self, task: Task, out: Output) -> dict:
        tests = task.public_tests + task.hidden_tests
        fn = task.meta["func_name"]
        ok = bool(tests) and (lcb_call(out.code, fn, tests, self.timeout, task.memory_mb) if fn
                              else lcb_stdio(out.code, tests, self.timeout, task.memory_mb))
        return {"task_id": task.id, "passed": ok}

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        return pmap(self.score, [tasks[o.task_id] for o in outs], outs)

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"pass@1": pass_at_1(rows)}


CF = ("https://codeforces.com", "https://mirror.codeforces.com")
CF_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
TESTLIB = "https://raw.githubusercontent.com/MikeMirzayanov/testlib/1e4e8a24c79c6bad3becbdb5a332ffc352b7d5dd/testlib.h"
PRO_LANG = {"cpp": ("C++", "cpp"), "python": ("Python 3", "python"), "java": ("Java", "java")}
PRO_BUILD = {"cpp": ["g++", "main.cpp", "-O2", "-pipe", "-std=gnu++17", "-o", "main"]}
PRO_PROMPT = ("You are a competitive programmer. You will be given a problem statement, please implement solution in {}. "
              "The execution time and memory limit are also stated in the statement so be aware of the complexity of the "
              "program. Please wrap the code in ```{} and ``` so that it is properly formatted.\n\n")
CHECKERS = ("checker.cpp", "checker.cc", "chk.cpp", "chk.cc")


class FileTest(Test):
    """An io test whose input and answer stay on disk until read."""

    def __init__(self, inp: Path, ans: Path):
        self.kind, self.code, self.inp, self.ans = "io", "", inp, ans

    stdin = property(lambda self: self.inp.read_text())
    expected = property(lambda self: self.ans.read_text())


def io_tests(d: Path) -> list[Test]:
    key = lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]
    pairs = ((i, next((i.with_suffix(s) for s in (".ans", ".out") if i.with_suffix(s).exists()), None))
             for i in sorted(d.glob("*.in"), key=key))
    return [FileTest(i, a) for i, a in pairs if a]


def checker(pdir: Path) -> Path | None:
    src = next((pdir / n for n in CHECKERS if (pdir / n).exists()), None)
    if src is None:
        return None
    binary = pdir / f"{src.name}.bin"
    if not binary.exists() or binary.stat().st_mtime < src.stat().st_mtime:
        inc = pdir if (pdir / "testlib.h").exists() else pdir.parent
        r = execute(["g++", str(src), "-O2", "-pipe", "-std=gnu++17", "-I", str(inc), "-o", str(binary)], pdir, timeout=180, mem_mb=4096)
        if not r.ok:
            raise RuntimeError(f"checker build failed for {pdir.name}: {r.stderr[-1000:]}")
    return binary


def pro_judge(code: str, lang: str, tests: list[Test], tl: float, ml: int, chk: Path | None) -> str:
    """LightCPVerifier semantics: compile, run each test under the problem's limits, stop at the first non-accepted test,
    accept by the testlib checker's exit code (token comparison without a checker)."""
    spec = LANGS[lang]
    with Workdir({spec["file"]: code}) as d:
        b = execute(PRO_BUILD[lang], d, timeout=60, mem_mb=4096) if lang in PRO_BUILD else sandbox.build(lang, d)
        if not b.ok:
            return "Compile Error"
        for t in tests:
            inp, ans = t.stdin, str(t.expected)
            r = execute(spec["run"], d, inp, 2 * tl, ml)
            if r.timeout or r.seconds > tl:
                return "Time Limit Exceeded"
            if r.returncode != 0:
                return "Runtime Error"
            if chk is None:
                ok = r.stdout.split() == ans.split()
            else:
                with Workdir({"in.txt": inp, "out.txt": r.stdout, "ans.txt": ans}) as cd:
                    ok = execute([str(chk), "in.txt", "out.txt", "ans.txt"], cd, timeout=30, mem_mb=1024).returncode == 0
            if not ok:
                return "Wrong Answer"
    return "Accepted"


def _cf_api(query: str):
    import httpx
    time.sleep(2.1)
    d = httpx.get(f"{CF[0]}/api/{query}", headers=CF_UA, timeout=600).json()
    if d.get("status") != "OK":
        raise RuntimeError(f"Codeforces API {query}: {d.get('comment')}")
    return d["result"]


def _cf_page(cid: int, idx: str) -> str:
    import httpx
    headers = dict(CF_UA, **({"Cookie": os.environ["CF_COOKIE"]} if os.environ.get("CF_COOKIE") else {}))
    for host in CF:
        try:
            r = httpx.get(f"{host}/contest/{cid}/problem/{idx}", headers=headers, timeout=60, follow_redirects=True)
        except httpx.HTTPError:
            continue
        if r.status_code == 200 and 'class="problem-statement"' in r.text:
            return r.text
    raise RuntimeError(f"could not fetch problem {cid}{idx}; if Cloudflare blocks it, set CF_COOKIE to a browser session cookie")


def _txt(el) -> str:
    return re.sub(r"\s+", " ", el.text_content()).strip().replace("$$$$$$", "$$").replace("$$$", "$")


def _pre(pre) -> str:
    lines = pre.xpath('./div[contains(@class, "test-example-line")]')
    if lines:
        return "\n".join(ln.text_content() for ln in lines)
    for br in pre.iter("br"):
        br.tail = "\n" + (br.tail or "")
    return pre.text_content().strip("\n")


def _md(el) -> str:
    parts = [el.text.strip()] if el.text and el.text.strip() else []
    for c in el:
        if c.get("class") == "section-title":
            continue
        if c.tag in ("ul", "ol"):
            parts.append("\n".join(f"- {_txt(li)}" for li in c.xpath("./li")))
        elif c.tag == "pre":
            parts.append(f"```\n{_pre(c)}\n```")
        else:
            parts.append(_txt(c))
    return "\n\n".join(p for p in parts if p)


def parse_problem(html: str) -> dict:
    from lxml import html as lh
    st = lh.fromstring(html).xpath('//div[@class="problem-statement"]')[0]
    hdr = st.xpath('./div[@class="header"]')[0]
    prop = lambda c: (hdr.xpath(f'./div[@class="{c}"]/div[@class="property-title"]')[0].tail or "").strip()
    sections, samples = [], []
    for el in st.xpath("./div"):
        cls = el.get("class") or ""
        if cls == "header":
            continue
        if cls == "sample-tests":
            for s in el.xpath('.//div[@class="sample-test"]'):
                samples += zip(map(_pre, s.xpath('./div[@class="input"]/pre')), map(_pre, s.xpath('./div[@class="output"]/pre')))
            sections.append(("Examples", "\n\n".join(f"### Input\n```\n{i}\n```\n### Output\n```\n{o}\n```" for i, o in samples)))
            continue
        title = el.xpath('./div[@class="section-title"]')
        sections.append((_txt(title[0]) if title else "", _md(el)))
    text = " ".join(b for _, b in sections).lower()
    return {"title": _txt(hdr.xpath('./div[@class="title"]')[0]), "time_limit": float(prop("time-limit").split()[0]),
            "memory_limit": int(float(prop("memory-limit").split()[0])), "input": prop("input-file"), "output": prop("output-file"),
            "sections": sections, "samples": samples,
            "interactive": any(t == "Interaction" for t, _ in sections) or "interactive problem" in text}


def statement(p: dict) -> str:
    tl = p["time_limit"]
    head = f"# {p['title']}\n\ntime limit per test: {tl:g} second{'' if tl == 1 else 's'}\n" \
           f"memory limit per test: {p['memory_limit']} megabytes"
    return "\n\n".join([head] + [f"## {t}\n\n{b}" if t else b for t, b in p["sections"]]) + "\n"


def build(after: str | None = None, root: Path | None = None) -> list[str]:
    """Collect Codeforces contests held after `after` into root/<contestId><index>/ (statement.md, meta.json, samples/).
    Hidden tests (tests/*.in + *.ans|*.out) and an optional testlib checker.cpp are added to each directory separately."""
    after = after or config.load("benchmarks")["lcb_pro"]["contests_after"]
    root = root or config.DATA / "lcb_pro"
    fetch(TESTLIB, root / "testlib.h")
    contests = sorted((c for c in _cf_api("contest.list?gym=false") if c["phase"] == "FINISHED" and c["type"] != "IOI"
                       and "Kotlin Heroes" not in c["name"] and day(c.get("startTimeSeconds", 0)) > after),
                      key=lambda c: (c["startTimeSeconds"], c["id"]))
    seen, made = set(), []
    for c in contests:
        for q in _cf_api(f"contest.standings?contestId={c['id']}")["problems"]:
            key, d = (c["startTimeSeconds"], q["name"]), root / f"{c['id']}{q['index']}"
            if q.get("type") != "PROGRAMMING" or key in seen:
                continue
            seen.add(key)
            if (d / "meta.json").exists():
                continue
            p = parse_problem(_cf_page(c["id"], q["index"]))
            if p["input"] != "standard input" or p["output"] != "standard output":
                continue
            (d / "samples").mkdir(parents=True, exist_ok=True)
            for k, (i, o) in enumerate(p["samples"], 1):
                (d / "samples" / f"{k}.in").write_text(i.rstrip("\n") + "\n")
                (d / "samples" / f"{k}.out").write_text(o.rstrip("\n") + "\n")
            (d / "statement.md").write_text(statement(p))
            (d / "meta.json").write_text(json.dumps({
                "contest_id": c["id"], "index": q["index"], "name": q["name"], "contest": c["name"], "date": day(c["startTimeSeconds"]),
                "time_limit": p["time_limit"], "memory_limit": p["memory_limit"], "rating": q.get("rating"), "tags": q.get("tags", []),
                "interactive": p["interactive"], "url": f"{CF[0]}/contest/{c['id']}/problem/{q['index']}"}, indent=1))
            made.append(d.name)
    return made


@register
class LCBPro(Benchmark):
    name, dims, metrics, mode = "lcb_pro", ("COR",), {"pass@1": True}, "stdio"

    def __init__(self, limit: int | None = None, after: str | None = None, before: str | None = None, language: str = "cpp"):
        super().__init__(limit, after or config.load("benchmarks")["lcb_pro"]["contests_after"], before)
        self.language = language

    def load(self) -> list[Task]:
        tasks = []
        for m in sorted((config.DATA / "lcb_pro").glob("*/meta.json")):
            meta, d = json.loads(m.read_text()), m.parent
            if meta.get("interactive"):
                continue
            tasks.append(Task(id=d.name, benchmark=self.name, prompt=PRO_PROMPT.format(*PRO_LANG[self.language]) + (d / "statement.md").read_text(),
                              language=self.language, mode="stdio", public_tests=io_tests(d / "samples"), hidden_tests=io_tests(d / "tests"),
                              date=meta["date"], time_limit=meta["time_limit"], memory_mb=meta["memory_limit"], meta={**meta, "dir": str(d)}))
        return tasks

    def score(self, task: Task, out: Output) -> dict:
        tests = task.hidden_tests or task.public_tests
        v = pro_judge(out.code, task.language, tests, task.time_limit, task.memory_mb, checker(Path(task.meta["dir"])))
        return {"task_id": task.id, "passed": v == "Accepted", "verdict": v, "hidden": bool(task.hidden_tests)}

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        for d in {tasks[o.task_id].meta["dir"] for o in outs}:
            checker(Path(d))
        return pmap(self.score, [tasks[o.task_id] for o in outs], outs)

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"pass@1": pass_at_1(rows), "samples_only": float(len({r["task_id"] for r in rows if not r.get("hidden", True)}))}
