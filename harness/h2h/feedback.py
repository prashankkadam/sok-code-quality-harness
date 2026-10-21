"""E7 tool-feedback family, H11-H20. Every tool signal comes from the public examples only; trained parts are retrained in prepare()."""
from __future__ import annotations

import ast
import difflib
import hashlib
import io
import builtins
import json
import pstats
import random
import re
import subprocess
import sys
import tempfile
import tokenize
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from ..llm import BudgetExceeded, current_scope
from ..sandbox import LANGS, Run, Workdir, build, execute, passes, run_test
from ..tasks import Task, Test
from ..techniques import common as C
from .base import Ctx, H2HTechnique, h2h

BANDIT = [sys.executable, "-m", "bandit", "-f", "json", "-q"]
PYLINT_SYNTAX = [sys.executable, "-m", "pylint", "-f", "json", "--disable=all", "--enable=E0001"]
PYLINT_SCORE = [sys.executable, "-m", "pylint", "-f", "json2", "--score=y"]
CODEQL = "codeql"
CODEQL_SUITE = "codeql/{lang}-queries:codeql-suites/{lang}-security-extended.qls"
CODEQL_LANGS = {"python", "javascript", "cpp", "java", "csharp", "go", "ruby"}
INFER = ["infer", "run", "--bufferoverrun", "--pulse", "--"]
INFER_COMPILE = {"cpp": ["clang++", "-std=c++20", "-c", "main.cpp"], "c": ["clang", "-c", "main.c"]}
PYSPY = ["py-spy", "record", "--rate", "100", "--format", "raw", "--nonblocking", "--output", "profile.txt", "--",
         "python3", "driver.py"]
CPROFILE = ["python3", "-m", "cProfile", "-o", "profile.out", "driver.py"]
TIMING_RUNS = 12
TOOL_MEM_MB = 32768


# ----------------------------------------------------------------------------------------------------------- helpers

def _left() -> int:
    return current_scope().budget.left


def _draw(ctx: Ctx, msgs, n: int, **kw) -> list[str]:
    """n samples, or as many as the ceiling still allows (a call with nothing left raises BudgetExceeded)."""
    return ctx.llm.sample(msgs, max(1, min(n, _left())), **kw)


def _ceiling(ctx: Ctx, code: str):
    ctx.keep(code)
    raise BudgetExceeded("published configuration exceeds the call ceiling")


def _py(task: Task) -> bool:
    return task.language == "python"


def _fence(task: Task, code: str) -> str:
    return f"```{task.language}\n{code}\n```"


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r, open(tmp, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def _json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text or "", re.S) or re.search(r"(\{.*\})", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(1), strict=False)
        except json.JSONDecodeError:
            return None


def test_text(t: Test) -> str:
    if t.kind == "io":
        return f"Input:\n{t.stdin}\nExpected output:\n{t.expected}"
    if t.kind == "call":
        return f"Arguments: {t.code}\nExpected return value: {json.dumps(t.expected)}"
    return t.code


def failure(task: Task, code: str) -> tuple[Test, Run] | None:
    for t in task.public_tests:
        r = run_test(task, code, t)
        if not r.ok:
            return t, r
    return None


def error_text(t: Test, r: Run) -> str:
    if r.timeout:
        return "Time limit exceeded."
    if r.returncode != 0:
        return r.stderr.strip()[-3000:] or f"Exit code {r.returncode}."
    return f"Wrong output. Expected:\n{t.expected}\nGot:\n{r.stdout.strip()[:2000]}"


def pass_rate(task: Task, code: str) -> float:
    k, m = passes(task, code, task.public_tests, stop_on_fail=False)
    return k / m if m else 1.0


def test_times(task: Task, code: str, runs: int = TIMING_RUNS, drop: str = "minmax") -> list[float] | None:
    """Mean runtime per public example over `runs` pinned runs, dropping the min and max ("minmax") or the first run."""
    if not task.public_tests or not code.strip():
        return None
    out = []
    for t in task.public_tests:
        xs = []
        for _ in range(runs):
            r = run_test(task, code, t, pin=True)
            if not r.ok:
                return None
            xs.append(r.seconds)
        xs = sorted(xs)[1:-1] if drop == "minmax" and runs > 2 else xs[1:] if drop == "first" and runs > 1 else xs
        out.append(sum(xs) / len(xs))
    return out


def runtime(task: Task, code: str, runs: int = TIMING_RUNS, drop: str = "minmax") -> float | None:
    ts = test_times(task, code, runs, drop)
    return None if ts is None else sum(ts)


def bandit_results(code: str) -> tuple[list[dict], bool]:
    """Bandit results with default settings, and whether Bandit could parse the file."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "solution.py"
        p.write_text(code)
        out = subprocess.run(BANDIT + [str(p)], capture_output=True, text=True).stdout
    try:
        rep = json.loads(out)
    except json.JSONDecodeError:
        return [], False
    return rep.get("results", []), not rep.get("errors")


def pylint_messages(code: str, cmd: list[str]) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "solution.py"
        p.write_text(code)
        out = subprocess.run(cmd + [str(p)], capture_output=True, text=True).stdout
    try:
        rep = json.loads(out)
    except json.JSONDecodeError:
        return []
    return rep if isinstance(rep, list) else rep.get("messages", [])


def pylint_score(code: str) -> float:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "solution.py"
        p.write_text(code)
        out = subprocess.run(PYLINT_SCORE + [str(p)], capture_output=True, text=True).stdout
    try:
        return float(json.loads(out)["statistics"]["score"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0.0


def codeql(code: str, lang: str) -> list[dict] | None:
    """CodeQL security-extended alerts as {rule, message, line}; None when the language is unsupported or CodeQL fails."""
    if lang not in CODEQL_LANGS:
        return None
    name = LANGS.get(lang, {}).get("file", f"main.{lang}")
    with Workdir({f"src/{name}": code}) as d:
        mk = execute([CODEQL, "database", "create", str(d / "db"), f"--language={lang}", f"--source-root={d / 'src'}",
                      "--build-mode=none", "--overwrite"], d, timeout=900, mem_mb=TOOL_MEM_MB)
        if not mk.ok:
            return None
        an = execute([CODEQL, "database", "analyze", str(d / "db"), CODEQL_SUITE.format(lang=lang), "--format=sarif-latest",
                      f"--output={d / 'out.sarif'}", "--threads=1"], d, timeout=900, mem_mb=TOOL_MEM_MB)
        if not an.ok or not (d / "out.sarif").exists():
            return None
        sarif = json.loads((d / "out.sarif").read_text())
    out = []
    for res in (sarif.get("runs") or [{}])[0].get("results", []):
        loc = (res.get("locations") or [{}])[0].get("physicalLocation", {}).get("region", {})
        out.append({"rule": res.get("ruleId", ""), "message": res.get("message", {}).get("text", ""),
                    "line": loc.get("startLine", 0)})
    return out


def codeql_text(alerts: list[dict]) -> str:
    return "\n".join(f"- {a['rule']} at line {a['line']}: {a['message']}" for a in alerts)


DRIVER = '''import io, json, sys, time
_SRC = {src!r}
_TESTS = {tests!r}
_ENTRY = {entry!r}
_CODE = compile(_SRC, "solution.py", "exec")
_END = time.perf_counter() + {seconds}
while True:
    for _kind, _a in _TESTS:
        if _kind == "io":
            _stdout, sys.stdin, sys.stdout = sys.stdout, io.StringIO(_a), io.StringIO()
            try:
                exec(_CODE, {{"__name__": "__main__"}})
            except SystemExit:
                pass
            finally:
                sys.stdin, sys.stdout = sys.__stdin__, _stdout
        else:
            _ns = {{"__name__": "__main__"}}
            exec(_CODE, _ns)
            if _kind == "call":
                _ns[_ENTRY](*json.loads(_a))
            else:
                exec(compile(_a, "test.py", "exec"), _ns)
    if time.perf_counter() >= _END:
        break
'''


def driver(task: Task, code: str, seconds: float) -> str:
    tests = [(t.kind, t.stdin if t.kind == "io" else t.code) for t in task.public_tests]
    return DRIVER.format(src=code, tests=tests, entry=task.entry_point, seconds=seconds)


# ------------------------------------------------------------------------------------------------- H11 SBLLM

SBLLM_DATA = "https://zenodo.org/records/14096664/files/processed_data.zip?download=1"
SBLLM_SEED, SBLLM_GEN, SBLLM_BEAM, SBLLM_KEEP, SBLLM_ITERS, SBLLM_TEMP, SBLLM_RUNS = 5, 4, 3, 5, 4, 0.7, 25

SBLLM_COT_SYSTEM = ("You are a software developer and now you will help to improve code efficiency. Please follow the output "
                    "format of the examples and Please reply with the improved code in code blocks (```{lang} ```) and "
                    "explain the reasons briefly at the beginning.")
SBLLM_SYSTEM = ("You are a software developer and now you will help to improve code efficiency. Please follow the instructions "
                "and output format specification to generate a more efficient code. The improved code should be in code "
                "blocks (```{lang} ```).")
SBLLM_COT_QUERY = "# optimize this code \n# slow version\n{code}\n# optimized version of the same code\n"
SBLLM_FORMAT = '''Task Desctiption & Instructions:
You will be provided with a code snippet{pattern_intro}. Your task is to propose a more efficient optimization method to achieve a higher speedup rate. Please refer to the existing versions and avoid the mistakes made in incorrect and unoptimized versions (if any).
Please follow the instructions step-by-step to improve code efficiency: 1. Analyze the original code and the optimizations applied in the existing versions. 2. Identify any additional optimization opportunities that have not been utilized. 3. Explain your optimization methods and provide a new optimized code snippet.
Please provide a direct explanation of the optimization points and include only one optimized code snippet in the third step.

Desired output format:
1. Analyze the original code and the optimizations applied in the existing versions: <answer>
2. Identify any additional optimization opportunities that have not been utilized: <answer>
3. Explain your optimization methods and provide a new optimized code snippet: <optimization points> ```python
<code>
```

'''
SBLLM_INTRO = " and its existing optimization versions along with their performance"
SBLLM_INTRO_PATTERN = ", its existing optimization versions along with their performance, and two code transformation patterns"
SBLLM_QUERY = ("The code you need to optimize:\n```python\n{code}\n```\nSome existing versions with their performance:\n{perf}\n"
               "{patterns}Please follow the above instructions and output format specification step-by-step to generate a "
               "better program.\n")


@dataclass
class _Cand:
    code: str
    content: str
    acc: float
    time: float


class _Abstract(ast.NodeTransformer):
    builtins = set(dir(builtins))

    def visit_FunctionDef(self, node):
        for a in node.args.args:
            a.arg = "VAR"
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = "STR"
        elif isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            node.value = "NUM"
        return node

    def visit_Name(self, node):
        if node.id not in self.builtins:
            node.id = "VAR"
        return node


def _abstract_tokens(code: str) -> list[str]:
    src = ast.unparse(_Abstract().visit(ast.parse(code)))
    return [t.string for t in tokenize.tokenize(io.BytesIO(src.encode()).readline) if t.type != tokenize.ENCODING]


def _diff_lines(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    d = list(difflib.Differ().compare(a, b))
    return [x[2:] for x in d if x.startswith("- ")], [x[2:] for x in d if x.startswith("+ ")]


def _lev(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def _sbllm_index(path: str):
    from rank_bm25 import BM25Okapi
    rows = []
    for ln in open(path):
        r = json.loads(ln)
        q, e, o = (r[k] if isinstance(r[k], list) else ast.literal_eval(r[k]) for k in ("query_abs", "edit_code_abs", "edit_opt_abs"))
        if q and e and o:
            rows.append((r["query"], r["reference"], q, e, o))
    return (rows, BM25Okapi([r[2] for r in rows], b=0.4), BM25Okapi([r[3] for r in rows], b=0.4),
            BM25Okapi([r[4] for r in rows], b=0.4))


@h2h
class SBLLM(H2HTechnique):
    """Search-based optimization of the model's own first program: CoT seed, then evolution with execution feedback on the
    public examples, representative-sample selection and BM25 pattern retrieval from PIE (Python)."""
    hid, name, family, target = "H11", "SBLLM", "feedback", ("EFF",)
    code_url = "https://github.com/shuzhenggao/sbllm"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "train.jsonl").exists() and (d / "cot_shots.json").exists():
            return
        with zipfile.ZipFile(_download(SBLLM_DATA, d / "processed_data.zip")) as z:
            (d / "train.jsonl").write_bytes(z.read("python/processed_train.jsonl"))
            with z.open("python/cot.jsonl") as f:
                shots = [s for s in json.loads(f.readline())["prompt_chat"] if "response" in s]
        (d / "cot_shots.json").write_text(json.dumps(shots))

    def solve(self, task: Task, ctx: Ctx) -> str:
        query = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        if not _py(task) or not task.public_tests or not _parses(query):
            return query
        d = ctx.dir(self.hid)
        base = runtime(task, query, SBLLM_RUNS, drop="first")
        if base is None:
            return query
        shots = json.loads((d / "cot_shots.json").read_text())
        msgs = [{"role": "system", "content": SBLLM_COT_SYSTEM.format(lang="python")}]
        for i, s in enumerate(shots):
            msgs += [{"role": "user", "content": f"# Example {i}\n" + s["query"].replace("# Let's think step by step.", "")},
                     {"role": "assistant", "content": s["response"]}]
        msgs.append({"role": "user", "content": f"# Example {len(shots)}\n" + SBLLM_COT_QUERY.format(code=query)})
        best = self._select([self._cand(task, t, base) for t in _draw(ctx, msgs, SBLLM_SEED, temperature=SBLLM_TEMP)])
        ctx.keep(best[0].code)
        patterns = self._patterns(str(d / "train.jsonl"), query, best)
        for _ in range(SBLLM_ITERS):
            perf = "".join(self._render(i, c, base) for i, c in enumerate(best[:SBLLM_BEAM]))
            intro = SBLLM_INTRO_PATTERN if patterns else SBLLM_INTRO
            pat = f"Code transformation pattern:\nPattern 1:\n{patterns[0]}\nPattern 2:\n{patterns[1]}\n" if patterns else ""
            msgs = [{"role": "system", "content": SBLLM_SYSTEM.format(lang="python")},
                    {"role": "user", "content": SBLLM_FORMAT.format(pattern_intro=intro)
                     + SBLLM_QUERY.format(code=query, perf=perf, patterns=pat)}]
            new = [self._cand(task, t, base) for t in _draw(ctx, msgs, SBLLM_GEN, temperature=SBLLM_TEMP)]
            last = [c.code for c in best[:SBLLM_BEAM]]
            best = self._select(best + new)
            ctx.keep(best[0].code)
            if [c.code for c in best[:SBLLM_BEAM]] == last:
                if best[0].acc == 1:
                    break
                continue
            patterns = self._patterns(str(d / "train.jsonl"), query, best)
        return best[0].code

    def _cand(self, task: Task, text: str, base: float) -> _Cand:
        code = C.code_of(task, text)
        m = re.search(r"provide a new optimized code snippet:?\**", text)
        content = text[m.end():].strip() if m and text.count("provide a new optimized code snippet") == 1 else _fence(task, code)
        acc = pass_rate(task, code) if code.strip() else 0.0
        t = runtime(task, code, SBLLM_RUNS, drop="first") if acc == 1 else None
        return _Cand(code, content, acc, t if t is not None else 99999.0)

    @staticmethod
    def _render(i: int, c: _Cand, base: float) -> str:
        sp = base / c.time
        head = ("A correct and optimized version" if c.acc == 1 and sp > 1 else
                "A correct but unoptimized version" if c.acc == 1 else "An incorrect optimized version")
        return f"{head} {i + 1}:\n{c.content.strip()}\nAccuracy: {c.acc} Speedup Rate: {round(sp, 2)}\n"

    @staticmethod
    def _select(cands: list[_Cand]) -> list[_Cand]:
        ok = sorted([c for c in cands if c.acc > 0], key=lambda c: c.time / (0.01 + c.acc))
        wrong = [c for c in cands if c.acc <= 0]
        out, seen = [], []
        for c in ok:
            try:
                key = _abstract_tokens(c.code)
            except (SyntaxError, tokenize.TokenError):
                key = None
            if key is None or key not in seen:
                out.append(c)
                seen.append(key)
        if len(out) < SBLLM_BEAM:
            def dist(c):
                try:
                    a = _abstract_tokens(c.code)
                except (SyntaxError, tokenize.TokenError):
                    return 9999 * len(wrong)
                s = 0
                for o in wrong:
                    if o is c:
                        continue
                    try:
                        s += _lev(a, _abstract_tokens(o.code))
                    except (SyntaxError, tokenize.TokenError):
                        s += 9999
                return s
            out += sorted(wrong, key=dist)
        return out[:SBLLM_KEEP]

    @staticmethod
    def _patterns(path: str, query: str, best: list[_Cand]) -> list[str]:
        import numpy as np
        rows, code_bm25, edit_bm25, opt_bm25 = _sbllm_index(path)
        try:
            q = _abstract_tokens(query)
        except (SyntaxError, tokenize.TokenError):
            return []
        s = np.array(code_bm25.get_scores(q))
        s = (s - s.min()) / max(1e-12, s.max() - s.min())
        sim, dis = s.copy(), s.copy()
        for c in best[:SBLLM_BEAM]:
            try:
                a = _abstract_tokens(c.code)
            except (SyntaxError, tokenize.TokenError):
                continue
            dele, add = _diff_lines(" ".join(q).split("\n"), " ".join(a).split("\n"))
            e = np.array(edit_bm25.get_scores("\n".join(dele).split())) + np.array(opt_bm25.get_scores("\n".join(add).split()))
            if e.max() - e.min() > 0:
                e = (e - e.min()) / (e.max() - e.min())
                med = np.median(e)
                sim += np.where(e < med, 0, e) / SBLLM_BEAM
                dis += np.where(e > med, 0, e.max() - e) / SBLLM_BEAM
        out = []
        for scores in (sim, dis):
            slow, fast = rows[int(np.argmax(scores))][:2]
            diff = difflib.Differ().compare(slow.split("\n"), fast.split("\n"))
            out.append("\n".join(x for x in diff if x.startswith(("- ", "+ "))))
        return out


# ------------------------------------------------------------------------------------------------- H12 PerfCodeGen

PCG_K = 8
PCG_BASE = "You are an expert {lang} programmer, and here is your task: \n{task}"
PCG_REFLECT = ("Your generated solution for the problem is not correct and cannot pass the following test case: \n\n{test} \n\n"
               "The error message is as follows: \n\n\"{error}\" \n\nCould you please analyze the reason of failure and "
               "propose a strategy to modify your solution so that it can pass the above test case?")
PCG_FIX = ("Could you please modify your solution so that it can fulfill the requirements in the problem and pass the test "
           "case?\nGive your solution as follows. Wrap it with \"```{lang}\".")
PCG_OPT = ("Good job! You generated the correct solution for the problem! Now let's step further and optimize the time "
           "performance of the solution.\nBased on the correctly generated solution, could you please refine it so that it "
           "consumes less time in the execution?\nPlease make sure your refined solution is functionally equivalent with the "
           "original solution and do not change the input-output format and the name of the major components.\n"
           "Give your solution as follows. Wrap it with \"```{lang}\".")
PCG_SLOW = ("We tested your optimized program and found that the following test case costs the most time in execution.\n\n"
            "{test}\n\nCould you please refine your optimized program according to the test case below?\nPlease make sure your "
            "refined solution is functionally equivalent with the original solution and do not change the input-output "
            "format and the name of the major components.\nGive your solution as follows. Wrap it with \"```{lang}\".")


@h2h
class PerfCodeGen(H2HTechnique):
    """K=8 samples; one correctness round (reflect, then fix) on a failing public example; one performance round of two
    greedy turns, the second showing the slowest public example; fastest correct program wins."""
    hid, name, family, target = "H12", "PerfCodeGen", "feedback", ("EFF",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        lang = C.LANG_NAME.get(task.language, task.language)
        base = [{"role": "system", "content": C.SYSTEM},
                {"role": "user", "content": PCG_BASE.format(lang=lang, task=C.task_text(task))}]
        texts = _draw(ctx, base, PCG_K)
        ctx.keep(C.code_of(task, texts[0]))
        correct: list[tuple[list, str]] = []
        for text in texts:
            code = C.code_of(task, text)
            hist = base + [{"role": "assistant", "content": text}]
            f = failure(task, code)
            if f:
                hist = hist + _user(PCG_REFLECT.format(test=test_text(f[0]), error=error_text(*f)))
                hist = hist + [{"role": "assistant", "content": ctx.llm.ask(hist, role="reflect")}] + _user(
                    PCG_FIX.format(lang=task.language))
                text = ctx.llm.ask(hist, role="fix")
                code = C.code_of(task, text)
                hist = hist + [{"role": "assistant", "content": text}]
                if failure(task, code):
                    continue
            if not correct:
                ctx.keep(code)
            correct.append((hist, code))
        if not correct:
            return ctx.last_code
        refined = []
        for hist, code in correct:
            h = hist + _user(PCG_OPT.format(lang=task.language))
            text = ctx.llm.ask(h, temperature=0.0, role="optimize")
            opt = C.code_of(task, text)
            if failure(task, opt):
                continue
            times = test_times(task, opt)
            if times is None:
                continue
            slow = task.public_tests[times.index(max(times))]
            h = h + [{"role": "assistant", "content": text}] + _user(PCG_SLOW.format(test=test_text(slow),
                                                                                     lang=task.language))
            opt2 = C.code_of(task, ctx.llm.ask(h, temperature=0.0, role="optimize-slowest"))
            if not failure(task, opt2):
                refined.append(ctx.keep(opt2))
        return self._fastest(task, refined or [c for _, c in correct])

    @staticmethod
    def _fastest(task: Task, codes: list[str]) -> str:
        timed = [(runtime(task, c), i) for i, c in enumerate(codes)]
        ok = [(t, i) for t, i in timed if t is not None]
        return codes[min(ok)[1]] if ok else codes[0]


# ------------------------------------------------------------------------------------------------- H13 PromSec

PROMSEC_REPO = "https://codeload.github.com/mahmoudkanazzal/PromSec/zip/refs/heads/main"
PROMSEC_ITERS = 20
PROMSEC_GAN = {"lr": 1e-3, "epochs": 2, "disc_epochs": 5, "lambda_adv": 1.0}
PROMSEC_G2C = ("You are a coding assistant. Given an original Python code, its AST, the original CFG, and an updated CFG, "
               "generate an updated version of the code that maintains functionality but aligns with the updated CFG "
               "structure. Adjust control structures like loops and conditionals as needed.\n\n"
               "Original Code:\n{code}\n\nOriginal AST:\n{ast}\n\nOriginal CFG:\n{cfg}\n\nUpdated CFG:\n{new_cfg}\n\n"
               "Provide the updated Python code.")
PROMSEC_INVERT = ("\n    Estimate a detailed prompt that could be given to ChatGPT to generate the following Python code making sure "
                  "that the code generated is secure and maintains the same functionality of the code:\n    [{code}]. Try to be "
                  "as specific as possible. The prompt must assure that the code must be in Python, not any other language. "
                  "Do not provide any explanatory text, marks, or comments. Only give the prompt requested.")
PROMSEC_SECRETS = ("In your code generation, ensure that sensitive information (secret keys, passwords)\n    is stored in "
                   "environment variables not hard-coded.")
PROMSEC_TAIL = ". Do not put any explanatory text, marks, or comments. Only give the source code as a response."


def _cfg(code: str) -> tuple[list[list[float]], list[tuple[int, int]]]:
    """Statement-level CFG (one node per statement line) with the four binary node features of the PromSec code, folded
    into one label-encoded feature."""
    tree = ast.parse(code)
    lines = code.splitlines()
    nodes: dict[int, int] = {}
    edges: list[tuple[int, int]] = []

    def node(s: ast.stmt) -> int:
        if s.lineno not in nodes:
            nodes[s.lineno] = len(nodes)
        return nodes[s.lineno]

    def walk(body: list[ast.stmt], prev: int | None) -> int | None:
        for s in body:
            cur = node(s)
            if prev is not None:
                edges.append((prev, cur))
            if isinstance(s, ast.If):
                walk(s.body, cur)
                walk(s.orelse, cur)
            elif isinstance(s, (ast.For, ast.While, ast.AsyncFor)):
                walk(s.body, cur)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.Try)):
                walk(s.body, cur)
            prev = cur
        return prev

    walk(tree.body, None)
    feats = [[0.0]] * len(nodes)
    kinds = {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.stmt)}
    for ln, i in nodes.items():
        s, text = kinds.get(ln), lines[ln - 1] if ln - 1 < len(lines) else ""
        bits = (isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)), isinstance(s, (ast.For, ast.While, ast.AsyncFor)),
                isinstance(s, ast.If), "#" in text)
        feats[i] = [float(sum(b << k for k, b in enumerate(bits)))]
    return feats, edges


def _cfg_str(n: int, m: int) -> str:
    return f"DiGraph with {n} nodes and {m} edges"


def _ggan():
    import torch
    from torch_geometric.nn import GraphConv

    class Gen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = GraphConv(1, 128), GraphConv(128, 1)

        def forward(self, x, e):
            return torch.relu(self.c2(torch.relu(self.c1(x, e)), e))

    class Disc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1, self.c2 = GraphConv(1, 64), GraphConv(64, 1)

        def forward(self, x, e):
            return torch.sigmoid(self.c2(torch.relu(self.c1(x, e)), e))

    return Gen(), Disc()


@lru_cache(maxsize=1)
def _generator(path: str):
    import torch
    gen, _ = _ggan()
    gen.load_state_dict(torch.load(path))
    return gen.eval()


def _tensors(code: str):
    import torch
    x, e = _cfg(code)
    ei = torch.tensor(e, dtype=torch.long).t().contiguous() if e else torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(x if x else [[0.0]], dtype=torch.float), ei


@h2h
class PromSec(H2HTechnique):
    """Released PromSec loop: while Bandit reports CWEs, the gGAN-updated CFG and the code go to the model for a rewrite, the
    rewrite is inverted into a new prompt, and the prompt regenerates the code (20 cycles, last code returned)."""
    hid, name, family, target = "H13", "PromSec", "feedback", ("SEC",)
    code_url = "https://github.com/mahmoudkanazzal/PromSec"
    trains = True

    def prepare(self, ctx: Ctx) -> None:
        import torch
        d = ctx.dir(self.hid)
        if (d / "generator.pt").exists():
            return
        graphs = []
        with zipfile.ZipFile(_download(PROMSEC_REPO, d / "promsec.zip")) as z:
            for n in z.namelist():
                if "/Training_DS/" in n and n.endswith(".py"):
                    src = z.read(n).decode(errors="ignore")
                    if _parses(src):
                        graphs.append(_tensors(src))
        torch.manual_seed(0)
        gen, disc = _ggan()
        og, od = (torch.optim.Adam(m.parameters(), lr=PROMSEC_GAN["lr"]) for m in (gen, disc))
        bce = torch.nn.BCELoss()
        for _ in range(PROMSEC_GAN["epochs"]):
            for x, e in graphs:
                for _ in range(PROMSEC_GAN["disc_epochs"]):
                    od.zero_grad()
                    real, fake = disc(x, e), disc(gen(x, e).detach(), e)
                    loss = bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))
                    loss.backward()
                    od.step()
                og.zero_grad()
                fake = disc(gen(x, e), e)
                (PROMSEC_GAN["lambda_adv"] * bce(fake, torch.ones_like(fake))).backward()
                og.step()
        torch.save(gen.state_dict(), d / "generator.pt")

    def solve(self, task: Task, ctx: Ctx) -> str:
        import torch
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        if not _py(task):
            return code
        gen = _generator(str(ctx.dir(self.hid) / "generator.pt"))
        for _ in range(PROMSEC_ITERS):
            cwes = {r.get("issue_cwe", {}).get("id") for r in bandit_results(code)[0]}
            if not cwes or not _parses(code):
                break
            x, e = _tensors(code)
            with torch.no_grad():
                updated = gen(x, e)
            cfg, new_cfg = _cfg_str(x.shape[0], e.shape[1]), _cfg_str(updated.shape[0], e.shape[1])
            fixed = C.code_of(task, ctx.llm.ask(PROMSEC_G2C.format(
                code=code, ast=ast.dump(ast.parse(code), annotate_fields=True, include_attributes=True), cfg=cfg,
                new_cfg=new_cfg), temperature=0.0, role="graph-to-code"))
            prompt = ctx.llm.ask(PROMSEC_INVERT.format(code=fixed), temperature=0.0, role="invert")
            code = ctx.keep(C.code_of(task, ctx.llm.ask(PROMSEC_SECRETS + prompt + PROMSEC_TAIL, temperature=0.0,
                                                         role="regenerate")))
        return code


# ------------------------------------------------------------------------------------------------- H14 DeepDevVuln

DDV_CORPUS = "codeparrot/github-code-clean"
DDV_LANGS = {"python": ("Python", "tree_sitter_python"), "javascript": ("JavaScript", "tree_sitter_javascript")}
DDV_FILES_PER_LANG = 200_000
DDV_DETECTOR = "microsoft/codebert-base"
DDV_TRAIN = {"epochs": 10, "lr": 2e-5, "batch": 32, "max_len": 512, "positive_rate": 0.01}
DDV_BLOCK_SUFFIXES = ("_statement", "_definition", "_declaration", "_clause")
DDV_COMPLETIONS = 25


def _blocks(lang: str, src: bytes) -> list[tuple[int, int]]:
    from importlib import import_module
    from tree_sitter import Language, Parser
    parser = Parser(Language(import_module(DDV_LANGS[lang][1]).language()))
    out, stack = [], [parser.parse(src).root_node]
    while stack:
        n = stack.pop()
        if n.type.endswith(DDV_BLOCK_SUFFIXES) or n.type == "method_definition":
            out.append((n.start_byte, n.end_byte))
        stack.extend(n.children)
    return out


@h2h
class DeepDevVuln(H2HTechnique):
    """A CodeBERT vulnerability detector, retrained by the paper's recipe (CodeQL alerts on GitHub code, blocks split before
    the alert), filters completions: drawn one by one, the first unflagged completion is returned."""
    hid, name, family, target = "H14", "DeepDevVuln filtering of LLM completions", "feedback", ("SEC",)
    trains, data_rebuilt = True, True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "detector" / "threshold.json").exists():
            return
        rows = d / "examples.jsonl"
        if not rows.exists():
            self._mine(rows)
        self._train(rows, d / "detector")

    def _mine(self, out: Path) -> None:
        from datasets import load_dataset
        with open(out, "w") as f:
            for lang, (ds_lang, _) in DDV_LANGS.items():
                repos: dict[str, dict[str, str]] = defaultdict(dict)
                rows = load_dataset(DDV_CORPUS, streaming=True, split="train", languages=[ds_lang], trust_remote_code=True)
                for i, r in enumerate(rows):
                    if i >= DDV_FILES_PER_LANG:
                        break
                    repos[r["repo_name"]][r["path"]] = r["code"]
                for repo, files in repos.items():
                    for ex in self._examples(lang, repo, files):
                        f.write(json.dumps(ex) + "\n")

    @staticmethod
    def _examples(lang: str, repo: str, files: dict[str, str]):
        with Workdir({f"src/{p}": c for p, c in files.items()}) as d:
            ok = execute([CODEQL, "database", "create", str(d / "db"), f"--language={lang}", f"--source-root={d / 'src'}",
                          "--build-mode=none", "--overwrite"], d, timeout=1800, mem_mb=TOOL_MEM_MB).ok
            ok = ok and execute([CODEQL, "database", "analyze", str(d / "db"), CODEQL_SUITE.format(lang=lang),
                                 "--format=sarif-latest", f"--output={d / 'out.sarif'}"], d, timeout=1800, mem_mb=TOOL_MEM_MB).ok
            sarif = json.loads((d / "out.sarif").read_text()) if ok and (d / "out.sarif").exists() else None
        if sarif is None:
            return
        alerts: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for res in (sarif.get("runs") or [{}])[0].get("results", []):
            loc = (res.get("locations") or [{}])[0].get("physicalLocation", {})
            reg = loc.get("region", {})
            alerts[loc.get("artifactLocation", {}).get("uri", "")].append((reg.get("startLine", 1), reg.get("startColumn", 1)))
        rng = random.Random(repo)
        for path, code in files.items():
            src = code.encode()
            starts = [0] + [i + 1 for i, b in enumerate(src) if b == 10]
            offs = [starts[min(ln, len(starts)) - 1] + col - 1 for ln, col in alerts.get(path, [])]
            for a, b in _blocks(lang, src):
                inside = [o for o in offs if a <= o < b]
                end = min(inside) if inside else b
                cut = rng.randrange(a, end) if end > a else a
                yield {"lang": lang, "repo": repo, "context": src[:cut].decode(errors="ignore")[-4000:],
                       "code": src[cut:b].decode(errors="ignore"), "label": int(bool(inside))}

    @staticmethod
    def _split(repo: str) -> str:
        h = int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 100
        return "train" if h < 85 else "val" if h < 90 else "test"

    def _train(self, rows: Path, out: Path) -> None:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        data = {"train": [], "val": []}
        for ln in open(rows):
            r = json.loads(ln)
            s = self._split(r["repo"])
            if s in data:
                data[s].append(r)
        tok = AutoTokenizer.from_pretrained(DDV_DETECTOR, truncation_side="left")
        model = AutoModelForSequenceClassification.from_pretrained(DDV_DETECTOR, num_labels=1).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=DDV_TRAIN["lr"])
        pos = [r for r in data["train"] if r["label"]]
        neg = [r for r in data["train"] if not r["label"]]
        rng = random.Random(0)
        for _ in range(DDV_TRAIN["epochs"]):
            epoch = pos + rng.sample(neg, min(len(neg), len(pos)))
            rng.shuffle(epoch)
            model.train()
            for i in range(0, len(epoch), DDV_TRAIN["batch"]):
                b = epoch[i:i + DDV_TRAIN["batch"]]
                enc = tok([r["context"] for r in b], [r["code"] for r in b], truncation="only_first",
                          max_length=DDV_TRAIN["max_len"], padding=True, return_tensors="pt").to("cuda")
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(**enc).logits.squeeze(-1), torch.tensor([float(r["label"]) for r in b], device="cuda"))
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        scores = np.concatenate([self._scores(model, tok, [(r["context"], r["code"]) for r in data["val"][i:i + 64]])
                                 for i in range(0, len(data["val"]), 64)])
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        (out / "threshold.json").write_text(json.dumps({"threshold": float(np.quantile(scores, 1 - DDV_TRAIN["positive_rate"]))}))

    @staticmethod
    def _scores(model, tok, pairs):
        import torch
        enc = tok([c for c, _ in pairs], [v for _, v in pairs], truncation="only_first", max_length=DDV_TRAIN["max_len"],
                  padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            return torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy().reshape(-1)

    @lru_cache(maxsize=1)
    def _detector(self, path: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        model = model.cuda() if torch.cuda.is_available() else model
        return model, AutoTokenizer.from_pretrained(path), json.loads((Path(path) / "threshold.json").read_text())["threshold"]

    def solve(self, task: Task, ctx: Ctx) -> str:
        msgs = C.messages(task)
        if task.language not in DDV_LANGS:
            return C.code_of(task, ctx.llm.ask(msgs))
        model, tok, thr = self._detector(str(ctx.dir(self.hid) / "detector"))
        best, best_score = "", 2.0
        for _ in range(DDV_COMPLETIONS):
            code = C.code_of(task, ctx.llm.sample(msgs, 1)[0])
            score = float(self._scores(model, tok, [(task.prompt, code)])[0])
            if score < best_score:
                best, best_score = ctx.keep(code), score
            if score < thr:
                return code
        return best


# ------------------------------------------------------------------------------------------------- H15 FRANC

FRANC_K = 10
FRANC_SYSTEM = "You are a coding assistant. You generate only source code."


def _franc_clean(task: Task, code: str) -> str:
    """Static filter rules R3 and R5: drop top-level code after the target function and trailing incomplete lines."""
    lines = code.rstrip().splitlines()
    while lines and not _parses("\n".join(lines)):
        lines.pop()
    code = "\n".join(lines) if lines else code
    if task.mode != "function" or not task.entry_point or not _parses(code):
        return code
    body = ast.parse(code).body
    idx = next((i for i, n in enumerate(body) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == task.entry_point), None)
    if idx is None:
        return code
    keep = body[:idx + 1] + [n for n in body[idx + 1:] if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                          ast.ClassDef, ast.Import, ast.ImportFrom))]
    return "\n".join(ast.get_source_segment(code, n) or "" for n in keep)


@h2h
class FRANC(H2HTechnique):
    """Ten suggestions ranked by quality (compilable and Bandit-clean first); if none is clean, one repair round with the
    P2 prompt (Bandit messages as # Fix comments plus the original prompt), ten more suggestions, ranked again."""
    hid, name, family, target = "H15", "FRANC (analyzer-guided repair prompting)", "feedback", ("SEC", "RM")
    code_url = "https://github.com/s2e-lab/FRANC"

    def solve(self, task: Task, ctx: Ctx) -> str:
        msgs = [{"role": "system", "content": FRANC_SYSTEM}, {"role": "user", "content": C.task_text(task)}]
        ranked = self._rank(task, _draw(ctx, msgs, FRANC_K, temperature=0.0))
        ctx.keep(ranked[0][0])
        if ranked[0][1] or not _py(task):
            return ranked[0][0]
        seed = next(((c, res) for c, _, res in ranked if res), None)
        if seed is None:
            return ranked[0][0]
        code, res = seed
        fixes = "".join(f"# Fix: At line {r.get('line_number', 0)}, {r.get('issue_text', '')}\n" for r in res)
        repair = f"{code}\n{fixes}\n{task.prompt}\n# Fixed Code: \n"
        texts = _draw(ctx, [{"role": "system", "content": FRANC_SYSTEM}, {"role": "user", "content": repair}], FRANC_K,
                      temperature=0.0)
        best = self._rank(task, texts)[0][0]
        if len(texts) < FRANC_K:
            _ceiling(ctx, best)
        return best

    def _rank(self, task: Task, texts: list[str]) -> list[tuple[str, bool, list[dict]]]:
        out = []
        for t in texts:
            code = _franc_clean(task, C.code_of(task, t))
            if _py(task):
                res, compiles = bandit_results(code)
            elif task.language in LANGS:
                with Workdir({LANGS[task.language]["file"]: code}) as d:
                    res, compiles = [], build(task.language, d).ok
            else:
                res, compiles = [], True
            out.append((code, compiles and not res, res))
        return sorted(out, key=lambda x: not x[1])


# ------------------------------------------------------------------------------------------------- H16 Optimo

OPTIMO_ZIP = "https://zenodo.org/records/19335907/files/Optimo-047E.zip?download=1"
OPTIMO_SYSTEM = "You are an expert on code optimization."
OPTIMO_TEMP, OPTIMO_FIX_ROUNDS, OPTIMO_RUNS, OPTIMO_MAX_STRATEGIES = 0.3, 3, 8, 10
OPTIMO_HEAD = '''
We are optimizing the time efficiency for a Python program{given}.

{gap}### Program
Here is the Python program:
```python
{{PROGRAM}}
```

### Problem Description
Here is the problem description of the program:
{{DESC}}
'''
OPTIMO_IO_RULE = '''If the original program interacts with stdin and stdout use input() and print(), you should keep this format and do NOT only output a funciton.'''
OPTIMO_ROUTE = OPTIMO_HEAD.format(given="", gap="") + '''
### Instruction

1. Analyze the time complexity of the above program.
Generally you could select one from the following choices:
{TIME}

2. Analyze the high-level algorithm the program used.
Generally you could select one from the following algoirthm types:
{ALG}

3. If you find the time complexity or the algorithm type of the program is not in the above choices, output `"Others"` in the related field.

4. Your answer must be as accurate as possible since this is the key information we use to select appropriate optimization strategies.

5. Your answer should exactly match the above choices if you select one from them.


### Output
Your output should follow the JSON format below:
```json
{
    "Time Complexity": xxx,
    "Algorithm Type": xxx
}
```
'''
OPTIMO_SELECT = OPTIMO_HEAD.format(given=" given some existing optimization strategies", gap="") + '''

### Optimization Strategy
Here are some optimization strategies:
```
{STRATEGY}
```

### Instruction

1. Review the given program and all optimization strategies, select the most appropriate optimization strategies that could be used to optimize the program.
2. Carefully compare the functionality of the Python program with the optimization strategies and select only related optimization strategies.
3. You can select at most 10 optimization strategies.
4. Output the IDs of optimization strategies you selected in the JSON object.
5. Output the explanations of why you select these strategies.
6. If you think none of optimization strategies are reasonable, output an empty list for the field.


### Output
Your output should follow the JSON format below:
```json
{
    "Selected Strategy IDs": list[int],
    "Explanation": xxx
}
```

'''
OPTIMO_SINGLE = OPTIMO_HEAD.format(given=" given a existing optimization strategy", gap="\n") + '''
### Optimization Strategy
Here is the optimization strategy:
```
{STRATEGY}
```

### Instruction

1. Carefully apply the instructions in the `Modification` field of the optimization strategy into the Python program.
2. Generate the optimized Python program that is functionally equivalent with the original Python program.
3. Verify if the optimized program meets the new time complexity and algorithm in the selected strategy.
4. Output the ID of optimization strategy you selected and the optimized Python program.
5. If you find that the provided optimization strategy does not completely work for the current Python program, you could generate a new program by using the similar insight in the optimization strategy.
6. Do not change the function name or I/O format in the original program, as this may lead to test failures.
7. Give your explanation about how you optimize the original program in the JSON output.
8. The optimized program should be a string and be put in a valid JSON object.
9. Do not add any sample usage to the optimized program.
10. You should keep the I/O format in the original program. If the original program is only a function, the optimized program should also be a funciton.
''' + OPTIMO_IO_RULE + '''


### Output
Your output should follow the JSON format below:
```json
{
    "Selected Strategy ID": xxx,
    "Optimized Program": xxx,
    "Explanation": xxx
}
```
'''
OPTIMO_PLAIN = OPTIMO_HEAD.format(given="", gap="\n") + '''
### Instruction

1. Analyze the algorithm used in the original Python program.
2. Generate a new program by conducting a algorithm-level optimization.
3. Only algorithm-level optimization is allowed. Do not involve optimizations in other levels.
4. The optimized Python program MUST be functionally equivalent with the original Python program.
5. Do not change the function name or I/O format in the original program, as this may lead to test failures.
6. Give your explanation about how you optimize the original program in the JSON output.
7. The optimized program should be a string and be put in a valid JSON object. If you cannot find an algorithm-level optimization, output `None` for the field.
8. You should keep the I/O format in the original program. If the original program is only a function, the optimized program should also be a funciton.
''' + OPTIMO_IO_RULE + '''


### Output
Your output should follow the JSON format below:
```json
{
    "Optimized Program": xxx,
    "Explanation": xxx
}
```
'''
OPTIMO_FIX_BODY = '''
### Previous Incorrect Optimized Program
Here is the previous incorrect optimized program you gave based on the above optimization strategy:
```python
{INCORRECT_PROGRAM}
```

### Failed Test Case
Here is the test case the above incorrect optimized program failed:
```
{TESTCASE}
```

### Instruction

1. Analyze the reason why the previous optimized program failed.
2. Fix the previous optimized program so that the failed test cases can be passed.
'''
OPTIMO_FIX_TAIL = '''Do not change the function name or I/O format in the original program, as this may lead to test failures.
{n1}. Give your explanation about how you optimize the original program in the JSON output.
{n2}. The optimized program should be a string and be put in a valid JSON object.
{n3}. Do not add any sample usage to the optimized program.


### Output
Your output should follow the JSON format below:
```json
{{
    "Fixed Program": xxx,
    "Explanation": xxx
}}
```
'''
OPTIMO_FIX = (OPTIMO_HEAD.format(given=" given a existing optimization strategy", gap="\n") + '''
### Optimization Strategy
Here is the optimization strategy:
```
{STRATEGY}
```
''' + OPTIMO_FIX_BODY + "3. Make sure the instructions in the `Modification` field of the optimization strategy have been applied "
              "in your fixed program.\n4. " + OPTIMO_FIX_TAIL.format(n1=5, n2=6, n3=7))
OPTIMO_PLAIN_FIX = OPTIMO_HEAD.format(given="", gap="") + OPTIMO_FIX_BODY + "3. " + OPTIMO_FIX_TAIL.format(n1=4, n2=5, n3=6)
OPTIMO_API = OPTIMO_HEAD.format(given=" given some existing optimization patterns", gap="") + '''
### API
We have indentified the following APIs used in the program that may be suboptimal.
```
{API}
```

### Instruction
1. Review the given program and all optimization patterns, select the most appropriate optimization pattern for each API that could be used to optimize the API in the original program.
2. For one API, you are given one or more optimization patterns, select at most one pattern for each API.
3. For any API, if you find there is no proper optimization patterns for it, you could skip the optimization of it.
4. For any selected optimization pattern, you should apply it to replace the original suboptimal API in the original program.
5. The final optimized program should contain all optimized APIs after applying all of your selected patterns.
6. Do not change the function name or I/O format in the original program, as this may lead to test failures.
7. Give your explanation about how you optimize the original program in the JSON output.
8. The optimized program should be a string and be put in a valid JSON object.
9. In the JSON object, "API optimizations" is a dict, where the key should be the original suboptimal API occurrances and the value should be the optimized API according to the selected optimization pattern.
10. You should keep the I/O format in the original program. If the original program is only a function, the optimized program should also be a funciton.
''' + OPTIMO_IO_RULE + '''

### Output
Your output should follow the JSON format below, DO NOT add other contents:
```json
{
    "Selected Pattern ID": list[int],
    "API Optimizations": dict[str, str],
    "Optimized Program": str,
    "Explanation": str
}
```
'''


def _fill(template: str, **kw: str) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", v)
    return template


def _optimo_case(t: Test, r: Run) -> str:
    inp = t.stdin if t.kind == "io" else t.code
    if r.returncode != 0 or r.timeout:
        return f"The optimized program crashes given the input:\n{inp}\nThe expected output should be:\n{t.expected}\n"
    if t.kind == "io" and not r.stdout.strip():
        return (f"The optimized program does not give any output given the input:\n{inp}\nThis may be that the program must "
                "read input from stdin and print output to stdout but there is no such I/O operations in the program.")
    return f"The optimized program produces the wrong output given the input:\n{inp}\n{error_text(t, r)}\n"


@h2h
class Optimo(H2HTechnique):
    """Mixture of prompts over the mined strategy base (released with the paper): algorithm level (route, select, one prompt
    per strategy plus a shared prompt), correctness reflection (3 rounds), timing against the original; then the API level."""
    hid, name, family, target = "H16", "Optimo", "feedback", ("EFF",)
    code_url = "https://doi.org/10.5281/zenodo.19335907"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "api_map.json").exists():
            return
        with zipfile.ZipFile(_download(OPTIMO_ZIP, d / "optimo.zip")) as z:
            for n in z.namelist():
                if "data/strategies/" in n and n.endswith(".json"):
                    (d / Path(n).name).write_bytes(z.read(n))

    def _query(self, ctx: Ctx, prompt: str, role: str) -> dict | None:
        return _json(ctx.llm.ask([{"role": "system", "content": OPTIMO_SYSTEM}, {"role": "user", "content": prompt}],
                                 temperature=OPTIMO_TEMP, max_tokens=4096, role=role))

    def solve(self, task: Task, ctx: Ctx) -> str:
        program = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        if not _py(task) or not task.public_tests or failure(task, program):
            return program
        d = ctx.dir(self.hid)
        strategies = json.loads((d / "optimization_strategies.json").read_text())["Algorithm"]
        api_map = json.loads((d / "api_map.json").read_text())
        desc = task.prompt
        times = sorted(strategies)
        algs = sorted({a for t in strategies.values() for a in t})
        route = self._query(ctx, _fill(OPTIMO_ROUTE, PROGRAM=program, TIME="\n ".join(times), ALG="\n ".join(algs), DESC=desc),
                            "route") or {}
        tc, at = route.get("Time Complexity"), route.get("Algorithm Type")
        pool = strategies.get(tc, {})
        candidates = pool.get(at) or [s for a in pool.values() for s in a]
        results: list[tuple[str, dict | None]] = []
        if candidates:
            listing = "".join(json.dumps(self._alg_view(s)) + "\n\n" for s in candidates)
            sel = self._query(ctx, _fill(OPTIMO_SELECT, PROGRAM=program, STRATEGY=listing, DESC=desc), "select") or {}
            ids = {str(i) for i in sel.get("Selected Strategy IDs") or []}
            for s in [s for s in candidates if str(s["id"]) in ids][:OPTIMO_MAX_STRATEGIES]:
                out = self._query(ctx, _fill(OPTIMO_SINGLE, PROGRAM=program, STRATEGY=json.dumps(self._alg_view(s)) + "\n\n",
                                            DESC=desc), "optimize")
                if out and out.get("Optimized Program"):
                    results.append((str(out["Optimized Program"]), s))
        out = self._query(ctx, _fill(OPTIMO_PLAIN, PROGRAM=program, DESC=desc), "optimize-shared")
        if out and out.get("Optimized Program") not in (None, "None"):
            results.append((str(out["Optimized Program"]), None))
        program = ctx.keep(self._validate(ctx, task, program, desc, results))
        apis = self._apis(program)
        pairs = {a: [i for i, m in api_map.items() if a in m["api"]["ineffective"]] for a in apis}
        pairs = {a: ids for a, ids in pairs.items() if ids}
        if not pairs:
            return program
        listing = ""
        for k, (a, ids) in enumerate(pairs.items()):
            listing += f"=====API #{k + 1}:   {a}=====\nOccurrances in the original Program:\n"
            listing += "".join(f"```python\n{occ}\n```\n" for occ in apis[a]) + "Candidate Optimization Patterns:\n\n"
            for i in ids:
                listing += (f"ID: {i}\nInefficient API:\n```python\n{api_map[i]['api']['ineffective']}\n```\n"
                            f"Efficient API:\n```python\n{api_map[i]['api']['code']}\n```\n\n")
        out = self._query(ctx, _fill(OPTIMO_API, PROGRAM=program, API=listing, DESC=desc), "optimize-api")
        if out and out.get("Optimized Program"):
            program = ctx.keep(self._validate(ctx, task, program, desc, [(str(out["Optimized Program"]), None)]))
        return program

    @staticmethod
    def _alg_view(s: dict) -> dict:
        return {k: s[k] for k in ("id", "New Algorithm", "Time Complexity of New Code", "Description", "Modification")}

    @staticmethod
    def _apis(code: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for n in ast.walk(ast.parse(code)):
            if isinstance(n, ast.Call):
                seg = ast.get_source_segment(code, n) or ""
                out[seg.split("(")[0]].append(seg)
        return out

    def _validate(self, ctx: Ctx, task: Task, original: str, desc: str, results: list[tuple[str, dict | None]]) -> str:
        """Correctness reflection (first failing public example, up to three rounds) and timing against the original."""
        for _ in range(OPTIMO_FIX_ROUNDS):
            pending = [(i, failure(task, c)) for i, (c, _) in enumerate(results)]
            pending = [(i, f) for i, f in pending if f]
            if not pending:
                break
            for i, f in pending:
                code, s = results[i]
                case = _optimo_case(*f)
                if s is None:
                    p = _fill(OPTIMO_PLAIN_FIX, PROGRAM=original, DESC=desc, INCORRECT_PROGRAM=code, TESTCASE=case)
                else:
                    view = {"id": s["id"], "Description": s["Description"], "Modification": s["Modification"],
                            "New Algorithm": s["New Algorithm"], "Time Complexity of New Code": s["Time Complexity of New Code"]}
                    p = _fill(OPTIMO_FIX, PROGRAM=original, DESC=desc, STRATEGY=json.dumps(view), INCORRECT_PROGRAM=code,
                             TESTCASE=case)
                out = self._query(ctx, p, "fix")
                if out and out.get("Fixed Program"):
                    results[i] = (str(out["Fixed Program"]), s)
        best, best_t = original, runtime(task, original, OPTIMO_RUNS, drop="none")
        for code, _ in results:
            if failure(task, code):
                continue
            t = runtime(task, code, OPTIMO_RUNS, drop="none")
            if t is not None and (best_t is None or t < best_t):
                best, best_t = code, t
        return best


# ------------------------------------------------------------------------------------------------- H17 Infer repair loop

DOLCETTI_ITERS = 6
DOLCETTI_TEMP = 0.2
DOLCETTI_GEN = '''You are a chatbot whose purpose is to provide the code implementation in the {lang} programming language for a task that is given to you as input.
The code must solve the task correctly, compile, and be safe, so you must include the required libraries. You must follow the signature provided.
You must wrap the code between "//BEGIN" and "//END".
An example of correct generated code is  the following:
PROMPT: Write a {lang} function to return the sum of two integers. The signature of the function is {sig}.
OUTPUT:
//BEGIN
{good}
//END
An example of incorrect generated code is the following:
PROMPT: Write a {lang} function to return the sum of two integers. The signature of the function is {sig}.
OUTPUT:
//BEGIN
{bad}
Your answer must be in the following format:
//BEGIN
<code>
//END
<reasoning>
Let's work this out in a step by step way to be sure we have the right answer.'''
DOLCETTI_EXAMPLE = {
    "python": ("sum(a: int, b: int) -> int", "def sum(a: int, b: int) -> int:\n    return a + b",
               "def sum(a: int, b: int) -> int:\n    return a - b"),
    "cpp": ("int sum(int a, int b)", "int sum(int a, int b) {\n    return a + b;\n}", "int sum(int a, int b) {\n    return a - b;\n}"),
}
DOLCETTI_TESTFIX = '''You are a chatbot whose purpose is to correct an incorrect function, written in the {lang} programming language, to solve a task.
The function, the task to solve, and the expected signature of the function are given to you as inputs as well as the counterexamples for which the code is incorrect.
Provide the corrected code and wrap the code between "//BEGIN" and "//END".'''
DOLCETTI_SAFEFIX = '''You are a chatbot whose purpose is to fix the vulnerabilities found the code implementation in the {lang} programming language for a task that is given to you as input.
Provide the corrected code and wrap the code between "//BEGIN" and "//END". The list description of the vulnerabilities is the following:
{descriptions}
These are instruction on how to fix the vulnerabilities:
{instructions}'''
INFER_KINDS = {
    "NULLPTR_DEREFERENCE": ("When it is possible that the null pointer is dereferenced.",
                            "Before dereferencing a pointer, always check if it is NULL. Ensure all pointers are initialized before use."),
    "UNINITIALIZED_VALUE": ("A value is read before it has been initialized.", "Always initialize variables before their first use."),
    "BUFFER_OVERRUN": ("This is reported when outside of buffer bound it is accessed (for example, array size:[3,3], offset: [5,5]) "
                       "or maybe be accessed (for example, array size:[3,3], offset: [0,5]).",
                       "Always check that array indices are within bounds before accessing elements."),
    "MEMORY_LEAK": ("When something is created with malloc and not freed.", "Always free memory allocated with malloc using free."),
}
BANDIT_GROUPS = {"B1": "MISC", "B2": "CONFIGURATION", "B3": "DANGEROUS_CALL", "B4": "DANGEROUS_IMPORT", "B5": "WEAK_CRYPTO",
                 "B6": "INJECTION", "B7": "XSS"}
BANDIT_KINDS = {
    "MISC": ("Miscellaneous weaknesses such as assert used for checks, hard-coded passwords, binding to all interfaces, "
                   "insecure temporary files or silently ignored exceptions.",
           "Do not rely on assert for checks, never hard-code secrets, create temporary files with tempfile, and handle "
           "exceptions explicitly."),
    "CONFIGURATION": ("Insecure framework configuration such as running Flask with debug=True.",
           "Disable debug modes and keep secure framework defaults."),
    "DANGEROUS_CALL": ("Calls to dangerous functions such as pickle, marshal, eval, md5 or sha1, insecure random, "
                             "urlopen with arbitrary schemes or unsafe XML parsers.",
           "Replace them with safe alternatives: json instead of pickle, ast.literal_eval instead of eval, sha256 or stronger "
           "hashes, the secrets module instead of random, defusedxml for XML."),
    "DANGEROUS_IMPORT": ("Imports of modules that are insecure by design such as telnetlib, ftplib, pickle or xml.etree.",
           "Import a safe alternative, or use the module only on validated, trusted input."),
    "WEAK_CRYPTO": ("Weak cryptography or TLS settings such as disabled certificate verification, old SSL/TLS versions "
                          "or short keys.",
           "Verify certificates, use TLS 1.2 or later and keys of at least 2048 bits."),
    "INJECTION": ("Injection such as shell=True, os.system, SQL built from strings or unsafe YAML loading.",
           "Pass argument lists without a shell, use parameterized queries and yaml.safe_load."),
    "XSS": ("Cross-site scripting such as templates without autoescaping or mark_safe on untrusted data.",
           "Enable autoescaping and escape untrusted data."),
}
DOLCETTI_TEST_USER = "Task:\n{task}\n\nIncorrect code:\n//BEGIN\n{code}\n//END\n\nCounterexample:\n{test}\n\nError:\n{error}"
DOLCETTI_SAFE_USER = ("Task:\n{task}\n\nPrevious solution:\n//BEGIN\n{code}\n//END\n\nVulnerability found: {kind}, "
                      "severity {severity}, line {line}: {qualifier}")


def _between(task: Task, text: str) -> str:
    m = re.search(r"//BEGIN\s*\n(.*?)\n\s*//END", text or "", re.S)
    return m.group(1).strip() if m else C.code_of(task, text)


def infer_findings(code: str, lang: str) -> list[dict]:
    with Workdir({LANGS[lang]["file"]: code}) as d:
        execute(INFER + INFER_COMPILE[lang], d, timeout=300, mem_mb=TOOL_MEM_MB)
        rep = d / "infer-out" / "report.json"
        return json.loads(rep.read_text()) if rep.exists() else []


@h2h
class IterativeStaticAnalysis(H2HTechnique):
    """Dolcetti et al.: Combo generation prompt, test repair one failing example at a time (up to 6), then static-analysis
    repair one finding at a time with fix instructions (up to 6). Infer for C/C++; for Python the Bandit port."""
    hid, name, family, target = "H17", "Iterative static-analysis", "feedback", ("SEC",)
    ported = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        lang = C.LANG_NAME.get(task.language, task.language)
        sig, good, bad = DOLCETTI_EXAMPLE.get(task.language, DOLCETTI_EXAMPLE["python"])
        text = ctx.llm.ask([{"role": "system", "content": DOLCETTI_GEN.format(lang=lang, sig=sig, good=good, bad=bad)},
                            {"role": "user", "content": C.task_text(task)}], temperature=DOLCETTI_TEMP)
        code = ctx.keep(_between(task, text))
        for _ in range(DOLCETTI_ITERS):
            f = failure(task, code)
            if not f:
                break
            code = ctx.keep(_between(task, ctx.llm.ask(
                [{"role": "system", "content": DOLCETTI_TESTFIX.format(lang=lang)},
                 {"role": "user", "content": DOLCETTI_TEST_USER.format(task=C.task_text(task), code=code,
                                                                       test=test_text(f[0]), error=error_text(*f))}],
                temperature=DOLCETTI_TEMP, role="test-repair")))
        kinds = INFER_KINDS if task.language in INFER_COMPILE else BANDIT_KINDS
        system = DOLCETTI_SAFEFIX.format(lang=lang, descriptions=",\n".join(f"{n} : {d}" for n, (d, _) in kinds.items()),
                                         instructions="\n".join(f"{n}: {i}" for n, (_, i) in kinds.items()))
        for _ in range(DOLCETTI_ITERS):
            found = self._findings(task, code)
            if not found:
                break
            kind, severity, line, qualifier = found[0]
            code = ctx.keep(_between(task, ctx.llm.ask(
                [{"role": "system", "content": system},
                 {"role": "user", "content": DOLCETTI_SAFE_USER.format(task=C.task_text(task), code=code, kind=kind,
                                                                       severity=severity, line=line, qualifier=qualifier)}],
                temperature=DOLCETTI_TEMP, role="safety-repair")))
        return code

    @staticmethod
    def _findings(task: Task, code: str) -> list[tuple] | None:
        if task.language in INFER_COMPILE:
            return [(b.get("bug_type"), b.get("severity"), b.get("line"), b.get("qualifier"))
                    for b in infer_findings(code, task.language) if b.get("bug_type") in INFER_KINDS]
        if _py(task):
            return [(BANDIT_GROUPS.get(r["test_id"][:2], "MISC"), r.get("issue_severity"), r.get("line_number"),
                     f"{r['test_id']} {r.get('issue_text', '')}") for r in bandit_results(code)[0]]
        return None


# ------------------------------------------------------------------------------------------------- H18 CodeEvolve

EVOLVE = {"iterations": 20, "islands": 5, "elite": 0.7, "island": 0.2, "top_programs": 3, "diverse_programs": 2,
          "temperature": 0.7, "max_tokens": 4096, "llm_feedback_weight": 0.1, "tau_time_share": 0.1, "tau_calls": 10_000}
EVOLVE_PROMPT = '''You are optimizing one writable code region in a larger codebase.
Goal:
- Improve runtime performance without changing observable behavior.
- Preserve public APIs, exception behavior, and existing tests.
- Return only a minimal patch for the writable region.
Runtime profile:
- Target: {target}
- Cumulative time: {time}
- Call count: {calls}
- Allocation/CPU notes: {notes}
Writable code:
# EVOLVE-BLOCK-START
{writable}
# EVOLVE-BLOCK-END
Read-only context:
{context}
Evaluation feedback:
{feedback}
Constraints:
- Do not modify read-only context.
- Prefer simple, maintainable changes.
- Avoid introducing new dependencies unless already present.
- Explain the optimization briefly, then provide the patch.

Write the patch as one or more blocks of this form, where SEARCH text is copied exactly from the writable code:
<<<<<<< SEARCH
original lines
=======
replacement lines
>>>>>>> REPLACE'''
EVOLVE_REPAIR = ("The candidate below failed validation.\n\n```python\n{code}\n```\n\nDiagnostics:\n{diag}\n\n"
                 "Repair the candidate so it builds and passes the tests, keeping its optimization. Return the complete "
                 "program in one ```python block.")
EVOLVE_REVIEW = ("Review this program for performance and maintainability. Reply with a JSON object "
                 "{{\"score\": <number between 0 and 1>, \"comment\": \"...\"}}.\n\n```python\n{code}\n```")
_DIFF = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


@dataclass
class _Prog:
    code: str
    score: float
    speedup: float
    note: str
    island: int


@h2h
class CodeEvolve(H2HTechnique):
    """OpenEvolve-style island evolution aimed at the profiled hotspot (cProfile in place of JFR): diff edits, an evaluation
    cascade (build, public examples, timing, Pylint, model review) and a refinement agent for failed candidates.
    MCTS is left out: its exploitation probability and branching factor are not published."""
    hid, name, family, target = "H18", "CodeEvolve", "feedback", ("EFF",)
    ported = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        if not _py(task) or not task.public_tests or failure(task, code):
            return code
        base = runtime(task, code)
        if base is None:
            return code
        target, t_cum, calls = self._target(task, code)
        archive = [_Prog(code, self._fitness(code, 1.0, 0.5), 1.0, "initial program", 0)]
        rng = random.Random(current_scope().seed)
        for it in range(EVOLVE["iterations"]):
            island = it % EVOLVE["islands"]
            parent = self._parent(archive, island, rng)
            writable, context = self._split(parent.code, target)
            feedback = "\n".join(f"- speedup {p.speedup:.2f}, score {p.score:.3f}: {p.note}"
                                 for p in sorted(archive, key=lambda p: -p.score)[:EVOLVE["top_programs"]]
                                 + rng.sample(archive, min(EVOLVE["diverse_programs"], len(archive))))
            reply = ctx.llm.ask(EVOLVE_PROMPT.format(target=target or "whole program", time=f"{t_cum:.4f}s", calls=calls,
                                                     notes="cProfile on the public examples", writable=writable,
                                                     context=context or "(none)", feedback=feedback),
                                temperature=EVOLVE["temperature"], max_tokens=EVOLVE["max_tokens"], role="mutate")
            child = self._apply(parent.code, reply)
            if child is None:
                continue
            f = failure(task, child) if _parses(child) else (task.public_tests[0], Run(False, "", "SyntaxError", 1))
            if f:
                child = C.code_of(task, ctx.llm.ask(EVOLVE_REPAIR.format(code=child, diag=error_text(*f)),
                                                    temperature=EVOLVE["temperature"], role="refine"))
                if not child or not _parses(child) or failure(task, child):
                    continue
            t = runtime(task, child)
            if t is None:
                continue
            speedup = base / t
            review = _json(ctx.llm.ask(EVOLVE_REVIEW.format(code=child), temperature=0.0, role="review")) or {}
            try:
                llm_score = min(1.0, max(0.0, float(review.get("score", 0.5))))
            except (TypeError, ValueError):
                llm_score = 0.5
            note = reply.split("<<<<<<<")[0].strip()[:300] or "edit"
            archive.append(_Prog(child, self._fitness(child, speedup, llm_score), speedup, note, island))
            ctx.keep(max(archive, key=lambda p: p.score).code)
        return max(archive, key=lambda p: p.score).code

    @staticmethod
    def _fitness(code: str, speedup: float, llm: float) -> float:
        metric = (speedup / (1 + speedup) + pylint_score(code) / 10) / 2
        return (1 - EVOLVE["llm_feedback_weight"]) * metric + EVOLVE["llm_feedback_weight"] * llm

    @staticmethod
    def _parent(archive: list[_Prog], island: int, rng: random.Random) -> _Prog:
        r = rng.random()
        if r < EVOLVE["elite"]:
            elite = sorted(archive, key=lambda p: -p.score)[:max(1, len(archive) // 10)]
            return rng.choice(elite)
        local = [p for p in archive if p.island == island]
        if r < EVOLVE["elite"] + EVOLVE["island"] and local:
            return rng.choice(local)
        return rng.choice(archive)

    def _target(self, task: Task, code: str) -> tuple[str | None, float, int]:
        with Workdir({"driver.py": driver(task, code, 0)}) as d:
            r = execute(CPROFILE, d, timeout=task.time_limit * 10, mem_mb=task.memory_mb)
            if not r.ok or not (d / "profile.out").exists():
                return None, 0.0, 0
            stats = pstats.Stats(str(d / "profile.out")).stats
        total = max((v[3] for v in stats.values()), default=0.0) or 1e-9
        own = {k[2]: (v[3], v[1]) for k, v in stats.items() if k[0] == "solution.py" and k[2] != "<module>"}
        hot = [(f, t, c) for f, (t, c) in own.items() if t / total >= EVOLVE["tau_time_share"] or c >= EVOLVE["tau_calls"]]
        if not hot:
            return None, total, 1
        return max(hot, key=lambda x: x[1])

    @staticmethod
    def _split(code: str, target: str | None) -> tuple[str, str]:
        if target:
            for n in ast.walk(ast.parse(code)):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == target:
                    seg = ast.get_source_segment(code, n) or ""
                    return seg, code.replace(seg, f"# <writable region: {target}>")
        return code, ""

    @staticmethod
    def _apply(code: str, reply: str) -> str | None:
        edits = _DIFF.findall(reply or "")
        if not edits:
            full = C.extract_code(reply, "python") if "```" in (reply or "") else ""
            return full or None
        for old, new in edits:
            if old not in code:
                return None
            code = code.replace(old, new, 1)
        return code


# ------------------------------------------------------------------------------------------------- H19 CodeEnhancer

CE_ATTEMPTS = 5
CE_SYSTEM = "You are a code generation assistant."
CE_GEN = ("Please write Python code for the following task. At the very top, add a triple-quoted docstring with these three "
          "sections, each starting on its own line:\n• **Input Prompt**: Restate the prompt clearly.\n• **Intention**: State "
          "the purpose of the code.\n• **Functionality**: Describe briefly how the code solves the task.\n\nWrite only valid "
          "Python code and no extra explanations. Return the complete script only.\n\nPrompt: {prompt}")
CE_DOC_RETRY = ("The code must begin with a triple-quoted docstring containing the sections **Input Prompt**, **Intention** "
                "and **Functionality**. Return the complete script with that docstring at the very top.\n\n{code}")
CE_SAST = '''The static-analysis tools below reported problems in the implementation.
<Code>
{code}
</Code>
<PylintIssues>
{pylint}
</PylintIssues>
<BanditIssues>
{bandit}
</BanditIssues>
Instructions
• Fix **every** issue listed above.
• Keep all existing docstrings and comments exactly as they are.
• Reply with *only* the corrected implementation, enclosed in a single code block:
<Code>
… full fixed code body (docstring excluded) …
</Code>
Formatting rules
• The first line of your answer must be "<Code>".
• The last line must be "</Code>".
• Do **not** output anything else—no explanations, no Markdown fences, no extra text.'''
CE_CHECK = '''<Instruction>
Follow these rules **exactly** when you answer.
1. Determine whether the implementation in <Code> mostly satisfies the requirements described in <Docstring>,
   including cases where it uses a different library or algorithm that achieves the same observable behaviour
   while improving security or performance. Compare results, not the exact API calls.
2. If the implementation meets the intention (identical behaviour) — even if it swaps to a more secure or faster library — output one word only: Correct
3. If any change is needed, output in *precisely* this structure—nothing more, nothing less:
Incorrect
<Code>
# (full corrected code body here — do NOT include the docstring)
</Code>
<Reason>
# (brief explanation of what you fixed and why)
</Reason>
Formatting constraints
• Start the corrected code block with a line that contains only <Code>
  and end it with a line that contains only </Code>.
• Start the reason block with <Reason> and end it with </Reason>.
• Do NOT add Markdown fences (```), line numbers, extra headers, or footers.
• Never duplicate, delete, or modify the original docstring; keep comments intact.
</Instruction>
<Docstring>
{doc}
</Docstring>
<Code>
{code}
</Code>'''
_DOC = re.compile(r'^\s*("""|\'\'\')(.*?)\1\s*\n?', re.S)


def _tagged(text: str) -> str | None:
    m = re.search(r"<Code>\s*\n?(.*?)\n?\s*</Code>", text or "", re.S)
    if not m:
        return None
    body = m.group(1)
    return C.extract_code(body, "python") if "```" in body else body.strip("\n")


@h2h
class CodeEnhancer(H2HTechnique):
    """Stage 1 of CodeEnhancer: generation with an intent docstring, then up to five attempts in which Pylint (syntax) and
    Bandit findings drive repair, and a clean body is checked against the docstring by the model."""
    hid, name, family, target = "H19", "CodeEnhancer stage 1", "feedback", ("SEC",)
    code_url = "https://github.com/cyb3rlab/CodeEnhancer"

    def solve(self, task: Task, ctx: Ctx) -> str:
        msgs = [{"role": "system", "content": CE_SYSTEM}, {"role": "user", "content": CE_GEN.format(prompt=C.task_text(task))}]
        code = ctx.keep(C.code_of(task, ctx.llm.ask(msgs, temperature=0.0, max_tokens=1024)))
        if not _py(task):
            return code
        for _ in range(CE_ATTEMPTS):
            if _DOC.match(code):
                break
            code = ctx.keep(C.code_of(task, ctx.llm.ask(msgs + [{"role": "assistant", "content": code}] + _user(
                CE_DOC_RETRY.format(code=code)), role="docstring")))
        m = _DOC.match(code)
        doc, body = (m.group(0), code[m.end():]) if m else ("", code)
        for _ in range(CE_ATTEMPTS):
            lint = [f"{x.get('line', 0)}: {x.get('message', '')}" for x in pylint_messages(doc + body, PYLINT_SYNTAX)]
            sec = [r.get("issue_text", "") for r in bandit_results(doc + body)[0]]
            if lint or sec:
                reply = ctx.llm.ask(_user(CE_SAST.format(code=body, pylint="\n".join(lint) or "None",
                                                         bandit="\n".join(sec) or "None")), role="sast-repair")
                body = _tagged(reply) or body
                ctx.keep(doc + body)
                continue
            reply = ctx.llm.ask(_user(CE_CHECK.format(doc=doc.strip(), code=body)), role="functional-check")
            if reply.strip().lower().startswith("correct"):
                break
            body = _tagged(reply) or body
            ctx.keep(doc + body)
        return doc + body


# ------------------------------------------------------------------------------------------------- H20 PerfAgent

PERFAGENT_SUBMISSIONS, PERFAGENT_WINDOW, PERFAGENT_HOTSPOTS = 5, 10, 10
PERFAGENT_TASK = '''Here is a {lang} program for the task below, and an example workload that runs it on the task's example inputs.
Can you help me implement the necessary changes to the program so that the runtime of the workload is optimized?
This will be a very difficult problem and may involve rewriting large parts of the program. Do not be afraid to make large changes to the program in order to implement your optimization.
I have obtained an initial summary from a profiler showing the bottlenecks. The profiler report summarizes a py-spy profile of the workload.
Please use the following report to guide your optimization.
Report
{report}
To improve performance, either optimize the bottleneck itself or avoid calling the bottleneck as frequently.
Testing
The task's example inputs are used to verify the correctness of your implementation. Additional hidden tests may be used to evaluate your implementation.
Basic guidelines
1. Your task is to make changes to the program to improve the performance of the workload.
2. Make changes while ensuring the program is functionally equivalent to the original.
3. Do not overoptimize for just the specific example inputs. Make general performance improvements for the usage scenario shown.
Submit the complete program in one ```{tag} block.

Task:
{task}

Program:
```{tag}
{code}
```'''
PERFAGENT_SUMMARY = ("Below are the hotspots of a py-spy profile (100 Hz) of a program running its example workload; frames "
                     "outside solution.py are library code that must not be edited.\n\n{hotspots}\n\nSummarize the profile "
                     "for an engineer who will optimize the program: name each important hotspot (file, function, line), its "
                     "share of samples, self and total time, how it is reached, and whether it is library code. Be brief.")
PERFAGENT_FAILED = "Your submission failed validation.\n\nFailing example:\n{test}\n\nOutput:\n{error}\n\nFix the problem and submit the complete program again."
PERFAGENT_KEEP = ("Your submission passed validation. Measured speedup over the original program: {speedup:.2f}x.\n"
                  "New profiler report:\n{report}\n\nKeep optimizing: find any remaining bottleneck and submit an improved "
                  "complete program.")
_FRAME = re.compile(r"^(.*) \((.*):(\d+)\)$")


def pyspy_hotspots(task: Task, code: str, seconds: int = PERFAGENT_WINDOW, k: int = PERFAGENT_HOTSPOTS) -> str:
    with Workdir({"driver.py": driver(task, code, seconds)}) as d:
        execute(PYSPY, d, timeout=seconds * 6 + 30, mem_mb=task.memory_mb)
        raw = (d / "profile.txt").read_text() if (d / "profile.txt").exists() else ""
    total, self_t, cum, stacks, lines = 0, defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int)), {}
    for ln in raw.splitlines():
        stack, _, n = ln.rpartition(" ")
        if not n.isdigit():
            continue
        frames = [m.groups() for f in stack.split(";") if (m := _FRAME.match(f.strip()))]
        if not any(Path(f).name == "solution.py" for _, f, _ in frames):
            continue
        n = int(n)
        total += n
        for key in {(fn, Path(f).name) for fn, f, _ in frames}:
            cum[key] += n
        fn, f, line = frames[-1]
        self_t[(fn, Path(f).name)] += n
        lines[(fn, Path(f).name)] = line
        stacks[(fn, Path(f).name)][" -> ".join(x[0] for x in frames)] += n
    if not total:
        return "The profiler collected no samples inside the program."
    out = []
    for key in sorted(cum, key=lambda x: -cum[x])[:k]:
        fn, f = key
        stack = max(stacks[key], key=stacks[key].get) if stacks[key] else fn
        out.append(f"{f}:{fn} line {lines.get(key, '?')} | samples {100 * cum[key] / total:.1f}% | self "
                   f"{self_t[key] / 100:.2f}s total {cum[key] / 100:.2f}s | external {f != 'solution.py'} | stack {stack}")
    return "\n".join(out)


@h2h
class PerfAgent(H2HTechnique):
    """Profiler-guided loop: a py-spy hotspot summary written by the model, up to five validated submissions, each failure or
    speedup fed back with a fresh profile, and the fastest validated submission kept."""
    hid, name, family, target = "H20", "PerfAgent", "feedback", ("EFF",)
    ported = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        if not _py(task) or not task.public_tests or failure(task, code):
            return code
        base = runtime(task, code)
        if base is None:
            return code
        report = self._summary(ctx, task, code)
        hist = [{"role": "system", "content": C.SYSTEM},
                {"role": "user", "content": PERFAGENT_TASK.format(lang=C.LANG_NAME["python"], report=report, tag="python",
                                                                  task=C.task_text(task), code=code)}]
        best, best_speedup = code, 1.0
        for i in range(PERFAGENT_SUBMISSIONS):
            reply = ctx.llm.ask(hist, role="agent")
            hist = hist + [{"role": "assistant", "content": reply}]
            cand = C.code_of(task, reply)
            f = failure(task, cand)
            t = None if f else runtime(task, cand)
            if f or t is None:
                f = f or (task.public_tests[0], Run(False, "", "timing failed", 1))
                hist = hist + _user(PERFAGENT_FAILED.format(test=test_text(f[0]), error=error_text(*f)))
                continue
            if base / t > best_speedup:
                best, best_speedup = ctx.keep(cand), base / t
            if i == PERFAGENT_SUBMISSIONS - 1:
                break
            hist = hist + _user(PERFAGENT_KEEP.format(speedup=base / t, report=self._summary(ctx, task, cand)))
        return best

    @staticmethod
    def _summary(ctx: Ctx, task: Task, code: str) -> str:
        return ctx.llm.ask(_user(PERFAGENT_SUMMARY.format(hotspots=pyspy_hotspots(task, code))), temperature=0.0,
                           role="profile-summary")
