"""Efficiency benchmarks: EvalPerf, EffiBench-X, Mercury, EffiBench."""
from __future__ import annotations

import ast
import doctest
import json
import pickle
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from statistics import mean

from ..config import DATA
from ..measure.metrics import dps, effibench_net, effibenchx_et, mercury_beyond, share
from ..sandbox import Workdir, execute, run_program
from ..tasks import Output, Task, Test
from .base import Benchmark, register

NAN = float("nan")


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else NAN


def prompt_examples(prompt: str) -> list[Test]:
    """Examples visible in a HumanEval/MBPP-style prompt: doctest lines and bare asserts."""
    tests = []
    for ex in doctest.DocTestParser().get_examples(prompt):
        src, want = ex.source.strip(), ex.want.strip()
        try:
            ast.parse(src, mode="eval"), ast.parse(want, mode="eval")
        except SyntaxError:
            continue
        tests.append(Test("assert", f"assert ({src}) == ({want})"))
    tests += [Test("assert", l.strip()) for l in prompt.splitlines() if l.strip().startswith("assert ")]
    return tests


def _io_test(entry: str, args: list, expected, atol: float) -> Test:
    blob = pickle.dumps((args, expected))
    return Test("assert", f"import math, pickle\n_i, _o = pickle.loads({blob!r})\n_r = {entry}(*_i)\n"
                          f"assert _r == _o or (isinstance(_o, float) and math.isclose(_r, _o, abs_tol={atol or 1e-6})), _r")


# ---------------------------------------------------------------- EvalPerf

@lru_cache
def _evalperf():
    from evalplus.data import get_evalperf_data
    from evalplus.evalperf import get_evalplus_data
    problems, expected = get_evalplus_data()
    return get_evalperf_data(), problems, expected


def _pe_input(tid: str, ptask: dict):
    from evalplus.data.mbpp import mbpp_deserialize_inputs
    return mbpp_deserialize_inputs(tid, ptask["pe_input"])[0] if tid.startswith("Mbpp/") else ptask["pe_input"][0]


def _instructions(tid: str, code: str, retries: int = 0) -> float | None:
    """Mean CPU instruction count of the entry point on the performance-exercising input (evalplus.perf.profile)."""
    from evalplus.config import PERF_EVAL_TIMEOUT_SECOND
    from evalplus.perf.profile import are_profiles_broken, profile
    ptask, timeout = _evalperf()[0][tid], PERF_EVAL_TIMEOUT_SECOND
    for _ in range(retries + 1):
        ps = profile(code, ptask["entry_point"], [_pe_input(tid, ptask)], timeout_second_per_test=timeout)
        if not are_profiles_broken(ps):
            return float(mean(ps))
        timeout += 10
    return None


@register
class EvalPerf(Benchmark):
    """EvalPerf (Liu et al. 2024): DPS and DPS_norm against clustered reference solutions on CPU instruction counts,
    correctness by the evalplus base+plus oracle. With one sample per task, a task is profiled when its sample is
    correct (the COLM paper's activation rule; evalplus' min_correct=10 presumes 100 samples)."""
    name = "evalperf"
    dims = ("EFF", "COR")
    metrics = {"dps_norm": True, "dps": True, "pass@1": True}

    _refs: dict[str, list[tuple[float | None, float]]] = {}

    def load(self) -> list[Task]:
        from evalplus.config import PERF_EVAL_TIMEOUT_SECOND
        ptasks, problems, expected = _evalperf()
        out = []
        for tid, p in ptasks.items():
            prob, exp = problems[tid], expected[tid]
            inputs, outputs = list(prob["base_input"]) + list(prob["plus_input"]), exp["base"] + exp["plus"]
            out.append(Task(id=tid, benchmark=self.name, prompt=p["prompt"], entry_point=p["entry_point"],
                            public_tests=prompt_examples(p["prompt"]),
                            hidden_tests=[_io_test(p["entry_point"], i, o, prob["atol"]) for i, o in zip(inputs, outputs)],
                            reference=p["reference"][-1], time_limit=PERF_EVAL_TIMEOUT_SECOND,
                            meta={"n_clusters": len(p["reference"]), "target_dim": "EFF"}))
        return out

    def _ref_counts(self, tid: str) -> list[tuple[float | None, float]]:
        if tid not in self._refs:
            p = _evalperf()[0][tid]
            self._refs[tid] = [(_instructions(tid, r, retries=1), s) for r, s in zip(p["reference"], p["scores"])]
        return self._refs[tid]

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        from evalplus.eval import PASS
        from evalplus.evalperf import correctness_check
        from evalplus.perf.profile import default_parallelism, simple_test_profiler
        from evalplus.sanitize import sanitize
        simple_test_profiler()
        _, problems, expected = _evalperf()
        codes = [sanitize(o.code, tasks[o.task_id].entry_point) if o.code.strip() else "" for o in outs]

        def check(i: int) -> bool:
            tid = outs[i].task_id
            return bool(codes[i]) and correctness_check(codes[i], tid.split("/")[0].lower(), problems[tid], expected[tid])[0][0] == PASS

        with ThreadPoolExecutor(max(1, default_parallelism(divisor=4))) as ex:
            ok = list(ex.map(check, range(len(outs))))
            list(ex.map(self._ref_counts, sorted({o.task_id for o, k in zip(outs, ok) if k})))
            counts = list(ex.map(lambda i: _instructions(outs[i].task_id, codes[i]) if ok[i] else None, range(len(outs))))
        rows = []
        for o, k, c in zip(outs, ok, counts):
            if not k:
                rows.append({"task": o.task_id, "passed": False})
                continue
            d, dn = dps(c, self._ref_counts(o.task_id))
            rows.append({"task": o.task_id, "passed": True, "instructions": c, "dps": d, "dps_norm": dn})
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        ok = [r for r in rows if r["passed"]]
        return {"dps_norm": _mean(r["dps_norm"] for r in ok), "dps": _mean(r["dps"] for r in ok),
                "pass@1": share([r["passed"] for r in rows]), "n_profiled": float(len(ok))}


# ---------------------------------------------------------------- EffiBench-X

EFFIBENCH_X_DIR = DATA / "EffiBench-X"             # git clone https://github.com/EffiBench/EffiBench-X; sandbox via start_sandbox.py, URL in $BACKEND_BASE_URL
EFFIBENCH_X_HF = "EffiBench/effibench-x"
EFFIBENCH_X_LANGS = ("python3", "cpp", "java", "javascript", "ruby", "golang")
EFFIBENCH_X_CANON = DATA / "effibench_x" / "canonical"
LANG_OF = {"python3": "python", "cpp": "cpp", "java": "java", "javascript": "javascript", "ruby": "ruby", "golang": "go"}


@lru_cache
def _effibench_x():
    from datasets import load_dataset
    if str(EFFIBENCH_X_DIR) not in sys.path:
        sys.path.insert(0, str(EFFIBENCH_X_DIR))
    return load_dataset(EFFIBENCH_X_HF, split="test")


def _ebx_key(row: dict) -> str:
    return f"{row['source']}_{row['id']}_{row['title_slug']}"


def _ebx_runtime_sum(row: dict, lang: str, code: str | None) -> float | None:
    """Official run (effibench.run_tests over the 100 stored tests) and runtime_sum in ms, None unless every test passes."""
    from effibench.run_tests import postprocess_text, run_tests
    if not code or not code.strip():
        return None
    res = run_tests(lang=lang, solution=code, test_cases=json.loads(row["generated_tests"]), evaluator=row["evaluator"],
                    test_runner=(row.get("test_runners") or {}).get(lang), early_stop=False, raise_on_error=False,
                    as_batch=True, polling_interval=5)
    res =[r for r in res if r.get("runtime") is None or r["runtime"] < 10_000_000_000]

    def ok(r: dict) -> bool:
        return bool(r.get("passed")) or (r["status"] == "done" and r["text"] == "" and r["exit_code"] == 0) or \
            postprocess_text(r["text"] or "") == postprocess_text(r["input"] + r["output"])

    if not res or not all(ok(r) and r.get("runtime") is not None for r in res):
        return None
    return sum(r["runtime"] / 1_000_000 for r in res)


def _ebx_canonical(idx: int, lang: str) -> float | None:
    """Human-expert time baseline: fastest passing of canonical_runtime, canonical_memory and canonical_normal."""
    row = _effibench_x()[idx]
    f = EFFIBENCH_X_CANON / f"{_ebx_key(row)}_{lang}.json"
    if f.exists():
        return json.loads(f.read_text())["runtime_sum"]
    sol = (row.get("solutions") or {}).get(lang) or {}
    cands = [next((it[2] for it in json.loads(sol.get(d) or "[]") if it[2] is not None), None)
             for d in ("runtimeDistribution", "memoryDistribution")] + [sol.get("code")]
    sums = [s for s in (_ebx_runtime_sum(row, lang, c) for c in dict.fromkeys(c for c in cands if c)) if s is not None]
    best = min(sums) if sums else None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"runtime_sum": best}))
    return best


@register
class EffiBenchX(Benchmark):
    """EffiBench-X (Qing et al. 2025): ET = mean over problems of clip(T_human / T_llm, 0, 1) (0 if any test fails),
    per language, then averaged over the six languages. Problems count only when all six languages have a passing
    human solution, as in the official report."""
    name = "effibench_x"
    dims = ("EFF", "COR")
    metrics = {"et": True, "pass@1": True}

    def __init__(self, langs: tuple[str, ...] = EFFIBENCH_X_LANGS, workers: int = 8, **kw):
        super().__init__(**kw)
        self.langs, self.workers = tuple(langs), workers

    def load(self) -> list[Task]:
        from effibench.prompts import prompt_solve_problem_functional, prompt_solve_problem_io
        ds = _effibench_x().select_columns(["id", "source", "title_slug", "description_md", "type", "starter_code",
                                            "release_date", "difficulty", "time_limit_nanos", "memory_limit_bytes"])
        out = []
        for idx, row in enumerate(ds):
            for lang in self.langs:
                if row["type"] == "functional":
                    starter = (row["starter_code"] or {}).get(lang)
                    if not starter:
                        continue
                    prompt = prompt_solve_problem_functional(problem=row["description_md"], target_lang=lang, starter_code=starter)
                else:
                    prompt = prompt_solve_problem_io(problem=row["description_md"], target_lang=lang)
                out.append(Task(id=f"{_ebx_key(row)}/{lang}", benchmark=self.name, prompt=prompt, language=LANG_OF[lang],
                                mode="function" if row["type"] == "functional" else "stdio", time_limit=10.0, memory_mb=1024,
                                date=(row["release_date"] or "")[:10] or None,
                                meta={"index": idx, "problem": _ebx_key(row), "lang": lang, "type": row["type"],
                                      "source": row["source"], "difficulty": row["difficulty"], "target_dim": "EFF"}))
        return out

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        ds = _effibench_x()
        pairs = [(i, l) for i in sorted({tasks[o.task_id].meta["index"] for o in outs}) for l in EFFIBENCH_X_LANGS]
        with ThreadPoolExecutor(self.workers) as ex:
            canon = dict(zip(pairs, ex.map(lambda p: _ebx_canonical(*p), pairs)))
            sums = list(ex.map(lambda o: _ebx_runtime_sum(ds[tasks[o.task_id].meta["index"]], tasks[o.task_id].meta["lang"],
                                                          o.code), outs))
        rows = []
        for o, s in zip(outs, sums):
            m = tasks[o.task_id].meta
            human = canon[m["index"], m["lang"]]
            eligible = all(canon[m["index"], l] is not None for l in EFFIBENCH_X_LANGS)
            rows.append({"task": o.task_id, "problem": m["problem"], "lang": m["lang"], "passed": s is not None,
                         "runtime_sum": s, "human_runtime_sum": human, "eligible": eligible,
                         "et": effibenchx_et(s, human) if eligible else None})
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        el = [r for r in rows if r["eligible"]]
        langs = sorted({r["lang"] for r in el})
        et = {l: _mean(r["et"] for r in el if r["lang"] == l) for l in langs}
        ps = {l: share([r["passed"] for r in el if r["lang"] == l]) for l in langs}
        out = {"et": _mean(et.values()), "pass@1": _mean(ps.values()), "n_problems": float(len({r["problem"] for r in el}))}
        out.update({f"et/{l}": v for l, v in et.items()})
        return out


# ---------------------------------------------------------------- Mercury

MERCURY_HF, MERCURY_SPLIT, MERCURY_TIMEOUT, MERCURY_RUNS = "Elfsong/Mercury", "eval", 10.0, 5

MERCURY_PRELUDE = """import re
import itertools
import collections
import heapq
import bisect
import string
import sys
import lctk
import functools
import math
import copy
import sortedcontainers
from math import floor, ceil, factorial, sqrt, inf
from sys import maxsize, stdin
from bisect import bisect_left, bisect_right
from itertools import permutations, zip_longest
from heapq import heappush, heappop, heapify
from collections import deque, defaultdict, OrderedDict
from typing import List, Optional, Tuple
from functools import lru_cache, cache
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def print(*args):pass
"""

# The bigcode-evaluation-harness Mercury sandbox (beyond_eval.Sandbox.unsafe_execute): one namespace, timed test loop.
MERCURY_RUNNER = r'''
import contextlib, io, json, signal, sys, time
s = json.loads(sys.stdin.read())
def _alarm(*a):
    raise TimeoutError
signal.signal(signal.SIGALRM, _alarm)
signal.setitimer(signal.ITIMER_REAL, s["timeout"])
status, runtime, ns = "passed", 0.0, {}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(s["prelude"], ns)
        try:
            exec(s["solution"], ns)
            exec("solution=Solution()", ns)
            exec(s["convert_offline"], ns)
            exec(s["evaluate_offline"], ns)
        except Exception:
            status = "failed@load"
        total = passed = 0
        try:
            t0 = time.time()
            for tc in s["test_cases"]:
                ns["inputs"], ns["expected"] = tc["input"], tc["expected"]
                exec("inputs, expected = convert_offline((inputs, expected))", ns)
                exec(f"outputs = solution.{s['entry_point']}(*inputs)", ns)
                exec("passed = evaluate_offline(inputs, outputs, expected)", ns)
                total += 1
                passed += 1 if ns["passed"] else 0
            runtime = time.time() - t0
        except Exception:
            status = "failed@eval" if status == "passed" else status
        if status == "passed" and total != passed:
            status = "failed@cases"
except TimeoutError:
    status = "failed@timeout"
except BaseException:
    status = "failed@error"
signal.setitimer(signal.ITIMER_REAL, 0)
sys.__stdout__.write(json.dumps({"status": status, "runtime": runtime}))
'''


@lru_cache
def _mercury():
    from datasets import load_dataset
    return {r["slug_name"]: r for r in load_dataset(MERCURY_HF, split=MERCURY_SPLIT)}


def _mercury_run(row: dict, solution: str) -> float | None:
    """Runtime (s) of the timed test loop if every stored test passes, else None."""
    spec = {"solution": solution, "prelude": MERCURY_PRELUDE, "convert_offline": row["convert_offline"],
            "evaluate_offline": row["evaluate_offline"], "entry_point": row["entry_point"],
            "test_cases": json.loads(row["test_cases"]), "timeout": MERCURY_TIMEOUT}
    r = run_program(MERCURY_RUNNER, "python", json.dumps(spec), MERCURY_TIMEOUT + 1, 2048, pin=True)
    try:
        res = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return res["runtime"] if res["status"] == "passed" else None


def _mercury_test(row: dict, tc: dict) -> Test:
    return Test("assert", f"import json\n{MERCURY_PRELUDE}\n{row['convert_offline']}\n{row['evaluate_offline']}\n"
                          f"solution = Solution()\ninputs, expected = convert_offline((json.loads({json.dumps(tc['input'])!r}), "
                          f"json.loads({json.dumps(tc['expected'])!r})))\noutputs = solution.{row['entry_point']}(*inputs)\n"
                          f"assert evaluate_offline(inputs, outputs, expected)")


@register
class Mercury(Benchmark):
    """Mercury (Du et al. 2024): Beyond = mean over tasks of (max R - clip(r, min R, max R)) / (max R - min R), R the
    locally re-run runtimes of the task's historical solutions, r the mean of five runs (0 if incorrect)."""
    name = "mercury"
    dims = ("EFF", "COR")
    metrics = {"beyond": True, "pass@1": True}

    _ref: dict[str, list[float]] = {}

    def __init__(self, workers: int = 4, **kw):
        super().__init__(**kw)
        self.workers = workers

    def load(self) -> list[Task]:
        return [Task(id=slug, benchmark=self.name, entry_point=r["entry_point"], time_limit=MERCURY_TIMEOUT, memory_mb=2048,
                     prompt=f"### Question\n{r['pretty_content'][0]}\n\n### Code Prompt\n{r['prompt']}\n",
                     hidden_tests=[_mercury_test(r, tc) for tc in json.loads(r["test_cases"])],
                     reference=r["solutions"][0]["solution"] if r["solutions"] else None,
                     meta={"difficulty": r["difficulty"], "starter": r["prompt"], "n_solutions": len(r["solutions"]),
                           "target_dim": "EFF"})
                for slug, r in _mercury().items()]

    def _runtimes(self, slug: str) -> list[float]:
        if slug not in self._ref:
            row = _mercury()[slug]
            rts = sorted(t for t in (_mercury_run(row, s["solution"]) for s in row["solutions"]) if t is not None)
            if not rts:
                raise RuntimeError(f"mercury {slug}: no historical solution passed (check lctk/sortedcontainers in the sandbox)")
            self._ref[slug] = rts
        return self._ref[slug]

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        row_of = _mercury()

        def sample(o: Output) -> float | None:
            ts = [_mercury_run(row_of[o.task_id], o.code) for _ in range(MERCURY_RUNS)]
            return None if ts[0] is None else mean(t for t in ts if t is not None)

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self._runtimes, sorted({o.task_id for o in outs})))
            rts = list(ex.map(sample, outs))
        return [{"task": o.task_id, "difficulty": row_of[o.task_id]["difficulty"], "passed": rt is not None,
                 "runtime": rt, "beyond": mercury_beyond(rt, self._runtimes(o.task_id))} for o, rt in zip(outs, rts)]

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        out = {"beyond": _mean(r["beyond"] for r in rows), "pass@1": share([r["passed"] for r in rows])}
        for d in ("Easy", "Medium", "Hard"):
            out[f"beyond/{d}"] = _mean(r["beyond"] for r in rows if r["difficulty"] == d)
        return out


# ---------------------------------------------------------------- EffiBench

EFFIBENCH_URL = "https://raw.githubusercontent.com/huangd1999/EffiBench/main/data/dataset_with_difficulty_and_algorithm.json"
EFFIBENCH_JSON = DATA / "effibench" / "dataset_with_difficulty_and_algorithm.json"
EFFIBENCH_TIMEOUT, EFFIBENCH_INTERVAL = 5, 0.001
EFFIBENCH_SYSTEM = "You are a code developer."

EFFIBENCH_INSTRUCTION = """Please based on the task description write Solution to pass the provided test cases.
You must follow the following rules:
First, the code should be in ```python\\n[Code]\\n``` block.
Second, You should not add the provided test cases into your ```\\npython[Code]\\n``` block.
Third, You are not need to write the test cases, we will provide the test cases for you.
Finally, You should make sure that the provided test cases can pass your solution.

Here is a example:
Example:
# Task description
```python
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
Example:
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

# Test cases
```python
solution = Solution()
assert solution.twoSum([2, 7, 11, 15], 9) == [[0, 1]]
```

# Code
```python
from typing import *
import random

class Solution:
    def twoSum(self, nums, target):
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []
```"""

EFFIBENCH_PRELUDE = """
from typing import *
from bisect import *
from collections import *
from copy import *
from datetime import *
from heapq import *
from math import *
from re import *
from string import *
from random import *
from itertools import *
from functools import *
from operator import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import itertools
import functools
import operator

class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""


@lru_cache
def _effibench() -> dict[str, dict]:
    if not EFFIBENCH_JSON.exists():
        EFFIBENCH_JSON.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EFFIBENCH_URL, EFFIBENCH_JSON)
    return {str(r["problem_idx"]): r for r in json.loads(EFFIBENCH_JSON.read_text())}


def _effibench_profile(row: dict, completion: str) -> dict | None:
    """src/code_efficiency_calculator.py + scripts/run_code.sh: the whole script under `mprof run --interval 0.001`;
    ET = last minus first sample time, MU = peak MB, TMU = trapezoid MB*s. None if it fails (nonzero exit or stderr)."""
    if "class Solution" not in completion:
        return None
    code = EFFIBENCH_PRELUDE + "\n" + completion + "\nsolution=Solution()\n" + row["test_case"]
    with Workdir({"main.py": code}) as d:
        r = execute(["timeout", str(EFFIBENCH_TIMEOUT), "mprof", "run", "--interval", str(EFFIBENCH_INTERVAL), "--output",
                     "main.dat", "main.py"], d, timeout=EFFIBENCH_TIMEOUT + 5, mem_mb=4096, pin=True)
        dat = d / "main.dat"
        if r.returncode != 0 or r.stderr.strip() or not dat.exists():
            return None
        samples = [(float(p[1]), float(p[2])) for p in (l.split() for l in dat.read_text().splitlines()[1:])
                   if len(p) > 2 and "__main__." not in " ".join(p)]
    if not samples:
        return None
    ts = [t for _, t in samples]
    tmu = sum((m0 + m1) / 2 * (t1 - t0) for (m0, t0), (m1, t1) in zip(samples, samples[1:]))
    return {"et": max(max(ts) - min(ts), 0.0), "mu": max(m for m, _ in samples), "tmu": tmu}


@register
class EffiBench(Benchmark):
    """EffiBench (Huang et al. 2024): NET = mean over correctly solved problems of ET_code / ET_canonical (lower is
    better, the paper's definition); net_total is the official report script's total-over-total ratio."""
    name = "effibench"
    dims = ("EFF", "COR")
    metrics = {"net": False, "net_total": False, "nmu": False, "ntmu": False, "pass@1": True}

    _canon: dict[str, dict | None] = {}

    def __init__(self, workers: int = 4, **kw):
        super().__init__(**kw)
        self.workers = workers

    def load(self) -> list[Task]:
        return [Task(id=k, benchmark=self.name, reference=r["canonical_solution"], time_limit=EFFIBENCH_TIMEOUT,
                     prompt=f"{EFFIBENCH_INSTRUCTION}\n# Task description:\n```python\n{r['markdown_description']}\n```\n"
                            f"# Test case:\n```python\n{r['small_test_cases']}\n```",
                     public_tests=[Test("assert", r["small_test_cases"])],
                     hidden_tests=[Test("assert", "solution = Solution()\n" + r["test_case"])],
                     meta={"system": EFFIBENCH_SYSTEM, "task_name": r["task_name"], "difficulty": r.get("difficulty"),
                           "algorithms": r.get("algorithms"), "target_dim": "EFF"})
                for k, r in _effibench().items()]

    def _canonical(self, pid: str) -> dict | None:
        if pid not in self._canon:
            self._canon[pid] = _effibench_profile(_effibench()[pid], _effibench()[pid]["canonical_solution"])
        return self._canon[pid]

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        rows_of = _effibench()
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self._canonical, sorted({o.task_id for o in outs})))
            profs = list(ex.map(lambda o: _effibench_profile(rows_of[o.task_id], o.code), outs))
        rows = []
        for o, p in zip(outs, profs):
            c = self._canonical(o.task_id)
            both = p is not None and c is not None
            rows.append({"task": o.task_id, "passed": p is not None, "et": p and p["et"], "mu": p and p["mu"],
                         "tmu": p and p["tmu"], "canonical": c,
                         "net": effibench_net(p["et"], c["et"]) if both else None,
                         "nmu": effibench_net(p["mu"], c["mu"]) if both else None,
                         "ntmu": effibench_net(p["tmu"], c["tmu"]) if both else None})
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        both = [r for r in rows if r["net"] is not None]

        def tot(k: str) -> float:
            den = sum(r["canonical"][k] for r in both)
            return sum(r[k] for r in both) / den if den > 0 else NAN

        return {"net": _mean(r["net"] for r in both), "nmu": _mean(r["nmu"] for r in both),
                "ntmu": _mean(r["ntmu"] for r in both), "net_total": tot("et"), "nmu_total": tot("mu"),
                "ntmu_total": tot("tmu"), "pass@1": share([r["passed"] for r in rows]), "n_compared": float(len(both))}
