"""Quality benchmarks: RACE readability and maintainability subsets, NoFunEval, PythonSecurityEval."""
from __future__ import annotations

import ast
import contextlib
import csv
import io
import json
import math
import os
import re
import subprocess
import tarfile
import tempfile
import tokenize
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .. import config
from ..measure.metrics import per_kloc, share
from ..sandbox import Workdir, execute, run_program
from ..tasks import Output, Task, Test
from .base import Benchmark, register
from .correctness import day, doc_examples, fetch, pmap

RACE_R_PROMPT = "Please generate the Python code to solve the following problem{req}.\n\nProblem:\n\n{problem}"
CAMEL = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)*$")
SNAKE = re.compile(r"^(?![0-9])_?[a-z0-9]+(?:_[a-z0-9]+)*$")


def _parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(bytes(code, "utf8"))
    except (SyntaxError, ValueError):
        return None


def _targets(t: ast.AST) -> set[str]:
    if isinstance(t, ast.Name):
        return {t.id}
    return {e.id for e in t.elts if isinstance(e, ast.Name)} if isinstance(t, ast.Tuple) else set()


def _var_names(tree: ast.Module) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                out |= _targets(t)
        elif isinstance(n, ast.For):
            out |= _targets(n.target)
        elif isinstance(n, ast.FunctionDef):
            out |= {a.arg for a in n.args.args}
    return out


def _functions(tree: ast.Module | None) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)] if tree else []


def naming(code: str, pat: re.Pattern, funcs: bool = True, variables: bool = True) -> bool:
    tree = _parse(code)
    if tree is None:
        return False
    names = ([f.name for f in _functions(tree)] if funcs else []) + (list(_var_names(tree)) if variables else [])
    return bool(names) and all(pat.match(n) for n in names)


def length(code: str, max_line: int, max_fn: int) -> bool:
    if any(len(ln) > max_line for ln in code.split("\n")):
        return False
    bodies = [ast.get_source_segment(code, f) or "" for f in _functions(_parse(code))]
    return bool(bodies) and all(len(b.split("\n")) <= max_fn for b in bodies)


def _is_doc(body: list) -> bool:
    return bool(body) and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
        and isinstance(body[0].value.value, str)


def docstrings(code: str) -> bool:
    tree = _parse(code)
    if tree is None:
        return False
    if _is_doc(tree.body):
        return True
    return all(_is_doc(n.body) for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))


def line_comments(code: str, threshold: float = 0.75) -> bool:
    n = len(re.findall(r"\s#+\s", code)) + bool(re.search(r'""".*?"""', code, flags=re.S))
    code = re.sub(r'""".*?"""', "", code, flags=re.S).strip()
    try:
        lines = ast.unparse(ast.parse(code)).split("\n")
        return n / sum(1 for ln in lines if ln.strip() and "def" not in ln and "class" not in ln and "import" not in ln) >= threshold
    except Exception:
        def commented(ln: str) -> bool:
            ln = ln.strip()
            return not ln or ln.startswith("#") or ("#" in ln and bool(ln.split("#")[0].strip()))
        lines = code.split("\n")
        return sum(not commented(ln) for ln in lines) / len(lines) < 1 - threshold


RACE_R = {
    "readability_name_camel": (", and use camel case for both function names and variable names", lambda c: naming(c, CAMEL)),
    "readability_name_snake": (", and use snake case for both function names and variable names", lambda c: naming(c, SNAKE)),
    "readability_name_function_camel": (", and use camel case for function names", lambda c: naming(c, CAMEL, variables=False)),
    "readability_name_function_snake": (", and use snake case for function names", lambda c: naming(c, SNAKE, variables=False)),
    "readability_name_var_camel": (", and use camel case for variable names", lambda c: naming(c, CAMEL, funcs=False)),
    "readability_name_var_snake": (", and use snake case for variable names", lambda c: naming(c, SNAKE, funcs=False)),
    "readability_length_setting_1": (", where each line is less than 60 characters long and each function is less than 20 lines long",
                                     lambda c: length(c, 60, 20)),
    "readability_length_setting_2": (", where each line is less than 70 characters long and each function is less than 30 lines long",
                                     lambda c: length(c, 70, 30)),
    "readability_length_setting_3": (", where each line is less than 79 characters long and each function is less than 40 lines long",
                                     lambda c: length(c, 79, 40)),
    "readability_comment_by_function": (", and add the necessary docstring for each function", docstrings),
    "readability_comment_by_line": (", and add comments for each line in each function", line_comments),
}
RACE_R_GROUPS = {"RN": "readability_name", "RL": "readability_length", "RC": "readability_comment"}


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w[:1].upper() + w[1:] for w in rest)


def race_evalplus(problem: dict, solution: str, expected: dict) -> bool:
    """RACE's EvalPlus check: try every defined function as the entry point, keep the last attempt."""
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
    from evalplus.evaluate import check_correctness
    res = None
    for ep in re.findall(r"def\s+([a-zA-Z_]\w*)\s*\(", solution) or [problem["entry_point"]]:
        res = check_correctness("humaneval", 0, dict(problem, entry_point=ep), solution, expected, False, True, None,
                                DEFAULT_MIN_TIME_LIMIT, DEFAULT_GT_TIME_LIMIT_FACTOR)
        if res["base"][0] == "pass" or res["plus"][0] == "pass":
            break
    return res["base"][0] == "pass" and res["plus"][0] == "pass"


@register
class RaceR(Benchmark):
    name, dims = "race_r", ("RM",)
    metrics = {"Readability": True, "RN": True, "RL": True, "RC": True, "P": True, "IF": True}

    def load(self) -> list[Task]:
        from evalplus.data import get_human_eval_plus
        tasks = []
        for tid, p in get_human_eval_plus().items():
            ep = p["entry_point"]
            pre = f"{ep} = globals().get({ep!r}) or globals().get({_camel(ep)!r})\n"
            public, hidden = doc_examples(p["prompt"], ep, pre), [Test("assert", f"{pre}{p['test']}\ncheck({ep})\n")]
            tasks += [Task(id=f"{dim}/{tid}", benchmark=self.name, prompt=RACE_R_PROMPT.format(req=req, problem=p["prompt"].strip()),
                           entry_point=ep, public_tests=public, hidden_tests=hidden, reference=p["prompt"] + p["canonical_solution"],
                           meta={"dim": dim, "source_id": tid}) for dim, (req, _) in RACE_R.items()]
        return tasks

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        if not outs:
            return []
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        from evalplus.evaluate import get_groundtruth
        probs = get_human_eval_plus()
        with contextlib.redirect_stdout(io.StringIO()):
            gt = get_groundtruth(probs, get_human_eval_plus_hash(), [])
        src = [tasks[o.task_id].meta["source_id"] for o in outs]
        passed = pmap(race_evalplus, [probs[s] for s in src], [o.code for o in outs], [gt[s] for s in src])
        rows = []
        for o, ok in zip(outs, passed):
            dim = tasks[o.task_id].meta["dim"]
            rows.append({"task_id": o.task_id, "dim": dim, "passed": ok, "if": bool(RACE_R[dim][1](o.code))})
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        rows = [r for r in rows if "dim" in r]
        by = defaultdict(list)
        for r in rows:
            by[r["dim"]].append(r)
        pif = {d: share([r["passed"] and r["if"] for r in rs]) for d, rs in by.items()}
        res = {k: float(np.mean(v)) for k, p in RACE_R_GROUPS.items() if (v := [x for d, x in pif.items() if d.startswith(p)])}
        if len(res) == 3:
            res["Readability"] = float(np.mean([res[k] for k in RACE_R_GROUPS]))
        res.update(P=share([r["passed"] for r in rows]), IF=share([r["if"] for r in rows]))
        return res


CLASSEVAL = "https://github.com/FudanSELab/ClassEval/raw/8ac9909c134b3cd760774cb7bb8d34202f47786a/data/ClassEval_data.json"
RACE_LEETCODE = ("https://media.githubusercontent.com/media/jszheng21/RACE/3b8ee591abd5febd8ae8ec17c7b9907949c5e1d5/"
                 "data/leetcode/evaluation_tests.jsonl")
MI_PROMPT = ("Please complete the class {class_name} in the following code, and ensure that the code has good maintainability. "
             "Code maintainability refers to how easy it is to support and change the code.\n\n```python\n{skeleton}\n```")
LC_MARK = "Please complete the code below to solve above prblem:"
MC = {
    "maintainability_module_count_1": ("Please complete the code below to solve above problem, and use only the given function.", 1),
    "maintainability_module_count_2": ("Please complete the code below to solve above problem, and use only the given function and "
                                       "one addition sub-function.", 2),
    "maintainability_module_count_3": ("Please complete the code below to solve above problem, and use only the given function and "
                                       "two addition sub-functions.", 3),
}
LC_IMPORTS = "\n".join([
    "import copy", "import string", "import math", "import collections", "import bisect", "import heapq", "import functools",
    "import random", "import itertools", "import operator", "import re", "import numpy as np", "import sys",
    "from itertools import accumulate, chain, combinations, count, cycle, permutations, product, pairwise, groupby, islice, "
    "repeat, zip_longest",
    "from copy import deepcopy", "from string import ascii_lowercase",
    "from math import floor, log, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt, prod",
    "from bisect import bisect_left, bisect_right, insort", "from heapq import heappush, heappop, heapify, merge, nsmallest",
    "from random import randrange, shuffle", "from operator import itemgetter, sub, iand, xor", "from re import search as re_search",
    "from os.path import commonprefix", "from typing import *", "from functools import *", "from collections import *",
]) + "\n\n\n"
CE_DOCTESTS = '''
import ast as _ast, doctest as _doctest
_dr = _doctest.DocTestRunner(optionflags=_doctest.ELLIPSIS | _doctest.NORMALIZE_WHITESPACE)
for _n in _ast.walk(_ast.parse({skeleton!r})):
    _d = _ast.get_docstring(_n) if isinstance(_n, (_ast.FunctionDef, _ast.ClassDef)) else None
    if _d and ">>>" in _d:
        _dr.run(_doctest.DocTestParser().get_doctest(_d, dict(globals()), _n.name, None, 0), out=lambda _s: None)
assert _dr.failures == 0, _dr.failures
'''
CE_RUNNER = '''
import io as _io, json as _json, signal as _signal, unittest as _unittest
{head}


def _ce_alarm(*_):
    raise TimeoutError


_signal.signal(_signal.SIGALRM, _ce_alarm)
_ce = {{}}
for _c in {classes!r}:
    _signal.alarm(5)
    try:
        _r = _unittest.TextTestRunner(stream=_io.StringIO()).run(_unittest.TestLoader().loadTestsFromTestCase(_ns[_c]))
        _ce[_c] = [len(_r.errors), len(_r.failures), _r.testsRun]
    except BaseException:
        _ce[_c] = [0, 0, 0]
    finally:
        _signal.alarm(0)
print("__CLASSEVAL__" + _json.dumps(_ce))
{tail}
'''


def add_static(code: str) -> str:
    """ClassEval's post-processing: re-attach @staticmethod to class-level methods without self/cls."""
    out = []
    for ln in (x for x in code.split("\n") if "@staticmethod" not in x):
        if ln.strip().startswith("def ") and "self" not in ln and "cls" not in ln and len(ln) - len(ln.lstrip()) == 4:
            out.append("    @staticmethod")
        out.append(ln)
    return "\n".join(out)


def maintainability_index(code: str) -> float:
    from radon.metrics import mi_visit
    try:
        return round(mi_visit(code, True), 2)
    except Exception:
        return 0.0


def _class_success(res: dict, classes: list[str]) -> bool:
    return len(res) == len(classes) and all(n > 0 and e + f == 0 for e, f, n in res.values())


@register
class RaceM(Benchmark):
    name, dims = "race_m", ("RM",)
    metrics = {"Maintainability": True, "MI": True, "MC": True, "MI_p": True, "MC_p": True, "P": True}

    def load(self) -> list[Task]:
        return self.classeval() + self.leetcode()

    def classeval(self) -> list[Task]:
        data = json.loads(fetch(CLASSEVAL, config.DATA / "race" / "ClassEval_data.json").read_text())
        return [Task(id=f"maintainability_mi/{t['task_id']}", benchmark=self.name, prompt=MI_PROMPT.format(**t), entry_point=t["class_name"],
                     public_tests=[Test("assert", CE_DOCTESTS.format(skeleton=t["skeleton"]))],
                     hidden_tests=[Test("assert", t["test"] + CE_RUNNER.format(
                         head="_ns = globals()", classes=t["test_classes"],
                         tail="assert all(_n > 0 and _e + _f == 0 for _e, _f, _n in _ce.values()), _ce"))],
                     reference=t["solution_code"],
                     meta={"dim": "maintainability_mi", "imports": t["import_statement"], "test": t["test"], "test_classes": t["test_classes"]})
                for t in data]

    def leetcode(self) -> list[Task]:
        rows = [json.loads(ln) for ln in fetch(RACE_LEETCODE, config.DATA / "race" / "leetcode_evaluation_tests.jsonl")
                .read_text().splitlines() if ln.strip()]
        tasks = []
        for p in rows:
            blocks = re.split(r"\n\s*\n", p["test"].strip())
            k = len(re.findall(r"Example \d+:", p["prompt"]))
            public = [Test("assert", "\n\n".join(blocks[: 1 + k]) + "\n")] if k else []
            fn = re.search(r"my_solution\.(\w+)\(", p["test"])
            tasks += [Task(id=f"{dim}/{p['task_id']}", benchmark=self.name, prompt=p["prompt"].strip().replace(LC_MARK, inst),
                           entry_point=f"Solution().{fn.group(1)}" if fn else None, public_tests=public,
                           hidden_tests=[Test("assert", p["test"])], date=day(int(p["start_time"])), time_limit=60,
                           meta={"dim": dim, "modules": n, "prelude": LC_IMPORTS}) for dim, (inst, n) in MC.items()]
        return tasks

    def score(self, task: Task, out: Output) -> dict:
        dim = task.meta["dim"]
        if dim == "maintainability_mi":
            full = "\n".join(task.meta["imports"]) + "\n" + add_static(out.code)
            classes = task.meta["test_classes"]
            runner = CE_RUNNER.format(head="import sol as _sol\n_ns = vars(_sol)", classes=classes, tail="")
            with Workdir({"sol.py": full + "\n" + task.meta["test"], "main.py": runner}) as d:
                r = execute(["python3", "main.py"], d, "", 5 * len(classes) + 10, task.memory_mb)
            m = re.search(r"__CLASSEVAL__(\{.*\})", r.stdout)
            return {"task_id": task.id, "dim": dim, "passed": _class_success(json.loads(m.group(1)) if m else {}, classes),
                    "mi": maintainability_index(full)}
        r = run_program(LC_IMPORTS + out.code + "\n\n" + task.hidden_tests[0].code + "\n", "python", "", task.time_limit, task.memory_mb)
        return {"task_id": task.id, "dim": dim, "passed": r.ok, "if": len(_functions(_parse(out.code))) == task.meta["modules"]}

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        return pmap(self.score, [tasks[o.task_id] for o in outs], outs)

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        res = {}
        ce = [r for r in rows if r.get("dim") == "maintainability_mi"]
        if ce:
            ok = [r["mi"] for r in ce if r["passed"]]
            res.update(MI=float(np.mean(ok)) if ok else float("nan"), MI_p=share([r["passed"] for r in ce]))
        by = defaultdict(list)
        for r in rows:
            if r.get("dim") in MC:
                by[r["dim"]].append(r)
        if by:
            res.update(MC=float(np.mean([share([r["passed"] and r["if"] for r in rs]) for rs in by.values()])),
                       MC_p=float(np.mean([share([r["passed"] for r in rs]) for rs in by.values()])))
        if "MI" in res and "MC" in res:
            res["Maintainability"] = (res["MI"] + res["MC"]) / 2
        res["P"] = share([r["passed"] for r in rows if "passed" in r])
        return res


NFE = "https://raw.githubusercontent.com/microsoft/NoFunEval/c82a903259767e20be36c1a95d96e5b064587b66/datasets/{}.jsonl"
NFE_SYNTAX = re.compile(r"\[WARN\] \[\d*\] Failed to analyse imports of ([a-zA-Z0-9\\/.:_\-\(\)\']*) : Syntax Error \(line \d*\)")


def _strip_py_comments(src: str) -> str:
    out, prev, last_line, last_col = "", tokenize.INDENT, -1, 0
    for typ, s, (sl, sc), (el, ec), _ in tokenize.generate_tokens(io.StringIO(src).readline):
        if sl > last_line:
            last_col = 0
        if sc > last_col:
            out += " " * (sc - last_col)
        if typ == tokenize.STRING:
            if prev != tokenize.INDENT and prev != tokenize.NEWLINE and sc > 0:
                out += s
        elif typ != tokenize.COMMENT:
            out += s
        prev, last_col, last_line = typ, ec, el
    return out


def nfe_clean(code: str) -> str:
    """NoFunEval remove_comments + remove_blank_lines for Python."""
    try:
        code = _strip_py_comments(code)
    except Exception:
        code = re.sub(r"\'{3}[\s\S]*?\'{3}|\"{3}[\s\S]*?\"{3}|\#[^\n]*", "", code)
    return "\n".join(ln for ln in code.split("\n") if ln.strip())


def _git_diff(a: str, b: str) -> str:
    with tempfile.TemporaryDirectory(prefix="nfe_") as t:
        pa, pb = Path(t, "source"), Path(t, "edited")
        pa.write_text(a)
        pb.write_text(b)
        p = subprocess.run(["git", "diff", "-U0", "--no-index", "--ignore-all-space", "--ignore-blank-lines", str(pa), str(pb)],
                           capture_output=True, text=True)
    return "".join(ln for ln in p.stdout.splitlines(keepends=True)[4:] if "No newline at end of file" not in ln)


def sentence_bleu(ref: list[str], hyp: list[str]) -> float:
    """NLTK sentence_bleu, 4-gram uniform weights, SmoothingFunction().method1 (epsilon 0.1)."""
    num, den = [], []
    for n in range(1, 5):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        num.append(sum(min(c, r[g]) for g, c in h.items()))
        den.append(max(1, sum(h.values())))
    if num[0] == 0:
        return 0.0
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return bp * math.exp(sum(0.25 * math.log((x or 0.1) / y) for x, y in zip(num, den)))


def diff_bleu(source: str, target: str, generated: str) -> float:
    gen = nfe_clean(generated)
    if not gen.strip():
        return 0.0
    tok = lambda s: re.findall(r"\w+|[^\w\s]+", s)
    src = nfe_clean(source)
    return sentence_bleu(tok(_git_diff(src, nfe_clean(target))), tok(_git_diff(src, gen)))


def codeql_flags(files: dict[str, str], query: str, home: Path) -> set[str]:
    """Files flagged by the CodeQL query or failing CodeQL's Python extraction, as in NoFunEval's evaluation."""
    cli = str(home / "codeql" / "codeql")
    with tempfile.TemporaryDirectory(prefix="nfe_codeql_") as t:
        src, db, out = Path(t, "src"), Path(t, "db"), Path(t, "out.csv")
        src.mkdir()
        for name, code in files.items():
            (src / name).write_text(code)
        subprocess.run([cli, "database", "create", "--quiet", "--language=python", f"--source-root={src}", str(db)], capture_output=True)
        p = subprocess.run([cli, "database", "analyze", "--rerun", str(db), str(home / query.removeprefix("codeql-home/")),
                            "--format=csv", f"--output={out}", "--threads=0"], capture_output=True, text=True)
        if not out.exists():
            raise RuntimeError(f"CodeQL analysis failed for {query}: {p.stderr[-2000:]}")
        flagged = {row[-5].split("/")[1] for row in csv.reader(out.open())}
        for log in (db / "log").glob("database-create*"):
            flagged |= {Path(m).name for m in NFE_SYNTAX.findall(log.read_text(errors="ignore"))}
    return flagged


@register
class NoFunEval(Benchmark):
    name, dims, mode = "nofuneval", ("RM",), "edit"
    metrics = {"DiffBLEU": True, "CodeQL": True, "CodeQL-DiffBLEU": True}

    def __init__(self, limit: int | None = None, after: str | None = None, before: str | None = None,
                 subsets: tuple[str, ...] = ("maintainability",)):
        super().__init__(limit, after, before)
        self.subsets = tuple(subsets)

    def load(self) -> list[Task]:
        tasks = []
        for s in self.subsets:
            path = fetch(NFE.format(s), config.DATA / "nofuneval" / f"{s}.jsonl")
            for i, r in enumerate(json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()):
                if r["pl"].lower() != "python":
                    continue
                m = re.search(r"### Instruction:\n(.*?)\n*### Response:", r["base_prompt"], re.S)
                tasks.append(Task(id=f"{s}/{i}", benchmark=self.name, prompt=m.group(1).strip() if m else r["base_prompt"], mode="edit",
                                  source=r["source_code"], reference=r["target_code"],
                                  meta={"subset": s, "query": r.get("query"), "codeql_check": r.get("codeql_check"),
                                        "file_path": r.get("file_path"), "target_dim": "SEC" if s == "security" else "RM"}))
        return tasks

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        rows = [{"task_id": o.task_id, "diffbleu": diff_bleu(tasks[o.task_id].source, tasks[o.task_id].reference, o.code)} for o in outs]
        home = os.environ.get("CODEQL_HOME")
        if not home:
            return rows
        groups = defaultdict(dict)
        for i, o in enumerate(outs):
            t = tasks[o.task_id]
            if t.meta.get("codeql_check"):
                groups[t.meta["codeql_check"]][f"g{i}.py"] = o.code if nfe_clean(o.code).strip() else t.source
        for q, files in groups.items():
            flagged = codeql_flags(files, q, Path(home))
            for f in files:
                r = rows[int(f[1:-3])]
                r.update(codeql=f not in flagged, codeql_diffbleu=r["diffbleu"] * (f not in flagged))
        return rows

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        res = {"DiffBLEU": 100 * float(np.mean([r["diffbleu"] for r in rows]))} if rows else {}
        cq = [r for r in rows if "codeql" in r]
        if cq:
            res.update({"CodeQL": share([r["codeql"] for r in cq]), "CodeQL-DiffBLEU": 100 * float(np.mean([r["codeql_diffbleu"] for r in cq]))})
        return res


PSE = "https://codeload.github.com/Kamel773/LLM-code-refine/tar.gz/0278aa04404908eff8129a2ed0be58c6c736bf36"


@register
class PythonSecurityEval(Benchmark):
    """Scored only through the measurement stack's SonarQube rows (see sonar_metrics)."""
    name, dims = "pythonsecurityeval", ("SEC", "RM")
    metrics = {"C%": False, "Ins": False, "B/k": False}

    def load(self) -> list[Task]:
        path = config.DATA / "pythonsecurityeval" / "prompts.json"
        if not path.exists():
            prompts = {}
            with tarfile.open(fetch(PSE, config.DATA / "pythonsecurityeval" / "LLM-code-refine.tar.gz")) as tf:
                for m in tf.getmembers():
                    k = re.search(r"/data/PythonSecurityEval/(\d+)/text\.txt$", m.name)
                    if k and m.isfile():
                        prompts[k.group(1)] = tf.extractfile(m).read().decode().strip()
            if len(prompts) != 470:
                raise RuntimeError(f"PythonSecurityEval: expected 470 prompts, found {len(prompts)}")
            path.write_text(json.dumps(prompts, indent=1))
        prompts = json.loads(path.read_text())
        return [Task(id=k, benchmark=self.name, prompt=v) for k, v in sorted(prompts.items(), key=lambda kv: int(kv[0]))]

    def score(self, task: Task, out: Output) -> dict:
        return {}

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {}


def sonar_metrics(quality_rows: list[dict]) -> dict[str, float | None]:
    """PythonSecurityEval paper metrics from the stack's SonarQube rows: C% (solutions with a maintainability issue),
    Ins (solutions with a security issue), B/k (security issues per kLOC)."""
    rs = [r for r in quality_rows if "sonar_maint" in r]
    return {"C%": share([r["sonar_maint"] > 0 for r in rs]), "Ins": share([r["sonar_sec"] > 0 for r in rs]),
            "B/k": per_kloc(sum(r["sonar_sec"] for r in rs), sum(r["ncloc"] for r in rs))}
