"""E7 multi-agent family (H27-H36). Every agent role is the E7 model `ctx.llm`; every role's call counts toward the ceiling."""
from __future__ import annotations

import ast
import inspect
import json
import math
import os
import random
import re
import statistics
import threading
import time
from functools import lru_cache, partial
from pathlib import Path

import httpx
import numpy as np

from .. import sandbox as S
from ..llm import BudgetExceeded, current_scope, scope
from ..tasks import Task, Test
from ..techniques import common as C
from .base import Ctx, H2HTechnique, h2h
from .retrieval import decontaminate, draft, embed, fetch, load_index, save_index, top_k

TIMING_REPEATS = 5
TAVILY_URL = "https://api.tavily.com/search"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"
_LOCK = threading.Lock()


def _lang(task: Task) -> str:
    return C.LANG_NAME.get(task.language, task.language)


def _block(task: Task, code: str) -> str:
    return f"```{task.language}\n{code}\n```"


def _json(text: str):
    for pat in (r"```(?:json)?\s*\n(.*?)```", r"(\{.*\}|\[.*\])"):
        for m in re.finditer(pat, text or "", re.S):
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _compiles(task: Task, code: str) -> tuple[bool, str]:
    if not code.strip():
        return False, "empty program"
    if task.language != "python" and task.mode != "stdio":
        return True, ""
    if task.language == "python":
        try:
            compile(code, "main.py", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
    spec = S.LANGS.get(task.language)
    if not spec or "build" not in spec:
        return True, ""
    with S.Workdir({spec["file"]: code}) as d:
        r = S.build(task.language, d)
    return r.ok, r.stderr


def _describe(task: Task, t: Test) -> str:
    if t.kind == "io":
        return f"input:\n{t.stdin}\nexpected output:\n{t.expected}"
    if t.kind == "call":
        return f"{task.entry_point}(*{t.code}) should return {t.expected!r}"
    return t.code


def _run_public(task: Task, code: str) -> tuple[int, int, str]:
    """Public examples passed, their number, and a report of the failures."""
    k, fails = 0, []
    for i, t in enumerate(task.public_tests, 1):
        r = S.run_test(task, code, t)
        if r.ok:
            k += 1
            continue
        got = "timeout" if r.timeout else (f"got:\n{r.stdout}\n" if t.kind == "io" else "") + r.stderr[-1500:]
        fails.append(f"Example {i} failed.\n{_describe(task, t)}\n{got}".strip())
    return k, len(task.public_tests), "\n\n".join(fails)


def _cost(task: Task, code: str) -> float | None:
    """Summed median wall time of the public examples; None if any fails or there are none."""
    if not task.public_tests:
        return None
    total = 0.0
    for t in task.public_tests:
        w = S.wallclock(task, code, t, TIMING_REPEATS)
        if w is None:
            return None
        total += w[0]
    return total


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _test_program(task: Task, code: str, t: Test) -> tuple[str, str]:
    if t.kind == "io":
        return code, t.stdin
    if t.kind == "assert":
        return code + "\n\n" + t.code + "\n", ""
    call = (f"import json\n_r = {task.entry_point}(*json.loads({t.code!r}))\n"
            f"assert _r == json.loads({json.dumps(t.expected)!r}), _r\n")
    return code + "\n\n" + call, ""


CASE_RUNNER = '''

import json as _json, signal as _signal

def assert_raises(candidate, args=(), kwargs=None, allowed_types=(Exception,), message_substrings=None):
    allowed = tuple(allowed_types) if isinstance(allowed_types, (list, tuple)) else (allowed_types,)
    names = [t.__name__ for t in allowed]
    try:
        candidate(*args, **(kwargs or {}))
    except allowed as e:
        if message_substrings and not any(s.lower() in str(e).lower() for s in message_substrings):
            raise AssertionError(f"raised {type(e).__name__}({e}) without any of {message_substrings}")
        return
    except Exception as e:
        raise AssertionError(f"raised {type(e).__name__}({e}), expected one of {names}")
    raise AssertionError(f"no exception raised, expected one of {names}")

def _alarm(*_):
    raise TimeoutError("test case timed out")

_signal.signal(_signal.SIGALRM, _alarm)
_ns = dict(globals(), candidate=__ENTRY__)
_out = []
for _kind, _src in __CASES__:
    _signal.alarm(__LIMIT__)
    try:
        exec(_src, _ns)
        _out.append([_kind, _src, ""])
    except BaseException as _e:
        _out.append([_kind, _src, f"{type(_e).__name__}: {_e}"[:300]])
    finally:
        _signal.alarm(0)
print("@@CASES@@" + _json.dumps(_out))
'''


def _run_cases(task: Task, code: str, cases: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Runs Python test statements one by one against the task's function; returns (kind, source, error) per case."""
    if not cases:
        return []
    limit = max(1, int(task.time_limit))
    prog = code + CASE_RUNNER.replace("__ENTRY__", task.entry_point).replace("__CASES__", repr(cases)).replace(
        "__LIMIT__", str(limit))
    r = S.run_program(prog, "python", "", limit * (len(cases) + 2), task.memory_mb)
    for ln in r.stdout.splitlines():
        if ln.startswith("@@CASES@@"):
            return [tuple(x) for x in json.loads(ln[len("@@CASES@@"):])]
    err = "timeout" if r.timeout else (r.stderr[-300:] or "the program did not run")
    return [(k, s, err) for k, s in cases]


def _run_io_cases(task: Task, code: str, cases: list[dict]) -> list[tuple[str, str, str]]:
    out = []
    for c in cases:
        stdin, want = str(c.get("stdin", "")), str(c.get("expected_stdout", ""))
        r = S.run_program(code, task.language, stdin, task.time_limit, task.memory_mb)
        ok = r.ok and S.outputs_match(r.stdout, want)
        err = "" if ok else ("timeout" if r.timeout else f"expected:\n{want}\ngot:\n{r.stdout}\n{r.stderr[-500:]}".strip())
        out.append((str(c.get("kind", "functional")), f"input:\n{stdin}", err))
    return out


def _executable(task: Task) -> str:
    """How synthesized tests run: 'call' (Python function), 'io' (stdin program) or '' (not executable here)."""
    if task.language == "python" and task.mode == "function" and task.entry_point:
        return "call"
    return "io" if task.mode == "stdio" else ""


def _statements(src: str) -> list[ast.stmt]:
    try:
        return ast.parse(src).body
    except SyntaxError:
        return []


_SEARCH_LOCK = threading.Lock()


def _cached(cache: Path, key: str, fetch) -> list[dict]:
    with _SEARCH_LOCK:
        if cache.exists():
            for ln in cache.read_text().splitlines():
                d = json.loads(ln)
                if d["key"] == key:
                    return d["results"]
    res = fetch()
    with _SEARCH_LOCK, open(cache, "a") as f:
        f.write(json.dumps({"key": key, "results": res}) + "\n")
    return res


def _web_search(cache: Path, query: str, **opts) -> list[dict]:
    """Tavily web search (key from TAVILY_API_KEY); results are cached per query so reruns see the same evidence."""
    def fetch():
        r = httpx.post(TAVILY_URL, json={"query": query, **opts}, timeout=60,
                       headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"})
        r.raise_for_status()
        return [{"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
                for x in r.json().get("results", [])]
    return _cached(cache, json.dumps(["tavily", query, opts], sort_keys=True), fetch)


def _wikipedia(cache: Path, query: str) -> list[dict]:
    def fetch():
        with httpx.Client(timeout=60) as c:
            hits = c.get(WIKIPEDIA_URL, params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1,
                                                "format": "json"}).raise_for_status().json()["query"]["search"]
            if not hits:
                return []
            pages = c.get(WIKIPEDIA_URL, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                                 "titles": hits[0]["title"], "format": "json"}).raise_for_status().json()
        page = next(iter(pages["query"]["pages"].values()))
        return [{"title": page.get("title", ""), "content": page.get("extract", "")}]
    return _cached(cache, json.dumps(["wikipedia", query]), fetch)


# --------------------------------------------------------------------------------------------------------------------
# H27 SecAwareCoder: threat model -> constraint-guided coding -> security-aware test synthesis -> closed-loop repair.

SAC_ANALYZER = """You are the Analyzer of a secure code generation pipeline.

Task:
{task}

Step 1. Write a Stepwise Solution Blueprint: the algorithm as numbered steps in plain language, without code syntax.
Step 2. Perform Semantic Attack Surface Inspection over the blueprint: identify trust-boundary violations, attacker-controllable data flows, and high-risk sinks (for example subprocess calls, deserialization, file paths, SQL, eval, cryptography).
Step 3. Derive task-specific vulnerability hypotheses. Each hypothesis links a CWE to the blueprint step where it can arise and gives a concrete mitigation. Also list functional pitfalls (edge cases and input validation the task requires, including the exceptions the task says to raise).

After the blueprint and the inspection, output one JSON object in this format:
{{"functional_issues": [{{"issue_location": "...", "issue_description": "...", "fix_needed": "..."}}],
 "security_vulnerabilities": [{{"cwe_id": "CWE-...", "vulnerability_location": "...", "vulnerability_description": "...", "enhancement_needed": "..."}}]}}"""

SAC_CODER = """Generate the solution under the following security constraints.

Solution blueprint and vulnerability hypotheses:
{analysis}

Defense-in-depth directive: implement every mitigation in the hypotheses, validate and sanitize every input at the trust boundary, and fail securely: on malicious or invalid input raise an appropriate exception instead of continuing. Keep the exact signature required by the task."""

SAC_TESTER_CALL = """You are the Tester of a secure code generation pipeline. Write a dual-objective test suite for the function `{entry}` of the task below.

Task:
{task}

Vulnerability hypotheses:
{hypotheses}

Write one Python function `check(candidate)`, where `candidate` is the function under test. Respect its signature strictly.
- Under the comment `# Functional tests`, write assert statements on representative inputs, edge cases and their expected outputs.
- Under the comment `# Security tests`, turn every hypothesis into concrete adversarial payloads and assert that the candidate rejects them, with the helper `assert_raises(candidate, args=(...), allowed_types=(ValueError,), message_substrings=[...])`, which is already defined.
One assertion per statement. Do not define or import the candidate. Return only `check` in one ```python block."""

SAC_TESTER_IO = """You are the Tester of a secure code generation pipeline. Write a dual-objective test suite for the program of the task below, which reads standard input and writes standard output.

Task:
{task}

Vulnerability hypotheses:
{hypotheses}

Write functional tests (representative inputs and edge cases with their exact expected output) and security tests (adversarial or malformed inputs derived from the hypotheses, with the output the task requires for them). Return one JSON list: [{{"kind": "functional" or "security", "stdin": "...", "expected_stdout": "..."}}]."""

SAC_REPAIRER = """You are the Repairer of a secure code generation pipeline. The code below was executed against the synthesized test suite. Functional failures come from benign inputs; security failures come from adversarial inputs derived from the vulnerability hypotheses. Map each failure to the requirement or hypothesis it violates and make targeted edits that fix it while minimizing disruption to the parts that work. Keep the exact signature required by the task.

Task:
{task}

Vulnerability hypotheses:
{hypotheses}

Code:
{code}

{feedback}

Return the complete repaired code in one fenced code block."""


@h2h
class SecAwareCoder(H2HTechnique):
    """Analyzer, Coder and Tester calls, then at most two Repairer calls at temperature 0; the Executor runs the synthesized
    and public tests. Prompts are reconstructed from the paper (the repository has not released its code)."""
    hid, name, family, target = "H27", "SecAwareCoder", "multiagent", ("SEC",)
    code_url = "https://github.com/CY-SCUT-MS/CodeSecEval"
    MAX_REPAIRS = 2

    def solve(self, task: Task, ctx: Ctx) -> str:
        text = C.task_text(task)
        analysis = ctx.llm.ask(SAC_ANALYZER.format(task=text), role="analyzer", temperature=0)
        hyp = _json(analysis)
        hypotheses = json.dumps(hyp, indent=1) if isinstance(hyp, dict) else analysis
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task, post=SAC_CODER.format(analysis=analysis)),
                                                    role="coder", temperature=0)))
        how = _executable(task)
        if not how:
            return code
        if how == "call":
            raw = ctx.llm.ask(SAC_TESTER_CALL.format(entry=task.entry_point, task=text, hypotheses=hypotheses),
                              role="tester", temperature=0)
            suite = _check_cases(C.code_of(task, raw))
        else:
            raw = _json(ctx.llm.ask(SAC_TESTER_IO.format(task=text, hypotheses=hypotheses), role="tester", temperature=0))
            suite = [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []
        for _ in range(self.MAX_REPAIRS):
            feedback = self._execute(task, code, how, suite)
            if not feedback:
                break
            fixed = C.code_of(task, ctx.llm.ask(SAC_REPAIRER.format(task=text, hypotheses=hypotheses,
                                                                    code=_block(task, code), feedback=feedback),
                                                role="repairer", temperature=0))
            code = ctx.keep(fixed or code)
        return code

    @staticmethod
    def _execute(task: Task, code: str, how: str, suite) -> str:
        res = _run_cases(task, code, suite) if how == "call" else _run_io_cases(task, code, suite)
        k, n, public = _run_public(task, code)
        func = [f"- {s}\n  {e}" for kind, s, e in res if e and kind != "security"]
        sec = [f"- {s}\n  {e}" for kind, s, e in res if e and kind == "security"]
        parts = ["Functional test failures:\n" + "\n".join(func) if func else "",
                 "Security test failures:\n" + "\n".join(sec) if sec else "",
                 f"Failures on the task's own examples ({n - k} of {n}):\n{public}" if k < n else ""]
        return "\n\n".join(p for p in parts if p)


def _check_cases(src: str) -> list[tuple[str, str]]:
    fn = next((n for n in _statements(src) if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if fn is None:
        return []
    lines, out = src.splitlines(), []
    kind, prev = "functional", fn.lineno
    for st in fn.body:
        for ln in lines[prev:st.lineno - 1]:
            s = ln.strip().lower()
            if s.startswith("#") and ("security" in s or "functional" in s):
                kind = "security" if "security" in s else "functional"
        out.append((kind, ast.unparse(st)))
        prev = st.end_lineno or st.lineno
    return out


# --------------------------------------------------------------------------------------------------------------------
# H28 FlowGen (SOEN-101): process-model emulation with role agents; prompt templates verbatim from the FlowGen repository.

FLOW_BASE = {"Role": "", "Instruction": "", "Example": "", "Context": "", "Question": ""}
ENGINEER = ("You are a professional software engineer. Your duty is to write code in Python that meets the requirements. "
            "Ensure that the code you write is efficient, readable, and follows best practices. Remember, do not need to "
            "explain the code you wrote.")
TEST_CASE_EXAMPLE = ("{\"requirement analysis\":\"analysis\",\"test_cases\":[{\"Test Title\":\"title\", \"Input Data\":"
                     "\"parameter1=[1,2,3],parameter2='123'\", \"Expected Output\":\"456\"},]}")
FLOW = {
    "write_analyze": {
        "Role": "You need to act as a professional product owner. Your task is to write a ",
        "Instruction": "According to the Context, please analyze the requirement and write your response. Response in JSON "
                       "format. Your response should be high-level, rather than providing implementation details.",
        "Question": "Follow the Instruction. The requirement is as follows:\n"},
    "analyze_meetings": {
        "Role": "I want you to act as a different roles and then give some suggestions about how to revise the document.",
        "Instruction": "1. Read the Context section. 2. Act as an architect, give your suggestions. 3. Act as a tester, give "
                       "your suggestions. 4. Response in JSON format.",
        "Example": "{\"Architect\":\"Suggestion\",\"Tester\":\"Suggestion\"}",
        "Question": "Follow the Instruction. The Context section include the analysis."},
    "revise_analyze": {
        "Role": "You need to act as a professional product owner. Your duty is to revise the document based on the "
                "suggestions.",
        "Instruction": "1. Read the suggestions. 2. Revise your draft version document. 3. Response in JSON format. 4. "
                       "REMEMBER: Your analysis should be high-level, rather than providing implementation details.",
        "Question": "Follow the Instruction. Your colleague give you some suggestions(in the Context section) about this "
                    "draft document:"},
    "write_design": {
        "Role": "You need to act as a professional architect.",
        "Instruction": "According to the Context, please write your design. Your design should be high-level and focused on "
                       "guiding the developer in writing code, rather than providing implementation details.",
        "Question": "Follow the Instruction. Write your design for this requirement:\n"},
    "design_meetings": {
        "Role": "I want you to act as a different roles and then give some suggestions about how to revise the design.",
        "Instruction": "1. Read the Context section. 2. Act as an Developer, give your suggestions. 3. Act as a tester, give "
                       "your suggestions. 4. Response in JSON format.",
        "Example": "{\"Developer\":\"Suggestion\",\"Tester\":\"Suggestion\"}",
        "Question": "Follow the Instruction. The Context section include the design."},
    "revise_design": {
        "Role": "I want you to act as a professional architect. Your duty is to write a new design based on the suggestions.",
        "Instruction": "1. Read the suggestions. 2. Write your new design base on the draft version. 3. Only response your new "
                       "design.",
        "Question": "Follow the Instruction. Your colleague give you some suggestions(in the Context section) about this "
                    "draft version design:"},
    "write_code": {
        "Role": "You are a professional engineer. Your duty is to write code in Python that meets the requirements. Ensure "
                "that the code you write is efficient, readable, and follows best practices. Remember, do not need to "
                "explain the code you wrote.",
        "Instruction": "According to the Context section, write your code.",
        "Question": "Follow the Instruction section. Complete the following code:\n"},
    "code_review_meetings": {
        "Role": "I want you to act as a different roles and then give some suggestions about how to revise the code.",
        "Instruction": "1. Read the Context section. 2. Act as an architect, give your suggestions. 3. Act as a tester, give "
                       "your suggestions. 4. Response in JSON format.",
        "Example": "{\"Architect\":\"Suggestion\",\"Tester\":\"Suggestion\"}",
        "Question": "Follow the Instruction. The Context section include the code."},
    "revise_code": {
        "Role": ENGINEER,
        "Instruction": "1. Understand the suggestions(in the Context section) about the original code. 2. Rewrite your code.",
        "Question": "Follow the Instruction section. Complete the following code:\n"},
    "write_test_cases": {
        "Role": "You are a quality assurance engineer. Your duty is to design test cases for the developers.",
        "Instruction": "According to the Context section, write test cases. Don't write long strings in your test cases. "
                       "Response in JSON format.",
        "Example": TEST_CASE_EXAMPLE,
        "Question": "Follow the Instruction section, output the test cases for this code:\n"},
    "test_cases_meetings": {
        "Role": "I want you to act as a different roles and then give some suggestions about how to revise the test cases.",
        "Instruction": "1. Read the Context section. 2. Act as an Architect, give your suggestions. 3. Act as a Developer, give "
                       "your suggestions. 4. Response in JSON format.",
        "Example": "{\"Architect\":\"Suggestion\",\"Developer\":\"Suggestion\"}",
        "Question": "Follow the Instruction. The Context section include the test cases."},
    "revise_test_cases": {
        "Role": "You are a quality assurance engineer. Your duty is to revise the test cases based on the suggestions.",
        "Instruction": "1. Your colleague give you some suggestions(in the Context section) about the draft test cases, read "
                       "these suggestions. 2. Revise your draft version test cases, but don't use long parameters. 3. "
                       "Response in JSON format.",
        "Example": TEST_CASE_EXAMPLE,
        "Question": "Follow the Instruction. Please make sure your response length are less than 512. Your test cases is for "
                    "the below question:\n"},
    "self_test": {
        "Role": ENGINEER,
        "Instruction": "1. Understand the test cases(in the Context section) about the original code. 2. Rewrite your code, "
                       "make sure your code pass all test cases.",
        "Question": "Follow the Instruction section. Complete the following code:\n:"},
    "write_test_script": {
        "Role": "You are a quality assurance engineer. Your duty is to write a python test script in unittest framework.",
        "Instruction": "1. Write a python test script. 2. Use the input and output given by the question.",
        "Example": "import unittest\n\nclass Test(unittest.TestCase):\n    def test(self):\n        "
                   "self.assertEqual(function_under_test('a'), 1)\n\nif __name__ == '__main__':\n    unittest.main()",
        "Question": "Follow the Instruction section. The function definition you need to test:"},
    "test_code": {
        "Role": "You are a quality assurance engineer. Your duty is to write a test report.",
        "Instruction": "1. Read the test script's output(in the Context section). 2. Analyze and generate your test report "
                       "base on this output. 3. If the code has passed the tests, write a conclusion 'Code Test Passed'. 4. "
                       "If the code FAIL, shows the input and expected output example in your report.",
        "Question": "Follow the Instruction section."},
    "repair_code": {
        "Role": ENGINEER,
        "Instruction": "1. Rewrite a bug fix version of the original code. 2. Response in JSON format.",
        "Example": "{\"revised_code\":\"...\"}",
        "Question": "The original code has some bugs, according to the test report and suggestions, rewrite the code. "
                    "Complete the following code:\n"},
}


def _flow(name: str, *, role: str = "", context: str = "", add: str = "", question: str | None = None) -> str:
    d = FLOW_BASE | FLOW[name]
    d["Role"] += role
    d["Context"] = context
    d["Question"] = question if question is not None else d["Question"] + add
    return json.dumps(d)


@h2h
class FlowGen(H2HTechnique):
    """FlowGen-Waterfall at temperature 0.8: requirements, design, code and test cases each written, reviewed and revised,
    a self-test rewrite, then execution of the example-derived unittest script with at most two report-and-repair rounds.
    FlowGen-Scrum needs at least 24 calls (22 before any execution), so the Waterfall model, which fits, is used."""
    hid, name, family, target = "H28", "FlowGen", "multiagent", ("RM",)
    code_url = "https://anonymous.4open.science/r/FlowGen-LLM-E842/README.md"
    TEMPERATURE = 0.8
    MAX_REPAIRS = 2
    REPEAT_GUARD = 3

    def solve(self, task: Task, ctx: Ctx) -> str:
        ask = lambda prompt, role: ctx.llm.ask(prompt, role=role, temperature=self.TEMPERATURE)
        problem = C.task_text(task)
        prd = ask(_flow("write_analyze", role="Product Requirement Document", add=problem), "requirements")
        review = ask(_flow("analyze_meetings", context=prd), "requirements-review")
        prd = ask(_flow("revise_analyze", context=f"# Suggestion:\n{review}", add=prd), "requirements-revise")
        ctx_prd = f"# Product Requirement Document:\n{prd}"
        design = ask(_flow("write_design", context=ctx_prd, add=problem), "design")
        review = ask(_flow("design_meetings", context=design), "design-review")
        design = ask(_flow("revise_design", context=f"# Suggestions:\n{review}", add=design), "design-revise")
        ctx_design = f"{ctx_prd}\n# Design:\n{design}"
        code = ctx.keep(C.code_of(task, ask(_flow("write_code", context=ctx_design, add=problem), "developer")))
        review = ask(_flow("code_review_meetings", context=_block(task, code)), "code-review")
        code = ctx.keep(C.code_of(task, ask(_flow("revise_code", context=f"# Original Code:\n{code}\n# Advices:\n{review}",
                                                  add=problem), "developer-revise")) or code)
        cases = ask(_flow("write_test_cases", context=ctx_design, question=problem), "tester")
        review = ask(_flow("test_cases_meetings", context=cases), "tester-review")
        cases = ask(_flow("revise_test_cases", context=f"# Suggestions:\n{review}\n # Draft Test cases:\n{cases}",
                          add=problem), "tester-revise")
        code = ctx.keep(C.code_of(task, ask(_flow("self_test", context=f"# Original Code:\n{code}\n# Test Cases:\n{cases}",
                                                  question=problem), "developer-selftest")) or code)
        script = ""
        if task.language == "python" and task.mode == "function":
            raw = C.code_of(task, ask(_flow("write_test_script", question=problem), "test-script"))
            script = re.sub(r"(def[\s\S]+)class Test", "class Test", raw)
        versions = [code]
        for i in range(self.MAX_REPAIRS + 1):
            passed, output = self._test(task, code, script)
            if passed or i == self.MAX_REPAIRS:
                break
            report = ask(_flow("test_code", context=f"# test script's output:\n{output}"), "test-report")
            for _ in range(self.REPEAT_GUARD):
                raw = ask(_flow("repair_code", context=f"# the original code:\n{code}\n# test report:\n{report}",
                                add=problem), "developer-repair")
                d = _json(raw)
                fixed = C.code_of(task, d["revised_code"]) if isinstance(d, dict) and isinstance(d.get("revised_code"), str) \
                    else C.code_of(task, raw)
                if fixed and fixed not in versions:
                    break
            code = ctx.keep(fixed or code)
            versions.append(code)
        return code

    @staticmethod
    def _test(task: Task, code: str, script: str) -> tuple[bool, str]:
        if script:
            r = S.run_program(code + "\n" + script, "python", "", 480, task.memory_mb)
            return r.ok, (r.stdout + "\n" + r.stderr).strip()
        k, n, report = _run_public(task, code)
        return k == n, report or f"{k} of {n} examples passed."


# --------------------------------------------------------------------------------------------------------------------
# H29 INDICT: safety and helpfulness critics in an internal dialogue, grounded by tools, preemptive and post-hoc feedback.
# Prompts and the loop follow the released code (indict_llama strategy).

IND_ROUNDS = 3
IND_CRITIC_TURNS = 1
IND_HEADER = "Based on the above analysis, here's the updated version of the code in a single code block (wrap in ```):\n"
IND_ACTOR = """
{question}
{scratchpad}"""

IND_SAFETY = """You are an advanced intelligent agent with a strong focus for system security and risk management. You will be given a previous potential solution in which you were given a task to solve. In a few sentences, diagnose whether the solution is completely safe in every execution step and will not pose any potential harmful effect to the world, including any damage to cybersecurity or physical world.

You are also required to debate and discuss with another critic who focuses on system performance and accuracy. Always advocate and focus on system security and risk management over system performance or correctness.

Limit your critic to a few sentences.

Task: {question}
Solution: {answer}

{scratchpad}"""

IND_HELPFUL = """You are an advanced intelligent agent with a strong focus for system performance and accuracy. You will be given a previous potential solution in which you were given a task to solve. In a few sentences, diagnose whether the solution is completely correct in every execution step and will satisfy all the requirements in the given task and pass any corner test cases.

You are also required to debate and discuss with another critic who focuses on system security and risk management. Always advocate and focus on system performance and accuracy over system security or risk management.

Limit your critic to a few sentences.

Task: {question}
Solution: {answer}

{scratchpad}"""

IND_SUMMARY = """You are an advanced intelligent agent with equal focuses to both system performance and security. You will be given a discussion about a system solution between two critics against system safety and correctness. In a few sentences, summarize the discussion and determine whether the solution is completely correct and secure in every execution step. The perfect solution should satisfy all the requirements in the given task while not violating any security standards or causing any harmful effect to the society or cybersecurity.
There might be some supporting facts in the discussion between the critics. Incorporate as much as possible those details into your response to make your analysis informative and convincing.

{scratchpad}"""

IND_SUMMARY_POSTHOC = """You are an advanced intelligent agent with equal focuses to both system performance and security. You will be given a discussion about a system solution between two critics against system safety and correctness. In a few sentences, summarize the discussion and determine whether the solution is completely correct and secure in every execution step. The perfect solution should satisfy all the requirements in the given task while not violating any security standards or causing any harmful effect to the society or cybersecurity.
There might be some supporting facts in the discussion between the critics, including relevant document snippets, code snippets and their execution results and test results. Incorporate as much as possible those details into your response to make your analysis informative and convincing to be used to improve the current initial solution.

{scratchpad}"""

IND_QUERY = """You are an advanced intelligent agent with direct access to Internet. You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. To improve the analysis with relevant evidence and fact, generate a relevant keyword or query to search for related information on Internet. You may also search for information that is relevant to the task or solution but is missing in the analysis. Use the following format: Search[<query or keyword>].

Task: {question}
Solution: {answer}

{scratchpad}

Query (in the form of Search[<query or keyword>]):"""

IND_QUERY_CODE = """You are an advanced intelligent agent with direct access to Internet. You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. To improve the analysis with relevant evidence and fact, a query might be provided to extract more information. To make the query more informative, extract or create a relevant short code snippet to be used together the query. If the query is empty, provide a representative code snippet that could be used to search for more information to support the analysis.

The code snippet should be indepedent (does not refer to external operating systems, databases, repositories, or custom libraries) and limited to few lines of codes only. Use `print` or `assert` statements in the code snippets if needed (to execute and perform debugging on a code interpreter).

Wrap the code snippet in ```.

Task: {question}
Solution: {answer}

{scratchpad}

Query: {query}

Short code snippet in a single code block (wrap in ```):"""

IND_PAST = ("\nThe following critic(s) provide some analysis of system safety and correctness of previous solution(s). Use them "
            "as reference only to improve and update your critic based on the above latest solution.")
IND_OBS = ("\nThe following provides some observation(s) when executing the current solution. If these observations are "
           "relevant, you might use them as reference (if needed) to improve and update your critic:")
IND_REFUSALS = ("i can't", "i cannot", "i don't know", "i do not know", "i am not sure", "i'm not sure", "sorry i", "i refuse")
IND_COMPILED = "Solution is compiled successfully without any error."
IND_EXEC_TIMEOUT = 120


@h2h
class INDICT(H2HTechnique):
    """Three outer rounds as in the released script; per round: act, preemptive debate (safety then helpfulness critic,
    each grounded by one tool query: query and snippet generation, a model-knowledge answer, web search and Wikipedia),
    summary, act, then for Python an executed post-hoc debate and a final act. The model-knowledge tool is the E7 model;
    Tavily stands in for the Yahoo scraper. A Python round needs 21 calls, so the run stops at the ceiling."""
    hid, name, family, target = "H29", "INDICT", "multiagent", ("SEC",)
    code_url = "https://github.com/SalesforceAIResearch/indict_code_gen"

    def prepare(self, ctx: Ctx) -> None:
        if not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError("H29 needs TAVILY_API_KEY for its web-search tool")

    def solve(self, task: Task, ctx: Ctx) -> str:
        q, posthoc = C.task_text(task), task.language == "python"
        actor = lambda pad: C.code_of(task, ctx.llm.ask(IND_ACTOR.format(question=q, scratchpad=pad), role="actor",
                                                         max_tokens=2048, stop=["\nCritic:"]))
        action, past = "", []
        for _ in range(IND_ROUNDS):
            if not action:
                action = ctx.keep(actor(IND_HEADER))
            critic = self._debate(task, ctx, q, action, past, None)
            mid = ctx.keep(actor(f"\nInitial Solution: {_block(task, action)}\nCritic: {critic}\nImproved Solution: "
                                 f"{IND_HEADER}") or action)
            if not posthoc:
                past, action = past + [critic], mid
                continue
            run = S.run_program(mid, "python", "", IND_EXEC_TIMEOUT, task.memory_mb)
            obs = (run.stdout + run.stderr).strip() or IND_COMPILED
            critic2 = self._debate(task, ctx, q, mid, past + [critic], obs)
            final = actor(f"\nInitial Solution: {_block(task, action)}\nCritic: {critic}\nFirst Improved Solution: "
                          f"{_block(task, mid)}\nCritic: {critic2}\nSecond Improved Solution: {IND_HEADER}")
            past, action = past + [critic, critic2], ctx.keep(final or mid)
        return action

    def _debate(self, task: Task, ctx: Ctx, q: str, answer: str, past: list[str], obs: str | None) -> str:
        ans = _block(task, answer)
        pad = IND_PAST + "".join("\nPast Critic: " + c for c in past) if past else ""
        if obs is not None:
            pad += IND_OBS + "\nCurrent Solution Observation: " + obs
        for turn in range(IND_CRITIC_TURNS):
            upd = bool(turn or past)
            for name, tmpl, label in (("Safety", IND_SAFETY, "safety"), ("Correctness", IND_HELPFUL, "correctness")):
                pad += (f"\n{name} Critic: based on the above discussion, here is my updated analysis of solution {label}: "
                        if upd else f"\n{name} Critic: here is my analysis of solution {label}: ")
                critique = ctx.llm.ask(tmpl.format(question=q, answer=ans, scratchpad=pad), role=f"{label}-critic",
                                       max_tokens=128)
                facts = self._tools(task, ctx, q, ans, critique, obs is not None)
                pad += critique + (f"\nSupporting Fact(s) for {name}: {facts}" if facts else "")
        summary = IND_SUMMARY_POSTHOC if obs is not None else IND_SUMMARY
        return ctx.llm.ask(summary.format(scratchpad=pad), role="critic-summary", max_tokens=2048)

    def _tools(self, task: Task, ctx: Ctx, q: str, ans: str, critique: str, posthoc: bool) -> str:
        pad = "Analysis:" + critique
        raw = ctx.llm.ask(IND_QUERY.format(question=q, answer=ans, scratchpad=pad), role="tool-query", max_tokens=64)
        line = next((l for l in raw.splitlines() if "Search" in l), (raw.strip().splitlines() or [""])[0])
        m = re.search(r"\[(.*?)\]", line)
        query = (m.group(1) if m else line).strip()
        snip_raw = ctx.llm.ask(IND_QUERY_CODE.format(question=q, answer=ans, scratchpad=pad, query=query),
                               role="tool-snippet", max_tokens=128)
        snippet = C.extract_code(snip_raw, "python") if "```" in snip_raw else ""
        if posthoc:
            r = S.run_program(snippet, "python", "", IND_EXEC_TIMEOUT, task.memory_mb) if snippet else None
            out = (r.stdout + r.stderr).strip() if r else ""
            combined = (f"Code context:\n```{snippet}\n```\nCode output: {out or 'the code is compiled successfully without any error.'}"
                        f"\nQuery: {query}")
        else:
            combined = (f"Code context:\n```{snippet}\n```\nQuery: {query}" if snippet
                        else f"Provide critical and useful information about the following: {query}")
        cache = ctx.dir(self.hid) / "search_cache.jsonl"
        facts = [("Model knowledge", ctx.llm.ask(combined, role="tool-knowledge", max_tokens=256))]
        facts += [(f"{query} - {w['title']}", w["content"]) for w in _web_search(cache, combined[:400], max_results=1)[:1]]
        if query:
            facts += [(w["title"], w["content"]) for w in _wikipedia(cache, query)]
        return "".join(f"\nSupporting Fact: {t} - {d.strip()}" for t, d in facts
                       if d.strip() and not any(x in d.lower() for x in IND_REFUSALS))


# --------------------------------------------------------------------------------------------------------------------
# H30 ContextEvolve: OpenEvolve-style evolutionary search with Summarizer, Navigator and Sampler agents (RL mode).
# Prompts verbatim from ContextEvolve-ACC; hyperparameters from its task configs.

CE_MAX_ITERATIONS = 100
CE_NUM_TOP = 5
CE_NUM_DIVERSE = 5
CE_NUM_INSPIRATIONS = 5
CE_NUM_REFERENCES = 6
CE_ARCHIVE = 100
CE_EXPLORATION, CE_EXPLOITATION = 0.2, 0.7
CE_NAV_INTERVAL = 3
CE_ROLLOUT = 20
CE_ROLLOUT_WEIGHTS = {"all-improved": 0.6, "mixed": 0.2, "all-degraded": 0.2}
CE_TEMP_EVOLVE, CE_TEMP_SUMMARY, CE_TEMP_NAV, CE_TEMP_SAMPLER = 0.8, 0.6, 0.8, 0.8
CE_MAX_CODE = 10000
CE_MAX_ARTIFACT = 20000
CE_SIMPLIFY_AFTER = 500
CE_NO_GRADIENT = "No gradient generated yet. Perform broad exploration."

CE_SUM_SYSTEM = ("You are an expert code analyst specializing in algorithmic optimization and code comprehension. Your task is "
                 "to summarize the core algorithmic idea from code implementations, focusing on the underlying strategy, data "
                 "structures, and optimization techniques used in brief and concise manner.")
CE_SUM_USER = ("# Task Overview:\nYour goal is to summarize the core algorithmic idea from the child programs.\n\n"
               "# Requirements:\n1. Concise but dense with technical details.\n2. Highlight the new or modified aspects (2-4 "
               "phrases).\n3. Keep core shared characteristics from the parent (if available) that are still present in the "
               "child (2-4 phrases).\n4. Each phrase should be less than 8 words.\n\n")
CE_SUM_PARENT = "# Parent Program Abstract:\n{abstract}\n\n"
CE_SUM_CHILD = "# Child Program Code:\n```{lang}\n{code}\n```\n\n# Output Format:\nOutput the summary of the child program directly: \n- phrase 1\n- phrase 2\n- ...\n\n"

CE_NAV_SYSTEM = ("You are an expert algorithmic strategist (Meta-Critic). Your task is to analyze the recent history of code "
                 "evolution (both successes and failures) and synthesize 3-5 brief high-level **Optimization Directions** for "
                 "the next batch of optimization attempts.")
CE_NAV_USER = ("# Task Overview:\nYour goal is to synthesize 3-5 brief high-level **Optimization Directions** from the history "
               "of optimization for the next batch of attempts.\n\n# Requirements:\n1. **Identify Trends:** Identify which "
               "directions seem promising vs. dead ends.\n2. **Potential Ideas:** Though some attempts may failed, the "
               "high-level ideas may still be worth exploring.\n3. **Abstraction:** Do NOT suggest specific code lines. Suggest "
               "algorithmic paradigms or architectural patterns.\n4. **Focus:** The goal is to maximize the evaluation "
               "metrics.\n\n# Population Samples (Metrics):\n{history}\nOutput the directions directly: \n")

CE_SAMPLER_SYSTEM = ("You are an expert program selector specializing in evolutionary algorithm optimization. Your task is to "
                     "select the best reference programs from a population to guide the evolution of a parent program based "
                     "on its current state and improvement needs.")
CE_SAMPLER_USER = ("# Task Overview:\nYour goal is to select the best {k} reference programs based on its current state and "
                   "improvement needs.\n\n# Parent Program Context:\n## Abstract:\n{abstract}\n\n## Optimization Directions:\n"
                   "{gradient}\n\n# Candidate Programs (Total: {n}):\n{candidates}\n\n# Selection Criteria:\n1. The candidate "
                   "should logically address or relate to the current optimization directions.\n2. Prefer candidates with "
                   "better evaluation metrics.\n3. Avoid selecting multiple programs with identical or similar abstracts. "
                   "Programs should complement each other.\n4. Provide potential, attractive, and innovative ideas.\n# Output "
                   "Format:\nYou MUST return a JSON object with the following structure:\n{{\n  \"reason\": \"Brief reasoning "
                   "statement.\",\n  \"selected_indices\": [index1, index2, index3]\n}}\n\n")

CE_EVOLVE_SYSTEM = """You are an expert programmer specializing in optimization algorithms. Your task is to improve the program below, which solves the following task, so that it stays correct and runs as fast as possible. The evaluator runs the program on the task's examples and reports correctness (fraction of examples passed), speedup (execution time of the initial program divided by this program's) and combined_score (correctness plus speedup when every example passes).

Task:
{task}"""

CE_EVOLVE_USER = """# Current Program Information
- Current performance metrics: {metrics}
- Areas identified for improvement: {improvement_areas}

{artifacts}

# Optimization Opportunities
{gradient}

# Reference Programs
{references}


# Current Program
```{language}
{code}
```

# Instructions
1. **Analyze** the current program, focusing on the improvement areas and parent's gradient
2. **Learn** from the evolution rollout and reference programs
3. **Identify** specific code sections that need modification
4. **Modify** using the SEARCH/REPLACE format below

# Required Output Format
You MUST use the exact SEARCH/REPLACE diff format:

<<<<<<< SEARCH
[Original code block - must match exactly]
=======
[Improved code block]
>>>>>>> REPLACE

# Example
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better cache locality
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

# Important Constraints
- Each SEARCH block must exactly match existing code
- Focus on targeted improvements
- You can provide multiple SEARCH/REPLACE blocks
- Explain your reasoning for each change"""

CE_ATTEMPT = "### Attempt {n}\n- Changes: {changes}\n- Performance: {performance}\n- Outcome: {outcome}\n"
CE_PROGRAM = "### Program {n} (Score: {score})\n```{language}\n{code}\n```\nKey features: {features}\n"
CE_DIFF = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)


@h2h
class ContextEvolve(H2HTechnique):
    """Evolution of an initial program for up to 100 iterations: parent sampling (exploration/exploitation), Sampler-chosen
    references (no call while the pool has at most k candidates), a diff from the generator, evaluation on the public
    examples, a Summarizer abstract per child and a Navigator every 3 children. Single island: the ceiling allows about six
    iterations, far below the population and migration scales of the configs. Runs until the ceiling."""
    hid, name, family, target = "H30", "ContextEvolve", "multiagent", ("EFF",)
    code_url = "https://anonymous.4open.science/r/ContextEvolve-ACC"

    def solve(self, task: Task, ctx: Ctx) -> str:
        rng = random.Random(f"{task.id}:{current_scope().seed}")
        code0 = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        base = _cost(task, code0)
        root = self._evaluate(task, code0, base)
        root["abstract"] = self._summarize(task, ctx, code0, None)
        pop, gradient, best, children = [root], CE_NO_GRADIENT, root, 0
        system = CE_EVOLVE_SYSTEM.format(task=C.task_text(task))
        for _ in range(CE_MAX_ITERATIONS):
            parent = self._parent(pop, rng)
            refs = self._references(task, ctx, pop, parent, gradient, rng)
            raw = ctx.llm.ask([{"role": "system", "content": system},
                               {"role": "user", "content": self._prompt(task, parent, gradient, refs)}],
                              role="generator", temperature=CE_TEMP_EVOLVE)
            child_code, applied = parent["code"], 0
            for search, replace in CE_DIFF.findall(raw):
                if search in child_code:
                    child_code, applied = child_code.replace(search, replace, 1), applied + 1
            if not applied or len(child_code) > CE_MAX_CODE:
                continue
            child = self._evaluate(task, child_code, base) | {"parent": parent}
            child["abstract"] = self._summarize(task, ctx, child_code, parent["abstract"])
            pop.append(child)
            if child["metrics"]["combined_score"] > best["metrics"]["combined_score"]:
                best = child
                ctx.keep(child_code)
            children += 1
            if children % CE_NAV_INTERVAL == 0:
                gradient = self._navigate(ctx, pop, rng)
        return best["code"]

    @staticmethod
    def _evaluate(task: Task, code: str, base: float | None) -> dict:
        ok, log = _compiles(task, code)
        k, n, report = _run_public(task, code) if ok else (0, max(1, len(task.public_tests)), log)
        correct = k / n if n else float(ok)
        speed = 0.0
        if correct == 1.0:
            c = _cost(task, code)
            speed = base / c if base and c else 1.0
        metrics = {"correctness": correct, "speedup": speed, "combined_score": correct + speed}
        return {"code": code, "metrics": metrics, "artifacts": report[:CE_MAX_ARTIFACT], "abstract": "", "parent": None}

    @staticmethod
    def _parent(pop: list[dict], rng: random.Random) -> dict:
        r = rng.random()
        if r < CE_EXPLORATION:
            return rng.choice(pop)
        if r < CE_EXPLORATION + CE_EXPLOITATION:
            return rng.choice(sorted(pop, key=lambda p: -p["metrics"]["combined_score"])[:CE_ARCHIVE])
        return rng.choice(pop)

    def _references(self, task: Task, ctx: Ctx, pop: list[dict], parent: dict, gradient: str,
                    rng: random.Random) -> list[tuple[str, dict]]:
        ranked = sorted(pop, key=lambda p: -p["metrics"]["combined_score"])
        top = ranked[:CE_NUM_TOP + CE_NUM_DIVERSE]
        prev = [p for p in ranked if all(p is not q for q in top)][:CE_NUM_TOP]
        others = [p for p in pop if p is not parent]
        insp = rng.sample(others, min(CE_NUM_INSPIRATIONS, len(others)))
        pool: list[tuple[str, dict]] = []
        for kind, group in (("top", top), ("previous", prev), ("inspiration", insp)):
            pool += [(kind, p) for p in group if p is not parent and all(p is not q for _, q in pool)]
        if len(pool) <= CE_NUM_REFERENCES:
            return pool
        cands = "".join(f"**Index {i}**:\n- Abstract: {p['abstract']}\n- Metrics:\n{json.dumps(p['metrics'], indent=2)}\n"
                        for i, (_, p) in enumerate(pool))
        msgs = [{"role": "system", "content": CE_SAMPLER_SYSTEM},
                {"role": "user", "content": CE_SAMPLER_USER.format(k=CE_NUM_REFERENCES, abstract=parent["abstract"],
                                                                   gradient=gradient, n=len(pool), candidates=cands)}]
        try:
            idx = json.loads(ctx.llm.ask(msgs, role="sampler", temperature=CE_TEMP_SAMPLER))["selected_indices"]
            return [pool[i] for i in idx if isinstance(i, int) and 0 <= i < len(pool)][:CE_NUM_REFERENCES]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    @staticmethod
    def _prompt(task: Task, parent: dict, gradient: str, refs: list[tuple[str, dict]]) -> str:
        parts = []
        for n, (kind, p) in enumerate(refs, 1):
            m = p["metrics"]
            if kind == "previous" and p["parent"] is not None:
                diffs = [m[k] - p["parent"]["metrics"][k] for k in m]
                outcome = ("Improvement in all metrics" if all(d > 0 for d in diffs) else
                           "Regression in all metrics" if all(d < 0 for d in diffs) else "Mixed results")
                parts.append(CE_ATTEMPT.format(n=n, changes=p["abstract"], performance=_fmt_metrics(m), outcome=outcome))
            else:
                best_metric = max((k for k in m if k != "combined_score"), key=lambda k: m[k])
                parts.append(CE_PROGRAM.format(n=n, score=f"{m['combined_score']:.4f}", language=task.language,
                                               code=p["code"], features=f"Performs well on {best_metric} ({m[best_metric]:.4f})"))
        areas = ("Consider simplifying - code length exceeds 500 characters" if len(parent["code"]) > CE_SIMPLIFY_AFTER
                 else "Focus on optimizing the code for better performance on the target metrics")
        artifacts = f"## Last Execution Output\n\n### failures\n```\n{parent['artifacts']}\n```" if parent["artifacts"] else ""
        return CE_EVOLVE_USER.format(metrics=_fmt_metrics(parent["metrics"]), improvement_areas=areas, artifacts=artifacts,
                                     gradient=gradient, references="\n".join(parts), language=task.language,
                                     code=parent["code"])

    @staticmethod
    def _summarize(task: Task, ctx: Ctx, code: str, parent_abstract: str | None) -> str:
        user = CE_SUM_USER + (CE_SUM_PARENT.format(abstract=parent_abstract) if parent_abstract else "") + \
            CE_SUM_CHILD.format(lang=task.language, code=code)
        return ctx.llm.ask([{"role": "system", "content": CE_SUM_SYSTEM}, {"role": "user", "content": user}],
                           role="summarizer", temperature=CE_TEMP_SUMMARY)

    @staticmethod
    def _navigate(ctx: Ctx, pop: list[dict], rng: random.Random) -> str:
        keys = sorted(pop[0]["metrics"], key=lambda k: (k == "combined_score", k))
        z = {}
        for k in keys:
            xs = [p["metrics"][k] for p in pop]
            mu, sd = statistics.fmean(xs), statistics.pstdev(xs) or 1.0
            z[k] = [(x - mu) / sd for x in xs]
        status = []
        for i in range(len(pop)):
            zs = [z[k][i] for k in keys]
            status.append("all-improved" if all(v > 0 for v in zs) else "all-degraded" if all(v <= 0 for v in zs) else "mixed")
        idx, picked = list(range(len(pop))), []
        while idx and len(picked) < CE_ROLLOUT:
            i = rng.choices(idx, weights=[CE_ROLLOUT_WEIGHTS[status[j]] for j in idx])[0]
            idx.remove(i)
            picked.append(i)
        history = "".join(f"## Attempt {n + 1} [{status[i]}]:\nAbstract: {pop[i]['abstract']}\nRaw Metrics: {{\n" +
                          ",\n".join(f" \"{k}\": {pop[i]['metrics'][k]:.4f}" for k in keys) + "\n}\n\n"
                          for n, i in enumerate(picked))
        return ctx.llm.ask([{"role": "system", "content": CE_NAV_SYSTEM},
                            {"role": "user", "content": CE_NAV_USER.format(history=history)}],
                           role="navigator", temperature=CE_TEMP_NAV)


def _fmt_metrics(m: dict) -> str:
    return ", ".join(f"{k}: {v:.4f}" for k, v in m.items())


# --------------------------------------------------------------------------------------------------------------------
# H31 Just-in-time vulnerability detection and remediation, Pipeline 1: generator -> LLM validator (CWE findings) ->
# enrichment from MITRE CWE observed examples, ATT&CK techniques and remediation guidance -> remediation.
# Prompts verbatim from the repository; enrichment data fetched at the pinned commit.

JIT_COMMIT = "e81dbdc2730b2560513d7b359a5229303b9059fb"
JIT_RAW = f"https://raw.githubusercontent.com/Droidsurikov/securing-ai-generated-code/{JIT_COMMIT}/"
JIT_SOURCES = {"cwe": "pipeline/enrich/data/cwe_enrichment_data.json", "guidance": "pipeline/enrich/cwe_guidance.py",
               "attack": "pipeline/enrich/cwe_to_attack.py", "techniques": "tools/mitre_db.py",
               "names": "pipeline/enrich/enricher.py"}
JIT_LITERALS = {"guidance": "CWE_REMEDIATION", "attack": "CWE_TO_ATTACK", "techniques": "TECHNIQUES", "names": "_CWE_NAMES"}

JIT_GENERATE = ("You are an expert {lang} developer. Generate complete, runnable {lang} code for the following task. Output "
                "only code, no explanations, no markdown fences.")

JIT_VALIDATOR = """You are a security code reviewer specialising in the CWE Top 25 Most Dangerous Software Weaknesses.

You will be given Python code. Your task is to identify any security vulnerabilities present.

Apply the following Python security best practices when reviewing:

Input & Injection
- Never use string formatting or concatenation to build SQL queries — use parameterised queries
- Never pass user input to subprocess with shell=True or to os.system()
- Validate and whitelist all user-supplied file paths; use os.path.realpath() to prevent traversal
- Never pass user input to eval(), exec(), or compile()

Deserialization & Parsing
- Avoid pickle.loads() on untrusted data; prefer JSON
- Use yaml.safe_load() not yaml.load() — the latter executes arbitrary Python
- Disable XML external entity (XXE) processing when using xml.etree or lxml

Secrets & Credentials
- Never hardcode passwords, API keys, or tokens — use environment variables or a secrets manager
- Use the secrets module for security-sensitive randomness, not random
- Use strong password hashing (bcrypt, argon2, pbkdf2_hmac) — never MD5/SHA1 or plaintext storage

Cryptography & Network
- Avoid weak algorithms: MD5, SHA1, DES, RC4
- Never set verify=False in requests calls (disables TLS certificate verification)
- Use TLS for all network communication

Permissions & Files
- Avoid world-writable permissions (0o777, 0o666) on files or directories
- Use the tempfile module for temporary files, not predictable paths under /tmp

Output & Errors
- Never log or expose stack traces, credentials, or PII in error messages
- Escape all user-supplied content before rendering in HTML; use auto-escaping templates

Defaults
- Never run with debug=True in production (Flask, Django, etc.)
- Apply the principle of least privilege — request only the permissions the code actually needs

For each vulnerability found, reason through:
1. Which CWE applies and why
2. The exact line(s) where the vulnerability exists
3. What an attacker could do to exploit it
4. What the fix should be

Then output your findings as a JSON array. Each item must have exactly these fields:
- "cwe_id": the CWE identifier (e.g. "CWE-89")
- "line_number": integer line number where the issue occurs, or null if not line-specific
- "severity": "high", "medium", or "low"
- "description": one sentence describing the specific vulnerability in this code
- "recommendation": one sentence describing the required fix

If no vulnerabilities are found, output an empty array: []

Output only the JSON array. No explanations, no markdown fences."""

JIT_REMEDIATE = """You are a security-focused Python developer.
You will be given Python code with identified security vulnerabilities,
enrichment context about each vulnerability (CWE details, ATT&CK techniques,
CVE examples, and remediation guidance), and you must produce a fixed version
of the code.

Before generating the fix, reason through:
1. Which specific CWEs are present and what they enable an attacker to do
2. What the ATT&CK techniques mapped to these CWEs tell you about the attack path
3. What the CVE examples show about real-world exploitation
4. What language-specific Python safeguards apply

Then output ONLY the fixed Python code. No explanations. No markdown fences.

Universal rules that must always apply:
- Never concatenate user input into SQL, shell commands, or file paths
- Never use pickle or yaml.load() with untrusted data
- Never hardcode credentials — use environment variables
- Always validate and sanitise all inputs before use
- Always use the most restrictive file permissions possible"""

JIT_FALLBACK_GUIDANCE = ("No specific remediation guidance is available for {cwe}. Apply defence-in-depth: validate all inputs, "
                         "enforce least privilege, keep dependencies up to date, and follow MITRE CWE mitigations for this "
                         "weakness.")


@h2h
class JustInTimeRemediation(H2HTechnique):
    """Generation (retried once on a syntax error), the LLM validator, local enrichment and one remediation call (retried once
    on a syntax error): 3 calls. The classifier call and the post-remediation judge do not affect the output and are omitted."""
    hid, name, family, target = "H31", "Just-in-time vulnerability remediation pipeline", "multiagent", ("SEC",)
    code_url = "https://github.com/Droidsurikov/securing-ai-generated-code"

    def prepare(self, ctx: Ctx) -> None:
        path = ctx.dir(self.hid) / "enrichment.json"
        if path.exists():
            return
        data = {}
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            for key, rel in JIT_SOURCES.items():
                text = c.get(JIT_RAW + rel).raise_for_status().text
                data[key] = json.loads(text) if key == "cwe" else _literal(text, JIT_LITERALS[key])
        path.write_text(json.dumps(data))

    def solve(self, task: Task, ctx: Ctx) -> str:
        lang = _lang(task)
        gen = f"{JIT_GENERATE.format(lang=lang)}\n\nTask: {C.task_text(task)}"
        code = ctx.keep(self._call(task, ctx, gen, "generate"))
        raw = ctx.llm.ask(f"{JIT_VALIDATOR}\n\nCODE TO REVIEW:\n{code}", role="validator")
        enrichment = self._enrich(ctx, _findings(raw))
        fix = (f"{JIT_REMEDIATE}\n\nORIGINAL CODE:\n{code}\n\n{enrichment}\n\n"
               f"Generate the fixed {lang} code:")
        return self._call(task, ctx, fix, "remediate") or code

    @staticmethod
    def _call(task: Task, ctx: Ctx, prompt: str, role: str) -> str:
        code = C.code_of(task, ctx.llm.ask(prompt, role=role))
        if task.language == "python" and not _compiles(task, code)[0]:
            code = C.code_of(task, ctx.llm.ask(prompt, role=role + "-retry"))
        return code

    def _enrich(self, ctx: Ctx, findings: list[dict]) -> str:
        if not findings:
            return "No vulnerability findings detected."
        d = _jit_data(str(ctx.dir(self.hid) / "enrichment.json"))
        lines = ["VULNERABILITY FINDINGS:"]
        for i, f in enumerate(findings, 1):
            cwe = f["cwe_id"] or ""
            lines += [f"[Finding {i}]", f"- CWE: {cwe or 'Unknown'} ({d['names'].get(cwe or 'Unknown', 'Unknown')})",
                      f"- Severity: {f['severity']}",
                      f"- Line: {f['line_number'] if f['line_number'] is not None else 'N/A'}",
                      f"- Description: {f['description']}"]
            techniques = [t for t in d["attack"].get(cwe, []) if t in d["techniques"]]
            if techniques:
                t = d["techniques"][techniques[0]]
                lines.append(f"- ATT&CK: {techniques[0]} ({t['name']}) [{t['tactic']}]")
            entry = d["cwe"].get(cwe)
            if entry:
                ctx_lines = [f"CWE Context — {entry['cwe_id']}: {entry['name']}"]
                if entry.get("description"):
                    ctx_lines.append(f"  Description: {entry['description'][:300]}")
                if entry.get("observed_examples"):
                    ctx_lines.append("  Observed Real-World Examples:")
                    ctx_lines += [f"    - {ex['cve_id']}: {ex['description'][:120]}" for ex in entry["observed_examples"][:3]]
                if entry.get("consequences"):
                    c0 = entry["consequences"][0]
                    ctx_lines.append(f"  Consequence: {', '.join(c0['scope'])} — {', '.join(c0['impact'])}")
                lines.append("\n".join(ctx_lines))
            lines.append(f"- Recommended fix: {d['guidance'].get(cwe) or JIT_FALLBACK_GUIDANCE.format(cwe=cwe)}")
        return "\n".join(lines)


def _literal(src: str, name: str):
    for n in ast.parse(src).body:
        tgt = n.target if isinstance(n, ast.AnnAssign) else n.targets[0] if isinstance(n, ast.Assign) else None
        if isinstance(tgt, ast.Name) and tgt.id == name:
            return ast.literal_eval(n.value)
    raise KeyError(name)


@lru_cache(maxsize=1)
def _jit_data(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _findings(raw: str) -> list[dict]:
    items = _json(raw)
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        cwe = str(it.get("cwe_id") or "").strip()
        cwe = cwe.upper() if re.fullmatch(r"CWE-\d+", cwe, re.I) else None
        line = it.get("line_number") if isinstance(it.get("line_number"), int) else None
        sev = str(it.get("severity", "medium")).lower()
        desc = str(it.get("description", ""))
        key = (cwe, line, desc[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append({"cwe_id": cwe, "line_number": line, "severity": sev if sev in ("high", "medium", "low") else "medium",
                    "description": desc})
    return out


# --------------------------------------------------------------------------------------------------------------------
# H32 LessonL (code optimization): three agents optimize the same source each round, write a lesson about their attempt,
# and lessons are banked and selected (k=4, speedup and quality) for later rounds. Prompts verbatim from code-opt.

LL_AGENTS = 3
LL_ROUNDS = 4
LL_K = 4
LL_TEMP = 0.2
LL_LESSON_TOKENS = 250
LL_HIGH_SPEEDUP = 1.1
LL_CODEBERT = {"python": "neulab/codebert-python", "cpp": "neulab/codebert-cpp"}

LL_OPT = ("You will be given a piece of code written in {lang}. Your task is to rewrite it in the same language to improve its "
          "performance (i.e., execution time). Do not change the input/output behaviors of the function. Include the "
          "generated code between ```{tag} and ```.\n\n// Code:\n{code}\n\n")
LL_OPT_LESSONS = ("You are given a piece of code written in {lang}. Your task is to rewrite it in the same language to improve "
                  "its performance (i.e., execution time). Do not change the input/output behaviors of the code. Some lessons "
                  "regarding correctness and performance are provided to help you rewrite the code. Include the generated code "
                  "between ```{tag} and ```.\n\n// Code:\n{code}\n\nWhile you rewrite the code, consider the following lessons. "
                  "If Code A and Code B appear in the lessons, Code A refers to the given code and Code B refers to an attempted "
                  "rewrite. Code B may not be optimal and it could be even worse than Code A.\n\n{lessons}\n\n")
LL_CORRECT = ("The following are two functionally equivalent codes. They are compiled by using the same compiler and executed in "
              "the same environment. Code B runs {fs} than Code A with a speedup {speedup}x. Explain why Code B is {fs}. Be "
              "brief in the explanations. Use only one or two sentences.\n\n// Code A:\n{a}\n\n// Code B:\n{b}\n\n")
LL_INCORRECT = ("The following two codes are not functionally equivalent; that is, given the same input, they produce different "
                "outputs. Explain the reasons that make Code B nonequivalent to Code A. Be brief in the explanations. Use only "
                "one or two sentences.\n\n// Code A:\n{a}\n\n// Code B:\n{b}\n\n")
LL_NOT_COMPILABLE = ("The following are two codes. Code B attempts to improve the performance of Code A, but it has syntactic "
                     "errors. Explain why Code B cannot be compiled. You may get hints from the compiler output provided after "
                     "Code B. Be brief in the explanations. Use only one or two sentences.\n\n// Code A:\n{a}\n\n// Code B:\n{b}"
                     "\n\nCompiler output:\n{log}\n\n")


@h2h
class LessonL(H2HTechnique):
    """The ParEval code-optimization configuration (3 agents, 4 rounds, k=4, temperature 0.2) ported to Python tasks: the
    source is the model's initial solution, correctness is the public examples and speedup is their wall time. All three
    agents are the E7 model. The configuration needs 25 calls, so the run stops at the ceiling (inside round 2)."""
    hid, name, family, target = "H32", "LessonL", "multiagent", ("EFF",)
    code_url = "https://github.com/MITIBM-FastCoder/LessonL"
    ported = True

    def prepare(self, ctx: Ctx) -> None:
        for name in LL_CODEBERT.values():
            _codebert(name)

    def solve(self, task: Task, ctx: Ctx) -> str:
        src = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        base = _cost(task, src)
        bank: list[dict] = []
        best = (1.0, src)
        count = 0
        for t in range(LL_ROUNDS):
            chosen = self._select(bank, t)
            used = {l["count"] for l in chosen}
            speedups = []
            for _ in range(LL_AGENTS):
                prompt = (LL_OPT_LESSONS.format(lang=_lang(task), tag=task.language, code=src, lessons=_lessons(chosen))
                          if chosen else LL_OPT.format(lang=_lang(task), tag=task.language, code=src))
                tgt = C.extract_code(ctx.llm.ask(prompt, role="optimizer", temperature=LL_TEMP), task.language)
                tag, speedup, log = self._evaluate(task, tgt, base)
                if speedup > best[0]:
                    best = (speedup, ctx.keep(tgt))
                speedups.append(speedup)
                lesson = ctx.llm.ask(self._lesson_prompt(src, tgt, tag, speedup, log), role="lesson",
                                     temperature=LL_TEMP, max_tokens=LL_LESSON_TOKENS)
                count += 1
                bank.append({"lesson": lesson, "tag": tag, "speedup": speedup, "factor": 1.0, "count": count,
                             "quality": _quality(lesson, src, task.language)})
            if t > 0:
                for l in bank:
                    if l["count"] in used:
                        l["factor"] = sum(1.1 if s > l["speedup"] else 0.9 for s in speedups) / len(speedups)
        return best[1]

    @staticmethod
    def _select(bank: list[dict], t: int) -> list[dict]:
        if t == 0:
            return []
        if t * LL_AGENTS <= LL_K:
            return list(bank)
        half = math.ceil(LL_K / 2)
        high = [l for l in bank if l["speedup"] * l["factor"] >= LL_HIGH_SPEEDUP]
        rest = [l for l in bank if l["speedup"] * l["factor"] < LL_HIGH_SPEEDUP]
        if len(high) > half:
            high = sorted(high, key=lambda l: -l["speedup"] * l["factor"])
            rest, high = rest + high[half:], high[:half]
        return high + sorted(rest, key=lambda l: -l["quality"])[:LL_K - len(high)]

    @staticmethod
    def _evaluate(task: Task, code: str, base: float | None) -> tuple[str, float, str]:
        ok, log = _compiles(task, code)
        if not ok:
            return "NOT_COMPILABLE", 0.0, log
        k, n, _ = _run_public(task, code)
        if k < n:
            return "INCORRECT", 0.0, ""
        c = _cost(task, code)
        return "CORRECT", (base / c if base and c else 1.0), ""

    @staticmethod
    def _lesson_prompt(src: str, tgt: str, tag: str, speedup: float, log: str) -> str:
        if tag == "CORRECT":
            fs = "slightly faster" if 1.0 <= speedup < 1.1 else "significantly faster" if speedup >= 1.1 else "slower"
            return LL_CORRECT.format(fs=fs, speedup=round(speedup, 2), a=src, b=tgt)
        if tag == "INCORRECT":
            return LL_INCORRECT.format(a=src, b=tgt)
        return LL_NOT_COMPILABLE.format(a=src, b=tgt, log="\n".join(log.splitlines()[:7]))


def _lessons(chosen: list[dict]) -> str:
    out = ""
    for i, l in enumerate(chosen, 1):
        if l["tag"] == "CORRECT":
            s = l["speedup"]
            if 1 <= s < 1.1:
                out += (f"\nLesson {i} slightly improves the code performance. {l['lesson']}. However, despite the code "
                        f"performance improvement, the speedup is only marginal.")
            elif s > 1.1:
                out += f"\nLesson {i} significantly improves the code performance. {l['lesson']}."
            else:
                out += f"\nLesson {i} degrades the code performance. {l['lesson']}"
        elif l["tag"] == "INCORRECT":
            out += f"\nLesson {i} compromises code equivalence. {l['lesson']}"
        else:
            out += f"\nLesson {i} produces non-compilable code. {l['lesson']}"
    return out + "\nBesides the above lessons, consider other optimization strategies that can more significantly improve the " \
                 "performance of the given code."


@lru_cache(maxsize=2)
def _codebert(name: str):
    from transformers import AutoModel, AutoTokenizer
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


def _quality(lesson: str, src: str, lang: str) -> float:
    """Cosine similarity of the [CLS] CodeBERT embeddings of the lesson and the source code."""
    import torch
    tok, model = _codebert(LL_CODEBERT.get(lang, LL_CODEBERT["python"]))
    with _LOCK, torch.no_grad():
        e = [model(**tok(x, return_tensors="pt", truncation=True)).last_hidden_state[0, 0] for x in (lesson, src)]
    return float(torch.nn.functional.cosine_similarity(e[0], e[1], dim=0))


# --------------------------------------------------------------------------------------------------------------------
# H33 MARCO: context LLM writes search queries; per iteration web search, technique summarization, code generation and
# a performance evaluator that runs the code. Reconstructed from the paper (no prompts or code released).

MARCO_ITERATIONS = 3
MARCO_RESULTS = 5
MARCO_DOMAINS = ["acm.org", "ieee.org", "arxiv.org", "researchgate.net"]

MARCO_CONTEXT = """You are an expert in high-performance computing and code optimization. Interpret the {lang} code below and write {n} web search queries, each aimed at a different optimization technique from the research literature that could reduce its execution time or memory use (algorithms, data structures, language-specific performance techniques).

Task the code solves:
{task}

Code:
{code}

Return one JSON list of {n} query strings."""

MARCO_SUMMARIZE = """You are an expert in code optimization. Condense the search results below into ONE concrete optimization technique for this code: name it, explain how to apply it to this code and why it should reduce execution time or memory use. Take the latest performance report into account. Be concise.

Code:
{code}

Latest performance report:
{report}

Search results:
{results}"""

MARCO_OPTIMIZE = """You are the code optimizer agent. Apply the optimization technique below to the code to reduce its execution time and memory use without changing its input/output behaviour or signature.

Task the code solves:
{task}

Optimization technique:
{technique}

Performance evaluator report on the current code:
{report}

Current code:
{code}

Return the complete optimized code in one fenced code block."""

PEAK_PREFIX = "import tracemalloc as _tm\n_tm.start()\n"
PEAK_SUFFIX = "\nimport sys as _sys\n_sys.stderr.write(f'@@PEAK {_tm.get_traced_memory()[1]}\\n')\n"


@h2h
class MARCO(H2HTechnique):
    """Initial code, one context call for the queries, then three iterations of Tavily search (academic domains), technique
    summary and code generation; the performance evaluator runs the public examples (time, peak memory, correctness) and
    its report feeds the next iteration. The paper leaves the iteration count to the user; 3 is used (8 calls)."""
    hid, name, family, target = "H33", "MARCO", "multiagent", ("EFF",)

    def prepare(self, ctx: Ctx) -> None:
        if not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError("H33 needs TAVILY_API_KEY for its web-search tool")

    def solve(self, task: Task, ctx: Ctx) -> str:
        cache = ctx.dir(self.hid) / "search_cache.jsonl"
        text = C.task_text(task)
        best = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        best_eval = _perf(task, best)
        queries = _json(ctx.llm.ask(MARCO_CONTEXT.format(lang=_lang(task), n=MARCO_ITERATIONS, task=text,
                                                         code=_block(task, best)), role="context"))
        queries = [str(q) for q in queries if str(q).strip()] if isinstance(queries, list) else []
        report = _perf_report(best_eval, None)
        for i in range(MARCO_ITERATIONS):
            query = queries[i] if i < len(queries) else f"{_lang(task)} performance optimization {text[:200]}"
            hits = _web_search(cache, query, search_depth="advanced", max_results=MARCO_RESULTS,
                               include_domains=MARCO_DOMAINS)
            results = "\n\n".join(f"[{j}] {h['title']} ({h['url']})\n{h['content']}" for j, h in enumerate(hits, 1))
            technique = ctx.llm.ask(MARCO_SUMMARIZE.format(code=_block(task, best), report=report,
                                                           results=results or "No results."), role="summarizer")
            cand = C.code_of(task, ctx.llm.ask(MARCO_OPTIMIZE.format(task=text, technique=technique, report=report,
                                                                     code=_block(task, best)), role="optimizer"))
            ev = _perf(task, cand)
            report = _perf_report(ev, best_eval)
            faster = ev["time"] is not None and best_eval["time"] is not None and ev["time"] < best_eval["time"]
            unmeasured = not task.public_tests and _compiles(task, cand)[0]
            if ev["correct"] and (not best_eval["correct"] or faster or unmeasured):
                best, best_eval = ctx.keep(cand), ev
        return best


def _perf(task: Task, code: str) -> dict:
    k, n, fails = _run_public(task, code) if code.strip() else (0, len(task.public_tests), "empty program")
    ev = {"passed": k, "total": n, "correct": bool(code.strip()) and k == n, "failures": fails, "time": None, "peak": None}
    if ev["correct"]:
        ev["time"] = _cost(task, code)
        if task.language == "python" and task.public_tests:
            peaks = []
            for t in task.public_tests:
                prog, stdin = _test_program(task, code, t)
                r = S.run_program(PEAK_PREFIX + prog + PEAK_SUFFIX, "python", stdin, task.time_limit, task.memory_mb)
                peaks += [int(m) for m in re.findall(r"@@PEAK (\d+)", r.stderr)]
            ev["peak"] = max(peaks) if peaks else None
    return ev


def _perf_report(ev: dict, prev: dict | None) -> str:
    lines = [f"Correctness: {ev['passed']} of {ev['total']} examples passed."]
    if ev["failures"]:
        lines.append(f"Failures:\n{ev['failures']}")
    if ev["time"] is not None:
        lines.append(f"Execution time (median, summed over examples): {ev['time'] * 1000:.2f} ms"
                     + (f" (previous best {prev['time'] * 1000:.2f} ms)" if prev and prev.get("time") else ""))
    if ev["peak"] is not None:
        lines.append(f"Peak traced memory: {ev['peak'] / 1024:.1f} KiB"
                     + (f" (previous best {prev['peak'] / 1024:.1f} KiB)" if prev and prev.get("peak") else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# H34 C3, decentralized communication-aware competition (DCC) in a MetaGPT-style pipeline. Reconstructed from the paper.

C3_COMPETITORS = 3
C3_SCORES = re.compile(r'\{\s*"scores"\s*:\s*\{[^{}]*\}\s*\}')
C3_ROUNDS = 1
C3_CRITERIA = {"Architect": "novelty, feasibility and compliance with the standard operating procedure",
               "Engineer": "implementability and performance"}

C3_PM = """You are a Product Manager in a software company that follows a standard operating procedure. Write a concise product requirement document for the task below: goals, user stories or use cases, requirement analysis, a requirement pool with priorities, and anything unclear. Do not write code.

Task:
{task}"""

C3_ARCH = """You are an Architect in a software company that follows a standard operating procedure. Based on the product requirement document, write the system design: the implementation approach (algorithms and data structures with their complexity), the structure of functions and classes with signatures, the program call flow, and anything unclear. Do not write the implementation.

Task:
{task}

Product requirement document:
{prd}"""

C3_ENG = """You are an Engineer in a software company that follows a standard operating procedure. Implement the task following the product requirement document and the system design. Write complete, correct, readable and maintainable code.

Product requirement document:
{prd}

System design:
{design}"""

C3_EVOLVE = """You are {role} {me} in an open competition with the other {role}s working on the same phase. All proposals are shared in a message pool.

Task:
{task}

{inputs}Your proposal ({role} {me}):
{own}

Peer proposals:
{peers}

First, score each peer proposal from 1 to 10 for {criteria}, with one line of justification each. Then improve your own proposal: keep its strengths, adopt the strong elements of the peers' proposals and drop weaker strategies. Output the scores as one JSON object {{"scores": {{"<peer number>": <score>, ...}}}}, then your complete improved proposal{suffix}."""

C3_QA_CALL = """You are a QA Engineer. Write unit tests for the function `{entry}` of the task below, from the requirement document and the examples in the task: representative inputs, edge cases and invalid inputs, with their expected results.

Task:
{task}

Product requirement document:
{prd}

Write only assert statements that call `{entry}` directly, one per line, in one ```python block."""

C3_QA_IO = """You are a QA Engineer. Write tests for the program of the task below, which reads standard input and writes standard output: representative inputs and edge cases with their exact expected output, derived from the requirement document and the examples in the task.

Task:
{task}

Product requirement document:
{prd}

Return one JSON list: [{{"stdin": "...", "expected_stdout": "..."}}]."""

C3_FIX = """You are the Engineer whose implementation won the competition. The QA Engineer ran the tests and these failed:

{report}

Task:
{task}

Code:
{code}

Fix the code. Return the complete code in one fenced code block."""


@h2h
class C3(H2HTechnique):
    """Product Manager, then 3 competing Architects and 3 competing Engineers (engineer i builds on design i), each with
    one DCC round in which a competitor scores its peers' proposals and evolves its own; a QA engineer writes tests, the
    pool is ranked by tests passed then peer utility, and the winner gets one fix if it fails (at most 15 calls). The paper
    runs 3-5 rounds per phase; one round is what fits the ceiling."""
    hid, name, family, target = "H34", "C3", "multiagent", ("RM",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        text = C.task_text(task)
        prd = ctx.llm.ask(C3_PM.format(task=text), role="product-manager")
        designs = ctx.llm.sample(C3_ARCH.format(task=text, prd=prd), C3_COMPETITORS, role="architect")
        designs, _ = self._compete(task, ctx, "Architect", designs, f"Product requirement document:\n{prd}\n\n", "")
        codes = [C.code_of(task, ctx.llm.ask(C.messages(task, post=C3_ENG.format(prd=prd, design=d)), role="engineer"))
                 for d in designs]
        ctx.keep(codes[0])
        suffix = f" in one fenced ```{task.language} code block"
        raw, utility = self._compete(task, ctx, "Engineer", [_block(task, c) for c in codes],
                                     f"Product requirement document:\n{prd}\n\n", suffix)
        codes = [C.code_of(task, r) or c for r, c in zip(raw, codes)]
        ctx.keep(codes[0])
        how = _executable(task)
        if how == "call":
            src = C.code_of(task, ctx.llm.ask(C3_QA_CALL.format(entry=task.entry_point, task=text, prd=prd), role="qa"))
            suite = [("qa", ast.unparse(s)) for s in _statements(src)]
        elif how == "io":
            raw_qa = _json(ctx.llm.ask(C3_QA_IO.format(task=text, prd=prd), role="qa"))
            suite = [c for c in raw_qa if isinstance(c, dict)] if isinstance(raw_qa, list) else []
        else:
            suite = []
        reports = [self._qa(task, c, how, suite) for c in codes]
        order = sorted(range(len(codes)), key=lambda i: (-reports[i][0], -utility[i], i))
        best = codes[order[0]]
        ctx.keep(best)
        if reports[order[0]][1]:
            fixed = C.code_of(task, ctx.llm.ask(C3_FIX.format(report=reports[order[0]][1], task=text,
                                                              code=_block(task, best)), role="engineer-fix"))
            if fixed and self._qa(task, fixed, how, suite)[0] >= reports[order[0]][0]:
                best = ctx.keep(fixed)
        return best

    @staticmethod
    def _compete(task: Task, ctx: Ctx, role: str, proposals: list[str], inputs: str,
                 suffix: str) -> tuple[list[str], list[float]]:
        text = C.task_text(task)
        scores: list[list[float]] = [[] for _ in proposals]
        for _ in range(C3_ROUNDS):
            new = []
            for i, own in enumerate(proposals):
                peers = "\n\n".join(f"[{role} {j + 1}]\n{p}" for j, p in enumerate(proposals) if j != i)
                out = ctx.llm.ask(C3_EVOLVE.format(role=role, me=i + 1, task=text, inputs=inputs, own=own, peers=peers,
                                                   criteria=C3_CRITERIA[role], suffix=suffix), role=f"{role.lower()}-dcc")
                m = C3_SCORES.search(out)
                try:
                    given = json.loads(m.group())["scores"] if m else {}
                except json.JSONDecodeError:
                    given = {}
                for j, s in given.items():
                    n = re.search(r"\d+", str(j))
                    if n and 1 <= int(n.group()) <= len(proposals) and int(n.group()) != i + 1:
                        scores[int(n.group()) - 1].append(_num(s))
                body = out[:m.start()] + out[m.end():] if m else out
                new.append(re.sub(r"```(?:json)?\s*```", "", body).strip() or own)
            proposals = new
        return proposals, [statistics.fmean(s) if s else 0.0 for s in scores]

    @staticmethod
    def _qa(task: Task, code: str, how: str, suite) -> tuple[int, str]:
        res = _run_cases(task, code, suite) if how == "call" else _run_io_cases(task, code, suite) if how == "io" else []
        k, n, public = _run_public(task, code)
        fails = [f"- {s}\n  {e}" for _, s, e in res if e]
        report = "\n".join(fails) + (f"\n\nTask examples failed ({n - k} of {n}):\n{public}" if k < n else "")
        return k * 1000 + sum(1 for *_, e in res if not e), report.strip()


# --------------------------------------------------------------------------------------------------------------------
# H35 ACCLAIM: a guiding agent interleaves free compiler components with budgeted level-specific optimizing agents, scored
# by a testing agent's input-generator script. Prompts and loops from the repository (commit 9db78ec), ported to CPython.

ACC_BUDGET, ACC_SAMPLES, ACC_LOOPS = 18, 2, 2
ACC_TEMP, ACC_MAX_TOKENS = 1.0, 4096
ACC_TEST_TIMEOUT, ACC_WARMUP, ACC_RUNS = 30, 5, 10
ACC_CORRECTNESS = range(1, 11)
ACC_LARGE = (10, 100, 1000, 10000, 100000)
ACC_SCRIPT_TIMEOUT = 60
ACC_RUN_SECONDS = 3600
ACC_DIR = "work"

ACC_FRONTEND = "import ast, sys\nprint(ast.dump(ast.parse(sys.stdin.read()), indent=1))\n"
ACC_BACKEND = "import dis, sys\ndis.dis(compile(sys.stdin.read(), 'main.py', 'exec'))\n"
ACC_UNPARSE = "import ast, sys\nprint(ast.unparse(ast.fix_missing_locations(eval(sys.stdin.read(), dict(vars(ast))))))\n"
ACC_CALL = "\n\nimport json as _acc_json, sys as _acc_sys\nprint(repr({entry}(*_acc_json.loads(_acc_sys.stdin.read()))))\n"
ACC_ARGV = "import sys\nsys.argv = ['test_script.py', '{n}']\n"

ACC_PLANNER_PY = """You are a planning agent responsible for converting a given Python source code file into the most efficient possible Python program. Optimization means the rewritten code should achieve the minimum possible running time without changing the semantics of the original. The CPython compiler processes Python code at 3 levels of abstraction: Python source code, the CPython abstract syntax tree (AST), and CPython bytecode, which the interpreter executes. You are given tools that can convert between each of these levels, and tools which perform optimization at the source code and AST levels. The final output must be Python source code, so a program optimized at AST level must be converted back to source code with the unparse tool before it is written to the output file.

Moving between levels by calling the existing CPython compiler is computationally cheap, but performing optimization at a specific level using the provided tools is computationally expensive. You will therefore be given a computational budget, which is an integer that gives the number of times you may run level-specific optimization. This budget will decrease each time you call a tool to perform an optimization at a particular level of abstraction. You may continue optimizing until you conclude that there is no more room for optimization; there is no benefit to having compute budget left over at the end.

To decide at which levels to optimize, you should consider all of the available files, including source code any any code at levels of abstraction above the one you are considering. Select a level at which optimization will provide the most benefit for this specific input program. For example, if the program contains an implementation of an algorithm that runs in O(n^3) time for input size n, but there is another implementation of that algorithm that runs at O(n) time, you should optimize it at source code level to use that faster algorithm. On the other hand, if the bytecode CPython generates from the input repeats work in a hot loop that the compiler does not remove, such as a global or attribute lookup or a constant expression evaluated in every iteration, then you should invoke the AST level optimizer to restructure the tree, since the CPython compiler performs only light optimizations (constant folding and peephole rewrites) on its way to bytecode.

Once you have chosen a level of abstraction, use the provided compiler tools to compile to that level and then invoke the level-specific optimization tool you have access to. If you haven't used up the entire compute budget, you should try to optimize at more than one level. For example, you may first call the tool to optimize at source code level, then call the compiler frontend to convert to the AST, and then call the tool to optimize at AST level too. If you think that there is further opportunity for improvement at a single level, you may call optimization for that level again with some instructions on how to perform further optimization. For example, if you have called the tool for source code optimization and received an optimized source code, you might suspect that it's possible to further optimize by considering a new type of loop transformation. In that case, call the source code optimization tool again with the corresponding instructions on what types of optimization to try. Make sure that you pass the already optimized code as the source code to the level-specific tool instead of the original input source so that it can iterate on the best existing result.

After each level-specific optimization call, you will receive feedback giving the speedup of the optimized code. The speedup is given relative to 1.0, so a speedup of 1.0 means no change in performance, speedups below 1.0 mean performance got worse, and speedups above 1.0 mean performance got better. The higher the speedup, the better. It is possible that an optimization at some level of abstraction will actually slow down the performance. If this happens, you should revert to the previous best version and not use the one produced by that optimization. You might, for instance, call that level of optimization again with different feedback instructions to see if performance can be further improved. If you receive feedback indicating a speedup of -1, this means that optimization at the called level did not succeed.

Continue optimizing and using the existing tools until you either run out of budget or are satisfied that no further optimization at any level is possible. Make sure to write the final version to the specified output file. Never attempt to optimize at any level yourself; only use the provided level-specific tools to do so. If any of those tools fail, just report the error and do not continue."""

ACC_PLANNER_SRC = """You are a planning agent responsible for converting a given {lang} source code file into the most efficient possible {lang} program. You are given an LLM-based tool for optimizing at source code level only. You should not attempt to perform any optimizations at other levels of abstraction.

Performing optimization at source level using the provided tool is computationally expensive. You will therefore be given a computational budget, which is an integer that gives the number of times you may run source optimization. This budget will decrease each time you call the tool to perform an optimization. You may continue optimizing until you conclude that there is no more room for optimization; there is no benefit to having compute budget left over at the end.

Optimization means the rewritten code should achieve the minimum possible running time without changing the semantics of the original. For example, if the program contains an implementation of an algorithm that runs in O(n^3) time for input size n, but there is another implementation of that algorithm that runs at O(n) time, you should optimize it to use that faster algorithm. If you haven't used up the entire compute budget, you can try to call the optimization tool again. Continue optimizing until you either run out of budget or are satisfied that no further optimization is possible. Make sure to write the final version to the specified output file.

After each optimization call, you will receive feedback giving the speedup of the optimized code. The speedup is given relative to 1.0, so a speedup of 1.0 means no change in performance, speedups below 1.0 mean performance got worse, and speedups above 1.0 mean performance got better. The higher the speedup, the better. It is possible that an optimization will actually slow down the performance. If this happens, you should revert to the previous best version and not use the one produced by that optimization. You might, for instance, call the optimization tool again with different feedback instructions to see if performance can be further improved. If you receive feedback indicating a speedup of -1, this means that optimization did not succeed."""

ACC_LEVEL_DOC = """{name}(source_file, out_file, feedback=None): Optimizes {what} code by invoking an LLM to rewrite the code and make it as efficient as possible.
    There is no guarantee that this agent will make the code more efficient; it's possible that it could decrease performance.
    The feedback argument is an opportunity to give instructions about what type of optimization to perform, or feedback about lessons learned from past optimizations.
    If you are unsure what types of optimizations are required, do not pass anything as feedback. If there are particular types of optimization to perform, then this can be mentioned.
    In particular, if you have learned in the past that a particular type of optimization made performance worse, then make sure to mention not to perform that type of optimziation again in the feedback
    Only call this tool if you think that it is likely that LLM-rewriting at {where} level is likely to improve the performance of the code.
    Args:
        source_file: the input {what} code file to rewrite
        out_file: the output {what} file to write results
        feedback: A string with instructions on what type of optimization to perform on feedback about what was learned from previous optimizations
    Returns:
        A summary of whether the agent succeeded in optimizing the source code, and what optimizations it performed"""

ACC_COMPILER_DOC = """{name}({arg}, out_file): {what}
    Returns:
        A string containing "success" if call succeeded, and otherwise an error message from the compiler"""

ACC_TOOLS = {
    "file_read": "file_read(path): Reads a file and returns its contents.",
    "file_write": "file_write(path, content): Writes content to a file, replacing the file if it exists.",
    "frontend": ACC_COMPILER_DOC.format(name="frontend", arg="source_file", what="Invokes the frontend of the CPython compiler "
                                        "(its parser) to convert Python source code stored in a file into the CPython abstract "
                                        "syntax tree (AST), which is output to another file."),
    "backend": ACC_COMPILER_DOC.format(name="backend", arg="ir_file", what="Invokes the backend of the CPython compiler (code "
                                       "generation and its peephole optimizer) to convert a CPython AST or Python source stored in "
                                       "a file into CPython bytecode, which is output to another file as a disassembly listing."),
    "unparse": ACC_COMPILER_DOC.format(name="unparse", arg="ir_file", what="Converts a CPython AST stored in a file back into "
                                       "Python source code, which is output to another file."),
    "source": ACC_LEVEL_DOC.format(name="source", what="{lang} source", where="source code"),
    "ir": ACC_LEVEL_DOC.format(name="ir", what="CPython AST", where="CPython AST"),
}

ACC_PROTOCOL = """You have access to the following tools:

{tools}

To call tools, reply with one JSON object {{"tool": "<tool name>", "args": {{...}}}} in a ```json block, or with a JSON list of such objects to call several tools in order; their results are returned in the next message. When you are done, reply without a tool call."""

ACC_RECORDED = "agent.tool.file_read direct tool call.\nInput parameters: {params}\nResult:\n{content}"
ACC_OPEN = ("Optimize the {lang} source code file located at {src}, and write the final {lang} source code output to {out}. "
            "Please write any intermediate files into the directory {dir}. The initial budget is {budget}")

ACC_LEVEL = """You are a an expert at re-writing {language} code to make it more efficient. You may use any optimization strategies you know of.
Optimization means the rewritten code should achieve the minimum possible running time without changing the semantics of the original.
For example, here is a type of optimization not performed by a compiler:

int i = 0;
while( i < 10 ) {
  int j = 3 * i + 2;
  a[j] = a[j] - 2;
  i = i + 2;
}

In this code, the computation int j = 3 * i + 2, combined with the fact that i is incremented by 2 each time,
means that j has the initial value of 2 and is incremented by 6 at each loop iteration. Therefore, the following code is an optimization:

int i = 0;
int j = 2;
while( i < 10 ) {
  a[j] = a[j] - 2;
  i = i + 2;
  j = j + 6;
}

This code runs faster because it has removed a multiplication instruction from the loop body.

Make sure that your responses are always clear and concise. They should by only one or two sentences long and should clearly state whether you succeeded in performing an optimization, and if so, what optimizations you performed."""

ACC_QUERY = "Optimize the {what} code in the file {src} and write the optimized code to {out}. {feedback}"
ACC_CALLER = "Here is some feedback or instructions on optimization from the calling agent: {fb}."
ACC_INLINE = ("\n\nContents of {src}:\n```{tag}\n{code}\n```\n\nWrite {out} by replying with its complete contents in one fenced "
              "code block, followed by your summary.")

ACC_FB_UNKNOWN = ("Previously, you attempted to generate an optimized program based on the following instructions: {fb}. That "
                  "generation failed to generate a successful optimization of {src} for unknown reasons.")
ACC_FB_WRONG = ("Previously, you attempted to generate an optimized program based on the following instructions: {fb}. That "
                "generation failed to generate a successful optimization of {src}. The generation failed for the following "
                "reasons; please attempt to correct these issues and generate a correct and performant program: {reasons}")
ACC_FB_BETTER = ("Previously, you attempted to generate an optimized program based on the following instructions: {fb}. This "
                 "produced the program at {out}, with speedup of {s}x when measured against the original implementation in "
                 "{base}. Now, you should attempt to further boost this speedup.")
ACC_FB_WORSE = ("Previously, you attempted to generate an optimized program based on the following instructions: {fb}. This "
                "attempt was worse than the previous best, with speedup of {s}x when measured against the original "
                "implementation in {base}. Instead, return to the previous best implementation at {best} and try something else.")
ACC_DONE = "Generated successful program at {out}, with speedup f{s}x. Remaining budget is {b}."
ACC_FAILED = "Failed to generate successful optimization of {src}. Remaining budget is {b}."
ACC_EXHAUSTED = "Level optimization budget exceeded; no further optimization is possible."

ACC_TESTER = """You are a an expert at evaluating and testing software. You will be given the description of a competitive programming problem and several C programs that attempt to solve that problem. Your goal is to generate a python script that creates inputs which will be given to programs solving the problem via stdin. Do not read all of the example C files; if more than 3 are provided, simply read 3 of them to get a sense of the types of programs that exist for this problem.
First, you should generate specific test inputs which exercise important functionality which is not covered by random tests.
Next, you should generate a python function that generates a random test each time it is called, with the size of each test growing with the input passed. Then, combine these two sets of testing into a single python script that takes an integer from stdin starting with 1, 2, 3, etc. For 1, the script should output the first specific test case. For 2, it should output the second specific test case, etc, until all specific cases have been exhausted. Once all of the specific tests cases have been output, the next highest number and all subsequent inputs should produce a random test case that is proportional in size to the input given.
Here is an example of the type of reasoning you should follow. Below is a C function. The description of this function is that it is a program that counts how many of the characters at the start of two strings are the same, and outputs this result. If either string is "STOP", or if the length of either string is 0, the program returns 0.

int main(void)
{
    int count = 0;
    int len;
    char x[NUMBER];
    char y[NUMBER];

    scanf("%s %s",x,y);
    scanf("%d",&len);

    if (x == "STOP" || y == "STOP" || len == 0)
    {
        printf("0\\n",count);
        return 0;
    }

    for (int i = 0; i < len;i++){
        if (x[i] == y[i]){
            count++;
        }
    }

    printf("%d\\n",count);
    return 0;
}

First, evaluate what types of input this function takes. We see that it uses scanf two times: first, it reads two strings to the variables x and y. Then, it reads an integer to the variable len. That means that its inputs consist of two strings, then two integers, then one more string. All test inputs to the program should follow this format.
Now, we consider what types of values should be tested with. First, let's look for specific values that aren't likely to be caught by random testing. The line if (x == "STOP" || y == "STOP" || len == 0) checks for three specific values: x == "STOP", y == "STOP", and len == 0. All of these values are unlikely to come up if we test only with random inputs, so we should add tests that check them. In addition, the loop compares values from x and y up to len. This might be an error in the program, since len could be longer than the length of the strings, so we need to expose this possible error. That means we should have some test cases where len is longer than the length of either x or y. We should also test different combinations, like x being longer than y, x and y being the same length, etc.
In addition to these specific cases, your role is to generate a script that generates random test cases once the defined ones have been exhausted. Make sure that all of the random test cases have the right format to exercise the program. In this example, the random test cases should consist of two random strings, each with some length n, and then the third input should be n itself. The generated test cases should be proportional in size to the input to the testing script; in this example, that means that we can use the input n as value n in the test case.

Below is an example python script that generates such inputs for the above example. Note that for values up to n == 6, specific test cases are used, and for larger values of n, the test function produces random inputs that follow the format described earlier, using n as the length. You will need to modify the test function from this example for a different input program, but may reuse random_int, random_float, and random_string. Always set the random seed according to the input to the function test to make the testing script deterministic.

import random
import sys
import string

def random_int(lower = -2**31, upper = 2**31 - 1):
    return random.randint(lower, upper)

def random_floats(lower = -2**31, upper = 2**31 - 1):
    return random.uniform(lower, upper)

def random_string(length  = None):
    if not length:
        length = random.randint(0,20)
    return ''.join(random.choices(list(string.ascii_letters) + list(string.digits), k=length))


def test(n):
    random.seed(n)
    if n == 1:
        #To test the specific string STOP in the input
        return f"STOP {random_string(10)} 10"
    elif n == 2:
        #To test the specific string STOP as the second input string
        return f"{random_string(13)} STOP 13"
    elif n == 3:
        #To test when both input strings are STOP
        return f"STOP STOP {random_int(1,100)}"
    elif n == 4:
        #To test the specific input len = 0
        return f"{random_string()} {random_string()} 0"
    elif n == 5:
        #To handle where len is longer than the two strings
        return f"{random_string(10)} {random_string(10)} 20"
    elif n == 6:
        #Different lengths for the strings
        return f"{random_string(8)} {random_string(20)} 20"
    else:
        return f"{random_string(n)} {random_string(n)} {n}"

print(test(int(sys.argv[1])))
"""

ACC_TEST_QUERY = ("Create a test case generation script for the problem described below, with an example solution {lang} program "
                  "below. Write the python test generation script in one fenced code block.\n\nProblem description:\n{problem}"
                  "\n\nExample solution:\n```{tag}\n{code}\n```")
ACC_TEST_CALL = ("\n\nThe example solution is the function `{entry}`, which is called directly instead of reading standard input: "
                 "for each number, the script must print one JSON list holding the positional arguments of one call.")


@h2h
class ACCLAIM(H2HTechnique):
    """Published configuration b=18, n=2, k=2 at temperature 1 (4096 tokens): the guiding agent calls compiler tools for free
    and level-specific agents (n samples, k feedback loops, best correct sample kept) against the budget; the testing agent's
    script yields C=10 correctness and L=5 large-scale inputs, correctness is output equality with the initial program and
    performance a hyperfine-style mean (5 warmups, 10 runs, 30 s). Port from C/LLVM/x86 to CPython 3.12: source, AST (the IR,
    lifted back exactly by ast.unparse) and bytecode; frontend = parser, backend = bytecode compiler (inspection only, bytecode
    cannot be delivered as source, so no assembly agent), an unparse tool replaces lowering as the final step, and there is no
    middle-end tool since CPython exposes no standalone IR optimizer. Other languages run the source-only configuration. One
    call per guiding turn (JSON tool calls) and per level sample; b x n x k alone is 72 calls, so runs end at the ceiling with
    the best correct program."""
    hid, name, family, target = "H35", "ACCLAIM", "multiagent", ("EFF",)
    code_url = "https://github.com/amazon-science/acclaim"
    ported = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        base = draft(task, ctx)
        entry = _acc_entry(task, base)
        if not base.strip() or not (entry or task.mode == "stdio"):
            return base
        run = _AcclaimRun(task, ctx, base, entry)
        return run.guide() if run.testing_agent() else base


class _AcclaimRun:
    """One ACCLAIM run over a virtual file tree: compiler tools, level-specific agents, testing agent and guiding agent."""

    def __init__(self, task: Task, ctx: Ctx, base: str, entry: str):
        self.task, self.ctx, self.entry = task, ctx, entry
        self.ext = Path(S.LANGS[task.language]["file"]).suffix
        self.base, self.out = f"{ACC_DIR}/source{self.ext}", f"{ACC_DIR}/generated{self.ext}"
        self.files = {self.base: base}
        self.budget, self.counter, self.best = ACC_BUDGET, 0, (1.0, base)
        self.inputs: dict[str, str] = {}
        self.ref: dict[str, dict] = {}

    def testing_agent(self) -> bool:
        t, base = self.task, self.files[self.base]
        query = ACC_TEST_QUERY.format(lang=_lang(t), problem=t.prompt, tag=t.language, code=base)
        if self.entry:
            query += ACC_TEST_CALL.format(entry=self.entry)
        raw = self.ctx.llm.ask([{"role": "system", "content": ACC_TESTER}, {"role": "user", "content": query}], role="tester",
                               temperature=ACC_TEMP, max_tokens=ACC_MAX_TOKENS)
        script, generated = C.extract_code(raw, "python"), {}
        for n in dict.fromkeys([*ACC_CORRECTNESS, *ACC_LARGE]):
            r = S.run_program(ACC_ARGV.format(n=n) + script, "python", "", ACC_SCRIPT_TIMEOUT, t.memory_mb)
            if r.ok and r.stdout.strip():
                generated[f"t{n}.in"] = r.stdout
        kind = "call" if self.entry else "io"
        public = {f"example{i}.in": x.code if kind == "call" else x.stdin
                  for i, x in enumerate(t.public_tests, 1) if x.kind == kind}
        for inputs in (generated, public):
            ref = self.execute(base, "source", inputs)
            self.inputs = {k: v for k, v in inputs.items() if ref.get(k, {}).get("status") in ("ok", "timeout")}
            if self.inputs:
                self.ref = {k: ref[k] for k in self.inputs}
                return True
        return False

    def execute(self, code: str, level: str, inputs: dict[str, str] | None = None) -> dict[str, dict]:
        """The testing agent's second phase: compile from the given level, run every input, keep output and mean time."""
        t = self.task
        if level == "ir":
            code, err = self._py(ACC_UNPARSE, code)
            if err:
                return {"compiler": {"status": "failure", "output": err}}
        if t.language == "python":
            ok, log = _compiles(t, code)
            if not ok:
                return {"compiler": {"status": "failure", "output": log}}
            code = (t.meta["prelude"] + "\n" + code if t.meta.get("prelude") else code) + \
                (ACC_CALL.format(entry=self.entry) if self.entry else "")
        spec = S.LANGS[t.language]
        with S.Workdir({spec["file"]: code}) as d:
            b = S.build(t.language, d)
            if not b.ok:
                return {"compiler": {"status": "failure", "output": b.stderr}}
            return {k: _acc_hyperfine(spec["run"], d, x, t.memory_mb)
                    for k, x in (self.inputs if inputs is None else inputs).items()}

    def tools(self) -> dict:
        tools = {"file_read": self.file_read, "file_write": self.file_write, "source": partial(self.optimize, "source")}
        if self.task.language == "python":
            tools |= {"frontend": self.frontend, "backend": self.backend, "unparse": self.unparse,
                      "ir": partial(self.optimize, "ir")}
        return tools

    def guide(self) -> str:
        t, tools, lang = self.task, self.tools(), _lang(self.task)
        head = ACC_PLANNER_PY if t.language == "python" else ACC_PLANNER_SRC.format(lang=lang)
        docs = "\n\n".join(ACC_TOOLS[n].format(lang=lang) for n in tools)
        rec = ACC_RECORDED.format(params=json.dumps({"path": self.base, "mode": "view"}), content=self.files[self.base])
        msgs = [{"role": "system", "content": f"{head}\n\n{ACC_PROTOCOL.format(tools=docs)}"},
                {"role": "user", "content": f"{rec}\n\n" + ACC_OPEN.format(lang=lang, src=self.base, out=self.out, dir=ACC_DIR,
                                                                           budget=ACC_BUDGET)}]
        start = time.monotonic()
        while self.budget > 0 and time.monotonic() - start < ACC_RUN_SECONDS:
            try:
                reply = self.ctx.llm.ask(msgs, role="guiding", temperature=ACC_TEMP, max_tokens=ACC_MAX_TOKENS)
            except BudgetExceeded:
                raise
            except Exception:
                break
            calls = _acc_calls(reply)
            if not calls:
                final = self.files.get(self.out, "")
                return final if final.strip() and (t.language != "python" or _compiles(t, final)[0]) else self.best[1]
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "\n\n".join(f"[{c['tool']}] {self.call(tools, c)}" for c in calls)}]
        return self.best[1]

    def call(self, tools: dict, c: dict) -> str:
        fn, args = tools.get(str(c["tool"])), c.get("args") or {}
        if fn is None:
            return f"Error: unknown tool {c['tool']!r}"
        if not isinstance(args, dict):
            return "Error: args must be a JSON object"
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            return f"Error: {e}"
        return str(fn(**args))

    def file_read(self, path: str) -> str:
        return self.files.get(_acc_path(path), f"Error: file {path} does not exist")

    def file_write(self, path: str, content: str) -> str:
        self.files[_acc_path(path)] = str(content)
        return f"File write success: {path}"

    def frontend(self, source_file: str, out_file: str) -> str:
        return self._compiler(ACC_FRONTEND, source_file, out_file)

    def backend(self, ir_file: str, out_file: str) -> str:
        return self._compiler(ACC_BACKEND, ir_file, out_file)

    def unparse(self, ir_file: str, out_file: str) -> str:
        return self._compiler(ACC_UNPARSE, ir_file, out_file)

    def _compiler(self, script: str, src: str, out: str) -> str:
        code = self.files.get(_acc_path(src))
        if code is None:
            return f"Compilation failed for {src}: no such file"
        err = ""
        if script == ACC_BACKEND and code.lstrip().startswith("Module("):
            code, err = self._py(ACC_UNPARSE, code)
        if not err:
            code, err = self._py(script, code)
        if err:
            return f"Compilation failed for {src}: {err}"
        self.files[_acc_path(out)] = code
        return "success"

    def _py(self, script: str, stdin: str) -> tuple[str, str]:
        r = S.run_program(script, "python", stdin, ACC_SCRIPT_TIMEOUT, self.task.memory_mb)
        return (r.stdout, "") if r.ok else ("", r.stderr or "the compiler did not run")

    def what(self, level: str) -> str:
        return f"{_lang(self.task)} source" if level == "source" else "CPython AST"

    def name(self, level: str, out: str, *nums) -> str:
        return f"{os.path.splitext(out)[0]}_{'_'.join(map(str, nums))}{'.ast' if level == 'ir' else self.ext}"

    def optimize(self, level: str, source_file: str, out_file: str, feedback: str | None = None) -> str:
        """level_workflow: k feedback loops of n samples; the best correct sample replaces the input when it beats the best."""
        if self.budget <= 0:
            return ACC_EXHAUSTED
        self.budget -= 1
        self.counter += 1
        src, out = _acc_path(source_file), _acc_path(out_file)
        fb, cur, best = feedback, src, (-1.0, "")
        for l in range(ACC_LOOPS):
            cur_out = self.name(level, out, self.counter, l)
            results = self.sample(level, cur, cur_out, fb)
            good = {f: r for f, r in results.items() if _acc_match(self.ref, r)}
            if not results:
                fb = ACC_FB_UNKNOWN.format(fb=fb, src=source_file)
            elif not good:
                fb = ACC_FB_WRONG.format(fb=fb, src=source_file, reasons=_acc_report(self.ref, next(iter(results.values()))))
            else:
                speed = {f: _acc_speedup(self.ref, r) for f, r in good.items()}
                f = max(speed, key=speed.get)
                if speed[f] > best[0]:
                    self.files[cur_out] = self.files[f]
                    fb = ACC_FB_BETTER.format(fb=fb, out=cur_out, s=speed[f], base=self.base)
                    cur, best = cur_out, (speed[f], cur_out)
                    self.record(level, self.files[f], speed[f])
                else:
                    fb = ACC_FB_WORSE.format(fb=fb, s=speed[f], base=self.base, best=best[1])
        if not best[1]:
            return ACC_FAILED.format(src=source_file, b=self.budget)
        self.files[out] = self.files[best[1]]
        return ACC_DONE.format(out=out_file, s=best[0], b=self.budget)

    def sample(self, level: str, in_file: str, out_file: str, feedback: str | None) -> dict[str, dict]:
        names = [self.name(level, out_file, i) for i in range(ACC_SAMPLES)]
        code = self.files.get(in_file)
        if code is None:
            return {f: {"compiler": {"status": "failure", "output": f"no such file or directory: '{in_file}'"}} for f in names}
        what, tag = self.what(level), "ast" if level == "ir" else self.task.language
        query = ACC_QUERY.format(what=what, src=in_file, out=out_file,
                                 feedback=ACC_CALLER.format(fb=feedback) if feedback else "")
        msgs = [{"role": "system", "content": ACC_LEVEL.replace("{language}", what)},
                {"role": "user", "content": query + ACC_INLINE.format(src=in_file, tag=tag, code=code, out=out_file)}]
        try:
            outs = self.ctx.llm.sample(msgs, ACC_SAMPLES, role=f"level-{level}", temperature=ACC_TEMP,
                                       max_tokens=ACC_MAX_TOKENS)
        except BudgetExceeded:
            raise
        except Exception:
            return {}
        results = {}
        for f, raw in zip(names, outs):
            self.files[f] = C.extract_code(raw, tag)
            results[f] = self.execute(self.files[f], level)
        return results

    def record(self, level: str, code: str, speedup: float) -> None:
        if speedup > self.best[0]:
            src = self._py(ACC_UNPARSE, code)[0] if level == "ir" else code
            if src.strip():
                self.best = (speedup, self.ctx.keep(src))


def _acc_entry(task: Task, code: str) -> str:
    """The callable a function task is run through: its entry point, or the method of a LeetCode-style Solution class."""
    if task.language != "python" or task.mode != "function":
        return ""
    if task.entry_point:
        return task.entry_point
    cls = next((n for n in _statements(code) if isinstance(n, ast.ClassDef) and n.name == "Solution"), None)
    fn = next((n.name for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")), "")
    return f"Solution().{fn}" if fn else ""


def _acc_path(p) -> str:
    return str(p).strip().removeprefix("./")


def _acc_calls(reply: str) -> list[dict]:
    d = _json(reply)
    calls = [d] if isinstance(d, dict) else d if isinstance(d, list) else []
    return [c for c in calls if isinstance(c, dict) and "tool" in c]


def _acc_hyperfine(cmd: list[str], d: Path, stdin: str, mem: int) -> dict:
    """hyperfine -i semantics: warmups then timed runs under one overall timeout; a non-zero exit still records its output."""
    times, spent, out, status = [], 0.0, "", "ok"
    for i in range(ACC_WARMUP + ACC_RUNS):
        if spent >= ACC_TEST_TIMEOUT:
            return {"status": "timeout"}
        r = S.execute(cmd, d, stdin, ACC_TEST_TIMEOUT - spent, mem, pin=True)
        spent += r.seconds
        if r.timeout:
            return {"status": "timeout"}
        out, status = r.stdout, "ok" if r.ok else "error"
        if i >= ACC_WARMUP:
            times.append(r.seconds)
    return {"status": status, "output": out, "runtime": statistics.fmean(times)}


def _acc_match(ref: dict, res: dict) -> bool:
    """output_matches: same inputs, equal outputs; a timeout on either side counts as a match."""
    return ref.keys() == res.keys() and all(
        "timeout" in (ref[k]["status"], res[k]["status"]) or ref[k].get("output") == res[k].get("output") for k in res)


def _acc_speedup(ref: dict, res: dict) -> float:
    xs = [ref[k]["runtime"] / res[k]["runtime"] if "runtime" in ref[k] and "runtime" in res[k] else 1.0
          for k in ref.keys() & res.keys()]
    return statistics.geometric_mean(xs) if xs else 1.0


def _acc_report(ref: dict, res: dict) -> str:
    """correctness_feedback for one sample."""
    if list(res) == ["compiler"]:
        return f"Solution failed to compile for the following reason: {res['compiler']['output']}"
    out = []
    for k, r in res.items():
        f = ref.get(k)
        if f is None:
            out.append(f"Input {k}: No solution produced")
        elif f["status"] == "timeout" and r["status"] != "timeout":
            out.append(f"Input {k}: Solution suceeded with unkown correctness (reference implementation timed out)")
        elif f["status"] != "timeout" and r["status"] == "timeout":
            out.append(f"Input {k}: Solution timed out, reference implementation finished in time")
        elif f["status"] == "timeout":
            out.append(f"Input {k}: Solution timed out, reference implementation also timed out")
        elif f["output"] == r["output"]:
            out.append(f"Input {k}: Test passed, success")
        else:
            out.append(f'Input {k}: Test failed; expected "{f["output"]}" but solution produced "{r["output"]}"')
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------------------------
# H36 MACGen: planner, security advisor (attack-surface triage, CWE groups, standards KB RAG, code-based quick wins,
# validation), coder and perspective-separated reviewer, passing artifacts only. Prompts from the repository (commit f3339f2).

MG_COMMIT = "f3339f2fcb29e33f4d38300a039a64e8c852ad51"
MG_RAW = f"https://raw.githubusercontent.com/Mishuni/macgen-secure-code/{MG_COMMIT}/data/rag_databases/"
MG_KB = {"asvs": "ASVS_faiss/ASVS.json", "python": "Python_faiss/python_refined.json", "c": "CERT_C_faiss/CERT_C_refined.json",
         "cpp": "CERT_CPP_faiss/CERT_CPP_refined.json", "go": "Go_faiss/Go_refined.json",
         "javascript": "JavaScript_faiss/JavaScript_refined.json",
         "owasp": "OWASP_CheatSheets_faiss/owasp_cheatsheet_docs_refined.json"}
MG_STORES = {"c": ("c", "asvs"), "cpp": ("cpp",), "python": ("python", "asvs"), "javascript": ("javascript", "owasp", "asvs"),
             "go": ("go", "asvs")}
MG_EMBED = "jina"
MG_TEMP, MG_MAX_TOKENS = 0.0, 1500
MG_RAG_K, MG_SNIPPET_CHARS = 2, 7000
MG_REFINE_ATTEMPTS, MG_SYNTAX_FIXES, MG_PLAN_RETRIES, MG_CWE_TRIES = 2, 2, 3, 2
MG_OS = "linux"
MG_NO_COMPILE = ("go", "javascript", "php", "rust", "ruby")
MG_FENCE_TAGS = {"python", "py", "c", "cpp", "c++", "js", "jsx", "javascript", "go", "java", "ruby", "rust", "php", "csharp"}
MG_PLAN_MARKERS = ("### plan", "###plan", "##plan", "## plan", "## new plan", "### functional plan", "### final plan",
                   "functional plan", "final plan")
MG_GUIDE_MARKERS = ("### security guidelines", "### security_guidelines", "### security guideline", "### security_guideline",
                    "security guidelines", "security_guidelines")

MG_HEADER_CWEVAL = ("Your response should include all dependencies, headers and function declaration to be directly usable.\n"
                    "You should NOT call or test the function and should NOT implement a main function in your response."
                    "\nDo NOT include any print statements, logging. Do NOT add authentication unless explicitly required."
                    "\nHere's the code in a single code block (wrap in ```{language})")
MG_HEADER_LLMSECEVAL = "Based on the above analysis, here's the updated version of the code in a single code block (wrap in ```):\n"
MG_PRESETS = {"cweval": (MG_HEADER_CWEVAL, 3), "humaneval_plus": ("", 2), "mbpp_plus": ("", 2),
              "llmseceval": (MG_HEADER_LLMSECEVAL, 3)}

MG_PLAN = """{task}
You are a programmer tasked with generating appropriate plan to solve a given problem using the **{language}** programming language.

**Expected Output:**
Your response must be structured as follows:

### Problem Understanding
- Think about the original problem. Develop an initial understanding about the problem.

### Recall Example Problem
Recall a relevant and distinct problems (different from problem mentioned above) and
- Describe it
- Generate {language} code step by step to solve that problem
- Discuss the algorithm to solve this problem
 within 5 bullet points

### Algorithm to solve the original problem
- Write down the algorithm that is well suited for the original problem
- Give some tutorials to about the algorithm for example:
    - How to approach this type of algorithm
    - Important things to consider

### Plan
- Write down a step-by-step plan to solve the original problem with bullet points.
- Do not consider testing, deployment, document and future maintenance in your plan.

--------
**Important Instruction:**
- Strictly follow the instructions.
- Do not generate code.
"""

MG_DRAFT = """{task_description}
{action_prompt_header}

 **Important Instructions:**
- Do not add any explanation.
- Write secure code.
- You MUST strictly follow the function signature specified in the problem."""

MG_SURFACE = """
You are a security engineer. You will be given a problem statement and draft code.

Goal:
- Skip simple code with no external interaction.
- Only flag code that clearly exposes an attack surface.

Decision rules (IMPORTANT):
- Judge based on the **Draft Code only**.
- If the code is simple logic, or uses only constants → NO_ATTACK_SURFACE.
- If there is an attack surface BUT it appears already safely handled → NO_ATTACK_SURFACE.
- Only flag when there is an attack surface AND it is not clearly handled safely.

Categories (ONLY IF the code includes one of):
- Untrusted Input (user, request, file, env)
- Network/API (HTTP, sockets)
- Database/Storage (SQL, queries)
- File System (open, write, path join)
- Exec/Deserialization/rendering (subprocess, eval, pickle, yaml)
- Auth/Session/Secrets (hard-coded keys, debug flags, session, logging)
- Memory/Resources (buffers, unsafe ops, regax, DoS)
- Crypto (hash, verify flags, random generators)

Output format:
### Attack Surface
- [Category] Evidence: ... — Candidate: ...

### Overall
HAS_ATTACK_SURFACE | NO_ATTACK_SURFACE
---

## Problem:
{problem}

## Draft Code:
{draft_code}
"""

MG_SEC_ENGINEER = ("You are a security engineer."
                   "Your task is to identify the most likely CWE vulnerabilities that the following function might introduce.")

MG_CWE = """Use the provided context and guidelines to output a list of group of probable cause-effect CWE(s) pair(s). For each, provide a concise explanation, a likelihood score in [0, 1].
Reasoning levels:
1. **Cause (fine-grained CWE(s))**: the direct insecure practice or action(s).
2. **Effect (broad CWE(s))**: all higher-level security consequence(s).

## Context
- Language: {language}
- Problem: {problem_with_planning}

### Guidelines
1. Read the problem and plan carefully.
2. Define rules for identifying vulnerabilities groups.
3. For each vulnerability:
   - Map to cause CWE(s).
   - Map to effect CWEs.
   - Group similar categories.
4. Rank and return AT MOST {max_limit} groups. (Do NOT pad to {max_limit})

Output format:
* plan understanding: a short summary of the plan
* extract keywords: APIs, inputs(trust level), credential info, auth/cryptography(for what), memory/logic (include only if present, few lines).
* JSON array of objects. Each object must contain: (in wrap ```json)
  - `group_name` : string, <group_name>
  - `why`: short rationale (1-2 sentences)
  - `cause_cwes`: ["CWE-###", ...],
  - `effect_cwes`: ["CWE-###", ...],
  - `likelihood`: float ∈ [0, 1]
"""

MG_GUIDE_SYSTEM = ("You are a security agent."
                   "Using the retrieved secure-coding practices, write concise, actionable "
                   "security guidelines tailored to THIS problem.")
MG_GUIDE = """Retrieved practices (ranked, deduplicated):
{retrieved_snippets}

---
Context:
- Language/Os: {language}, {os_platform}
- Problem (with plan): {problem_with_planning}
{input}

{prompt_4_sec_tail}"""
MG_GUIDE_TAIL = """Output format:
### security guidelines
- Provide 1-3 clear guidelines that leave no room for error. (if needed, with at most one minimal code line as an example).
- Keep it precise; no fluff, no generic advice.
- Preserving functionality.
- Do NOT include (logging, audit trail) unless the problem explicitly asks for it.
"""
MG_FINDINGS = """
Findings (CWE groups):
{cwe_list}"""

MG_QUICK_WINS = """Your task is to:
- Review the draft code and identify up to **2 small, lightweight security analysis and one-line advices** that are *not already covered*.
- Focus on **direct, code-evident quick wins** (single-line changes or flags).
- If there are no new lightweight guidelines, return exactly: "No additional guideline needed."

Evidence-first rule:
- Prefer findings that are **directly visible in the draft code** (e.g., hard-coded secrets, debug/dev flags, unsafe API options).
- Deprioritize infrastructure-only guidance (e.g., “use HTTPS”) unless the code explicitly sets/assumes insecure behavior.

Priorities (apply in order; only include items that actually appear in code):
P0 (must-check quick wins):
  - Hard-coded credentials/secrets/keys (e.g., API keys in literals) → advise using environment variables or secret manager.
  - Development/test toggles in production path (e.g., `debug=True`) → advise disabling (`debug=False`) for production.
P1:
  - API call flags that need safe defaults
P2:
  - Other one-line configuration hardening directly observable in the code. (unsafe API usage, missing safe flags, etc.)

Constraints:
- Suggestions must be **minimal and actionable with a single line** of advice or configuration.
- Do NOT suggest heavy-weight practices (logging/auditing/compliance frameworks/large refactors).
- Do NOT include broad infra advice unless **explicitly evidenced in code**.

Output format (exactly):
- Advice 1: <one-line advice>
- Advice 2: <one-line advice>

## Code (draft):
{draft_code}"""

MG_VALIDATE = """You are a security agent. Your task is to carefully review and refine the provided Security Guidelines for the given problem.

Steps:
1. Think about the original problem and functional requirements which must be preserved.
2. Check if any security guideline conflicts with the functional constraints (e.g., removing an argument when signature must be preserved).
   - If conflict exists, modify or drop the guideline so that functional constraints are preserved.
   - If there are over-engineering which destroy functionality, modify or drop them.
3. If duplicate or overlapping guidelines exist, merge them into a single, coherent detailed guideline.

## Language/Os: {language}, {os_platform}
## Problem:
{question}

## Security Guidelines (draft):
{security_guidelines}

Output format:
### Functional Requirements
- Summarize must have to keep from the problem.
### Conflicting Identified
- List any security guidelines that conflict with functional constraints, and how you modified them to preserve Functional Requirements.
### Security Guidelines
- Provide improved guidelines for code engineers.
- Keep it ultra concise; include only essentials; no filler.
- Do NOT include (logging, audit trail) unless the problem explicitly asks for it."""

MG_CODE = """Your task is to write secure and functional code for the following problem in {language}.
### Problem Statement:
{task_description}

{implementation_plan}
### Security Guidelines:
{security_requirements}

Requirements:
- You MUST strictly follow the function signature specified in the problem. (e.g. do not change function name, its arguments and return type)
- Keep it simple. Avoid unnecessary complexity or over-engineering.

### Output format:
{action_prompt_header}"""

MG_SYNTAX_SYSTEM = ("You are given a programming task, a code snippet containing syntax errors, and compiler feedback. Your task "
                    "is to correct the code strictly according to the compiler feedback. Respond only with the corrected code, "
                    "nothing else.")
MG_SYNTAX = """## Task : {question}
## The code is
```{language}
{code}
```
## The feedback from compiler is
{compiler_feedback}

Your solution should be in the form of a code snippet with prefix code.
Wrap the code snippet in (wrap in ```{language}). """

MG_FUNC = """### Problem Statement:
{task_description}
{action_header}

### Current Code:
```{language}
{current_code}
```
---
You are a software engineer. You are given a problem statement and a implementation called the current code.
Goal: Decide whether the Current Code correctly implements the REQUIRED core functionalities in the Problem Statement.

Output format:
### Problem Understanding
Summarize the core task and goals of the problem.
Keywords and definitions that are critical to the problem and test case inputs. (what is it, specific things to consider)

### Example Scenarios
Describe a concrete example user-scenario that illustrates the problem requirements and expected behavior.
Describe how to handle these scenarios on the code.

### Current Code Analysis
For each core function or block:
Step-by-step explanation of logic
Highlight any missing logic or unclear behavior (e.g., return type, format, false positive/negative on edge cases etc.)
Your opinion on whether it fulfills the requirements correctly from the the scenario

### Final Opinion
- Provide your overall assessment of the current code’s functional completeness.
- If you find no issue, output ONLY the 'THE CODE IS FUNCTIONAL'.
"""

MG_SEC_REVIEW = """You are a security engineer. You are given a problem statement and an implementation called the **current code**.
Goal: Decide whether the code is safe related to the security guidelines.
Minimize false positives:
- Do NOT modify the provided function signature (including prefix/header, parameter names, order, or types).
- If you cannot provide a minimal concrete reproducer (input/test-case) for an issue, do NOT report it as a concern.
---
### Problem Statement:
{task_description}
{action_header}

### Current Code:
```{language}
{current_code}
```

### Security guidelines:
{security_requirements}
- other security issues when the code touches predefined risk categories (input, network, storage, FS, exec, auth, memory, crypto).

Output format:
### Current Code Analysis
Step-by-step explanation of logic and security mechanisms
### Final Opinion
- Provide your overall assessment of the current code’s security concerns.
- If you find no issue, output ONLY the ‘THE CODE IS SAFE’.
"""

MG_FIX = """You are a software engineer. Given the Problem and Feedback, produce the SMALLEST possible code change as ONE line (prefer 1 line; max 3 lines).
First, think step-by-step about the root cause and the minimal fix.

Constraints:
- Strictly follow the provided docstring and function signature; do not alter names, order, types, headers, or imports.
- No refactors, no scaffolding, no placeholders.
- If the Feedback asks to change the prefix/header, function signature, parameter list/order/types, return type, DROP and ignore those parts and proceed with an in-scope fix only.

Inputs:
[Problem]
{problem}

[Feedback]
{feedback}

Output (STRICT):
### Conflicts Identified
- Enumerate any Feedback items that (a) conflict with the Problem/docstring/signature (e.g., prefix/header/signature/import changes) or (b) would break Functional Requirements.
- State how you modified or dropped them to preserve functionality and constraints.
### Fix Approaches
- For **each identified issue**:
    - IdentifiedIssue: <max 2 sentences describing what the feedback flags (after dropping out-of-scope items)>
    - FixApproach: <max 2 sentences explaining how we will fix the issues>
    - CodeExampleLines({language}):
        <line 1>
        <line 2> (optional)
        <line 3> (optional)

Rules:
- Maximum 3 lines.
- Functionality is the top priority; do not break it.
- CodeExampleLines must be the final section and compile/parse correctly in {language}.
"""

MG_JUDGE_SYSTEM = """You will be provided with an analysis result of a code snippet.
If the analysis confirms that the code snippet correctly implements the **core functionalities** as required by the problem statement, output: "Yes". Otherwise, output: "No".
"""
MG_JUDGE_USER = 'Analysis Result:"""\n{MISTAKES}\n"""\n'
MG_JUDGE_PREFILL = "Final Answer (Yes or No):\n"

MG_REFINE = """You are a S/W engineer. Improve the code into a functional and secure implementation.

Problem:
{problem_with_planning}

Original Code:
```{language}
{code}
```

Feedback from Evaluation:
{feedback}

-----
Requirements:
1. Review the feedback carefully. Use it as a reference—not a rule—and adopt only what aligns with the problem logic and intended behavior.
2. Make the SMALLEST possible change: touch only the exact lines the feedback targets. Do NOT rewrite, restructure, or replace logic that already works.
3. Do NOT change function name, params, return type, prefix/header, or imports. Drop feedback that asks for these.
4. Ensure the final code is both functionally correct.
5. Keep it simple. Avoid unnecessary complexity or over-engineering.
6. If the feedback suggests replacing a working implementation with a completely different approach, ignore that suggestion.

{action_prompt_header}"""


@h2h
class MACGen(H2HTechnique):
    """Phase 1: plan, draft, attack-surface triage (early exit with the draft), CWE groups (C per preset), one guideline call
    over the top-2 KB practices per group query, code-based quick wins, validation, final code and syntax repair (2 tries).
    Phase 2: functional and security reviews (each with a Yes/No judge and a fix suggestion) and up to M=2 refinements.
    Temperature 0, 1500 tokens for planner and security agents; per-agent session histories as in the repository. The authors'
    released refined KB is re-embedded locally (jina-embeddings-v3 instead of OpenAI text-embedding-3-small); presets
    (action header, C) follow the repository configs for CWEval, HumanEval and LLMSecEval (all other benchmarks)."""
    hid, name, family, target = "H36", "MACGen", "multiagent", ("SEC",)
    code_url = "https://github.com/Mishuni/macgen-secure-code"

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        for name, rel in MG_KB.items():
            if (d / f"{name}.npy").exists():
                continue
            items = json.loads(fetch(MG_RAW + rel, d / "kb" / Path(rel).name).read_text())
            rows = [{"text": x, "meta": {}} if isinstance(x, str) else
                    {"text": x.get("text") or x.get("content") or "", "meta": x.get("metadata", {})} for x in items]
            rows = decontaminate([r for r in rows if r["text"]], "text")
            save_index(d, rows, embed(MG_EMBED, [r["text"] for r in rows]), name)

    def solve(self, task: Task, ctx: Ctx) -> str:
        lang, q = task.language, C.task_text(task)
        header, limit = MG_PRESETS.get(task.benchmark, MG_PRESETS["llmseceval"])
        header = header.replace("{language}", lang)
        pre = "\n".join(header.split("\n")[:-1])
        plan = ""
        for _ in range(1 + MG_PLAN_RETRIES):
            plan = ctx.llm.ask(MG_PLAN.format(task=q + pre, language=lang), role="planner", temperature=MG_TEMP,
                               max_tokens=MG_MAX_TOKENS).strip()
            if plan:
                break
        plan = _mg_plan(plan)
        code_hist, sec_hist = [], []
        draft_code = ctx.keep(_mg_code(_mg_turn(ctx, code_hist, MG_DRAFT.format(task_description=f"{q}\n\n{plan}\n",
                                                                                 action_prompt_header=header), "coder-draft"), lang))
        surface = _mg_turn(ctx, sec_hist, MG_SURFACE.format(problem=q, draft_code=draft_code), "advisor-surface",
                           max_tokens=MG_MAX_TOKENS).lower()
        if "no attack surface" in surface or "no_attack_surface" in surface:
            return self._syntax(task, ctx, draft_code, q) if draft_code else draft_code
        req = self._advise(task, ctx, sec_hist, q, pre, plan, draft_code, limit)
        prompt = (MG_CODE.format(language=lang, task_description=q, implementation_plan=plan, security_requirements=req,
                                 action_prompt_header=header) if req.strip() else
                  MG_DRAFT.format(task_description=q, action_prompt_header=header))
        code = _mg_code(_mg_turn(ctx, code_hist, prompt, "coder"), lang)
        code = self._syntax(task, ctx, ctx.keep(code), q) if code else draft_code
        return self._review(task, ctx, q, header, code, req) if code else code

    def _advise(self, task: Task, ctx: Ctx, hist: list, q: str, pre: str, plan: str, draft_code: str, limit: int) -> str:
        """Security Advisor after triage: CWE groups, standards-driven guideline, code-based quick wins, validation."""
        lang, groups = task.language, []
        for _ in range(MG_CWE_TRIES):
            msgs = [{"role": "user", "content": MG_SEC_ENGINEER},
                    {"role": "user", "content": MG_CWE.format(language=lang, problem_with_planning=f"{q}\n\n{plan}\n",
                                                              max_limit=limit)}]
            groups = _mg_groups(ctx.llm.ask(msgs, role="advisor-cwe", temperature=MG_TEMP, max_tokens=MG_MAX_TOKENS))
            if groups:
                break
        context = dict(language=lang, os_platform=MG_OS, problem_with_planning=f"{q}\n{pre}\n{plan}" if plan else q)
        guides = []
        if groups:
            queries = [" ".join([str(g.get("group_name") or "").strip(), str(g.get("why") or "").strip()]).strip()
                       for g in groups]
            findings = MG_FINDINGS.format(cwe_list=", ".join(queries))
            out = _mg_turn(ctx, hist, MG_GUIDE.format(retrieved_snippets=_mg_snippets(self._retrieve(ctx, lang, queries)),
                                                      input=findings, prompt_4_sec_tail=MG_GUIDE_TAIL, **context),
                           "advisor-guide", record=findings, system=MG_GUIDE_SYSTEM, max_tokens=MG_MAX_TOKENS)
            guides.append((str(groups[0].get("group_name") or "group_0"), _mg_collapse(_mg_after(out))))
        if draft_code:
            quick = MG_QUICK_WINS.format(draft_code=draft_code)
            out = _mg_turn(ctx, hist, MG_GUIDE.format(retrieved_snippets="", input=quick, prompt_4_sec_tail="", **context),
                           "advisor-code", record=quick, system=MG_GUIDE_SYSTEM, max_tokens=MG_MAX_TOKENS).strip()
            if "no additional guideline needed" not in out.lower():
                guides.append(("others", _mg_collapse(out)))
        summarized = "\n".join(g for _, g in guides)
        out = ctx.llm.ask(MG_VALIDATE.format(language=lang, os_platform=MG_OS, question=f"{q}\n{pre}",
                                             security_guidelines=summarized),
                          role="advisor-validate", temperature=MG_TEMP, max_tokens=MG_MAX_TOKENS).strip()
        if not out:
            return "\n".join(f"{n}\n{g}" for n, g in guides if n != "others")
        return _mg_collapse(_mg_after(out)).strip() or summarized

    def _retrieve(self, ctx: Ctx, lang: str, queries: list[str]) -> list[dict]:
        """Top-K practices per query over the language's merged stores (the general ASVS store when none is dedicated)."""
        parts = [load_index(ctx.dir(self.hid), s) for s in MG_STORES.get(lang, ("asvs",))]
        rows, vecs = [r for rs, _ in parts for r in rs], np.concatenate([v for _, v in parts])
        return [rows[i] for v in embed(MG_EMBED, queries) for i in top_k(vecs, v, MG_RAG_K)]

    def _review(self, task: Task, ctx: Ctx, q: str, header: str, code: str, req: str) -> str:
        """Phase 2: the coder's session is reseeded with the code-generation turn (the plan reads None, as in the repository)."""
        lang, seed_q = task.language, f"{q}\n\nNone".strip()
        seed = (MG_CODE.format(language=lang, task_description=seed_q, implementation_plan="", security_requirements=req,
                               action_prompt_header=header) if req.strip() else
                MG_DRAFT.format(task_description=seed_q, action_prompt_header=header))
        hist = [{"role": "user", "content": seed}, {"role": "assistant", "content": code}]
        req = _mg_collapse(re.sub(r"<CODE>.*?</CODE>", "", re.sub(r"```.*?```", "", req, flags=re.S), flags=re.S | re.I)).strip()
        current = code
        for _ in range(MG_REFINE_ATTEMPTS):
            func_ok, func_fb = self._functional(task, ctx, q, header, current)
            sec_ok, sec_fb = self._secure(task, ctx, q, header, current, req)
            if func_ok and sec_ok:
                break
            feedback = (f"Functionality: {'Passed' if func_ok else 'Failed'}, Security: {'Passed' if sec_ok else 'Failed'}"
                        + ("" if func_ok else f"\n\nFunctionality Feedback:\n{func_fb}")
                        + ("" if sec_ok else f"\n\nSecurity Feedback:\n{sec_fb}"))
            out = _mg_turn(ctx, hist, MG_REFINE.format(problem_with_planning=q, language=lang, code=current, feedback=feedback,
                                                       action_prompt_header=header), "coder-refine")
            current = ctx.keep(_mg_code(out, lang) or current)
        return self._syntax(task, ctx, current, q) if current.strip() != code.strip() else current

    @staticmethod
    def _functional(task: Task, ctx: Ctx, q: str, header: str, code: str) -> tuple[bool, str]:
        hist: list = []
        under = _mg_turn(ctx, hist, MG_FUNC.format(task_description=q, action_header=header, language=task.language,
                                                   current_code=code), "reviewer-func")
        if "the code is functional" in under.lower():
            passed = True
        else:
            ans = ctx.llm.ask(_mg_judge(under), role="reviewer-judge", temperature=MG_TEMP).strip().lower()
            passed = "yes" in ans and "no" not in ans
        analysis = re.sub(r"\n{2,}", "\n", _mg_section(under).strip())
        if not passed:
            fix = _mg_turn(ctx, hist, MG_FIX.format(problem=q, feedback=analysis, language=task.language), "reviewer-func-fix")
            analysis = _mg_collapse(fix).strip()
            passed = "no actionable fix" in analysis.lower()
        return passed, analysis

    @staticmethod
    def _secure(task: Task, ctx: Ctx, q: str, header: str, code: str, req: str) -> tuple[bool, str]:
        hist: list = []
        under = _mg_turn(ctx, hist, MG_SEC_REVIEW.format(task_description=q, action_header=header, language=task.language,
                                                         current_code=code, security_requirements=req),
                         "reviewer-sec", max_tokens=MG_MAX_TOKENS)
        if "the code is safe" in under.lower():
            return True, under
        if "yes" in ctx.llm.ask(_mg_judge(under), role="reviewer-judge", temperature=MG_TEMP).lower():
            return True, under
        fix = _mg_turn(ctx, hist, MG_FIX.format(problem=q, feedback=_mg_section(under), language=task.language),
                       "reviewer-sec-fix", max_tokens=MG_MAX_TOKENS)
        m = re.search(r"(### Fix Approaches.*)", fix, re.S)
        total = _mg_collapse(m.group(1) if m else fix).strip()
        return "no actionable fix" in total.lower(), total

    @staticmethod
    def _syntax(task: Task, ctx: Ctx, code: str, q: str) -> str:
        """compilation_check_and_repair: at most two syntax-fix calls with the compiler feedback."""
        if task.language in MG_NO_COMPILE:
            return code
        ok, log = _mg_compiles(task, code)
        for _ in range(MG_SYNTAX_FIXES):
            if ok:
                break
            msgs = [{"role": "system", "content": MG_SYNTAX_SYSTEM},
                    {"role": "user", "content": MG_SYNTAX.format(question=q, language=task.language, code=code,
                                                                 compiler_feedback=("\n" + log)[:1000])}]
            code = ctx.keep(_mg_code(ctx.llm.ask(msgs, role="coder-syntax", temperature=MG_TEMP), task.language))
            ok, log = _mg_compiles(task, code)
        return code


def _mg_turn(ctx: Ctx, hist: list, content: str, role: str, record: str | None = None, system: str | None = None,
             max_tokens: int | None = None) -> str:
    """One call of an agent session: its history plus the new message; the session stores `record` (the chain input)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + hist + [{"role": "user", "content": content}]
    out = ctx.llm.ask(msgs, role=role, temperature=MG_TEMP, max_tokens=max_tokens)
    hist += [{"role": "user", "content": content if record is None else record}, {"role": "assistant", "content": out}]
    return out


def _mg_judge(analysis: str) -> list[dict]:
    return [{"role": "system", "content": MG_JUDGE_SYSTEM}, {"role": "user", "content": MG_JUDGE_USER.format(MISTAKES=analysis)},
            {"role": "assistant", "content": MG_JUDGE_PREFILL}]


def _mg_code(text: str, lang: str) -> str:
    """The repository's extract_code: the first fenced block, without its language tag."""
    m = re.search(r"```(.*?)```", text or "", re.S)
    code = (m.group(1) if m else text or "").strip()
    head, _, rest = code.partition("\n")
    return rest if head.strip().lower() in MG_FENCE_TAGS | {lang} else code


def _mg_compiles(task: Task, code: str) -> tuple[bool, str]:
    if task.language != "python":
        return _compiles(task, code)
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"[SyntaxError] {e.msg} (Line {e.lineno}, Column {e.offset}): {(e.text or '').strip()}"


def _mg_plan(plan: str) -> str:
    """_normalize_plan_text: lower-cased, from the last plan marker (or the last 1200 characters), blank lines dropped."""
    txt = plan.strip().lower()
    if not txt:
        return ""
    at = max(txt.rfind(m) for m in MG_PLAN_MARKERS)
    txt = txt[at:] if at != -1 else f"### Plan\n\n{txt[-1200:]}"
    return "\n".join(l for l in txt.splitlines() if l.strip())


def _mg_groups(raw: str) -> list[dict]:
    """CWE groups from the first fenced block, sorted by likelihood; [] when it does not parse to a list of objects."""
    m = re.search(r"```(.*?)```", raw or "", re.S)
    block = (m.group(1) if m else raw or "").strip()
    if block.lower().startswith("json\n"):
        block = block[4:]
    for a, b in ((']"', ']'), ('"[', '['), ("]'", ']'), ("'[", '['), ("cwe-", "CWE-"), ("CWE- ", "CWE-")):
        block = block.replace(a, b)
    try:
        data = json.loads(block, strict=False)
    except json.JSONDecodeError:
        return []
    for k in ("Security Requirements", "security_requirements"):
        if isinstance(data, dict) and k in data:
            data = data[k]
    if not isinstance(data, list) or not all(isinstance(g, dict) for g in data):
        return []
    return sorted(data, key=lambda g: _num(g.get("likelihood", 0)), reverse=True)


def _mg_snippets(docs: list[dict]) -> str:
    """_format_docs_for_prompt: deduplicated practices of at least 50 characters, stopping once past 7000 characters."""
    chunks, seen = [], set()
    for d in docs:
        txt = d["text"].strip()
        key = f"{d['meta']}||{txt[:100]}"
        if len(txt) < 50 or key in seen:
            continue
        seen.add(key)
        chunks.append(txt)
        if sum(len(c) for c in chunks) > MG_SNIPPET_CHARS:
            break
    return "\n\n".join(f"- {c}" for c in chunks) if chunks else "- (no high-quality practices found)"


def _mg_after(text: str) -> str:
    """extract_content_after_marker: the text after the first guideline marker found, else its last 700 characters."""
    low = text.lower()
    for m in MG_GUIDE_MARKERS:
        i = low.find(m)
        if i != -1:
            return text[i + len(m):].strip()
    return text[-700:].strip()


def _mg_section(text: str) -> str:
    m = re.search(r"(?:^|\n)###\s*Current Code Analysis.*?(.*)", text, re.S)
    return m.group(0).strip() if m else text


def _mg_collapse(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text)
