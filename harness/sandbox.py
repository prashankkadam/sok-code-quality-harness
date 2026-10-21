"""Isolated execution. Runs inside the pinned harness container (docker/Dockerfile); on Linux it pins cores with taskset."""
from __future__ import annotations

import os
import platform
import resource
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task, Test

PIN = os.environ.get("HARNESS_PIN_CORES", "2")
LINUX = platform.system() == "Linux"

LANGS = {
    "python": {"file": "main.py", "run": ["python3", "main.py"]},
    "cpp": {"file": "main.cpp", "build": ["g++", "-O2", "-std=gnu++20", "-o", "main", "main.cpp"], "run": ["./main"]},
    "java": {"file": "Main.java", "build": ["javac", "Main.java"], "run": ["java", "-Xss256m", "Main"]},
    "javascript": {"file": "main.js", "run": ["node", "main.js"]},
    "ruby": {"file": "main.rb", "run": ["ruby", "main.rb"]},
    "go": {"file": "main.go", "build": ["go", "build", "-o", "main", "main.go"], "run": ["./main"]},
}


@dataclass
class Run:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    seconds: float = 0.0
    timeout: bool = False


def _limits(mem_mb: int):
    def f():
        os.setsid()
        if LINUX:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20, 64 * 2**20))
    return f


def _prefix(pin: bool) -> list[str]:
    return ["taskset", "-c", PIN] if pin and LINUX and shutil.which("taskset") else []


def execute(cmd: list[str], cwd: Path, stdin: str = "", timeout: float = 10, mem_mb: int = 1024, pin: bool = False) -> Run:
    import time
    t0 = time.perf_counter()
    try:
        p = subprocess.run(_prefix(pin) + cmd, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limits(mem_mb), env={"PATH": os.environ["PATH"], "PYTHONHASHSEED": "0", "HOME": str(cwd)})
        return Run(p.returncode == 0, p.stdout, p.stderr[-4000:], p.returncode, time.perf_counter() - t0)
    except subprocess.TimeoutExpired as e:
        return Run(False, (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                   "timeout", -9, timeout, True)


class Workdir:
    def __init__(self, files: dict[str, str]):
        self.files = files

    def __enter__(self) -> Path:
        self.d = Path(tempfile.mkdtemp(prefix="hx_"))
        for name, text in self.files.items():
            (self.d / name).parent.mkdir(parents=True, exist_ok=True)
            (self.d / name).write_text(text)
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(self.d, ignore_errors=True)


def build(lang: str, d: Path) -> Run:
    spec = LANGS[lang]
    return execute(spec["build"], d, timeout=60, mem_mb=4096) if "build" in spec else Run(True)


def run_program(code: str, lang: str, stdin: str = "", timeout: float = 10, mem_mb: int = 1024, pin: bool = False) -> Run:
    spec = LANGS[lang]
    with Workdir({spec["file"]: code}) as d:
        b = build(lang, d)
        if not b.ok:
            return Run(False, "", b.stderr, b.returncode)
        return execute(spec["run"], d, stdin, timeout, mem_mb, pin)


def norm_out(s: str) -> list[str]:
    return [ln.rstrip() for ln in s.strip().splitlines()]


def outputs_match(got: str, want: str) -> bool:
    a, b = norm_out(got), norm_out(want)
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        xs, ys = x.split(), y.split()
        if len(xs) != len(ys):
            return False
        for u, v in zip(xs, ys):
            if u == v:
                continue
            try:
                if abs(float(u) - float(v)) > 1e-6 * max(1, abs(float(v))):
                    return False
            except ValueError:
                return False
    return True


def run_test(task: Task, code: str, t: Test, pin: bool = False) -> Run:
    if task.language == "python" and task.meta.get("prelude"):
        code = task.meta["prelude"] + "\n" + code
    if t.kind == "io":
        r = run_program(code, task.language, t.stdin, task.time_limit, task.memory_mb, pin)
        return Run(r.ok and outputs_match(r.stdout, str(t.expected)), r.stdout, r.stderr, r.returncode, r.seconds, r.timeout)
    if t.kind == "assert":
        return run_program(code + "\n\n" + t.code + "\n", "python", "", task.time_limit, task.memory_mb, pin)
    if t.kind == "call":
        call = f"import json\n_r = {task.entry_point}(*json.loads({t.code!r}))\nassert _r == json.loads({json_dumps(t.expected)!r}), _r\n"
        return run_program(code + "\n\n" + call, "python", "", task.time_limit, task.memory_mb, pin)
    raise ValueError(t.kind)


def json_dumps(x) -> str:
    import json
    return json.dumps(x)


def passes(task: Task, code: str, tests: list[Test], stop_on_fail: bool = True) -> tuple[int, int]:
    n = 0
    for t in tests:
        if run_test(task, code, t).ok:
            n += 1
        elif stop_on_fail:
            return n, len(tests)
    return n, len(tests)


def passes_public(task: Task, code: str) -> bool:
    return not task.public_tests or passes(task, code, task.public_tests)[0] == len(task.public_tests)


def passes_hidden(task: Task, code: str) -> bool:
    tests = task.hidden_tests or task.public_tests
    return bool(tests) and passes(task, code, tests)[0] == len(tests)


def instructions(task: Task, code: str, t: Test) -> int | None:
    """User-space CPU instruction count of one test via perf (Linux only), the EvalPerf measure."""
    if not (LINUX and shutil.which("perf")):
        return None
    spec = LANGS[task.language]
    body = code + "\n\n" + t.code + "\n" if t.kind == "assert" else code
    with Workdir({spec["file"]: body}) as d:
        if not build(task.language, d).ok:
            return None
        r = execute(["perf", "stat", "-x,", "-e", "instructions:u", *spec["run"]], d, t.stdin, task.time_limit * 3,
                    task.memory_mb, pin=True)
        for line in r.stderr.splitlines():
            parts = line.split(",")
            if len(parts) > 2 and parts[2].startswith("instructions") and parts[0].isdigit():
                return int(parts[0])
    return None


def wallclock(task: Task, code: str, t: Test, repeats: int = 5) -> tuple[float, float] | None:
    """Median and IQR of wall time over `repeats` pinned runs; None if any run fails."""
    xs = []
    for _ in range(repeats):
        r = run_test(task, code, t, pin=True)
        if not r.ok:
            return None
        xs.append(r.seconds)
    q = statistics.quantiles(xs, n=4)
    return statistics.median(xs), q[2] - q[0]
