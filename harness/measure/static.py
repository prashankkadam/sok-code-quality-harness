"""Layer 1: pinned Pylint, Bandit and Safety (the techniques' analyzers), SonarQube issue counts and cognitive complexity."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import CONFIGS

AN = CONFIGS / "analyzers"


@dataclass
class Finding:
    tool: str
    rule: str
    severity: str
    message: str
    line: int = 0
    security: bool = False

    def text(self) -> str:
        return f"[{self.tool} {self.rule}, {self.severity}] line {self.line}: {self.message}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    pylint_score: float | None = None

    @property
    def security(self) -> list[Finding]:
        return [f for f in self.findings if f.security]

    @property
    def maintainability(self) -> list[Finding]:
        return [f for f in self.findings if not f.security]

    def weighted(self) -> float:
        w = _weights()
        s = 0.0
        for f in self.findings:
            s += w["safety"] if f.tool == "safety" else w[f.tool].get(f.severity, 0.0)
        return s

    def top(self, k: int = 3) -> list[Finding]:
        """Up to k findings, security first, then by weight."""
        w = _weights()
        key = lambda f: (not f.security, -(w["safety"] if f.tool == "safety" else w[f.tool].get(f.severity, 0)), f.line)
        return sorted(self.findings, key=key)[:k]


@lru_cache
def _weights() -> dict:
    import yaml
    return yaml.safe_load((AN / "weights.yaml").read_text())


def _write(code: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="an_"))
    (d / "solution.py").write_text(code)
    return d / "solution.py"


def pylint(path: Path) -> tuple[list[Finding], float | None]:
    p = subprocess.run([sys.executable, "-m", "pylint", f"--rcfile={AN / 'pylintrc'}", "--output-format=json2",
                        "--score=y", str(path)], capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return [], None
    fs = [Finding("pylint", m["symbol"], m["type"], m["message"], m.get("line", 0)) for m in d.get("messages", [])]
    return fs, d.get("statistics", {}).get("score")


def bandit(path: Path) -> list[Finding]:
    p = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", "-c", str(AN / "bandit.yaml"), str(path)],
                       capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []
    return [Finding("bandit", r["test_id"], r["issue_severity"], r["issue_text"], r.get("line_number", 0), True)
            for r in d.get("results", [])]


STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def third_party_imports(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    return {m for m in mods if m not in STDLIB}


@lru_cache
def _env_versions() -> dict[str, str]:
    req = AN / "env_requirements.txt"
    if not req.exists():
        return {}
    out = {}
    for ln in req.read_text().splitlines():
        if "==" in ln:
            k, v = ln.split("==")
            out[k.strip().lower()] = v.strip()
    return out


IMPORT_TO_DIST = {"yaml": "pyyaml", "cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "pillow", "bs4": "beautifulsoup4",
                  "Crypto": "pycryptodome", "jwt": "pyjwt", "dateutil": "python-dateutil", "magic": "python-magic"}


def safety(code: str) -> list[Finding]:
    """Known-vulnerable dependencies among the imports, resolved against the pinned execution environment."""
    vers = _env_versions()
    pins = [f"{d}=={vers[d]}" for d in (IMPORT_TO_DIST.get(m, m).lower() for m in third_party_imports(code)) if d in vers]
    if not pins:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(pins))
    db = ["--db", os.environ["SAFETY_DB"]] if os.environ.get("SAFETY_DB") else []
    p = subprocess.run([sys.executable, "-m", "safety", "check", "--output", "json", *db, "-r", f.name],
                       capture_output=True, text=True)
    try:
        d, _ = json.JSONDecoder().raw_decode(p.stdout, p.stdout.index("{"))
    except ValueError as e:
        raise RuntimeError(f"safety produced no report: {p.stderr[-500:]}") from e
    return [Finding("safety", v.get("vulnerability_id", ""), "HIGH", f"{v.get('package_name')} {v.get('analyzed_version')}: "
                    f"{(v.get('advisory') or '')[:200]}", 0, True) for v in d.get("vulnerabilities", [])]


def analyze(code: str, tools: tuple[str, ...] = ("pylint", "bandit", "safety")) -> Report:
    path = _write(code)
    r = Report()
    if "pylint" in tools:
        fs, r.pylint_score = pylint(path)
        r.findings += fs
    if "bandit" in tools:
        r.findings += bandit(path)
    if "safety" in tools:
        r.findings += safety(code)
    return r


def cognitive_complexity(code: str) -> float | None:
    """Mean SonarSource cognitive complexity per function."""
    from cognitive_complexity.api import get_cognitive_complexity
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not fns:
        return float(get_cognitive_complexity(ast.parse("def _m():\n" + "\n".join("    " + l for l in code.splitlines() or ["pass"])).body[0]))
    return sum(get_cognitive_complexity(f) for f in fns) / len(fns)


def loc(code: str) -> int:
    return sum(1 for l in code.splitlines() if l.strip() and not l.strip().startswith("#"))
