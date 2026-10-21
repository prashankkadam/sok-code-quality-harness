"""SonarQube: ingests the Pylint and Bandit reports and runs SonarPython; maintainability and security issues are counted per output file."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
import yaml

from ..config import CONFIGS
from .static import AN

CFG = yaml.safe_load((CONFIGS / "analyzers" / "sonar.yaml").read_text())


def _client() -> httpx.Client:
    return httpx.Client(base_url=os.environ.get("SONAR_URL", CFG["url"]), auth=(os.environ[CFG["token_env"]], ""), timeout=60)


def _reports(src: Path) -> tuple[Path, Path]:
    pl, bd = src / "pylint.txt", src / "bandit.json"
    files = sorted(str(p.relative_to(src)) for p in src.rglob("*.py"))
    with open(pl, "w") as f:
        subprocess.run([sys.executable, "-m", "pylint", f"--rcfile={AN / 'pylintrc'}", "--exit-zero",
                        "--msg-template={path}:{line}: [{msg_id}({symbol}), {obj}] {msg}", *files], cwd=src, stdout=f, text=True)
    with open(bd, "w") as f:
        subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", "-c", str(AN / "bandit.yaml"), "-r", "."],
                       cwd=src, stdout=f, text=True)
    return pl, bd


def analyze_dir(src: Path, project: str | None = None) -> dict[str, dict]:
    """Returns {relative file: {maintainability, security, ncloc, cognitive_complexity}} for every .py file under src."""
    key = project or f"hx-{uuid.uuid4().hex[:12]}"
    _reports(src)
    url = os.environ.get("SONAR_URL", CFG["url"])
    net = ["--network", "host"] if "localhost" in url else []
    subprocess.run(["docker", "run", "--rm", *net, "-v", f"{src.resolve()}:/usr/src", "-e", f"SONAR_HOST_URL={url}",
                    "-e", f"SONAR_TOKEN={os.environ[CFG['token_env']]}", CFG["image"],
                    f"-Dsonar.projectKey={key}", "-Dsonar.sources=.", "-Dsonar.python.version=3.12",
                    "-Dsonar.python.pylint.reportPaths=pylint.txt", "-Dsonar.python.bandit.reportPaths=bandit.json",
                    "-Dsonar.scm.disabled=true", "-Dsonar.qualitygate.wait=false"], check=True, capture_output=True)
    c = _client()
    _wait(c, key)
    out: dict[str, dict] = defaultdict(lambda: {"maintainability": 0, "security": 0, "ncloc": 0, "cognitive_complexity": 0})
    for quality in ("MAINTAINABILITY", "SECURITY"):
        page = 1
        while True:
            r = c.get("/api/issues/search", params={"components": key, "impactSoftwareQualities": quality, "ps": 500,
                                                     "p": page, "resolved": "false"}).json()
            if r["paging"]["total"] > 10000:
                raise RuntimeError(f"{key}: {r['paging']['total']} {quality} issues exceed the 10,000-result search limit")
            for it in r["issues"]:
                out[it["component"].split(":", 1)[1]][quality.lower()] += 1
            if page * 500 >= r["paging"]["total"]:
                break
            page += 1
    page = 1
    while True:
        r = c.get("/api/measures/component_tree", params={"component": key, "metricKeys": "ncloc,cognitive_complexity",
                                                           "qualifiers": "FIL", "ps": 500, "p": page}).json()
        for comp in r["components"]:
            for m in comp["measures"]:
                if "value" in m:
                    out[comp["path"]][m["metric"]] = float(m["value"])
        if page * 500 >= r["paging"]["total"]:
            break
        page += 1
    c.post("/api/projects/delete", data={"project": key})
    return dict(out)


def _wait(c: httpx.Client, key: str, timeout: float = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get("/api/ce/component", params={"component": key})
        if r.status_code == 200:
            d = r.json()
            if not d.get("queue") and d.get("current", {}).get("status") in {"SUCCESS", "FAILED", "CANCELED"}:
                if d["current"]["status"] != "SUCCESS":
                    raise RuntimeError(json.dumps(d["current"]))
                return
        time.sleep(3)
    raise TimeoutError(key)


def analyze_outputs(codes: dict[str, str]) -> dict[str, dict]:
    """codes: {output id: python source}. Writes one file per output and maps results back to ids."""
    import tempfile
    src = Path(tempfile.mkdtemp(prefix="sonar_"))
    names = {}
    for i, (oid, code) in enumerate(codes.items()):
        name = f"o{i:06d}.py"
        (src / name).write_text(code)
        names[name] = oid
    res = analyze_dir(src)
    return {oid: res.get(name, {"maintainability": 0, "security": 0, "ncloc": 0, "cognitive_complexity": 0})
            for name, oid in names.items()}
