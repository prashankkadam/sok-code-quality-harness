"""Repository- and agent-level benchmarks, and the agent runner: the mini-SWE-agent loop in the task's container, or Harbor
(Terminal-Bench's own harness) for Harbor-native suites. Every model call goes through LLM.chat."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import DATA, RUNS
from ..llm import LLM, Ledger, current_scope, scope
from ..measure.static import AN, Finding, Report, analyze, safety
from ..tasks import Output, Task
from .base import Benchmark, register

try:
    import tomllib
except ImportError:
    import tomli as tomllib

DOCKER = os.environ.get("MSWEA_DOCKER_EXECUTABLE", "docker")
HARBOR = [shutil.which("harbor")] if shutil.which("harbor") else ["uvx", "harbor"]
WORKERS = int(os.environ.get("HARNESS_EVAL_WORKERS", "4"))
STEP_LIMIT, COST_LIMIT, COMMAND_TIMEOUT = 250, 3.0, 60
ACTION_REGEX = r"```mswea_bash_command\s*\n(.*?)\n```"
VISIBLE_DEFAULT = {"python": ["python -m pytest -q -p no:cacheprovider"], "go": ["go test ./..."], "rust": ["cargo test"],
                   "javascript": ["npm test --silent"], "typescript": ["npm test --silent"]}

DEEPSWE_LAYOUT = """$HARNESS_DATA/deepswe/tasks/<task_id>/   the public v1.1 release (github.com/datacurve-ai/deep-swe), Harbor task format
  task.toml       [environment] docker_image (pinned image), cpus, memory_mb; [agent] timeout_sec;
                  [verifier] timeout_sec, env, optional environment.docker_image; [metadata] language, optional workdir
                  and visible_tests (commands running the repository's own tests)
  instruction.md  task statement
  tests/test.sh   the authors' verifier entry point; writes /logs/verifier/reward.json or reward.txt
  solution/       reference solution, never read"""

FRONTIERCODE_LAYOUT = """$HARNESS_DATA/frontiercode/   obtained from the authors (tasks are not public)
  tasks/<task_id>/task.toml       [environment] docker_image; [agent] timeout_sec; [metadata] subset = diamond | main | extended
  tasks/<task_id>/instruction.md  task description plus codebase guidelines
  tasks/<task_id>/rubric.json     [{"id", "weight", "blocker", "prompt"} | {"id", "weight", "blocker", "command", "expect_fail"}]
  tasks/<task_id>/tests/          held-out files the command items use, copied to /tests
  grader/prompt.txt               reference grader prompt with {instruction}, {diff}, {criterion}
  grader/config.json              {"model": <configs/models.yaml key>, "max_tokens": int, "verdict_regex": str}"""


@dataclass
class AgentRun:
    patch: str = ""
    steps: int = 0
    exit_status: str = ""
    messages: list[dict] = field(default_factory=list, repr=False)
    rewards: dict = field(default_factory=dict)

    @property
    def reward(self) -> float | None:
        return self.rewards.get("reward")

    def summary(self) -> dict:
        """Text-free record of the run, for Output.meta["agent"]."""
        return {"exit_status": self.exit_status, "steps": self.steps, "rewards": self.rewards}


class _Box:
    """A container started from an image; commands run with docker exec, inputs go through stdin."""

    def __init__(self, image: str, run_args: list[str] | None = None, cid: str | None = None):
        self.image, self.run_args, self.id, self.owned = image, run_args or [], cid, cid is None

    def __enter__(self) -> "_Box":
        if self.owned:
            p = subprocess.run([DOCKER, "run", "-d", "--rm", "--entrypoint", "", *self.run_args, self.image, "sleep", "86400"],
                               capture_output=True, text=True, timeout=3600)
            if p.returncode:
                raise RuntimeError(f"container start failed for {self.image}")
            self.id = p.stdout.strip()
        return self

    def __exit__(self, *a):
        if self.owned and self.id:
            subprocess.run([DOCKER, "rm", "-f", self.id], capture_output=True)

    def run(self, cmd: str, cwd: str | None = None, input: str | None = None, timeout: float | None = 600,
            env: dict | None = None) -> subprocess.CompletedProcess:
        args = [DOCKER, "exec", "-i", *(["-w", cwd] if cwd else [])]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        try:
            return subprocess.run([*args, self.id, "bash", "-c", cmd], input=input, capture_output=True, text=True,
                                  timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, -9, "", "timeout")

    def apply(self, patch: str, cwd: str) -> bool:
        if not patch.strip():
            return True
        return any(self.run(c, cwd, input=patch).returncode == 0
                   for c in ("git apply --whitespace=nowarn -", "patch --batch --fuzz=5 -p1"))

    def put(self, src: Path, dst: str) -> None:
        self.run(f"mkdir -p {dst}")
        subprocess.run([DOCKER, "cp", f"{src}/.", f"{self.id}:{dst}"], capture_output=True)


@lru_cache
def _image_workdir(image: str) -> str:
    p = subprocess.run([DOCKER, "image", "inspect", "-f", "{{.Config.WorkingDir}}", image], capture_output=True, text=True)
    return p.stdout.strip() or "/"


def _cwd(task: Task) -> str:
    return task.meta.get("workdir") or _image_workdir(task.meta["image"])


def _resources(meta: dict) -> list[str]:
    return [*(["--cpus", str(meta["cpus"])] if meta.get("cpus") else []),
            *(["--memory", f"{meta['memory_mb']}m"] if meta.get("memory_mb") else [])]


def _box(task: Task) -> _Box:
    return _Box(task.meta["image"], _resources(task.meta))


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""


def _augment(text: str, instructions: str = "", context: str = "", feedback: str = "", revising: bool = False) -> str:
    parts = [text, instructions]
    if context:
        parts.append(f"Reference material:\n{context}")
    if feedback:
        lead = ("Your previous changes are already applied in the repository. Revise them to address this feedback:"
                if revising else "Feedback on a previous attempt at this task:")
        parts.append(f"{lead}\n{feedback}")
    return "\n\n".join(p for p in parts if p)


def run_agent(task: Task, llm: LLM, *, instructions: str = "", context: str = "", feedback: str = "", seed: int = 0,
              prior_patch: str = "") -> AgentRun:
    """One complete agent run, charged as one unit of the current scope's budget; its model calls are logged, not charged.

    instructions: prompt additions (T1, T2); context: retrieved text (T6); feedback: findings, critique or review (T3-T5);
    prior_patch: start from an earlier run's patch so the run revises it (the returned patch is then cumulative)."""
    sc = current_scope()
    sc.budget.charge(1)
    run_seed = sc.seed * 1000 + seed
    with scope(sc.task_id or task.id, sc.technique, None, sc.seed):
        if task.meta.get("harness") == "harbor":
            return _harbor_run(task, llm, instructions, context, feedback, run_seed)
        return _mini_run(task, llm, _augment(task.prompt, instructions, context, feedback, bool(prior_patch)), run_seed,
                         prior_patch)


class _HarnessModel:
    """mini-swe-agent text-based model whose queries go through LLM.chat."""

    def __init__(self, llm: LLM, seed: int, cfg: dict):
        self.llm, self.seed, self.config = llm, seed, cfg

    def query(self, messages: list[dict], **kw) -> dict:
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_text import parse_regex_actions
        msgs = [{"role": m["role"], "content": _text(m.get("content"))} for m in messages
                if m.get("role") in ("system", "user", "assistant")]
        c = self.llm.chat(msgs, role="agent", seed=self.seed)[0]
        cost = (c.input_tokens * self.llm.price["input"] + c.output_tokens * self.llm.price["output"]) / 1e6
        finish = "length" if c.finish_reason in ("length", "max_tokens") else c.finish_reason
        try:
            actions = parse_regex_actions(c.text, action_regex=ACTION_REGEX,
                                          format_error_template=self.config["format_error_template"],
                                          template_kwargs={"finish_reason": finish})
        except FormatError as e:
            e.messages[0]["extra"]["cost"] = cost
            raise
        return {"role": "assistant", "content": c.text, "extra": {"actions": actions, "cost": cost, "timestamp": time.time()}}

    def format_message(self, **kw) -> dict:
        return dict(kw)

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        from minisweagent.models.utils.actions_text import format_observation_messages
        return format_observation_messages(outputs, observation_template=self.config["observation_template"],
                                           template_vars=template_vars)

    def get_template_vars(self, **kw) -> dict:
        return dict(self.config)

    def serialize(self) -> dict:
        return {}


@lru_cache
def _mini_config() -> dict:
    import logging

    import yaml
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    from minisweagent import package_dir
    logging.getLogger("minisweagent").setLevel(logging.WARNING)
    return yaml.safe_load((Path(package_dir) / "config" / "mini_textbased.yaml").read_text())


def _baseline(box: _Box, cwd: str) -> str:
    cmd = ("git config --global --add safe.directory '*' >/dev/null 2>&1; "
           "if ! git rev-parse --git-dir >/dev/null 2>&1; then git init -q && git add -A && "
           "git -c user.name=harness -c user.email=harness@localhost commit -qm base >/dev/null 2>&1; fi; "
           "git add -A >/dev/null 2>&1; git write-tree")
    return box.run(cmd, cwd).stdout.strip()


def _diff(box: _Box, cwd: str, base: str) -> str:
    return box.run(f"git add -A >/dev/null 2>&1; git diff --cached --binary {base}", cwd).stdout


def _mini_run(task: Task, llm: LLM, text: str, seed: int, prior_patch: str) -> AgentRun:
    cfg = _mini_config()
    from minisweagent.agents.default import AgentConfig, DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    cwd = _cwd(task)
    env = DockerEnvironment(image=task.meta["image"], cwd=cwd, env=cfg.get("environment", {}).get("env", {}),
                            timeout=COMMAND_TIMEOUT, executable=DOCKER, container_timeout="24h", pull_timeout=3600,
                            run_args=["--rm", "--entrypoint", "", *_resources(task.meta)])
    box = _Box(task.meta["image"], cid=env.container_id)
    try:
        base = _baseline(box, cwd)
        if prior_patch and not box.apply(prior_patch, cwd):
            return AgentRun(prior_patch, 0, "PriorPatchFailed")
        uname = box.run("uname -s; uname -r; uname -v; uname -m").stdout.splitlines()
        kw = {k: v for k, v in cfg["agent"].items() if k in AgentConfig.model_fields}
        kw |= {"step_limit": STEP_LIMIT, "cost_limit": COST_LIMIT, "output_path": None,
               "wall_time_limit_seconds": int(task.meta.get("agent_timeout") or 0)}
        agent = DefaultAgent(_HarnessModel(llm, seed, cfg["model"]), env, **kw)
        try:
            status = agent.run(text, **dict(zip(("system", "release", "version", "machine"), uname))).get("exit_status", "")
        except Exception as e:
            status = type(e).__name__
        return AgentRun(_diff(box, cwd, base), agent.n_calls, status, agent.messages)
    finally:
        env.cleanup()


class _Proxy:
    """Local OpenAI-compatible endpoint for Harbor's agent; every request is an LLM.chat call in the caller's scope."""

    def __init__(self, llm: LLM, seed: int):
        sc, proxy = current_scope(), self
        self.calls = 0

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                msgs = [{"role": m["role"] if m.get("role") in ("system", "user", "assistant") else "user",
                         "content": _text(m.get("content"))} for m in body.get("messages", [])]
                stop = body.get("stop")
                try:
                    with scope(sc.task_id, sc.technique, None, sc.seed):
                        c = llm.chat(msgs, role="agent", seed=seed, stop=[stop] if isinstance(stop, str) else stop,
                                     max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"))[0]
                    proxy.calls += 1
                except Exception as e:
                    return self._send(500, {"error": {"message": type(e).__name__, "type": "server_error"}})
                finish = {"max_tokens": "length", "end_turn": "stop", "stop_sequence": "stop"}.get(c.finish_reason,
                                                                                                    c.finish_reason or "stop")
                usage = {"prompt_tokens": c.input_tokens, "completion_tokens": c.output_tokens,
                         "total_tokens": c.input_tokens + c.output_tokens}
                head = {"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time()), "model": body.get("model", llm.key)}
                if body.get("stream"):
                    chunk = head | {"object": "chat.completion.chunk", "usage": usage, "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": c.text}, "finish_reason": finish}]}
                    return self._send(200, f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n", "text/event-stream")
                self._send(200, head | {"object": "chat.completion", "usage": usage, "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": c.text}, "finish_reason": finish}]})

            def _send(self, code: int, payload, ctype: str = "application/json"):
                data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "_Proxy":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def _trial(job: Path) -> dict:
    for p in sorted(job.rglob("result.json")):
        r = json.loads(p.read_text())
        if "trial_name" in r:
            return r
    return {}


def _harbor_run(task: Task, llm: LLM, instructions: str, context: str, feedback: str, seed: int) -> AgentRun:
    src, job = Path(task.meta["dir"]), RUNS / "harbor" / task.benchmark / uuid.uuid4().hex[:12]
    job.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hb_") as tmp, _Proxy(llm, seed) as px:
        d = Path(tmp) / src.name
        shutil.copytree(src, d)
        for f in d.rglob("instruction.md"):
            f.write_text(_augment(f.read_text(), instructions, context, feedback))
        cmd = [*HARBOR, "run", "-p", str(d), "-a", task.meta.get("agent", "terminus-2"), "-m", f"openai/{llm.key}",
               "--ak", f"api_base={px.url}", "-o", str(job), "--job-name", "run", "-n", "1", "-k", "1"]
        p = subprocess.run(cmd, env=os.environ | {"OPENAI_API_KEY": "harness-local-proxy"}, capture_output=True, text=True)
        (job / "harbor.log").write_text(p.stdout + p.stderr)
        steps = px.calls
    r = _trial(job)
    rewards = dict((r.get("verifier_result") or {}).get("rewards") or {})
    checkpoints = [dict((s.get("verifier_result") or {}).get("rewards") or {}) for s in r.get("step_results") or []]
    if checkpoints:
        rewards["checkpoints"] = checkpoints
    status = (r.get("exception_info") or {}).get("exception_type") or ("completed" if r else f"harbor_exit_{p.returncode}")
    return AgentRun("", steps, status, [], rewards)


def _count(out: str, rc: int) -> tuple[int, int]:
    ok = sum(map(int, re.findall(r"(\d+) passed", out))) + len(re.findall(r"^ok\s+\S", out, re.M))
    bad = sum(map(int, re.findall(r"(\d+) (?:failed|errors?)\b", out))) + len(re.findall(r"^FAIL\t", out, re.M))
    if ok + bad == 0:
        return (1, 1) if rc == 0 else (0, 1)
    return ok, ok + bad


def visible_tests(task: Task, patch: str) -> tuple[int, int]:
    """Tests passed and run among the repository's visible tests with the patch applied; hidden tests are never run here."""
    cmds = task.meta.get("visible_tests") or []
    if task.meta.get("harness") != "mini" or not cmds:
        return 0, 0
    with _box(task) as box:
        cwd = _cwd(task)
        if not box.apply(patch, cwd):
            return 0, len(cmds)
        passed = total = 0
        for c in cmds:
            r = box.run(c, cwd, timeout=task.meta.get("test_timeout", 1800))
            p, t = _count(r.stdout + r.stderr, r.returncode)
            passed, total = passed + p, total + t
    return passed, total


def _added_lines(patch: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    path, line = None, 0
    for ln in patch.splitlines():
        if ln.startswith("+++ "):
            path = None if ln[4:].strip() == "/dev/null" else re.sub(r"^b/", "", ln[4:].strip())
            if path:
                out.setdefault(path, set())
        elif m := re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", ln):
            line = int(m.group(1))
        elif path and ln.startswith("+"):
            out[path].add(line)
            line += 1
        elif path and ln.startswith(" "):
            line += 1
    return out


def patched_files(task: Task, patch: str, suffix: str = ".py") -> dict[str, str]:
    """Post-patch contents of the changed files ending in suffix (kept in memory)."""
    paths = [p for p in _added_lines(patch) if p.endswith(suffix)]
    if not paths or task.meta.get("harness") != "mini":
        return {}
    with _box(task) as box:
        cwd = _cwd(task)
        if not box.apply(patch, cwd):
            return {}
        return {p: r.stdout for p in paths if (r := box.run(f"cat -- '{p}'", cwd)).returncode == 0}


def _analyze_stdin(path: str, code: str) -> Report:
    """Pylint, Bandit and Safety with the pinned configs, fed through stdin so nothing is written to disk."""
    r = Report()
    p = subprocess.run([sys.executable, "-m", "pylint", f"--rcfile={AN / 'pylintrc'}", "--output-format=json2", "--score=y",
                        "--from-stdin", path], input=code, capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
        r.findings += [Finding("pylint", m["symbol"], m["type"], m["message"], m.get("line", 0)) for m in d.get("messages", [])]
        r.pylint_score = d.get("statistics", {}).get("score")
    except json.JSONDecodeError:
        pass
    b = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", "-c", str(AN / "bandit.yaml"), "-"], input=code,
                       capture_output=True, text=True)
    try:
        r.findings += [Finding("bandit", x["test_id"], x["issue_severity"], x["issue_text"], x.get("line_number", 0), True)
                       for x in json.loads(b.stdout).get("results", [])]
    except json.JSONDecodeError:
        pass
    r.findings += safety(code)
    return r


def findings(task: Task, patch: str) -> Report:
    """Analyzer findings on the lines the patch adds (plus dependency findings), each message prefixed with its file."""
    added, out, scores = _added_lines(patch), Report(), []
    for path, code in patched_files(task, patch).items():
        r = _analyze_stdin(path, code) if task.meta.get("private") else analyze(code)
        for f in r.findings:
            if f.line in added.get(path, ()) or f.tool == "safety":
                f.message = f"{path}: {f.message}"
                out.findings.append(f)
        if r.pylint_score is not None:
            scores.append(r.pylint_score)
    out.pylint_score = statistics.mean(scores) if scores else None
    return out


def select(task: Task, runs: list[AgentRun]) -> AgentRun:
    """T7 over complete runs: keep runs passing every visible test (all, if none does), then fewest weighted findings."""
    cands = [r for r in runs if r.patch.strip()]
    if not cands:
        return runs[0]
    scored = [(i, r, visible_tests(task, r.patch), findings(task, r.patch).weighted()) for i, r in enumerate(cands)]
    pool = [s for s in scored if s[2][1] and s[2][0] == s[2][1]] or scored
    return min(pool, key=lambda s: (s[3], s[0]))[1]


def _toml(p: Path) -> dict:
    return tomllib.loads(p.read_text()) if p.exists() else {}


def _harbor_task(d: Path, bench: str, harness: str, **meta) -> Task:
    cfg = _toml(d / "task.toml")
    env, agent, ver, md = (cfg.get(k, {}) for k in ("environment", "agent", "verifier", "metadata"))
    lang = md.get("language", "python").lower()
    visible = md.get("visible_tests", VISIBLE_DEFAULT.get(lang, []))
    instr = d / "instruction.md"
    return Task(d.name, bench, instr.read_text() if instr.exists() else "", language=lang, mode="repo",
                meta={"harness": harness, "dir": str(d), "image": env.get("docker_image") or md.get("image"),
                      "cpus": env.get("cpus"), "memory_mb": env.get("memory_mb"), "agent_timeout": agent.get("timeout_sec"),
                      "verifier_timeout": ver.get("timeout_sec", 600), "verifier_env": ver.get("env", {}),
                      "verifier_image": (ver.get("environment") or {}).get("docker_image"), "workdir": md.get("workdir"),
                      "visible_tests": [visible] if isinstance(visible, str) else list(visible), "subset": md.get("subset"),
                      **meta})


def _harbor_tasks(root: Path, bench: str, harness: str, **meta) -> list[Task]:
    base = root / "tasks" if (root / "tasks").is_dir() else root
    return [_harbor_task(d, bench, harness, **meta) for d in sorted(base.iterdir()) if (d / "task.toml").exists()]


def _parse_reward(text: str) -> dict:
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(v, (int, float)):
        return {"reward": float(v)}
    return {k: x for k, x in v.items() if isinstance(x, (int, float))} if isinstance(v, dict) else {}


def _verify(task: Task, patch: str) -> dict:
    """The task's own verifier (tests/test.sh) in a clean pinned container with the patch applied through stdin."""
    m = task.meta
    with _Box(m.get("verifier_image") or m["image"], _resources(m)) as box:
        cwd = _cwd(task)
        if not box.apply(patch, cwd):
            return {"reward": 0.0}
        box.run("mkdir -p /logs/verifier")
        box.put(Path(m["dir"]) / "tests", "/tests")
        box.run("bash /tests/test.sh", cwd, timeout=m["verifier_timeout"], env=m["verifier_env"])
        out = box.run("cat /logs/verifier/reward.json 2>/dev/null || cat /logs/verifier/reward.txt").stdout
    r = _parse_reward(out.strip())
    if "reward" not in r and len(r) == 1:
        r["reward"] = next(iter(r.values()))
    return r


def _agent_rewards(out: Output) -> dict | None:
    return (out.meta.get("agent") or out.meta).get("rewards")


def _pct(rows: list[dict], key: str) -> float:
    return 100 * statistics.mean(bool(r.get(key)) for r in rows) if rows else float("nan")


def _key(o: Output) -> str:
    return f"{o.task_id}|{o.model}|{o.technique}|{o.seed}"


@register
class DeepSWE(Benchmark):
    """DeepSWE v1.1 (Huang et al. 2026, Datacurve), public tasks run in the benchmark's pinned containers with its
    official verifiers; see DEEPSWE_LAYOUT. Patches and scores are stored like any public benchmark."""
    name, dims, metrics, mode, language, private = "deepswe", ("COR",), {"pass@1": True}, "repo", "multi", False

    def load(self) -> list[Task]:
        ts = _harbor_tasks(DATA / "deepswe", self.name, "mini", private=False)
        if missing := [t.id for t in ts if not t.meta["image"]]:
            raise ValueError(f"deepswe tasks without a pinned image: {missing}")
        return ts

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        def one(o: Output) -> dict:
            try:
                return {"task_id": o.task_id, "passed": _verify(tasks[o.task_id], o.code).get("reward", 0.0) >= 1.0}
            except Exception:
                return {"task_id": o.task_id, "passed": False, "error": True}
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(one, outs))

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"pass@1": _pct(rows, "passed")}


@register
class TerminalBench(Benchmark):
    """Terminal-Bench 4.0 (Merrill et al. 2026) on its own harness (Harbor, Terminus 2 agent): task directories of
    terminal-bench/terminal-bench@4.0.0 under $HARNESS_DATA/terminal_bench/; the verifier runs inside each agent run."""
    name, dims, metrics, mode, language = "terminal_bench", ("COR",), {"resolved": True}, "repo", "multi"

    def load(self) -> list[Task]:
        return _harbor_tasks(DATA / "terminal_bench", self.name, "harbor", agent="terminus-2")

    def score(self, task: Task, out: Output) -> dict:
        r = _agent_rewards(out)
        return {"task_id": out.task_id, "resolved": bool(r) and (r.get("reward") or 0) >= 1.0, "missing": r is None}

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"resolved": _pct(rows, "resolved")}


@register
class SlopCodeBench(Benchmark):
    """SlopCodeBench (Orlanski et al. 2026), Harbor multi-step conversion (gabeorlanski/slopcodebench, one step per
    checkpoint) under $HARNESS_DATA/slopcodebench/. Erosion is scb-check's per-checkpoint score; the run-level value is
    the mean over all checkpoints (lower is better)."""
    name, dims, metrics, mode = "slopcodebench", ("RM", "COR"), {"erosion": False, "checkpoint_pass": True}, "repo"

    def load(self) -> list[Task]:
        return _harbor_tasks(DATA / "slopcodebench", self.name, "harbor", agent="terminus-2")

    def score(self, task: Task, out: Output) -> dict:
        r = _agent_rewards(out) or {}
        cps = r.get("checkpoints") or [r]
        return {"task_id": out.task_id, "erosion": [c["erosion"] for c in cps if "erosion" in c],
                "verbosity": [c["verbosity"] for c in cps if "verbosity" in c],
                "passed": [(c.get("reward") or 0) >= 1.0 for c in cps if "reward" in c], "missing": not r}

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        def pool(k: str) -> list:
            return [x for r in rows for x in r.get(k, [])]

        def mean(xs: list) -> float:
            return statistics.mean(xs) if xs else float("nan")

        return {"erosion": mean(pool("erosion")), "verbosity": mean(pool("verbosity")),
                "checkpoint_pass": 100 * mean([float(x) for x in pool("passed")])}


@register
class FrontierCode(Benchmark):
    """FrontierCode v1.1 (Cognition 2026), maintainability with its reference rubric grader; see FRONTIERCODE_LAYOUT.
    A submission scores the weighted share of rubric items it passes, or 0 if any blocker fails."""
    name, dims, metrics = "frontiercode", ("RM", "COR"), {"score": True, "correct": True}
    mode, language, private = "repo", "multi", True
    SUBSETS = {"diamond": {"diamond"}, "main": {"diamond", "main"}, "extended": {"diamond", "main", "extended"}}

    def __init__(self, limit: int | None = None, after: str | None = None, before: str | None = None, subset: str = "main",
                 grader: LLM | None = None):
        super().__init__(limit, after, before)
        self.subset, self._grader = subset, grader
        self.root = DATA / "frontiercode"

    def load(self) -> list[Task]:
        return [t for t in _harbor_tasks(self.root, self.name, "mini", private=True)
                if t.meta.get("subset") in self.SUBSETS[self.subset]]

    @property
    def grader_cfg(self) -> dict:
        return json.loads((self.root / "grader" / "config.json").read_text())

    @property
    def grader(self) -> LLM:
        if self._grader is None:
            self._grader = LLM(self.grader_cfg["model"], ledger=Ledger(RUNS / "frontiercode" / "grader_calls.jsonl"))
        return self._grader

    def _judge(self, task: Task, patch: str, item: dict) -> bool:
        cfg = self.grader_cfg
        prompt = (self.root / "grader" / "prompt.txt").read_text()
        for k, v in (("instruction", task.prompt), ("diff", patch), ("criterion", item["prompt"])):
            prompt = prompt.replace("{" + k + "}", v)
        with scope(task.id, "frontiercode-grader", None):
            txt = self.grader.ask(prompt, role="grader", temperature=0.0, seed=0, max_tokens=cfg.get("max_tokens"))
        m = re.search(cfg.get("verdict_regex", r"(?i)verdict\s*:\s*(pass|fail)"), txt)
        return bool(m) and m.group(1).lower() in ("pass", "yes", "true", "1")

    def _commands(self, task: Task, patch: str, items: list[dict]) -> dict[str, bool]:
        if not items:
            return {}
        with _box(task) as box:
            cwd = _cwd(task)
            if not box.apply(patch, cwd):
                return {i["id"]: False for i in items}
            if (tests := Path(task.meta["dir"]) / "tests").is_dir():
                box.put(tests, "/tests")
            return {i["id"]: (box.run(i["command"], cwd, timeout=i.get("timeout", 1800)).returncode == 0)
                    != bool(i.get("expect_fail")) for i in items}

    def score(self, task: Task, out: Output) -> dict:
        items = json.loads((Path(task.meta["dir"]) / "rubric.json").read_text())
        ok = self._commands(task, out.code, [i for i in items if "command" in i]) if out.code.strip() else \
            {i["id"]: False for i in items}
        ok |= {i["id"]: bool(out.code.strip()) and self._judge(task, out.code, i) for i in items if "prompt" in i}
        correct = all(ok[i["id"]] for i in items if i.get("blocker"))
        w = sum(i.get("weight", 1.0) for i in items) or 1.0
        score = 100 * sum(i.get("weight", 1.0) for i in items if ok[i["id"]]) / w if correct else 0.0
        return {"task_id": out.task_id, "key": _key(out), "model": out.model, "technique": out.technique, "seed": out.seed,
                "score": score, "correct": correct}

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(lambda o: self.score(tasks[o.task_id], o), outs))

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"score": statistics.mean(r["score"] for r in rows) if rows else float("nan"), "correct": _pct(rows, "correct")}


def rescore_sample(rows: list[dict], frac: float = 1 / 3, seed: int = 0) -> list[dict]:
    """A stratified share of submissions (per model and technique) for the rescoring check."""
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        strata.setdefault((r["model"], r["technique"]), []).append(r)
    rng = random.Random(seed)
    return [r for k in sorted(strata) for r in rng.sample(strata[k], math.ceil(len(strata[k]) * frac))]


def rescore_check(rows: list[dict], our_judge_scores: dict[str, float], frac: float = 1 / 3, seed: int = 0) -> float:
    """Spearman rho between official FrontierCode scores and our length-controlled judge (keyed by row["key"])."""
    from scipy.stats import spearmanr
    pairs = [(r["score"], our_judge_scores[r["key"]]) for r in rescore_sample(rows, frac, seed)
             if our_judge_scores.get(r["key"]) is not None]
    if len(pairs) < 3:
        return float("nan")
    return float(spearmanr(*zip(*pairs)).statistic)


class _Official(Benchmark):
    """Benchmarks scored by their authors' prediction-file harness."""
    mode, language = "repo", "python"

    def _rundir(self, label: str) -> Path:
        d = RUNS / "official" / self.name / f"{label}-{uuid.uuid4().hex[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _cells(outs: list[Output]) -> list[tuple[str, list[Output]]]:
        groups: dict[tuple, list[list[Output]]] = {}
        for o in outs:
            chunks = groups.setdefault((o.model, o.technique, o.seed), [])
            free = next((c for c in chunks if all(x.task_id != o.task_id for x in c)), None)
            if free is None:
                chunks.append([o])
            else:
                free.append(o)
        return [(re.sub(r"[^\w.-]", "_", f"{m}.{t}.s{s}.{i}"), c) for (m, t, s), cs in groups.items() for i, c in enumerate(cs)]

    @staticmethod
    def _predictions(d: Path, label: str, outs: list[Output]) -> Path:
        p = d / "predictions.jsonl"
        p.write_text("".join(json.dumps({"instance_id": o.task_id, "model_name_or_path": label, "model_patch": o.code}) + "\n"
                             for o in outs))
        return p

    @staticmethod
    def _call(cmd: list[str], cwd: Path, log: Path) -> int:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n")
        return p.returncode

    def score_batch(self, tasks: dict[str, Task], outs: list[Output]) -> list[dict]:
        rows: dict[int, dict] = {}
        for label, group in self._cells(outs):
            d = self._rundir(label)
            found = self._evaluate(d, label, group, self._predictions(d, label, group))
            for o in group:
                rows[id(o)] = {"task_id": o.task_id, **found.get(o.task_id, {"missing": True})}
        return [rows[id(o)] for o in outs]

    def _evaluate(self, d: Path, label: str, outs: list[Output], preds: Path) -> dict[str, dict]:
        raise NotImplementedError


GSO_PROMPT = """<uploaded_files>
/testbed
</uploaded_files>
I've uploaded a python code repository in the directory /testbed. Consider the following performance test script, which
exercises {api}:

<test_script>
{prob_script}
</test_script>

Can you help me implement the necessary changes to the repository so that the runtime of the test script is improved?
Make general optimizations to the repository's source code rather than to the script; do not modify the test script or
any tests, and keep the repository's behavior unchanged so that its existing tests still pass."""


@register
class GSO(_Official):
    """GSO (Shetty et al. 2025): optimization tasks from gso-bench/gso, scored with the official opt_at_k.py from a checkout
    of github.com/gso-bench/gso at $HARNESS_DATA/gso/gso. Opt@1: correct and at least 95% of the expert commit's speedup."""
    name, dims, metrics = "gso", ("EFF", "COR"), {"opt@1": True, "passed": True}
    repo = DATA / "gso" / "gso"

    def load(self) -> list[Task]:
        from datasets import load_dataset
        hub = os.environ.get("GSO_DOCKERHUB_ID")
        ts = []
        for r in load_dataset("gso-bench/gso", split="test"):
            key = f"gso.eval.{r['arch']}.{r['instance_id'].lower()}"
            prompt = GSO_PROMPT.replace("{api}", r["api"] or "the repository").replace("{prob_script}", r["prob_script"])
            ts.append(Task(r["instance_id"], self.name, prompt, mode="repo", date=str(r["created_at"])[:10],
                           meta={"harness": "mini", "image": f"{hub}:{key}" if hub else f"{key}:{r['instance_image_tag'] or 'latest'}",
                                 "workdir": "/testbed", "repo": r["repo"], "base_commit": r["base_commit"], "visible_tests": []}))
        return ts

    def _evaluate(self, d: Path, label: str, outs: list[Output], preds: Path) -> dict[str, dict]:
        self._call(["uv", "run", "src/gso/harness/opt_at_k.py", "--prediction_paths", str(preds), "--run_id", d.name,
                    "--k", "1", "--model_name", label, "--timeout", "3600", "--max_workers", str(WORKERS),
                    "--instance_ids", *[o.task_id for o in outs]], self.repo, d / "harness.log")
        rep = self.repo / "reports" / f"{label}.opt@1.{d.name}.report.json"
        if not rep.exists():
            return {}
        sets = json.loads(rep.read_text())["instance_sets"]
        return {o.task_id: {"passed": o.task_id in sets["passed_ids"], "opt": o.task_id in sets["opt_commit_ids"]} for o in outs}

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"opt@1": _pct(rows, "opt"), "passed": _pct(rows, "passed")}


SWEF_PROMPT = """<uploaded_files>
{workdir}
</uploaded_files>
I've uploaded a python code repository in the directory {workdir}. Consider the following performance workload:

<performance_workload>
{workload}
</performance_workload>

Can you help me implement the necessary changes to the repository so that the runtime of the `workload()` function is
faster?

Guidelines:
1. Edit only non-test source files in the repository; do not modify the `workload()` function itself.
2. Keep the repository functionally equivalent: tests that pass now must still pass.
3. Make sure the workload runs successfully and gets faster; allow generous timeouts.
4. After changes to non-Python code (C, C++, Cython), rebuild the repository with: {rebuild_cmd}
5. Do not install or upgrade dependencies.

Steps:
1. Explore the repository to find the code the workload exercises.
2. Write a script that runs the workload and reports its runtime.
3. Edit the source code to make the workload faster.
4. Rebuild the repository if needed.
5. Rerun the script to confirm the speedup.
6. Run the related tests with `{test_cmd} <test file>`, e.g. on: {tests}
7. Reflect on each attempt and keep the fastest correct change.
8. Finish when done.

The speedup must come only from changes to the repository source, not from changes to the workload."""


@register
class SWEfficiency(_Official):
    """SWE-fficiency (Ma et al. 2025): swefficiency/swefficiency tasks, scored with the official `swefficiency` CLI against a
    gold baseline run. Speedup ratio = LM speedup / expert speedup per task (failed patches count as no edit), floored at
    0.001 and aggregated with the harmonic mean; reported x100."""
    name, dims, metrics = "swe_fficiency", ("EFF", "COR"), {"speedup": True, "correct": True}
    GOLD = "gold"

    def load(self) -> list[Task]:
        from datasets import load_dataset
        ts = []
        for r in load_dataset("swefficiency/swefficiency", split="test"):
            fill = {"workdir": "/testbed", "workload": r["workload"], "rebuild_cmd": r["rebuild_cmd"] or "(none needed)",
                    "test_cmd": r["test_cmd"], "tests": ", ".join(r["covering_tests"] or []) or "the related test files"}
            prompt = SWEF_PROMPT
            for k, v in fill.items():
                prompt = prompt.replace("{" + k + "}", v)
            ts.append(Task(r["instance_id"], self.name, prompt, mode="repo", date=str(r["created_at"])[:10],
                           meta={"harness": "mini", "image": r["image_name"], "workdir": "/testbed", "repo": r["repo"],
                                 "expert_speedup": r["speedup"],
                                 "visible_tests": [f"{r['test_cmd']} {t}" for t in r["covering_tests"] or []]}))
        return ts

    def _evaluate(self, d: Path, label: str, outs: list[Output], preds: Path) -> dict[str, dict]:
        home, log = RUNS / "official" / self.name, d / "harness.log"
        gold = home / "logs" / "run_evaluation" / self.GOLD / "gold"
        if not gold.exists():
            self._call(["swefficiency", "eval", "--run_id", self.GOLD, "--num_workers", str(WORKERS)], home, log)
        self._call(["swefficiency", "eval", "--run_id", d.name, "--num_workers", str(WORKERS), "--prediction_path",
                    str(preds)], home, log)
        self._call(["swefficiency", "report", "--gold_run", str(gold), "--pred_run",
                    str(home / "logs" / "run_evaluation" / d.name / label), "--report_output", str(d / "report")], home, log)
        rep = d / "report" / f"eval_report_{label}.csv"
        if not rep.exists():
            return {}
        out = {}
        for r in csv.DictReader(rep.open()):
            sr = next((r[k] for k in ("speedup_ratio", "SR", "sr", "score") if r.get(k) not in (None, "")), None)
            ok = next((r[k] for k in ("correct", "is_correct", "passed") if r.get(k) not in (None, "")), None)
            out[r["instance_id"]] = {"sr": float(sr) if sr is not None else None,
                                     "correct": str(ok).lower() in ("true", "1", "yes") if ok is not None else None}
        return out

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        srs = [max(r["sr"], 0.001) for r in rows if r.get("sr") is not None]
        return {"speedup": 100 * statistics.harmonic_mean(srs) if srs else float("nan"), "correct": _pct(rows, "correct")}


@register
class SusVibes(_Official):
    """SusVibes (Zhao et al. 2026): feature requests in real Python projects, scored with the official
    `python -m susvibes.eval.core` from a checkout of github.com/LeiLiLab/susvibes at $HARNESS_DATA/susvibes/susvibes.
    SecPass = functionally correct and secure; FuncPass = functionally correct."""
    name, dims, metrics = "susvibes", ("SEC", "COR"), {"sec_pass": True, "func_pass": True}
    repo = DATA / "susvibes" / "susvibes"

    def load(self) -> list[Task]:
        rows = [json.loads(l) for l in (self.repo / "datasets" / "default" / "susvibes_dataset.jsonl").read_text().splitlines()
                if l.strip()]
        return [Task(r["instance_id"], self.name, r["problem_statement"], mode="repo",
                     meta={"harness": "mini", "image": r["image_name"], "workdir": "/project", "visible_tests": []})
                for r in rows]

    def _evaluate(self, d: Path, label: str, outs: list[Output], preds: Path) -> dict[str, dict]:
        py = os.environ.get("SUSVIBES_PYTHON", sys.executable)
        self._call([py, "-m", "susvibes.eval.core", "--run_id", d.name, "--predictions_path", str(preds), "--max_workers",
                    str(WORKERS)], self.repo, d / "harness.log")
        base, out = self.repo / "logs" / "eval" / d.name / "none" / label, {}
        for o in outs:
            p = base / o.task_id / "report.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            run = r.get("run") or {}
            done = r.get("eval_status") == "completed"
            func = done and bool((run.get("func") or {}).get("pass"))
            out[o.task_id] = {"func": func, "sec": func and bool((run.get("sec") or {}).get("pass"))}
        return out

    def aggregate(self, rows: list[dict]) -> dict[str, float]:
        return {"sec_pass": _pct(rows, "sec"), "func_pass": _pct(rows, "func")}
