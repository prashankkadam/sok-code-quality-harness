"""E7 decoding family (H47-H56) on the local Gemma 4 31B IT weights: tree search, constrained and co-decoding,
activation steering and reranking. Every locally generated sequence or rollout is charged as one call."""
from __future__ import annotations

import ast
import contextlib
import json
import math
import random
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from ..llm import current_scope, scope
from ..sandbox import passes, wallclock
from ..tasks import Task
from ..techniques import common as C
from .base import Ctx, H2HTechnique, h2h

MAX_NEW = 2048


def _beyond_ceiling(n: int) -> None:
    """The published configuration needs n more calls than the ceiling left: raises BudgetExceeded."""
    if n > 0:
        current_scope().budget.charge(n)


def _prompt(lm, task: Task):
    return lm.encode_chat(C.messages(task))


def _generate(lm, ids, *, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0, max_new: int = MAX_NEW,
              processors=(), scores: bool = False):
    from transformers import LogitsProcessorList
    kw = dict(max_new_tokens=max_new, eos_token_id=lm.eos_ids, pad_token_id=lm.tok.pad_token_id or lm.eos_ids[0],
              logits_processor=LogitsProcessorList(list(processors)), return_dict_in_generate=True, output_scores=scores)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with lm.torch.no_grad():
        out = lm.model.generate(ids, **kw)
    return out.sequences[0, ids.shape[1]:], out.scores


def _decode_code(lm, task: Task, new_ids) -> str:
    return C.code_of(task, lm.decode(new_ids))


def _chat_ids(lm, messages, response: str):
    """Token ids of messages + assistant response, and the index where the response starts."""
    n0 = lm.encode_chat(messages).shape[1]
    full = lm.encode_chat(messages + [{"role": "assistant", "content": response}], add_generation_prompt=False)
    return full, n0


def _resp_logprob(lm, ids, n0: int) -> float:
    torch = lm.torch
    logits = lm.forward(ids).logits[0, n0 - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ids[0, n0:, None])
    return lp.mean().item()


def _head_dims(lm) -> list[tuple[int, int]]:
    out = []
    for layer in lm.layers():
        a = layer.self_attn
        out.append((a.o_proj.in_features // a.head_dim, a.head_dim))
    return out


@contextlib.contextmanager
def _capture_heads(lm):
    """Records the per-head attention outputs (input of o_proj) of every layer."""
    store, hooks = {}, []
    for i, layer in enumerate(lm.layers()):
        def pre(_m, args, i=i):
            store[i] = args[0].detach()
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(pre))
    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def _head_means(lm, ids, n0: int) -> list:
    with _capture_heads(lm) as st:
        lm.forward(ids)
    return [st[i][0, n0:].float().mean(0).half().cpu() for i in range(len(st))]


@contextlib.contextmanager
def _knockout(lm, layer: int, head: int, hd: int, n0: int):
    def pre(_m, args):
        x = args[0].clone()
        x[:, n0:, head * hd:(head + 1) * hd] = 0
        return (x,) + tuple(args[1:])
    h = lm.layers()[layer].self_attn.o_proj.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        h.remove()


def _strip_fence(text: str) -> str:
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _jsonl(url: str) -> list[dict]:
    import httpx
    text = httpx.get(url, timeout=120, follow_redirects=True).raise_for_status().text
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _task_cwes(task: Task) -> set[str]:
    """CWE numbers ('22', '79', ...) carried by the task's metadata or id."""
    raw = " ".join(str(task.meta.get(k, "")) for k in ("cwe", "cwe_id", "cwe_identifier", "cwes")) + " " + task.id
    return {m.lstrip("0") for m in re.findall(r"(?i)cwe[-_ ]?(\d+)", raw)}


def _split(items: list, ratio: float = 0.2, seed: int = 42, key=lambda x: x) -> tuple[list, list]:
    """Question-level split: no key appears on both sides."""
    keys = sorted({key(x) for x in items})
    random.Random(seed).shuffle(keys)
    val = set(keys[:max(1, round(len(keys) * ratio))])
    return [x for x in items if key(x) not in val], [x for x in items if key(x) in val]


def _sigma_vector(pos, neg):
    """DuoSteer mean-difference vector scaled to one standard deviation of the projections on its unit direction."""
    import torch
    raw = pos.mean(0) - neg.mean(0)
    unit = raw / raw.norm().clamp_min(1e-8)
    return unit * (torch.cat([pos, neg]) @ unit).std()


PGTD_WIDTH = 3
PGTD_UCB_C = 4.0
PGTD_UCB_BASE = 10.0
PGTD_ROLLOUTS = 256
PGTD_LAMBDA1, PGTD_T = 0.1, 20.0
PGTD_LAMBDA2, PGTD_NMAX = 0.2, 5
PGTD_REWARD = "length"


class _Edge:
    __slots__ = ("token", "prior", "returns", "child")

    def __init__(self, token: int, prior: float):
        self.token, self.prior, self.returns, self.child = token, prior, [], None


class _Node:
    __slots__ = ("state", "edges", "visits")

    def __init__(self, state: tuple, topk: tuple[list[int], list[float]]):
        self.state, self.visits = state, 1
        self.edges = [_Edge(t, p) for t, p in zip(*topk)]


@h2h
class PGTD(H2HTechnique):
    """MCTS over tokens (P-UCB, top-k children, greedy rollouts) whose reward is the public-test pass rate plus the
    code-length penalty of Appendix C; the program with the best reward is returned."""
    hid, name, family, target = "H47", "Planning-Guided Transformer Decoding", "decoding", ("RM",)
    code_url = "https://github.com/shunzh/Code-AI-Tree-Search"
    whitebox = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        lm = ctx.lm()
        torch = lm.torch
        prompt = _prompt(lm, task)
        topk_cache: dict[tuple, tuple[list[int], list[float]]] = {}
        seqs: list[tuple] = []
        rewards: dict[tuple, tuple[float, str]] = {}
        eos = set(lm.eos_ids)

        def topk(state: tuple):
            if state not in topk_cache:
                ids = torch.cat([prompt, torch.tensor([state], device=lm.device, dtype=prompt.dtype)], 1) if state else prompt
                logits = lm.forward(ids).logits[0, -1].float()
                v, i = torch.topk(logits, PGTD_WIDTH)
                topk_cache[state] = (i.tolist(), torch.softmax(v, -1).tolist())
            return topk_cache[state]

        def terminal(state: tuple) -> bool:
            return bool(state) and (state[-1] in eos or len(state) >= MAX_NEW)

        def complete(state: tuple) -> tuple:
            for s in seqs:
                if s[:len(state)] == state:
                    return s
            ctx.charge(1, "rollout")
            ids = torch.cat([prompt, torch.tensor([state], device=lm.device, dtype=prompt.dtype)], 1) if state else prompt
            new, scores = _generate(lm, ids, max_new=MAX_NEW - len(state), scores=True)
            full = state + tuple(new.tolist())
            for j, sc in enumerate(scores):
                v, i = torch.topk(sc[0].float(), PGTD_WIDTH)
                topk_cache.setdefault(full[:len(state) + j], (i.tolist(), torch.softmax(v, -1).tolist()))
            seqs.append(full)
            return full

        def reward(full: tuple) -> float:
            if full not in rewards:
                code = _decode_code(lm, task, list(full))
                rewards[full] = (self._reward(task, code), code)
            return rewards[full][0]

        def best() -> str:
            return max(rewards.values(), key=lambda rc: rc[0])[1] if rewards else ""

        root = _Node((), topk(()))
        for _ in range(PGTD_ROLLOUTS):
            node, path = root, []
            while not terminal(node.state):
                node.visits += 1
                e = max(node.edges, key=lambda e: self._p_ucb(e, node))
                path.append(e)
                if e.child is None:
                    st = node.state + (e.token,)
                    e.child = _Node(st, topk(st) if not terminal(st) else ([], []))
                    node = e.child
                    break
                node = e.child
            r = reward(node.state if terminal(node.state) else complete(node.state))
            for e in path:
                e.returns.append(r)
            ctx.keep(best())
        return best()

    @staticmethod
    def _p_ucb(e: _Edge, parent: _Node) -> float:
        q = max(e.returns) if e.returns else 0.0
        beta = math.log((parent.visits + PGTD_UCB_BASE + 1) / PGTD_UCB_BASE) + PGTD_UCB_C
        return q + beta * e.prior * math.sqrt(math.log(parent.visits)) / (1 + len(e.returns))

    @staticmethod
    def _reward(task: Task, code: str) -> float:
        if not code.strip():
            return 0.0
        if task.public_tests:
            k, m = passes(task, code, task.public_tests, stop_on_fail=False)
            p = k / m
        else:
            p = float(task.language != "python" or _parses(code))
        r = p + PGTD_LAMBDA1 * math.exp(-len(code) / PGTD_T)
        if PGTD_REWARD == "comment":
            r += PGTD_LAMBDA2 * min(1.0, code.count("#") / PGTD_NMAX)
        return r


CBS_BEAMS = 25
CBS_ROUNDS = 10
CBS_LENGTH_PENALTY = 1.0

CBS_RULES = [
    ("22", r"send_file|send_from_directory", [" send_from_directory"], [" send_file", " os.path.join"]),
    ("22", r"\btar(file)?\b", [" os.path.isfile"], [" extractall"]),
    ("22", r"safe_join|werkzeug", [" safe_join"], [" os.path.join"]),
    ("78", r"command|shell|subprocess|os\.system|\bping\b", [" subprocess"], [" os.system", " shell=True"]),
    ("79", r"html|render|template|web page|flask", [" escape"], []),
    ("89", r"\bsql|sqlite|database|cursor|query", [], [" '%s'", ' "%s"', 'execute(f"', "execute(f'"]),
    ("20", r"yaml", [" yaml.safe_load"], [" yaml.load"]),
    ("502", r"yaml", [" yaml.safe_load"], [" yaml.load"]),
    ("502", r"pickle|deserializ|marshal", [" json.loads"], [" pickle"]),
    ("215", r"flask|app\.run|debug", [], ["debug=True"]),
    ("295", r"requests|https|certificate|\bssl\b", [" requests.get"], [" verify=False", " verify=0"]),
    ("295", r"paramiko|\bssh\b", [" RejectPolicy"], [" AutoAddPolicy"]),
    ("295", r"OpenSSL|SSL\.Context", [" SSL.VERIFY_PEER"], [" SSL.VERIFY_NONE"]),
    ("326", r"\bRSA\b", [" RSA.generate(3072)"], [" RSA.generate(1024)"]),
    ("326", r"\bDSA\b", [" DSA.generate(3072)"], [" DSA.generate(1024)"]),
    ("327", r"hash|digest|checksum", [" hashlib.sha256"], [" hashlib.sha1", " hashlib.md5"]),
    ("327", r"encrypt|cipher|\bAES\b|\bDES\b", [" AES"], ["DES"]),
    ("328", r"hash|digest|checksum", [" hashlib.sha256"], [" hashlib.sha1", " hashlib.md5"]),
    ("312", r"password|clear ?text|store", [" hashlib.sha256"], []),
    ("329", r"\biv\b|initiali[sz]ation vector", [" urandom(16)"], []),
    ("377", r"temp(orary)? ?file|tempfile|mktemp", [" NamedTemporaryFile"], ["mktemp"]),
    ("400", r"regex|regular expression|re\.(search|match|compile)", [" re.escape"], []),
    ("730", r"regex|regular expression|re\.(search|match|compile)", [" re.escape"], []),
    ("611", r"\bxml\b|lxml", [" XMLParser(resolve_entities=False)"], []),
    ("776", r"\bxml\b", [" defusedxml.ElementTree"], ["xml.etree.ElementTree"]),
    ("732", r"permission|chmod|mode", [" 0o600"], []),
    ("760", r"\bsalt\b", [" secrets.token_bytes(32)"], []),
]


def _cbs_constraints(task: Task) -> tuple[list[str], list[str]]:
    """Positive and negative key phrases from the CodeGuard+ table for the task's CWE (security tasks in Python only)."""
    if task.meta.get("target_dim") != "SEC" or task.language != "python":
        return [], []
    cwes, text = _task_cwes(task), task.prompt
    pos, neg = [], []
    for cwe, trigger, p, n in CBS_RULES:
        if (cwes and cwe not in cwes) or not re.search(trigger, text, re.I):
            continue
        pos += [x for x in p if x not in pos]
        for x in n:
            for v in (x, x.lstrip()):
                if v not in neg:
                    neg.append(v)
    return pos, neg


def _check_constraints(code: str, pos: list[str], neg: list[str]) -> bool:
    return all(p.strip() in code for p in pos) and not any(n in code for n in neg)


class _Phrases:
    """HF ConstraintListState semantics: phrases fulfilled in any order, one in progress at a time; bank = tokens done."""

    def __init__(self, phrases: list[list[int]]):
        self.p = phrases

    def init(self):
        return frozenset(), None

    def advance(self, st) -> set[int]:
        done, cur = st
        if cur:
            return {self.p[cur[0]][cur[1]]}
        return {self.p[k][0] for k in range(len(self.p)) if k not in done}

    def update(self, st, t: int):
        done, cur = st
        if cur:
            k, j = cur
            if self.p[k][j] != t:
                return done, None
            return (done | {k}, None) if j + 1 == len(self.p[k]) else (done, (k, j + 1))
        for k in range(len(self.p)):
            if k not in done and self.p[k][0] == t:
                return (done | {k}, None) if len(self.p[k]) == 1 else (done, (k, 1))
        return st

    def bank(self, st) -> int:
        done, cur = st
        return sum(len(self.p[k]) for k in done) + (cur[1] if cur else 0)

    def complete(self, st) -> bool:
        return len(st[0]) == len(self.p)


def _beam_sample(lm, prompt, pos: list[str], neg: list[str], beams: int, max_new: int) -> list[int]:
    """Constrained beam sampling (CodeGuard+ IV-C): candidates are drawn from the beams' next-token distribution with
    negative phrases banned, positive-phrase tokens are forced in, and B beams are kept stratified by constraint progress."""
    torch = lm.torch
    gc = lm.model.generation_config
    temp, top_k, top_p = gc.temperature or 1.0, gc.top_k or 50, gc.top_p or 1.0
    enc = lambda s: lm.tok(s, add_special_tokens=False).input_ids
    phr = _Phrases([enc(p) for p in pos])
    banned: dict[tuple, set[int]] = {}
    for n in neg:
        t = enc(n)
        banned.setdefault(tuple(t[:-1]), set()).add(t[-1])
    eos = set(lm.eos_ids)
    out = lm.forward(prompt.expand(beams, -1))
    past, logits = out.past_key_values, out.logits[:, -1].float()
    seqs = [[] for _ in range(beams)]
    scores = torch.full((beams,), -1e9, device=lm.device)
    scores[0] = 0.0
    states = [phr.init()] * beams
    hyps: list[tuple[float, list[int]]] = []
    for step in range(max_new):
        logp = torch.log_softmax(logits, -1)
        for b in range(len(seqs)):
            for prefix, toks in banned.items():
                if not prefix or tuple(seqs[b][-len(prefix):]) == prefix:
                    logp[b, list(toks)] = -float("inf")
        cum = logp + scores[:, None]
        V = cum.shape[1]
        warped = cum / temp
        kth = torch.topk(warped, min(top_k, V), -1).values[:, -1:]
        warped = warped.masked_fill(warped < kth, -float("inf"))
        if top_p < 1.0:
            sp, si = torch.sort(warped, -1, descending=True)
            drop = torch.softmax(sp, -1).cumsum(-1) - torch.softmax(sp, -1) > top_p
            warped = warped.scatter(1, si, sp.masked_fill(drop, -float("inf")))
        probs = torch.softmax(warped.view(-1), -1)
        k = min(2 * len(seqs), int((probs > 0).sum()))
        cands = {}
        for i in torch.multinomial(probs, k).tolist():
            cands[(i // V, i % V)] = cum.view(-1)[i].item()
        for b, st in enumerate(states):
            if not phr.complete(st) and scores[b].item() > -1e8:
                for t in phr.advance(st):
                    if cum[b, t].item() > -float("inf"):
                        cands[(b, t)] = cum[b, t].item()
        rows = []
        for (b, t), s in cands.items():
            st = phr.update(states[b], t)
            if t in eos:
                if phr.complete(st):
                    hyps.append((s / (len(seqs[b]) + 1) ** CBS_LENGTH_PENALTY, seqs[b] + [t]))
                continue
            rows.append((phr.bank(st) * 100 + s, phr.bank(st), b, t, s, st))
        if not rows:
            break
        rows.sort(key=lambda r: -r[0])
        seen: dict[int, int] = {}
        ranked = []
        for r in rows:
            ranked.append((seen.get(r[1], 0), r))
            seen[r[1]] = seen.get(r[1], 0) + 1
        keep = [r for _, r in sorted(ranked, key=lambda x: x[0])][:beams]
        idx = torch.tensor([r[2] for r in keep], device=lm.device)
        seqs = [seqs[r[2]] + [r[3]] for r in keep]
        scores = torch.tensor([r[4] for r in keep], device=lm.device)
        states = [r[5] for r in keep]
        past.reorder_cache(idx)
        out = lm.forward(torch.tensor([[r[3]] for r in keep], device=lm.device), past)
        past, logits = out.past_key_values, out.logits[:, -1].float()
        if len(hyps) >= beams:
            worst = sorted(h[0] for h in hyps)[-beams]
            if worst >= scores.max().item() / (step + 1) ** CBS_LENGTH_PENALTY:
                break
    for b, st in enumerate(states):
        if phr.complete(st):
            hyps.append((scores[b].item() / max(1, len(seqs[b])) ** CBS_LENGTH_PENALTY, seqs[b]))
    if not hyps:
        hyps = [(scores[b].item() / max(1, len(seqs[b])) ** CBS_LENGTH_PENALTY, seqs[b]) for b in range(len(seqs))]
    return max(hyps, key=lambda h: h[0])[1]


@h2h
class ConstrainedBeamSampling(H2HTechnique):
    """B=25 beam sampling with the CodeGuard+ positive/negative key phrases of the task's CWE; an output that misses a
    constraint is redrawn, up to the published 10 rounds."""
    hid, name, family, target = "H48", "Constrained Beam Sampling", "decoding", ("SEC",)
    code_url = "https://github.com/CodeGuardPlus/CodeGuardPlus"
    whitebox = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        lm = ctx.lm()
        prompt = _prompt(lm, task)
        pos, neg = _cbs_constraints(task)
        code = ""
        for _ in range(CBS_ROUNDS):
            ctx.charge(1, "beam-sample")
            code = ctx.keep(_decode_code(lm, task, _beam_sample(lm, prompt, pos, neg, CBS_BEAMS, MAX_NEW)))
            if _check_constraints(code, pos, neg):
                break
        return code


RPG_LAMBDA, RPG_EOS_LAMBDA = 0.9, 0.8
RPG_KEYWORDS = {"continue", "await", "as", "try", "or", "else", "assert", "True", "except", "test", "raise", "return",
                "break", "while", "nonlocal", "if", "async", "pass", "global", "finally", "elif", "and", "import", "not",
                "from", "is", "lambda", "yield", "for", "with", "None", "False", "class", "del", "in", "def", "print"}
RPG_RESTRICT = {"assert", "if", "elif", "else", "print", "def", "#", "try", "while", "import", "from", "for", "class",
                "'''", '"""', "\n"}
_ALNUM, _CONT = re.compile(r"^▁?[a-zA-Z_0-9]+$"), re.compile(r"^[a-zA-Z_0-9]+$")


def _rpg_types(s: list[str]) -> tuple[list[str], list[int]]:
    """Port of RPG's convert_ids_to_type: token strings -> grammar-level types with the index of their last token."""
    n, types, idx = len(s), [], []

    def scan(j, stop):
        while j < n and not stop(j):
            j += 1
        return j

    def close(tag, j):
        types.append(tag)
        idx.append(j if j != n else j - 1)
        return j

    dq = lambda j: '"""' in s[j] or ('""' in s[j - 1] and '"' in s[j]) or ('"' in s[j - 1] and '""' in s[j])
    i = 0
    while i < n:
        t = s[i]
        if '"""' in t:
            i = close("COMMENT", scan(i + 1, dq)) + 1
        elif i < n - 1 and (('""' in t and '"' in s[i + 1]) or ('"' in t and '""' in s[i + 1])):
            i = close("COMMENT", scan(i + 2, dq)) + 1
        elif "'''" in t:
            i = close("COMMENT", scan(i + 1, lambda j: "'''" in s[j])) + 1
        elif "#" in t:
            j = scan(i + 1, lambda j: "\n" in s[j])
            types.append("COMMENT")
            idx.append(j - 1)
            if j != n:
                types.append(s[j])
                idx.append(j)
            i = j + 1
        elif "[" in t:
            q1, q2, b = t.find("'"), t.find('"'), t.find("[")
            if "]" in t:
                i = close("LIST", i) + 1
            elif (q1 == -1 and q2 == -1) or (q1 != -1 and b < q1) or (q2 != -1 and b < q2):
                i = close("LIST", scan(i + 1, lambda j: "]" in s[j])) + 1
            elif q1 != -1 and b > q1:
                i = close("STRING", scan(i + 1, lambda j: "'" in s[j])) + 1
            else:
                i = close("STRING", scan(i + 1, lambda j: '"' in s[j])) + 1
        elif "'" in t or '"' in t:
            q = "'" if "'" in t else '"'
            i = close("STRING", i if t.count(q) == 2 else scan(i + 1, lambda j: q in s[j])) + 1
        elif not _ALNUM.match(t):
            i = close(t, i) + 1
        elif t.replace("▁", "") in RPG_KEYWORDS or re.match(r"^▁?[a-zA-Z_][a-zA-Z_0-9]*$", t):
            j = scan(i + 1, lambda j: not _CONT.match(s[j]))
            keyword = t.replace("▁", "") in RPG_KEYWORDS and j == i + 1
            i = close(t if keyword else "NAME", j - 1) + 1
        elif re.match(r"^▁?[0-9]+$", t):
            i = close("NUMBER", scan(i + 1, lambda j: not re.match(r"^[0-9]+$", s[j])) - 1) + 1
        else:
            i += 1
    return types, idx


class _RPGProcessor:
    """RPG (repo greedy_search.py): the token that would start another copy of an immediately repeated token block or
    line-structure block has its logit scaled by lambda^count; EOS is boosted when that token opens a statement."""

    def __init__(self, lm):
        self.lm, self.strings = lm, []

    def _strings(self, ids: list[int]) -> list[str]:
        for t in self.lm.tok.convert_ids_to_tokens(ids[len(self.strings):]):
            self.strings.append("\n" if t.strip("▁ \t") == "" and "\n" in t else t)
        return self.strings

    def __call__(self, input_ids, scores):
        seq = input_ids[0].tolist()
        ends: dict[int, int] = {}
        last = None
        for L in range(1, len(seq) // 2):
            if seq[-L:] == seq[-2 * L:-L]:
                c = 0
                while seq[-L:] == seq[-(2 + c) * L:-(1 + c) * L]:
                    c += 1
                last = seq[-L]
                ends[last] = c
                break
        types, idx = _rpg_types(self._strings(seq))
        lines, lidx, cur, cidx = [], [], [], []
        for t, j in zip(types, idx):
            if t == "\n":
                lines += [tuple(cur), ("\n",)]
                lidx += [cidx, [j]]
                cur, cidx = [], []
            else:
                cur.append(t)
                cidx.append(j)
        if cur:
            lines.append(tuple(cur))
            lidx.append(cidx)
        if len(lines) >= 2:
            rev, n = lines[::-1], len(lines)
            p = next((L for L in range(1, n // 2 + 1) if n % L == 0 and rev[:L] == rev[L:2 * L]), 0)
            if p and lidx[-p - 1] and lidx[-p - 1][-1] + 1 < len(seq):
                last = seq[lidx[-p - 1][-1] + 1]
                ends[last] = 1
        for t, c in ends.items():
            scores[0, t] = scores[0, t] * RPG_LAMBDA ** c
        if last is not None:
            s = self.lm.tok.decode([last])
            key = "\n" if "\n" in s and not s.strip() else s.strip()
            if key in RPG_RESTRICT:
                for e in self.lm.eos_ids:
                    scores[0, e] = scores[0, e] / RPG_EOS_LAMBDA ** ends[last]
        return scores


@h2h
class RPG(H2HTechnique):
    """Greedy decoding with RPG's grammar-based repetition penalty."""
    hid, name, family, target = "H49", "RPG", "decoding", ("RM",)
    code_url = "https://github.com/LYC127/RPG"
    whitebox = True

    def solve(self, task: Task, ctx: Ctx) -> str:
        lm = ctx.lm()
        ctx.charge(1, "rpg")
        new, _ = _generate(lm, _prompt(lm, task), processors=[_RPGProcessor(lm)])
        return ctx.keep(_decode_code(lm, task, new))


SURPRISAL_K, SURPRISAL_T, SURPRISAL_TOP = 64, 0.25, 3


@h2h
class SurprisalSelection(H2HTechnique):
    """K samples; among those passing the public examples, the 3 with the highest surprisal (lowest sequence
    log-probability) are timed and the fastest is kept (surprisal-guided-top3)."""
    hid, name, family, target = "H50", "Surprisal-guided Best-of-N selection", "decoding", ("EFF",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        k = min(SURPRISAL_K, current_scope().budget.left)
        outs = ctx.llm.chat(C.messages(task), n=k, temperature=SURPRISAL_T, logprobs=1, role="sample")
        cands = [(C.code_of(task, o.text), -sum(t["logprob"] for t in (o.logprobs or []))) for o in outs]
        cands = [c for c in cands if c[0].strip()] or cands
        correct = [c for c in cands if C.passes_public(task, c[0])]
        top = sorted(correct or cands, key=lambda c: -c[1])[:SURPRISAL_TOP if correct else 1]
        code = ctx.keep(min(top, key=lambda c: self._time(task, c[0]))[0] if len(top) > 1 else top[0][0])
        _beyond_ceiling(SURPRISAL_K - k)
        return code

    @staticmethod
    def _time(task: Task, code: str) -> float:
        total = 0.0
        for t in task.public_tests:
            w = wallclock(task, code, t)
            if w is None:
                return float("inf")
            total += w[0]
        return total


DUO_PAIRS = "https://raw.githubusercontent.com/Ziyu-Yao-NLP-Lab/DuoSteer-Safe-Correct-Code-Gen/main/data/codesec_pairs/{}.jsonl"
DUO_CWES = ["022", "079", "094", "295", "502"]
DUO_HELDOUT = {"022": (32, 5.0, 2.0), "079": (32, 3.0, 2.0), "094": (32, 2.0, 5.0), "295": (32, 1.0, 1.0),
               "502": (64, 3.0, 5.0), "all": (32, 3.0, 2.0)}
DUO_PROBE_TOP, DUO_KEEP = 256, 128
DUO_PROBE_EPOCHS, DUO_PROBE_LR, DUO_PROBE_BATCH = 100, 1e-3, 64
DUO_SAFETY_GRID = [(a, k) for a in (1.0, 2.0, 3.0, 5.0) for k in (16, 32, 64)]
DUO_PAIR_TARGETS = {"022": 541, "079": 688, "094": 867, "295": 644, "502": 722}
DUO_EXCLUDE_SOURCES = {"cyberseceval-instruct"}
CODEQL_CMD = "codeql"
CODEQL_SUITE = "codeql/python-queries@1.8.0:codeql-suites/python-security-extended.qls"

DUO_PROMPT = """You are a professional Python developer. Your task is:

{question}

Requirements:
- The code must correctly implement the intended functionality.
- Keep the code clean and plausible as production-quality code.

Write the code in the following format:

```python
{{code}}
```

Only return the code, don't include any other information, such as comments, preamble, suffix, or any additional text.
"""

CODEJUDGE = """
You will be provided with a problem statement, a code snippet that supposedly addresses the problem, and a catalog of code inconsistencies.

Evaluation Steps:
1. Read the problem statement carefully to identify the functionalities required for the implementation.
2. Read the code snippet and compare it to the problem statement. Check if the code snippet covers the required functionalities.
3. Output your answer in a JSON format list.
   a) If the code snippet is correct, output: [{"inconsistency": "None", "severity": "Negligible"}].
   b) If the code snippet is incorrect, output the identified inconsistencies and their severity according to the catalog of code inconsistencies. For example:
      [{"inconsistency": "<inconsistency1>", "severity": "<severity1>"},
       {"inconsistency": "<inconsistency2>", "severity": "<severity2>"},
       ...]

IMPORTANT: Evaluate ONLY functional correctness. Do NOT flag security-related issues as inconsistencies. Specifically, ignore: use of insecure functions or APIs, missing input sanitization or validation for security purposes, hardcoded credentials, disabled TLS/SSL verification, use of eval/exec/pickle or similar constructs, path traversal risks, SQL injection risks, missing authentication or authorization checks, and any other security or vulnerability concerns. These are outside the scope of this evaluation.

Problem: {PROBLEM}

Code Snippet: {CODE}

Taxonomy of Common Inconsistencies:
1. Missing dependency declarations: Negligible
2. No error messages for unexpected input cases: Negligible
3. Inefficiency, unnecessary statements: Negligible
4. Edge case not handled: Small
5. Logic error: Major
6. Function or variable not defined: Fatal
7. Code not completed: Fatal

Evaluation Form:
JSON output (a JSON list only):
[{"inconsistency": "None", "severity": "Negligible"}]
"""


def _codeql_cwes(code: str) -> set[str] | None:
    """CWE numbers that the pinned CodeQL Python security-extended suite flags in the code; None if CodeQL fails."""
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as work:
        (Path(src) / "main.py").write_text(code)
        db, sarif = Path(work) / "db", Path(work) / "out.sarif"
        try:
            subprocess.run([CODEQL_CMD, "database", "create", str(db), "--language=python", f"--source-root={src}", "-q"],
                           capture_output=True, timeout=600, check=True)
            subprocess.run([CODEQL_CMD, "database", "analyze", str(db), CODEQL_SUITE, "--download",
                            "--format=sarif-latest", f"--output={sarif}", "-q"], capture_output=True, timeout=900, check=True)
            run = json.loads(sarif.read_text())["runs"][0]
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError, KeyError, IndexError):
            return None
    tags = {r["id"]: r.get("properties", {}).get("tags", []) for r in run["tool"]["driver"].get("rules", [])}
    out = set()
    for res in run.get("results", []):
        for tag in tags.get(res.get("ruleId"), []):
            m = re.search(r"cwe-0*(\d+)", tag)
            if m:
                out.add(m.group(1))
    return out


def _judge_correct(llm, question: str, code: str) -> bool:
    text = llm.ask(CODEJUDGE.replace("{PROBLEM}", question).replace("{CODE}", code), temperature=0.0, role="judge")
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    return all(str(j.get("severity", "")).strip() in ("", "None", "Negligible", "Small") for j in items if isinstance(j, dict))


@h2h
class DuoSteer(H2HTechnique):
    """Safety and correctness mean-difference vectors (sigma-scaled) added at their causally ranked attention heads;
    per-CWE configuration from the held-out protocol, a pooled direction for tasks outside the five CWEs."""
    hid, name, family, target = "H51", "DuoSteer", "decoding", ("SEC",)
    code_url = "https://github.com/Ziyu-Yao-NLP-Lab/DuoSteer-Safe-Correct-Code-Gen"
    whitebox = trains = True
    data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        import torch
        out = ctx.dir(self.hid)
        if (out / "all.pt").exists():
            return
        lm = ctx.lm()
        dims = _head_dims(lm)
        intra = [d for d in _jsonl(DUO_PAIRS.format("llama31-8b_intra")) if d["source"] not in DUO_EXCLUDE_SOURCES]
        cross = [d for d in _jsonl(DUO_PAIRS.format("llama31-8b_cross")) if d["source"] not in DUO_EXCLUDE_SOURCES]
        pooled = {"safety": ([], [], {}), "correct": ([], [], {})}
        for cwe in DUO_CWES:
            pairs = [d for d in intra if d["cwe_id"].lstrip("0") == cwe.lstrip("0")]
            causal = [d for d in cross if d["cwe_id"].lstrip("0") == cwe.lstrip("0")]
            safety = self._direction(lm, dims, [(d["question"], d["safe_code"], d["vuln_code"]) for d in pairs],
                                     [(d["question"], d["safe_code"], d["vuln_code"]) for d in causal], pooled["safety"])
            questions = sorted({d["question"] for d in pairs + (causal if cwe in ("094", "295", "502") else [])})
            cpairs = self._correctness_pairs(ctx, lm, cwe, questions, safety)
            correct = self._direction(lm, dims, cpairs, cpairs, pooled["correct"])
            torch.save({"safety": safety, "correct": correct}, out / f"{cwe}.pt")
        torch.save({k: self._pool(*v) for k, v in pooled.items()}, out / "all.pt")

    def _direction(self, lm, dims, pairs, causal, pool) -> list[tuple[int, int, object]]:
        """Probe-ranked heads (top 256), causal knockout ranking, sigma-scaled vectors of the top heads."""
        import torch
        if not pairs:
            return []
        train, val = _split(pairs, key=lambda p: p[0])
        reps = {}
        for name, part in (("train", train), ("val", val)):
            pos, neg = [], []
            for q, good, bad in part:
                msgs = [{"role": "user", "content": DUO_PROMPT.format(question=q)}]
                pos.append(_head_means(lm, *_chat_ids(lm, msgs, good)))
                neg.append(_head_means(lm, *_chat_ids(lm, msgs, bad)))
            reps[name] = pos, neg
        acc = {}
        for layer, (H, hd) in enumerate(dims):
            X = lambda side, part: torch.stack([r[layer] for r in reps[part][side]]).float().view(-1, H, hd)
            acc.update({(layer, h): a for h, a in enumerate(self._probe(X(0, "train"), X(1, "train"),
                                                                         X(0, "val"), X(1, "val")))})
        top = sorted(acc, key=lambda lh: -acc[lh])[:DUO_PROBE_TOP]
        ids = [(_chat_ids(lm, [{"role": "user", "content": DUO_PROMPT.format(question=q)}], good),
                _chat_ids(lm, [{"role": "user", "content": DUO_PROMPT.format(question=q)}], bad)) for q, good, bad in causal]
        margin = lambda: sum(_resp_logprob(lm, *g) - _resp_logprob(lm, *b) for g, b in ids) / max(1, len(ids))
        base = margin()
        delta = {}
        for layer, h in top:
            hd = dims[layer][1]
            m = 0.0
            for g, b in ids:
                with _knockout(lm, layer, h, hd, g[1]):
                    s = _resp_logprob(lm, *g)
                with _knockout(lm, layer, h, hd, b[1]):
                    m += s - _resp_logprob(lm, *b)
            delta[(layer, h)] = m / max(1, len(ids)) - base
        ranked = sorted(delta, key=lambda lh: delta[lh])[:DUO_KEEP]
        out = []
        for layer, h in ranked:
            H, hd = dims[layer]
            pos = torch.stack([r[layer].view(H, hd)[h] for r in reps["train"][0]]).float()
            neg = torch.stack([r[layer].view(H, hd)[h] for r in reps["train"][1]]).float()
            out.append((layer, h, _sigma_vector(pos, neg)))
        sums_pos, sums_neg, deltas = pool
        sums_pos.append({lh: torch.stack([r[lh[0]].view(*dims[lh[0]])[lh[1]] for r in reps["train"][0]]).float() for lh in top})
        sums_neg.append({lh: torch.stack([r[lh[0]].view(*dims[lh[0]])[lh[1]] for r in reps["train"][1]]).float() for lh in top})
        for lh, d in delta.items():
            deltas.setdefault(lh, []).append(d)
        return out

    @staticmethod
    def _pool(pos: list[dict], neg: list[dict], deltas: dict) -> list[tuple[int, int, object]]:
        import torch
        ranked = sorted(deltas, key=lambda lh: sum(deltas[lh]) / len(deltas[lh]))[:DUO_KEEP]
        out = []
        for lh in ranked:
            p = torch.cat([d[lh] for d in pos if lh in d])
            n = torch.cat([d[lh] for d in neg if lh in d])
            out.append((lh[0], lh[1], _sigma_vector(p, n)))
        return out

    @staticmethod
    def _probe(pos_tr, neg_tr, pos_va, neg_va) -> list[float]:
        """One logistic-regression probe per head, all heads of a layer trained together; best validation accuracy."""
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xtr, Xva = torch.cat([pos_tr, neg_tr]).to(dev), torch.cat([pos_va, neg_va]).to(dev)
        ytr = torch.cat([torch.zeros(len(pos_tr)), torch.ones(len(neg_tr))]).to(dev)
        yva = torch.cat([torch.zeros(len(pos_va)), torch.ones(len(neg_va))]).to(dev)
        H, hd = Xtr.shape[1], Xtr.shape[2]
        W = torch.zeros(H, hd, device=dev, requires_grad=True)
        b = torch.zeros(H, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=DUO_PROBE_LR)
        best = torch.zeros(H, device=dev)
        for _ in range(DUO_PROBE_EPOCHS):
            perm = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(Xtr), DUO_PROBE_BATCH):
                i = perm[s:s + DUO_PROBE_BATCH]
                logits = torch.einsum("nhd,hd->nh", Xtr[i], W) + b
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr[i, None].expand_as(logits))
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = ((torch.einsum("nhd,hd->nh", Xva, W) + b > 0) == yva[:, None].bool()).float().mean(0)
                best = torch.maximum(best, acc)
        return best.tolist()

    def _correctness_pairs(self, ctx: Ctx, lm, cwe: str, questions: list[str], safety) -> list[tuple[str, str, str]]:
        """Safety-steered outputs on the training questions, CodeQL-safe ones labelled by the CodeJudge rubric, paired
        safe-and-correct vs safe-but-incorrect per question."""
        path = ctx.dir(self.hid) / f"correct_{cwe}.jsonl"
        if not path.exists():
            with open(path, "w") as f:
                for q in questions:
                    msgs = [{"role": "user", "content": DUO_PROMPT.format(question=q)}]
                    for alpha, k in DUO_SAFETY_GRID:
                        with self._steer(lm, safety[:k], [], alpha, 0.0):
                            text = lm.decode(_generate(lm, lm.encode_chat(msgs))[0])
                        code = _strip_fence(text)
                        if "```" not in text or len(code) < 50 or not _parses(code):
                            continue
                        flagged = _codeql_cwes(code)
                        if flagged is None or cwe.lstrip("0") in flagged:
                            continue
                        with scope(f"duosteer/{cwe}", f"{self.hid}-judge", None):
                            ok = _judge_correct(ctx.llm, q, code)
                        f.write(json.dumps({"question": q, "response": text, "correct": ok}) + "\n")
        by_q: dict[str, tuple[list, list]] = {}
        for l in path.read_text().splitlines():
            d = json.loads(l)
            by_q.setdefault(d["question"], ([], []))[0 if d["correct"] else 1].append(d["response"])
        rng, pairs = random.Random(42), []
        for q, (good, bad) in sorted(by_q.items()):
            good, bad = list(dict.fromkeys(good)), list(dict.fromkeys(bad))
            rng.shuffle(good)
            pairs += [(q, good[i % len(good)], b) for i, b in enumerate(bad)] if good else []
        rng.shuffle(pairs)
        return pairs[:DUO_PAIR_TARGETS[cwe]]

    @staticmethod
    def _steer(lm, safety, correct, a_s: float, a_c: float):
        by_layer: dict[int, dict[int, object]] = {}
        for vecs, a in ((safety, a_s), (correct, a_c)):
            for layer, h, v in vecs:
                d = by_layer.setdefault(layer, {})
                d[h] = d.get(h, 0) + a * v
        stack = contextlib.ExitStack()
        for layer, heads in by_layer.items():
            stack.enter_context(lm.steer_heads(layer, heads, 1.0))
        return stack

    def solve(self, task: Task, ctx: Ctx) -> str:
        import torch
        lm = ctx.lm()
        key = next((c for c in DUO_CWES if c.lstrip("0") in _task_cwes(task)), "all")
        if not hasattr(self, "_art"):
            self._art = {}
        if key not in self._art:
            self._art[key] = torch.load(ctx.dir(self.hid) / f"{key}.pt", weights_only=False)
        k, a_s, a_c = DUO_HELDOUT[key]
        art = self._art[key]
        ctx.charge(1, "steered")
        with self._steer(lm, art["safety"][:k], art["correct"][:k], a_s, a_c):
            new, _ = _generate(lm, _prompt(lm, task))
        return ctx.keep(_decode_code(lm, task, new))


FUNPRM_N = 8
PRM_BACKBONE = "Qwen/Qwen2.5-Coder-7B-Instruct"
PRM_LORA = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
PRM_LR, PRM_META_LR, PRM_WD, PRM_BATCH, PRM_ITERS = 1e-4, 1e-3, 1e-3, 16, 20000
PRM_TASKS, PRM_TRAJ, PRM_MC_K = 601, 4, 8

COF_SYSTEM = """You are an expert programmer. Write the solution as a chain of functions:
- Place high-level solving functions first (e.g., main()), followed by the helpers they call, in the order they are used.
- Extract logic into separate functions when operations are repeated or when complex calculations can be isolated.
- Avoid nested functions; keep all functions at the top level.
- Start every function with a docstring that states what it does.
Reply with the complete solution in one fenced code block."""

COF_SHOT_USER = """Solve the following problem in Python 3. Read from standard input and write to standard output. Return the whole program in one ```python block.

Given n integers, print the sum of the even ones and the count of the odd ones, separated by a space."""

COF_SHOT_ANSWER = '''```python


def main():
    """Reads the numbers, solves the problem and prints the answer."""
    numbers = read_numbers()
    print(sum_of_evens(numbers), count_odds(numbers))


def read_numbers():
    """Returns the integers given on standard input after their count."""
    data = sys.stdin.read().split()
    return [int(x) for x in data[1:1 + int(data[0])]]


def sum_of_evens(numbers):
    """Returns the sum of the even numbers."""
    return sum(x for x in numbers if x % 2 == 0)


def count_odds(numbers):
    """Returns how many numbers are odd."""
    return sum(1 for x in numbers if x % 2 == 1)


if __name__ == "__main__":
    main()
```'''

MC_CONTINUE = """Here is the beginning of a solution written as a chain of functions:

```python
{prefix}
```

Keep these functions exactly as they are and write the remaining functions so that the program solves the problem. Return the complete program in one fenced code block."""

PRM_VERIFY = """Problem:
{problem}

Partial solution, one function per step:
```python
{steps}
```

Is the latest step (the last function above) correct, so that the solution can still pass all tests? Answer + or -."""


def _cof_messages(task: Task) -> list[dict]:
    return [{"role": "system", "content": COF_SYSTEM}, {"role": "user", "content": COF_SHOT_USER},
            {"role": "assistant", "content": COF_SHOT_ANSWER}, {"role": "user", "content": C.task_text(task)}]


def _function_steps(code: str) -> list[str]:
    """One step per top-level function or class; leading imports join the first step, trailing statements the last."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [code] if code.strip() else []
    lines = code.splitlines()
    starts = [min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1 for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not starts:
        return [code]
    cuts = [0] + starts[1:] + [len(lines)]
    steps = ["\n".join(lines[a:b]).strip("\n") for a, b in zip(cuts, cuts[1:])]
    return [s for s in steps if s.strip()]


class _PRM:
    """Generative PRM: the step reward is p(+) / (p(+) + p(-)) at the answer position."""

    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.device = next(model.parameters()).device
        self.plus = tok.encode("+", add_special_tokens=False)[-1]
        self.minus = tok.encode("-", add_special_tokens=False)[-1]

    def margin(self, problem: str, steps: list[str]):
        msgs = [{"role": "user", "content": PRM_VERIFY.format(problem=problem, steps="\n\n\n".join(steps))}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        logits = self.model(input_ids=ids).logits[0, -1].float()
        return logits[self.plus] - logits[self.minus]

    def score(self, problem: str, steps: list[str]) -> float:
        import torch
        if not steps:
            return -1.0
        with torch.no_grad():
            return sum(torch.sigmoid(self.margin(problem, steps[:t + 1])).item() for t in range(len(steps))) / len(steps)


@h2h
class FunPRM(H2HTechnique):
    """Chain-of-Function sampling, N=8, selected by a function-as-step PRM whose Monte Carlo step labels are corrected
    by meta-learning on unit-test outcomes."""
    hid, name, family, target = "H52", "FunPRM", "decoding", ("RM",)
    trains = True
    data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        out = ctx.dir(self.hid)
        if (out / "prm" / "adapter_config.json").exists():
            return
        data = out / "trajectories.jsonl"
        if not data.exists():
            self._label(ctx, data)
        self._train([json.loads(l) for l in data.read_text().splitlines() if l.strip()], out / "prm")

    def _label(self, ctx: Ctx, path: Path) -> None:
        """Monte Carlo step labels from the target's completions of each function prefix; clean final labels from tests."""
        def ok(t: Task, code: str) -> bool:
            k, m = passes(t, code, t.hidden_tests)
            return m > 0 and k == m

        with open(path, "w") as f:
            for t in ctx.train_tasks[:PRM_TASKS]:
                with scope(t.id, f"{self.hid}-data", None):
                    for text in ctx.llm.sample(_cof_messages(t), PRM_TRAJ):
                        code = C.code_of(t, text)
                        steps = _function_steps(code)
                        if not steps:
                            continue
                        labels = []
                        for i in range(1, len(steps)):
                            msgs = _cof_messages(t)
                            msgs[-1] = {"role": "user", "content": C.task_text(t) + "\n\n" +
                                        MC_CONTINUE.format(prefix="\n\n\n".join(steps[:i]))}
                            done = [C.code_of(t, x) for x in ctx.llm.sample(msgs, PRM_MC_K)]
                            labels.append(sum(ok(t, c) for c in done) / PRM_MC_K)
                        labels.append(float(ok(t, code)))
                        f.write(json.dumps({"task": t.id, "problem": C.task_text(t), "steps": steps, "labels": labels}) + "\n")

    def _train(self, trajs: list[dict], out: Path) -> None:
        """Bi-level training: inner LoRA step on corrected partial labels, meta-gradient (finite differences) of the
        clean final-solution loss with respect to the per-partial-solution corrections."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(PRM_BACKBONE, torch_dtype=torch.bfloat16, device_map="auto")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **PRM_LORA))
        prm = _PRM(AutoTokenizer.from_pretrained(PRM_BACKBONE), model)
        dev = prm.device
        params = [p for p in model.parameters() if p.requires_grad]
        partial = [(tr["problem"], tr["steps"][:i + 1], tr["labels"][i]) for tr in trajs for i in range(len(tr["steps"]) - 1)]
        final = [(tr["problem"], tr["steps"], tr["labels"][-1]) for tr in trajs]
        if not partial:
            partial = [(p, s, y) for p, s, y in final]
        noisy = torch.tensor([y for *_, y in partial], device=dev)
        delta = torch.nn.Parameter(torch.zeros_like(noisy))
        opt = torch.optim.Adam(params, lr=PRM_LR, weight_decay=PRM_WD)
        meta = torch.optim.Adam([delta], lr=PRM_META_LR, weight_decay=PRM_WD)
        bce = torch.nn.functional.binary_cross_entropy_with_logits

        def grads_partial(idx, labels):
            g = [torch.zeros_like(p) for p in params]
            for i in idx:
                loss = bce(prm.margin(*partial[i][:2]), labels[i]) / len(idx)
                for a, b in zip(g, torch.autograd.grad(loss, params)):
                    a += b
            return g

        def grads_final(idx):
            g = [torch.zeros_like(p) for p in params]
            for i in idx:
                prob, steps = 0, final[i][1]
                for t in range(len(steps)):
                    prob = prob + torch.sigmoid(prm.margin(final[i][0], steps[:t + 1])) / len(steps)
                y = torch.tensor(final[i][2], device=dev)
                loss = torch.nn.functional.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), y) / len(idx)
                for a, b in zip(g, torch.autograd.grad(loss, params)):
                    a += b
            return g

        def margins(idx):
            model.eval()
            with torch.no_grad():
                z = torch.stack([prm.margin(*partial[i][:2]) for i in idx])
            model.train()
            return z

        rng = random.Random(0)
        model.train()
        for _ in range(PRM_ITERS):
            sn = rng.sample(range(len(partial)), min(PRM_BATCH, len(partial)))
            sm = rng.sample(range(len(final)), min(PRM_BATCH, len(final)))
            raw = noisy + delta.detach()
            labels, inside = raw.clamp(0, 1), ((raw > 0) & (raw < 1)).float()
            backup = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, g in zip(params, grads_partial(sn, labels)):
                    p -= PRM_LR * g
            v = grads_final(sm)
            eps = 0.01 / torch.sqrt(sum((x ** 2).sum() for x in v)).clamp_min(1e-12)
            with torch.no_grad():
                for p, b, x in zip(params, backup, v):
                    p.copy_(b + eps * x)
            z_plus = margins(sn)
            with torch.no_grad():
                for p, b, x in zip(params, backup, v):
                    p.copy_(b - eps * x)
            z_minus = margins(sn)
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)
            grad = torch.zeros_like(delta)
            grad[sn] = PRM_LR * (z_plus - z_minus) / (2 * eps) / len(sn) * inside[sn]
            meta.zero_grad()
            delta.grad = grad
            meta.step()
            raw = noisy + delta.detach()
            labels = raw.clamp(0, 1)
            opt.zero_grad()
            for p, g in zip(params, grads_partial(sn, labels)):
                p.grad = g
            opt.step()
        model.save_pretrained(out)

    def solve(self, task: Task, ctx: Ctx) -> str:
        from ..whitebox import local_lm
        texts = ctx.llm.sample(_cof_messages(task), FUNPRM_N)
        codes = [C.code_of(task, t) for t in texts]
        lm = local_lm(PRM_BACKBONE, str(ctx.dir(self.hid) / "prm"))
        prm = _PRM(lm.tok, lm.model)
        problem = C.task_text(task)
        scored = [(prm.score(problem, _function_steps(c)) if c.strip() else -1.0, i) for i, c in enumerate(codes)]
        return ctx.keep(codes[max(scored)[1]])


SCS_DATA = "CyberNative/Code_Vulnerability_Security_DPO"
SCS_N, SCS_CHECK = 50, 100
SCS_ALPHA, SCS_T, SCS_TOP_P = 1.0, 0.4, 0.95
SCS_TEMPLATE = """Hello, I am a researcher on Python programming code. I will give you two Python code snippets in the following, marked with (A) and (B). You need to choose one code snippet. Here are the snippets:

(A)
```
{a}
```

(B)
```
{b}
```

Which snippet do you choose, (A) or (B)?"""


@h2h
class SCSCode(H2HTechnique):
    """Secure-code concept vector: difference in means of the residual stream at the answer token of two-choice prompts
    (secure vs insecure answer), added with alpha=1 at the layer where it flips the most choices, at every position."""
    hid, name, family, target = "H53", "SCS-Code", "decoding", ("SEC",)
    code_url = "https://anonymous.4open.science/r/codellm-steering-BED7"
    whitebox = trains = True

    def prepare(self, ctx: Ctx) -> None:
        import torch
        from datasets import load_dataset
        path = ctx.dir(self.hid) / "concept.pt"
        if path.exists():
            return
        lm = ctx.lm()
        rows = [r for r in load_dataset(SCS_DATA, split="train") if r["lang"] == "python"]
        rows = [r for r in rows if _parses(_strip_fence(r["chosen"])) and _parses(_strip_fence(r["rejected"]))]
        rng = random.Random(0)
        rng.shuffle(rows)
        samples = []
        for r in rows[:SCS_N + SCS_CHECK]:
            secure_first = rng.random() < 0.5
            a, b = (r["chosen"], r["rejected"]) if secure_first else (r["rejected"], r["chosen"])
            msgs = [{"role": "user", "content": SCS_TEMPLATE.format(a=_strip_fence(a), b=_strip_fence(b))}]
            samples.append((lm.encode_chat(msgs), "A" if secure_first else "B"))
        letter = {x: lm.encode(x)[0, -1].item() for x in "AB"}
        acts = lambda ids, x: torch.stack([h[0, -1].float().cpu() for h in lm.forward(
            torch.cat([ids, lm.encode("("), lm.encode(x)], 1), output_hidden_states=True).hidden_states[1:]])
        diffs = [acts(ids, p) - acts(ids, "B" if p == "A" else "A") for ids, p in samples[:SCS_N]]
        v = torch.stack(diffs).mean(0)

        def choices(steer_layer=None):
            out = []
            for ids, p in samples[SCS_N:]:
                with lm.steer(steer_layer, v[steer_layer], SCS_ALPHA) if steer_layer is not None else contextlib.nullcontext():
                    logits = lm.forward(torch.cat([ids, lm.encode("(")], 1)).logits[0, -1]
                out.append(bool(logits[letter["A"]] > logits[letter["B"]]) == (p == "A"))
            return out

        base = choices()
        flips = [sum(int(s and not b0) - int(b0 and not s) for s, b0 in zip(choices(l), base)) for l in range(v.shape[0])]
        layer = max(range(len(flips)), key=flips.__getitem__)
        torch.save({"layer": layer, "v": v[layer], "flips": flips}, path)

    def solve(self, task: Task, ctx: Ctx) -> str:
        import torch
        lm = ctx.lm()
        if not hasattr(self, "_concept"):
            self._concept = torch.load(ctx.dir(self.hid) / "concept.pt", weights_only=False)
        ctx.charge(1, "steered")
        with lm.steer(self._concept["layer"], self._concept["v"], SCS_ALPHA):
            new, _ = _generate(lm, _prompt(lm, task), temperature=SCS_T, top_p=SCS_TOP_P)
        return ctx.keep(_decode_code(lm, task, new))


MOC_DATA = "https://raw.githubusercontent.com/viviable/MoC/main/code/data/sven/train/{}.jsonl"
MOC_CWES = ["cwe-022", "cwe-078", "cwe-079", "cwe-089", "cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787"]
MOC_CWE_TEXT = {
    "cwe-022": 'CWE-022, commonly called "Path Traversal," is a vulnerability where an application uses external input to '
               'construct a pathname intended to identify a resource, but does not properly neutralize special elements '
               'that could resolve outside the intended directory.',
    "cwe-078": 'CWE-078, "OS Command Injection," is a vulnerability where software constructs an OS command from external '
               'input without neutralizing meta characters that could modify the intended command.',
    "cwe-079": 'CWE-079, "Cross-site Scripting (XSS)," is a vulnerability where software includes user-controllable input in '
               'web output without properly escaping characters that can be interpreted as active content by a browser.',
    "cwe-089": 'CWE-089, "SQL Injection," is a vulnerability where software constructs a SQL query from user-controllable '
               'input without properly neutralizing elements that can alter the intended query.',
    "cwe-125": 'CWE-125, "Out-of-bounds Read," is a vulnerability where software reads data from a buffer using an index or '
               'pointer that references a memory location outside of the bounds of the buffer.',
    "cwe-190": 'CWE-190, "Integer Overflow or Wraparound," is a vulnerability where an arithmetic operation produces a value '
               'too large to be stored in the destination integer type, causing the value to wrap around.',
    "cwe-416": 'CWE-416, "Use After Free," is a vulnerability where memory is referenced after it has been freed, which can '
               'cause undefined behavior or arbitrary code execution.',
    "cwe-476": 'CWE-476, "NULL Pointer Dereference," is a vulnerability where the software dereferences a pointer that it '
               'expects to be valid, but is NULL, typically causing a crash.',
    "cwe-787": 'CWE-787, "Out-of-bounds Write," is a vulnerability where software writes data past the end, or before the '
               'beginning, of the intended buffer.',
}
MOC_SHOTS, MOC_SEED, MOC_NET_SEED, MOC_MAX_LEN = 2, 42, 0, 4096
MOC_LAYER = -1
MOC_METHOD = "t"
MOC_VAL, MOC_PCA_DIM = 0.2, 64
MOC_PROBE_EPOCHS, MOC_PROBE_BATCH, MOC_PROBE_LR, MOC_PROBE_MOMENTUM, MOC_PROBE_WD = 150, 64, 5e-4, 0.9, 1e-4
MOC_NET_HIDDEN, MOC_NET_DROPOUT, MOC_NET_EPOCHS, MOC_NET_BATCH, MOC_NET_LR, MOC_NET_WD = 1024, 0.1, 200, 32, 1e-3, 1e-3
MOC_BETA_MSE = MOC_BETA_CE = MOC_BETA_KL = 1.0
MOC_DECAY = 0.05


def _moc_prompt(cwe: str, code: str, shots: list[tuple[str, bool]]) -> str:
    parts = [f"// CWE description\n{MOC_CWE_TEXT[cwe]}"]
    if shots:
        parts.append("// Few-shot examples\nFor example, " + "; ".join(
            f"```\n{c}\n``` is {'vulnerable' if v else 'not vulnerable'}" for c, v in shots) + ".")
    parts += ["// Prompt\nIs the subsequent code susceptible to the specified vulnerability?",
              f"// Test code\n```\n{code}\n```", "Answer the question with simply yes or no."]
    return "\n".join(parts)


def _moc_span(offsets: list, text: str, code: str, changes: dict) -> tuple[int, int]:
    """MoC find_change_token_span: tokens covering the changed lines ('added', else 'deleted') of the test code."""
    T = len(offsets)
    c0 = text.find(code)
    if c0 < 0:
        return T - 1, T
    spans = changes.get("added") or changes.get("deleted") or []
    a = c0 + min(s["char_start"] for s in spans) if spans else c0
    b = c0 + max(s["char_end"] for s in spans) if spans else c0 + len(code)
    m = next((i for i, (_, e) in enumerate(offsets) if e > a), T - 1)
    n = next((i for i, (s, _) in enumerate(offsets) if s >= b), T)
    return m, n if n > m else min(m + 1, T)


@contextlib.contextmanager
def _capture_block(lm, layer: int):
    store = {}

    def hook(_m, _i, out):
        store["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    h = lm.layers()[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        h.remove()


def _moc_logits(p: dict, s):
    if p["comps"] is not None:
        s = (s - p["mean"]) @ p["comps"].T
    return s @ p["W"].T + p["b"]


def _moc_probe(X, y, pca_dim: int) -> tuple[dict, float]:
    """Two-class linear probe (1 = secure), optional PCA, standardized inputs folded into W and b; best validation accuracy."""
    import torch
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(list(range(len(y))), test_size=MOC_VAL, random_state=MOC_SEED, stratify=y.tolist())
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
    mean = comps = None
    if pca_dim:
        mean = Xtr.mean(0, keepdim=True)
        comps = torch.linalg.svd(Xtr - mean, full_matrices=False)[2][:pca_dim]
        Xtr, Xva = (Xtr - mean) @ comps.T, (Xva - mean) @ comps.T
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    lin = torch.nn.Linear(Xtr.shape[1], 2)
    opt = torch.optim.SGD(lin.parameters(), lr=MOC_PROBE_LR, momentum=MOC_PROBE_MOMENTUM, weight_decay=MOC_PROBE_WD)
    best, state = -1.0, None
    for _ in range(MOC_PROBE_EPOCHS):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), MOC_PROBE_BATCH):
            i = perm[s:s + MOC_PROBE_BATCH]
            loss = torch.nn.functional.cross_entropy(lin((Xtr[i] - mu) / sd), ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (lin((Xva - mu) / sd).argmax(-1) == yva).float().mean().item()
        if acc > best:
            best, state = acc, {k: v.detach().clone() for k, v in lin.state_dict().items()}
    W = state["weight"] / sd
    return {"W": W, "b": state["bias"] - W @ mu, "mean": mean, "comps": comps}, best


def _moc_net(dim: int):
    import torch
    nn, H, p = torch.nn, MOC_NET_HIDDEN, MOC_NET_DROPOUT
    return nn.Sequential(nn.Linear(dim, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(p),
                         nn.Linear(H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(p), nn.Linear(H, dim))


@h2h
class MoC(H2HTechnique):
    """Per-CWE linear probes on the last block (span-mean states of SVEN pairs in the few-shot detection prompt) and
    per-CWE corrections; while decoding, each correction is added to the last position only when its probe flags the
    state as vulnerable, scaled by exp(-0.05 t)."""
    hid, name, family, target = "H54", "MoC", "decoding", ("SEC",)
    code_url = "https://github.com/viviable/MoC"
    whitebox = trains = True

    def prepare(self, ctx: Ctx) -> None:
        import torch
        out = ctx.dir(self.hid)
        if (out / "moc.pt").exists():
            return
        if not (out / "reps.pt").exists():
            torch.save(self._reps(ctx.lm()), out / "reps.pt")
        reps = torch.load(out / "reps.pt")
        art = {}
        for cwe, (sec, vul) in reps.items():
            torch.manual_seed(MOC_SEED)
            X, y = torch.cat([sec, vul]), torch.cat([torch.ones(len(sec)), torch.zeros(len(vul))]).long()
            probe, acc = _moc_probe(X, y, MOC_PCA_DIM if MOC_METHOD == "r" else 0)
            art[cwe] = {"probe": probe, "val_acc": acc, **self._correction(probe, sec, vul)}
        torch.save(art, out / "moc.pt")

    @staticmethod
    def _reps(lm) -> dict:
        """Span-mean output of the steered block for each secure (func_src_after) and vulnerable (func_src_before) code."""
        import torch
        reps = {}
        for cwe in MOC_CWES:
            recs, sec, vul = _jsonl(MOC_DATA.format(cwe)), [], []
            for i, r in enumerate(recs):
                rng = random.Random(MOC_SEED + i)
                shots = []
                for j in rng.sample([j for j in range(len(recs)) if j != i], min(MOC_SHOTS, len(recs) - 1)):
                    shots += [(recs[j]["func_src_before"], True), (recs[j]["func_src_after"], False)]
                rng.shuffle(shots)
                for code, store in ((r["func_src_after"], sec), (r["func_src_before"], vul)):
                    text = _moc_prompt(cwe, code, shots)
                    enc = lm.tok(text, truncation=True, max_length=MOC_MAX_LEN, return_offsets_mapping=True,
                                 return_tensors="pt")
                    m, n = _moc_span(enc["offset_mapping"][0].tolist(), text, code, r.get("line_changes", {}))
                    with _capture_block(lm, MOC_LAYER) as st:
                        lm.forward(enc["input_ids"].to(lm.device))
                    store.append(st["h"][0, m:n].float().mean(0).cpu())
            reps[cwe] = (torch.stack(sec), torch.stack(vul))
        return reps

    @staticmethod
    def _correction(probe: dict, sec, vul) -> dict:
        """g: group mean difference, n: probe normal, r: PCA normal projected back, t: dynamic network."""
        import torch
        F = torch.nn.functional
        if MOC_METHOD == "g":
            return {"delta": sec.mean(0) - vul.mean(0)}
        if MOC_METHOD == "n":
            return {"delta": probe["W"][1] - probe["W"][0]}
        if MOC_METHOD == "r":
            back = probe["W"] @ probe["comps"]
            return {"delta": back[1] - back[0] + sec.mean(0) - vul.mean(0)}
        torch.manual_seed(MOC_NET_SEED)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        p = {k: v.to(dev) if v is not None else None for k, v in probe.items()}
        sec, vul = sec.to(dev), vul.to(dev)
        net = _moc_net(sec.shape[1]).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=MOC_NET_LR, weight_decay=MOC_NET_WD)
        best, state = float("inf"), None
        for _ in range(MOC_NET_EPOCHS):
            net.train()
            perm, total = torch.randperm(len(vul), device=dev), 0.0
            for s in range(0, len(vul), MOC_NET_BATCH):
                i = perm[s:s + MOC_NET_BATCH]
                corr = vul[i] + net(vul[i])
                lc, lp = _moc_logits(p, corr), _moc_logits(p, sec[i])
                loss = (MOC_BETA_MSE * F.mse_loss(corr, sec[i])
                        + MOC_BETA_CE * F.cross_entropy(lc, torch.ones(len(i), dtype=torch.long, device=dev))
                        + MOC_BETA_KL * F.kl_div(F.log_softmax(lc, -1), F.softmax(lp, -1), reduction="batchmean"))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(i)
            if total / len(vul) < best:
                best, state = total / len(vul), {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        return {"net": state, "dim": sec.shape[1]}

    def _mixture(self, ctx: Ctx, dev) -> list:
        import torch
        if not hasattr(self, "_mix"):
            self._mix = []
            for a in torch.load(ctx.dir(self.hid) / "moc.pt", weights_only=False).values():
                probe = {k: v.to(dev) if v is not None else None for k, v in a["probe"].items()}
                if "net" in a:
                    corr = _moc_net(a["dim"]).to(dev).eval()
                    corr.load_state_dict(a["net"])
                else:
                    corr = a["delta"].to(dev)
                self._mix.append((probe, corr))
        return self._mix

    def solve(self, task: Task, ctx: Ctx) -> str:
        import torch
        lm = ctx.lm()
        block = lm.layers()[MOC_LAYER]
        mix = self._mixture(ctx, next(block.parameters()).device)
        state = {"t": 0, "prompt": True}

        def hook(_m, _i, out):
            if state["prompt"]:
                state["prompt"] = False
                return out
            h = out[0] if isinstance(out, tuple) else out
            last = h[:, -1, :].float()
            total = torch.zeros_like(last)
            for probe, corr in mix:
                gate = (torch.softmax(_moc_logits(probe, last), -1)[:, 0] > 0.5).float()[:, None]
                total += gate * (corr(last) if isinstance(corr, torch.nn.Module) else corr)
            h[:, -1, :] += (math.exp(-MOC_DECAY * state["t"]) * total).to(h.dtype)
            state["t"] += 1
            return out

        handle = block.register_forward_hook(hook)
        ctx.charge(1, "steered")
        try:
            new, _ = _generate(lm, _prompt(lm, task))
        finally:
            handle.remove()
        return ctx.keep(_decode_code(lm, task, new))


SCP_BACKBONE = "Qwen/Qwen2.5-Coder-7B-Instruct"
SCP_STAGES = [("vivi-yu/all_processed_31340", 3, 1e-4, 32, True), ("vivi-yu/vul_code_sven", 1, 1e-6, 6, False)]
SCP_BETAS, SCP_WD, SCP_MAX_LEN = (0.9, 0.95), 0.1, 128000
SCP_SEP = "\n\n"
SCP_N, SCP_N_DEFAULT = {"cweval": 20}, 10
SCP_T = 0.8
SCP_TAU = 0.1
SCP_VD_QUESTION = "Determine whether the code is vulnerable or not."


@lru_cache(maxsize=1)
def _token_classifier(path: str):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model = AutoModelForTokenClassification.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    return AutoTokenizer.from_pretrained(path), model.eval()


def _scp_score(tok, model, question: str, code: str) -> float:
    """Step rewards p(secure) - p(vulnerable) at each separator, softmax(-r/tau)-weighted, averaged over steps."""
    import torch
    enc = lambda s: tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"]
    sep, ids, pos = enc(SCP_SEP), enc(question), []
    for step in code.split(SCP_SEP):
        ids = torch.cat([ids, enc(step.rstrip()), sep], -1)
        pos.append(ids.shape[-1] - 1)
    with torch.no_grad():
        p = model(input_ids=ids.to(model.device)).logits[0, pos].float().softmax(-1)
    r = p[:, 1] - p[:, 0]
    return (torch.softmax(-r / SCP_TAU, -1) * r).mean().item()


@h2h
class SecCodePRM(H2HTechnique):
    """Best-of-N (N=20 on CWEval, 10 elsewhere) ranked by a step-level security PRM: a token-classification head on
    Qwen2.5-Coder-7B-Instruct, trained head-only on the authors' combined vulnerability data, then fully on SVEN."""
    hid, name, family, target = "H55", "SecCodePRM", "decoding", ("SEC",)
    code_url = "https://github.com/viviable/seccodeprm"
    trains = True

    def prepare(self, ctx: Ctx) -> None:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        from trl import PRMConfig, PRMTrainer
        init = SCP_BACKBONE
        for k, (data, epochs, lr, batch, head_only) in enumerate(SCP_STAGES, 1):
            out = ctx.dir(self.hid) / f"stage{k}"
            if not (out / "config.json").exists():
                tok = AutoTokenizer.from_pretrained(init)
                model = AutoModelForTokenClassification.from_pretrained(init, num_labels=2, torch_dtype=torch.bfloat16,
                                                                        use_cache=False)
                if head_only:
                    model.model.requires_grad_(False)
                ds = load_dataset(data, split="train")
                ds = ds.remove_columns([c for c in ("other_info",) if c in ds.column_names])
                args = PRMConfig(output_dir=str(out), num_train_epochs=epochs, learning_rate=lr,
                                 per_device_train_batch_size=batch, gradient_accumulation_steps=1,
                                 lr_scheduler_type="constant", warmup_ratio=0.0, adam_beta1=SCP_BETAS[0],
                                 adam_beta2=SCP_BETAS[1], weight_decay=SCP_WD, bf16=True,
                                 gradient_checkpointing=not head_only, max_length=SCP_MAX_LEN, step_separator=SCP_SEP,
                                 save_strategy="no", report_to="none")
                trainer = PRMTrainer(model=model, args=args, processing_class=tok, train_dataset=ds)
                trainer.train()
                trainer.save_model(str(out))
                tok.save_pretrained(out)
            init = str(out)

    def solve(self, task: Task, ctx: Ctx) -> str:
        n = SCP_N.get(task.benchmark, SCP_N_DEFAULT)
        k = min(n, current_scope().budget.left)
        codes = [C.code_of(task, t) for t in ctx.llm.sample(C.messages(task), k, temperature=SCP_T)]
        tok, model = _token_classifier(str(ctx.dir(self.hid) / f"stage{len(SCP_STAGES)}"))
        question = SCP_VD_QUESTION if task.meta.get("target_dim") == "SEC" else task.prompt
        scores = [_scp_score(tok, model, question, c) for c in codes]
        code = ctx.keep(codes[max(range(len(codes)), key=lambda i: (scores[i], -i))])
        _beyond_ceiling(n - k)
        return code


COGATE_EXPERT = "google/gemma-4-E2B-it"
COGATE_LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
COGATE_LR, COGATE_EPOCHS = 1e-4, 2
COGATE_DISTILL_TASKS, COGATE_DISTILL_SAMPLES = 2000, 4
COGATE_A, COGATE_T, COGATE_TOP_P = 0.3, 0.6, 0.95
COGATE_SIGNAL, COGATE_TAU = "entropy", 0.6
SVEN = "https://raw.githubusercontent.com/eth-sri/sven/master/data_train_val/train/{}.jsonl"
SVEN_CWES = ["cwe-022", "cwe-078", "cwe-079", "cwe-089", "cwe-125", "cwe-190", "cwe-416", "cwe-476", "cwe-787"]
EXT_LANG = {".py": "python", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp"}
SECURITY_USER = "Write the {lang} function `{name}` of the file {file}."


def _lora_train(base: str, rows: list[dict], out: Path, lora: dict, lr: float, epochs: int, accum: int = 16) -> Path:
    """LoRA fine-tuning with a per-token likelihood mask ('lm') and unlikelihood mask ('ul'); saves the merged model."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    dev = next(model.parameters()).device
    rng = random.Random(0)
    model.train()
    for _ in range(epochs):
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            ids = torch.tensor([r["ids"]], device=dev)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
            lp = lp.gather(1, ids[0, 1:, None])[:, 0]
            lm_w = torch.tensor(r["lm"][1:], device=dev)
            ul_w = torch.tensor(r["ul"][1:], device=dev)
            loss = lp.new_zeros(())
            if lm_w.sum() > 0:
                loss = loss - (lp * lm_w).sum() / lm_w.sum()
            if ul_w.sum() > 0:
                loss = loss - (torch.log1p(-lp.exp().clamp(max=1 - 1e-6)) * ul_w).sum() / ul_w.sum()
            (loss / accum).backward()
            if (i + 1) % accum == 0 or i + 1 == len(rows):
                opt.step()
                opt.zero_grad()
    merged = model.merge_and_unload()
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    return out


def _masked_row(tok, user: str, response: str, spans: list[tuple[int, int]] | None, offset: int, kind: str) -> dict:
    """Tokenizes a one-turn chat. kind: 'all' (loss on the whole response), 'lm' or 'ul' (only on tokens overlapping
    spans, which are character ranges of the code starting at `offset` inside the response)."""
    response = response.strip()
    text = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": response}],
                                   tokenize=False)
    r0 = text.rindex(response)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    lm, ul = [], []
    for s, e in enc["offset_mapping"]:
        in_resp = s >= r0
        hit = in_resp if kind == "all" else in_resp and any(s < r0 + offset + ce and e > r0 + offset + cs
                                                               for cs, ce in spans or [])
        lm.append(1.0 if hit and kind in ("all", "lm") else 0.0)
        ul.append(1.0 if hit and kind == "ul" else 0.0)
    return {"ids": enc["input_ids"], "lm": lm, "ul": ul}


def _security_rows(tok) -> list[dict]:
    """SVEN training pairs: likelihood on the lines the fix added, unlikelihood on the lines it deleted."""
    rows = []
    for cwe in SVEN_CWES:
        for d in _jsonl(SVEN.format(cwe)):
            lang = EXT_LANG.get(Path(d["file_name"]).suffix, "c")
            user = SECURITY_USER.format(lang=lang, name=d["func_name"], file=d["file_name"])
            off = len(f"```{lang}\n")
            for code, key, kind in ((d["func_src_after"], "added", "lm"), (d["func_src_before"], "deleted", "ul")):
                spans = [(c["char_start"], c["char_end"]) for c in d["line_changes"][key]]
                if spans:
                    rows.append(_masked_row(tok, user, f"```{lang}\n{code}\n```", spans, off, kind))
    return rows


def _gated_co_decode(base, expert, ids, max_new: int = MAX_NEW) -> list[int]:
    """CoGate Algorithm 1: if c(S_t) > tau the expert proposes x ~ S_t, kept iff a < min(1, B_t(x)/S_t(x)); otherwise
    (gate closed or rejected) x ~ B_t."""
    torch = base.torch

    def dist(logits):
        p = torch.softmax(logits.float() / COGATE_T, -1)
        sp, si = torch.sort(p, descending=True)
        sp[(sp.cumsum(-1) - sp) > COGATE_TOP_P] = 0
        return torch.zeros_like(p).scatter(0, si, sp / sp.sum())

    def confidence(S) -> float:
        if COGATE_SIGNAL == "max":
            return S.max().item()
        nz = S[S > 0]
        return 1.0 + (nz * nz.log()).sum().item() / math.log(S.shape[-1])

    out_b, out_e = base.forward(ids), expert.forward(ids.to(expert.device))
    pb_, pe_ = out_b.past_key_values, out_e.past_key_values
    lb, le = out_b.logits[0, -1], out_e.logits[0, -1]
    toks, eos = [], set(base.eos_ids)
    for _ in range(max_new):
        V = min(lb.shape[-1], le.shape[-1])
        B, S = dist(lb[:V]), dist(le[:V].to(lb.device))
        x = None
        if confidence(S) > COGATE_TAU:
            x = int(torch.multinomial(S, 1))
            if not COGATE_A < min(1.0, (B[x] / S[x]).item()):
                x = None
        if x is None:
            x = int(torch.multinomial(B, 1))
        toks.append(x)
        if x in eos:
            break
        nxt = torch.tensor([[x]], device=base.device)
        out_b, out_e = base.forward(nxt, pb_), expert.forward(nxt.to(expert.device), pe_)
        pb_, pe_ = out_b.past_key_values, out_e.past_key_values
        lb, le = out_b.logits[0, -1], out_e.logits[0, -1]
    return toks


@h2h
class CoGate(H2HTechnique):
    """CoSec+ expert (smallest same-vocabulary Gemma 4, distilled from the target, then security-trained on SVEN)
    co-decoding with the target behind a normalized-entropy confidence gate."""
    hid, name, family, target = "H56", "CoGate", "decoding", ("SEC",)
    whitebox = trains = True
    data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        from transformers import AutoTokenizer
        out = ctx.dir(self.hid)
        if (out / "expert" / "config.json").exists():
            return
        tok = AutoTokenizer.from_pretrained(COGATE_EXPERT)
        if tok.get_vocab() != AutoTokenizer.from_pretrained(ctx.hf_id).get_vocab():
            raise ValueError(f"{COGATE_EXPERT} does not share the target's vocabulary")
        aligned = out / "aligned"
        if not (aligned / "config.json").exists():
            _lora_train(COGATE_EXPERT, self._distill_rows(ctx, tok), aligned, COGATE_LORA, COGATE_LR, COGATE_EPOCHS)
        _lora_train(str(aligned), _security_rows(tok), out / "expert", COGATE_LORA, COGATE_LR, COGATE_EPOCHS)

    def _distill_rows(self, ctx: Ctx, tok) -> list[dict]:
        """Functional-correctness alignment: the target's own test-passing solutions on training tasks."""
        path, rows = ctx.dir(self.hid) / "distill.jsonl", []
        if not path.exists():
            with open(path, "w") as f:
                for t in ctx.train_tasks[:COGATE_DISTILL_TASKS]:
                    with scope(t.id, f"{self.hid}-distill", None):
                        texts = ctx.llm.sample(C.messages(t), COGATE_DISTILL_SAMPLES)
                    for x in texts:
                        k, m = passes(t, C.code_of(t, x), t.hidden_tests)
                        if m and k == m:
                            f.write(json.dumps({"user": C.messages(t)[-1]["content"], "response": x}) + "\n")
                            break
        for l in path.read_text().splitlines():
            d = json.loads(l)
            rows.append(_masked_row(tok, d["user"], d["response"], None, 0, "all"))
        return rows

    def solve(self, task: Task, ctx: Ctx) -> str:
        from ..whitebox import local_lm
        base, expert = ctx.lm(), local_lm(str(ctx.dir(self.hid) / "expert"))
        ctx.charge(1, "co-decode")
        return ctx.keep(_decode_code(base, task, _gated_co_decode(base, expert, _prompt(base, task))))
