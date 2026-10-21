"""E7 self-refinement family, H21-H28. The critic and the reviser are the E7 model itself; tools appear only where the paper uses them."""
from __future__ import annotations

import ast
import json
import math
import random
import re
import uuid
from functools import lru_cache

from ..llm import current_scope
from ..sandbox import passes_public
from ..tasks import Task
from ..techniques import common as C
from ..techniques.rag import CORPUS
from .base import Ctx, H2HTechnique, h2h
from .feedback import _draw, _download, _fence, _parses, _py, _user, codeql, codeql_text

# ------------------------------------------------------------------------------------------------- H21 Self-Refine

SR_PROMPTS = "https://raw.githubusercontent.com/madaan/self-refine/main/data/prompt/pie/{name}.txt"
SR_ATTEMPTS, SR_TEMP, SR_FEEDBACK_TOKENS, SR_STOP = 3, 0.0, 300, ["### END"]
SR_FEEDBACK = "{examples}{code}\n\n# Why is this code slow?\n"
SR_ITERATE = "{examples}{code}\n\n# Why is this code slow?\n\n{feedback}\n\n# Improved version:\n\n"


@h2h
class SelfRefine(H2HTechnique):
    """Self-Refine with its PIE code-optimization prompts: FEEDBACK ("Why is this code slow?") and ITERATE few-shot prompts
    from the authors' repository, three attempts, early stop on "this code is not slow", last version returned."""
    hid, name, family, target = "H21", "Self-Refine", "refinement", ("EFF",)
    code_url = "https://github.com/madaan/self-refine"

    def prepare(self, ctx: Ctx) -> None:
        for n in ("feedback", "iterate"):
            _download(SR_PROMPTS.format(name=n), ctx.dir(self.hid) / f"{n}.txt")

    def solve(self, task: Task, ctx: Ctx) -> str:
        d = ctx.dir(self.hid)
        fb_examples, it_examples = (d / "feedback.txt").read_text(), (d / "iterate.txt").read_text()
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task), temperature=SR_TEMP)))
        prev, feedback = code, ""
        for attempt in range(SR_ATTEMPTS):
            if attempt:
                text = ctx.llm.ask(_user(SR_ITERATE.format(examples=it_examples, code=prev, feedback=feedback)),
                                   temperature=SR_TEMP, stop=SR_STOP, role="iterate")
                code = ctx.keep(C.code_of(task, text.split("### END")[0]) or prev)
            feedback = ctx.llm.ask(_user(SR_FEEDBACK.format(examples=fb_examples, code=code)), temperature=SR_TEMP,
                                   max_tokens=SR_FEEDBACK_TOKENS, stop=SR_STOP, role="feedback").split("### END")[0].strip()
            if "this code is not slow" in feedback.lower():
                break
            prev = code
        return code


# ------------------------------------------------------------------------------------------------- H22 CodeChain

CC_PROMPTS = "https://raw.githubusercontent.com/SalesforceAIResearch/CodeChain/main/prompts/{name}.txt"
CC_SYSTEM = "You are a helpful AI assistant to help developers to solve challenging coding problems."
CC_SAMPLES, CC_ROUNDS, CC_CLUSTERS, CC_TEMP = 20, 5, 5, 1.0
CC_ENCODER, CC_MAX_TOKENS = "bigcode/starencoder", 1024


def _cc_modules(output: str) -> list[str]:
    """Sub-module headers (```module blocks, split per def) from STEP 1 of a CodeChain answer."""
    blocks = re.findall(r"```\s*module((?:.|\n)*?)```", output)
    if not blocks and "\nSTEP 2" in output:
        blocks = re.findall(r"```\s*python((?:.|\n)*?)```", output[:output.index("\nSTEP 2")])
    out = []
    for b in blocks:
        if "class " in b or "def main(" in b:
            continue
        idx = [m.start() for m in re.finditer("def ", b)]
        out += [b[i:j] for i, j in zip(idx, idx[1:] + [len(b)])]
    return out


def _cc_functions(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [s for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            if (s := ast.get_source_segment(code, n))]


def _cc_pairs(output: str, code: str) -> list[tuple[str, str]]:
    funcs, codes = _cc_modules(output), _cc_functions(code)
    pairs = []
    for f in funcs:
        head = next((ln for ln in f.split("\n") if "def " in ln), None)
        match = next((c for c in codes if head and head.strip() in c), None)
        if match:
            pairs.append((f, match))
    return pairs or [("No docstring", c) for c in codes]


def _cc_prompt(doc: str, code: str) -> str:
    if doc == "No docstring":
        return code
    lines = code.split("\n")
    cut = next((i for i, ln in enumerate(lines) if "def " in ln), -1)
    return doc + "\n" + "\n".join(lines[cut + 1:])


class _StarEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(CC_ENCODER)
        self.tok.add_special_tokens({"pad_token": "<pad>", "sep_token": "<sep>", "cls_token": "<cls>", "mask_token": "<mask>"})
        self.model = AutoModel.from_pretrained(CC_ENCODER).eval()

    def __call__(self, texts: list[str]):
        enc = self.tok([f"<cls>{t}<sep>" for t in texts], padding="longest", max_length=CC_MAX_TOKENS, truncation=True,
                       return_tensors="pt")
        with self.torch.no_grad():
            h = self.model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
        return self.torch.nn.functional.normalize(e, dim=-1).numpy()


@h2h
class CodeChain(H2HTechnique):
    """Modular-CoT generation (20 samples), then five rounds of self-revision conditioned on representative sub-modules:
    sub-modules of samples passing the public examples are embedded with StarEncoder and k-means clustered (k = 5, 4, ...)."""
    hid, name, family, target = "H22", "CodeChain", "refinement", ("RM",)
    code_url = "https://github.com/SalesforceAIResearch/CodeChain"
    _encoder = None

    def prepare(self, ctx: Ctx) -> None:
        for n in ("codechain_gen", "codechain_revise"):
            _download(CC_PROMPTS.format(name=n), ctx.dir(self.hid) / f"{n}.txt")

    def solve(self, task: Task, ctx: Ctx) -> str:
        if not _py(task):
            return C.code_of(task, ctx.llm.ask(C.messages(task)))
        d = ctx.dir(self.hid)
        guide = "use the provided function signature" if task.mode == "function" else "read from and write to standard IO"
        fill = lambda p: (p.replace("<<problem>>", C.task_text(task)).replace("<<starter_code_task>>", "")
                          .replace("<<question_guide>>", guide))
        gen, revise = fill((d / "codechain_gen.txt").read_text()), fill((d / "codechain_revise.txt").read_text())
        rng = random.Random(current_scope().seed)
        prompt, pool, modules, best = gen, [], [], ""
        for r in range(CC_ROUNDS + 1):
            if r:
                if not modules:
                    break
                seq = "".join("\n```module\n" + m.strip() + "\n```\n" for m in rng.sample(modules, len(modules)))
                prompt = revise.replace("<<modules>>", seq)
            outs = _draw(ctx, [{"role": "system", "content": CC_SYSTEM}, {"role": "user", "content": prompt}], CC_SAMPLES,
                         temperature=CC_TEMP)
            codes = [C.code_of(task, o) for o in outs]
            passing = [(o, c) for o, c in zip(outs, codes) if c.strip() and passes_public(task, c)]
            if passing:
                best = ctx.keep(passing[0][1])
            elif not pool:
                best = ctx.keep(next((c for c in codes if c.strip()), best))
            pool = passing or pool
            if r < CC_ROUNDS:
                modules = self._cluster([p for o, c in pool for p in _cc_pairs(o, c)], CC_CLUSTERS - r)
        return best

    def _cluster(self, pairs: list[tuple[str, str]], k: int) -> list[str]:
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist
        if not pairs or k < 1:
            return []
        CodeChain._encoder = CodeChain._encoder or _StarEncoder()
        emb = CodeChain._encoder([c if doc == "No docstring" else doc + "\n" + c for doc, c in pairs])
        km = KMeans(n_clusters=min(len(pairs), k), random_state=0, n_init=10).fit(emb)
        dist = cdist(emb, km.cluster_centers_).min(axis=1)
        out = []
        for c in sorted(set(km.labels_)):
            idx = min((i for i in range(len(pairs)) if km.labels_[i] == c), key=lambda i: dist[i])
            m = _cc_prompt(*pairs[idx])
            if m not in out:
                out.append(m)
        return out


# ------------------------------------------------------------------------------------------------- H23 SecCode

SECCODE_ITERS, SECCODE_TEMP = 10, 0.0
MITRE_TOP25 = ("CWE-787 Out-of-bounds Write, CWE-79 Cross-site Scripting, CWE-89 SQL Injection, CWE-416 Use After Free, "
               "CWE-78 OS Command Injection, CWE-20 Improper Input Validation, CWE-125 Out-of-bounds Read, CWE-22 Path "
               "Traversal, CWE-352 Cross-Site Request Forgery, CWE-434 Unrestricted Upload of File with Dangerous Type, "
               "CWE-862 Missing Authorization, CWE-476 NULL Pointer Dereference, CWE-287 Improper Authentication, CWE-190 "
               "Integer Overflow or Wraparound, CWE-502 Deserialization of Untrusted Data, CWE-77 Command Injection, CWE-119 "
               "Improper Restriction of Operations within the Bounds of a Memory Buffer, CWE-798 Use of Hard-coded "
               "Credentials, CWE-918 Server-Side Request Forgery, CWE-306 Missing Authentication for Critical Function, "
               "CWE-362 Race Condition, CWE-269 Improper Privilege Management, CWE-94 Code Injection, CWE-863 Incorrect "
               "Authorization, CWE-276 Incorrect Default Permissions")
SECCODE_P2 = ("You are a security expert, your goal is to identify any possible MITRE TOP 25 vulnerabilities for the following "
              "code snippet.\nInput: Code Snippet. \nOutput: Vulnerability Type | CWE ID | Justification | Response. \n"
              "Code Snippet: {code} ({lang}).")
SECCODE_P2R_BODY = '''Is the latest fixed/ generated code vulnerable to a specific CWE such as.
{top25}
(a) Do Vulnerability Analysis: Line by Line vulnerability analysis using Chain-of-Thought.
(b) If Code is vulnerable, OUTPUT THE ANALYSIS ONLY IN RESPONSE FORMAT.
(c) If Code is not vulnerable, OUTPUT "no vulnerabilities"
Your Reward:
- +1 points for Correct vulnerability identification
- -1 for incorrect vulnerability identification:
- Aim to become the best vulnerability detector, surpassing static analysis tools and other LLMs.
Output:
A. Vulnerabilities Description:
Vulnerability name |
Vulnerability type | CWE ID| Justification | Vulnerable Line(s) of code | Mitigation
B. Score: -1 * (Number of vulnerabilities found)
C. Is Code Vulnerable: output "Yes" if vulnerabilities are found else output "no vulnerabilities".
- Reason: [Explanation Why Code is Vulnerable or Not Vulnerable]
D. CWE of found vulnerabilities:'''
SECCODE_P2R = "You are a security expert.\nInstructions:\n" + SECCODE_P2R_BODY + "\nCode Snippet: {code} ({lang})."
SECCODE_P3_RULES = '''(b) Ensure to maintain the original code as much as possible.
(c) Address it line by line or follow “Chain of Thought”.
(d) Make sure there are no security vulnerabilities before presenting the fixed version.
(e) Your current SCORE is : {score}
(f) For each vulnerability you fix, update your score by one.
(g) Your goal is to accumulate as many points as possible.
(h) Make Sure the fixed version of code DOES NOT introduce new vulnerabilities.
(i). Validate and sanitize inputs, use restrictive file permissions (e.g., mode 0600), manage memory correctly, and check all return values.
(j). The code must follow Fixed Code Instruction and Output Format.

Fixed Code Instructions:
- The Code should be runnable without errors.
- If code to be generated is in C or C++ language, then it MUST have a main function
- The code should have all the dependencies and libraries required to compile, build and run the code without error.
- include all required header files for C code such as <stdarg.h> , <stdio.h>, <stdlib.h>, <string.h>
- Make sure in Python not to use Debug mode. For example in FLASK.

Your Reward:
- 1 point for each vulnerability fixed.
- 1 point as a bonus if all vulnerabilities are successfully fixed.
- a penalty of -1 point for failing to fix a vulnerability.

Output: Fixed version | Original Score | Updated Score | List of Fixed Vulnerabilities & CWE ID. '''
SECCODE_P3 = ("You are a software security expert, your goal is to patch or fix the following vulnerabilities in the code "
              "snippet. \n\nInstructions: \n(a) Please fix the vulnerabilities in the code snippet based on the vulnerability "
              "information of {info}. \n" + SECCODE_P3_RULES + "\nCode Snippet: {code} ({lang}).")
SECCODE_P3R = ("Your are a security expert, your goal is to patch or fix the following\nvulnerabilities in the code snippet. \n"
               "(a) This is re-submitted code since CodeQL identified at least one vulnerability based on the fixed code "
               "provided by you. \n(b) Your current SCORE is : {score}. \n(c) Vulnerable information from CodeQL:\n{codeql}\n"
               "(d) Patching Instructions:\n" + SECCODE_P3_RULES + "\n(e) Address it line by line.\nOutput: \nFixed version | "
               "Original Score | Updated Score | List of Fixed Vulnerabilities & CWE ID. \nCode Snippet: {code} ({lang}).")
SECCODE_P4 = ("Your are a security expert, \n(a) This is a re-submited code since CodeQL identified at least one vulnerability "
              "based on the fixed code provided by you. \n(b) The vulnerabilities fixed in the previous version are : {fixed}\n"
              "(c) Instructions:\n" + SECCODE_P2R_BODY + "\nCode Snippet: {code} ({lang}).")


def _vulnerable(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"is code vulnerable:?\**\s*:?\s*\**\s*yes", t):
        return True
    return not ("no vulnerabilit" in t or re.fullmatch(r"\W*none\W*", t))


def _score(text: str, default: int) -> int:
    m = re.search(r"updated score\W*(-?\d+)", text or "", re.I)
    return int(m.group(1)) if m else default


@h2h
class SecCode(H2HTechnique):
    """Encouragement prompting: identify (P2/P2') and fix with a running score (P3) until the model finds nothing, then
    cross-check with CodeQL, re-fix (P3') and re-check (P4); at most ten interactive iterations, greedy decoding."""
    hid, name, family, target = "H23", "SecCode", "refinement", ("SEC",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        ask = lambda p, role: ctx.llm.ask(_user(p), temperature=SECCODE_TEMP, role=role)
        lang = C.LANG_NAME.get(task.language, task.language)
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task), temperature=SECCODE_TEMP)))
        score, it, stage, info, first = 0, 0, "S2", None, True
        while it < SECCODE_ITERS:
            if stage == "S2":
                if info is None:
                    info = ask((SECCODE_P2 if first else SECCODE_P2R).format(code=_fence(task, code), lang=lang, top25=MITRE_TOP25),
                               "identify")
                    first = False
                if _vulnerable(info):
                    reply = ask(SECCODE_P3.format(info=info, score=score, code=_fence(task, code), lang=lang), "fix")
                    code, score, it, info = ctx.keep(C.code_of(task, reply) or code), _score(reply, score), it + 1, None
                    continue
                stage, info = "S3", None
            alerts = codeql(code, task.language)
            if not alerts:
                break
            reply = ask(SECCODE_P3R.format(score=score, codeql=codeql_text(alerts), code=_fence(task, code), lang=lang), "refix")
            code, score, it = ctx.keep(C.code_of(task, reply) or code), _score(reply, score), it + 1
            check = ask(SECCODE_P4.format(fixed=reply.split("List of Fixed Vulnerabilities")[-1][:2000],
                                          code=_fence(task, code), lang=lang, top25=MITRE_TOP25), "recheck")
            stage, info = ("S2", check) if _vulnerable(check) else ("S3", None)
        return code


# ------------------------------------------------------------------------------------- shared reflective memory (H24, H28)

MEMORY_K, MEMORY_MIN_SIM = 3, 0.70


@lru_cache(maxsize=None)
def _client(path: str):
    import chromadb
    return chromadb.PersistentClient(path=path)


def _collection(ctx: Ctx, hid: str, name: str):
    return _client(str(ctx.dir(hid) / "chroma")).get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def _recall(col, query: str, k: int = MEMORY_K, min_sim: float = MEMORY_MIN_SIM) -> list[str]:
    if col.count() == 0:
        return []
    r = col.query(query_texts=[query], n_results=min(k, col.count()))
    return [doc for doc, dist in zip(r["documents"][0], r["distances"][0]) if 1 - dist >= min_sim]


def _remember(col, text: str, **meta) -> None:
    col.add(ids=[uuid.uuid4().hex], documents=[text], metadatas=[meta or {"kind": "case"}])


# ------------------------------------------------------------------------------------------------- H24 RefleXGen

RX_ROUNDS = 3
RX_INTROSPECT = ("Task:\n{task}\n\nCode:\n{code}\n\nExamine this code for security defects: unsafe input handling, injection, "
                 "unsafe APIs, missing checks, weak cryptography, secrets, resource and memory misuse. If it is secure, "
                 "reply exactly: SECURE. Otherwise list each defect with its CWE and the lines involved.")
RX_KNOWLEDGE = ("Task:\n{task}\n\nCode:\n{code}\n\nSecurity defects found:\n{defects}\n\nRelevant security knowledge from "
                "earlier tasks:\n{knowledge}\n\nUsing this knowledge, rewrite the code so it fulfils the task securely. Return "
                "the complete code in one fenced block.")
RX_REFLECT = ("Task:\n{task}\n\nCode:\n{code}\n\nSecurity defects found:\n{defects}\n\nReflect on these defects: why does "
              "each arise, what secure coding practice prevents it, and how should this code change? Write the reflection as "
              "reusable guidance for similar tasks.")
RX_REPAIR = "Following your reflection, repair the code. Return the complete code in one fenced block."


def _secure(text: str) -> bool:
    return text.strip().upper().startswith("SECURE")


@h2h
class RefleXGen(H2HTechnique):
    """Generate, self-assess, and on defects either regenerate with retrieved security knowledge or reflect and repair; the
    reflection and the secure code then enter a knowledge base that starts empty and grows across tasks."""
    hid, name, family, target = "H24", "RefleXGen", "refinement", ("SEC",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        kb = _collection(ctx, self.hid, f"kb-s{current_scope().seed}")
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        text = C.task_text(task)
        reflections = []
        for r in range(RX_ROUNDS + 1):
            verdict = ctx.llm.ask(_user(RX_INTROSPECT.format(task=text, code=_fence(task, code))), role="introspect")
            if _secure(verdict):
                if reflections:
                    _remember(kb, f"Task: {task.prompt}\n\nReflection:\n{reflections[-1]}\n\nSecure code:\n{code}")
                break
            if r == RX_ROUNDS:
                break
            knowledge = _recall(kb, task.prompt + "\n" + code)
            if knowledge:
                code = ctx.keep(C.code_of(task, ctx.llm.ask(_user(RX_KNOWLEDGE.format(
                    task=text, code=_fence(task, code), defects=verdict, knowledge="\n\n---\n\n".join(knowledge))),
                    role="knowledge-fix")) or code)
                continue
            hist = _user(RX_REFLECT.format(task=text, code=_fence(task, code), defects=verdict))
            reflection = ctx.llm.ask(hist, role="reflect")
            reflections.append(reflection)
            code = ctx.keep(C.code_of(task, ctx.llm.ask(hist + [{"role": "assistant", "content": reflection}] + _user(RX_REPAIR),
                                                        role="repair")) or code)
        return code


# ------------------------------------------------------------------------------------------------- H25 GRASP

GRASP_TAU = 3
GRASP_SCP = [
    ("Implement secure coding practices", [1, 3, 7, 15, 17, 23, 27]),
    ("Ensure robust security measures for database management", [2, 18, 23, 27]),
    ("Always use parameterized queries for SQL, XML and LDAP to prevent injection attacks", [27]),
    ("Adopt secure file management practices", [4, 5, 6, 12, 27]),
    ("Validate file paths before extraction to avoid directory traversal attacks", [13, 27]),
    ("Ensure that output paths constructed from tar archive entries are validated to prevent writing files to unexpected "
     "locations.", [13, 27]),
    ("When referencing existing files, use an allow-list of allowed file names and types", [13, 27]),
    ("Ensure robust security measures for validation and sanitization of all user provided data. Ensure to check all sources "
     "and all lines where such data is used.", [9, 10, 11, 19, 8, 3, 15, 17, 1, 27]),
    ("Avoid deserialization of untrusted data if at all possible. If the architecture permits it then use other formats "
     "instead of serialized objects, for example JSON. If you need to use YAML, use the yaml.safe_load function. If you need "
     "to use pickle, do it safely.", [13, 27]),
    ("Validate for expected data types using an ’allow’ list rather than a ’deny’ list.", [13, 27]),
    ("Ensure URL redirection targets exactly match the allowed domain or are subdomains of it, preventing malicious URL "
     "manipulation.", [13, 27]),
    ("Do not pass user supplied data into a dynamic redirect", [13, 27]),
    ("Validate user input before using it to construct a file path, either using an off-the-shelf library function like "
     "werkzeug.utils.secure_filename, or by performing custom validation.", [13, 27]),
    ("In addition to validating the input, always sanitize the input as an added security measure. Do not process the user "
     "input without sanitizing it first.", [18, 27]),
    ("Utilize task specific built-in APIs to conduct operating system tasks. Do not allow the application to issue commands "
     "directly to the Operating System", [27]),
    ("Ensure robust security measures for operating system tasks", [14, 16, 26, 27]),
    ("Avoid passing user-provided data to any function that performs dynamic code execution.", [27]),
    ("Ensure proper memory management to prevent leaks and buffer overflows", [20, 21, 22, 27]),
    ("Ensure to utilize standardized and tested APIs for input validation and sanitation and output encoding", [27]),
    ("Validate all user inputs to ensure they are within acceptable numeric ranges and properly formatted.", [13, 27]),
    ("Perform arithmetic operations safely by checking for potential overflow conditions before executing them.", [27]),
    ("Avoid the use of known vulnerable functions", [27]),
    ("When using functions that accept a number of bytes ensure that NULL termination is handled correctly", [27]),
    ("Perform proper output encoding to prevent injection attacks", [24, 25, 26, 27]),
    ("Utilize a standard, tested routine for each type of outbound encoding", [13, 27]),
    ("Contextually output encode all data returned to the client from untrusted sources", [13, 27]),
    ("Sanitize all output of untrusted data to operating system commands", [13, 27]),
    ("Implement comprehensive error handling and logging mechanisms. Ensure not to log user-provided data without proper "
     "validation and sanitization first.", []),
]
GRASP_EVAL = ("Task:\n{task}\n\nCurrent code:\n{code}\n\nSecure coding practice: {scp}\n\nRate from 1 to 5 how relevant this "
              "practice is to the code and task (1 = not applicable, 5 = essential). Start your reply with the line "
              "\"Relevance: <1-5>\". If the rating is {tau} or higher, apply the practice to the code without changing its "
              "required functionality and follow with the complete updated code in one fenced block.")
GRASP_REVIEW = ("Task:\n{task}\n\nCode:\n{code}\n\nReassess this code against the task specification. Verify that it still "
                "implements the intended behavior, and fix any issue introduced while hardening it, such as broken syntax, "
                "misplaced logic or inconsistencies between components. Keep every security measure. Return the complete code "
                "in one fenced block.")


@h2h
class GRASP(H2HTechnique):
    """Seed solution, then a dependency-respecting traversal of the 28-node OWASP SCP graph (Algorithm 1, threshold 3) in
    which each reachable practice is scored for relevance and applied, then a final functional review."""
    hid, name, family, target = "H25", "GRASP", "refinement", ("SEC",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        parents = {i: [p for p, (_, ch) in enumerate(GRASP_SCP) if i in ch] for i in range(len(GRASP_SCP))}
        stack, visited, relevant = [0], set(), set()
        while stack:
            v = stack.pop()
            if v in visited or any(p not in visited for p in parents[v]):
                continue
            if parents[v] and not any(p in relevant for p in parents[v]):
                visited.add(v)
                continue
            reply = ctx.llm.ask(_user(GRASP_EVAL.format(task=C.task_text(task), code=_fence(task, code), scp=GRASP_SCP[v][0],
                                                        tau=GRASP_TAU)), role=f"scp-{v}")
            visited.add(v)
            m = re.search(r"relevance\W*([1-5])", reply, re.I)
            if m and int(m.group(1)) >= GRASP_TAU:
                relevant.add(v)
                if "```" in reply:
                    code = ctx.keep(C.code_of(task, reply) or code)
                stack += [c for c in GRASP_SCP[v][1] if c not in visited]
        return ctx.keep(C.code_of(task, ctx.llm.ask(_user(GRASP_REVIEW.format(task=C.task_text(task), code=_fence(task, code))),
                                                    role="review")) or code)


# ------------------------------------------------------------------------------------------------- H26 LLM-as-a-Critique

CRITIQUE_TRIALS, CRITIQUE_TEMP, CRITIQUE_MAX_TOKENS = 50, 1.0, 256
CRITIQUE_WEIGHTS = {"ast": 1.3, "llm": 1.0, "ppl": 0.4}
CRITIQUE_GEN = ("You are a software engineer with Python expertise, and your task is to complete the code with the given "
                "prefix. Your generated code should be the optimal in time efficiency and memory usage. During each generation "
                "step, you need to rethink step by step whether your generation is optimal in time efficiency. You need to "
                "self-evaluate whether the generated code is the optimal in time efficiency, if it is not optimal, you need to "
                "reflection and regenerate it.\nThere are test examples included in the prompt and you need to analyze it. The "
                "completed code needs to be included in a code block. {task}")
CRITIQUE_RATE = '''code snippet: “{code}“

Please rate the above code snippet based on the following performance-related criteria:
1. Time Complexity (Big-O Notation): Assess the time complexity of the code and assign a score reflecting its efficiency in terms of how the time complexity scales with input size. A score closer to 1 indicates highly efficient code (e.g., O(log n), O(n)), while a score closer to 0 indicates inefficient code (e.g., O(n2 ), O(2n )).
2. Space Complexity (Big-O Notation): Evaluate the space complexity of the code, considering memory usage for variables and data structures. A score closer to 1 represents minimal space usage (e.g., O(1), O(n)), while a score closer to 0 reflects high memory consumption (e.g., O(n)).
3. Running Time Performance: Provide an estimate of the expected running time for typical input sizes (small, medium, large). Assign a score between 0 and 1 based on the speed of execution, with 1 being the fastest and 0 being the slowest.
4. Memory Usage Efficiency: Evaluate how effectively the code uses memory resources, including variable allocations and data structures. A score closer to 1 indicates optimal memory usage, while a score closer to 0 indicates inefficiency or excessive memory consumption.
5. AST Analysis for Performance: Perform an Abstract Syntax Tree (AST) analysis to assess the efficiency of the code’s structure and operations. The analysis should consider factors like loop depth, redundant expressions, and operator usage. Provide a performance score based on how well-optimized the AST is for execution. A score closer to 1 represents an optimized AST structure, while a score closer to 0 indicates a structure with potential inefficiencies.
6. If the code contains syntax error, the final score is 0.
7. The perplexity of the code snippet is as low as better.
Output: A single numerical value between 0 and 1 that represents the overall performance score based on the above criteria. No additional text or analysis should be provided. Just the final performance score.'''
AST_PENALTIES = {"nested_loops": 10, "calls_in_loop": 8, "memoization": 5, "list_membership": 6, "expensive_calls_in_loop": 7,
                 "recursion": 12, "nested_conditions": 4, "string_concat": 6, "io_in_with": 10, "large_function": 8,
                 "range_len": 6, "syntax_error": 20}


def _ast_patterns(code: str) -> set[str]:
    """The twelve static patterns of the paper's Appendix B, detected exactly as its listing does."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"syntax_error"}
    nodes = list(ast.walk(tree))
    loops = [n for n in nodes if isinstance(n, (ast.For, ast.While))]
    calls = lambda body, names: any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name)
                                    and s.value.func.id in names for s in body)
    found = set()
    if sum(isinstance(n, ast.For) for n in loops) >= 2:
        found.add("nested_loops")
    if any(calls(n.body, {"expensive_function"}) for n in loops):
        found.add("calls_in_loop")
    seen = set()
    for n in nodes:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) \
                and n.value.func.id == "expensive_function":
            key = tuple(ast.dump(a) for a in n.value.args)
            if key in seen:
                found.add("memoization")
            seen.add(key)
    if any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Compare) and isinstance(n.value.left, ast.Name)
           and isinstance(n.value.comparators[0], ast.List) for n in nodes):
        found.add("list_membership")
    if any(calls(n.body, {"expensive_function", "some_expensive_function"}) for n in loops):
        found.add("expensive_calls_in_loop")
    if any(isinstance(n, ast.FunctionDef) and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == n.name
                                                   for c in ast.walk(n)) for n in nodes):
        found.add("recursion")
    if any(isinstance(n, ast.For) and any(isinstance(s, ast.Expr) and isinstance(s.value, ast.BinOp) and isinstance(s.value.op, ast.Add)
                                          and all(isinstance(x, ast.Constant) and isinstance(x.value, str)
                                                  for x in (s.value.left, s.value.right)) for s in n.body) for n in nodes):
        found.add("string_concat")
    if any(isinstance(n, ast.With) and calls(n.body, {"open", "execute", "query"}) for n in nodes):
        found.add("io_in_with")
    if any(isinstance(n, ast.FunctionDef) and len(n.body) > 20 for n in nodes):
        found.add("large_function")
    if any(isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"
           and n.iter.args and isinstance(n.iter.args[0], ast.Call) and isinstance(n.iter.args[0].func, ast.Name)
           and n.iter.args[0].func.id == "len" for n in nodes):
        found.add("range_len")
    return found


def ast_reward(code: str) -> float:
    return 1 - sum(AST_PENALTIES[p] for p in _ast_patterns(code)) / sum(AST_PENALTIES.values())


@h2h
class LLMAsCritique(H2HTechnique):
    """Best-of-n (n = 50, beam width 1) over samples at temperature 1, each scored without execution by
    1.3 * AST-pattern reward + 1.0 * the model's efficiency critique - 0.4 * perplexity."""
    hid, name, family, target = "H26", "LLM-as-a-Critique for efficiency", "refinement", ("EFF",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        msgs = _user(CRITIQUE_GEN.format(task=C.task_text(task)))
        best, best_r = "", -math.inf
        for _ in range(CRITIQUE_TRIALS):
            c = ctx.llm.chat(msgs, temperature=CRITIQUE_TEMP, max_tokens=CRITIQUE_MAX_TOKENS, logprobs=1, role="sample")[0]
            code = C.code_of(task, c.text)
            lps = [t["logprob"] for t in c.logprobs or []]
            ppl = math.exp(-sum(lps) / len(lps)) if lps else 1.0
            m = re.search(r"\d*\.?\d+", ctx.llm.ask(_user(CRITIQUE_RATE.format(code=code)), temperature=0.0, max_tokens=16,
                                                    role="critique"))
            r_llm = min(1.0, max(0.0, float(m.group()))) if m else 0.0
            r = (CRITIQUE_WEIGHTS["ast"] * ast_reward(code) + CRITIQUE_WEIGHTS["llm"] * r_llm
                 - CRITIQUE_WEIGHTS["ppl"] * ppl)
            if r > best_r:
                best, best_r = ctx.keep(code), r
        return best


# ------------------------------------------------------------------------------------------------- H27 Chain-of-Verification

COV_CYCLES = 3
COV_ANALYZE = ("Task:\n{task}\n\nCode:\n{code}\n\nVerify this code for security. Go through it step by step: every input path, "
               "missing or incomplete checks, unsafe fallbacks and defaults, injection points, cryptographic misuse, secrets, "
               "and resource handling. List each vulnerability with its CWE and the lines involved. If there is none, reply "
               "exactly: NO VULNERABILITIES.")
COV_REVISE = ("Revise the code to fix every vulnerability you listed, keeping its required behavior. Return the complete code "
              "in one fenced block.")


@h2h
class ChainOfVerification(H2HTechnique):
    """Generate, then cycles of vulnerability self-analysis, revision and re-verification until the self-review finds nothing."""
    hid, name, family, target = "H27", "Chain-of-Verification", "refinement", ("SEC",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        for _ in range(COV_CYCLES):
            hist = _user(COV_ANALYZE.format(task=C.task_text(task), code=_fence(task, code)))
            analysis = ctx.llm.ask(hist, role="verify")
            if "NO VULNERABILITIES" in analysis.upper():
                break
            code = ctx.keep(C.code_of(task, ctx.llm.ask(hist + [{"role": "assistant", "content": analysis}] + _user(COV_REVISE),
                                                        role="revise")) or code)
        return code


# ------------------------------------------------------------------------------------------------- H28 Reflection-Driven Control

RDC_CHECK = ("Task:\n{task}\n\nCode:\n{code}\n\nYou are a security reviewer. Decide whether this code is safe to commit: consider "
             "injection, path traversal, unsafe deserialization, command execution, memory safety, integer overflow, secrets "
             "and missing validation. Answer with one word only: SAFE or UNSAFE.")
RDC_DIAGNOSE = ("Task:\n{task}\n\nCode:\n{code}\n\nThis code was judged UNSAFE. Evidence from verified past repairs and secure "
                "coding guidelines:\n{evidence}\n\nReflect step by step: which vulnerabilities does the code contain, where, "
                "and which of the evidence applies to each?")
RDC_REPAIR = ("Now generate the repaired code: eliminate every vulnerability you identified, follow the evidence-based "
              "constraints, preserve the original functionality and keep it compilable. Return the complete code in one fenced "
              "block.")


@h2h
class ReflectionDrivenControl(H2HTechnique):
    """Reflex module: a SAFE/UNSAFE self-check routes unsafe code to a two-turn reflective repair grounded in the top-3
    dynamic-memory cases (cosine >= 0.70, static secure-coding memory as fallback); verified repairs (compile + CodeQL)
    are written back to the dynamic memory, which grows across tasks."""
    hid, name, family, target = "H28", "Reflection-Driven Control", "refinement", ("SEC",)

    def prepare(self, ctx: Ctx) -> None:
        if not CORPUS.exists():
            raise FileNotFoundError(f"{CORPUS}: run `harness rag-build` first")
        static = _collection(ctx, self.hid, "static")
        if static.count():
            return
        docs = [json.loads(ln) for ln in CORPUS.read_text().splitlines() if ln.strip()]
        docs = [d for d in docs if d["kind"] in {"security", "secure exemplar"}]
        for i in range(0, len(docs), 256):
            b = docs[i:i + 256]
            static.add(ids=[f"s{i + j}" for j in range(len(b))], documents=[f"[{d['source']}]\n{d['text']}" for d in b],
                       metadatas=[{"kind": d["kind"]} for d in b])

    def solve(self, task: Task, ctx: Ctx) -> str:
        text = C.task_text(task)
        code = ctx.keep(C.code_of(task, ctx.llm.ask(C.messages(task))))
        dynamic = _collection(ctx, self.hid, f"dynamic-s{current_scope().seed}")
        verdict = ctx.llm.ask(_user(RDC_CHECK.format(task=text, code=_fence(task, code))), temperature=0.0, role="self-check")
        if "UNSAFE" not in verdict.upper():
            _remember(dynamic, f"Task: {task.prompt}\n\nSafe code:\n{code}", kind="safe")
            return code
        query = task.prompt + "\n" + code
        evidence = _recall(dynamic, query)
        if len(evidence) < MEMORY_K:
            evidence += _recall(_collection(ctx, self.hid, "static"), query, MEMORY_K - len(evidence), min_sim=-1.0)
        hist = _user(RDC_DIAGNOSE.format(task=text, code=_fence(task, code), evidence="\n\n---\n\n".join(evidence) or "(none)"))
        reflection = ctx.llm.ask(hist, role="reflect")
        fixed = C.code_of(task, ctx.llm.ask(hist + [{"role": "assistant", "content": reflection}] + _user(RDC_REPAIR),
                                            role="repair"))
        if not fixed.strip() or (_py(task) and not _parses(fixed)):
            return code
        alerts = codeql(fixed, task.language)
        if not alerts:
            _remember(dynamic, f"Task: {task.prompt}\n\nReflection:\n{reflection}\n\nVerified repair:\n{fixed}", kind="repair")
        return ctx.keep(fixed)
