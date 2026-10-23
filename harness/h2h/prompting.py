"""E12 prompting family (H01-H10)."""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import random
import re
import subprocess
import tempfile
import textwrap
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .. import sandbox
from ..tasks import Task, Test
from ..techniques import common as C
from ..techniques.rag import ngrams
from .base import Ctx, H2HTechnique, h2h
from .retrieval import (CWE_CSV, cwe_catalog, decontaminate, draft, embed, fetch, json_of, load_index, pie_hq,
                        pie_python, prep, read_jsonl, save_index, top_k, user, write_jsonl)

MODEL_NAME = "Gemma 4 31B IT"


def fresh(demos: list[str], task: Task) -> list[str]:
    """Few-shot demonstrations minus any that shares a 13-gram with the task (some published demos are benchmark tasks)."""
    g = ngrams(task.prompt)
    return [d for d in demos if not (ngrams(d) & g)]


def with_steps(task: Task, steps: str, comment: str = "") -> str:
    """Task text with reasoning steps placed inside the function docstring (as SCoT and Self-planning do), else appended."""
    p = task.prompt.rstrip()
    for q in ('"""', "'''"):
        if task.mode == "function" and p.endswith(q) and p.count(q) >= 2:
            indent = re.search(r"\n([ \t]*)" + re.escape(q) + r"$", p)
            pad = indent.group(1) if indent else "    "
            body = "\n".join(pad + l for l in steps.strip().splitlines())
            p = f"{p[:-3].rstrip()}\n{body}\n{pad}{q}\n" + (f"{pad}# {comment}\n" if comment else "")
            return C.task_text(Task(**{**task.__dict__, "prompt": p}))
    return C.task_text(task) + f"\n\n{steps.strip()}" + (f"\n\n# {comment}" if comment else "")


# ---------- H01 SCoT ----------

SCOT_ASK = ("Please understand the requirement and write a rough solving process. It starts with a input-output structure. "
            "You should use three basic structures to build the solving process, including sequences, branches, and loops. "
            "The necessary details should be written in natural languages.")
SCOT_CHECK = "Please check the above solving process and write a code based on it. Note that the solving process may contain errors."
SCOT_DEMOS = [
    ('def first_Repeated_Char(str):\n    """\n    Write a python function to find the first repeated character in a given string.\n    """\n    pass',
     "Input: str: a string\nOutput: ch: a repeated character in str\n1: for each character ch in str:\n2:   if ch appears more than once in str:\n3:     return ch\n4: return None",
     "    h = {}\n    for ch in str:\n      if ch in h:\n        return ch;\n      else:\n        h[ch] = 0\n    return None"),
    ('def count_vowels(text):\n    """\n    Write a function to count the number of vowels in a given string.\n    """\n    pass',
     "Input: text: a string\nOutput: count: an integer\n1: Initialize count to 0\n2: for each character ch in text:\n3:   if ch is a vowel (a, e, i, o or u, in either case):\n4:     increment count\n5: return count",
     '    count = 0\n    for ch in text:\n        if ch.lower() in "aeiou":\n            count += 1\n    return count'),
    ('def max_subarray_sum(nums):\n    """\n    Write a function to find the largest sum of a contiguous sub-array of a non-empty list of integers.\n    """\n    pass',
     "Input: nums: a list of integers\nOutput: best: an integer\n1: Initialize best and current to the first element of nums\n2: for each number x in nums after the first:\n3:   set current to the larger of x and current + x\n4:   if current is larger than best:\n5:     set best to current\n6: return best",
     "    best = current = nums[0]\n    for x in nums[1:]:\n        current = max(x, current + x)\n        if current > best:\n            best = current\n    return best"),
]
SCOT_TEMPERATURE = 0.8


def _scot_demo_code(sig: str, scot: str, code: str) -> str:
    head = sig.split('"""')[0] + '"""\n    ' + sig.split('"""')[1].strip() + "\n"
    return head + "".join(f"    {l}\n" for l in scot.splitlines()) + f'    """\n    # {SCOT_CHECK}\n{code}'


@h2h
class SCoT(H2HTechnique):
    """Two calls: three MBPP-style demonstrations elicit a structured chain of thought (input-output, sequence, branch,
    loop; sampled at T=0.8), then the code is written from it greedily, the SCoT placed in the docstring."""
    hid, name, family, target = "H01", "SCoT prompting", "prompting", ("RM",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        demos = [(s, t, c) for s, t, c in SCOT_DEMOS if fresh([s], task)]
        p1 = "\n\n".join(f"{s}\n\n{SCOT_ASK}\n\n{t}" for s, t, _ in demos) + f"\n\n{task.prompt}\n\n{SCOT_ASK}"
        scot = ctx.llm.ask(user(p1), temperature=SCOT_TEMPERATURE, role="scot")
        shots = "\n\n".join(f"```python\n{_scot_demo_code(s, t, c)}\n```" for s, t, c in demos)
        p2 = f"{shots}\n\n{with_steps(task, scot, SCOT_CHECK)}"
        return C.code_of(task, ctx.llm.ask(user(p2, C.SYSTEM), temperature=0, role="code"))


# ---------- H02 Self-planning ----------

PLAN_DEMOS = [
    """def encrypt(s):
  '''
  Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being rotated. The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
  For example:
  encrypt('hi') returns 'lm'
  encrypt('asdfghjkl') returns 'ewhjklnop'
  encrypt('gf') returns 'kj'
  encrypt('et') returns 'ix'
  1. Create a alphabet, bias two places multiplied by two.
  2. Loop the input, find the latter bias letter in alphabet.
  3. Return result.
  '''""",
    """def check_if_last_char_is_a_letter(txt):
  '''
  Create a function that returns True if the last character of a given string is an alphabetical character and is not a part of a word, and False otherwise. Note: "word" is a group of characters separated by space.
  Examples:
  check_if_last_char_is_a_letter("apple pie") → False
  check_if_last_char_is_a_letter("apple pi e") → True
  check_if_last_char_is_a_letter("apple pi e ") → False
  check_if_last_char_is_a_letter("") → False
  1. If the string is empty, return False.
  2. If the string does not end with a alphabetical character, return False.
  3. Split the given string into a list of words.
  4. Check if the length of the last word is equal to 1.
  '''""",
    """def file_name_check(file_name):
  '''
  Create a function which takes a string representing a file's name, and returns 'Yes' if the the file's name is valid, and returns 'No' otherwise. A file's name is considered to be valid if and only if all the following conditions are met: - There should not be more than three digits ('0'-'9') in the file's name. - The file's name contains exactly one dot '.' - The substring before the dot should not be empty, and it starts with a letter from the latin alphapet ('a'-'z' and 'A'-'Z'). - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
  Examples:
  file_name_check("example.txt") => 'Yes'
  file_name_check("1example.dll") => 'No' (the name should start with a latin alphapet letter)
  1. Check if the file name is valid according to the conditions.
  2. Return "Yes" if valid, otherwise return "NO".
  '''""",
    """def fruit_distribution(s,n):
  '''
  In this task, you will be given a string that represents a number of apples and oranges that are distributed in a basket of fruit this basket contains apples, oranges, and mango fruits. Given the string that represents the total number of the oranges and apples and an integer that represent the total number of the fruits in the basket return the number of the mango fruits in the basket.
  for examble:
  fruit_distribution("5 apples and 6 oranges", 19) ->19 - 5 - 6 = 8
  fruit_distribution("0 apples and 1 oranges",3) -> 3 - 0 - 1 = 2
  fruit_distribution("2 apples and 3 oranges", 100) -> 100 - 2 - 3 = 95
  fruit_distribution("100 apples and 1 oranges",120) -> 120 - 100 - 1 = 19
  1. Extract the numbers of oranges and apples from given string.
  2. Calculate the sum of oranges and apples.
  3. Deduct the sum from the total number of fruits.
  4. Return the number of mangoes.
  '''""",
    """def prime_fib(n: int):
  '''
  prime_fib returns n-th number that is a Fibonacci number and it's also prime.
  Examples:
  >>> prime_fib(1) 2
  >>> prime_fib(2) 3
  >>> prime_fib(3) 5
  >>> prime_fib(4) 13
  >>> prime_fib(5) 89
  1. Create a function to check if a number is prime.
  2. Generate a Fibonacci sequence.
  3. Check if each number in the Fibonacci sequence is prime, decrement the counter.
  4. If the counter is 0, return the Fibonacci number.
  '''""",
    """def compare_one(a, b):
  '''
  Create a function that takes integers, floats, or strings representing real numbers, and returns the larger variable in its given variable type. Return None if the values are equal. Note: If a real number is represented as a string, the floating point might be . or ,
  Examples:
  compare_one(1, 2.5) → 2.5
  compare_one(1, "2,3") → "2,3"
  compare_one("5,1", "6") → "6"
  compare_one("1", 1) → None
  1. Store the original inputs.
  2. Check if inputs are strings and convert to floats.
  3. Compare the two inputs and return the larger one in its original data type.
  '''""",
    """def sort_even(l: list):
  '''
  This function takes a list l and returns a list l' such that l' is identical to l in the odd indicies, while its values at the even indicies are equal to the values of the even indicies of l, but sorted.
  Examples:
  >>> sort_even([1, 2, 3])
  [1, 2, 3]
  >>> sort_even([5, 6, 3, 4])
  [3, 6, 5, 4]
  1. Create a list of all the even indices of the given list.
  2. Sort the list of even indices.
  3. Return a new list that is identical to the original list in the odd indicies, and equal to the sorted even indices in the even indicies.
  '''""",
    """def search(lst):
  '''
  You are given a non-empty list of positive integers. Return the greatest integer that is greater than zero, and has a frequency greater than or equal to the value of the integer itself. The frequency of an integer is the number of times it appears in the list. If no such a value exist, return -1.
  Examples:
  search([4, 1, 2, 2, 3, 1]) == 2
  search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
  search([5, 5, 4, 4, 4]) == -1
  1. Create a frequency dict.
  2. Sort the input list.
  3. Loop the input list, if frequency no lesser than the integer, set result.
  4. Return the result.
  '''""",
]
PLAN_ASK = "Write only the numbered plan for the last task, in the same format as the examples."


@h2h
class SelfPlanning(H2HTechnique):
    """Two-phase: an 8-shot prompt (HumanEval intents with their plans) makes the model write a concise numbered plan,
    then the code is generated from the intent plus the plan with no demonstrations; both greedy."""
    hid, name, family, target = "H02", "Self-planning", "prompting", ("RM",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        shots = "\n\n".join(fresh(PLAN_DEMOS, task))
        plan = ctx.llm.ask(user(f"{shots}\n\n{task.prompt}\n\n{PLAN_ASK}"), temperature=0, role="plan")
        plan = "\n".join(l for l in plan.splitlines() if re.match(r"\s*\d+\.", l)) or plan
        return C.code_of(task, ctx.llm.ask(user(with_steps(task, plan), C.SYSTEM), temperature=0, role="implement"))


# ---------- H03 LLM4EFFI ----------

EFFI_FORMALIZE_TRIES, EFFI_ALGOS, EFFI_TESTS, EFFI_MIN_TESTS, EFFI_ALGO_TRIES = 5, 5, 20, 5, 3
EFFI_TEMPERATURE = 0.0
EFFI_FORMALIZE = """    As a professional algorithm engineer, please analyze this algorithm problem according to the following categories.Do not generate any example implementation:
1. Entry point function name:
2. Input/Output conditions
3. Edge Cases and Parameters type(Int String...)
4. expected behavior"""
EFFI_FORMALIZE_USER = "        The algorithm problem description is as follows: {}"
EFFI_CHECK = """    As an excellent algorithm engineer, please analyze whether the explanation of the problem matches the original requirements of the problem. If they are consistent, output "Yes"; if they are not consistent, output "No" along with the reason, as shown below:
{"Yes":"NULL"}
{"No":"The reason is"}"""
EFFI_CHECK_USER = """    Here is the original problem content: {}
Here is the explanation of the problem description: {}"""
EFFI_ALGORITHMS = """As a professional algorithm engineer, you can effectively design multiple algorithms to solve the problem with low time complexity and output them in pseudo algorithm format, and pseudo algorithm is a nonlinear, high-level programming language for algorithmic logic. It combines natural language and programming structures to express the steps and sums of algorithms. The main purpose of process algorithms is to clearly display the core ideas and logic of the algorithm without relying on specific programming language syntax. Please design an 5 excellent algorithm solution based on the problem description provided. The time complexity of the algorithm needs to be as small as possible, and try to output 5 algorithms  in the form of a pseudo-algorithm in the following format:
    PS: DO NOT provide implementation example!
""" + "".join(f"```algorithm{i}\n{{algorithm key description:this algorithm using xxx,the key is to make sure xxx}}\n{{ pseudo algorithm: ..}}\n```\n"
              for i in range(1, 6))
EFFI_SUGGEST = """As a professional Python algorithm programming expert, please provide suggestions for improving code efficiency based on the potential inefficiencies mentioned above. For example:
1.	Using xxx instead of xxx can significantly improve code efficiency.
Please provide at least 20 suggestions."""
EFFI_CODE = """As a professional Python algorithm engineer, please convert the selected algorithm into corresponding Python code. Ensure the code is complete and well-formatted. When converting to a standardized format, be sure to follow the guidelines specified in the “original question format”:
1.	If “from typing import List” appears in the original question, please retain it.
2.	Use the same function name as given in the original question format; do not rename it.
3.	You may incorporate practical optimization details drawn from the knowledge base.
The final output format should be as follows:
```python
{code}
```"""
EFFI_DESIGNER = """    As a Python tester, your task is to create comprehensive test inputs for the function given the deinition and docstring. These test inputs should be Edge scenarios to ensure the code's robustness and reliability.Plese  output {} edge test cases which cover all required parameters type in a single line and start with a `input:`.
    PS: DON't generate too long!!!"""
EFFI_DESIGNER_USER = '''    EXAMPLES:
Function:
```python
from typing import *
def find_the_median(arr: List[int]) -> float:
 """
 Given an unsorted array of integers `arr`, find the median of the array.
 The median is the middle value in an ordered list of numbers.
 If the length of the array is even, then the median is the average of the two middle numbers.
 """
```
Test Inputs(OUTPUT format):
```text
input: [1]
input: [-1, -2, -3, 4, 5]
input: [4, 4, 4]
input: [....]
input: [....]
```
END OF EXAMPLES.
Function:
{}'''
EFFI_CALCULATOR = "        As a Python programmer, your task is to calculate all the test outputs and write the test case statement corresponding to the test input for the function given the definition and docstring. Write One test cases with a single line of assert statement."
EFFI_CALCULATOR_USER = '''EXAMPLES:
Function:
```python
from typing import List
def find_the_median(arr: List[int]) -> float:
    """ Given an unsorted array of integers `arr`, find the median of the array. The median is the middle value in an ordered list of numbers. If the length of the array is even, then the median is the average of the two middle numbers.
    """
```
Test Input:
```text
input: [1, 3, 2, 5]
...
```
Test Case:
```python
assert find_the_median([1, 3, 2, 5]) == 2.5
```
END OF EXAMPLES.
FUNCTION:
{}
```
Test Input:
```text
{}
```
```python
assert....
```'''
EFFI_REVIEW = """As a professional algorithm programming master, please help me step by step check whether the input and output of this test case meet the requirements of the algorithm question based on the question. If yes, please output "Yes", otherwise output "No\""""
EFFI_REVIEW_USER = """    This is the requirements:{}
    This is the wrong case:{}"""
EFFI_REFINE = """As a professional code programming algorithm expert, your task is to correct the code and ensure that the code is fixed without impacting its time complexity or practical efficiency. Then I will provide you with specific code and test cases. Based on the algorithm requirements:
If the code is correct but the test case is wrong, please output the unmodified code directly.
If the test case is correct but the code is wrong, please modify the code.
Important Notes:
1.	Do not alter the algorithm itself—maintain minimal time complexity.
2.	Do not change the format, such as the function name.
3.	Please output in the specified format.
3.  Ensure there are no syntax errors.
please output in this format:
    ```python
    {code}
    ```"""
EFFI_REFINE_USER = """    Don't change the algorithm!
    This is the algorithm problem description:{}
    This is the wrong code: {}
    This is the correct case and output: {}"""
EFFI_CHOOSE = """As a professional Python algorithm engineer, please help me choose the most efficient Python code from the following codes. It is worth mentioning that it is necessary to consider the time complexity and practical level comprehensively:
INPUT:
{"1":"def ...()....",
"2": "def ...()..."
}
OUTPUT:
```text
{key}
EXAMPLE:
INPUT:
{"1":"def ...()....",
"2": "def ...()..."
}
OUTPUT:
```text
1
``` """


def _effi(ctx: Ctx, system: str, *users: str, role: str) -> str:
    msgs = [{"role": "system", "content": system}] + [{"role": "user", "content": u} for u in users if u]
    if len(msgs) > 2:
        msgs = msgs[:1] + [{"role": "user", "content": "\n\n".join(m["content"] for m in msgs[1:])}]
    return ctx.llm.ask(msgs, temperature=EFFI_TEMPERATURE, role=role)


@h2h
class LLM4EFFI(H2HTechnique):
    """Task formalization with a consistency check, five low-complexity pseudo-algorithms, per algorithm efficiency
    suggestions and an implementation; synthetic edge tests (20 inputs, outputs computed one call each) verified both
    ways, one refinement round, final pick by tests then the model. The published run (about 45 calls) exceeds the
    ceiling: the test-independent stages run first, so the ceiling cuts test synthesis and the candidates are ranked by the
    public examples."""
    hid, name, family, target = "H03", "LLM4EFFI", "prompting", ("EFF",)
    code_url = "https://anonymous.4open.science/r/LLM4EFFI-04B2"

    def solve(self, task: Task, ctx: Ctx) -> str:
        ques = C.task_text(task)
        gen, check = "", ""
        for _ in range(EFFI_FORMALIZE_TRIES):
            gen = _effi(ctx, EFFI_FORMALIZE, EFFI_FORMALIZE_USER.format(ques), f"{gen}\n{check}".strip(), role="formalize")
            check = _effi(ctx, EFFI_CHECK, EFFI_CHECK_USER.format(ques, gen), role="formalize-check")
            if "yes" in check.lower():
                break
        T = ques + gen
        algos: list[str] = []
        for _ in range(EFFI_ALGO_TRIES):
            out = _effi(ctx, EFFI_ALGORITHMS, T, role="algorithms")
            algos = [b.strip() for b in re.findall(r"```algorithm\d\s*(.*?)```", out, re.S)][:EFFI_ALGOS]
            if len(algos) == EFFI_ALGOS:
                break
        cands: list[str] = []
        for a in algos or [""]:
            tip = _effi(ctx, EFFI_SUGGEST, f"{T}\n{a}", role="suggestions")
            code = C.code_of(task, _effi(ctx, EFFI_CODE, f"{T}\n{a}", "This is a knowledge base:" + tip, role="implement"))
            cands.append(code)
            ctx.keep(next((c for c in cands if sandbox.passes_public(task, c)), cands[0]))
        tests = self._tests(task, ctx, T) if task.mode == "function" else []
        return self._select(task, ctx, T, cands, tests)

    @staticmethod
    def _tests(task: Task, ctx: Ctx, T: str) -> list[Test]:
        tests: list[str] = []
        for _ in range(EFFI_FORMALIZE_TRIES):
            out = _effi(ctx, EFFI_DESIGNER.format(EFFI_TESTS), EFFI_DESIGNER_USER.format(T), role="test-inputs")
            block = (re.findall(r"```text\s*(.*?)```", out, re.S) or [out])[0]
            for line in [l.split("input:", 1)[1].strip() for l in block.splitlines() if "input:" in l]:
                a = C.code_of(task, _effi(ctx, EFFI_CALCULATOR, EFFI_CALCULATOR_USER.format(T, line), role="test-outputs"))
                a = next((l.strip() for l in a.splitlines() if l.strip().startswith("assert")), "")
                if a and a not in tests:
                    tests.append(a)
            if len(tests) >= EFFI_MIN_TESTS:
                break
        return [Test("assert", code=a) for a in tests]

    @staticmethod
    def _select(task: Task, ctx: Ctx, T: str, cands: list[str], tests: list[Test]) -> str:
        tests = list(task.public_tests) + tests

        def run(code):
            return [sandbox.run_test(task, code, t) for t in tests]
        res = [run(c) for c in cands]
        order = sorted(range(len(cands)), key=lambda i: -sum(r.ok for r in res[i]))
        if all(r.ok for r in res[order[0]]):
            return ctx.keep(cands[order[0]])
        failing = {j for rs in res for j, r in enumerate(rs) if not r.ok}
        timeouts = {j for rs in res for j, r in enumerate(rs) if r.timeout}
        kept = [j for j in range(len(tests)) if j not in failing or j < len(task.public_tests)]
        for j in sorted(failing - timeouts):
            if j >= len(task.public_tests):
                if "yes" in _effi(ctx, EFFI_REVIEW, EFFI_REVIEW_USER.format(T, tests[j].code), role="test-review").lower():
                    kept.append(j)
        checked = [tests[j] for j in sorted(set(kept) - timeouts)]
        score = lambda c: sum(sandbox.run_test(task, c, t).ok for t in checked)
        scores = [score(c) for c in cands]
        if any(s < len(checked) for s in scores):
            for i, c in enumerate(cands):
                bad = [t.code or f"input: {t.stdin!r} output: {t.expected!r}" for t in checked if not sandbox.run_test(task, c, t).ok]
                fixed = C.code_of(task, _effi(ctx, EFFI_REFINE, EFFI_REFINE_USER.format(T, c, json.dumps(bad)), role="refine"))
                if fixed and score(fixed) > scores[i]:
                    cands[i], scores[i] = fixed, score(fixed)
                ctx.keep(cands[max(range(len(cands)), key=lambda k: scores[k])])
        top = [i for i, s in enumerate(scores) if s == max(scores)]
        if len(top) == 1:
            return cands[top[0]]
        out = _effi(ctx, EFFI_CHOOSE, json.dumps({str(k + 1): cands[i] for k, i in enumerate(top)}), role="choose")
        m = re.search(r"\d+", (re.findall(r"```text\s*(.*?)(?:```|$)", out, re.S) or [out])[-1])
        k = int(m.group()) - 1 if m else 0
        return cands[top[k if 0 <= k < len(top) else 0]]


# ---------- H04 Code2API ----------

CODE2API = """Give you a context including a question title, a question post and an answer post,
your task is to transform the python code snippet in the answer post into a python function based on the context.

To solve the problem, do the following:
step 1 - Recover import statements based on the code snippet. If necessary, it can be none.
step 2 - Create the method name based on the context or the code snippet itself.
step 3 - Infer parameter list based on the code snippet. If necessary, it can be none.
step 4 - Infer return statements based on the code snippet. If necessary, it can be none.
step 5 - Output the complete code based on above results.

When performing step 3, you can consider the following principle:
Any variable or literal that can be rewritten as input parameters must be a parameter.
When performing step 4, you can consider the following principle:
According to the intent of the code snippet or context, necessary calculation results must be used as return values.

Here are some examples:
Example one:
Question title: <How to check whether a file is empty or not>
Qustion post: <I have a text file. How can I check whether it's empty or not?>
Answer post: <>>> import os
>>> os.stat("file").st_size == 0
True
>
Code snippet in the answer post: <>>> import os
>>> os.stat("file").st_size == 0
True
>
Specific steps:
Step 1: The import statement is "import os".
Step 2: The method name can be "is_file_empty".
Step 3: The parameter is the file path, which can be named "file_path".
Step 4: The return statement is "os.stat(file_path).st_size == 0", which checks whether the file is empty or not.

Complete code:
```python
import os

def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0
```

Now, give you the following context:
Question title: <{title}>
Qustion post: <{post}>
Answer post: <{answer}>
Code snippet in the answer post: <{snippet}>
{constraint}
Please output the results in the following format:
Specific steps: <the results of step 1-4>
Complete code: <the result of step 5>"""


@h2h
class Code2API(H2HTechnique):
    """The model's own first answer plays the Stack Overflow answer; Code2API's one-call 5-step chain of thought with its
    released Python example turns it into a reusable, well-named function (temperature 0). The interface the task
    requires is kept, since tests call it."""
    hid, name, family, target = "H04", "Code2API", "prompting", ("RM",)
    code_url = "https://doi.org/10.6084/m9.figshare.24219856.v1"

    def solve(self, task: Task, ctx: Ctx) -> str:
        snippet = draft(task, ctx)
        title = next((l.strip(' #"\'') for l in task.prompt.splitlines() if l.strip(' #"\'')
                      and not l.strip().startswith(("from ", "import ", "def ", "class ", "@"))), task.id)
        constraint = (f"The function must keep the name and signature of `{task.entry_point}` that the question asks for.\n"
                      if task.mode == "function" and task.entry_point else
                      "The result must stay a complete program that reads standard input and writes standard output.\n"
                      if task.mode == "stdio" else "")
        out = ctx.llm.ask(user(CODE2API.format(title=title, post=C.task_text(task), answer=snippet, snippet=snippet,
                                               constraint=constraint)), temperature=0, role="code2api")
        m = re.search(r"Complete code:\s*```(?:python)?\s*(.*?)(?=```)", out, re.S)
        return (m.group(1).strip() if m else C.code_of(task, out)) or snippet


# ---------- H05 MPCO ----------

MPCO_META = """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}.
Consider the project context, task context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

## Target LLM Context
- Target Model: {target_llm}
- Considerations: {llm_considerations}"""
MPCO_PROJECTS = {
    "humaneval_plus": "HumanEval+: hand-written Python function-completion problems checked by extended unit tests",
    "mbpp_plus": "MBPP+: crowd-sourced basic Python programming problems checked by extended unit tests",
    "lcb_v6": "LiveCodeBench: competitive-programming problems from LeetCode, AtCoder and Codeforces",
    "lcb_pro": "LiveCodeBench Pro: recent Codeforces contest problems with strict time and memory limits",
    "cweval": "CWEval: security-sensitive function completion tasks checked for functionality and security",
    "cyberseceval": "CyberSecEval: instruction and autocomplete prompts from security-relevant open-source code",
    "effibench": "EffiBench: LeetCode problems whose solutions are scored by execution time and memory",
    "evalperf": "EvalPerf: performance-exercising Python tasks scored by CPU instruction counts",
}
MPCO_TASK = ("Optimize the given code snippet for runtime performance; return only the optimized code.",
             "Target metrics: execution time and CPU instructions on the project's tests. Domain constraints: the code must "
             "keep its interface, behaviour and output format exactly and stay within the standard library. Optimization "
             "considerations: algorithmic complexity, data structures, redundant computation, loops and I/O.")
MPCO_LLM = ("Open-weight 31B-parameter dense instruction-tuned model; strong at short self-contained Python programs and "
            "standard algorithms; may over-engineer or drop edge cases when asked for aggressive rewrites, so prompts "
            "should be explicit, structured and ask to preserve behaviour.")


@h2h
class MPCO(H2HTechnique):
    """A meta-prompter call (the same model) writes a project-, task- and model-aware optimization prompt; the model then
    optimizes the code with it. The context strings were not released and are written from the paper's description."""
    hid, name, family, target = "H05", "MPCO", "prompting", ("EFF",)

    def solve(self, task: Task, ctx: Ctx) -> str:
        code = draft(task, ctx)
        meta = MPCO_META.format(target_llm=MODEL_NAME, objective="runtime performance", project_name=task.benchmark,
                                project_description=MPCO_PROJECTS.get(task.benchmark, task.benchmark),
                                project_languages=C.LANG_NAME.get(task.language, task.language), task_description=MPCO_TASK[0],
                                task_considerations=MPCO_TASK[1], llm_considerations=MPCO_LLM)
        prompt = ctx.llm.ask(user(meta), role="meta-prompt")
        out = ctx.llm.ask(user(f"{prompt}\n\n```{task.language}\n{code}\n```", C.SYSTEM), role="optimize")
        return C.code_of(task, out) or code


# ---------- H06 BRACE ----------

BRACE_K = 3
BRACE_SELECT = """You are a software security analyst. Read the coding task below (target language: {lang}) and predict the Common Weakness Enumeration (CWE) weaknesses that could plausibly arise while implementing it. Do not claim that any program is already vulnerable; list the weaknesses an implementation of this task is at risk of, most relevant first.

Return only a JSON array; each item: {{"cwe_id": "CWE-<number>", "cwe_name": "<name>", "relevance": <number in [0, 1]>, "reason": "<one short sentence>"}}.

Coding task:
{task}"""
BRACE_CUE = ("Security guidance: when implementing the task above, avoid the following weaknesses.\n\n{items}\n\n"
             "Return the complete final code, preserve the functional requirements, apply the listed mitigations, and avoid "
             "the identified weaknesses.")


@h2h
class BRACE(H2HTechnique):
    """Risk selector call (the same model, deterministic) predicts the task's likely CWEs; ids are validated against the
    MITRE CWE catalog, deduplicated, kept in the selector's order and cut to K=3; a fixed formatter appends each CWE's
    canonical name, description and mitigations to the unchanged task prompt."""
    hid, name, family, target = "H06", "BRACE", "prompting", ("SEC",)

    def prepare(self, ctx: Ctx) -> None:
        cwe_catalog(ctx.dir("cwe"))

    def solve(self, task: Task, ctx: Ctx) -> str:
        cat = cwe_catalog(ctx.dir("cwe"))
        lang = C.LANG_NAME.get(task.language, task.language)
        z = json_of(ctx.llm.ask(user(BRACE_SELECT.format(lang=lang, task=task.prompt)), temperature=0, role="risk-selector"))
        ids = []
        for p in z if isinstance(z, list) else []:
            m = re.search(r"\d+", str(p.get("cwe_id", "")) if isinstance(p, dict) else "")
            cid = f"CWE-{int(m.group())}" if m else ""
            if cid in cat and cid not in ids:
                ids.append(cid)
        items = "\n\n".join(f"{c}: {cat[c]['name']}\nDescription: {cat[c]['description']}\nMitigations:\n" +
                            "\n".join(f"- {x}" for x in cat[c]["mitigations"]) for c in ids[:BRACE_K])
        msgs = C.messages(task, post=BRACE_CUE.format(items=items)) if items else C.messages(task)
        return C.code_of(task, ctx.llm.ask(msgs, role="generate"))


# ---------- H07 ECO ----------

ECO_K, ECO_TEMPERATURE = 2, 0.7
ECO_DISTILL = """Identify each optimization in the Slow Code
and explain how it speeds up the Fast Code.
Respond in JSON array form, with objects containing:
[
  {{
    "description": "Briefly describe the inefficiency in slow_code and how fast_code fixes it.",
    "runtime_improvement": "Integer (1-10) rating of runtime gain.",
    "category": "One of: Algorithm | Data Structure | Memory Management | Code Execution | System Interaction | Other"
  }},
  ...
]
Slow Code:
{slow}

Fast Code:
{fast}"""
ECO_ANALYZE = """You are a competitive-programming performance analyst.

### Task
1. A **slow Python program** is given between ```python``` blocks below.
2. Analyse it **only from a runtime-performance standpoint** - do **NOT** propose fixes or rewrites.
3. Identify every major **bottleneck** that contributes to slower runtime.
4. Cover these angles where applicable:
   * algorithmic complexity
   * data-structure choiceb
   * I/O or library usage
   * memory-access patterns / allocations
5. For each bottleneck, estimate its relative impact on a **1-10 scale** (10 = largest slowdown factor).

```python
{code}
```"""
ECO_PROMPT = """Given a program, optimization tips and retrieved examples, optimize the program and provide a more efficient version.

### Explanation:
{tips}

Followings are retrieved examples for optimization.
{examples}

Now, optimize the following code.

### Original Code:
{code}

### Optimized Code:
"""


def _loop_nodes(fn: ast.AST) -> list[ast.AST]:
    return [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.SetComp, ast.DictComp,
                                                       ast.GeneratorExp))]


def eco_advisor(code: str) -> list[str]:
    """ECO's symbolic advisor rules (recursion without memoization, data-structure misuse, slow library/I-O use, loop-
    invariant expensive calls) ported from Joern queries on C++ to Python AST queries."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    tips: list[str] = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        calls_self = any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == fn.name for n in ast.walk(fn))
        memo = any("cache" in ast.unparse(d) for d in fn.decorator_list) or any(
            isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) for n in ast.walk(fn))
        if calls_self and not memo:
            tips.append(f"The following methods are purely recursive: [method: {fn.name}, lines: {fn.lineno}-{fn.end_lineno}]. "
                        "Applying memoization or dynamic programming can significantly reduce its execution time.")
    lists = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, (ast.List, ast.ListComp))
             for t in n.targets if isinstance(t, ast.Name)}
    member, front = [], []
    for loop in _loop_nodes(tree):
        for n in ast.walk(loop):
            if isinstance(n, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in n.ops):
                member += [f"[variable: {c.id}, lines: {n.lineno}-{n.end_lineno}]" for c in n.comparators
                           if isinstance(c, ast.Name) and c.id in lists]
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id in lists and ((n.func.attr == "pop" and n.args and ast.unparse(n.args[0]) == "0")
                                                  or (n.func.attr == "insert" and n.args and ast.unparse(n.args[0]) == "0")):
            front.append(f"[variable: {n.func.value.id}, lines: {n.lineno}-{n.end_lineno}]")
    if member:
        tips.append(f"The following lists are used for membership tests inside loops: {', '.join(dict.fromkeys(member))}. "
                    "Replacing them with a set or dict can improve performance.")
    if front:
        tips.append(f"The following lists are modified at the front: {', '.join(dict.fromkeys(front))}. Replacing them with "
                    "collections.deque can improve performance.")
    io_calls, invariant = [], []
    for loop in _loop_nodes(tree):
        assigned = {t.id for n in ast.walk(loop) for t in ast.walk(n) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
        for n in ast.walk(loop):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id in ("input", "print"):
                    io_calls.append(f"call: {n.func.id}, lines: {n.lineno}-{n.end_lineno}")
                elif n.func.id in ("sorted", "min", "max", "sum", "set", "list", "len") and n.args and not (
                        {x.id for a in n.args for x in ast.walk(a) if isinstance(x, ast.Name)} & assigned):
                    invariant.append(f"call: {n.func.id}, lines: {n.lineno}-{n.end_lineno}")
    if io_calls:
        tips.append(f"The following I/O library calls rely on slow operations: [{', '.join(dict.fromkeys(io_calls))}]. Replacing "
                    "them with faster alternatives (sys.stdin.readline, collecting output and writing it once) can improve "
                    "performance.")
    if invariant:
        tips.append(f"The following redundant calls are placed inside loops: [{', '.join(dict.fromkeys(invariant))}]. Moving these "
                    "calls outside the loop, or caching their results, can eliminate redundant work and improve efficiency.")
    return tips


@h2h
class ECO(H2HTechnique):
    """Performance-aware prompt: rule-based bottleneck tips from the symbolic advisor plus the 2 optimization instructions
    (with their slow/fast pairs) whose distilled text is closest to the model's own bottleneck analysis (Qodo-Embed).
    The C++ rules and PIE-HQ knowledge base are not released: rules ported to Python, base distilled from PIE's Python
    high-quality pairs with the paper's prompt."""
    hid, name, family, target = "H07", "ECO", "prompting", ("EFF",)
    ported = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "index.npy").exists():
            return
        rows = decontaminate(pie_hq(pie_python(ctx)), "input", "target")
        kb = []
        with prep(self.hid):
            for r in rows:
                ins = json_of(ctx.llm.ask(user(ECO_DISTILL.format(slow=r["input"], fast=r["target"])), role="distill"))
                text = " ".join(str(x.get("description", "")) for x in ins if isinstance(x, dict)) if isinstance(ins, list) else ""
                if text:
                    kb.append({"slow": r["input"], "fast": r["target"], "instruction": text})
        save_index(d, kb, embed("qodo", [k["instruction"] for k in kb]))

    def solve(self, task: Task, ctx: Ctx) -> str:
        rows, vecs = load_index(ctx.dir(self.hid))
        code = draft(task, ctx)
        analysis = ctx.llm.ask(user(ECO_ANALYZE.format(code=code)), role="analyze")
        shots = [rows[i] for i in top_k(vecs, embed("qodo", [analysis])[0], ECO_K)]
        tips = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(eco_advisor(code))) or "1. No rule-based bottleneck found."
        examples = "\n\n".join(f"### Original Example Code{i + 1}:\n```{r['slow']}```\n\n### Optimized Example Code{i + 1}:\n"
                               f"```{r['fast']}```\n\n[{r['instruction']}]" for i, r in enumerate(shots))
        out = ctx.llm.ask(user(ECO_PROMPT.format(tips=tips, examples=examples, code=code), C.SYSTEM), temperature=ECO_TEMPERATURE,
                          role="optimize")
        return C.code_of(task, out) or code


# ---------- H08 StoryCoder ----------

# Narrative transformation guidelines, verbatim from the paper (Figure A) and instruction_template.py of
# https://github.com/gu-ni/StoryCoder.
STORY_GUIDELINES = """Please transform the coding problem into a narrative story using the following guidelines.
### Guidelines for Narrative Conversion:
Before writing the narrative, you must complete two preliminary steps:
1. Review the major categories of coding test algorithms:
- Graph Algorithms
- Dynamic Programming
- Greedy Algorithms
- Sorting and Searching
- String Algorithms
- Data Structures
- Mathematics and Number Theory
- Simulation and Implementation
2. Decide which algorithm category the given problem most closely belongs to.
Then, select a narrative genre that naturally aligns with the chosen algorithm.
### Output Format:
You must write the output in the exact following order with the specified headers:
- Algorithm Category: (one of the categories above)
- Narrative Genre: (a fitting genre of your choice)
- Task Overview: Describe the background and objective of the problem in a clear, narrative-inspired manner. The selected algorithm should be introduced naturally here, with its logic explained as part of the setting or scenario.
- Constraints: State input sizes, value ranges, conditions, and key operational rules. If efficiency or time limits exist, express them as natural constraints. The chosen algorithm should also shape these rules.
- Example Input/Output: Reframe the examples as part of the scenario's flow. Present them as clear, contextual situations.
The narrative must include all essential parts of the original problem, ensuring no constraints, goals, or examples are omitted.
Do not include any other text outside these five sections.
Do not attempt to solve the problem or provide any code. Your task is only to transform the problem statement into the narrative format as specified.
### The coding problem is as follows:
{problem}"""
STORY_NARR_TEMPERATURE, STORY_CODE_TEMPERATURE = 1.0, 0.2
STORY_MIN_WORDS = 40  # the paper's validity filter drops near-empty narratives (< 50 tokens)


def strip_tags(narrative: str) -> str:
    """As split_narratives.py: drop the Algorithm Category and Narrative Genre sections, keep the other three."""
    t = re.sub(r"(?im)^\W*algorithm category:.*$\n?", "", narrative)
    t = re.sub(r"(?im)^\W*narrative genre:.*$\n?", "", t)
    return t.strip()


def story_valid(narrative: str) -> bool:
    """The paper's validity rule: enough content and all three required components present."""
    low = narrative.lower()
    return (len(narrative.split()) >= STORY_MIN_WORDS
            and all(h in low for h in ("task overview", "constraints", "example input/output")))


@h2h
class StoryCoder(H2HTechnique):
    """Self-solving setting (generator and solver are the same model): one call reformulates the problem into a narrative
    (algorithm category and genre chosen by the model, then task overview, constraints and example input/output;
    T=1.0); the algorithm and genre lines are stripped as in the released code, and a second call writes the code at
    T=0.2. For stdin problems the narrative replaces the statement (narrative-only variant, as in split_narratives.py);
    for function and edit tasks it is concatenated with the original task, so the signature and code to edit are kept
    (the paper's Orig.+Narr. variant). The paper samples five narratives and ten solutions to report pass@10; one
    narrative and one solution is its k=1 point. An invalid narrative is redrawn once (paper's validity filter)."""
    hid, name, family, target = "H08", "StoryCoder", "prompting", ("RM",)
    code_url = "https://github.com/gu-ni/StoryCoder"

    def solve(self, task: Task, ctx: Ctx) -> str:
        ask = user(STORY_GUIDELINES.format(problem=task.prompt if task.mode != "edit" else C.task_text(task)))
        narrative = ctx.llm.ask(ask, temperature=STORY_NARR_TEMPERATURE, role="narrative")
        if not story_valid(narrative):
            narrative = ctx.llm.ask(ask, temperature=STORY_NARR_TEMPERATURE, role="narrative")
        story = strip_tags(narrative)
        if task.mode == "stdio" and story_valid(narrative):
            msgs = C.messages(Task(**{**task.__dict__, "prompt": story}))
        else:
            msgs = C.messages(task, pre=story)
        return C.code_of(task, ctx.llm.ask(msgs, temperature=STORY_CODE_TEMPERATURE, role="code"))


# ---------- H09 Constitutional Spec-Driven Development ----------

CSDD_CONSTITUTION = ("https://raw.githubusercontent.com/srinivasraom/banking-ms-by-constitution/001-banking-crud/"
                     ".specify/memory/constitution.md")
CSDD_SPECIFY = """You are following spec-driven development under the project constitution below; every artifact must comply with it.

{constitution}

/speckit.specify: Write the feature specification for the request below: functional requirements, security requirements derived from the constitution's principles (cite the CWE of each), data handling requirements, the interface contract, and acceptance criteria. Do not write code.

Request:
{task}"""
CSDD_PLAN = """/speckit.plan: Write the implementation plan for this specification: design decisions with their security justification, the components, and a Constitution Check listing each applicable principle and how the design satisfies it. Do not write code."""
CSDD_TASKS = """/speckit.tasks: Break the plan into ordered, atomic implementation tasks, each with acceptance criteria and the constitutional requirements it must meet. Do not write code."""
CSDD_IMPLEMENT = """/speckit.implement: Implement all tasks within the constitution's bounds. {format} For each constitutional principle the code relies on, add a one-line comment naming it at the line that satisfies it."""
CSDD_REJECT = """The implementation was rejected: it violates the constitution.
{violations}
Regenerate the complete implementation so that it complies with these principles while meeting the specification. {format}"""
BANDIT_CWE = {
    **dict.fromkeys(("B105", "B106", "B107"), 798), "B108": 377, **dict.fromkeys(("B301", "B302", "B403", "B506"), 502),
    **dict.fromkeys(("B303", "B304", "B305", "B324", "B413"), 327), "B311": 330, "B608": 89, "B201": 94, "B310": 22,
    **dict.fromkeys(("B602", "B603", "B604", "B605", "B606", "B607", "B609", "B102", "B307"), 78), "B701": 79,
    **dict.fromkeys(("B501", "B502", "B503", "B504"), 295), **dict.fromkeys(("B314", "B320", "B405", "B410"), 20), "B104": 605,
}


@h2h
class CSDD(H2HTechnique):
    """The authors' versioned CWE-mapped constitution governs four Spec Kit steps (specify, plan, tasks, implement) in one
    conversation; generated code is validated against the constitution with Bandit and, on a violation, rejected and
    regenerated once with a prompt naming the violated principles."""
    hid, name, family, target = "H09", "Constitutional Spec-Driven Development", "prompting", ("SEC",)
    code_url = "https://github.com/srinivasraom/banking-ms-by-constitution"

    def prepare(self, ctx: Ctx) -> None:
        fetch(CSDD_CONSTITUTION, ctx.dir(self.hid) / "constitution.md")

    def solve(self, task: Task, ctx: Ctx) -> str:
        constitution = (ctx.dir(self.hid) / "constitution.md").read_text()
        rules = {int(m.group(1)): m.group(0).strip("- ") for m in re.finditer(r"\*\*CWE-(\d+)[^*]*\*\*:[^\n]*", constitution)}
        fmt = f"Return the complete code in one ```{task.language} block, as the task requires."
        msgs = user(CSDD_SPECIFY.format(constitution=constitution, task=C.task_text(task)))
        for step in (CSDD_PLAN, CSDD_TASKS):
            msgs += [{"role": "assistant", "content": ctx.llm.ask(msgs, role="spec-kit")}, {"role": "user", "content": step}]
        msgs += [{"role": "assistant", "content": ctx.llm.ask(msgs, role="spec-kit")},
                 {"role": "user", "content": CSDD_IMPLEMENT.format(format=fmt)}]
        out = ctx.llm.ask(msgs, role="implement")
        code = ctx.keep(C.code_of(task, out))
        found = [f for f in C.report(task, code).security if f.tool == "bandit"]
        if not found:
            return code
        lines = [f"- Per {rules.get(BANDIT_CWE.get(f.rule, 0), 'principle I (Security-First)')}: line {f.line}, {f.message}"
                 for f in found]
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": CSDD_REJECT.format(violations="\n".join(dict.fromkeys(lines)), format=fmt)}]
        return C.code_of(task, ctx.llm.ask(msgs, role="regenerate")) or code


# ---------- H10 SecureForge ----------

SF_SEEDS = ("https://raw.githubusercontent.com/sisl/SecureForge/3d10af48cf375b4cbad6057d837cc41c8e621bab/secureforge/data/"
            "scenario_dow.jsonl")
SF_SEMGREP = ("semgrep", "scan", "--config=auto", "--no-git-ignore", "--json")
SF_CWES = (79, 787, 89, 352, 22, 125, 78, 416, 862, 434, 94, 20, 77, 287, 269, 502, 200, 863, 918, 119, 476, 798, 190, 400, 306)
SF_SEED_ROWS, SF_SCENARIOS, SF_ROLLOUTS = 20, 10, 10
SF_MIN_ROLLOUTS, SF_VARIANCE, SF_BURN_IN, SF_STEPS = 5, 0.015, 3, 32
SF_GEPA_N, SF_MINIBATCH = 6, 3
SF_T_GEN, SF_T_AMP, SF_T_OPT = 1.0, 0.8, 1.0
SF_MAX_TOKENS, SF_WORKERS = 32768, 32
SF_LANG = {"python": "Python", "javascript": "JavaScript", "java": "Java", "cpp": "C/C++", "c": "C/C++"}
SF_GENERATE_CODE = ("Given the prompt, generate code that solves the task; as a reminder, you should be writing production code (i.e. "
                    "disable any debugging traces, etc.) Return generated code only, do NOT add extra explanation or instructions. "
                    "If test code is provided, ensure the solution passes those tests.")
SF_CODE_IN = (("task", "str", ""), ("language", "str", ""),
              ("test_code", "str", "Pytest test code the solution must pass (may be empty)"))
SF_CODE_OUT = (("reasoning", "str", ""),
               ("code", "str", "Generated source code to solve the task; do not add extra explanation, instructions, or text."))
SF_DESCRIBE = ("given a code snippet, describe what scenario/situation the code is trying to accomplish",
               (("code", "str", ""), ("language", "str", "")),
               (("reasoning", "str", ""), ("scenario", "str", "a brief description of what this code snippet is trying to do")))
SF_EXTRACT = ("Given the weakness description, provide a few tasks that would exercise the vulnerability",
              (("name", "str", ""), ("description", "str", ""),
               ("language", "str", "the programming language the task should be implemented in, mention this in the description")),
              (("scenarios", "list[str]", "scenarios that exercises this weakness; follow examples you are given"),))
SF_STRIP = ("Given a scenario, strip any mention of potential vulnerability from the text, leaving only the coding task",
            (("scenario", "str", ""),),
            (("coding_task", "str", "a description of the coding task without mention of vulnerability"),))
SF_SUGGEST = ("Make the coding task more specific by recommending the use of one of the suggested libraries; if not possible, "
              "return None",
              (("task", "str", ""), ("suggested_libraries", "list[str]", "")),
              (("chosen_library", "str", "choose a library that would best help solve the task, or say None"),
               ("rephrased_task", "str", "rephrase the task in terms of the chosen library, or say None")))
SF_TEST = ("""Given a coding task description, generate a test file that validates functional correctness.

Focus on functional correctness with concrete inputs and expected outputs.
Do NOT test for security vulnerabilities — only test that the code works correctly.
Include at least 2-3 test cases covering normal usage and edge cases.
Mock any external dependencies (databases, network, file I/O) if needed.

The `language` and `test_instructions` fields specify the target language and
framework-specific instructions for the test file.""",
           (("task", "str", ""), ("language", "str", ""), ("test_instructions", "str", "")),
           (("reasoning", "str", ""),
            ("test_code", "str", "A test file that validates correctness; do not add explanation or markdown fences.")))
SF_TEST_DOC = ("Generate a pytest test file that imports from `solution` (e.g., `from solution import ...`). Focus on functional "
               "correctness with concrete inputs and expected outputs. Do NOT test for security vulnerabilities — only test that "
               "the code works correctly. Include at least 2-3 test cases covering normal usage and edge cases. Mock any external "
               "dependencies (databases, network, file I/O) if needed.")
SF_REPHRASE = ("Come up with a rephrased coding task that exercises the same techniques and tools; think about this as an exercise "
               "that checks same skills and knowledge; make sure that the new task is meaningfully different such that its not "
               "just the first task with names changed, but also make sure the new task excercises the exact sample libraries and "
               "skills.", (("task", "str", ""),), (("reasoning", "str", ""), ("rephrased_task", "str", "")))
SF_LIBRARIES = [
    "Django", "Flask", "FastAPI", "Tornado", "Twisted", "Starlette", "Sanic", "Bottle", "CherryPy", "Pyramid", "Flask-Admin",
    "Gradio", "Streamlit", "aiohttp", "djangorestframework", "httpx", "pycurl", "requests", "urllib", "urllib2", "urllib3", "dill",
    "PyYAML", "ruamel.yaml", "simplejson", "toml", "ujson", "jsonpickle", "json", "aioch", "aiomysql", "aiopg", "aiosqlite",
    "asyncpg", "cassandra-driver", "clickhouse-driver", "cx_Oracle", "hdbcli", "mysql-connector", "mysql-connector-python",
    "MySQL-python", "mysqlclient", "oracledb", "phoenixdb", "psycopg2", "pymssql", "PyMySQL", "pyodbc", "sqlite3",
    "Flask-SQLAlchemy", "peewee", "SQLAlchemy", "PyMongo", "cryptography", "pycryptodome", "pycryptodomex", "rsa", "MarkupSafe",
    "libtaxii", "libxml2", "lxml", "xmltodict", "Jinja2", "Mako", "Genshi", "Chameleon", "Chevron", "Airspeed", "TRender",
    "fabric", "idna", "invoke", "jmespath", "multidict", "pydantic", "yarl", "joblib", "paramiko", "pexpect", "setuptools",
    "anyio", "numpy", "pandas", "torch", "AWS Lambda",
]
SF_VULN = "The generated trace contains vulnerabilities. Please fix the code to remove all vulnerabilities.\n{findings}"
SF_CLEAN = "The generated trace does not contain any vulnerabilities. Good job!"
SF_REFLECT = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""
SF_LIST_NOTE = '        # note: the value you produce must adhere to the JSON schema: {"type": "array", "items": {"type": "string"}}'
SF_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def _sf_value(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _sf_msgs(sig: tuple, inputs: dict, demos: list[dict] = ()) -> list:
    """DSPy 3.3.1 ChatAdapter messages for a signature (instructions, input fields, output fields)."""
    instr, ins, outs = sig
    desc = lambda fs: "\n".join(f"{i + 1}. `{n}` ({a}): {d}" for i, (n, a, d) in enumerate(fs)).strip()
    shape = lambda fs, out: "\n\n".join(f"[[ ## {n} ## ]]\n{{{n}}}" + (SF_LIST_NOTE if out and a != "str" else "")
                                        for n, a, _ in fs)
    goal = ("\n" + " " * 8).join([""] + textwrap.dedent(instr).splitlines())
    system = (f"Your input fields are:\n{desc(ins)}\nYour output fields are:\n{desc(outs)}\n"
              "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
              f"{shape(ins, False)}\n\n{shape(outs, True)}\n\n[[ ## completed ## ]]\n"
              f"In adhering to this structure, your objective is: {goal}")
    req = ("Respond with the corresponding output fields, starting with the field " + ", then ".join(
        f"`[[ ## {n} ## ]]`" + ("" if a == "str" else f" (must be formatted as a valid Python {a})") for n, a, _ in outs)
        + ", and then ending with the marker for `[[ ## completed ## ]]`.")

    def body(vals, *extra, prefix=""):
        return "\n\n".join([prefix, *(f"[[ ## {n} ## ]]\n{_sf_value(vals[n])}" for n, _, _ in ins if n in vals), *extra]).strip()

    def answer(vals, missing):
        return "\n\n".join(f"[[ ## {n} ## ]]\n{_sf_value(vals.get(n, missing))}" for n, _, _ in outs).strip() + \
            "\n\n[[ ## completed ## ]]\n"
    fields = [n for n, _, _ in ins + outs]
    full = [x for x in demos if all(x.get(n) is not None for n in fields)]
    part = [x for x in demos if x not in full and any(n in x for n, _, _ in ins) and any(n in x for n, _, _ in outs)]
    msgs = [{"role": "system", "content": system}]
    for x in part:
        msgs += [{"role": "user", "content": body(x, prefix="This is an example of the task, though some input or output "
                                                            "fields are not supplied.")},
                 {"role": "assistant", "content": answer(x, "Not supplied for this particular example. ")}]
    for x in full:
        msgs += [{"role": "user", "content": body(x)},
                 {"role": "assistant", "content": answer(x, "Not supplied for this conversation history message. ")}]
    return msgs + [{"role": "user", "content": body(inputs, req)}]


def _sf_parse(outs: tuple, text: str) -> dict | None:
    """ChatAdapter parsing: first section per output field; None when a field is missing or a list does not parse."""
    secs = [(None, [])]
    for line in (text or "").splitlines():
        m = SF_HEADER.match(line.strip())
        if m:
            secs.append((m.group(1), [line.strip()[m.end():].strip()]))
        else:
            secs[-1][1].append(line)
    got: dict = {}
    for k, v in secs:
        if k not in got:
            got[k] = "\n".join(v).strip()
    res = {}
    for n, a, _ in outs:
        if n not in got:
            return None
        res[n] = got[n]
        if a != "str":
            try:
                res[n] = json.loads(got[n])
            except json.JSONDecodeError:
                try:
                    res[n] = ast.literal_eval(got[n])
                except (ValueError, SyntaxError):
                    return None
            if not isinstance(res[n], list):
                return None
    return res


def _sf_ask(ctx: Ctx, sig: tuple, inputs: dict, temperature: float, role: str, demos: list[dict] = ()) -> dict:
    text = ctx.llm.ask(_sf_msgs(sig, inputs, demos), temperature=temperature, max_tokens=SF_MAX_TOKENS, role=role)
    return _sf_parse(sig[2], text) or {}


def _sf_fences(text: str) -> str:
    return text.replace("```python", "").replace("```", "").strip()


def _sf_codes(ctx: Ctx, instr: str, task: str, test: str, n: int, temperature: float) -> list[str]:
    """n rollouts of the ChainOfThought(GenerateCode) coder, fences stripped as the authors' run_k does."""
    outs = ctx.llm.sample(_sf_msgs((instr, SF_CODE_IN, SF_CODE_OUT), {"task": task, "language": "Python", "test_code": test}), n,
                          temperature=temperature, max_tokens=SF_MAX_TOKENS, role="rollout")
    return [_sf_fences(f["code"]) for f in (_sf_parse(SF_CODE_OUT, t) for t in outs) if f]


@lru_cache(maxsize=65536)
def _sf_scan(code: str) -> tuple | None:
    """Semgrep findings (rule, line, message) for code as a standalone file; None if the scan is incomplete."""
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "program.py").write_text(code)
        p = subprocess.run([*SF_SEMGREP, t], capture_output=True, text=True, timeout=600)
    try:
        r = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None
    if p.returncode or r.get("errors") or not r.get("paths", {}).get("scanned"):
        return None
    return tuple({"rule": x["check_id"], "line": x["start"]["line"], "message": x["extra"]["message"]}
                 for x in r.get("results", []))


@lru_cache(maxsize=1)
def _sf_cwe(root: Path) -> dict[int, tuple[str, str, str]]:
    """MITRE CWE research view: id -> (name, description, extended description)."""
    with zipfile.ZipFile(fetch(CWE_CSV, root / "cwe-1000.csv.zip")) as f:
        text = f.read(f.namelist()[0]).decode("utf-8", "replace")
    return {int(r["CWE-ID"]): (r["Name"], r["Description"], r["Extended Description"]) for r in csv.DictReader(io.StringIO(text))}


def _sf_pmap(hid: str, fn, items) -> list:
    def run(x):
        with prep(hid):
            return fn(x)
    with ThreadPoolExecutor(SF_WORKERS) as ex:
        return list(ex.map(run, items))


def _sf_budget(v: int, preds: int = 1, m: int = 35, full_every: int = 5) -> int:
    """dspy.GEPA auto_budget for the light preset."""
    n = int(max(2 * (preds * 2) * math.log2(SF_GEPA_N), 1.5 * SF_GEPA_N))
    return v + SF_GEPA_N * 5 + n * m + ((n + 1) // full_every + 1 + (1 if n < full_every else 0)) * v


def _sf_pick(scores: list[list[float]], rng: random.Random) -> int:
    """GEPA Pareto candidate selection: per-instance fronts, dominated programs removed, frequency-weighted draw."""
    fronts = [{k for k, s in enumerate(scores) if s[j] == max(x[j] for x in scores)} for j in range(len(scores[0]))]
    agg = [sum(s) for s in scores]
    progs = sorted({p for f in fronts for p in f}, key=lambda p: agg[p])
    dominated: set[int] = set()
    found = True
    while found:
        found = False
        for y in progs:
            others = set(progs) - dominated - {y}
            if y not in dominated and all(f & others for f in fronts if y in f):
                dominated.add(y)
                found = True
                break
    freq = Counter(p for f in fronts for p in sorted(f) if p not in dominated)
    return rng.choice([p for p, c in freq.items() for _ in range(c)])


def _sf_minibatch(i: int, order: list[int], epoch: int, n: int, rng: random.Random) -> tuple[list[int], list[int], int]:
    """GEPA EpochShuffledBatchSampler: reshuffle each epoch, pad to the minibatch size with the least used ids."""
    base = i * SF_MINIBATCH
    cur = 0 if epoch == -1 else base // len(order)
    if not order or cur > epoch:
        order, epoch = list(range(n)), cur
        rng.shuffle(order)
        order += order[::-1][:-n % SF_MINIBATCH]
    k = base % len(order)
    return order[k:k + SF_MINIBATCH], order, epoch


def _sf_md(v, level: int = 3) -> str:
    if isinstance(v, dict):
        return "".join(f"{'#' * level} {k}\n" + _sf_md(x, min(level + 1, 6)) for k, x in v.items()) or "\n"
    return f"{str(v).strip()}\n\n"


def _sf_instruction(out: str) -> str:
    """GEPA InstructionProposalSignature.output_extractor."""
    s, e = out.find("```") + 3, out.rfind("```")
    if s >= e:
        t = out.strip()
        if t.startswith("```"):
            m = re.match(r"^```\S*\n?", out)
            if m:
                return out[m.end():].strip()
        elif t.endswith("```"):
            return t[:-3].strip()
        return t
    c = out[s:e]
    m = re.match(r"^\S*\n", c)
    return (c[m.end():] if m else c).strip()


@h2h
class SecureForge(H2HTechnique):
    """Offline, as the authors' pipeline: benign falsification (few-shot CWE scenarios for the CWE Top 25, vulnerability
    mentions stripped, a library suggested, 10 rollouts each at T=1.0, Semgrep labels), MCMC amplification of every
    vulnerable rollout's prompt (LM rephrasing kernel, Metropolis-Hastings on the Beta-posterior failure rate, rollouts until
    variance < 0.015, 3 burn-in + 32 steps, T=0.8), then GEPA (light preset, Pareto selection, minibatch 3, +-1 Semgrep
    reward with line-level feedback, T=1.0) on the vulnerable amplified prompts optimizes the coder's instruction. Every
    helper (test writer, scenario writer, reflection) is the same model. At test time the optimized instruction drives the
    DSPy ChainOfThought coder: one call. GEPA's merge is enabled in the paper but never fires with a single predictor."""
    hid, name, family, target = "H10", "SecureForge", "prompting", ("SEC",)
    code_url = "https://github.com/sisl/SecureForge"
    trains = data_rebuilt = True

    def prepare(self, ctx: Ctx) -> None:
        d = ctx.dir(self.hid)
        if (d / "instruction.txt").exists():
            return
        gen, amp = d / "generated.jsonl", d / "amplified.jsonl"
        with prep(self.hid):
            if not gen.exists():
                write_jsonl(gen, self._falsify(ctx, d))
            tasks = [(r["scenario"], x["vulnerabilities"][0]["rule"]) for r in read_jsonl(gen) for x in r["rollouts"]
                     if x["vulnerabilities"]]
            done = len(read_jsonl(amp)) if amp.exists() else 0
            for i in range(done, len(tasks), SF_WORKERS):
                recs = _sf_pmap(self.hid, lambda t: self._amplify(ctx, *t), tasks[i:i + SF_WORKERS])
                with open(amp, "a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in recs))
            train = decontaminate([{"task": e["prompt"], "test_code": r["tests"] or ""} for r in read_jsonl(amp)
                                   for e in r["mcmc_failures"] for x in e["rollouts"] if x["vulnerabilities"]], "task")
            if not train:
                raise ValueError("SecureForge: no vulnerable amplified rollouts to optimize against")
            (d / "instruction.txt").write_text(self._gepa(ctx, train))

    def _falsify(self, ctx: Ctx, d: Path) -> list[dict]:
        cwe = _sf_cwe(ctx.dir("cwe"))
        rows = [json.loads(l) for l in fetch(SF_SEEDS, d / "scenario_dow.jsonl").read_text().splitlines() if l.strip()]
        rows = rows[:SF_SEED_ROWS]
        described = _sf_pmap(self.hid, lambda r: _sf_ask(ctx, SF_DESCRIBE, {"code": r["prompt"], "language": r["language"]},
                                                         SF_T_GEN, "describe").get("scenario"), rows)
        by: dict[int, list[str]] = {}
        for r, s in zip(rows, described):
            m = re.search(r"CWE-(\d+)", r["scenario_id"])
            if m and s:
                by.setdefault(int(m.group(1)), []).append(s)
        demos = [{"name": cwe[c][0], "description": cwe[c][2] or cwe[c][1], "scenarios": s} for c, s in by.items() if c in cwe]
        demos = random.Random(0).sample(demos, len(demos))
        scen = _sf_pmap(self.hid, lambda c: [(c, s) for s in self._scenarios(ctx, cwe[c], demos)], SF_CWES)
        return _sf_pmap(self.hid, lambda cs: self._rollout(ctx, *cs), [x for s in scen for x in s])

    @staticmethod
    def _scenarios(ctx: Ctx, entry: tuple[str, str, str], demos: list[dict]) -> list[str]:
        found: list[str] = []
        for _ in range(SF_SCENARIOS):
            if len(found) >= SF_SCENARIOS:
                break
            got = _sf_ask(ctx, SF_EXTRACT, {"name": entry[0], "description": entry[2], "language": "Python"}, SF_T_GEN,
                          "scenarios", demos)
            found += [str(s) for s in got.get("scenarios") or []]
        out = []
        for s in found:
            t = _sf_ask(ctx, SF_STRIP, {"scenario": s}, SF_T_GEN, "strip").get("coding_task") or s
            r = _sf_ask(ctx, SF_SUGGEST, {"task": t, "suggested_libraries": SF_LIBRARIES}, SF_T_GEN,
                        "library").get("rephrased_task")
            out.append(r if r and r.lower().strip() != "none" else t)
        return out

    @staticmethod
    def _rollout(ctx: Ctx, c: int, s: str) -> dict:
        test = _sf_fences(_sf_ask(ctx, SF_TEST, {"task": s, "language": "Python", "test_instructions": SF_TEST_DOC}, SF_T_GEN,
                                  "test").get("test_code", ""))
        codes = _sf_codes(ctx, SF_GENERATE_CODE, s, test, SF_ROLLOUTS, SF_T_GEN)
        return {"cwe": c, "scenario": s, "tests": test,
                "rollouts": [{"code": x, "vulnerabilities": _sf_scan(x)} for x in codes]}

    @staticmethod
    def _quantify(ctx: Ctx, prompt: str) -> tuple[int, int, list[dict]] | None:
        """Rollouts (5, then one at a time) until the Beta(1+fail, 1+safe) variance is below 0.015."""
        seen: dict[str, tuple | None] = {}
        f, n, k = 1, 1, SF_MIN_ROLLOUTS
        while True:
            for c in _sf_codes(ctx, SF_GENERATE_CODE, prompt, "", k, SF_T_AMP):
                if c not in seen:
                    seen[c] = _sf_scan(c)
                if seen[c] is None:
                    return None
                f, n = (f + 1, n) if seen[c] else (f, n + 1)
            if f * n / ((f + n) ** 2 * (f + n + 1)) <= SF_VARIANCE:
                return f, n, [{"code": c, "vulnerabilities": v} for c, v in seen.items()]
            k = 1

    def _amplify(self, ctx: Ctx, seed: str, rule: str) -> dict:
        rate = lambda q: (q[0] - 1) / (q[0] + q[1] - 2)
        q = self._quantify(ctx, seed)
        chain, tau, cur = [], seed, rate(q) if q else 0.0
        for i in range(SF_BURN_IN + SF_STEPS if q else 0):
            prop = _sf_ask(ctx, SF_REPHRASE, {"task": tau}, SF_T_AMP, "rephrase").get("rephrased_task")
            q = self._quantify(ctx, prop) if prop else None
            if prop and q is None:
                chain = []
                break
            p = rate(q) if q else 0.0
            if p > 0 and (cur == 0 or random.random() < math.exp(math.log(p) - math.log(cur))):
                tau, cur = prop, p
                if i >= SF_BURN_IN:
                    chain.append({"prompt": prop, "num_successes": q[1] - 1, "num_failures": q[0] - 1, "rollouts": q[2]})
        test = _sf_fences(_sf_ask(ctx, SF_TEST, {"task": seed, "language": "Python", "test_instructions": SF_TEST_DOC},
                                  SF_T_AMP, "test").get("test_code", "")) if chain else ""
        return {"type": rule, "seed": seed, "tests": test or None, "mcmc_failures": chain}

    @staticmethod
    def _attempt(ctx: Ctx, instr: str, ex: dict) -> tuple[float, dict | None, bool]:
        """One GEPA metric call: (score, reflective record, metric ran). Format failures score 0 and are not reflected on."""
        inputs = {"task": ex["task"], "language": "Python", "test_code": ex["test_code"]}
        f = _sf_parse(SF_CODE_OUT, ctx.llm.ask(_sf_msgs((instr, SF_CODE_IN, SF_CODE_OUT), inputs), temperature=SF_T_OPT,
                                               max_tokens=SF_MAX_TOKENS, role="gepa-rollout"))
        if not f:
            return 0.0, None, True
        v = _sf_scan(f["code"])
        if v is None:
            return 0.0, None, False
        fb = SF_VULN.format(findings="\n".join(f'- Line {x["line"]}: {x["rule"]} - {x["message"]}' for x in v)) if v else SF_CLEAN
        return (-1.0 if v else 1.0), {"Inputs": inputs, "Generated Outputs": f, "Feedback": fb}, True

    def _gepa(self, ctx: Ctx, train: list[dict]) -> str:
        """GEPA as run by dspy.GEPA (seed 0, valset = trainset): reflective mutation on a Pareto-selected parent, strict
        minibatch improvement then a full evaluation, budget from the light preset; best mean validation score wins."""
        v, rng = len(train), random.Random(0)
        ev = lambda instr, ids: _sf_pmap(self.hid, lambda j: self._attempt(ctx, instr, train[j]), list(ids))
        cands, scores = [SF_GENERATE_CODE], [[s for s, _, _ in ev(SF_GENERATE_CODE, range(v))]]
        used, budget, order, epoch, i = v, _sf_budget(v), [], -1, -1
        while used < budget:
            i += 1
            p = _sf_pick(scores, rng)
            ids, order, epoch = _sf_minibatch(i, order, epoch, v, rng)
            before = ev(cands[p], ids)
            used += len(ids)
            recs = [r for _, r, _ in before if r]
            if all(s >= 1 for s, _, _ in before) or not all(ok for _, _, ok in before) or not recs:
                continue
            side = "\n\n".join(f"# Example {k + 1}\n" + "".join(f"## {key}\n" + _sf_md(val) for key, val in r.items())
                               for k, r in enumerate(recs))
            out = ctx.llm.ask(user(SF_REFLECT.replace("<curr_param>", cands[p]).replace("<side_info>", side)),
                              temperature=SF_T_OPT, max_tokens=SF_MAX_TOKENS, role="reflect")
            new = _sf_instruction(out)
            after = ev(new, ids)
            used += len(ids)
            if sum(s for s, _, _ in after) > sum(s for s, _, _ in before):
                cands.append(new)
                scores.append([s for s, _, _ in ev(new, range(v))])
                used += v
        return cands[max(range(len(cands)), key=lambda k: (sum(scores[k]), -k))]

    def solve(self, task: Task, ctx: Ctx) -> str:
        instr = (ctx.dir(self.hid) / "instruction.txt").read_text()
        inputs = {"task": C.task_text(task), "language": SF_LANG.get(task.language, task.language), "test_code": ""}
        out = ctx.llm.ask(_sf_msgs((instr, SF_CODE_IN, SF_CODE_OUT), inputs), temperature=SF_T_GEN, max_tokens=SF_MAX_TOKENS,
                          role="generate")
        f = _sf_parse(SF_CODE_OUT, out)
        return ctx.keep(C.code_of(task, f["code"] if f else out))
