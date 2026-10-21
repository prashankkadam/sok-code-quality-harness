You are performing full-text ELIGIBILITY assessment (stage A, before detailed extraction) for a systematic literature review in software engineering. You receive the full text of ONE paper (extracted from its PDF; tables may be garbled). Read all of it before answering.

# Review topic
Techniques that improve the NON-FUNCTIONAL quality of code generated or edited by large language models (LLMs).
- Technique: any intervention that changes the code an LLM produces: input (prompt/context), decoding, a post-generation loop (tool feedback, self-refinement, agents, critics), or the model's weights (fine-tuning, preference optimization, RL).
- Non-functional (NF) quality: readability-maintainability (naming, structure, complexity, smells, idiomaticity, documentation, style), efficiency (runtime, memory, energy), security (vulnerabilities, CWE). Correctness is NOT an NF dimension; record it as a co-outcome.

# Eligibility (apply to the full text)
Include only if ALL hold: IC1 proposes or empirically evaluates a technique applied to an LLM; IC2 the evaluated artifact is code generated or edited by an LLM; IC3 at least one quantitative NF outcome is reported for that code; IC4 full paper (peer-reviewed or preprint of 4+ pages).
Exclude if ANY holds: EC1 not English; EC2 secondary study; EC3 artifact is not production code (tests only, comments, commit messages, review comments, explanations); EC4 purely analysis/repair/refactoring of human-written code with no LLM generation; EC5 quality measured only as correctness/compilability/syntax validity; EC6 model robustness/safety/attack papers whose outcome is not generated-code quality; EC7 pre-LLM neural code models; EC8 duplicate/earlier version.
Side list for excluded papers: context (measures NF quality of LLM code without an intervention), adjacent (LLM refactoring/repair of human-written code), correctness-only (code-generation technique with correctness-only outcomes), secondary, none.

# Output
Return ONLY one JSON object:
{"eligibility": "include" | "exclude" | "uncertain",
 "criterion": "IC" for include, the single most decisive code (EC1-EC8 or IC1-IC4) for exclude, or the criterion you could not decide for uncertain,
 "side_list": "none" | "context" | "adjacent" | "correctness-only" | "secondary",
 "reason": "<one sentence, max 30 words, citing what in the text decided it>",
 "nf_outcomes_found": "<the quantitative non-functional outcome(s) the paper reports for LLM-generated code, or 'none'>"}

Rules: this stage only removes papers that clearly fail. Choose "exclude" only when the full text makes an exclusion criterion unambiguous (for example, no quantitative readability/maintainability, efficiency or security outcome is reported for LLM-generated or LLM-edited code anywhere in the paper, or the paper only analyzes human-written code). When in doubt, choose "uncertain".
