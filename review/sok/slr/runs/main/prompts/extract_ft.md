You are performing full-text eligibility assessment and data extraction for a systematic literature review in software engineering. You receive the full text of ONE paper (extracted from its PDF; tables may be garbled). Read all of it before answering.

# Review topic
Techniques that improve the NON-FUNCTIONAL quality of code generated or edited by large language models (LLMs).
- Technique: any intervention that changes the code an LLM produces: input (prompt/context), decoding, a post-generation loop (tool feedback, self-refinement, agents, critics), or the model's weights (fine-tuning, preference optimization, RL).
- Non-functional (NF) quality: readability-maintainability (naming, structure, complexity, smells, idiomaticity, documentation, style), efficiency (runtime, memory, energy), security (vulnerabilities, CWE). Correctness is NOT an NF dimension; record it as a co-outcome.

# Eligibility (apply to the full text)
Include only if ALL hold: IC1 proposes or empirically evaluates a technique applied to an LLM; IC2 the evaluated artifact is code generated or edited by an LLM; IC3 at least one quantitative NF outcome is reported for that code; IC4 full paper (peer-reviewed or preprint of 4+ pages).
Exclude if ANY holds: EC1 not English; EC2 secondary study; EC3 artifact is not production code (tests only, comments, commit messages, review comments, explanations); EC4 purely analysis/repair/refactoring of human-written code with no LLM generation; EC5 quality measured only as correctness/compilability/syntax validity; EC6 model robustness/safety/attack papers whose outcome is not generated-code quality; EC7 pre-LLM neural code models; EC8 duplicate/earlier version.
Side list for excluded papers: context (measures NF quality of LLM code without an intervention), adjacent (LLM refactoring/repair of human-written code), correctness-only (code-generation technique with correctness-only outcomes), secondary, none.

# Output
Return ONLY one JSON object with these keys. Use the controlled vocabularies exactly. If the paper is excluded, fill `eligibility`, `exclusion_criterion`, `side_list`, `eligibility_reason`, `confidence` and set the other fields to "" (empty) except `title`, `year`, `venue`.
If a paper evaluates several distinct techniques, put one object per technique in `techniques` (most papers have one).

{
 "eligibility": "include" | "exclude",
 "exclusion_criterion": "" | "EC1".."EC8" | "IC1".."IC4",
 "side_list": "none" | "context" | "adjacent" | "correctness-only" | "secondary",
 "eligibility_reason": "<one or two sentences>",
 "title": "...", "year": 2025, "venue": "<as stated in the paper, or 'preprint'>",
 "venue_type": "peer-reviewed-conf" | "peer-reviewed-journal" | "workshop" | "preprint",
 "study_type": "technique-proposal" | "technique-evaluation",
 "techniques": [{
   "technique_name": "...",
   "family_v0": "prompting" | "feedback-loop" | "self-refinement" | "multi-agent" | "retrieval" | "decoding" | "training" | "hybrid" | "other",
   "stage": "pre-generation" | "in-decoding" | "post-generation-loop" | "training-time",
   "signal_source": ["none" | "self" | "other-LLM" | "static-analyzer" | "compiler" | "test-execution" | "profiler" | "retrieved-knowledge" | "human" | "learned-verifier" | "reward-model", ...],
   "tools": "<analyzers/compilers/profilers with versions if given, else ''>",
   "iterations": "<max rounds and stopping rule, or ''>",
   "dims_targeted": ["readability-maintainability" | "efficiency" | "security", ...],
   "dims_measured": ["readability-maintainability" | "efficiency" | "security", ...],
   "correctness_measured": true | false,
   "metrics": "<per dimension: metric name and instrument type [tool-static | execution | llm-judge | human | similarity-to-reference]>",
   "benchmarks": "<name, version/subset, number of tasks>",
   "languages": "...",
   "models": "<name, size, open/closed, base/instruct/code-specialized/reasoning; semicolon-separated>",
   "baselines": "...",
   "effect_target": "+" | "0" | "-" | "mixed" | "NR",
   "effect_target_headline": "<the headline number copied EXACTLY as printed, with its setting; '' if none>",
   "effect_correctness": "+" | "0" | "-" | "mixed" | "NR",
   "effect_correctness_headline": "<exact number with setting, or ''>",
   "effect_other_dims": {"readability-maintainability": "+|0|-|mixed|NR|targeted", "efficiency": "...", "security": "..."}
 }],
 "compute_reported": ["calls" | "tokens" | "cost" | "time" | "none", ...],
 "contamination_handled": "yes" | "partial" | "no",
 "human_eval": "none" | "rating" | "preference",
 "human_eval_details": "<number of annotators and agreement, or ''>",
 "artifact_url": "<code/data URL if given, else ''>",
 "reimplementable": "yes" | "partial" | "no",
 "qa": {"QA1": 0|0.5|1, "QA2": ..., "QA3": ..., "QA4": ..., "QA5": ..., "QA6": ..., "QA7": ..., "QA8": ..., "QA9": ..., "QA10": ...},
 "confidence": "high" | "medium" | "low",
 "notes": "<anything a human reviewer should check; mention garbled tables or missing sections>"
}

Quality-assessment items (score 0, 0.5 or 1): QA1 technique described well enough to reimplement; QA2 compared with an appropriate baseline (no-technique or a strong prior technique); QA3 the NF metric is established or validated (standard tool, or human evaluation with reported agreement); QA4 correctness reported alongside NF quality; QA5 more than one model evaluated, or a single model justified; QA6 variance, repeated runs or a statistical test reported; QA7 compute or cost (calls, tokens) reported; QA8 artifacts (code or data) available; QA9 contamination considered; QA10 threats to validity discussed.

Rules: never invent numbers; copy headline numbers exactly as printed. Use "NR" when a direction is not reported. "effect_other_dims" uses "targeted" for targeted dimensions and "NR" for dimensions not measured. Set confidence to "low" when the text is garbled, key sections are missing, or eligibility is borderline.
