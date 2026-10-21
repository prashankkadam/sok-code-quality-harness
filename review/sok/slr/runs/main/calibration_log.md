# Search calibration log (PROTOCOL §4.3)

QGS: `sok/slr/qgs.csv`, 32 seed technique studies meeting IC1–IC4, frozen before
any S1 result was inspected (decisions and reasons: `sok/slr/qgs_decisions.csv`).

## Iteration 0: S1 as written in PROTOCOL §4.2

- Sources: OpenAlex (title_and_abstract.search, wildcards expanded), Semantic Scholar bulk search,
  arXiv API (ti/abs). Dates 2021-07-07 to 2026-10-14. Exact strings: `queries_S1.json`.
- Raw hits: OpenAlex 2,769; Semantic Scholar 1,442; arXiv 672 (4 chunks: 169, 83, 204, 216).
- Records retrieved 4,819; unique after dedup 2,773 (`dedup_S1_stats.json`).
- QGS recall: OpenAlex 0.594, Semantic Scholar 0.688, arXiv 0.562; **combined 0.688 (22/32)**,
  below the 0.85 target.
- Missed (10): CodePromptEval 2412.20545; CORE 2309.12938; Abtahi & Azim 2506.10330;
  SecureFixAgent 2509.16275; LLMLOOP 2603.23613; Self-Refine 2303.17651; SecCoder 2410.01488;
  PIE 2302.07867; CodeUltraFeedback 2403.09032; Afterburner 2505.23387.
- Diagnosis (facet matching on each missed title+abstract):
  - F2 (task) failed for 7: the papers describe editing, repairing, revising or optimizing code
    ("code revisions", "vulnerability repair", "program optimization", "codebases") without the
    phrase "code generation".
  - F1 (model) failed for SecCoder, which says "code LM".
  - F3 (quality) failed for CodePromptEval, LLMLOOP and CodeUltraFeedback ("quality of complete
    functions", "static analysis issues", "coding preferences").
  - Self-Refine matches neither F2 nor F3 (as PROTOCOL §4.2 anticipates); left to snowballing.

## Iteration 1: revise F1/F2, add S2 with prompting terms in F4

- F1 += "code LM*", LMs.
- F2 += "source code", "code edit*", "code repair", "vulnerability repair", "code refinement",
  "program optimization", "code optimization", "programming task*", "coding task*",
  "code-related task*", "coding preference*", "code revision*", codebase*, "code efficiency",
  "Python code", "code snippet*". (IC2 covers editing tasks, so the task facet must too.)
- F4 += "prompt engineering", "prompt technique*", "prompting technique*", "chain-of-thought",
  "few-shot", "in-context learning" (F4 had no term for the prompting family).
- Run S1r1 = F1' ∧ F2' ∧ F3 and S2r1 = F1' ∧ F2' ∧ F4'; the union is screened.
- Overfitting caveat: some added terms were found by inspecting QGS misses, so QGS recall after
  iteration 1 is optimistic. The OpenAlex venue sweep (replacing DBLP, see below) and snowballing
  give independent recall evidence; both are reported.
- Count-only raw hits: S1r1 OpenAlex 5,362, arXiv 1,068, Semantic Scholar 2,718; S2r1 OpenAlex 7,165,
  arXiv 2,923, Semantic Scholar 3,617.

### Iteration 1 results

- OpenAlex could not be queried for S1r1/S2r1: the free no-key daily budget for this network was
  exhausted (HTTP 429 "Insufficient budget", resets 00:00 UTC; it also rejects queries with many
  boolean operators without a key). S1r1 and S2r1 were therefore run on Semantic Scholar and arXiv
  only; OpenAlex contributes the iteration-0 S1 records. **Deviation, logged in REPORT/INBOX.** Re-run
  `search.py --string S1r1/S2r1 --sources openalex` once an OpenAlex key is available, then screen the
  new records.
- Retrieved: S1r1 arXiv 980 (1,068 hits over chunks), Semantic Scholar 2,718; S2r1 arXiv 2,530
  (2,923), Semantic Scholar 3,617.
- Union with iteration-0 records: 14,664 records → **6,942 unique** after dedup (`dedup_all_stats.json`).
- **QGS recall, combined: 0.938 (30/32)** (`qgs_recall_iter1.json`). Missed: LLMLOOP (2603.23613; not
  in the Semantic Scholar/arXiv results for S2r1 despite matching the terms locally; the OpenAlex
  S2r1 run would likely find it) and Self-Refine (2303.17651; matches no task or quality term, left to
  snowballing). Target 0.85 met; iteration stops.
