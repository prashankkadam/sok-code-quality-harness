# Review data for "Beyond Passing Tests"

Everything behind the systematic literature review in the paper: queries, raw search exports, screening
decisions, prompts, full-text assessment, extraction data, quality assessment and the highlighted-technique
selection, plus the scripts that produce the paper's counts, tables and figures.

- `sok/slr/`: the data (see `sok/slr/README.md` for the layout).
- `src/slr/`: the review pipeline; `synth.py` regenerates the paper's counts, tables and figures.

The folder layout is the one the scripts expect; run them from this folder after `mkdir -p sok/paper/figures sok/paper/generated` (where they write their output).

## Full-text rule

Only reports whose full text was retrieved are assessed (`synth.retrieved`): a report without a retrievable full text
is excluded as not retrieved and never synthesized from its abstract. Every count, table and figure applies this rule.

## Corrections

`sok/slr/corrections.csv` logs every correction made to `extraction.csv` after extraction (`src/slr/corrections.py`):
targets filled in from the quote-checked selection review for highlighted techniques, and the outcome of a criteria
audit of every included study (`runs/main/scope_audit.csv`: duplicates, studies outside the criteria, studies moved to
the context list, a corrected year). The highlighted-technique exclusions are in `top/exclusions.csv`.

## Completing the review (`round2.sh`)

`round2.sh` runs the remaining steps with the frozen prompts: the revised strings on OpenAlex, the manual ACM DL and
IEEE Xplore exports (`src/slr/manual_search.py`), screening of the new records, full-text assessment and extraction of
every forwarded report not yet extracted (including the snowballing reports in `runs/main/snow_pending_ids.txt`), and
regeneration of all outputs. Until it has run, `synth.py` adds the estimates in `runs/main/round2_estimates.json` to the
affected counts and marks them blue in the paper.

## First consistency pass

Twelve records were moved from `extraction.csv` to
`runs/*/side_lists.csv`, each with its criterion and reason:

| Removed | Criterion | Reason |
| --- | --- | --- |
| R04129 | EC8 | Duplicate of R02327 |
| R04143 | EC8 | Duplicate of R04473 |
| R04257 | EC8 | Duplicate of R04482 |
| R06166 | EC8 | Duplicate of R02031 |
| R03729 | EC8 | Duplicate of R02515 |
| R05893 | EC8 | Duplicate of R04526 |
| R02793 | IC1 | Quantization study, not a quality-improving technique (context list) |
| R06253 | IC1 | Quantization study, not a quality-improving technique (context list) |
| R01946 | IC1 | Benchmark paper; only intervention evaluated for correctness (context list) |
| R02279 | IC1 | Benchmark paper; no quality-improving technique evaluated (context list) |
| R02821 | IC1 | Benchmark paper; no quality-improving technique evaluated (context list) |
| R01826 | IC1 | Mathematical problem-solving benchmark, not code quality |

Rule for benchmark papers: a benchmark paper stays included only if it reports a technique's effect on a
quality dimension (so Mercury, RTLBench, EnergyBench and SlopCodeBench stay).

The three duplicates that had reached the highlighted-technique pool (R06166, R03729, R05893) were also
removed from `top/pool.csv` and `top/merged.csv` (388 to 385 rows). The intermediate selection passes
(`top/pass1_*`, `top/pass2_*`) are kept as they ran.

## Check

Running `src/slr/synth.py --run main` on this data reproduces the paper's review counts and the generated tables
(effects, non-targeted dimensions, quality assessment, included-studies and quality-assessment appendices)
exactly.
