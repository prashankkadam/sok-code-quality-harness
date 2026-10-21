# Beyond Passing Tests: code and data

Code and data for "Beyond Passing Tests: A Systematic Review and Controlled Re-evaluation of Techniques for
Improving the Quality of LLM-Generated Code".

- `review/`: the systematic literature review. All queries, raw search exports, screening decisions, prompts,
  full-text assessment, extraction data and quality assessment, plus the scripts that produce the paper's
  review counts, tables and figures. See `review/README.md`.
- `harness/`: the controlled re-evaluation (Section "Controlled Re-evaluation"): the main grid, the extended grid,
  the agent benchmarks and the E7 head-to-head of the 68 highlighted techniques.
- `configs/`: models, benchmarks, pinned analyzer configurations, training settings.
- `docker/`: the execution and measurement environment and a SonarQube server.

## Layout

| Path | What it does |
| --- | --- |
| `harness/llm.py` | Model clients (Claude API, Fireworks, Vercel AI Gateway with one pinned provider, local weights). Every call is charged to the task's budget and logged with tokens, latency and cost. |
| `harness/sandbox.py` | Isolated execution, test running, CPU instruction counts, wall-clock medians and IQR on pinned cores. |
| `harness/measure/` | The four-layer measurement stack: Pylint, Bandit, Safety, SonarQube and cognitive complexity; the length-controlled judge; metric definitions. |
| `harness/benchmarks/` | Loaders and official scorers: HumanEval+, MBPP+, LiveCodeBench v6, LiveCodeBench Pro protocol, RACE, NoFunEval, PythonSecurityEval, EvalPerf, EffiBench-X, Mercury, EffiBench, CWEval, BaxBench, CyberSecEval, DeepSWE, Terminal-Bench 4.0, FrontierCode, SlopCodeBench, GSO, SWE-fficiency, SusVibes. |
| `harness/techniques/` | T0-T10 (Table E1c), their agent versions for repository-level benchmarks, and the T6 retrieval corpus. |
| `harness/training/` | T8-T10 data from the base model, LoRA SFT and DPO on Fireworks, GRPO with TRL, adapter upload and serving. |
| `harness/h2h/` | E7: the T0 and best-of-n references and the 68 highlighted techniques, one module per family. |
| `harness/run.py`, `harness/stats.py`, `harness/tables.py` | Experiment drivers, paired bootstrap half-widths, trade-off matrix, budget curves, E7 counts, manual-rating agreement, LaTeX tables. |

## Setup

```bash
cp .env.example .env            # fill in keys; .env is never committed
docker compose -f docker/compose.yaml up -d sonarqube
pip install -e ".[whitebox,h2h]"
```

All runs execute inside the `harness` image (`docker/Dockerfile`) so tests, profiles and analyzers see the same
pinned environment; on Linux, measurements are pinned to the cores in `HARNESS_PIN_CORES`.

Data that is not public must be placed under `$HARNESS_DATA` before a run:

- `deepswe/`: the DeepSWE v1.1 tasks, environments and scorer obtained from its authors (layout in
  `harness/benchmarks/agent.py`, `DEEPSWE_LAYOUT`). Under the evaluation agreement no task text, patch or trajectory is
  written to disk; only per-task pass/fail is stored.
- `frontiercode/`: the FrontierCode tasks, rubric and reference grader (`FRONTIERCODE_LAYOUT`).
- `lcb_pro/<problem>/tests/`: hidden tests for the post-2026-07-01 Codeforces problems; `harness lcbpro-build`
  collects statements, limits and sample tests.

## Running the experiments

| Paper | Command |
| --- | --- |
| T6 corpus | `harness rag-build` |
| T8-T10 adapters | `harness train g31`, `harness train g26` |
| E2-E6, E8, E9 (main grid) | `harness main` |
| E10 and the extended technique table | `harness extended` |
| E2b and the agent table | `harness agent` |
| E7 head-to-head | `harness h2h` |
| Manual ratings | `harness manual-sample`, then `harness manual-agreement runs/main/manual_sample.csv` |
| All tables | `harness tables` |

Options: `--models`, `--techniques`, `--benches`, `--budgets`, `--seeds` (comma-separated). Runs resume where they
stopped. Results are stored as `runs/<experiment>/<model>/<technique>/B<b>/seed<s>/<benchmark>.jsonl`, and every
model call is logged in `runs/<experiment>/calls.jsonl`.

## Design as in the paper

- Main grid: Claude Opus 5.5, gpt-oss-120b, Gemma 4 31B IT and Gemma 4 26B-A4B IT; T0 at B=1, T1-T10 at
  B in {1, 4, 8, 16}; every call counts, including critic and reviewer calls; single-call techniques spend extra budget
  as best-of-B with the T7 ranker; rankers and gates see only the public examples. T8-T10 on the Gemma pair only.
- Open models at temperature 0.2 (0.8 for best-of-n) with fixed seeds; closed models at default sampling with a
  recorded effort. Every configuration runs three times with fresh seeds; tables report the mean.
- Clean splits: LiveCodeBench v6 only after each model's cutoff (gpt-oss and Gemma 4); LiveCodeBench Pro on
  Codeforces contests after 2026-07-01 for every model.
- Every output is scored on all four dimensions; effects are paired per-task differences against T7 at the same B
  (and T0 at B=1) with 95% bootstrap half-widths.
- E7: Gemma 4 31B IT, each technique in its published configuration with a ceiling of 16 calls; components that call
  another LLM use the same model; trained components are retrained for it; training-time techniques use LoRA.

## Where the implementation departs from the paper or its sources

- Fireworks stopped accepting new reinforcement fine-tuning jobs on 2026-10-01, so GRPO (T10 and the E7 RL
  techniques) trains locally with TRL and the adapter is uploaded to Fireworks for serving.
- Some highlighted techniques publish no prompts or code, or only an abstract; these are reconstructed from the
  paper's description and flagged in their class (`data_rebuilt`, `ported`). Techniques whose published configuration
  needs more than 16 calls run at the ceiling and are marked in the results (`capped`).
- Mean calls per task come from the runs; they may differ from the Calls column of the E7 table.
