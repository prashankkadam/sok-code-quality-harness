#!/usr/bin/env bash
# Round-2 completion of the review. Needs ANTHROPIC_API_KEY and OPENALEX_API_KEY in ../.env, and the ACM DL / IEEE
# Xplore exports in sok/slr/runs/main/manual/ (see src/slr/manual_search.py). Every step resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ../.env; set +a
PY="${PY:-python3} -I"
R=main

# 1. Revised strings on OpenAlex, deduplicated against every record already screened.
for s in S1r1 S2r1; do $PY src/slr/search.py --run $R --string $s --sources openalex --end 2026-10-14; done
$PY src/slr/dedup.py --run $R --inputs records_S1r1_openalex.jsonl,records_S2r1_openalex.jsonl --against dedup_all.jsonl,snowball_it1.jsonl,other_sources_sections.jsonl,venue_sweep_new.jsonl --prefix A --out dedup_all_openalex_r1.jsonl

# 2. Manual ACM DL / IEEE Xplore exports.
$PY src/slr/manual_search.py ingest --run $R

# 3. Title/abstract screening of the new records with the frozen prompt.
for inp in dedup_all_openalex_r1.jsonl manual_new.jsonl; do
  $PY src/slr/screen.py submit --run $R --input $inp --tag "${inp%.jsonl}"
  $PY src/slr/screen.py collect --run $R --tag "${inp%.jsonl}"
done
$PY src/slr/screen.py merge --run $R

# 4. Full texts, then the two-stage full-text assessment and extraction for every forwarded report not yet extracted
#    (the snowballing reports and the new records). Reports without a full text are excluded as not retrieved.
$PY - <<'PYEOF'
import csv, json
from pathlib import Path
rd = Path("sok/slr/runs/main")
done = {json.loads(l)["rid"] for p in rd.glob("extraction/batch_stageB*/parsed.jsonl") for l in open(p)}
done |= {json.loads(l)["rid"] for p in rd.glob("extraction/batch_snowB*/parsed.jsonl") for l in open(p)}
fwd = [d["rid"] for d in csv.DictReader(open(rd / "decisions_ta.csv")) if d["forward_to_fulltext"] == "True"]
(rd / "round2_ids.txt").write_text("\n".join(r for r in fwd if r not in done) + "\n")
PYEOF
$PY src/slr/extract.py fetch --run $R --ids round2_ids.txt
$PY src/slr/extract.py submit --run $R --ids round2_ids.txt --tag round2A --prompt screen_ft.md --model claude-haiku-5-5
$PY src/slr/extract.py collect --run $R --tag round2A --wait
$PY src/slr/extract.py submit --run $R --ids round2_ids.txt --tag round2B
$PY src/slr/extract.py collect --run $R --tag round2B --wait
$PY src/slr/extract.py table --run $R

# 5. Regenerate every count, figure and table; the highlighted selection is re-run on the new eligible pool.
: > sok/slr/runs/$R/snow_pending_ids.txt
$PY src/slr/top_techniques.py pool
$PY src/slr/synth.py --run $R
$PY src/slr/appendix_gen.py --run $R
