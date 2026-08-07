#!/usr/bin/env bash
set -euo pipefail
RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${STATIC_HH_PAPER_RECORDS_PATH:-chtc/phase3_optuna/input/static_hh_paper_l2_records.tsv}"
OUT_ROOT="raw_outputs/static_hh_paper_l2/${RECORD_ID}"
mkdir -p "$OUT_ROOT" logs
python - <<'PY' "$RECORD_ID" "$RECORDS_PATH" "$OUT_ROOT"
import csv, json, shlex, subprocess, sys
from pathlib import Path
record_id, records_path, out_root = sys.argv[1:]
rows = list(csv.DictReader(Path(records_path).read_text().splitlines(), delimiter='\t'))
row = next((r for r in rows if r.get('record_id') == record_id), None)
if row is None:
    raise SystemExit(f'record_id {record_id!r} not found in {records_path}')
out = Path(out_root)
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, '-u', '-m', 'pipelines.exact_bench.hh_static_paper_l2_benchmark',
    '--regenerate-static',
    '--output-dir', str(out / 'paper'),
    '--case-ids', row.get('case_ids') or 'hh_L2_strong_canonical,hh_L2_weak_diagnostic',
    '--compile-policy', row.get('compile_policy') or 'paper',
    '--backend-name', row.get('backend_name') or 'FakeMarrakesh',
    '--seed-transpiler', row.get('seed_transpiler') or '7',
    '--optimization-level', row.get('optimization_level') or '2',
]
(out / 'record.json').write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
(out / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')
print('RUN', ' '.join(shlex.quote(x) for x in cmd), flush=True)
subprocess.run(cmd, check=True)
PY
