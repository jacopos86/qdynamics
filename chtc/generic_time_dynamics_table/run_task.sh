#!/usr/bin/env bash
set -euo pipefail
RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${2:-${GENERIC_TD_TABLE_RECORDS_PATH:-chtc/generic_time_dynamics_table/input/records.tsv}}"
OUT_ROOT="${GENERIC_TD_OUTPUT_ROOT:-raw_outputs/generic_time_dynamics_table}/${RECORD_ID}"
mkdir -p "$OUT_ROOT" logs
python - <<'PY' "$RECORD_ID" "$RECORDS_PATH" "$OUT_ROOT"
import csv, json, os, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import validate_dynamics_tuning_class
record_id, records_path, out_root = sys.argv[1:]
rows = list(csv.DictReader(Path(records_path).read_text().splitlines(), delimiter='\t'))
row = next((r for r in rows if r.get('record_id') == record_id), None)
if row is None:
    raise SystemExit(f'record_id {record_id!r} not found in {records_path}')
out = Path(out_root)
out.mkdir(parents=True, exist_ok=True)
kind = (row.get('kind') or '').strip().lower()
family = row.get('family', '').strip()
case_id = row.get('case_id', '').strip()
if row.get('tuning_class', '').strip():
    validate_dynamics_tuning_class(family=family, tuning_class=row.get('tuning_class', '').strip())
case_manifest = (row.get('case_manifest') or os.environ.get('GENERIC_TD_CASE_MANIFEST', '')).strip()
if kind == 'benchmark':
    if not family or not case_id:
        raise SystemExit(f'record {record_id!r}: family and case_id are required for benchmark')
    algorithm_id = row.get('algorithm_id', '').strip()
    if not algorithm_id:
        raise SystemExit(f'record {record_id!r}: algorithm_id is required for benchmark')
    cmd = [
        sys.executable, '-u', '-m', 'pipelines.time_dynamics.tables.generic_dynamics_benchmark',
        '--run-single',
        '--family', family,
        '--case-id', case_id,
        '--algorithm-id', algorithm_id,
        '--output-dir', str(out / 'result'),
    ]
    if case_manifest:
        cmd.extend(['--case-manifest', case_manifest])
    qiskit_supported = {
        'dyn_fixed_mclachlan',
        'dyn_product_formula_envelope',
        'dyn_qdrift',
        'dyn_fixed_pvqd',
        'dyn_adaptive_pvqd',
    }
    qiskit_mode = os.environ.get('GENERIC_TD_QISKIT_DYNAMICS_MODE', '').strip()
    if not qiskit_mode and algorithm_id in qiskit_supported:
        qiskit_mode = 'parity_required'
    if qiskit_mode:
        if qiskit_mode != 'off' and algorithm_id not in qiskit_supported:
            raise SystemExit(
                f'record {record_id!r}: Qiskit dynamics mode {qiskit_mode!r} requested for unsupported algorithm {algorithm_id!r}'
            )
        if algorithm_id in qiskit_supported:
            cmd.extend(['--qiskit-dynamics-mode', qiskit_mode])
            qiskit_qubit_cap = os.environ.get('GENERIC_TD_QISKIT_QUBIT_CAP', 'none').strip() or 'none'
            cmd.extend(['--qiskit-qubit-cap', qiskit_qubit_cap])
elif kind == 'ablation':
    if not family or not case_id:
        raise SystemExit(f'record {record_id!r}: family and case_id are required for ablation')
    cmd = [
        sys.executable, '-u', '-m', 'pipelines.time_dynamics.tables.generic_dynamics_ablation_matrix',
        '--family', family,
        '--case-id', case_id,
        '--output-dir', str(out / 'result'),
    ]
    if case_manifest:
        cmd.extend(['--case-manifest', case_manifest])
    variants = [item.strip() for item in (row.get('variants') or '').split(',') if item.strip()]
    for variant in variants:
        cmd.extend(['--variant', variant])
elif kind == 'legacy_optuna':
    legacy_record_id = (row.get('legacy_record_id') or '').strip()
    if not legacy_record_id:
        raise SystemExit(f'record {record_id!r}: legacy_record_id is required for legacy_optuna')
    legacy_records = (row.get('legacy_records') or 'chtc/time_dynamics_optuna/input/records.tsv').strip()
    cmd = [
        sys.executable, '-u', 'chtc/time_dynamics_optuna/run_task.py',
        '--record-id', legacy_record_id,
        '--records', legacy_records,
        '--output-root', str(out / 'result'),
    ]
else:
    raise SystemExit(f'record {record_id!r}: kind must be benchmark, ablation, or legacy_optuna; got {kind!r}')
manifest = __import__('os').environ.get('GENERIC_TD_CLASS_SETTINGS_MANIFEST', '').strip()
require_locked = __import__('os').environ.get('GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS', '').strip().lower() in {'1', 'true', 'yes', 'on'}
if kind in {'benchmark', 'ablation'}:
    if require_locked and not manifest:
        raise SystemExit('GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1 requires GENERIC_TD_CLASS_SETTINGS_MANIFEST')
    if manifest and not Path(manifest).exists():
        raise SystemExit(f'GENERIC_TD_CLASS_SETTINGS_MANIFEST not found: {manifest}')
    if manifest:
        cmd.extend(['--class-settings-manifest', manifest])
    if require_locked:
        cmd.append('--require-locked-class-settings')
row_with_env = dict(row)
row_with_env['effective_class_settings_manifest'] = (manifest or None) if kind in {'benchmark', 'ablation'} else None
row_with_env['require_locked_class_settings'] = bool(require_locked) if kind in {'benchmark', 'ablation'} else False
(out / 'record.json').write_text(json.dumps(row_with_env, indent=2, sort_keys=True) + '\n')
(out / 'command.sh').write_text(' '.join(shlex.quote(x) for x in cmd) + '\n')
print('RUN', ' '.join(shlex.quote(x) for x in cmd), flush=True)
subprocess.run(cmd, check=True)
status = {
    'record_id': record_id,
    'kind': kind,
    'return_code': 0,
    'result_dir': str(out / 'result'),
    'effective_class_settings_manifest': (manifest or None) if kind in {'benchmark', 'ablation'} else None,
    'require_locked_class_settings': bool(require_locked) if kind in {'benchmark', 'ablation'} else False,
}
(out / 'chtc_status.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
PY
