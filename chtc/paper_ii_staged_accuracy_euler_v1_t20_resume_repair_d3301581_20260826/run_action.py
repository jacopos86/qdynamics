#!/usr/bin/env python3
import json, subprocess, sys
from pathlib import Path
action_id = sys.argv[1] if len(sys.argv) == 2 else None
if not action_id: raise SystemExit('usage: run_action.py ACTION_ID')
rows = Path('chtc/paper_ii_staged_accuracy_euler_v1_t20_resume_repair_d3301581_20260826/input/actions.jsonl').read_text(encoding='utf-8').splitlines()
records = [json.loads(row) for row in rows]
record = next((row for row in records if row['action_id'] == action_id), None)
if record is None: raise SystemExit(f'unknown action_id {action_id}')
command = [sys.executable, '-m', record['runner_module'], *record['runner_argv']]
if not record['runner_argv']: raise SystemExit('empty runner argv')
subprocess.run(command, check=True)
trajectory = Path(record['trajectory_json'])
if not trajectory.is_file() or trajectory.stat().st_size == 0:
    raise SystemExit(f'runner returned without artifact: {trajectory}')
if record['compile_terminal_cost']:
    profile = {'profile_id': 'paper_ii_fake_marrakesh_fixed_patch_v1', 'backend_name': 'FakeMarrakesh', 'optimization_level': 1, 'seed_transpiler': 7, 'layout_policy': 'shared_initial_seed_fixed_patch_v1', 'graph_proxy_profile_id': 'paper_i_fake_marrakesh_graph_span_v1', 'native_basis': ['sx', 'rz', 'x', 'cz', 'id'], 'source': 'agent_guidance/time-dynamics/paper-ii-comparison-protocol.md', 'note': 'Paper-II terminal-resource profile: the full ansatz including reference-state preparation receives the Paper-I analytical graph proxy and is also transpiled to FakeMarrakesh on the physical patch selected from the common initial seed.', 'primary_reported_metrics': ['absolute_c_hat_2q', 'absolute_c_hat_d', 'absolute_c_hat_1q', 'absolute_theta_count'], 'secondary_reported_metrics': ['N2q', 'D2q', 'Dc']}
    cost_command = [
        sys.executable, '-m',
        'pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost',
        '--trajectory-json', str(trajectory),
        '--output-cost-table-json', str(trajectory.with_name('terminal_qiskit_cost.json')),
        '--label', action_id,
        '--backend-name', str(profile['backend_name']),
        '--seed-transpiler', str(profile['seed_transpiler']),
        '--optimization-level', str(profile['optimization_level']),
    ]
    subprocess.run(cost_command, check=True)
