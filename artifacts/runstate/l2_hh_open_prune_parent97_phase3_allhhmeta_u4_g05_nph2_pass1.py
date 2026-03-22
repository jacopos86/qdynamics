from __future__ import annotations
import json
import sys
from pathlib import Path as _PathBoot
REPO_ROOT = _PathBoot('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pathlib import Path
import numpy as np
from dataclasses import replace

from pipelines.exact_bench import hh_l2_heavy_prune_workflow as hpw
import pipelines.exact_bench.hh_l2_logical_screen_workflow as logical_wf
import pipelines.exact_bench.hh_l2_stage_unit_audit_workflow as audit_wf
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded.hh_staged_workflow import _build_hh_context
from docs.reports.qiskit_circuit_report import adapt_ops_to_circuit

REPO = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2')
HANDOFF = REPO / 'artifacts/json/l2_hh_open_fromsnap_phase3_allhhmeta_u4_g05_nph2_adapt_handoff.json'
WARM = REPO / 'artifacts/json/l2_hh_open_teacher4_phase3_allhhmeta_u4_g05_nph2_warm_checkpoint_state.json'
OUTDIR = REPO / 'artifacts/json/hh_l2_prune_saved_parent/l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1'
OUTDIR.mkdir(parents=True, exist_ok=True)

cfg = hpw.HeavyPruneConfig(
    output_json=OUTDIR / 'summary.json',
    output_csv=OUTDIR / 'summary.csv',
    run_root=OUTDIR,
    tag='l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1',
    t=1.0,
    u=4.0,
    dv=0.0,
    omega0=1.0,
    g_ep=0.5,
    warm_ansatz='hh_hva_ptw',
    adapt_pool='all_hh_meta_v1',
    adapt_continuation_mode='phase3_v1',
    ordering='blocked',
    boundary='open',
    include_prefix_50=True,
    weakest_single_count=6,
    weakest_cumulative_count=6,
    final_vqe_reps_override=2,
    final_vqe_restarts_override=8,
    final_vqe_maxiter_override=6000,
)

staged_cfg, audit_cfg, run_dir = hpw.build_heavy_prune_staged_cfg(cfg)
run_dir.mkdir(parents=True, exist_ok=True)

warm_payload = json.loads(WARM.read_text())
handoff_payload = json.loads(HANDOFF.read_text())

# HH context
h_poly, hmat, ordered_labels_exyz, coeff_map_exyz, psi_hf = _build_hh_context(staged_cfg)

# Reconstruct states
nq = int(handoff_payload['initial_state']['nq_total'])
psi_adapt = replay_mod._amplitudes_qn_to_q0_to_statevector(handoff_payload['initial_state']['amplitudes_qn_to_q0'], nq=nq)
psi_warm = replay_mod._amplitudes_qn_to_q0_to_statevector(warm_payload['initial_state']['amplitudes_qn_to_q0'], nq=nq)

# Reconstruct selected operators in final order
base_replay_cfg = logical_wf._build_replay_cfg(staged_cfg, adapt_input_json=HANDOFF, run_dir=run_dir, ablation_id='full_replay_baseline')
adapt_labels, adapt_theta = replay_mod._extract_adapt_operator_theta_sequence(handoff_payload)
replay_terms, pool_meta, family_terms_count = replay_mod._build_replay_terms_for_family(
    base_replay_cfg,
    family='all_hh_meta_v1',
    h_poly=h_poly,
    adapt_labels=adapt_labels,
    payload=handoff_payload,
)
if len(replay_terms) != len(adapt_theta):
    raise RuntimeError(f'replay term mismatch: {len(replay_terms)} vs {len(adapt_theta)}')

# Audit reconstruction assumption: append-only path => final order == acceptance order
units = []
for idx, (term, theta) in enumerate(zip(replay_terms, adapt_theta), start=1):
    units.append(
        audit_wf._make_unit(
            stage='adapt_vqe',
            unit_index=idx,
            unit_kind='accepted_operator_insertion',
            unit_label=f'accept{idx}:{term.label}',
            base_label=str(term.label),
            theta_value=float(theta),
            polynomials=[term.polynomial],
            insertion_position=idx - 1,
            final_order_index=idx - 1,
        )
    )
full_order_ids = tuple(unit.unit_id for unit in units)
prefix_order_ids = tuple(tuple(unit.unit_id for unit in units[:k]) for k in range(1, len(units) + 1))
adapt_spec = audit_wf.StageAuditSpec(
    stage='adapt_vqe',
    reference_state=np.asarray(psi_warm, dtype=complex).reshape(-1),
    expected_full_state=np.asarray(psi_adapt, dtype=complex).reshape(-1),
    units_in_acceptance_order=tuple(units),
    full_order_ids=full_order_ids,
    prefix_order_ids=prefix_order_ids,
    reference_energy=audit_wf._state_energy(hmat, psi_warm),
    stage_metadata={
        'pool_type': 'all_hh_meta_v1',
        'continuation_mode': 'phase3_v1',
        'ansatz_depth': int(len(units)),
        'seed_prefix_depth': 0,
        'reconstruction_assumption': 'append_only_final_order_equals_acceptance_order',
    },
)
adapt_rows, adapt_stage_summary = audit_wf.compute_stage_audit_rows(adapt_spec, hmat)
ranked_adapt_units = logical_wf._rank_adapt_units(units_in_acceptance_order=units, adapt_rows=adapt_rows)
weakest_adapt_unit = logical_wf._select_weakest_adapt_unit(units_in_acceptance_order=units, adapt_rows=adapt_rows)

audit_json = OUTDIR / 'audit_from_saved_parent.json'
audit_csv = OUTDIR / 'audit_from_saved_parent.csv'
audit_payload = {
    'generated_utc': hpw._now_utc(),
    'source_handoff_json': str(HANDOFF),
    'source_warm_json': str(WARM),
    'assumption': 'append_only_final_order_equals_acceptance_order',
    'ranked_adapt_units': ranked_adapt_units,
    'weakest_adapt_unit': weakest_adapt_unit,
    'adapt_stage_summary': adapt_stage_summary,
    'adapt_rows': adapt_rows,
}
hpw._write_json(audit_json, audit_payload)
hpw._write_csv(audit_csv, adapt_rows)

audit_ctx = {
    'audit_json': audit_json,
    'audit_csv': audit_csv,
    'ranked_adapt_units': ranked_adapt_units,
    'weakest_adapt_unit': weakest_adapt_unit,
    'seed_prefix_depth': 0,
    'accepted_insertion_count': len(adapt_rows),
    'final_adapt_depth': len(units),
}

stage_metrics = {
    'warm_energy': float(warm_payload['adapt_vqe']['energy']),
    'warm_exact_energy': float(warm_payload['ground_state']['exact_energy_filtered']),
    'warm_delta_abs': float(warm_payload['adapt_vqe']['abs_delta_e']),
    'adapt_energy': float(handoff_payload['adapt_vqe']['energy']),
    'adapt_exact_energy': float(handoff_payload['ground_state']['exact_energy_filtered']),
    'adapt_delta_abs': float(handoff_payload['adapt_vqe']['abs_delta_e']),
    'final_adapt_depth': int(handoff_payload['adapt_vqe']['ansatz_depth']),
    'history_length': int(handoff_payload['adapt_vqe']['ansatz_depth']),
    'rescue_count': 0,
    'stall_step_count': None,
    'drop_low_signal_count': None,
    'depth_rollback_count': 0,
    'optimizer_memory_reuse_count': None,
    'nonpositive_marginal_gain_count': None,
}

# Parent adapt circuit cost
parent_circuit = adapt_ops_to_circuit(
    replay_terms,
    np.asarray(adapt_theta, dtype=float),
    num_qubits=int(nq),
    reference_state=np.asarray(psi_warm, dtype=complex),
)
parent_adapt_cost = hpw._circuit_cost_metrics(
    parent_circuit,
    basis_gates=tuple(str(x) for x in cfg.basis_gates),
    optimization_level=int(cfg.transpile_optimization_level),
)

# Baseline replay
baseline_diag = {}
baseline_payload = replay_mod.run(base_replay_cfg, diagnostics_out=baseline_diag)
baseline_cost = hpw._replay_circuit_cost_from_diagnostics(
    baseline_diag,
    basis_gates=tuple(str(x) for x in cfg.basis_gates),
    optimization_level=int(cfg.transpile_optimization_level),
)

plans = hpw._build_ranked_prune_plans(
    handoff_payload=handoff_payload,
    ranked_adapt_units=ranked_adapt_units,
    seed_prefix_depth=0,
    include_prefix_50=bool(cfg.include_prefix_50),
    weakest_single_count=int(cfg.weakest_single_count),
    weakest_cumulative_count=int(cfg.weakest_cumulative_count),
)
rows = []
baseline_plan = next(plan for plan in plans if str(plan.ablation_id) == 'full_replay_baseline')
rows.append(
    hpw._build_heavy_prune_row(
        cfg=cfg,
        stage_metrics=stage_metrics,
        audit_ctx=audit_ctx,
        baseline_payload=baseline_payload,
        replay_payload=baseline_payload,
        plan=baseline_plan,
        replay_output_json=Path(base_replay_cfg.output_json),
        handoff_input_json=HANDOFF,
        replay_cost=baseline_cost,
        baseline_replay_cost=baseline_cost,
    )
)

for plan in plans:
    if str(plan.ablation_id) == 'full_replay_baseline':
        continue
    handoff_path = Path(run_dir) / f'handoff_{plan.ablation_id}.json'
    hpw._write_json(handoff_path, hpw._build_heavy_prune_handoff_payload(handoff_payload, plan=plan))
    replay_cfg = logical_wf._build_replay_cfg(staged_cfg, adapt_input_json=handoff_path, run_dir=run_dir, ablation_id=str(plan.ablation_id))
    diag = {}
    replay_payload = replay_mod.run(replay_cfg, diagnostics_out=diag)
    replay_cost = hpw._replay_circuit_cost_from_diagnostics(
        diag,
        basis_gates=tuple(str(x) for x in cfg.basis_gates),
        optimization_level=int(cfg.transpile_optimization_level),
    )
    rows.append(
        hpw._build_heavy_prune_row(
            cfg=cfg,
            stage_metrics=stage_metrics,
            audit_ctx=audit_ctx,
            baseline_payload=baseline_payload,
            replay_payload=replay_payload,
            plan=plan,
            replay_output_json=Path(replay_cfg.output_json),
            handoff_input_json=handoff_path,
            replay_cost=replay_cost,
            baseline_replay_cost=baseline_cost,
        )
    )

pareto_rows = hpw._pareto_rows(rows)
payload = {
    'generated_utc': hpw._now_utc(),
    'pipeline': 'hh_l2_prune_saved_parent_pass1',
    'source_handoff_json': str(HANDOFF),
    'source_warm_json': str(WARM),
    'settings': hpw.asdict(cfg),
    'assumption': 'append_only_final_order_equals_acceptance_order',
    'parent_adapt_cost': parent_adapt_cost,
    'baseline_replay': {
        'energy': float(baseline_payload.get('vqe',{}).get('energy', float('nan'))),
        'abs_delta_e': float(baseline_payload.get('vqe',{}).get('abs_delta_e', float('nan'))),
        'num_parameters': int(baseline_payload.get('vqe',{}).get('num_parameters', 0)),
        'transpiled_depth': int(baseline_cost.get('transpiled',{}).get('depth', 0)),
        'transpiled_cx_count': int(baseline_cost.get('transpiled',{}).get('cx_count', 0)),
    },
    'artifacts': {
        'output_json': str(cfg.output_json),
        'output_csv': str(cfg.output_csv),
        'run_root': str(run_dir),
        'audit_json': str(audit_json),
        'audit_csv': str(audit_csv),
    },
    'rows': rows,
    'pareto_front': pareto_rows,
}
hpw._write_json(cfg.output_json, payload)
hpw._write_csv(cfg.output_csv, rows)
print('done', cfg.output_json)
print('baseline_delta', payload['baseline_replay']['abs_delta_e'])
print('baseline_depth', payload['baseline_replay']['transpiled_depth'])
print('baseline_cx', payload['baseline_replay']['transpiled_cx_count'])
print('pareto_count', len(pareto_rows))
