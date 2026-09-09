"""Prepare source-bound Paper II working-report data; never runs a trajectory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def project(raw, target):
    """Stream raw evidence, omitting only two unused ranking arrays."""
    import ijson
    from ijson.common import ObjectBuilder
    digest = sha(raw)
    receipt_path = target.with_name('projection_receipt.json')
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['raw_sha256'] == digest and receipt['projection_sha256'] == sha(target):
            return json.loads(target.read_text()), receipt
    level = 'trajectory.points.item.patch_decision.metadata.selector_route.levels.item'
    omitted = {level + '.ranked', level + '.population_receipt.normalization.records'}
    builder = ObjectBuilder()
    skip = None
    counts = Counter()
    with raw.open('rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if skip:
                if prefix == skip + '.item' and event == 'start_map':
                    counts[skip] += 1
                if prefix == skip and event == 'end_array':
                    builder.event(event, None)
                    skip = None
                continue
            builder.event(event, value)
            if prefix in omitted and event == 'start_array':
                skip = prefix
    if skip:
        raise ValueError('Unfinished raw ranking array')
    write(target, builder.value)
    receipt = dict(raw_path=str(raw), raw_sha256=digest, raw_bytes=raw.stat().st_size,
                   projection_path=str(target), projection_sha256=sha(target),
                   omitted_array_record_counts=dict(counts),
                   purpose='Reporting only. Full raw selection evidence remains unchanged.')
    write(receipt_path, receipt)
    return builder.value, receipt


def prepare(config_path, output):
    import numpy as np
    from pipelines.time_dynamics.diagnostics.exact_reference_store import ExactReferenceStore
    from pipelines.time_dynamics.diagnostics.physical_errors import physical_errors
    from pipelines.time_dynamics.diagnostics.paper_ii_four_route_comparison import (
        complete_estimator_work, deletion_gate_outcomes, support_motion)
    from pipelines.time_dynamics.diagnostics.q_compiled_manuscript_assets import (
        _generator_motion, _reuse_adjusted_estimator_total)
    from pipelines.time_dynamics.diagnostics.ap_terminal_qiskit_cost import build_ap_terminal_cost_table
    from pipelines.time_dynamics.diagnostics.paper_ii_benchmark_calibration import run_calibration
    from pipelines.time_dynamics.run_lock import assert_comparable

    cfg = json.loads(config_path.read_text())
    repo = Path(cfg['repository'])
    seed = Path(cfg['seed'])
    assert sha(seed) == cfg['seed_sha256']
    assert sha(cfg['manuscript_table']) == cfg['manuscript_table_sha256']
    reference_manifest = json.loads((Path(cfg['exact_reference_store']) /
        cfg['exact_reference_key'] / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    records, locks, calibration_paths = [], [], []
    for arm in cfg['arms']:
        print('Preparing ' + arm['id'], flush=True)
        raw = Path(arm['raw'])
        directory = Path(cfg.get('data_directory', output / 'data')) / arm['id']
        projected = directory / 'run.json'
        run, receipt = project(raw, projected)
        assert receipt['raw_sha256'] == arm['raw_sha256'], 'source digest mismatch'
        completion = json.loads(Path(arm['completion']).read_text())
        bound_hash = completion.get('trajectory_sha256', completion.get('output_sha256'))
        assert bound_hash == receipt['raw_sha256'], 'completion does not bind this run'
        assert completion.get('status') in {'SUCCESS_EXACT', 'SUCCESS', 'COMPLETED_PENDING_SCIENTIFIC_VALIDATION'}
        points, rows, summary = run['trajectory']['points'], run['plot_rows'], run['summary']
        assert len(points) == len(rows) == 501 and rows[0]['time'] == 0 and rows[-1]['time'] == 20
        assert all(run['decision_data_flow'].get(k) is False for k in (
            'uses_reference_for_decision', 'uses_exact_reference_for_decision', 'uses_future_exact_forecast_for_decision'))
        lock = run['run_lock']
        locks.append(lock)
        assert lock['physics']['seed_sha256'] == cfg['seed_sha256']
        times = [row['time'] for row in rows]
        # Saved float grids differ at roundoff; use the paper's immutable reference.
        assert np.allclose(times, reference_manifest['times'], rtol=0, atol=1e-12)
        ref = ExactReferenceStore(cfg['exact_reference_store']).get_or_build(
            seed_path=seed, run_path=projected, times=reference_manifest['times'])
        assert ref.key == cfg['exact_reference_key'] and ref.data_sha256 == cfg['exact_reference_sha256']
        metrics = physical_errors(rows, ref.rows).to_json_dict()
        observed_sites = np.array([row['site_occupations'] for row in rows])
        exact_sites = np.array([row['site_occupations'] for row in ref.rows])
        dn_error = float(np.mean(np.abs((observed_sites[:, 0] - observed_sites[:, 1]) -
                                        (exact_sites[:, 0] - exact_sites[:, 1]))))
        # The dimer's conserved total charge makes the paper's shortcut exact.
        assert abs(dn_error - 2 * metrics['mean_max_abs_site_occupation_error']) < 1e-11
        metrics['mean_abs_charge_imbalance_error'] = dn_error
        route = arm['route']
        work = complete_estimator_work([run], route=route)
        assert work['logical_total_estimator_work_status'] == 'exact'
        assert work['selection_round_ledger_complete']
        if arm.get('stored_cost'):
            cost_path = Path(arm['stored_cost'])
            assert sha(cost_path) == completion['terminal_cost_sha256']
            cost = json.loads(cost_path.read_text())
        else:
            cost_path = directory / 'terminal_qiskit_cost.json'
            if cost_path.exists():
                cost = json.loads(cost_path.read_text())
            else:
                print('Compiling terminal circuit ' + arm['id'], flush=True)
                cost = build_ap_terminal_cost_table(trajectory_path=projected, label=arm['id'],
                    backend_name='FakeMarrakesh', seed_transpiler=7, optimization_level=1,
                    source_artifact_path=seed)
                write(cost_path, cost)
        assert cost['compile_defaults']['backend_name'] == 'FakeMarrakesh'
        assert cost['compile_defaults']['optimization_level'] == 1
        assert cost['compile_defaults']['seed_transpiler'] == 7
        assert cost['rows'][0]['terminal_reconstruction_parity']['passed']
        cost_source_hash = arm['raw_sha256'] if arm.get('stored_cost') else sha(projected)
        if arm.get('worker_projection_receipt'):
            worker_receipt = json.loads(Path(arm['worker_projection_receipt']).read_text())
            assert worker_receipt['source_sha256'] == arm['raw_sha256']
            worker_projection = Path(arm['worker_projection_receipt']).with_name('reporting_projection.json')
            assert sha(worker_projection) == worker_receipt['projection_sha256']
            cost_source_hash = worker_receipt['projection_sha256']
        assert cost['rows'][0]['trajectory_json_sha256'] == cost_source_hash, 'stale terminal-cost cache'
        assert cost['rows'][0]['source_artifact_json_sha256'] == cfg['seed_sha256']
        route_record = dict(route=route, estimator_work=work, run_sources=[dict(path=str(projected))])
        adjusted = _reuse_adjusted_estimator_total(route_record)
        assert adjusted is not None
        motion = _generator_motion(route_record)
        gates = deletion_gate_outcomes([run], applicable=route != 'avqds')
        validation = dict(grid_points=501, horizon=20, raw_hash_verified=True,
            decision_reference_separation=True, terminal_reconstruction=True,
            exact_all_call_estimator_ledger=True,
            max_state_norm_error=max(abs(sum(x*x for x in row['psi_real']) +
                                           sum(x*x for x in row['psi_imag'])-1) for row in rows),
            terminal_fixed_step_warning_count=summary.get('unsupported_fixed_step_count'),
            integration_unsupported_fields={k:v for k,v in summary.items()
                if ('unsupported' in k or 'uncured' in k) and not isinstance(v,(dict,list))})
        if arm['id'].startswith('new_'):
            import importlib.util
            spec = importlib.util.spec_from_file_location('working_report_receipt_contracts', cfg['receipt_auditor'])
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            if arm.get('effective_arguments'):
                effective = arm['effective_arguments']
            else:
                case = json.loads(Path(arm['case']).read_text())
                from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import _build_parser
                effective = vars(_build_parser().parse_args(case['runner_argv']))
            for key in ('search_policy', 'gain_evaluation', 'max_insertion_batch_size',
                        'max_joint_patch_evaluations', 'patch_rung_schedule'):
                assert lock['policy']['guards'][key] == effective[key], 'effective setting drift: '+key
            assert effective['max_insertion_batch_size'] == 1, 'Batch reuse accounting requires separate review'
            if arm.get('code_revision'):
                assert lock['code_revision'] == arm['code_revision'], 'worker source identity drift'
            searches, commits = module.audit_search_receipts(run, {'effective_arguments':effective})
            write(directory/'search_validation.json', dict(searches=searches, commits=commits))
            validation.update(search_passes_verified=len(searches), committed_patches_verified=len(commits))
        calibrated = dict(method=arm['id'], regime='hh_snake_nph1', drive='fastweak',
            controller='state_motion_5e-5', horizon='t20', seed_sha256=cfg['seed_sha256'],
            seed_source=str(seed), time_grid=times, n_ph_max=1,
            reference_policy='independent_exact_post_run', observable_set=['energy','doublon','charge_imbalance'],
            reference_key=ref.key, compile_target='FakeMarrakesh-opt1-seed7', **metrics)
        calibration_paths.append(str(projected))
        record = dict(**arm, raw_receipt=receipt, metrics=metrics, work=work,
            reuse_adjusted_work=adjusted, motion=motion, coordinate_motion=support_motion([run]),
            deletion_gates=gates, terminal_cost=cost['rows'][0], cost_path=str(cost_path),
            cost_sha256=sha(cost_path), steps=summary['accepted_internal_substep_count'],
            attempted_steps=summary['attempted_internal_step_count'],
            n_theta=summary['active_parameter_count_final'], validation=validation,
            code_revision=lock['code_revision'], run_lock=lock,
            calibration_row=calibrated, times=times,
            series=dict(energy=[row['energy_expectation'] for row in rows],
                        doublon=[row['doublon'] for row in rows],
                        imbalance=(observed_sites[:,0]-observed_sites[:,1]).tolist()))
        write(directory/'record.json', record)
        records.append(record)
        print(arm['id'], metrics, 'resources', {k:record['terminal_cost'][k] for k in ('N2q','D2q','Dc')}, flush=True)
        del run
    # Raw locks remain unmodified; version differences must be investigated explicitly.
    try:
        comparable = dict(status='PASS', fingerprint=assert_comparable(locks))
    except ValueError as error:
        comparable = dict(status='REVIEW_REQUIRED', reason=str(error))
    write(output/'comparability.json', comparable)
    calibration_input = output/'calibration_input.json'
    write(calibration_input, dict(rows=[r['calibration_row'] for r in records],
        author_override='Working report requested by author: show completed rows independent of matched-accuracy qualification; no evidence promotion.'))
    calibration = run_calibration(root=repo, artifact_paths=[str(calibration_input), *calibration_paths])
    write(output/'calibration_receipt.json', calibration)
    print('Calibration', calibration['status'], [(x['check_id'],x['status']) for x in calibration['checks']], flush=True)
    dataset = dict(schema='paper_ii_new_route_working_results_v1', config=cfg,
        config_sha256=sha(config_path), producer_sha256=sha(__file__),
        records=records, comparability=comparable, calibration_status=calibration['status'],
        exact=dict(key=ref.key, data_sha256=ref.data_sha256, manifest=str(ref.manifest_path), times=times,
            series=dict(energy=[row['energy'] for row in ref.rows],doublon=[row['doublon'] for row in ref.rows],
                imbalance=[row['site_occupations'][0]-row['site_occupations'][1] for row in ref.rows])))
    write(output/'results.json', dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.inputs,args.output)
