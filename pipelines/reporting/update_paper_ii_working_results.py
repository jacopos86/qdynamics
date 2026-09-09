"""Append validated t=20 arrivals to the working PDF, once per raw trajectory."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

from pipelines.reporting.prepare_paper_ii_working_results import prepare, sha, write


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def add_once(arms, candidate):
    """Keep page order; never replace an existing run with different evidence."""
    for arm in arms:
        if arm['id'] == candidate['id']:
            require(arm['raw_sha256'] == candidate['raw_sha256'], 'Existing run ID has changed evidence')
            return False
        if arm['raw_sha256'] == candidate['raw_sha256']:
            return False
    arms.append(candidate)
    return True


def validate_arrival(receipt_path, package, manifest):
    receipt=read(receipt_path)
    require(receipt['source_package_sha256'] == manifest['source']['archive_sha256'], 'Wrong source archive')
    require(receipt['code_revision'] == manifest['source_revision'], 'Wrong source revision')
    actions={a['action_id']:a for a in map(json.loads,(package/'input/actions.jsonl').read_text().splitlines())}
    require(receipt['action_id'] in actions, 'Unknown action')
    action=actions[receipt['action_id']]
    effective=action['effective_arguments']
    require(effective['t_final'] == 20 and effective['num_times'] == 501, 'Not a complete t=20 action')
    archive=receipt_path.parent/Path(receipt['archive']).name
    require(sha(archive) == receipt['archive_sha256'] == receipt_path.parent.name, 'Archive digest mismatch')
    extraction=receipt_path.parent/'extracted'
    # Re-extract from the verified archive. Refuse links and traversal.
    with tarfile.open(archive,'r:gz') as stream:
        for item in stream.getmembers():
            require(item.isfile() or item.isdir(), 'Archive contains a link or special file')
            require(not Path(item.name).is_absolute() and '..' not in Path(item.name).parts, 'Unsafe archive path')
        stream.extractall(extraction,filter='data')
    raw=extraction/action['trajectory_json']; directory=raw.parent
    completion=read(directory/'completion_receipt.json')
    require(completion['status'] == 'SUCCESS_EXACT', 'Worker did not complete exact reporting')
    for key in ('action_id','campaign_id','cell_id','code_revision','source_package_sha256',
                'matched_cell_lock_sha256','exchange_route_bundle'):
        require(completion[key] == action[key], 'Completion identity mismatch: '+key)
    for name,key in [('run.json','trajectory_sha256'),('terminal_qiskit_cost.json','terminal_cost_sha256'),
                     ('exact_estimator_work.json','estimator_work_sha256')]:
        require(sha(directory/name) == completion[key], 'Completed artifact digest mismatch: '+name)
    projection=read(directory/'reporting_projection_receipt.json')
    require(projection['source_sha256'] == completion['trajectory_sha256'], 'Projection has a different source')
    require(sha(directory/'reporting_projection.json') == projection['projection_sha256'], 'Projection digest mismatch')
    report=read(directory/'exploratory_summary.json')
    require(report['raw_run_sha256'] == completion['trajectory_sha256'], 'Report has a different source')
    require(report['scientific_config_sha256'] == action['scientific_config_sha256'], 'Report settings mismatch')
    require(report['source_files_sha256'] == manifest['source']['files_sha256'], 'Report source map mismatch')
    require(report['work'] == read(directory/'exact_estimator_work.json')['estimator_work'], 'Worker work ledgers disagree')
    label_search={'greedy':'greedy','block_exchange':'block','bounded_exhaustive':'exhaustive'}[effective['search_policy']]
    label_gain={'joint_batch':'joint','singleton_factorized':'factorized'}[effective['gain_evaluation']]
    label=f"APM - {label_search}, {label_gain}, W{effective['max_joint_patch_evaluations']}"
    candidate=dict(id='new_'+action['action_id'],label=label,table_label='APM - new',route='q_compiled',
        raw=str(raw),raw_sha256=completion['trajectory_sha256'],completion=str(directory/'completion_receipt.json'),
        stored_cost=str(directory/'terminal_qiskit_cost.json'),worker_projection_receipt=str(directory/'reporting_projection_receipt.json'),
        effective_arguments=effective,code_revision=action['code_revision'],action_id=action['action_id'],
        archive_receipt=str(receipt_path),archive_sha256=receipt['archive_sha256'],
        work_budget=effective['max_joint_patch_evaluations'],color='#D55E00',linestyle='-',
        receipt_kind='chtc_worker',scientific_config_sha256=action['scientific_config_sha256'])
    write(receipt_path.parent/'report_arrival_validation.json',dict(status='PASS',raw_sha256=candidate['raw_sha256'],
        archive_sha256=receipt['archive_sha256'],action_id=action['action_id'],
        scope='Source/action/artifact identity; scientific validation runs during report preparation.'))
    return candidate


def discover(cfg):
    added=[]; failures=[]
    known={a.get('archive_sha256') for a in cfg['arms']}
    for source in cfg.get('arrival_sources',[]):
        package=Path(source['package']); manifest=read(package/'input/manifest.json')
        require(sha(package/'input/manifest.json') == source['manifest_sha256'], 'Package manifest changed')
        require(sha(package/'input/actions.jsonl') == manifest['package_files']['input/actions.jsonl'], 'Actions changed')
        require(sha(package/manifest['source']['archive']) == manifest['source']['archive_sha256'], 'Source archive changed')
        for receipt in sorted(Path(source['arrivals']).glob('*/*.archive.json'),key=lambda p:(p.stat().st_mtime,str(p))):
            if receipt.parent.name in known:
                continue
            try:
                candidate=validate_arrival(receipt,package,manifest)
                if add_once(cfg['arms'],candidate):
                    added.append(candidate['id']); known.add(candidate['archive_sha256'])
            except (ValueError,KeyError,OSError,tarfile.TarError) as error:
                failures.append(dict(receipt=str(receipt),error=str(error)))
    return added,failures


def update(inputs, output, verify_python, rebuild=False):
    output.mkdir(parents=True,exist_ok=True)
    with (output/'refresh.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='ALREADY_REFRESHING')
        cfg=read(inputs)
        added,failures=discover(cfg)
        if failures:
            write(output/'arrival_review_required.json',dict(status='REVIEW_REQUIRED',failures=failures))
        # Repeated ticks do not build duplicate pages or re-score known evidence.
        if not added and not rebuild and (output/'pdf_build_receipt.json').exists():
            return dict(status='REVIEW_REQUIRED' if failures else 'UNCHANGED',failures=failures)
        discovered={arm['id']:arm for arm in cfg['arms'] if arm['id'] in added}
        cfg['arms']=[arm for arm in cfg['arms'] if arm['id'] not in discovered]
        published=[]
        for identity in (added or [None]):
            if identity is not None:
                cfg['arms'].append(discovered[identity])
            try:
                receipt,staging=publish(cfg,inputs,output,verify_python)
            except (ValueError,AssertionError,KeyError,OSError,subprocess.CalledProcessError) as error:
                if identity is None:
                    raise
                cfg['arms'].pop()
                failures.append(dict(run_id=identity,error=str(error)))
                continue
            if identity is not None:
                published.append(identity)
        if failures:
            write(output/'arrival_review_required.json',dict(status='REVIEW_REQUIRED',failures=failures))
        receipt=read(output/'pdf_build_receipt.json')
        result=dict(status='UPDATED' if published or rebuild else 'REVIEW_REQUIRED',added=published,
            page_count=len(receipt['pages']),pdf_sha256=receipt['pdf_sha256'],failures=failures)
        write(output/'refresh_receipt.json',result)
        return result

def publish(cfg, inputs, output, verify_python):
    cfg['data_directory']=str(output/'data')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    staging=output/'builds'/stamp; staging.mkdir(parents=True)
    staged_inputs=staging/'inputs.json'; write(staged_inputs,cfg)
    prepare(staged_inputs,staging)
    from pipelines.reporting.build_paper_ii_new_route_results import build
    build(staging/'results.json')
    subprocess.run([verify_python,'-B','-m','pipelines.reporting.verify_paper_ii_working_results',
        '--root',str(staging)],check=True)
    # All validation precedes replacing the stable PDF. Retain every build.
    receipt=read(staging/'pdf_build_receipt.json')
    receipt['pdf']=str(output/'paper_ii_new_route_results.pdf')
    write(staging/'pdf_build_receipt.json',receipt)
    for name in ['results.json','paper_ii_new_route_results.tex','paper_ii_new_route_results.log',
                 'provenance_notes.md','calibration_receipt.json','comparability.json',
                 'verification_receipt.json','pdf_build_receipt.json','paper_ii_new_route_results.pdf']:
        temporary=output/('.'+name+'.tmp')
        shutil.copy2(staging/name,temporary); os.replace(temporary,output/name)
    for figure in staging.glob('observables_*.pdf'):
        shutil.copy2(figure,output/figure.name)
    temporary=inputs.with_suffix('.json.tmp'); write(temporary,cfg); os.replace(temporary,inputs)
    return receipt, staging



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-python',required=True)
    parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args()
    print(json.dumps(update(args.inputs.resolve(),args.output.resolve(),args.verify_python,args.rebuild),indent=2))
