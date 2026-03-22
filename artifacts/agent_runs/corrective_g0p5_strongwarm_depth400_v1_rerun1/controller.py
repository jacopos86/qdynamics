from __future__ import annotations
import json, subprocess, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2')
BASE = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/agent_runs/corrective_g0p5_strongwarm_depth400_v1_rerun1')
MANIFEST = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/agent_runs/corrective_g0p5_strongwarm_depth400_v1_rerun1/manifest.json')
STATUS = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/agent_runs/corrective_g0p5_strongwarm_depth400_v1_rerun1/controller.status.json')
LOG = Path('/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_2/artifacts/agent_runs/corrective_g0p5_strongwarm_depth400_v1_rerun1/controller.stdout.log')
CONCURRENCY = 3
POLL_S = 20


def now_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_status(obj):
    STATUS.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    manifest = json.loads(MANIFEST.read_text())
    queue = list(manifest['runs'])
    running = []
    completed = []
    failed = []
    with LOG.open('a') as log:
        log.write(f"QUEUE_START {now_utc()} total={len(queue)}\n")
        while queue or running:
            while queue and len(running) < CONCURRENCY:
                run = queue.pop(0)
                tag = run['tag']
                stdout_path = BASE / f"{tag}.stdout.log"
                status_run = BASE / f"{tag}.status.json"
                with stdout_path.open('w') as f:
                    f.write(f"LAUNCH {now_utc()}\n")
                    f.write('CMD: ' + ' '.join(run['argv']) + '\n')
                    f.flush()
                    proc = subprocess.Popen(['nice','-n','10', *run['argv']], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                run_state = dict(run)
                run_state.update({'pid': proc.pid, 'started_utc': now_utc(), 'stdout_log': str(stdout_path), 'status': 'running'})
                status_run.write_text(json.dumps(run_state, indent=2) + '\n')
                running.append((proc, run_state, status_run))
                log.write(f"START {run['tag']} pid={proc.pid} at={run_state['started_utc']}\n")
                log.flush()
            new_running = []
            for proc, run_state, status_run in running:
                ret = proc.poll()
                if ret is None:
                    new_running.append((proc, run_state, status_run))
                    continue
                run_state['ended_utc'] = now_utc()
                run_state['returncode'] = int(ret)
                run_state['status'] = 'completed' if ret == 0 else 'failed'
                status_run.write_text(json.dumps(run_state, indent=2) + '\n')
                log.write(f"END {run_state['tag']} rc={ret} at={run_state['ended_utc']}\n")
                log.flush()
                if ret == 0:
                    completed.append(run_state)
                else:
                    failed.append(run_state)
            running = new_running
            write_status({
                'updated_utc': now_utc(),
                'queue_remaining': [r['tag'] for r in queue],
                'running': [r['tag'] for _, r, _ in running],
                'completed': [r['tag'] for r in completed],
                'failed': [r['tag'] for r in failed],
                'total': len(manifest['runs']),
                'concurrency': CONCURRENCY,
            })
            time.sleep(POLL_S)
        log.write(f"QUEUE_DONE {now_utc()} completed={len(completed)} failed={len(failed)}\n")
        log.flush()
        write_status({
            'updated_utc': now_utc(),
            'queue_remaining': [],
            'running': [],
            'completed': [r['tag'] for r in completed],
            'failed': [r['tag'] for r in failed],
            'total': len(manifest['runs']),
            'concurrency': CONCURRENCY,
            'done': True,
        })

if __name__ == '__main__':
    main()
