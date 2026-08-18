# Memory Budget Contract (HARD LIMIT)

User directive, 2026-08-18, after repeated machine crashes caused by
agent workloads: this Mac has 16 GB physical RAM and can sustain at
most ~15 GB in use. **All agent-launched work combined must never
exceed 10 GB of resident memory.** The remainder belongs to the OS,
the user's applications, and headroom. This contract binds every agent
session in every checkout and worktree of this repository.

## Hard rules

1. **10 GB aggregate ceiling** across all agent-launched processes,
   all sessions combined. Assume other agent sessions are running:
   your budget is a share, not the whole.
2. **Check before launching anything potentially heavy** (indexers,
   integration test suites, run harnesses, plotting over large data):

   ```bash
   memory_pressure -Q | head -2
   ```

   If system free memory is below ~30%, do not launch heavy work at
   all — wait or move the job to CHTC.
3. **Wrap heavy processes in the RAM guard**, which kills the process
   tree before it can take the machine down:

   ```bash
   python3 pipelines/shell/ram_guard.py --limit-mb 8000 -- <command...>
   ```

   Default limit is 8000 MB for a single process tree; never raise it
   above 10000. A killed process is an inconvenience; a crashed machine
   loses every agent's work at once.
4. **Never `json.load` GB-scale artifacts** — stream them
   (`ijson`, line-wise, or `gzip.open` + incremental parsing).
5. **One heavy job at a time, per machine — not per session.** No
   parallel test suites, no parallel local runs. Big compute belongs on
   CHTC, not this Mac.
6. Long-running local runs must checkpoint (they already do) so a
   guard kill or crash never loses more than one round.

## Precedent

The 2026-08-18 reboot killed every agent's in-flight local runs.
Before that, sustained swap pressure (millions of swapouts) had been
reported while multiple agent workloads ran concurrently. The guard
exists so this never recurs.
