# Memory Budget Contract (HARD LIMIT)

User directive, 2026-08-18, after repeated machine crashes caused by
agent workloads; **limits revised 2026-08-31 by user directive for the
replacement machine** (Apple M4, 24 GB unified RAM — the crashed 16 GB
machine was lost). **All agent-launched work combined must never exceed
18 GB of resident memory.** The remainder belongs to the OS, the user's
applications, and headroom. This contract binds every agent session in
every checkout and worktree of this repository.

## Hard rules

1. **18 GB aggregate ceiling** across all agent-launched processes,
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
   python3 pipelines/shell/ram_guard.py --limit-mb 12000 -- <command...>
   ```

   Default limit is 12000 MB for a single process tree; raise it to at
   most 17000 only for a lane whose CHTC profile requests 16 GB (e.g.
   the Paper II t=10 comparator cells), and only with nothing else
   heavy running. Never exceed 17000. A killed process is an
   inconvenience; a crashed machine loses every agent's work at once.
4. **Never `json.load` GB-scale artifacts** — stream them
   (`ijson`, line-wise, or `gzip.open` + incremental parsing).
5. **Concurrent heavy jobs are allowed within the aggregate ceiling.**
   Give every process tree its own RAM guard and choose limits whose sum
   is at most 18 GB. Check current resident memory before adding another
   job; do not launch it when the projected aggregate or the 30%-free
   headroom rule would be violated. Use CHTC when a campaign cannot fit
   within those bounds.
6. Long-running local runs must checkpoint (they already do) so a
   guard kill or crash never loses more than one round.

## Precedent

The 2026-08-18 reboot killed every agent's in-flight local runs.
Before that, sustained swap pressure (millions of swapouts) had been
reported while unbounded agent workloads ran concurrently. The aggregate
ceiling and per-tree guards exist so this never recurs. The 16 GB machine
that motivated the original 10 GB ceiling was subsequently lost; the 18 GB
ceiling is the same ~75% proportion applied to the 24 GB replacement, and
newly makes single 16 GB-profile jobs locally feasible.
