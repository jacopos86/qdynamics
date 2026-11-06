# Memory use

User revision, 2026-09-19: remove the mandatory RAM guard. A 3.5 GB
agent-selected cap killed a legitimate indexer on a 24 GB Mac; the same
indexer subsequently completed at about 5.8 GB resident memory.

Run commands directly. Use actual system headroom and concurrent usage when
sizing or scheduling work. Do not wrap tasks in `pipelines/shell/ram_guard.py`
or invent a per-process kill threshold unless the user specifically requests
one. This supersedes older instructions and copied commands requiring that
wrapper, fixed allocations, or a fixed free-memory percentage gate.

Before a substantial launch, inspect `memory_pressure -Q` and relevant process
resident memory. Account for the OS, user applications, and other active jobs.
The prior 18 GB aggregate planning allowance on this 24 GB machine is context,
not an automatic process-kill rule. Respond to demonstrated memory pressure
by reducing new concurrency or moving new work to an authorized compute
backend. Preserve unrelated jobs.

Stream GB-scale artifacts instead of loading them whole. Checkpoint long
scientific runs so an interruption does not discard completed work.

The legacy wrapper remains available for old command compatibility and
explicitly requested limits; its presence does not authorize its use.
