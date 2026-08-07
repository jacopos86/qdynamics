# Stale bundle — do not submit

This immutable v2 bundle predates the approved SR-SNAKE v4 Phase-I/II
energy-model contract that removes active `lambda_F F` curvature proxies.
It is retained only as historical build and submission provenance.

- CHTC cluster `8816561` was removed while all six jobs were still idle.
- No process recorded an exit code or remote wall time.
- `submit.sub` is deliberately fail-closed with `requirements = False`.
- A corrected bundle must be built as a new immutable revision; this directory
  must not be repaired in place or submitted again.
