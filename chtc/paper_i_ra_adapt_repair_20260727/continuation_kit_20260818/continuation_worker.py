"""Continuation wrapper for the sealed RA all-phase campaign workers.

Runs a package's own ``worker.py`` unmodified except that the engine's
operational resume policy is an ``AcceptedStateResume`` bound to a retrieved
mid-run checkpoint, instead of ``FreshStart``. Everything else — job
authority verification, protocol rematerialization, completion validation,
receipts — executes byte-identically to the sealed worker.

Usage (from the extracted package directory, image already entered):
    python3 continuation_worker.py \
        --package-worker worker.py \
        --resume-checkpoint <staged>/current.json \
        --job ... --protocol ... --authorization ... --output ...

The staged resume directory must contain ``current.json`` and its adjacent
``current.estimator_call_ledger_checkpoint.<hash>.json`` exactly as pulled
from the dying job's sandbox (see pull_checkpoint.sh). The checkpoint sha is
computed here and bound into the resume contract; the engine's canonical
resume loader re-authenticates it against the route contract before any
science runs, so a stale or foreign checkpoint fails closed.

Only mid-run checkpoints continue. A checkpoint at an authenticated natural
terminal is refused by the loader (by design), and a checkpoint already at
the k=50 horizon has nothing to continue — both are operator errors, not
recoverable states.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-worker", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.resume_checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"resume checkpoint missing: {checkpoint}")
    ledgers = sorted(
        checkpoint.parent.glob(
            "current.estimator_call_ledger_checkpoint.*.json"
        )
    )
    if not ledgers:
        raise SystemExit(
            "estimator ledger checkpoint must be staged next to current.json"
        )
    checkpoint_sha = _file_sha256(checkpoint)

    spec = importlib.util.spec_from_file_location(
        "sealed_package_worker", args.package_worker.resolve()
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load package worker: {args.package_worker}")
    sealed = importlib.util.module_from_spec(spec)
    sys.modules["sealed_package_worker"] = sealed
    spec.loader.exec_module(sealed)

    resume_contract = sealed.AcceptedStateResume(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
    )
    # The sealed worker's single FreshStart() call site becomes the
    # continuation contract; the class stays importable for the worker's own
    # post-run verification hydration, which uses AcceptedStateResume anyway.
    sealed.FreshStart = lambda: resume_contract

    print(
        "CONTINUATION resume bound:"
        f" checkpoint={checkpoint} sha256={checkpoint_sha}",
        flush=True,
    )
    return int(
        sealed.run(args.job, args.protocol, args.authorization, args.output)
    )


if __name__ == "__main__":
    raise SystemExit(main())
