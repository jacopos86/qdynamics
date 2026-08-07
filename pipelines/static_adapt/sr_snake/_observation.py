"""Shared post-run observation projection for SR-SNAKE execution paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pipelines.static_adapt.sr_snake._context import (
    _ResolvedExecutionContext,
)
from pipelines.static_adapt.sr_snake.contracts import (
    ObservationArtifactReceipt,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_receipt(
    *,
    kind: str,
    path: Path,
    every_controller_rounds: int | None = None,
) -> ObservationArtifactReceipt:
    return ObservationArtifactReceipt(
        kind=kind,
        path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        every_controller_rounds=every_controller_rounds,
    )


def _write_estimator_ledger_sidecar(
    path: Path,
    adapt_payload: Mapping[str, Any],
) -> None:
    accounting_raw = adapt_payload.get("estimator_call_accounting")
    if not isinstance(accounting_raw, Mapping):
        raise RuntimeError(
            "The characterized SR-SNAKE route did not return estimator accounting."
        )
    accounting = dict(accounting_raw)
    full_ledger = accounting.pop("full_ledger", None)
    if not isinstance(full_ledger, Mapping):
        raise RuntimeError(
            "The characterized SR-SNAKE route did not return its full estimator ledger."
        )
    sidecar = {
        "schema": "paper_i_estimator_call_ledger_sidecar_v2",
        "accounting": accounting,
        "ledger": dict(full_ledger),
        "adapt_success": bool(adapt_payload.get("success", False)),
        "adapt_error": adapt_payload.get("error"),
    }
    encoded = (json.dumps(sidecar, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _project_observation_artifacts(
    context: _ResolvedExecutionContext,
    payload: Mapping[str, Any],
) -> tuple[ObservationArtifactReceipt, ...]:
    """Materialize requested sidecars and return their portable receipts."""

    _prepare_observation_destinations(context)
    checkpoint = context.observation.checkpoint
    ledger = context.observation.estimator_ledger

    artifacts: list[ObservationArtifactReceipt] = []
    if checkpoint is not None:
        if not checkpoint.path.is_file():
            raise RuntimeError(
                "The characterized SR-SNAKE route did not write the requested "
                f"checkpoint: {checkpoint.path}"
            )
        artifacts.append(
            _artifact_receipt(
                kind="accepted_state_checkpoint",
                path=checkpoint.path,
                every_controller_rounds=checkpoint.every_controller_rounds,
            )
        )

    if ledger is not None:
        _write_estimator_ledger_sidecar(ledger.path, payload)
        artifacts.append(
            _artifact_receipt(
                kind="estimator_ledger",
                path=ledger.path,
            )
        )
    return tuple(artifacts)


def _prepare_observation_destinations(
    context: _ResolvedExecutionContext,
) -> None:
    """Create only the parent directories explicitly requested by policy."""

    checkpoint = context.observation.checkpoint
    if checkpoint is not None:
        checkpoint.path.parent.mkdir(parents=True, exist_ok=True)
    ledger = context.observation.estimator_ledger
    if ledger is not None:
        ledger.path.parent.mkdir(parents=True, exist_ok=True)


__all__: list[str] = []
