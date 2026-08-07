from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pipelines.excited_dynamics.schemas import (
    build_excited_state_seed_manifest,
    validate_excited_state_seed_manifest,
    validate_qse_result_manifest,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_qse_result_json(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = load_json(path)
    validate_qse_result_manifest(payload)
    return payload, sha256_file(path)


def write_excited_state_seed_from_qse_json(
    *,
    qse_json_path: str | Path,
    output_json_path: str | Path,
    state_index: int,
    allow_ground_state: bool = False,
) -> dict[str, Any]:
    qse_payload, qse_hash = load_qse_result_json(qse_json_path)
    seed = build_excited_state_seed_manifest(
        qse_payload,
        state_index=state_index,
        source_qse_path=str(qse_json_path),
        source_qse_sha256=qse_hash,
        allow_ground_state=allow_ground_state,
    )
    validate_excited_state_seed_manifest(seed)
    write_json(output_json_path, seed)
    return seed
