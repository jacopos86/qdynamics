#!/usr/bin/env python3
"""Promote Optuna class-settings candidates into a locked table-run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.tables.table_lock_contract import (
    DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
    validate_class_settings_lock_manifest,
)
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID,
    DYNAMICS_SETTINGS_KIND_CONTROLLER,
    json_safe,
    normalize_dynamics_tuning_class,
    validate_dynamics_tuning_class,
)


def _load_candidate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate {path} must contain a JSON object")
    if payload.get("schema") == "dynamics_class_settings_candidate_v1":
        return dict(payload)
    if isinstance(payload.get("class_settings_candidate"), Mapping):
        return dict(payload["class_settings_candidate"])
    raise ValueError(f"candidate {path} does not contain dynamics_class_settings_candidate_v1")


_EXACT_LEAK_VALUE_TOKENS = {"exact", "exact_v1", "benchmark_exact"}


def _reject_exact_leakage(payload: Mapping[str, Any], *, source_path: Path) -> None:
    for key, value in payload.items():
        key_text = str(key).strip().lower()
        if "exact_forecast" in key_text or "exact-v1" in key_text or "exact_v1" in key_text:
            raise ValueError(f"candidate {source_path} settings_payload contains exact-assisted key {key!r}")
        if isinstance(value, str) and value.strip().lower() in _EXACT_LEAK_VALUE_TOKENS:
            raise ValueError(f"candidate {source_path} settings_payload key {key!r} uses exact-assisted value {value!r}")


def _entry(candidate: Mapping[str, Any], *, source_path: Path) -> dict[str, Any]:
    tuning_class = normalize_dynamics_tuning_class(candidate.get("tuning_class", "")) or ""
    validate_dynamics_tuning_class(family=str(candidate.get("family", tuning_class)), tuning_class=tuning_class)
    payload = candidate.get("settings_payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate {source_path} settings_payload must be a mapping")
    _reject_exact_leakage(payload, source_path=source_path)
    algorithm_id = str(candidate.get("algorithm_id", DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID))
    settings_kind = str(candidate.get("settings_kind", DYNAMICS_SETTINGS_KIND_CONTROLLER))
    if (
        algorithm_id == DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID
        and settings_kind == DYNAMICS_SETTINGS_KIND_CONTROLLER
        and not bool(candidate.get("strict_online_feedback_exact_free", False))
    ):
        raise ValueError(
            f"candidate {source_path} is not strict_online_feedback_exact_free; "
            "canonical Paper-II controller locks must be strict exact-free"
        )
    return {
        "tuning_class": tuning_class,
        "algorithm_id": algorithm_id,
        "settings_kind": settings_kind,
        "settings_source": str(candidate.get("settings_source", "paper_ii_class_tuned_optuna_v1")),
        "settings_id": candidate.get("settings_id"),
        "settings_payload": dict(payload),
        "variant_id": candidate.get("variant_id"),
        "class_tuned_result_locked": True,
        "source_candidate_json": str(source_path),
        "source_summary_json": candidate.get("source_summary_json"),
        "selected_trial_number": candidate.get("selected_trial_number"),
        "study_profile": candidate.get("study_profile"),
        "training_role": candidate.get("training_role", "class_policy_candidate"),
        "strict_online_feedback_exact_free": bool(candidate.get("strict_online_feedback_exact_free", False)),
    }


def promote(candidates: Sequence[Path], *, output: Path, note: str = "") -> dict[str, Any]:
    entries = [_entry(_load_candidate(path), source_path=path) for path in candidates]
    seen: set[tuple[str, str, str, str | None]] = set()
    for item in entries:
        key = (
            str(item["tuning_class"]),
            str(item["algorithm_id"]),
            str(item["settings_kind"]),
            None if item.get("variant_id") in {None, ""} else str(item.get("variant_id")),
        )
        if key in seen:
            raise ValueError(f"duplicate class-settings entry for {key}")
        seen.add(key)
    manifest = {
        "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
        "lock_status": "locked",
        "require_canonical_controller_classes": True,
        "lock_policy": "manual_orchestrator_promotion_after_class_optuna_review",
        "note": str(note),
        "settings": entries,
    }
    validate_class_settings_lock_manifest(manifest, require_exact_controller_classes=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = promote(args.candidate_json, output=args.output, note=str(args.note))
    print(json.dumps({"output": str(args.output), "entry_count": len(payload["settings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
