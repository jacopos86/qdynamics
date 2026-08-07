from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--checkpoint-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    command = json.loads(args.command_json.read_text(encoding="utf-8"))
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint_json.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []

    expected = {
        "settings.problem": "molecular_vibronic_h2o_linear_fd",
        "settings.adapt_reopt_policy": "full",
        "settings.adapt_full_refit_every": 1,
        "settings.adapt_finite_angle_fallback": True,
        "settings.adapt_scipy_maxfev": 0,
        "settings.phase1_prune_enabled": False,
        "settings.phase2_enable_batching": False,
        "settings.phase2_gram_novelty_policy": "fallback_only_v1",
        "settings.phase3_gram_novelty_policy": "fallback_only_v1",
        "settings.phase3_novelty_ablation_mode": "off",
        "adapt_vqe.route_family": "singleton_response_snake",
        "adapt_vqe.adapt_beam_enabled": False,
        "adapt_vqe.accepted_refit.scope": "full_ansatz_v1",
        "adapt_vqe.accepted_refit.coordinate_chart": "supported_fs_whitened_fixed_v1",
        "adapt_vqe.accepted_refit.base_chart_policy": "expanded_runtime_projected_logical_v1",
    }
    for dotted, wanted in expected.items():
        actual = get(result, dotted)
        if actual != wanted:
            failures.append({"field": dotted, "expected": wanted, "actual": actual})

    mode = command["mode"]
    depth = int(get(result, "adapt_vqe.ansatz_depth"))
    if mode == "preflight" and depth != 12:
        failures.append({"field": "adapt_vqe.ansatz_depth", "expected": 12, "actual": depth})
    if mode == "full" and not 12 <= depth <= 30:
        failures.append({"field": "adapt_vqe.ansatz_depth", "expected": "12..30", "actual": depth})

    checkpoint_energy = float(get(checkpoint, "adapt_vqe.energy"))
    result_energy = float(get(result, "adapt_vqe.energy"))
    energy_abs_diff = abs(result_energy - checkpoint_energy)
    if mode == "preflight" and energy_abs_diff > 1.0e-8:
        failures.append(
            {
                "field": "resume_energy_abs_diff",
                "expected": "<=1e-8",
                "actual": energy_abs_diff,
            }
        )

    checkpoint_operators = get(checkpoint, "adapt_vqe.operators")
    result_operators = get(result, "adapt_vqe.operators")
    checkpoint_operator_hash = canonical_hash(checkpoint_operators)
    result_prefix_hash = canonical_hash(result_operators[:12])
    if result_prefix_hash != checkpoint_operator_hash:
        failures.append(
            {
                "field": "operator_prefix_sha256",
                "expected": checkpoint_operator_hash,
                "actual": result_prefix_hash,
            }
        )

    expected_checkpoint_sha = command["resume"]["checkpoint_sha256"]
    if sha256(args.checkpoint_json) != expected_checkpoint_sha:
        failures.append(
            {
                "field": "checkpoint_sha256",
                "expected": expected_checkpoint_sha,
                "actual": sha256(args.checkpoint_json),
            }
        )

    audit = {
        "schema": "paper_iv_h2o_source_locked_continuation_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "blocked",
        "mode": mode,
        "command_json": str(args.command_json),
        "command_sha256": sha256(args.command_json),
        "result_json": str(args.result_json),
        "result_sha256": sha256(args.result_json),
        "checkpoint_json": str(args.checkpoint_json),
        "checkpoint_sha256": sha256(args.checkpoint_json),
        "start_depth": 12,
        "result_depth": depth,
        "checkpoint_energy": checkpoint_energy,
        "result_energy": result_energy,
        "energy_abs_diff": energy_abs_diff,
        "checkpoint_operator_sha256": checkpoint_operator_hash,
        "result_prefix_operator_sha256": result_prefix_hash,
        "scientific_settings_changed_vs_stopped_run": command.get(
            "scientific_settings_changed_vs_stopped_run"
        ),
        "failures": failures,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": audit["status"], "audit": str(args.audit_json)}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
