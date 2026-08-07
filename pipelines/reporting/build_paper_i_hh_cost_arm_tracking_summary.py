#!/usr/bin/env python3
"""Build one pass-only tracking summary for a macro/beam/prune cost arm.

The source ``result.json`` for these historical-beam rows is multi-gigabyte.
This builder therefore consumes a separately streamed, compact trajectory
receipt plus the small v9 revalidation artifacts.  It never opens or
materializes ``result.json``.  The raw transfer archive is still hashed and its
identity is required to agree across the trajectory receipt, v9 revalidation
receipt, and generated validation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "paper_i_hh_cost_arm_tracking_summary_v1"
TRAJECTORY_SCHEMA = "paper_i_sr_macro_beam_cost_compact_trajectory_v1"
REVALIDATION_SCHEMAS = {
    "symmetric": "paper_i_sr_macro_beam_cost_v9_v6_archive_revalidation_v1",
    "one_sided": "paper_i_sr_macro_beam_cost_v10_v6_archive_revalidation_v1",
}
REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
ARM_CONTRACTS = {
    "symmetric": {
        "profile_contract_sha256": (
            "a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0"
        ),
        "cost_mode": "family_robust_symmetric_arctan_v1",
        "fallback_policy": "collective_span_novelty_over_symmetric_cost_v1",
        "behavioral_closure": "full_response_validated_each_controller_round_v1",
    },
    "one_sided": {
        "profile_contract_sha256": (
            "e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096"
        ),
        "cost_mode": "family_robust_v1",
        "fallback_policy": "collective_span_novelty_over_cost_v1",
        "behavioral_closure": "full_response_route_policy_locked_in_normalized_contract_v1",
    },
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is nonfinite: {value!r}")
    return parsed


def _integer(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    return parsed


def _resolved_declared_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "sha256": _sha256_path(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_generated_artifacts(
    *, receipt: Mapping[str, Any], receipt_path: Path
) -> dict[str, dict[str, Any]]:
    generated = receipt.get("generated_reporting_artifacts")
    if not isinstance(generated, Mapping):
        raise ValueError("v9 receipt lacks generated_reporting_artifacts")
    required = {
        "validation.json",
        "qiskit_cost_sidecar.json",
        "ground_space_projector_fidelity.json",
        "terminal_checkpoint.execution_order_repaired.json",
    }
    if not required.issubset(generated):
        raise ValueError(
            "v9 receipt lacks generated artifacts: "
            + ", ".join(sorted(required - set(generated)))
        )
    identities: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        declared = generated[name]
        if not isinstance(declared, Mapping):
            raise TypeError(f"generated artifact identity is malformed: {name}")
        path = receipt_path.parent / name
        if not path.is_file():
            raise FileNotFoundError(path)
        identity = _artifact_identity(path)
        if identity["sha256"] != str(declared.get("sha256") or ""):
            raise ValueError(f"generated artifact SHA-256 drift: {name}")
        if identity["size_bytes"] != _integer(
            declared.get("size_bytes"), label=f"{name} size"
        ):
            raise ValueError(f"generated artifact size drift: {name}")
        identities[name] = identity
    return identities


def _normalized_trajectory(compact: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = compact.get("trajectory")
    if not isinstance(raw, list) or len(raw) != 50:
        raise ValueError("compact cost-arm trajectory must contain exactly 50 rows")
    trajectory: list[dict[str, Any]] = []
    prior_s_alg = -1
    for expected_round, row in enumerate(raw, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"compact trajectory row {expected_round} is not an object")
        if _integer(row.get("round"), label="trajectory round") != expected_round:
            raise ValueError(f"compact trajectory order drift at round {expected_round}")
        active_depth = _integer(
            row.get("active_depth"), label=f"round {expected_round} active depth"
        )
        if active_depth < 0:
            raise ValueError(f"negative active depth at round {expected_round}")
        prune_accepted = row.get("prune_accepted")
        if not isinstance(prune_accepted, bool):
            raise ValueError(
                f"round {expected_round} prune_accepted must be boolean"
            )
        winning_s_alg = _integer(
            row.get("winning_lineage_S_alg", row.get("S_alg")),
            label=f"round {expected_round} winning-lineage S_alg",
        )
        if row.get("S_alg") is not None and _integer(
            row.get("S_alg"), label=f"round {expected_round} legacy S_alg"
        ) != winning_s_alg:
            raise ValueError(
                f"round {expected_round} legacy/winning-lineage S_alg drift"
            )
        s_alg = _integer(
            row.get("all_branch_S_alg", row.get("S_alg")),
            label=f"round {expected_round} all-branch S_alg",
        )
        if s_alg < prior_s_alg:
            raise ValueError(f"nonmonotone S_alg at round {expected_round}")
        prior_s_alg = s_alg
        trajectory.append(
            {
                "round": expected_round,
                "error": abs(
                    _finite(row.get("error"), label=f"round {expected_round} error")
                ),
                "active_depth": active_depth,
                "prune_accepted": prune_accepted,
                "S_alg": s_alg,
                "winning_lineage_S_alg": winning_s_alg,
            }
        )
    return trajectory


def _normalized_selected_winner_history(
    compact: Mapping[str, Any], *, selected_round: int
) -> list[dict[str, Any]]:
    raw = compact.get("selected_winner_history")
    if not isinstance(raw, list) or len(raw) != selected_round:
        raise ValueError(
            "one-sided compact receipt lacks the selected winner's exact history"
        )
    history: list[dict[str, Any]] = []
    prior_s_alg = -1
    for expected_round, row in enumerate(raw, start=1):
        if not isinstance(row, Mapping):
            raise TypeError("selected-winner history row is not an object")
        round_id = _integer(row.get("round"), label="selected-winner round")
        active_depth = _integer(
            row.get("active_depth"), label="selected-winner active depth"
        )
        winning_s_alg = _integer(
            row.get("winning_lineage_S_alg"),
            label="selected-winner winning-lineage S_alg",
        )
        checkpoint_sha = str(row.get("checkpoint_sha256") or "")
        if (
            round_id != expected_round
            or active_depth < 0
            or winning_s_alg < prior_s_alg
            or len(checkpoint_sha) != 64
        ):
            raise ValueError("selected-winner history identity/order drift")
        prior_s_alg = winning_s_alg
        history.append(
            {
                "round": round_id,
                "error": abs(
                    _finite(row.get("error"), label="selected-winner error")
                ),
                "active_depth": active_depth,
                "winning_lineage_S_alg": winning_s_alg,
                "checkpoint_sha256": checkpoint_sha,
            }
        )
    return history


def build_tracking_summary(
    *,
    archive_path: Path,
    revalidation_receipt_path: Path,
    compact_trajectory_path: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Validate one objectively passed row and create its tracker projection."""

    archive_path = archive_path.resolve()
    revalidation_receipt_path = revalidation_receipt_path.resolve()
    compact_trajectory_path = compact_trajectory_path.resolve()
    for path in (archive_path, revalidation_receipt_path, compact_trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    receipt = _read_json(revalidation_receipt_path)
    compact = _read_json(compact_trajectory_path)
    if (
        receipt.get("schema") not in set(REVALIDATION_SCHEMAS.values())
        or receipt.get("status") != "pass"
    ):
        raise ValueError("cost-arm revalidation receipt is not an objective pass")
    if receipt.get("scientific_rerun_required") is not False:
        raise ValueError("cost-arm row still requires a scientific rerun")
    if receipt.get("raw_transfer_archive_preserved") is not True:
        raise ValueError("v9 receipt does not preserve the raw transfer archive")
    if compact.get("schema") != TRAJECTORY_SCHEMA or compact.get("status") != "pass":
        raise ValueError("cost-arm compact trajectory receipt is not an objective pass")
    if _integer(compact.get("controller_rounds"), label="compact controller rounds") != 50:
        raise ValueError("cost-arm compact trajectory does not cover 50 controller rounds")

    regime = str(receipt.get("regime_slug") or "")
    if regime not in REGIMES or compact.get("regime_slug") != regime:
        raise ValueError("cost-arm regime identity drift")
    expected_n_ph = 3 if regime in REGIMES[:3] else 7
    digest = str(receipt.get("profile_contract_sha256") or "")
    arm = next(
        (
            name
            for name, contract in ARM_CONTRACTS.items()
            if contract["profile_contract_sha256"] == digest
        ),
        None,
    )
    if arm is None:
        raise ValueError(f"unsupported cost-arm route digest: {digest!r}")
    contract = ARM_CONTRACTS[arm]
    if receipt.get("schema") != REVALIDATION_SCHEMAS[arm]:
        raise ValueError("cost-arm route/revalidation-schema drift")
    if compact.get("profile_contract_sha256") != digest:
        raise ValueError("compact trajectory route digest drift")

    archive_identity = compact.get("raw_transfer_archive")
    if isinstance(archive_identity, Mapping):
        compact_archive_path = archive_identity.get("path")
        compact_archive_sha = archive_identity.get("sha256")
        compact_archive_size = archive_identity.get("size_bytes")
    else:
        compact_archive_path = archive_identity
        compact_archive_sha = compact.get("raw_transfer_archive_sha256")
        compact_archive_size = archive_path.stat().st_size
    declared_receipt_archive = _resolved_declared_path(
        receipt.get("raw_transfer_archive"), base=revalidation_receipt_path.parent
    )
    declared_compact_archive = _resolved_declared_path(
        compact_archive_path, base=REPO_ROOT
    )
    if declared_receipt_archive != archive_path or declared_compact_archive != archive_path:
        raise ValueError("cost-arm raw archive path drift")
    # The v9 validator already hashes the immutable 400--700 MB raw archive
    # before and after reporting repair.  Reopening it here would defeat the
    # bounded reporting path; require the two validator hashes and the compact
    # streaming receipt hash to agree instead.
    declared_archive_shas = {
        str(receipt.get("raw_transfer_archive_sha256_before") or ""),
        str(receipt.get("raw_transfer_archive_sha256_after") or ""),
        str(compact_archive_sha or ""),
    }
    if len(declared_archive_shas) != 1 or "" in declared_archive_shas:
        raise ValueError("cost-arm raw archive SHA-256 drift")
    archive_sha = next(iter(declared_archive_shas))
    if len(archive_sha) != 64:
        raise ValueError("cost-arm raw archive SHA-256 is malformed")
    if _integer(compact_archive_size, label="raw archive size") != archive_path.stat().st_size:
        raise ValueError("cost-arm raw archive size drift")

    result_member = compact.get("result_member")
    if isinstance(result_member, Mapping):
        result_member_name = str(result_member.get("name") or "")
        result_sha = str(result_member.get("sha256") or "")
        result_size = result_member.get("size_bytes")
    else:
        result_member_name = str(compact.get("source_result_member_name") or "")
        result_sha = str(compact.get("source_result_sha256") or "")
        result_size = compact.get("source_result_size_bytes")
    if not result_member_name.endswith(f"/{regime}/json/result.json"):
        raise ValueError("compact trajectory result-member name drift")
    if len(result_sha) != 64 or (_integer(result_size, label="result member size") <= 0):
        raise ValueError("compact trajectory result-member identity is incomplete")

    generated = _validate_generated_artifacts(
        receipt=receipt, receipt_path=revalidation_receipt_path
    )
    validation = _read_json(revalidation_receipt_path.parent / "validation.json")
    qiskit = _read_json(revalidation_receipt_path.parent / "qiskit_cost_sidecar.json")
    fidelity = _read_json(
        revalidation_receipt_path.parent / "ground_space_projector_fidelity.json"
    )
    executable_checkpoint_path = (
        revalidation_receipt_path.parent
        / "terminal_checkpoint.execution_order_repaired.json"
    )
    executable_checkpoint = _read_json(executable_checkpoint_path)
    if validation.get("status") != "pass" or _integer(
        validation.get("controller_horizon_round"), label="controller horizon"
    ) != 50:
        raise ValueError("cost-arm scientific validation did not pass 50 rounds")
    if validation.get("result_sha256") != result_sha:
        raise ValueError("validation/result-member SHA-256 drift")
    if qiskit.get("status") != "ok" or fidelity.get("status") != "pass":
        raise ValueError("cost-arm Qiskit or fidelity validation did not pass")
    if (
        qiskit.get("source", {}).get("result_sha256") != result_sha
        or fidelity.get("source_result_sha256") != result_sha
    ):
        raise ValueError("post-run reporting artifacts reference another result")

    scientific = receipt.get("scientific_evidence_validation")
    if not isinstance(scientific, Mapping):
        raise ValueError("v9 receipt lacks scientific evidence validation")
    source_runtime = receipt.get("source_only_runtime_settings_receipt")
    if not isinstance(source_runtime, Mapping):
        raise ValueError("v9 receipt lacks source-only runtime settings")
    runtime_settings = source_runtime.get("source_only_runtime_settings")
    if not isinstance(runtime_settings, Mapping):
        raise ValueError("source-only runtime settings are malformed")
    if (
        source_runtime.get("status") != "pass"
        or source_runtime.get("profile_contract_sha256") != digest
        or source_runtime.get("phase_live_hysteresis_disabled") is not True
        or source_runtime.get("behavioral_closure")
        != contract["behavioral_closure"]
        or runtime_settings.get("adapt_beam_live_branches") != 3
        or runtime_settings.get("adapt_beam_children_per_parent") != 2
        or runtime_settings.get("phase0_pilot_enabled") is not False
        or runtime_settings.get("phase3_enable_batching") is not False
        or runtime_settings.get("adapt_accepted_refit_coordinate_chart")
        != "supported_fs_whitened_fixed_v1"
    ):
        raise ValueError("source-only runtime route contract drift")

    compact_history_receipt = scientific.get("compact_current_history_receipt")
    active_receipts = scientific.get("active_prefix_estimator_ledger_receipts")
    ledger = scientific.get("ledger")
    if not isinstance(compact_history_receipt, Mapping):
        raise ValueError("v9 receipt lacks compact-current-history validation")
    if not isinstance(active_receipts, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError("v9 receipt lacks estimator-ledger validation")
    if (
        _integer(scientific.get("controller_rounds"), label="controller rounds") != 50
        or compact_history_receipt.get("status") != "pass"
        or _integer(compact_history_receipt.get("rounds"), label="history rounds") != 50
        or active_receipts.get("closure_passed") is not True
        or _integer(active_receipts.get("controller_horizon_rounds"), label="ledger horizon") != 50
        or scientific.get("expected_cost_mode") != contract["cost_mode"]
        or scientific.get("expected_fallback_policy") != contract["fallback_policy"]
        or _integer(ledger.get("finite_angle_guard_occurrence_count"), label="finite angle occurrences") != 0
        or abs(_finite(scientific.get("max_binary_padding_leakage"), label="padding leakage")) > 1.0e-10
        or abs(_finite(scientific.get("max_fixed_sector_leakage"), label="sector leakage")) > 1.0e-10
    ):
        raise ValueError("cost-arm scientific validation contract drift")

    selected_round = _integer(
        scientific.get("selected_final_controller_round"), label="selected round"
    )
    selected_depth = _integer(
        scientific.get("selected_final_active_depth"), label="selected depth"
    )
    all_branch_s_alg = _integer(
        active_receipts.get("all_branch_S_alg"), label="all-branch S_alg"
    )
    if (
        selected_round != _integer(validation.get("selected_winner_round"), label="validation selected round")
        or selected_round != _integer(compact.get("selected_final_controller_round"), label="compact selected round")
        or selected_depth != _integer(compact.get("selected_final_active_depth"), label="compact selected depth")
        or all_branch_s_alg != _integer(compact.get("all_branch_S_alg"), label="compact S_alg")
        or all_branch_s_alg != _integer(ledger.get("all_branch_s_alg"), label="ledger S_alg")
        or _integer(compact.get("winning_lineage_S_alg"), label="compact winning S_alg")
        != _integer(ledger.get("winning_lineage_s_alg"), label="ledger winning S_alg")
    ):
        raise ValueError("compact trajectory disagrees with selected winner or ledger")

    trajectory = _normalized_trajectory(compact)
    if trajectory[-1]["S_alg"] != all_branch_s_alg:
        raise ValueError("compact trajectory terminal S_alg drift")
    accepted_prune_rounds = sum(point["prune_accepted"] for point in trajectory)
    trajectory_role = "selected_terminal_path_v1"
    selected_winner_history: list[dict[str, Any]] = []
    controller_frontier: dict[str, Any] | None = None
    if arm == "one_sided":
        trajectory_role = "controller_frontier_non_selected_v1"
        raw_frontier = compact.get("controller_frontier")
        raw_selected = compact.get("selected_terminal")
        if not isinstance(raw_frontier, Mapping) or not isinstance(
            raw_selected, Mapping
        ):
            raise ValueError("one-sided compact receipt lacks split terminal identities")
        if compact.get("controller_frontier_trajectory") != compact.get("trajectory"):
            raise ValueError("one-sided compact frontier trajectory aliases drift")
        selected_winner_history = _normalized_selected_winner_history(
            compact, selected_round=selected_round
        )
        controller_frontier = {
            "status": str(raw_frontier.get("status") or ""),
            "round": _integer(raw_frontier.get("round"), label="frontier round"),
            "active_depth": _integer(
                raw_frontier.get("active_depth"), label="frontier depth"
            ),
            "error": abs(
                _finite(
                    raw_frontier.get("same_cutoff_absolute_error"),
                    label="frontier error",
                )
            ),
            "all_branch_S_alg": _integer(
                raw_frontier.get("all_branch_S_alg"), label="frontier S_alg"
            ),
            "eligible_for_selected_terminal_cost_reporting": raw_frontier.get(
                "eligible_for_selected_terminal_cost_reporting"
            ),
        }
        if (
            controller_frontier["status"]
            != "non_selected_recoverable_frontier"
            or controller_frontier["round"] != 50
            or controller_frontier["active_depth"]
            != trajectory[-1]["active_depth"]
            or not math.isclose(
                float(controller_frontier["error"]),
                trajectory[-1]["error"],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or controller_frontier["all_branch_S_alg"] != all_branch_s_alg
            or controller_frontier[
                "eligible_for_selected_terminal_cost_reporting"
            ]
            is not False
            or _integer(raw_selected.get("round"), label="selected terminal round")
            != selected_round
            or _integer(
                raw_selected.get("active_depth"), label="selected terminal depth"
            )
            != selected_depth
            or _integer(
                raw_selected.get("winning_lineage_S_alg"),
                label="selected terminal winning S_alg",
            )
            != _integer(
                ledger.get("winning_lineage_s_alg"), label="ledger winning S_alg"
            )
            or selected_winner_history[-1]["active_depth"] != selected_depth
            or selected_winner_history[-1]["winning_lineage_S_alg"]
            != _integer(
                ledger.get("winning_lineage_s_alg"), label="terminal winning S_alg"
            )
            or accepted_prune_rounds
            != _integer(
                scientific.get("controller_frontier_prune_rounds_accepted"),
                label="frontier accepted prune rounds",
            )
        ):
            raise ValueError("one-sided selected-terminal/frontier identity drift")
    else:
        if trajectory[-1]["active_depth"] != selected_depth:
            raise ValueError("compact trajectory terminal depth drift")
        if trajectory[-1]["winning_lineage_S_alg"] != _integer(
            ledger.get("winning_lineage_s_alg"), label="terminal winning S_alg"
        ):
            raise ValueError(
                "compact trajectory terminal winning-lineage S_alg drift"
            )
        if accepted_prune_rounds != _integer(
            scientific.get("selected_prune_rounds_accepted"),
            label="selected accepted prune rounds",
        ):
            raise ValueError("compact trajectory prune-acceptance count drift")
    construction = compact.get("construction_receipt")
    selected_identity = compact.get("selected_prefix_identity")
    if not isinstance(construction, Mapping) or not isinstance(
        selected_identity, Mapping
    ):
        raise ValueError("compact trajectory lacks construction/prefix identities")
    if (
        construction.get("all_branch_ledger_closure_matches_v9") is not True
        or construction.get("winning_lineage_ledger_closure_matches_v9") is not True
        or construction.get("archive_hash_matches_v9_before_and_after") is not True
        or construction.get("trajectory_rounds_exactly_1_through_50") is not True
        or _integer(
            construction.get("all_branch_unique_checkpoint_receipt_count"),
            label="all-branch receipt count",
        )
        != _integer(active_receipts.get("receipt_count"), label="v9 receipt count")
        or _integer(
            construction.get("estimator_ledger_unique_primitive_entry_count"),
            label="unique primitive entry count",
        )
        != all_branch_s_alg
        or selected_identity.get("ledger_fingerprint")
        != ledger.get("ledger_fingerprint")
        or selected_identity.get("terminal_checkpoint_sha256")
        != scientific.get("selected_terminal_checkpoint_sha256")
    ):
        raise ValueError("compact trajectory construction/ledger closure drift")
    exact_energy = _finite(validation.get("same_cutoff_exact_energy"), label="exact energy")
    fixed_replay = validation.get("fixed_prefix_replay")
    if not isinstance(fixed_replay, Mapping) or fixed_replay.get("status") != "pass":
        raise ValueError("fixed-prefix replay did not pass")
    reported_energy = _finite(fixed_replay.get("reported_energy"), label="reported energy")
    terminal_error = abs(reported_energy - exact_energy)
    selected_error_point = (
        selected_winner_history[-1]
        if trajectory_role == "controller_frontier_non_selected_v1"
        else trajectory[-1]
    )
    if not math.isclose(
        selected_error_point["error"],
        terminal_error,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("compact selected-terminal error drift")
    if trajectory_role == "controller_frontier_non_selected_v1":
        assert isinstance(raw_selected, Mapping)
        if (
            not math.isclose(
                abs(
                    _finite(
                        raw_selected.get("same_cutoff_absolute_error"),
                        label="selected terminal error",
                    )
                ),
                terminal_error,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _finite(raw_selected.get("energy"), label="selected terminal energy"),
                reported_energy,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                _finite(
                    raw_selected.get("same_cutoff_exact_energy"),
                    label="selected terminal exact energy",
                ),
                exact_energy,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or raw_selected.get("checkpoint_sha256")
            != scientific.get("selected_terminal_checkpoint_sha256")
            or _integer(
                raw_selected.get("all_branch_S_alg_at_controller_horizon"),
                label="selected terminal all-branch horizon S_alg",
            )
            != all_branch_s_alg
        ):
            raise ValueError("one-sided compact selected-terminal metric drift")
    if _integer(fixed_replay.get("active_ansatz_depth"), label="replay depth") != selected_depth:
        raise ValueError("fixed-prefix replay active-depth drift")

    repair = executable_checkpoint.get("repair")
    checkpoint_source = executable_checkpoint.get("source")
    checkpoint = executable_checkpoint.get("repaired_checkpoint")
    if not all(
        isinstance(record, Mapping)
        for record in (repair, checkpoint_source, checkpoint)
    ):
        raise ValueError("executable terminal checkpoint is malformed")
    checkpoint_ledger = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(checkpoint_ledger, Mapping):
        raise ValueError("executable terminal checkpoint lacks ledger identity")
    operator_labels = checkpoint.get("ordered_active_operator_labels")
    operators = checkpoint.get("ordered_active_operators")
    logical_parameters = checkpoint.get("signed_unwrapped_logical_parameters")
    runtime_parameters = checkpoint.get("signed_unwrapped_runtime_parameters")
    if (
        executable_checkpoint.get("schema")
        != "paper_i_checkpoint_execution_order_repair_v1"
        or repair.get("status") not in {"repaired_permutation_only", "not_required"}
        or repair.get("substantive_term_changes") is not False
        or repair.get("source_checkpoint_sha256")
        != selected_identity.get("terminal_checkpoint_sha256")
        or checkpoint.get("checkpoint_sha256")
        != repair.get("repaired_checkpoint_sha256")
        or checkpoint_source.get("result_sha256") != result_sha
        or _integer(checkpoint_source.get("outer_iteration"), label="checkpoint source round")
        != selected_round
        or checkpoint.get("schema") != "paper_i_signed_active_prefix_checkpoint_v1"
        or _integer(checkpoint.get("outer_iteration"), label="checkpoint round")
        != selected_round
        or _integer(checkpoint.get("active_ansatz_depth"), label="checkpoint active depth")
        != selected_depth
        or checkpoint.get("sr_route_profile_contract_sha256") != digest
        or checkpoint_ledger.get("status") != "complete"
        or _integer(checkpoint_ledger.get("outer_iteration"), label="checkpoint ledger round")
        != selected_round
        or not isinstance(operator_labels, list)
        or not isinstance(operators, list)
        or not isinstance(logical_parameters, list)
        or not isinstance(runtime_parameters, list)
        or len(operator_labels) != selected_depth
        or len(operators) != selected_depth
        or len(logical_parameters) != selected_depth
        or not runtime_parameters
    ):
        raise ValueError("executable terminal checkpoint identity/content drift")
    current_metrics = validation.get("current_fake_marrakesh_metrics")
    qiskit_metrics = qiskit.get("current_jr_fake_marrakesh_convention", {}).get(
        "metrics"
    )
    if not isinstance(current_metrics, Mapping) or not isinstance(qiskit_metrics, Mapping):
        raise ValueError("cost-arm current Qiskit metrics are missing")
    costs = {
        metric: _integer(current_metrics.get(metric), label=f"Qiskit {metric}")
        for metric in ("N2q", "D2q", "Dc")
    }
    if costs != {
        metric: _integer(qiskit_metrics.get(metric), label=f"sidecar Qiskit {metric}")
        for metric in ("N2q", "D2q", "Dc")
    }:
        raise ValueError("validation/Qiskit-sidecar metric drift")
    if trajectory_role == "controller_frontier_non_selected_v1":
        assert isinstance(raw_selected, Mapping)
        selected_qiskit = raw_selected.get("qiskit")
        if not isinstance(selected_qiskit, Mapping):
            raise ValueError("one-sided selected terminal lacks compact Qiskit identity")
        compact_costs = {
            "N2q": _integer(selected_qiskit.get("N2q"), label="compact N2q"),
            "D2q": _integer(selected_qiskit.get("D2q"), label="compact D2q"),
            "Dc": _integer(
                selected_qiskit.get("total_compiled_depth"), label="compact Dc"
            ),
        }
        if (
            compact_costs != costs
            or len(str(selected_qiskit.get("sidecar_sha256") or "")) != 64
        ):
            raise ValueError("one-sided selected-terminal compact Qiskit drift")
    historical_metrics = validation.get("historical_metrics")
    sidecar_historical = qiskit.get("historical_displayed_convention", {}).get(
        "metrics"
    )
    if not isinstance(historical_metrics, Mapping) or not isinstance(
        sidecar_historical, Mapping
    ):
        raise ValueError("cost-arm historical prefix Qiskit metrics are missing")
    prefix_costs = {
        metric: _integer(historical_metrics.get(metric), label=f"prefix Qiskit {metric}")
        for metric in ("N2q", "D2q", "Dc")
    }
    if prefix_costs != {
        metric: _integer(
            sidecar_historical.get(metric), label=f"sidecar prefix Qiskit {metric}"
        )
        for metric in ("N2q", "D2q", "Dc")
    }:
        raise ValueError("validation/historical-Qiskit metric drift")
    fidelity_value = _finite(fidelity.get("fidelity"), label="fidelity")
    if trajectory_role == "controller_frontier_non_selected_v1":
        assert isinstance(raw_selected, Mapping)
        if not math.isclose(
            _finite(raw_selected.get("fidelity"), label="compact fidelity"),
            fidelity_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("one-sided selected-terminal compact fidelity drift")
    ground_fidelity = fidelity.get("ground_space_fidelity")
    if (
        not isinstance(ground_fidelity, Mapping)
        or ground_fidelity.get("same_cutoff_verified") is not True
        or _integer(ground_fidelity.get("working_cutoff"), label="working cutoff") != expected_n_ph
        or _integer(ground_fidelity.get("reference_cutoff"), label="reference cutoff") != expected_n_ph
    ):
        raise ValueError("cost-arm same-cutoff fidelity contract drift")

    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "identity": {
            "arm": arm,
            "regime": regime,
            "n_ph_work": expected_n_ph,
            "n_ph_reference": expected_n_ph,
            "same_cutoff_reference": True,
            "profile_contract_sha256": digest,
            "cost_mode": contract["cost_mode"],
            "fallback_policy": contract["fallback_policy"],
        },
        "archive": {
            "path": _display_path(archive_path),
            "sha256": archive_sha,
            "size_bytes": archive_path.stat().st_size,
        },
        "executable_source": _artifact_identity(executable_checkpoint_path),
        "result_member": {
            "name": result_member_name,
            "sha256": result_sha,
            "size_bytes": _integer(result_size, label="result member size"),
        },
        "revalidation_receipt": _artifact_identity(revalidation_receipt_path),
        "compact_trajectory_receipt": _artifact_identity(compact_trajectory_path),
        "generated_reporting_artifacts": generated,
        "validation": {
            "schema": validation.get("schema"),
            "status": "pass",
            "controller_rounds": 50,
            "selected_final_controller_round": selected_round,
            "selected_final_active_depth": selected_depth,
            "all_branch_S_alg": all_branch_s_alg,
            "winning_lineage_S_alg": _integer(
                ledger.get("winning_lineage_s_alg"), label="winning-lineage S_alg"
            ),
            "prune_rounds_executed": _integer(
                scientific.get("selected_prune_rounds_executed"), label="prune rounds executed"
            ),
            "prune_rounds_accepted": _integer(
                scientific.get("selected_prune_rounds_accepted"), label="prune rounds accepted"
            ),
            "controller_frontier_active_depth": (
                _integer(
                    scientific.get("controller_frontier_active_depth"),
                    label="frontier active depth",
                )
                if arm == "one_sided"
                else selected_depth
            ),
            "controller_frontier_prune_rounds_executed": (
                _integer(
                    scientific.get("controller_frontier_prune_rounds_executed"),
                    label="frontier prune rounds executed",
                )
                if arm == "one_sided"
                else _integer(
                    scientific.get("selected_prune_rounds_executed"),
                    label="selected prune rounds executed",
                )
            ),
            "controller_frontier_prune_rounds_accepted": (
                _integer(
                    scientific.get("controller_frontier_prune_rounds_accepted"),
                    label="frontier prune rounds accepted",
                )
                if arm == "one_sided"
                else _integer(
                    scientific.get("selected_prune_rounds_accepted"),
                    label="selected prune rounds accepted",
                )
            ),
            "max_binary_padding_leakage": _finite(
                scientific.get("max_binary_padding_leakage"), label="padding leakage"
            ),
            "max_fixed_sector_leakage": _finite(
                scientific.get("max_fixed_sector_leakage"), label="sector leakage"
            ),
        },
        "result": {
            "status": "complete",
            "n_ph": expected_n_ph,
            "rounds": 50,
            "active_depth": selected_depth,
            "terminal_error": terminal_error,
            "s_alg": all_branch_s_alg,
            "s_alg_scope": "validated all-branch beam search work including accepted prune trials",
            "fidelity": fidelity_value,
            "trajectory_role": trajectory_role,
            "trajectory": trajectory,
            "selected_winner_history": (
                selected_winner_history if selected_winner_history else None
            ),
            "selected_terminal": {
                "round": selected_round,
                "active_depth": selected_depth,
                "error": terminal_error,
                "winning_lineage_S_alg": _integer(
                    ledger.get("winning_lineage_s_alg"),
                    label="selected winning-lineage S_alg",
                ),
                "selection_authority_all_branch_S_alg": all_branch_s_alg,
            },
            "controller_frontier": controller_frontier,
        },
        "qiskit": costs,
        "terminal_prefix_qiskit": prefix_costs,
    }
    if output_json is not None:
        output_json = output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_name(output_json.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--revalidation-receipt", type=Path, required=True)
    parser.add_argument("--compact-trajectory", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    summary = build_tracking_summary(
        archive_path=args.archive,
        revalidation_receipt_path=args.revalidation_receipt,
        compact_trajectory_path=args.compact_trajectory,
        output_json=args.output_json,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_json": str(args.output_json.resolve()),
                "archive_sha256": summary["archive"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
