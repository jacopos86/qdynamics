"""Optional Append-ADAPT/SNAKE overlays for the compact Geo report.

Only explicitly supplied result roots are searched.  Append plateau-prefix
ansatze are reconstructed coefficient-aware from their fetched runtime seeds.  SNAKE
trajectories are recovered from the emitted winner-checkpoint events, while
cost rows require a hash-linked native terminal payload and the same
coefficient-aware Table-I compilation convention.  No arbitrary SNAKE prefix
is reconstructed from labels, and device-mapped compile-scout counts are never
mixed into the common adjacent table.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.reporting.build_paper_i_geo_scaling_evidence import (
    PLOT_ERROR_FLOOR,
    compile_prefix_qiskit,
    prefix_query_ledger,
    read_json,
    reconstruct_structural_prefix,
    rel,
    resolve_runtime_seed_path,
    select_first_plateau,
    sha256,
    trajectory_points,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = ("Append-ADAPT", "Geo-ADAPT", "SNAKE")
METHOD_KEYS = {"Append-ADAPT": "append", "Geo-ADAPT": "geo", "SNAKE": "snake"}
METHOD_ALGORITHMS = {
    "append": "static_full_meta_append_adapt_vqe",
    "snake": "static_family_native_adapt_phase3",
}
MAX_DIRECT_NATIVE_RESULT_BYTES = 128 * 1024 * 1024
NATIVE_SIDECAR_NAMES = (
    "snake_native_compact_sidecar.json",
    "snake_terminal_compact_sidecar.json",
    "paper_i_compact_sidecar.json",
)
L2_SNAKE_REGIME_TO_CANONICAL_CASE = {
    "weak_weak": "hh_L2_nph2_three_model_sym_weak_weak",
    "intermediate_weak": "hh_L2_nph2_three_model_sym_strong_weak",
    "strong_weak": "hh_L2_nph2_three_model_sym_u8_strong_weak",
    "weak_strong": "hh_L2_nph4_three_model_sym_weak_strong",
    "intermediate_strong": "hh_L2_nph4_three_model_sym_strong_strong",
    "strong_strong": "hh_L2_nph4_three_model_sym_u8_strong_strong",
}


class OverlayBlocked(RuntimeError):
    """A fetched overlay cannot be displayed under the locked contract."""


def _finite(value: Any, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise OverlayBlocked(f"{label} is not finite: {value!r}")
    return out


def _normalized_curve(points: Sequence[tuple[int, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    previous = -1
    for k, error in points:
        x = int(k)
        value = abs(_finite(error, "trajectory error"))
        if x <= previous:
            raise OverlayBlocked(f"trajectory iterations are not strictly increasing: {x} <= {previous}")
        previous = x
        out.append(
            {
                "k": x,
                "error_raw": value,
                "error_plotted": max(value, PLOT_ERROR_FLOOR),
            }
        )
    if not out:
        raise OverlayBlocked("trajectory is empty")
    return out


def _record_contract(record_dir: Path) -> dict[str, Any]:
    path = record_dir / "cell_manifest.json"
    if not path.is_file():
        return {"status": "blocked:missing_cell_manifest", "path": rel(path), "row": None}
    payload = read_json(path)
    row = payload.get("row")
    if not isinstance(row, Mapping):
        return {"status": "blocked:missing_cell_manifest_row", "path": rel(path), "row": None}
    if str(payload.get("status")) != "ok" or int(payload.get("returncode", -1)) != 0:
        return {
            "status": "blocked:cell_not_completed",
            "path": rel(path),
            "row": dict(row),
        }
    return {
        "status": "ok",
        "path": rel(path),
        "sha256": sha256(path),
        "row": dict(row),
        "env_overlay": (
            dict(payload.get("env_overlay"))
            if isinstance(payload.get("env_overlay"), Mapping)
            else {}
        ),
        "command": list(payload.get("command") or []),
    }


def _validate_record_contract(
    contract: Mapping[str, Any],
    *,
    method_key: str,
    expected_horizon: int,
    case_id: str,
) -> dict[str, Any]:
    row = contract.get("row")
    if not isinstance(row, Mapping):
        raise OverlayBlocked(str(contract.get("status") or "missing cell-manifest row"))
    common_expected = {
        "method_key": method_key,
        "algorithm_id": METHOD_ALGORITHMS[method_key],
        "optimizer": "POWELL",
        "budget": "200",
        "pool_contract": "full_meta_unfiltered",
        "shared_pauli_pool_mode": "off",
    }
    problems = [
        f"{key}={row.get(key)!r} expected {value!r}"
        for key, value in common_expected.items()
        if str(row.get(key) or "") != value
    ]
    if method_key == "snake" and str(case_id).startswith("hh_L2_"):
        # These five already-completed L=2 SNAKE rows intentionally use the
        # archival singleton child-set route.  They are useful diagnostics,
        # but are not a parent-only policy match to Geo/Append.
        child_expected = {
            "child_policy": "native_phase3_singleton",
            "child_subset_size": "1",
            "snake_phase3_runtime_split_mode": "shortlist_pauli_children_v1",
            "snake_phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
            "snake_phase3_runtime_split_max_subset_size": "1",
            "snake_phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        }
        problems.extend(
            f"{key}={row.get(key)!r} expected {value!r}"
            for key, value in child_expected.items()
            if str(row.get(key) or "") != value
        )
        horizon_observed = row.get("expected_horizon") or row.get("max_depth")
        row_out = dict(row)
        row_out["overlay_policy_comparability"] = "mixed_child_set_diagnostic"
        row_out["common_pauli_child_policy"] = False
    else:
        parent_expected = {
            "child_policy": "macro_only",
            "generic_adapt_runtime_split_mode": "off",
        }
        problems.extend(
            f"{key}={row.get(key)!r} expected {value!r}"
            for key, value in parent_expected.items()
            if str(row.get(key) or "") != value
        )
        parent_policy = str(row.get("parent_generator_policy") or "")
        legacy_parent_evidence = False
        if parent_policy != "full_meta_parent_macro_generators_only_all_methods":
            env = contract.get("env_overlay")
            record_id = str(row.get("record_id") or "")
            legacy_parent_evidence = (
                method_key == "append"
                and str(case_id).startswith("hh_L2_")
                and not parent_policy
                and record_id.endswith("__fullmeta_parent")
                and isinstance(env, Mapping)
                and str(env.get("GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE"))
                == "off"
                and str(env.get("GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"))
                == "full_meta_unfiltered"
            )
            if not legacy_parent_evidence:
                problems.append(
                    "parent_generator_policy="
                    f"{row.get('parent_generator_policy')!r} expected "
                    "'full_meta_parent_macro_generators_only_all_methods'"
                )
        horizon_observed = row.get("expected_horizon") or row.get("max_depth")
        row_out = dict(row)
        row_out["overlay_policy_comparability"] = (
            "legacy_l2_parent_macro_only_env_verified"
            if legacy_parent_evidence
            else "parent_macro_only_matched"
        )
        row_out["common_pauli_child_policy"] = True
    if int(horizon_observed or -1) != int(expected_horizon):
        problems.append(
            f"expected_horizon/max_depth={horizon_observed!r} expected {expected_horizon}"
        )
    if problems:
        raise OverlayBlocked("cell-manifest contract mismatch: " + "; ".join(problems))
    return row_out


def index_explicit_roots(
    roots: Sequence[Path], *, method_key: str
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Index fetched comparator records under explicit roots only.

    Append and scaling SNAKE records use ``generic_static_single.json``.
    The earlier L=2 current-forward SNAKE batch is native-only, so its
    ``cell_manifest.json`` plus unique ``json/result.json`` are indexed without
    opening the native payload.  This keeps the six very large L=4 native
    payloads out of object-mode JSON loading.
    """

    expected_algorithm = METHOD_ALGORITHMS[method_key]
    by_case: dict[str, list[dict[str, Any]]] = {}
    root_audit: list[dict[str, Any]] = []
    for precedence, raw_root in enumerate(roots):
        root = Path(raw_root).resolve()
        if not root.is_dir():
            root_audit.append(
                {"root": str(root), "status": "blocked:source_root_missing", "result_count": 0}
            )
            continue
        accepted = 0
        unreadable = 0
        generic_record_dirs: set[Path] = set()
        for result_path in sorted(root.rglob("generic_static_single.json")):
            try:
                payload = read_json(result_path)
            except Exception:
                unreadable += 1
                continue
            if str(payload.get("algorithm_id") or "") != expected_algorithm:
                continue
            case_id = str(payload.get("case_id") or "")
            if not case_id:
                unreadable += 1
                continue
            record_dir = result_path.parent.parent
            generic_record_dirs.add(record_dir.resolve())
            contract = _record_contract(record_dir)
            by_case.setdefault(case_id, []).append(
                {
                    "result_path": result_path,
                    "record_dir": record_dir,
                    "source_kind": "generic_wrapper",
                    "root": root,
                    "root_precedence": precedence,
                    "record_contract": contract,
                }
            )
            accepted += 1
        native_only = 0
        if method_key == "snake":
            for manifest_path in sorted(root.rglob("cell_manifest.json")):
                record_dir = manifest_path.parent
                if record_dir.resolve() in generic_record_dirs:
                    continue
                try:
                    manifest = read_json(manifest_path)
                    row = manifest.get("row")
                except Exception:
                    unreadable += 1
                    continue
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("method_key") or "") != "snake":
                    continue
                if str(row.get("algorithm_id") or "") != expected_algorithm:
                    continue
                regime = str(row.get("internal_regime") or row.get("source_map_regime") or "")
                case_id = L2_SNAKE_REGIME_TO_CANONICAL_CASE.get(
                    regime,
                    str(row.get("case_id") or ""),
                )
                if not case_id:
                    unreadable += 1
                    continue
                candidates = sorted(record_dir.glob("json/result.json"))
                candidates.extend(sorted(record_dir.glob("result/*/json/result.json")))
                candidates = [path for path in candidates if path.is_file()]
                if len(candidates) != 1:
                    unreadable += 1
                    continue
                contract = _record_contract(record_dir)
                by_case.setdefault(case_id, []).append(
                    {
                        "result_path": candidates[0],
                        "record_dir": record_dir,
                        "source_kind": "native_result_direct",
                        "canonical_case_id": case_id,
                        "source_case_id": row.get("case_id"),
                        "source_internal_regime": regime or None,
                        "root": root,
                        "root_precedence": precedence,
                        "record_contract": contract,
                    }
                )
                accepted += 1
                native_only += 1
        root_audit.append(
            {
                "root": str(root),
                "status": "ok",
                "result_count": accepted,
                "native_only_result_count": native_only,
                "unreadable_result_count": unreadable,
                "selection_precedence": precedence,
            }
        )
    return by_case, root_audit


def select_explicit_source(
    indexed: Mapping[str, Sequence[Mapping[str, Any]]], case_id: str
) -> dict[str, Any]:
    candidates = list(indexed.get(str(case_id)) or [])
    if not candidates:
        raise OverlayBlocked("result_not_found_in_explicit_roots")
    highest = max(int(item.get("root_precedence", -1)) for item in candidates)
    finalists = [item for item in candidates if int(item.get("root_precedence", -1)) == highest]
    if len(finalists) != 1:
        record_ids = {
            str(item.get("record_contract", {}).get("row", {}).get("record_id") or "")
            for item in finalists
        }
        repairs = []
        for item in finalists:
            row = item.get("record_contract", {}).get("row", {})
            superseded = str(
                row.get("repair_source_record_id")
                or row.get("supersedes_record_id")
                or ""
            )
            if superseded and superseded in record_ids:
                repairs.append(item)
        if len(repairs) == 1:
            return dict(repairs[0])
        paths = [str(item.get("result_path")) for item in finalists]
        raise OverlayBlocked(f"ambiguous_results_at_same_precedence:{paths}")
    return dict(finalists[0])


def _check_exact_energy(observed: Any, expected: float, *, label: str) -> None:
    value = _finite(observed, label)
    if not math.isclose(value, float(expected), rel_tol=1.0e-11, abs_tol=1.0e-11):
        raise OverlayBlocked(f"{label}={value:.16g} differs from expected {expected:.16g}")


def _append_plateau_overlay(
    source: Mapping[str, Any],
    *,
    expected_horizon: int,
    expected_exact_energy: float,
    grouped_exact_max_active_qubits: int,
) -> dict[str, Any]:
    result_path = Path(source["result_path"])
    payload = read_json(result_path)
    if str(payload.get("status")) not in {"completed", "ok"}:
        raise OverlayBlocked(f"generic result status={payload.get('status')!r}")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise OverlayBlocked("generic result has no result object")
    contract_row = _validate_record_contract(
        source["record_contract"],
        method_key="append",
        expected_horizon=expected_horizon,
        case_id=str(payload.get("case_id") or result.get("case_id") or ""),
    )
    history = result.get("adapt_history")
    if not isinstance(history, list) or len(history) != int(expected_horizon):
        raise OverlayBlocked(
            f"append history/horizon mismatch: history={len(history) if isinstance(history, list) else None}, expected={expected_horizon}"
        )
    if not bool(result.get("optimizer_success_all")):
        # The narrowly accepted finite/nonincreasing Powell cap is effective
        # success but remains raw-success false.  Require its explicit policy.
        policy = str(contract_row.get("powell_maxiter_cap_policy") or "")
        accepted = all(bool(row.get("optimizer_effective_success")) for row in history)
        if policy != "accept_finite_nonincreasing_v1" or not accepted:
            raise OverlayBlocked("append optimizer success contract failed")
    _check_exact_energy(
        result.get("same_cutoff_exact_gs_energy"),
        expected_exact_energy,
        label="append same-cutoff exact energy",
    )
    points = trajectory_points(result)
    curve = _normalized_curve([(point.k, point.error_raw) for point in points])
    plateau = select_first_plateau(result, horizon=expected_horizon)
    marker = next((point for point in curve if int(point["k"]) == int(plateau.k_pl)), None)
    if marker is None or not math.isclose(
        float(marker["error_raw"]), float(plateau.error_raw), rel_tol=1.0e-11, abs_tol=1.0e-11
    ):
        raise OverlayBlocked("append plateau selection/trajectory mismatch")
    seed_path = resolve_runtime_seed_path(result_path, payload)
    seed = read_json(seed_path)
    reconstruction = reconstruct_structural_prefix(
        seed=seed,
        history=history,
        history_position=int(plateau.history_position),
    )
    qiskit = compile_prefix_qiskit(
        seed=seed,
        reconstruction=reconstruction,
        grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        source_kind="qiskit_coefficient_aware_append_selected_prefix",
    )
    ledger = prefix_query_ledger(history, int(plateau.history_position))
    cost_status = str(qiskit.get("status") or "blocked:missing_qiskit_status")
    overall_status = (
        "ok"
        if cost_status == "ok"
        else f"blocked:cost:{cost_status}:{qiskit.get('blocked_reason')}"
    )
    return {
        "schema": "paper_i_geo_compact_overlay_method_v1",
        "method": "Append-ADAPT",
        "method_key": "append",
        "status": overall_status,
        "trajectory_status": "ok",
        "cost_status": cost_status,
        "curve": curve,
        "marker": {
            "policy": "first_prefix_within_10_percent_of_best_observed_error_v1",
            "label": "k_pl",
            "k": int(marker["k"]),
            "error_raw": float(marker["error_raw"]),
            "error_plotted": float(marker["error_plotted"]),
        },
        "qiskit_cost": {
            **qiskit,
            "scope": "selected_k_pl_structural_prefix",
            "comparison_semantics": "table_i_basis_gate_transpile_v1",
        },
        "query_ledger": ledger,
        "prefix_reconstruction": reconstruction,
        "plateau_selection": {
            "history_position": int(plateau.history_position),
            "k_pl": int(plateau.k_pl),
            "best_observed_error": float(plateau.best_observed_error),
            "threshold": float(plateau.threshold),
        },
        "policy_comparability": contract_row.get("overlay_policy_comparability"),
        "common_pauli_child_policy": bool(contract_row.get("common_pauli_child_policy")),
        "source": {
            "result_json": rel(result_path),
            "result_sha256": sha256(result_path),
            "runtime_seed_json": rel(seed_path),
            "runtime_seed_sha256": sha256(seed_path),
            "cell_manifest": source["record_contract"].get("path"),
            "cell_manifest_sha256": source["record_contract"].get("sha256"),
            "record_id": contract_row.get("record_id"),
        },
    }


def _geo_plateau_overlay(
    geo: Mapping[str, Any], *, grouped_exact_max_active_qubits: int
) -> dict[str, Any]:
    if int(grouped_exact_max_active_qubits) != int(
        geo.get("qiskit_prefix_cost", {}).get("grouped_exact_max_active_qubits")
        or grouped_exact_max_active_qubits
    ):
        raise OverlayBlocked("Geo k_pl compile cap differs from requested overlay cap")
    qiskit = dict(geo.get("qiskit_prefix_cost") or {})
    if str(qiskit.get("status")) != "ok":
        raise OverlayBlocked(
            f"Geo k_pl Qiskit compile {qiskit.get('status')}: {qiskit.get('blocked_reason')}"
        )
    ledger = dict(geo.get("query_ledger") or {})
    if ledger.get("S") is None:
        raise OverlayBlocked("Geo k_pl query ledger is missing S")
    curve = [dict(point) for point in geo["trajectory_points"]]
    marker = dict(geo.get("marker") or {})
    if marker.get("k") is None:
        raise OverlayBlocked("Geo k_pl marker is missing")
    return {
        "schema": "paper_i_geo_compact_overlay_method_v1",
        "method": "Geo-ADAPT",
        "method_key": "geo",
        "status": "ok",
        "trajectory_status": "ok",
        "cost_status": "ok",
        "curve": curve,
        "marker": {
            "policy": "first_prefix_within_10_percent_of_best_observed_error_v1",
            "label": "k_pl",
            "k": int(marker["k"]),
            "error_raw": float(marker["error_raw"]),
            "error_plotted": float(marker["error_plotted"]),
        },
        "qiskit_cost": {
            **qiskit,
            "scope": "selected_k_pl_structural_prefix",
            "comparison_semantics": "table_i_basis_gate_transpile_v1",
        },
        "query_ledger": ledger,
        "prefix_reconstruction": dict(geo.get("prefix_reconstruction") or {}),
        "plateau_selection": dict(geo.get("plateau_selection") or {}),
        "policy_comparability": "parent_macro_only_matched",
        "common_pauli_child_policy": True,
        "source": dict(geo["sources"]),
    }


def _parse_snake_checkpoint_curve(stdout_path: Path, *, initial_error: float) -> list[dict[str, Any]]:
    if not stdout_path.is_file():
        raise OverlayBlocked("snake stdout checkpoint log missing")
    points: dict[int, float] = {0: abs(float(initial_error))}
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("AI_LOG "):
            continue
        try:
            payload = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if str(payload.get("event")) != "hardcoded_adapt_current_checkpoint_written":
            continue
        if str(payload.get("reason")) != "beam_round_done":
            continue
        k = int(payload.get("depth") or -1)
        error = payload.get("benchmark_target_abs_delta_e_current")
        if k < 1 or error is None:
            continue
        if k in points:
            raise OverlayBlocked(f"duplicate SNAKE checkpoint event at outer iteration {k}")
        points[k] = abs(_finite(error, "SNAKE checkpoint error"))
    return _normalized_curve(sorted(points.items()))


def _require_complete_outer_curve(
    curve: Sequence[Mapping[str, Any]], *, expected_horizon: int, label: str
) -> None:
    observed = [int(point["k"]) for point in curve]
    expected = list(range(int(expected_horizon) + 1))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise OverlayBlocked(
            f"{label} outer-iteration coverage mismatch: missing={missing}, extra={extra}"
        )


def _snake_history_round_audit(
    *,
    record_dir: Path,
    expected_horizon: int,
    terminal_generator_count: int,
) -> dict[str, Any]:
    """Classify zero-gain duplicate loops from the compact current checkpoint."""

    candidates = sorted(
        {
            path.resolve()
            for path in (
                list(record_dir.glob("current.json"))
                + list(record_dir.glob("result/*/json/current.json"))
            )
            if path.is_file()
        }
    )
    if len(candidates) != 1:
        raise OverlayBlocked(
            "SNAKE history-pathology audit requires exactly one current.json; "
            f"found {len(candidates)}"
        )
    current_path = candidates[0]
    current = read_json(current_path)
    adapt = current.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise OverlayBlocked("SNAKE current checkpoint has no adapt_vqe object")
    raw_history = adapt.get("history_tail")
    if raw_history is None:
        raw_history = adapt.get("history")
    if not isinstance(raw_history, list):
        raise OverlayBlocked("SNAKE current checkpoint has no history/history_tail list")
    history_count = int(adapt.get("history_count") or len(raw_history))
    history_tail_count = int(adapt.get("history_tail_count") or len(raw_history))
    if (
        history_count != int(expected_horizon)
        or history_tail_count != int(expected_horizon)
        or len(raw_history) != int(expected_horizon)
    ):
        raise OverlayBlocked(
            "SNAKE current checkpoint does not retain the full requested history horizon: "
            f"history_count={history_count}, history_tail_count={history_tail_count}, "
            f"tail_len={len(raw_history)}, expected={expected_horizon}"
        )
    for index, row in enumerate(raw_history):
        if not isinstance(row, Mapping):
            raise OverlayBlocked(f"SNAKE current history row {index} is not an object")
    current_ansatz_depth = adapt.get("ansatz_depth")
    return {
        "schema": "paper_i_snake_history_round_audit_v2",
        "trajectory_status": "ok",
        "trajectory_semantics": "outer_history_rounds_not_committed_admission_count",
        "history_round_count": int(history_count),
        "current_checkpoint_ansatz_depth": (
            None if current_ansatz_depth is None else int(current_ansatz_depth)
        ),
        "terminal_generator_count": int(terminal_generator_count),
        "terminal_history_round": int(expected_horizon),
        "current_json": rel(current_path),
        "current_json_sha256": sha256(current_path),
    }


def trajectory_status_is_displayable(value: Any) -> bool:
    text = str(value or "")
    return text == "ok" or text.startswith("diagnostic:")


def _native_result_candidates(record_dir: Path) -> list[Path]:
    candidates = sorted(record_dir.glob("json/result.json"))
    candidates.extend(sorted(record_dir.glob("result/*/json/result.json")))
    return [path for path in candidates if path.is_file()]


def _native_sidecar_candidates(record_dir: Path, native_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in (native_path.parent, record_dir, record_dir / "result"):
        for name in NATIVE_SIDECAR_NAMES:
            candidate = directory / name
            if candidate.is_file():
                candidates.append(candidate)
    candidates.extend(sorted(record_dir.glob("**/*native*compact*sidecar*.json")))
    for ancestor in list(record_dir.parents)[:4]:
        compact_dir = ancestor / "compact_native"
        if compact_dir.is_dir():
            exact = compact_dir / f"{record_dir.name}__native_compact.json"
            if exact.is_file():
                candidates.append(exact)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def _unwrap_native_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("payload", "extracted_payload", "native_result_subset"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and isinstance(nested.get("adapt_vqe"), Mapping):
            return nested
    return payload


def _load_native_payload(
    *, record_dir: Path, preferred_path: Path | None = None
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    candidates = [preferred_path] if preferred_path is not None else _native_result_candidates(record_dir)
    candidates = [path for path in candidates if path is not None and path.is_file()]
    if len(candidates) != 1:
        raise OverlayBlocked(
            f"native terminal result count={len(candidates)}; expected exactly one"
        )
    native_path = Path(candidates[0])
    if preferred_path is not None and native_path.stat().st_size <= MAX_DIRECT_NATIVE_RESULT_BYTES:
        raw = read_json(native_path)
        payload = _unwrap_native_payload(raw)
        if not isinstance(payload.get("adapt_vqe"), Mapping):
            raise OverlayBlocked("native result has no adapt_vqe object")
        return payload, {
            "native_result_json": rel(native_path),
            "native_result_sha256": sha256(native_path),
            "native_payload_kind": "direct_native_result",
            "native_sidecar_json": None,
            "native_sidecar_sha256": None,
            "native_sidecar_schema": None,
        }
    sidecars = _native_sidecar_candidates(record_dir, native_path)
    if len(sidecars) > 1:
        raise OverlayBlocked(f"ambiguous native compact sidecars: {[str(path) for path in sidecars]}")
    if sidecars:
        sidecar = sidecars[0]
        raw = read_json(sidecar)
        payload = _unwrap_native_payload(raw)
        if not isinstance(payload.get("adapt_vqe"), Mapping):
            raise OverlayBlocked("native compact sidecar has no adapt_vqe object")
        source_meta = raw.get("source") if isinstance(raw.get("source"), Mapping) else {}
        declared_path = (
            source_meta.get("result_json")
            or raw.get("source_result_json")
            or raw.get("native_result_json")
        )
        if declared_path:
            declared_resolved = Path(str(declared_path))
            if not declared_resolved.is_absolute():
                declared_resolved = REPO_ROOT / declared_resolved
            if declared_resolved.resolve() != native_path.resolve():
                raise OverlayBlocked("native compact sidecar source path mismatch")
        declared_sha = (
            source_meta.get("result_sha256")
            or raw.get("source_result_sha256")
            or raw.get("native_result_sha256")
        )
        if declared_sha:
            observed_sha = sha256(native_path)
            if str(declared_sha) != observed_sha:
                raise OverlayBlocked("native compact sidecar source SHA-256 mismatch")
        declared_size = source_meta.get("size_bytes")
        if declared_size is not None and int(declared_size) != native_path.stat().st_size:
            raise OverlayBlocked("native compact sidecar source size mismatch")
        return payload, {
            "native_result_json": rel(native_path),
            "native_result_sha256": str(declared_sha) if declared_sha else sha256(native_path),
            "native_payload_kind": "compact_streaming_sidecar",
            "native_sidecar_json": rel(sidecar),
            "native_sidecar_sha256": sha256(sidecar),
            "native_sidecar_schema": raw.get("schema"),
        }
    size = native_path.stat().st_size
    if size > MAX_DIRECT_NATIVE_RESULT_BYTES:
        raise OverlayBlocked(
            "native_result_requires_compact_sidecar:"
            f"{size}_bytes_exceeds_{MAX_DIRECT_NATIVE_RESULT_BYTES}"
        )
    raw = read_json(native_path)
    payload = _unwrap_native_payload(raw)
    if not isinstance(payload.get("adapt_vqe"), Mapping):
        raise OverlayBlocked("native result has no adapt_vqe object")
    return payload, {
        "native_result_json": rel(native_path),
        "native_result_sha256": sha256(native_path),
        "native_payload_kind": "direct_native_result",
        "native_sidecar_json": None,
        "native_sidecar_sha256": None,
        "native_sidecar_schema": None,
    }


def _execution_modes_from_native(native: Mapping[str, Any]) -> dict[str, str]:
    modes: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            label = value.get("candidate_label") or value.get("generator_id") or value.get("label")
            mode = value.get("execution_mode") or value.get("recommended_execution_mode")
            if label and mode:
                text_label = str(label)
                text_mode = str(mode)
                prior = modes.get(text_label)
                if prior is not None and prior != text_mode:
                    raise OverlayBlocked(
                        f"conflicting native execution modes for {text_label}: {prior}, {text_mode}"
                    )
                modes[text_label] = text_mode
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    adapt = native.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        for key in ("selected_scaffold_history", "selected_scaffold_record_chain"):
            visit(adapt.get(key))
        visit(adapt.get("continuation"))
    return modes


def _aligned_runtime_split_execution_modes(
    native: Mapping[str, Any],
    *,
    labels: Sequence[str],
) -> list[tuple[str | None, dict[str, Any] | None]]:
    """Recover index-aligned runtime-split execution modes when serialized.

    The L=2 current-forward rows place the selected label on the outer metadata
    record and the exact execution mode inside
    ``compile_metadata.runtime_split``.  A recursive label-to-mode walk cannot
    join those two levels, and label-only matching is unsafe when labels repeat.
    """

    adapt = native.get("adapt_vqe")
    continuation = adapt.get("continuation") if isinstance(adapt, Mapping) else None
    raw = (
        continuation.get("selected_generator_metadata")
        if isinstance(continuation, Mapping)
        else None
    )
    if raw is None:
        return [(None, None) for _ in labels]
    if not isinstance(raw, list) or len(raw) != len(labels):
        raise OverlayBlocked(
            "native selected_generator_metadata is not index-aligned with terminal operators"
        )
    out: list[tuple[str | None, dict[str, Any] | None]] = []
    for index, (label, meta) in enumerate(zip(labels, raw)):
        if not isinstance(meta, Mapping):
            raise OverlayBlocked(f"native selected_generator_metadata[{index}] is not an object")
        observed_label = meta.get("candidate_label") or meta.get("label")
        if observed_label is None or str(observed_label) != str(label):
            raise OverlayBlocked(
                "native selected_generator_metadata/operator label mismatch at "
                f"index {index}: {observed_label!r} != {label!r}"
            )
        compile_meta = meta.get("compile_metadata")
        runtime_split = (
            compile_meta.get("runtime_split")
            if isinstance(compile_meta, Mapping)
            else None
        )
        mode = (
            runtime_split.get("recommended_execution_mode")
            if isinstance(runtime_split, Mapping)
            else None
        )
        if mode in {None, ""}:
            out.append((None, None))
            continue
        out.append(
            (
                str(mode),
                {
                    "schema": "native_runtime_split_execution_mode_alignment_v1",
                    "metadata_index": int(index),
                    "candidate_label": str(label),
                    "runtime_split_mode": runtime_split.get("mode"),
                    "runtime_split_representation": runtime_split.get("representation"),
                },
            )
        )
    return out


_HH_POOL_REPLAY_SETTING_KEYS = (
    "problem",
    "L",
    "t",
    "u",
    "dv",
    "omega0",
    "g_ep",
    "n_ph_max",
    "boson_encoding",
    "ordering",
    "boundary",
    "include_zero_point",
    "v_nn",
    "t_prime",
    "n_fermions",
    "adapt_continuation_mode",
    "adapt_pool",
    "adapt_pool_requested",
    "adapt_pool_class_filter_json",
    "adapt_pool_label_filter_json",
    "adapt_selected_logical_source_json",
    "adapt_selected_logical_mode",
    "adapt_selected_logical_transfer_mode",
    "paop_r",
    "paop_split_paulis",
    "paop_prune_eps",
    "paop_normalization",
    "phase3_symmetry_mitigation_mode",
    "adapt_child_pool_expansion_mode",
    "shared_pauli_pool_mode",
    "phase3_runtime_split_mode",
)


def _canonical_polynomial_signature(polynomial: Any) -> tuple[tuple[str, float, float], ...]:
    coefficients: dict[str, complex] = {}
    for term in polynomial.return_polynomial():
        label = str(term.pw2strng()).lower()
        coefficients[label] = coefficients.get(label, 0.0j) + complex(term.p_coeff)
    return tuple(
        sorted(
            (
                label,
                round(float(coeff.real), 12),
                round(float(coeff.imag), 12),
            )
            for label, coeff in coefficients.items()
            if abs(coeff) > 1.0e-12
        )
    )


def _derive_execution_mode_from_contract(
    native: Mapping[str, Any],
    *,
    label: str,
    terms: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Replay the family-specific pool execution-mode contract exactly."""

    settings = native.get("settings")
    if not isinstance(settings, Mapping):
        return None, None, None
    problem = str(settings.get("problem") or "")
    if problem == "hubbard":
        if str(label).startswith("hva_block::"):
            return "grouped_exact", "hubbard_hva_block_pool_contract_v1", None
        return "termwise_product", "hubbard_uccsd_qeb_pool_contract_v1", None
    try:
        from src.quantum.pauli_polynomial_class import PauliPolynomial
        from src.quantum.qubitization_module import PauliTerm
        from src.quantum.vqe_latex_python_pairs import AnsatzTerm

        nq = len(str(terms[0]["pauli_exyz"]))
        polynomial = PauliPolynomial(
            "JW",
            [
                PauliTerm(
                    nq,
                    ps=str(term["pauli_exyz"]),
                    pc=complex(float(term["coeff_re"]), float(term["coeff_im"])),
                )
                for term in terms
            ],
        )
        if problem in {"spin_boson", "bose_hubbard", "hh"}:
            from pipelines.static_adapt.builders.legal_subspace_filter import (
                sanitize_pool_for_binary_boson_legal_subspace,
            )

            if problem == "hh":
                for field in (
                    "adapt_child_pool_expansion_mode",
                    "shared_pauli_pool_mode",
                    "phase3_runtime_split_mode",
                ):
                    if str(settings.get(field) or "off").strip().lower() != "off":
                        raise OverlayBlocked(
                            "HH selected base-pool execution replay requires "
                            f"{field}=off; got {settings.get(field)!r}"
                        )

            filtered, meta = sanitize_pool_for_binary_boson_legal_subspace(
                [AnsatzTerm(label=str(label), polynomial=polynomial)],
                problem_key=problem,
                num_sites=int(settings.get("L") or 0),
                n_ph_max=int(settings.get("n_ph_max") or 0),
                boson_encoding=str(settings.get("boson_encoding") or "binary"),
                total_register_width=nq,
                fail_on_unknown=True,
            )
            if len(filtered) != 1:
                raise OverlayBlocked(
                    "native selected boson generator fails legal-subspace execution replay"
                )
            mode = str(getattr(filtered[0], "execution_mode", "termwise_product"))
            if problem == "hh":
                replay_settings = {
                    key: settings.get(key) for key in _HH_POOL_REPLAY_SETTING_KEYS
                }
                settings_json = json.dumps(
                    replay_settings, sort_keys=True, separators=(",", ":")
                )
                return (
                    mode,
                    "hh_base_full_meta_selected_legal_subspace_execution_replay_v1",
                    {
                        "schema": "hh_base_full_meta_selected_execution_replay_v1",
                        "settings_sha256": hashlib.sha256(
                            settings_json.encode("utf-8")
                        ).hexdigest(),
                        "selected_label": str(label),
                        "selected_polynomial_signature": _canonical_polynomial_signature(
                            polynomial
                        ),
                        "legal_subspace_filter_schema": (
                            meta.get("schema") if isinstance(meta, Mapping) else None
                        ),
                        "execution_mode": mode,
                        "contract": (
                            "sanitize_pool_for_binary_boson_legal_subspace_on_exact_"
                            "coefficient_bearing_selected_base_generator"
                        ),
                    },
                )
            return mode, "binary_boson_legal_subspace_execution_replay_v1", None
        if problem != "hh":
            return None, None, None
        return None, None, None
    except OverlayBlocked:
        raise
    except Exception as exc:
        if problem == "hh":
            raise OverlayBlocked(
                "exact HH base-pool execution-mode replay failed: "
                f"{type(exc).__name__}:{exc}"
            ) from exc
        return None, None, None


def _native_curve_and_reconstruction(
    native: Mapping[str, Any],
    *,
    expected_horizon: int,
    curve_override: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    adapt = native.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise OverlayBlocked("native result has no adapt_vqe object")
    if adapt.get("success") is False:
        raise OverlayBlocked("native SNAKE terminal result is not successful")
    if curve_override is None:
        history = adapt.get("history")
        if not isinstance(history, list) or len(history) != int(expected_horizon):
            raise OverlayBlocked(
                f"native SNAKE history/horizon mismatch: history={len(history) if isinstance(history, list) else None}, expected={expected_horizon}"
            )
        points: list[tuple[int, float]] = []
        for index, row in enumerate(history):
            if not isinstance(row, Mapping):
                raise OverlayBlocked(f"native SNAKE history row {index} is not an object")
            k = int(row.get("depth") or row.get("depth_cumulative") or index + 1)
            if index == 0:
                points.append((0, _finite(row.get("delta_abs_prev"), "native initial error")))
            points.append((k, _finite(row.get("delta_abs_current"), f"native error at {k}")))
        curve = _normalized_curve(points)
    else:
        curve = _normalized_curve(
            [
                (int(point["k"]), _finite(point["error_raw"], "override SNAKE error"))
                for point in curve_override
            ]
        )
    if int(curve[-1]["k"]) != int(expected_horizon):
        raise OverlayBlocked(
            f"native SNAKE curve stops at {curve[-1]['k']}, expected {expected_horizon}"
        )
    _require_complete_outer_curve(
        curve,
        expected_horizon=expected_horizon,
        label="native SNAKE",
    )
    terminal_error = _finite(adapt.get("abs_delta_e"), "native SNAKE terminal error")
    if not math.isclose(
        float(curve[-1]["error_raw"]), terminal_error, rel_tol=1.0e-11, abs_tol=1.0e-11
    ):
        raise OverlayBlocked("native SNAKE history/result terminal error mismatch")
    exact_final_error = adapt.get("exact_abs_delta_e_from_final_state")
    if exact_final_error is not None and not math.isclose(
        terminal_error,
        _finite(exact_final_error, "native final-state exact error"),
        rel_tol=1.0e-11,
        abs_tol=1.0e-11,
    ):
        raise OverlayBlocked("native SNAKE result/final-state terminal error mismatch")

    operators = adapt.get("operators")
    logical_point = adapt.get("logical_optimal_point")
    runtime_point = adapt.get("optimal_point")
    parameterization = adapt.get("parameterization")
    blocks = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None
    if not all(isinstance(value, list) for value in (operators, logical_point, runtime_point, blocks)):
        raise OverlayBlocked("native SNAKE terminal structure arrays are incomplete")
    labels = [str(label) for label in operators]
    block_labels = [str(block.get("candidate_label") or "") for block in blocks]
    if labels != block_labels:
        raise OverlayBlocked("native SNAKE operators do not match serialized parameterization blocks")
    logical_count = int(parameterization.get("logical_operator_count") or -1)
    runtime_count = int(parameterization.get("runtime_parameter_count") or -1)
    if logical_count != len(labels) or len(logical_point) != len(labels):
        raise OverlayBlocked("native SNAKE logical parameterization length mismatch")
    if runtime_count != len(runtime_point):
        raise OverlayBlocked("native SNAKE runtime parameterization length mismatch")
    if adapt.get("ansatz_depth") is not None and int(adapt.get("ansatz_depth")) != len(labels):
        raise OverlayBlocked("native SNAKE ansatz depth does not match terminal operators")

    mode_by_label = _execution_modes_from_native(native)
    aligned_modes = _aligned_runtime_split_execution_modes(native, labels=labels)
    semantics: list[dict[str, Any]] = []
    mode_source_counts: Counter[str] = Counter()
    execution_mode_counts: Counter[str] = Counter()
    replay_contracts: dict[str, dict[str, Any]] = {}
    local_compile_blocker: str | None = None
    runtime_cursor = 0
    for index, (label, block) in enumerate(zip(labels, blocks)):
        if int(block.get("logical_index") or 0) != index:
            raise OverlayBlocked(f"native SNAKE block {index} logical_index mismatch")
        if int(block.get("runtime_start") or 0) != runtime_cursor:
            raise OverlayBlocked(f"native SNAKE block {index} runtime_start is not contiguous")
        terms_raw = block.get("runtime_terms_exyz")
        if not isinstance(terms_raw, list) or not terms_raw:
            raise OverlayBlocked(f"native SNAKE block {index} has no coefficient-bearing terms")
        terms = [
            {
                "pauli_exyz": str(term.get("pauli_exyz") or "").lower(),
                "coeff_re": _finite(term.get("coeff_re", 0.0), "native coeff_re"),
                "coeff_im": _finite(term.get("coeff_im", 0.0), "native coeff_im"),
            }
            for term in terms_raw
            if isinstance(term, Mapping)
        ]
        if len(terms) != len(terms_raw):
            raise OverlayBlocked(f"native SNAKE block {index} contains a non-object term")
        block_runtime_count = int(block.get("runtime_count") or 0)
        if block_runtime_count != len(terms):
            raise OverlayBlocked(f"native SNAKE block {index} runtime_count mismatch")
        runtime_cursor += block_runtime_count
        for term_index, (term, raw_term) in enumerate(zip(terms, terms_raw)):
            word = str(term["pauli_exyz"])
            if not word or set(word) - set("exyz"):
                raise OverlayBlocked(
                    f"native SNAKE block {index} term {term_index} has invalid Pauli word"
                )
            if int(raw_term.get("nq") or -1) != len(word):
                raise OverlayBlocked(
                    f"native SNAKE block {index} term {term_index} nq/word-length mismatch"
                )
        mode = block.get("execution_mode") or block.get("recommended_execution_mode")
        mode_source = "parameterization_block" if mode else None
        mode_details: dict[str, Any] | None = None
        if not mode:
            aligned_mode, aligned_details = aligned_modes[index]
            if aligned_mode:
                mode = aligned_mode
                mode_source = "native_runtime_split_metadata_aligned_v1"
                mode_details = aligned_details
        if not mode:
            mode = mode_by_label.get(label)
            if mode:
                mode_source = "selected_scaffold_metadata"
        if not mode:
            mode, mode_source, mode_details = _derive_execution_mode_from_contract(
                native,
                label=label,
                terms=terms,
            )
        if not mode and len(terms) == 1:
            mode = "termwise_product"
            mode_source = "singleton_exact_equivalence_fallback"
        if not mode:
            local_compile_blocker = (
                "multi_term_native_generator_has_no_serialized_execution_mode"
            )
            mode_source = "blocked_missing_execution_mode"
        mode_source_counts[str(mode_source)] += 1
        execution_mode_counts[str(mode or "blocked_missing_execution_mode")] += 1
        if isinstance(mode_details, Mapping):
            detail_key = hashlib.sha256(
                json.dumps(mode_details, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            replay_contracts[detail_key] = dict(mode_details)
        semantics.append(
            {
                "index": index,
                "label": label,
                "execution_mode": str(mode or "blocked_missing_execution_mode"),
                "execution_mode_source": str(mode_source),
                "execution_mode_provenance": (
                    dict(mode_details) if isinstance(mode_details, Mapping) else None
                ),
                "support": [],
                "pauli_terms": terms,
            }
        )
    if runtime_cursor != runtime_count:
        raise OverlayBlocked("native SNAKE parameterization block runtime total mismatch")
    semantics_sha = hashlib.sha256(
        json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    structure_audit = {
        "status": "pass",
        "terminal_operator_count": len(labels),
        "runtime_parameter_count": runtime_count,
        "logical_parameter_count": logical_count,
        "operators_match_parameterization_blocks": True,
        "local_table_i_compile_status": (
            "ok_to_compile" if local_compile_blocker is None else f"blocked:{local_compile_blocker}"
        ),
        "selected_generator_semantics_sha256": semantics_sha,
        "execution_mode_counts": dict(sorted(execution_mode_counts.items())),
        "execution_mode_source_counts": dict(sorted(mode_source_counts.items())),
        "execution_mode_replay_contracts": [
            replay_contracts[key] for key in sorted(replay_contracts)
        ],
    }
    reconstruction = None
    if local_compile_blocker is None:
        reconstruction = {
            "selected_generator_count": len(labels),
            "selected_labels": labels,
            "selected_generator_semantics": semantics,
            "selected_generator_semantics_sha256": semantics_sha,
            "terminal_generator_count": len(labels),
            "terminal_parameters_match_selected_structure": True,
            "selected_prefix_parameter_status": "terminal_native_parameters_available",
            "selected_prefix_theta": [float(value) for value in runtime_point],
        }
    return curve, structure_audit, reconstruction


def _native_query_ledger(native: Mapping[str, Any], *, record_dir: Path) -> dict[str, Any]:
    adapt = native.get("adapt_vqe")
    candidates: list[tuple[Any, Any, Any, str, Path | None, Mapping[str, Any] | None]] = []
    if isinstance(adapt, Mapping):
        candidates.append(
            (
                adapt.get("algorithmic_measurement_work"),
                adapt.get("table_i_measurement_event_ledger"),
                adapt.get("S_alg"),
                "native_result",
                None,
                None,
            )
        )
    sidecar_path = record_dir / "snake_algorithmic_work.json"
    if sidecar_path.is_file():
        sidecar = read_json(sidecar_path)
        candidates.append(
            (
                sidecar.get("algorithmic_measurement_work"),
                sidecar.get("table_i_measurement_event_ledger"),
                sidecar.get("S_alg"),
                "snake_algorithmic_work_sidecar",
                sidecar_path,
                sidecar,
            )
        )
    for work, event_ledger, s_alg, source_kind, path, raw_sidecar in candidates:
        if not isinstance(work, Mapping) or str(work.get("status")) != "ok":
            continue
        if not isinstance(event_ledger, Mapping) or str(event_ledger.get("status")) != "ok":
            continue
        value = s_alg if s_alg is not None else work.get("S_alg")
        if value is None:
            continue
        value_float = _finite(value, "native SNAKE S_alg")
        if raw_sidecar is not None:
            if str(raw_sidecar.get("S_alg_status")) != "ok":
                continue
            if str(event_ledger.get("source_kind")) != "snake_native_runtime_reconstruction_v1":
                continue
            if (
                str(event_ledger.get("operator_probe_charge_basis"))
                != "logical_estimator_request_pre_grouping_v1"
            ):
                continue
            reconstruction = raw_sidecar.get("reconstruction_audit")
            if not isinstance(reconstruction, Mapping):
                continue
            if str(reconstruction.get("S_alg_work_scope")) != "winner_lineage_terminal":
                continue
            if str(reconstruction.get("S_alg_row_policy")) != "beam_terminal_winner_history_v1":
                continue
            if not bool(reconstruction.get("beam_aggregate_summary_blocked_as_row_s_alg")):
                continue
            component_sets = (
                raw_sidecar.get("component_counts"),
                work.get("components"),
                event_ledger.get("component_totals"),
            )
            for components in component_sets:
                if not isinstance(components, Mapping):
                    raise OverlayBlocked("native SNAKE S_alg component identity is incomplete")
                component_sum = sum(
                    _finite(component_value, "native SNAKE work component")
                    for component_value in components.values()
                )
                if not math.isclose(component_sum, value_float, rel_tol=0.0, abs_tol=1.0e-9):
                    raise OverlayBlocked("native SNAKE S_alg component identity failed")
            cell_path = record_dir / "cell_manifest.json"
            if cell_path.is_file():
                cell = read_json(cell_path)
                cell_s = cell.get("S_alg")
                if cell_s is not None and not math.isclose(
                    _finite(cell_s, "cell-manifest S_alg"),
                    value_float,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                ):
                    raise OverlayBlocked("native SNAKE sidecar/cell-manifest S_alg mismatch")
        return {
            "S": int(round(value_float)),
            "status": "ok",
            "scope": "terminal_winner_lineage_native_event_ledger",
            "source_kind": source_kind,
            "source_json": rel(path) if path is not None else None,
            "source_sha256": sha256(path) if path is not None else None,
            "algorithmic_measurement_work": dict(work),
            "table_i_measurement_event_ledger_status": str(event_ledger.get("status")),
        }
    raise OverlayBlocked("native SNAKE terminal algorithmic-work ledger is missing or invalid")


def _native_terminal_qiskit_cost(
    native: Mapping[str, Any],
    reconstruction: Mapping[str, Any] | None,
    *,
    grouped_exact_max_active_qubits: int,
) -> dict[str, Any]:
    if reconstruction is None:
        return {
            "status": "blocked:missing_exact_execution_mode_semantics",
            "blocked_reason": (
                "native SNAKE terminal lacks exact execution-mode semantics for Table-I compilation"
            ),
            "N2q": None,
            "D2q": None,
            "Dcirc": None,
            "scope": "terminal_exact_native_structural_ansatz",
            "comparison_semantics": "table_i_basis_gate_transpile_v1",
        }
    state = native.get("ansatz_input_state")
    if not isinstance(state, Mapping):
        return {
            "status": "blocked:missing_ansatz_input_state",
            "blocked_reason": "native SNAKE terminal has no ansatz_input_state",
            "N2q": None,
            "D2q": None,
            "Dcirc": None,
            "scope": "terminal_exact_native_structural_ansatz",
            "comparison_semantics": "table_i_basis_gate_transpile_v1",
        }
    qiskit = compile_prefix_qiskit(
        seed={"ansatz_input_state": dict(state)},
        reconstruction=reconstruction,
        grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        source_kind="qiskit_coefficient_aware_snake_terminal",
    )
    return {
        **qiskit,
        "scope": "terminal_exact_native_structural_ansatz",
        "comparison_semantics": "table_i_basis_gate_transpile_v1",
    }


def _compile_scout_cost(result: Mapping[str, Any], *, record_dir: Path) -> dict[str, Any]:
    cost_values = {
        "N2q": result.get("compiled_count_2q_total") or result.get("count_2q"),
        "D2q": result.get("depth_2q"),
        "Dcirc": result.get("compiled_depth_total") or result.get("circuit_depth"),
    }
    if any(value is None for value in cost_values.values()):
        raise OverlayBlocked("SNAKE terminal compile-scout cost columns missing")
    compile_path = None
    raw_compile = result.get("compile_json")
    if isinstance(raw_compile, str) and raw_compile:
        candidates = list((record_dir / "result").rglob(Path(raw_compile).name))
        if len(candidates) == 1:
            compile_path = candidates[0]
    if compile_path is None or not compile_path.is_file():
        raise OverlayBlocked("SNAKE terminal compile-scout artifact missing")
    compile_payload = read_json(compile_path)
    if not bool(compile_payload.get("success")):
        raise OverlayBlocked("SNAKE terminal compile-scout artifact reports failure")
    selected = compile_payload.get("selected_backend")
    if not isinstance(selected, Mapping) or str(selected.get("transpile_status")) != "ok":
        raise OverlayBlocked("SNAKE terminal compile-scout selected backend is invalid")
    selected_values = {
        "N2q": selected.get("compiled_count_2q"),
        "D2q": selected.get("compiled_depth_2q"),
        "Dcirc": selected.get("compiled_depth"),
    }
    if any(value is None for value in selected_values.values()):
        raise OverlayBlocked("SNAKE terminal compile-scout selected-backend columns missing")
    if any(int(cost_values[key]) != int(selected_values[key]) for key in cost_values):
        raise OverlayBlocked("SNAKE generic/compile-scout cost identity failed")
    return {
        "status": "ok",
        **{key: int(value) for key, value in cost_values.items()},
        "scope": "terminal_native_winner_ansatz_compile_scout",
        "source_kind": "qiskit_terminal_compile_scout_fake_marrakesh",
        "comparison_semantics": (
            "terminal_diagnostic_only_not_k_pl_not_reconstructed_prefix; "
            "backend convention differs from Table-I basis-gate comparator compile"
        ),
        "compile_json": rel(compile_path),
        "compile_json_sha256": sha256(compile_path),
    }


def _snake_terminal_overlay(
    source: Mapping[str, Any],
    *,
    expected_horizon: int,
    expected_exact_energy: float,
    grouped_exact_max_active_qubits: int,
) -> dict[str, Any]:
    result_path = Path(source["result_path"])
    record_dir = Path(source["record_dir"])
    contract_raw = source.get("record_contract")
    contract_manifest_row = contract_raw.get("row") if isinstance(contract_raw, Mapping) else None
    case_id = str(source.get("canonical_case_id") or "")
    if not case_id and isinstance(contract_manifest_row, Mapping):
        case_id = str(contract_manifest_row.get("case_id") or "")
    contract_row = _validate_record_contract(
        source["record_contract"],
        method_key="snake",
        expected_horizon=expected_horizon,
        case_id=case_id,
    )
    source_kind = str(source.get("source_kind") or "generic_wrapper")
    native_meta: dict[str, Any] | None = None
    structure_audit: dict[str, Any] | None = None
    if source_kind == "native_result_direct":
        native, native_meta = _load_native_payload(
            record_dir=record_dir,
            preferred_path=result_path,
        )
        adapt = native.get("adapt_vqe")
        if not isinstance(adapt, Mapping):
            raise OverlayBlocked("native SNAKE result has no adapt_vqe object")
        _check_exact_energy(
            adapt.get("exact_gs_energy"),
            expected_exact_energy,
            label="native SNAKE same-cutoff exact energy",
        )
        curve, structure_audit, reconstruction = _native_curve_and_reconstruction(
            native,
            expected_horizon=expected_horizon,
        )
        qiskit_cost = _native_terminal_qiskit_cost(
            native,
            reconstruction,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        query_ledger = _native_query_ledger(native, record_dir=record_dir)
        primary_source = {
            "native_result_json": native_meta["native_result_json"],
            "native_result_sha256": native_meta["native_result_sha256"],
        }
    else:
        payload = read_json(result_path)
        if str(payload.get("status")) != "completed":
            raise OverlayBlocked(f"SNAKE generic result status={payload.get('status')!r}")
        result = payload.get("result")
        if not isinstance(result, Mapping) or not bool(result.get("success")):
            raise OverlayBlocked("SNAKE generic terminal result is not successful")
        _check_exact_energy(
            result.get("same_cutoff_exact_gs_energy"),
            expected_exact_energy,
            label="SNAKE same-cutoff exact energy",
        )
        native_candidates = _native_result_candidates(record_dir)
        if native_candidates:
            native, native_meta = _load_native_payload(record_dir=record_dir)
            adapt = native.get("adapt_vqe")
            if not isinstance(adapt, Mapping):
                raise OverlayBlocked("native SNAKE result has no adapt_vqe object")
            _check_exact_energy(
                adapt.get("exact_gs_energy"),
                expected_exact_energy,
                label="native SNAKE same-cutoff exact energy",
            )
            stdout_curve = _parse_snake_checkpoint_curve(
                record_dir / "stdout.log",
                initial_error=_finite(result.get("initial_abs_delta_e"), "SNAKE initial error"),
            )
            if int(stdout_curve[-1]["k"]) != int(expected_horizon):
                raise OverlayBlocked(
                    f"SNAKE checkpoint curve stops at {stdout_curve[-1]['k']}, expected {expected_horizon}"
                )
            _require_complete_outer_curve(
                stdout_curve,
                expected_horizon=expected_horizon,
                label="SNAKE stdout checkpoint",
            )
            native_error = _finite(adapt.get("abs_delta_e"), "native SNAKE terminal error")
            generic_error = _finite(
                result.get("abs_delta_e_same_cutoff"), "SNAKE generic terminal error"
            )
            if not math.isclose(native_error, generic_error, rel_tol=1.0e-11, abs_tol=1.0e-11):
                raise OverlayBlocked("SNAKE native/generic terminal error mismatch")
            checkpoint_terminal_error = float(stdout_curve[-1]["error_raw"])
            stdout_curve[-1] = {
                **dict(stdout_curve[-1]),
                "error_raw": native_error,
                "error_plotted": max(native_error, PLOT_ERROR_FLOOR),
            }
            curve, structure_audit, reconstruction = _native_curve_and_reconstruction(
                native,
                expected_horizon=expected_horizon,
                curve_override=stdout_curve,
            )
            structure_audit["trajectory_terminal_source"] = "native_final_after_pruning"
            structure_audit["stdout_checkpoint_terminal_error"] = checkpoint_terminal_error
            structure_audit["native_final_terminal_error"] = native_error
            structure_audit["terminal_checkpoint_to_final_delta"] = (
                native_error - checkpoint_terminal_error
            )
            qiskit_cost = _native_terminal_qiskit_cost(
                native,
                reconstruction,
                grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
            )
        else:
            curve = _parse_snake_checkpoint_curve(
                record_dir / "stdout.log",
                initial_error=_finite(result.get("initial_abs_delta_e"), "SNAKE initial error"),
            )
            if int(curve[-1]["k"]) != int(expected_horizon):
                raise OverlayBlocked(
                    f"SNAKE checkpoint curve stops at {curve[-1]['k']}, expected {expected_horizon}"
                )
            _require_complete_outer_curve(
                curve,
                expected_horizon=expected_horizon,
                label="SNAKE stdout checkpoint",
            )
            terminal_error = _finite(
                result.get("abs_delta_e_same_cutoff"), "SNAKE terminal error"
            )
            if not math.isclose(
                float(curve[-1]["error_raw"]),
                terminal_error,
                rel_tol=1.0e-11,
                abs_tol=1.0e-11,
            ):
                raise OverlayBlocked("SNAKE checkpoint/result terminal error mismatch")
            # The fetched compile scout is FakeMarrakesh / optimization level
            # 1 and is therefore not a comparable substitute for the common
            # abstract-basis Table-I optimization-level-0 column.
            qiskit_cost = {
                "status": "blocked:native_terminal_payload_required_for_common_table_i_cost",
                "blocked_reason": (
                    "No hash-linked native terminal payload is fetched for common Table-I compilation"
                ),
                "N2q": None,
                "D2q": None,
                "Dcirc": None,
                "scope": "terminal_exact_native_structural_ansatz",
                "comparison_semantics": "table_i_basis_gate_transpile_v1",
            }
        s_alg = result.get("S_alg")
        if str(result.get("S_alg_status")) != "ok" or s_alg is None:
            raise OverlayBlocked(f"SNAKE terminal S_alg status={result.get('S_alg_status')!r}")
        work = result.get("algorithmic_measurement_work")
        if not isinstance(work, Mapping) or str(work.get("status")) != "ok":
            raise OverlayBlocked("SNAKE terminal algorithmic-work ledger is missing or invalid")
        event_ledger = result.get("table_i_measurement_event_ledger")
        if not isinstance(event_ledger, Mapping) or str(event_ledger.get("status")) != "ok":
            raise OverlayBlocked("SNAKE terminal Table-I event ledger is missing or invalid")
        query_ledger = {
            "S": int(round(float(s_alg))),
            "status": "ok",
            "scope": "terminal_winner_lineage_from_generic_result",
            "algorithmic_measurement_work": dict(work),
            "table_i_measurement_event_ledger_status": str(event_ledger.get("status")),
        }
        primary_source = {
            "generic_result_json": rel(result_path),
            "generic_result_sha256": sha256(result_path),
        }
    if not isinstance(structure_audit, Mapping):
        raise OverlayBlocked(
            "SNAKE trajectory requires terminal native structure for pathology classification"
        )
    pathology = _snake_history_round_audit(
        record_dir=record_dir,
        expected_horizon=expected_horizon,
        terminal_generator_count=int(structure_audit["terminal_operator_count"]),
    )
    terminal = curve[-1]
    cost_status = str(qiskit_cost.get("status") or "blocked:missing_qiskit_status")
    trajectory_status = str(pathology["trajectory_status"])
    if trajectory_status.startswith("diagnostic:"):
        overall_status = trajectory_status
        if cost_status != "ok":
            overall_status += f":cost_blocked:{cost_status}"
    else:
        overall_status = (
            "ok"
            if cost_status == "ok"
            else f"blocked:cost:{cost_status}:{qiskit_cost.get('blocked_reason')}"
        )
    return {
        "schema": "paper_i_geo_compact_overlay_method_v1",
        "method": "SNAKE",
        "method_key": "snake",
        "status": overall_status,
        "trajectory_status": trajectory_status,
        "cost_status": cost_status,
        "curve": curve,
        "marker": {
            "policy": "terminal_history_round_diagnostic",
            "label": "r_hist",
            "k": int(terminal["k"]),
            "error_raw": float(terminal["error_raw"]),
            "error_plotted": float(terminal["error_plotted"]),
        },
        "qiskit_cost": qiskit_cost,
        "query_ledger": query_ledger,
        "policy_comparability": contract_row.get("overlay_policy_comparability"),
        "common_pauli_child_policy": bool(contract_row.get("common_pauli_child_policy")),
        "terminal_structure_validation": structure_audit,
        "trajectory_diagnostic": pathology,
        "source": {
            **primary_source,
            "stdout_checkpoint_log": (
                rel(record_dir / "stdout.log") if (record_dir / "stdout.log").is_file() else None
            ),
            "stdout_checkpoint_log_sha256": (
                sha256(record_dir / "stdout.log")
                if (record_dir / "stdout.log").is_file()
                else None
            ),
            "cell_manifest": source["record_contract"].get("path"),
            "cell_manifest_sha256": source["record_contract"].get("sha256"),
            "record_id": contract_row.get("record_id"),
            "native_terminal": native_meta,
        },
    }


def _blocked_method(method: str, case_id: str, exc: Exception) -> dict[str, Any]:
    reason = str(exc)
    status_prefix = "absent" if "result_not_found" in reason else "blocked"
    return {
        "schema": "paper_i_geo_compact_overlay_method_v1",
        "method": method,
        "method_key": METHOD_KEYS[method],
        "case_id": case_id,
        "status": f"{status_prefix}:{reason}",
        "trajectory_status": f"{status_prefix}:{reason}",
        "cost_status": f"{status_prefix}:{reason}",
        "curve": [],
        "marker": None,
        "qiskit_cost": {"status": f"{status_prefix}:{reason}"},
        "query_ledger": {"status": f"{status_prefix}:{reason}", "S": None},
        "source": None,
    }


def build_overlay_rows(
    *,
    geo_rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    append_roots: Sequence[Path],
    snake_roots: Sequence[Path],
    grouped_exact_max_active_qubits: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(geo_rows) != len(inventory_rows):
        raise ValueError("Geo/inventory row count mismatch")
    append_index, append_root_audit = index_explicit_roots(append_roots, method_key="append")
    snake_index, snake_root_audit = index_explicit_roots(snake_roots, method_key="snake")
    output: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    trajectory_status_counts: Counter[str] = Counter()
    cost_status_counts: Counter[str] = Counter()
    for geo_raw, inventory in zip(geo_rows, inventory_rows):
        geo = dict(geo_raw)
        case_id = str(geo["case_id"])
        expected_horizon = int(geo["completed_horizon"])
        expected_exact = _finite(
            inventory.get("energy", {}).get("same_cutoff_exact_energy"),
            f"{case_id} inventory same-cutoff exact energy",
        )
        methods: dict[str, Any] = {}
        # Geo remains the validated plateau-prefix row until overlays are
        # requested; overlay mode recompiles and displays its exact terminal
        # structural ansatz with the cap-8 Table-I convention.
        try:
            methods["Geo-ADAPT"] = _geo_plateau_overlay(
                geo,
                grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
            )
        except Exception as exc:
            methods["Geo-ADAPT"] = _blocked_method(
                "Geo-ADAPT",
                case_id,
                exc,
            )
        try:
            append_source = select_explicit_source(append_index, case_id)
            methods["Append-ADAPT"] = _append_plateau_overlay(
                append_source,
                expected_horizon=expected_horizon,
                expected_exact_energy=expected_exact,
                grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
            )
        except Exception as exc:
            methods["Append-ADAPT"] = _blocked_method("Append-ADAPT", case_id, exc)
        try:
            snake_source = select_explicit_source(snake_index, case_id)
            methods["SNAKE"] = _snake_terminal_overlay(
                snake_source,
                expected_horizon=expected_horizon,
                expected_exact_energy=expected_exact,
                grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
            )
        except Exception as exc:
            methods["SNAKE"] = _blocked_method("SNAKE", case_id, exc)
        geo["overlay_mode"] = True
        geo["method_overlays"] = methods
        for method in METHOD_ORDER:
            status_counts[f"{method}:{methods[method]['status'].split(':', 1)[0]}"] += 1
            trajectory_status_counts[
                f"{method}:{str(methods[method].get('trajectory_status')).split(':', 1)[0]}"
            ] += 1
            cost_status_counts[
                f"{method}:{str(methods[method].get('cost_status')).split(':', 1)[0]}"
            ] += 1
        output.append(geo)
    snake_history_audit_rows = [
        row["method_overlays"]["SNAKE"].get("trajectory_diagnostic")
        for row in output
        if isinstance(
            row["method_overlays"]["SNAKE"].get("trajectory_diagnostic"), Mapping
        )
    ]
    summary = {
        "schema": "paper_i_geo_compact_overlay_summary_v1",
        "status_counts": dict(sorted(status_counts.items())),
        "trajectory_status_counts": dict(sorted(trajectory_status_counts.items())),
        "cost_status_counts": dict(sorted(cost_status_counts.items())),
        "append_root_audit": append_root_audit,
        "snake_root_audit": snake_root_audit,
        "row_count": len(output),
        "complete_case_count": sum(
            all(row["method_overlays"][method]["status"] == "ok" for method in METHOD_ORDER)
            for row in output
        ),
        "complete_trajectory_case_count": sum(
            all(
                row["method_overlays"][method].get("trajectory_status") == "ok"
                for method in METHOD_ORDER
            )
            for row in output
        ),
        "displayable_three_method_case_count": sum(
            all(
                trajectory_status_is_displayable(
                    row["method_overlays"][method].get("trajectory_status")
                )
                for method in METHOD_ORDER
            )
            for row in output
        ),
        "displayable_three_method_cost_complete_case_count": sum(
            all(
                trajectory_status_is_displayable(
                    row["method_overlays"][method].get("trajectory_status")
                )
                and row["method_overlays"][method].get("cost_status") == "ok"
                for method in METHOD_ORDER
            )
            for row in output
        ),
        "mixed_policy_case_ids": [
            row["case_id"]
            for row in output
            if trajectory_status_is_displayable(
                row["method_overlays"]["SNAKE"].get("trajectory_status")
            )
            and not bool(row["method_overlays"]["SNAKE"].get("common_pauli_child_policy"))
        ],
        "snake_history_round_audit": {
            "audited_row_count": int(len(snake_history_audit_rows)),
            "trajectory_semantics": "outer_history_rounds_not_committed_admission_count",
        },
        "snake_cost_semantics_counts": dict(
            sorted(
                Counter(
                    str(row["method_overlays"]["SNAKE"].get("qiskit_cost", {}).get("comparison_semantics"))
                    for row in output
                    if row["method_overlays"]["SNAKE"].get("cost_status") == "ok"
                ).items()
            )
        ),
        "cost_scope_note": (
            "Geo and Append use their own k_pl structural-prefix costs; SNAKE uses its terminal "
            "native ansatz after the reported history horizon because "
            "arbitrary-prefix replay is not validated. Every displayed cost uses coefficient-aware "
            "Table-I basis-gate compilation at optimization level 0 and cap 8. FakeMarrakesh "
            "optimization-level-1 compile scouts are never inserted into the adjacent table."
        ),
    }
    return output, summary


def write_overlay_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write only the overlay methods retained on every report row.

    The compact report can intentionally omit a comparator family (for
    example, when SNAKE evidence has been withdrawn).  Keeping this writer
    keyed to the row payload prevents an omitted method from reappearing as an
    absent/blocked provenance row.
    """

    active_methods = tuple(
        method
        for method in METHOD_ORDER
        if rows
        and all(
            isinstance(row.get("method_overlays"), Mapping)
            and method in row["method_overlays"]
            for row in rows
        )
    )
    if not active_methods:
        raise ValueError("Overlay CSV has no common retained methods")
    fields = (
        "order_index",
        "case_id",
        "family",
        "L",
        "display_regime",
        "method",
        "status",
        "trajectory_status",
        "cost_status",
        "terminal_iteration",
        "terminal_error",
        "trajectory_semantics",
        "history_round_count",
        "terminal_generator_count",
        "N2q",
        "D2q",
        "Dcirc",
        "S",
        "qiskit_scope",
        "qiskit_comparison_semantics",
        "source_json",
        "source_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in active_methods:
                overlay = row["method_overlays"][method]
                marker = overlay.get("marker") if isinstance(overlay.get("marker"), Mapping) else {}
                cost = overlay.get("qiskit_cost") if isinstance(overlay.get("qiskit_cost"), Mapping) else {}
                ledger = overlay.get("query_ledger") if isinstance(overlay.get("query_ledger"), Mapping) else {}
                diagnostic = (
                    overlay.get("trajectory_diagnostic")
                    if isinstance(overlay.get("trajectory_diagnostic"), Mapping)
                    else {}
                )
                source = overlay.get("source") if isinstance(overlay.get("source"), Mapping) else {}
                source_json = (
                    source.get("result_json")
                    or source.get("generic_result_json")
                    or source.get("native_result_json")
                )
                source_sha = (
                    source.get("result_sha256")
                    or source.get("generic_result_sha256")
                    or source.get("native_result_sha256")
                )
                writer.writerow(
                    {
                        "order_index": row["order_index"],
                        "case_id": row["case_id"],
                        "family": row["family"],
                        "L": row["L"],
                        "display_regime": row["display_regime"],
                        "method": method,
                        "status": overlay["status"],
                        "trajectory_status": overlay.get("trajectory_status"),
                        "cost_status": overlay.get("cost_status"),
                        "terminal_iteration": marker.get("k"),
                        "terminal_error": marker.get("error_raw"),
                        "trajectory_semantics": diagnostic.get("trajectory_semantics"),
                        "history_round_count": diagnostic.get("history_round_count"),
                        "terminal_generator_count": diagnostic.get("terminal_generator_count"),
                        "N2q": cost.get("N2q"),
                        "D2q": cost.get("D2q"),
                        "Dcirc": cost.get("Dcirc"),
                        "S": ledger.get("S"),
                        "qiskit_scope": cost.get("scope"),
                        "qiskit_comparison_semantics": cost.get("comparison_semantics"),
                        "source_json": source_json,
                        "source_sha256": source_sha,
                    }
                )
