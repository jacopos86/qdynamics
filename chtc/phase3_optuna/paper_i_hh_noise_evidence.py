from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

SELECTED_ENERGY_ZERO_NOISE_PASS_EVIDENCE_SCHEMA = (
    "paper_i_hh_selected_energy_zero_noise_route_faithfulness_evidence_v1"
)
SELECTED_ENERGY_ZERO_NOISE_PASS_EVIDENCE_STATUS = "selected_energy_zero_noise_pass_validated"
NOISE_TARGET_HIT_PASS_EVIDENCE_SCHEMA = "paper_i_hh_noise_target_hit_pass_evidence_v1"
NOISE_TARGET_HIT_PASS_EVIDENCE_STATUS = "noise_target_hit_pass_validated"
SELECTED_ENERGY_ZERO_NOISE_PASS_REQUESTED_INNER_OBJECTIVE_MODE = "noisy_v1"
SELECTED_ENERGY_ZERO_NOISE_PASS_EFFECTIVE_INNER_OBJECTIVE_MODE = "exact"
SELECTED_ENERGY_ZERO_NOISE_PASS_GUARD_REASON = "zero_noise_noisy_v1_exact_equivalent_guard_v1"
TARGET_HIT_SUCCESS_STOP_REASON = "benchmark_abs_delta_e_target"
SOURCE_LOCK_CASE_ID = "hh_L2_nph2_three_model_sym_strong_weak"
SOURCE_LOCK_EXPECTED_ENERGY = -0.4948181229002955
SOURCE_LOCK_EXPECTED_ABS_DELTA_E = 0.0001775162084057813
SOURCE_LOCK_EXPECTED_DEPTH = 13
SOURCE_LOCK_EXPECTED_N_PH_WORK = 2
SOURCE_LOCK_EXPECTED_N_PH_REF = 5
SOURCE_LOCK_EXPECTED_U = 1.25
SOURCE_LOCK_EXPECTED_G_EP = 0.3535533905932738
EVIDENCE_NUMERIC_ABS_TOL = 1.0e-12
SOURCE_LOCK_EXPECTED_OPERATOR_SEQUENCE = (
    "hh_phonon::s(site=1)",
    "hh_phonon::s(site=0)",
    "hh_fermionic_reusable::bond_charge_current_nn_up(0,1)",
    "hh_fermionic_reusable::bond_charge_current_nn_dn(0,1)",
    "hh_fermionic_reusable::exchange_current_nn(0,1)",
    "hh_fermionic_reusable::exchange_current_nn(0,1)",
    "hh_fermionic_reusable::exchange_current_nn(0,1)",
    "paop_full:paop_disp(site=1)",
    "paop_full:paop_disp(site=1)",
    "paop_full:paop_disp(site=1)",
    "paop_full:paop_disp(site=0)",
    "paop_full:paop_disp(site=0)",
    "paop_full:paop_disp(site=0)",
)
SOURCE_LOCK_EXPECTED_OPERATOR_SEQUENCE_SHA256 = "4c1481f594e3b5c62b63fbfefbf6a55ff8a75d2a247e5858ae497a50f3681887"

EVIDENCE_ROW_FIELDNAMES = (
    "selected_energy_zero_noise_pass_evidence_schema",
    "selected_energy_zero_noise_pass_evidence_status",
    "selected_energy_zero_noise_pass_evidence_json",
    "selected_energy_zero_noise_pass_evidence_sha256",
    "selected_energy_zero_noise_pass_evidence_baseline_json",
    "selected_energy_zero_noise_pass_evidence_baseline_sha256",
    "selected_energy_zero_noise_pass_evidence_case_id",
    "selected_energy_zero_noise_pass_evidence_stop_reason",
    "selected_energy_zero_noise_pass_evidence_energy",
    "selected_energy_zero_noise_pass_evidence_abs_delta_e",
    "selected_energy_zero_noise_pass_evidence_ansatz_depth",
    "selected_energy_zero_noise_pass_evidence_operator_count",
    "selected_energy_zero_noise_pass_evidence_first_operator",
    "selected_energy_zero_noise_pass_evidence_sequence_sha256",
    "selected_energy_zero_noise_pass_evidence_baseline_sequence_sha256",
    "selected_energy_zero_noise_pass_evidence_requested_inner_objective_mode",
    "selected_energy_zero_noise_pass_evidence_effective_inner_objective_mode",
    "selected_energy_zero_noise_pass_evidence_runtime_guard_reason",
    "selected_energy_zero_noise_pass_evidence_validation_errors",
)

NOISE_TARGET_HIT_PASS_EVIDENCE_ROW_FIELDNAMES = (
    "noise_target_hit_pass_evidence_schema",
    "noise_target_hit_pass_evidence_status",
    "noise_target_hit_pass_evidence_json",
    "noise_target_hit_pass_evidence_sha256",
    "noise_target_hit_pass_evidence_rung_id",
    "noise_target_hit_pass_evidence_case_id",
    "noise_target_hit_pass_evidence_stop_reason",
    "noise_target_hit_pass_evidence_energy",
    "noise_target_hit_pass_evidence_abs_delta_e",
    "noise_target_hit_pass_evidence_ansatz_depth",
    "noise_target_hit_pass_evidence_n_eff",
    "noise_target_hit_pass_evidence_sigma0_abs",
    "noise_target_hit_pass_evidence_std_abs",
    "noise_target_hit_pass_evidence_validation_errors",
)


class EvidenceValidationError(ValueError):
    """Raised when the selected-energy zero-noise pass evidence is missing or stale."""


def resolve_repo_path(path_value: str | Path, *, repo_root: str | Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path(repo_root) / path


def repo_relative(path_value: str | Path, *, repo_root: str | Path = REPO_ROOT) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve(strict=False).relative_to(Path(repo_root).resolve(strict=False))).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def operator_sequence_sha256(operators: Sequence[str]) -> str:
    text = json.dumps([str(op) for op in operators], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _adapt_vqe(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(payload.get("adapt_vqe"))


def _settings(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(payload.get("settings"))


def _continuation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _as_mapping(_adapt_vqe(payload).get("continuation"))


def _operators_from_payload(payload: Mapping[str, Any]) -> list[str]:
    adapt = _adapt_vqe(payload)
    raw = adapt.get("operators") or adapt.get("operator_labels") or adapt.get("selected_operators")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) for item in raw]


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(str(value)))
    except Exception:
        return None


def _value_noise_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract scalar value-noise metadata from a HH noise result payload."""

    adapt = _adapt_vqe(payload)
    candidate_roots: list[tuple[str, Mapping[str, Any]]] = []
    for source, value in (
        (
            "adapt_vqe.continuation.oracle_inner_exact_structure_value_noise.last_draw",
            _get_nested(adapt, "continuation", "oracle_inner_exact_structure_value_noise", "last_draw"),
        ),
        (
            "adapt_vqe.phase3_oracle_inner_exact_structure_value_noise.last_draw",
            _get_nested(adapt, "phase3_oracle_inner_exact_structure_value_noise", "last_draw"),
        ),
        (
            "adapt_vqe.continuation.oracle_gradient_config.value_noise",
            _get_nested(adapt, "continuation", "oracle_gradient_config", "value_noise"),
        ),
    ):
        if isinstance(value, Mapping):
            candidate_roots.append((source, value))
    for source, root in candidate_roots:
        n_eff = _float_or_none(root.get("n_eff", root.get("N_eff")))
        sigma0_abs = _float_or_none(root.get("sigma0_abs"))
        if n_eff is not None and n_eff > 0.0 and sigma0_abs is not None and sigma0_abs >= 0.0:
            return {
                "status": "ok",
                "source": source,
                "n_eff": float(n_eff),
                "sigma0_abs": float(sigma0_abs),
                "std_abs": float(sigma0_abs / math.sqrt(n_eff)),
            }
    return {"status": "missing_value_noise_contract", "n_eff": None, "sigma0_abs": None, "std_abs": None}


def _check_float(
    errors: list[str],
    *,
    field: str,
    actual: Any,
    expected: float,
    tol: float = EVIDENCE_NUMERIC_ABS_TOL,
) -> None:
    parsed = _float_or_none(actual)
    if parsed is None or not math.isclose(parsed, float(expected), rel_tol=0.0, abs_tol=float(tol)):
        errors.append(f"{field}:{actual}!={expected}")


def _read_json(path_value: str | Path, *, repo_root: str | Path) -> tuple[Path, dict[str, Any], str]:
    resolved = resolve_repo_path(path_value, repo_root=repo_root)
    if not resolved.exists() or not resolved.is_file():
        raise EvidenceValidationError(f"evidence JSON missing: {repo_relative(resolved, repo_root=repo_root)}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvidenceValidationError(
            f"evidence JSON unreadable: {repo_relative(resolved, repo_root=repo_root)}:{type(exc).__name__}:{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"evidence JSON must be an object: {repo_relative(resolved, repo_root=repo_root)}")
    return resolved, payload, sha256_file(resolved)


def validate_selected_energy_zero_noise_pass_evidence(
    evidence_json: str | Path,
    *,
    baseline_reference_json: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate the guarded selected-energy zero-noise HH strong/weak pass artifact.

    The check intentionally binds the noisy-v1 unlock to a local target-hit
    artifact whose exact selected-energy route is indistinguishable from the
    original exact selected-energy baseline.  It is not a general noisy-row pass.
    """

    errors: list[str] = []
    try:
        evidence_path, evidence, evidence_sha = _read_json(evidence_json, repo_root=repo_root)
    except EvidenceValidationError:
        if raise_on_error:
            raise
        evidence_path = resolve_repo_path(evidence_json, repo_root=repo_root)
        evidence = {}
        evidence_sha = ""
        errors.append("evidence_json_missing_or_unreadable")

    baseline_path: Path | None = None
    baseline_sha = ""
    baseline_payload: Mapping[str, Any] = {}
    if baseline_reference_json not in {None, ""}:
        try:
            baseline_path, baseline_raw, baseline_sha = _read_json(baseline_reference_json, repo_root=repo_root)
            baseline_payload = baseline_raw
        except EvidenceValidationError as exc:
            errors.append(f"baseline_reference_json_invalid:{exc}")

    adapt = _adapt_vqe(evidence)
    settings = _settings(evidence)
    continuation = _as_mapping(adapt.get("continuation"))
    operators = _operators_from_payload(evidence)
    baseline_operators = _operators_from_payload(baseline_payload)
    if not baseline_operators:
        baseline_operators = list(SOURCE_LOCK_EXPECTED_OPERATOR_SEQUENCE)
    baseline_sequence_sha = operator_sequence_sha256(baseline_operators)
    sequence_sha = operator_sequence_sha256(operators) if operators else ""

    baseline_adapt = _adapt_vqe(baseline_payload)
    baseline_energy = baseline_adapt.get("energy", SOURCE_LOCK_EXPECTED_ENERGY)
    baseline_abs_delta_e = baseline_adapt.get("abs_delta_e", SOURCE_LOCK_EXPECTED_ABS_DELTA_E)
    baseline_depth = _int_or_none(baseline_adapt.get("ansatz_depth")) or len(baseline_operators) or SOURCE_LOCK_EXPECTED_DEPTH

    stop_reason = adapt.get("stop_reason")
    energy = adapt.get("energy")
    abs_delta_e = adapt.get("abs_delta_e")
    depth = _int_or_none(adapt.get("ansatz_depth"))
    requested_inner = continuation.get("oracle_inner_objective_mode_requested")
    effective_inner = continuation.get("oracle_inner_objective_mode")
    guard_reason = continuation.get("oracle_inner_objective_runtime_guard_reason")

    if stop_reason != TARGET_HIT_SUCCESS_STOP_REASON:
        errors.append(f"stop_reason:{stop_reason}!={TARGET_HIT_SUCCESS_STOP_REASON}")
    if depth not in {SOURCE_LOCK_EXPECTED_DEPTH, len(baseline_operators), baseline_depth}:
        errors.append(f"ansatz_depth:{depth}!={SOURCE_LOCK_EXPECTED_DEPTH}_or_baseline_count")
    if len(operators) != len(baseline_operators):
        errors.append(f"operator_count:{len(operators)}!={len(baseline_operators)}")
    if list(operators) != list(baseline_operators):
        errors.append("operator_sequence_mismatch")
    if sequence_sha != baseline_sequence_sha or sequence_sha != SOURCE_LOCK_EXPECTED_OPERATOR_SEQUENCE_SHA256:
        errors.append(
            "operator_sequence_sha256:"
            f"{sequence_sha}!={baseline_sequence_sha}!={SOURCE_LOCK_EXPECTED_OPERATOR_SEQUENCE_SHA256}"
        )
    _check_float(errors, field="energy", actual=energy, expected=float(baseline_energy))
    _check_float(errors, field="energy_expected_anchor", actual=energy, expected=SOURCE_LOCK_EXPECTED_ENERGY)
    _check_float(errors, field="abs_delta_e", actual=abs_delta_e, expected=float(baseline_abs_delta_e))
    _check_float(errors, field="abs_delta_e_expected_anchor", actual=abs_delta_e, expected=SOURCE_LOCK_EXPECTED_ABS_DELTA_E)

    observed_case_id = _first_nonempty(
        evidence.get("benchmark_ids"),
        evidence.get("benchmark_id"),
        evidence.get("case_id"),
        settings.get("benchmark_ids"),
        settings.get("benchmark_id"),
        settings.get("case_id"),
        adapt.get("benchmark_ids"),
        adapt.get("benchmark_id"),
        adapt.get("case_id"),
    )
    if observed_case_id is not None and str(observed_case_id) != SOURCE_LOCK_CASE_ID:
        errors.append(f"case_id:{observed_case_id}!={SOURCE_LOCK_CASE_ID}")
    if settings.get("problem") is not None and str(settings.get("problem")) != "hh":
        errors.append(f"problem:{settings.get('problem')}!=hh")
    if settings.get("L") is not None and _int_or_none(settings.get("L")) != 2:
        errors.append(f"L:{settings.get('L')}!=2")
    if settings.get("n_ph_max") is not None and _int_or_none(settings.get("n_ph_max")) != SOURCE_LOCK_EXPECTED_N_PH_WORK:
        errors.append(f"n_ph_work:{settings.get('n_ph_max')}!={SOURCE_LOCK_EXPECTED_N_PH_WORK}")
    if settings.get("u") is not None:
        _check_float(errors, field="u", actual=settings.get("u"), expected=SOURCE_LOCK_EXPECTED_U)
    if settings.get("g_ep") is not None:
        _check_float(errors, field="g_ep", actual=settings.get("g_ep"), expected=SOURCE_LOCK_EXPECTED_G_EP)

    if str(requested_inner or "") != SELECTED_ENERGY_ZERO_NOISE_PASS_REQUESTED_INNER_OBJECTIVE_MODE:
        errors.append(
            "requested_inner_objective_mode:"
            f"{requested_inner}!={SELECTED_ENERGY_ZERO_NOISE_PASS_REQUESTED_INNER_OBJECTIVE_MODE}"
        )
    if str(effective_inner or "") != SELECTED_ENERGY_ZERO_NOISE_PASS_EFFECTIVE_INNER_OBJECTIVE_MODE:
        errors.append(
            "effective_inner_objective_mode:"
            f"{effective_inner}!={SELECTED_ENERGY_ZERO_NOISE_PASS_EFFECTIVE_INNER_OBJECTIVE_MODE}"
        )
    if str(guard_reason or "") != SELECTED_ENERGY_ZERO_NOISE_PASS_GUARD_REASON:
        errors.append(f"runtime_guard_reason:{guard_reason}!={SELECTED_ENERGY_ZERO_NOISE_PASS_GUARD_REASON}")

    value_noise_model = _get_nested(continuation, "oracle_gradient_config", "value_noise", "model")
    value_noise_std = _get_nested(continuation, "oracle_gradient_config", "value_noise", "std")
    if value_noise_model is not None and str(value_noise_model) != "off":
        errors.append(f"value_noise_model:{value_noise_model}!=off")
    if value_noise_std is not None:
        _check_float(errors, field="value_noise_std", actual=value_noise_std, expected=0.0)

    status = SELECTED_ENERGY_ZERO_NOISE_PASS_EVIDENCE_STATUS if not errors else "invalid"
    payload = {
        "schema": SELECTED_ENERGY_ZERO_NOISE_PASS_EVIDENCE_SCHEMA,
        "status": status,
        "evidence_json": repo_relative(evidence_path, repo_root=repo_root),
        "evidence_sha256": evidence_sha,
        "baseline_json": repo_relative(baseline_path, repo_root=repo_root) if baseline_path is not None else "",
        "baseline_sha256": baseline_sha,
        "case_id": SOURCE_LOCK_CASE_ID,
        "stop_reason": str(stop_reason or ""),
        "energy": "" if energy is None else str(energy),
        "abs_delta_e": "" if abs_delta_e is None else str(abs_delta_e),
        "ansatz_depth": "" if depth is None else str(int(depth)),
        "operator_count": str(len(operators)),
        "first_operator": operators[0] if operators else "",
        "sequence_sha256": sequence_sha,
        "baseline_sequence_sha256": baseline_sequence_sha,
        "requested_inner_objective_mode": str(requested_inner or ""),
        "effective_inner_objective_mode": str(effective_inner or ""),
        "runtime_guard_reason": str(guard_reason or ""),
        "validation_errors": errors,
    }
    if errors and raise_on_error:
        raise EvidenceValidationError(
            "selected-energy zero-noise pass evidence validation failed: " + ";".join(str(err) for err in errors)
        )
    return payload


def validate_noise_target_hit_pass_evidence(
    evidence_json: str | Path,
    *,
    predecessor_rung_id: str,
    expected_n_eff: float | None = None,
    expected_sigma0_abs: float | None = None,
    repo_root: str | Path = REPO_ROOT,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate a same-noise target-hit predecessor result for later rungs.

    This evidence is deliberately narrower than a paper-facing result promotion:
    it proves that the exact same scalar-noise surface already produced a target
    hit, so a later same-noise stop-condition or confidence-scoring row may be
    generated.  It does not require selected-generator sequence parity.
    """

    errors: list[str] = []
    try:
        evidence_path, evidence, evidence_sha = _read_json(evidence_json, repo_root=repo_root)
    except EvidenceValidationError:
        if raise_on_error:
            raise
        evidence_path = resolve_repo_path(evidence_json, repo_root=repo_root)
        evidence = {}
        evidence_sha = ""
        errors.append("evidence_json_missing_or_unreadable")

    adapt = _adapt_vqe(evidence)
    settings = _settings(evidence)
    stop_reason = adapt.get("stop_reason")
    energy = adapt.get("energy")
    abs_delta_e = adapt.get("abs_delta_e")
    depth = _int_or_none(adapt.get("ansatz_depth"))
    target_hit = adapt.get("benchmark_target_hit_success")
    if stop_reason != TARGET_HIT_SUCCESS_STOP_REASON:
        errors.append(f"stop_reason:{stop_reason}!={TARGET_HIT_SUCCESS_STOP_REASON}")
    if target_hit is False or str(target_hit).strip().lower() == "false":
        errors.append("benchmark_target_hit_success:false")
    if depth is None or depth <= 0:
        errors.append(f"ansatz_depth:{depth}:invalid")

    observed_case_id = _first_nonempty(
        evidence.get("benchmark_ids"),
        evidence.get("benchmark_id"),
        evidence.get("case_id"),
        settings.get("benchmark_ids"),
        settings.get("benchmark_id"),
        settings.get("case_id"),
        adapt.get("benchmark_ids"),
        adapt.get("benchmark_id"),
        adapt.get("case_id"),
    )
    if observed_case_id is not None and str(observed_case_id) != SOURCE_LOCK_CASE_ID:
        errors.append(f"case_id:{observed_case_id}!={SOURCE_LOCK_CASE_ID}")
    if settings.get("problem") is not None and str(settings.get("problem")) != "hh":
        errors.append(f"problem:{settings.get('problem')}!=hh")

    contract = _value_noise_contract(evidence)
    n_eff = _float_or_none(contract.get("n_eff"))
    sigma0_abs = _float_or_none(contract.get("sigma0_abs"))
    std_abs = _float_or_none(contract.get("std_abs"))
    if contract.get("status") != "ok" or n_eff is None or sigma0_abs is None or std_abs is None:
        errors.append(str(contract.get("status") or "missing_value_noise_contract"))
    if expected_n_eff is not None:
        if n_eff is None or not math.isclose(float(n_eff), float(expected_n_eff), rel_tol=1e-12, abs_tol=0.0):
            errors.append(f"n_eff:{n_eff}!={expected_n_eff}")
    if expected_sigma0_abs is not None:
        if sigma0_abs is None or not math.isclose(float(sigma0_abs), float(expected_sigma0_abs), rel_tol=1e-12, abs_tol=0.0):
            errors.append(f"sigma0_abs:{sigma0_abs}!={expected_sigma0_abs}")

    if not str(predecessor_rung_id or "").strip():
        errors.append("predecessor_rung_id:missing")

    status = NOISE_TARGET_HIT_PASS_EVIDENCE_STATUS if not errors else "invalid"
    payload = {
        "schema": NOISE_TARGET_HIT_PASS_EVIDENCE_SCHEMA,
        "status": status,
        "evidence_json": repo_relative(evidence_path, repo_root=repo_root),
        "evidence_sha256": evidence_sha,
        "rung_id": str(predecessor_rung_id or ""),
        "case_id": SOURCE_LOCK_CASE_ID,
        "stop_reason": str(stop_reason or ""),
        "energy": "" if energy is None else str(energy),
        "abs_delta_e": "" if abs_delta_e is None else str(abs_delta_e),
        "ansatz_depth": "" if depth is None else str(int(depth)),
        "n_eff": "" if n_eff is None else f"{float(n_eff):.17g}",
        "sigma0_abs": "" if sigma0_abs is None else f"{float(sigma0_abs):.17g}",
        "std_abs": "" if std_abs is None else f"{float(std_abs):.17g}",
        "validation_errors": errors,
    }
    if errors and raise_on_error:
        raise EvidenceValidationError("noise target-hit pass evidence validation failed: " + ";".join(str(err) for err in errors))
    return payload


def evidence_row_fields(evidence: Mapping[str, Any] | None) -> dict[str, str]:
    if not evidence:
        return {field: "" for field in EVIDENCE_ROW_FIELDNAMES}
    errors = evidence.get("validation_errors", ()) or ()
    return {
        "selected_energy_zero_noise_pass_evidence_schema": str(evidence.get("schema") or ""),
        "selected_energy_zero_noise_pass_evidence_status": str(evidence.get("status") or ""),
        "selected_energy_zero_noise_pass_evidence_json": str(evidence.get("evidence_json") or ""),
        "selected_energy_zero_noise_pass_evidence_sha256": str(evidence.get("evidence_sha256") or ""),
        "selected_energy_zero_noise_pass_evidence_baseline_json": str(evidence.get("baseline_json") or ""),
        "selected_energy_zero_noise_pass_evidence_baseline_sha256": str(evidence.get("baseline_sha256") or ""),
        "selected_energy_zero_noise_pass_evidence_case_id": str(evidence.get("case_id") or ""),
        "selected_energy_zero_noise_pass_evidence_stop_reason": str(evidence.get("stop_reason") or ""),
        "selected_energy_zero_noise_pass_evidence_energy": str(evidence.get("energy") or ""),
        "selected_energy_zero_noise_pass_evidence_abs_delta_e": str(evidence.get("abs_delta_e") or ""),
        "selected_energy_zero_noise_pass_evidence_ansatz_depth": str(evidence.get("ansatz_depth") or ""),
        "selected_energy_zero_noise_pass_evidence_operator_count": str(evidence.get("operator_count") or ""),
        "selected_energy_zero_noise_pass_evidence_first_operator": str(evidence.get("first_operator") or ""),
        "selected_energy_zero_noise_pass_evidence_sequence_sha256": str(evidence.get("sequence_sha256") or ""),
        "selected_energy_zero_noise_pass_evidence_baseline_sequence_sha256": str(
            evidence.get("baseline_sequence_sha256") or ""
        ),
        "selected_energy_zero_noise_pass_evidence_requested_inner_objective_mode": str(
            evidence.get("requested_inner_objective_mode") or ""
        ),
        "selected_energy_zero_noise_pass_evidence_effective_inner_objective_mode": str(
            evidence.get("effective_inner_objective_mode") or ""
        ),
        "selected_energy_zero_noise_pass_evidence_runtime_guard_reason": str(
            evidence.get("runtime_guard_reason") or ""
        ),
        "selected_energy_zero_noise_pass_evidence_validation_errors": ";".join(str(err) for err in errors),
    }


def noise_target_hit_evidence_row_fields(evidence: Mapping[str, Any] | None) -> dict[str, str]:
    if not evidence:
        return {field: "" for field in NOISE_TARGET_HIT_PASS_EVIDENCE_ROW_FIELDNAMES}
    errors = evidence.get("validation_errors", ()) or ()
    return {
        "noise_target_hit_pass_evidence_schema": str(evidence.get("schema") or ""),
        "noise_target_hit_pass_evidence_status": str(evidence.get("status") or ""),
        "noise_target_hit_pass_evidence_json": str(evidence.get("evidence_json") or ""),
        "noise_target_hit_pass_evidence_sha256": str(evidence.get("evidence_sha256") or ""),
        "noise_target_hit_pass_evidence_rung_id": str(evidence.get("rung_id") or ""),
        "noise_target_hit_pass_evidence_case_id": str(evidence.get("case_id") or ""),
        "noise_target_hit_pass_evidence_stop_reason": str(evidence.get("stop_reason") or ""),
        "noise_target_hit_pass_evidence_energy": str(evidence.get("energy") or ""),
        "noise_target_hit_pass_evidence_abs_delta_e": str(evidence.get("abs_delta_e") or ""),
        "noise_target_hit_pass_evidence_ansatz_depth": str(evidence.get("ansatz_depth") or ""),
        "noise_target_hit_pass_evidence_n_eff": str(evidence.get("n_eff") or ""),
        "noise_target_hit_pass_evidence_sigma0_abs": str(evidence.get("sigma0_abs") or ""),
        "noise_target_hit_pass_evidence_std_abs": str(evidence.get("std_abs") or ""),
        "noise_target_hit_pass_evidence_validation_errors": ";".join(str(err) for err in errors),
    }


def validate_evidence_row_provenance(
    row: Mapping[str, str],
    *,
    baseline_reference_json: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    evidence_json = str(row.get("selected_energy_zero_noise_pass_evidence_json") or "").strip()
    if not evidence_json:
        raise EvidenceValidationError("missing selected_energy_zero_noise_pass_evidence_json")
    baseline_json = baseline_reference_json
    if baseline_json in {None, ""}:
        baseline_json = str(row.get("source_lock_reference_json") or "").strip() or None
    evidence = validate_selected_energy_zero_noise_pass_evidence(
        evidence_json,
        baseline_reference_json=baseline_json,
        repo_root=repo_root,
        raise_on_error=True,
    )
    expected_fields = evidence_row_fields(evidence)
    for field, expected in expected_fields.items():
        actual = str(row.get(field) or "")
        if actual != str(expected):
            raise EvidenceValidationError(
                f"selected-energy zero-noise pass evidence row provenance mismatch:{field}:{actual!r}!={expected!r}"
            )
    return evidence
