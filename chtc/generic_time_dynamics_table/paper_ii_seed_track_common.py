# ---------------------------------------------------------------------------
# RESTORED 2026-09-01 -- B4 of the open-decisions queue (classification row 31).
# Source: origin/codex/wip-local-active-source-snapshot-20260706-120000 @
# 517eca4c, blob 7f621d90835e8200b22e94042700c65c273707df, 2026-07-08.
# Content below this header is VERBATIM from that blob; nothing was
# forward-ported. Restored together with
# build_paper_ii_hh_seed_track_inputs.py, which imports it at module scope.
# ---------------------------------------------------------------------------
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path("chtc/generic_time_dynamics_table/input")
CLASS_SETTINGS_LOCK = "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"

SOURCE_REGISTRY_SCHEMA = "paper_ii_hh_static_seed_sources_v1"
SEED_TRACK_CASE_SCHEMA = "paper_ii_hh_seed_track_case_manifest_v1"
SEED_TRACK_LEDGER_SCHEMA = "paper_ii_hh_seed_track_ledger_v1"


class SeedTrackValidationError(ValueError):
    """Raised when a static seed source violates the Paper-II seed-track contract."""


@dataclass(frozen=True)
class SeedTrackSpec:
    seed_track: str
    display_label: str
    required_static_algorithm_id: str
    seed_selection_policy: str
    forbidden_source_tokens: tuple[str, ...] = ()
    forbidden_payload_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class HHRegimeSpec:
    regime_id: str
    static_case_id: str
    u_over_t: float
    lambda_ep: float
    n_ph_work: int
    n_ph_ref: int


SEED_TRACK_SPECS: tuple[SeedTrackSpec, ...] = (
    SeedTrackSpec(
        seed_track="geo",
        display_label="GeoAdapt",
        required_static_algorithm_id="static_geo_adapt_vqe",
        seed_selection_policy="strict_geo_adapt_current_revised_hh_seed_v1",
        forbidden_source_tokens=("posgeo", "pos_geo", "static_pos_geo_adapt_vqe"),
        forbidden_payload_values=("static_pos_geo_adapt_vqe", "best_insert_refit", "position_optimized_geo_adapt"),
    ),
    SeedTrackSpec(
        seed_track="snake",
        display_label="SNAKE",
        required_static_algorithm_id="paper_i_production_v1",
        seed_selection_policy="route_a_full_meta_paper_i_production_v1_seed_v1",
        forbidden_source_tokens=("selected_logical", "family_reduced", "lowwork", "stress"),
    ),
    SeedTrackSpec(
        seed_track="append",
        display_label="append-only ADAPT",
        required_static_algorithm_id="static_full_meta_append_adapt_vqe",
        seed_selection_policy="full_meta_append_only_current_revised_hh_seed_v1",
        forbidden_source_tokens=("staged", "fallback", "pending", "recovery"),
    ),
)

SEED_TRACKS_BY_ID: dict[str, SeedTrackSpec] = {spec.seed_track: spec for spec in SEED_TRACK_SPECS}

HH_REGIMES: tuple[HHRegimeSpec, ...] = (
    HHRegimeSpec("weak_weak", "hh_L2_nph2_three_model_sym_weak_weak", 0.25, 0.25, 2, 5),
    HHRegimeSpec("strong_weak", "hh_L2_nph2_three_model_sym_strong_weak", 1.25, 0.25, 2, 5),
    HHRegimeSpec("weak_strong", "hh_L2_nph4_three_model_sym_weak_strong", 0.25, 1.25, 4, 7),
    HHRegimeSpec("strong_strong", "hh_L2_nph4_three_model_sym_strong_strong", 1.25, 1.25, 4, 7),
)

HH_TRUE_STRONG_REGIMES: tuple[HHRegimeSpec, ...] = (
    HHRegimeSpec("strong_weak_u8", "hh_L2_nph2_three_model_sym_u8_strong_weak", 8.0, 0.25, 2, 5),
    HHRegimeSpec("strong_strong_u8", "hh_L2_nph4_three_model_sym_u8_strong_strong", 8.0, 1.25, 4, 7),
)

HH_LOCAL_PAPER_II_REGIMES: tuple[HHRegimeSpec, ...] = (
    HH_REGIMES[0],
    HH_REGIMES[2],
    *HH_TRUE_STRONG_REGIMES,
)

HH_ALL_REGIMES: tuple[HHRegimeSpec, ...] = (*HH_REGIMES, *HH_TRUE_STRONG_REGIMES)

HH_REGIMES_BY_ID: dict[str, HHRegimeSpec] = {regime.regime_id: regime for regime in HH_ALL_REGIMES}
HH_REGIMES_BY_CASE_ID: dict[str, HHRegimeSpec] = {regime.static_case_id: regime for regime in HH_ALL_REGIMES}

DRIVES: tuple[tuple[str, float], ...] = (("A0p2", 0.2), ("A0p6", 0.6))

VISIBLE_ALGORITHMS: tuple[str, ...] = (
    "dyn_controller_full",
    "dyn_fixed_mclachlan",
    "dyn_adaptive_pvqd",
    "dyn_avqds_t",
)

FULL_BENCHMARK_ALGORITHMS: tuple[str, ...] = (
    "dyn_fixed_mclachlan",
    "dyn_product_formula_envelope",
    "dyn_qdrift",
    "dyn_fixed_pvqd",
    "dyn_adaptive_pvqd",
    "dyn_avqds",
    "dyn_avqds_t",
)


def repo_path(path: str | Path, *, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def read_json(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    payload = json.loads(repo_path(path, root=root).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SeedTrackValidationError(f"Expected JSON object at {path}")
    return dict(payload)


def write_json(path: str | Path, payload: Mapping[str, Any], *, root: Path = ROOT) -> None:
    resolved = repo_path(path, root=root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path, *, root: Path = ROOT) -> str:
    return hashlib.sha256(repo_path(path, root=root).read_bytes()).hexdigest()


def drive_block(amplitude: float) -> dict[str, Any]:
    return {
        "enable_drive": True,
        "A": float(amplitude),
        "omega": 1.0,
        "pattern": "staggered",
        "phi": 0.0,
        "t0": 0.0,
        "tbar": 1.0,
        "time_sampling": "midpoint",
        "include_identity": False,
        "custom_weights": "",
    }


def hh_seed_track_case_id(regime_id: str, seed_track: str, drive_label: str) -> str:
    return f"table1_hh_{regime_id}_{seed_track}_{drive_label}_t8_dt321_seedtracks_v1"


def hh_same_seed_group_id(regime_id: str, seed_track: str, drive_label: str) -> str:
    return f"hh_{regime_id}_{seed_track}_{drive_label}_t8_dt321_same_seed_v1"


def hh_seed_artifact_relpath(regime_id: str, seed_track: str) -> Path:
    return Path("seed_artifacts_paper_ii_hh_seed_tracks_v1") / f"hh_{regime_id}_{seed_track}_seed.json"


def hh_record_id(case_id: str, algorithm_id: str) -> str:
    return f"paper_ii_hh_seedtracks_v1_{case_id}_{algorithm_id}"


def _result_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    return dict(result) if isinstance(result, Mapping) else {}


def _adapt_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = payload.get("adapt_vqe", {})
    return dict(adapt) if isinstance(adapt, Mapping) else {}


def _settings_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    return dict(settings) if isinstance(settings, Mapping) else {}


def _parse_static_base_args(args: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_map = {
        "problem": "problem",
        "L": "L",
        "t": "t",
        "u": "u",
        "dv": "dv",
        "omega0": "omega0",
        "g-ep": "g_ep",
        "n-ph-max": "n_ph_max",
        "boson-encoding": "boson_encoding",
        "ordering": "ordering",
        "boundary": "boundary",
        "n-fermions": "n_fermions",
    }
    float_keys = {"t", "u", "dv", "omega0", "g_ep"}
    int_keys = {"L", "n_ph_max", "n_fermions"}
    i = 0
    while i < len(args):
        token = str(args[i])
        if not token.startswith("--"):
            i += 1
            continue
        mapped = key_map.get(token[2:])
        if mapped is None:
            i += 2
            continue
        if i + 1 >= len(args):
            raise SeedTrackValidationError(f"Missing value after generic static base arg {token!r}.")
        raw = args[i + 1]
        if mapped in float_keys:
            value: Any = float(raw)
        elif mapped in int_keys:
            value = int(raw)
        else:
            value = str(raw)
        out[mapped] = value
        i += 2
    out.setdefault("problem", "hh")
    out.setdefault("L", 2)
    out.setdefault("t", 1.0)
    out.setdefault("u", 0.0)
    out.setdefault("dv", 0.0)
    out.setdefault("omega0", 1.0)
    out.setdefault("g_ep", 0.0)
    out.setdefault("n_ph_max", 1)
    out.setdefault("boson_encoding", "binary")
    out.setdefault("ordering", "blocked")
    out.setdefault("boundary", "open")
    return out


def _generic_static_wrapper_settings(payload: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    spec = payload.get("spec", {})
    base_args = spec.get("base_pipeline_args", []) if isinstance(spec, Mapping) else []
    settings = _parse_static_base_args(list(base_args) if isinstance(base_args, Sequence) else [])
    settings["problem"] = str(result.get("problem", settings.get("problem", "hh")))
    settings["L"] = int(result.get("L", settings.get("L", 2)))
    return settings


def _build_state_payloads_for_seed(seed_payload: Mapping[str, Any], *, source_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload
    from pipelines.static_adapt.builders.problem_registry import (
        ProblemRequest,
        canonical_problem_key,
        resolve_problem_context,
    )

    settings = _settings_block(seed_payload)
    request = ProblemRequest(
        problem_key=canonical_problem_key(settings.get("problem", "hh")),
        num_sites=int(settings.get("L", settings.get("num_sites", 0))),
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", settings.get("U", 0.0))),
        dv=float(settings.get("dv", 0.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.0)),
        n_ph_max=int(settings.get("n_ph_max", 0)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        n_fermions=(
            None
            if settings.get("n_fermions", None) in {None, ""}
            else int(settings.get("n_fermions"))
        ),
    )
    context = resolve_problem_context(request)
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    ansatz_input_state = build_statevector_manifest(
        psi_state=psi_ref,
        source="resolved_problem.reference_state",
        handoff_state_kind="reference_state",
        amplitude_cutoff=1.0e-12,
    )
    load_payload = copy.deepcopy(dict(seed_payload))
    load_payload["ansatz_input_state"] = ansatz_input_state
    loaded = load_scaffold_runtime_input_from_payload(
        load_payload,
        artifact_json=source_path,
        loader_mode="replay_family",
        generator_family="full_meta",
        fallback_family="full_meta",
    )
    initial_state = build_statevector_manifest(
        psi_state=loaded.psi_initial,
        source="paper_ii_hh_static_seed_export.reconstructed_from_scaffold",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=1.0e-12,
    )
    return ansatz_input_state, initial_state


def _convert_hh_generic_static_wrapper(
    payload: Mapping[str, Any],
    *,
    source_entry: Mapping[str, Any],
    source_path: str | Path,
) -> dict[str, Any]:
    result = _result_block(payload)
    if not result:
        raise SeedTrackValidationError(
            "HH seed source is neither a runtime seed nor a generic static result wrapper."
        )
    selected = list(result.get("selected_operators") or result.get("operators") or [])
    theta = list(result.get("theta") or result.get("optimal_point") or [])
    if not selected or not theta:
        raise SeedTrackValidationError(f"Cannot export HH generic static wrapper without selected operators/theta: {source_path}")
    if len(selected) != len(theta):
        raise SeedTrackValidationError(
            f"Selected operator/theta length mismatch in {source_path}: {len(selected)} vs {len(theta)}"
        )
    algorithm_id = str(payload.get("algorithm_id", payload.get("method_id", result.get("algorithm_id", "")))).strip()
    case_id = str(payload.get("case_id", result.get("case_id", source_entry.get("hh_static_case_id", "")))).strip()
    settings = _generic_static_wrapper_settings(payload, result)
    exact = result.get("exact_gs_energy", result.get("exact_energy", result.get("exact_reference_energy")))
    energy = result.get("energy", result.get("optimizer_reported_energy"))
    abs_delta = source_static_error({"result": result, "algorithm_id": algorithm_id})
    seed_payload: dict[str, Any] = {
        "generated_utc": str(payload.get("finished_utc", "2026-05-31T00:00:00+00:00")),
        "pipeline": "paper_ii_hh_seed_track_static_seed_export_from_generic_static_v1",
        "schema": "paper_ii_static_seed_runtime_payload_v1",
        "family": "hh",
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "method_id": algorithm_id,
        "settings": settings,
        "ground_state": {
            "exact_energy": None if exact is None or exact == "" else float(exact),
            "exact_energy_source": str(result.get("primary_reference_source", "generic_static_table_payload")),
            "method": "static_variant_reference_metric",
        },
        "adapt_vqe": {
            "algorithm_id": algorithm_id,
            "method_id": algorithm_id,
            "method_label": str(result.get("method_label", payload.get("method_id", algorithm_id))),
            "ansatz_depth": int(result.get("adapt_depth_reached", len(selected))),
            "operators": [str(item) for item in selected],
            "optimal_point": [float(item) for item in theta],
            "logical_optimal_point": [float(item) for item in theta],
            "theta": [float(item) for item in theta],
            "num_parameters": int(len(theta)),
            "pool_type": str(result.get("pool_name", result.get("required_pool_key", "full_meta"))),
            "adapt_pool": str(result.get("pool_name", result.get("required_pool_key", "full_meta"))),
            "required_pool_key": str(result.get("required_pool_key", "full_meta")),
            "selected_operator_pauli_labels_exyz": list(result.get("selected_operator_pauli_labels_exyz", [])),
            "selected_operator_supports": list(result.get("selected_operator_supports", [])),
            "abs_delta_e": abs_delta,
            "energy": energy,
            "exact_gs_energy": exact,
            "adapt_stop_reason": result.get("adapt_stop_reason"),
            "position_policy": result.get("position_policy"),
            "position_optimized_geo_adapt": result.get("position_optimized_geo_adapt"),
            "static_seed_source_runner": "pipelines.exact_bench.generic_static_adapt_variants",
        },
        "paper_ii_static_seed_export": {
            "schema": "paper_ii_static_seed_export_v1",
            "static_algorithm_id": algorithm_id,
            "static_seed_display_label": str(result.get("method_label", payload.get("method_id", algorithm_id))),
            "source_static_case_id": case_id,
            "source_family": "hh",
            "source_suite_profile": str((payload.get("spec", {}) if isinstance(payload.get("spec"), Mapping) else {}).get("split", "")),
            "static_abs_delta_e": abs_delta,
            "static_parameter_count": int(len(theta)),
            "runtime_loadability_status": "runtime_seed_sidecar_reconstructed_from_generic_static",
        },
        "source_static_table_result": {
            "schema": str(payload.get("schema", "")),
            "algorithm_id": algorithm_id,
            "case_id": case_id,
            "generic_static_single_json": str(source_path),
        },
        "source_artifact_json": str(source_path),
    }
    ansatz_input_state, initial_state = _build_state_payloads_for_seed(seed_payload, source_path=source_path)
    seed_payload["ansatz_input_state"] = ansatz_input_state
    seed_payload["initial_state"] = initial_state
    return seed_payload


def coerce_hh_seed_payload(
    payload: Mapping[str, Any],
    *,
    source_entry: Mapping[str, Any],
    source_path: str | Path,
) -> dict[str, Any]:
    if isinstance(payload.get("settings", None), Mapping) and isinstance(payload.get("adapt_vqe", None), Mapping):
        return copy.deepcopy(dict(payload))
    return _convert_hh_generic_static_wrapper(payload, source_entry=source_entry, source_path=source_path)


def _snake_payload_has_route_a_production_identity(payload: Mapping[str, Any]) -> bool:
    result = _result_block(payload)
    adapt = _adapt_block(payload)
    settings = _settings_block(payload)
    candidates: list[Mapping[str, Any]] = [payload, result, adapt, settings]
    static_route_identity = adapt.get("static_route_identity", None)
    if isinstance(static_route_identity, Mapping):
        candidates.append(static_route_identity)
        observed = static_route_identity.get("observed_components", None)
        required = static_route_identity.get("required_components", None)
        if isinstance(observed, Mapping):
            candidates.append(observed)
        if isinstance(required, Mapping):
            candidates.append(required)
    route_a = any(
        str(block.get(key, "")).strip().lower() in {"route_a", "route-a"}
        for block in candidates
        for key in ("static_route_id", "route_id")
    )
    full_meta_values = {
        "full_meta",
        "math_md_full_meta_v1",
    }
    full_meta = any(
        str(block.get(key, "")).strip().lower() in full_meta_values
        for block in candidates
        for key in ("route_base_pool_key", "pool_type", "adapt_pool", "required_pool_key", "base_pool_key")
    )
    production = any(
        str(block.get(key, "")).strip() == "paper_i_production_v1"
        for block in candidates
        for key in ("meta_feature_profile", "static_algorithm_id", "algorithm_id", "method_id")
    )
    selected_standard = True
    selected_values = [
        str(block.get("selected_logical_route", "")).strip().lower()
        for block in candidates
        if block.get("selected_logical_route", None) not in {None, ""}
    ]
    if selected_values:
        selected_standard = all(value in {"standard", "off"} for value in selected_values)
    return bool((production or route_a) and full_meta and selected_standard)


def source_static_algorithm_id(payload: Mapping[str, Any], source_entry: Mapping[str, Any] | None = None) -> str:
    entry_algorithm = "" if source_entry is None else str(source_entry.get("static_algorithm_id", "")).strip()
    if entry_algorithm:
        return entry_algorithm
    result = _result_block(payload)
    adapt = _adapt_block(payload)
    for key in ("algorithm_id", "method_id", "meta_feature_profile", "static_algorithm_id"):
        raw = payload.get(key, None)
        if raw not in {None, ""}:
            return str(raw)
        raw = result.get(key, None)
        if raw not in {None, ""}:
            return str(raw)
        raw = adapt.get(key, None)
        if raw not in {None, ""}:
            return str(raw)
    return ""


def source_case_id(payload: Mapping[str, Any], source_entry: Mapping[str, Any] | None = None) -> str:
    entry_case = "" if source_entry is None else str(source_entry.get("hh_static_case_id", source_entry.get("case_id", ""))).strip()
    if entry_case:
        return entry_case
    result = _result_block(payload)
    for key in ("case_id", "benchmark_case", "source_case_id"):
        raw = payload.get(key, None)
        if raw not in {None, ""}:
            return str(raw)
        raw = result.get(key, None)
        if raw not in {None, ""}:
            return str(raw)
    return ""


def source_static_error(payload: Mapping[str, Any]) -> float | None:
    result = _result_block(payload)
    adapt = _adapt_block(payload)
    lock = payload.get("paper_ii_seed_lock", {})
    lock_map = dict(lock) if isinstance(lock, Mapping) else {}
    for block in (lock_map, payload, adapt, result):
        for key in (
            "static_abs_delta_e",
            "abs_delta_e",
            "primary_abs_delta_e",
            "higher_cutoff_reference_abs_delta_e",
            "delta_E_abs",
            "abs_delta_e_same_cutoff",
        ):
            raw = block.get(key, None)
            if raw not in {None, ""}:
                try:
                    return abs(float(raw))
                except (TypeError, ValueError):
                    continue
    return None


def source_parameter_count(payload: Mapping[str, Any]) -> int | None:
    adapt = _adapt_block(payload)
    result = _result_block(payload)
    for block in (adapt, result, payload):
        for key in ("num_parameters", "selected_operator_count", "logical_operator_count", "ansatz_depth"):
            raw = block.get(key, None)
            if raw not in {None, ""}:
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    continue
    operators = adapt.get("operators", None)
    if isinstance(operators, Sequence) and not isinstance(operators, (str, bytes)):
        return int(len(operators))
    return None


def _canonical_hh_lambda(settings: Mapping[str, Any]) -> float | None:
    for key in ("lambda_ep", "lambda"):
        raw = settings.get(key, None)
        if raw not in {None, ""}:
            return float(raw)
    raw_g = settings.get("g_ep", None)
    if raw_g not in {None, ""}:
        omega0 = float(settings.get("omega0", 1.0) or 1.0)
        if omega0 == 0.0:
            return None
        g = float(raw_g)
        return float(2.0 * g * g / omega0)
    return None


def _canonical_u_over_t(settings: Mapping[str, Any]) -> float | None:
    raw_u = settings.get("u", settings.get("U", None))
    if raw_u in {None, ""}:
        return None
    t = float(settings.get("t", 1.0) or 1.0)
    if t == 0.0:
        return None
    return float(raw_u) / t


def _assert_close(name: str, got: float | None, expected: float, *, tol: float = 1e-9) -> None:
    if got is None or not math.isclose(float(got), float(expected), rel_tol=tol, abs_tol=tol):
        raise SeedTrackValidationError(f"{name} mismatch: expected {expected}, got {got}")


def validate_hh_replay_fields(payload: Mapping[str, Any]) -> None:
    for key in ("settings", "adapt_vqe", "ground_state"):
        if not isinstance(payload.get(key, None), Mapping):
            raise SeedTrackValidationError(
                f"HH seed source must be a replayable runtime seed with top-level {key!r}; "
                "generic static table wrappers are not accepted."
            )
    adapt = _adapt_block(payload)
    operators = adapt.get("operators", None)
    if not isinstance(operators, Sequence) or isinstance(operators, (str, bytes)) or len(operators) == 0:
        raise SeedTrackValidationError("HH replay seed adapt_vqe.operators must be a non-empty sequence.")
    theta = adapt.get("optimal_point", adapt.get("theta", None))
    if not isinstance(theta, Sequence) or isinstance(theta, (str, bytes)):
        raise SeedTrackValidationError("HH replay seed adapt_vqe.optimal_point/theta must be a sequence.")


def _payload_contains_forbidden_value(payload: Mapping[str, Any], forbidden_value: str) -> bool:
    needle = forbidden_value.strip().lower()
    if not needle:
        return False
    candidates: list[Any] = []
    result = _result_block(payload)
    adapt = _adapt_block(payload)
    for block in (payload, result, adapt):
        for key in (
            "algorithm_id",
            "method_id",
            "static_algorithm_id",
            "position_policy",
            "position_optimized_geo_adapt",
            "seed_selection_policy",
            "source_artifact_json",
        ):
            if key in block:
                candidates.append(block.get(key))
    return any(str(value).strip().lower() == needle for value in candidates)


def validate_hh_seed_source(
    payload: Mapping[str, Any],
    *,
    source_entry: Mapping[str, Any],
    source_path: str | Path,
) -> dict[str, Any]:
    seed_track = str(source_entry.get("seed_track", "")).strip()
    if seed_track not in SEED_TRACKS_BY_ID:
        raise SeedTrackValidationError(f"Unknown HH seed_track {seed_track!r}; expected geo, snake, or append.")
    track = SEED_TRACKS_BY_ID[seed_track]
    regime_id = str(source_entry.get("hh_regime_id", source_entry.get("regime_id", ""))).strip()
    if regime_id not in HH_REGIMES_BY_ID:
        raise SeedTrackValidationError(f"Unknown HH regime {regime_id!r} for seed_track {seed_track!r}.")
    regime = HH_REGIMES_BY_ID[regime_id]

    source_path_text = str(source_path).lower()
    for token in track.forbidden_source_tokens:
        if token and token.lower() in source_path_text:
            raise SeedTrackValidationError(
                f"{seed_track} source path contains forbidden token {token!r}: {source_path}"
            )
    for forbidden in track.forbidden_payload_values:
        if _payload_contains_forbidden_value(payload, forbidden):
            raise SeedTrackValidationError(
                f"{seed_track} source payload contains forbidden value {forbidden!r}."
            )
    if seed_track == "geo" and bool(_result_block(payload).get("position_optimized_geo_adapt", False)):
        raise SeedTrackValidationError("geo source payload is position-optimized Geo/PosGeo, not strict GeoAdapt.")

    algorithm_id = source_static_algorithm_id(payload, source_entry)
    observed_payload_algorithm_id = source_static_algorithm_id(payload, None)
    if seed_track in {"geo", "append"}:
        if observed_payload_algorithm_id != track.required_static_algorithm_id:
            raise SeedTrackValidationError(
                f"{seed_track} source payload algorithm mismatch: expected {track.required_static_algorithm_id!r}, "
                f"got {observed_payload_algorithm_id!r}."
            )
    elif seed_track == "snake" and not _snake_payload_has_route_a_production_identity(payload):
        raise SeedTrackValidationError(
            "snake source payload lacks Route-A/full_meta/paper_i_production_v1 identity markers."
        )
    if algorithm_id != track.required_static_algorithm_id:
        raise SeedTrackValidationError(
            f"{seed_track} requires static_algorithm_id={track.required_static_algorithm_id!r}; got {algorithm_id!r}."
        )

    case_id = source_case_id(payload, source_entry)
    if case_id and case_id != regime.static_case_id:
        raise SeedTrackValidationError(
            f"{seed_track}/{regime_id} source case mismatch: expected {regime.static_case_id!r}, got {case_id!r}."
        )

    family = str(payload.get("family", _result_block(payload).get("family", _settings_block(payload).get("problem", "")))).strip()
    if family and family not in {"hh", "hubbard_holstein"}:
        raise SeedTrackValidationError(f"HH seed source family must be hh/hubbard_holstein; got {family!r}.")
    for block in (source_entry, payload, _result_block(payload), _adapt_block(payload)):
        if bool(block.get("latest_phase3_source_artifact_missing_locally", False)):
            raise SeedTrackValidationError("latest_phase3_source_artifact_missing_locally=true blocks HH seed-track use.")

    validate_hh_replay_fields(payload)
    settings = _settings_block(payload)
    problem = str(settings.get("problem", "hh")).strip()
    if problem not in {"hh", "hubbard_holstein"}:
        raise SeedTrackValidationError(f"HH seed settings.problem must be hh/hubbard_holstein; got {problem!r}.")
    if int(settings.get("L", 0)) != 2:
        raise SeedTrackValidationError(f"HH seed settings.L must be 2; got {settings.get('L')!r}.")
    if int(settings.get("n_ph_max", -1)) != int(regime.n_ph_work):
        raise SeedTrackValidationError(
            f"HH seed n_ph_max mismatch for {regime_id}: expected {regime.n_ph_work}, got {settings.get('n_ph_max')!r}."
        )
    _assert_close("HH seed U/t", _canonical_u_over_t(settings), regime.u_over_t)
    _assert_close("HH seed lambda", _canonical_hh_lambda(settings), regime.lambda_ep)

    return {
        "hh_regime_id": regime.regime_id,
        "hh_static_case_id": regime.static_case_id,
        "seed_track": track.seed_track,
        "static_seed_display_label": track.display_label,
        "static_algorithm_id": track.required_static_algorithm_id,
        "seed_selection_policy": str(source_entry.get("seed_selection_policy", track.seed_selection_policy)),
        "static_abs_delta_e": source_static_error(payload),
        "static_parameter_count": source_parameter_count(payload),
        "runtime_loadability_status": "payload_fields_present_dry_load_not_run",
    }


def dry_load_hh_seed_payload(payload: Mapping[str, Any], *, artifact_json: str | Path = "runtime_seed.json") -> dict[str, Any]:
    try:
        from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input_from_payload

        loaded = load_scaffold_runtime_input_from_payload(
            payload,
            artifact_json=artifact_json,
            loader_mode="replay_family",
            generator_family="full_meta",
            fallback_family="full_meta",
        )
    except Exception as exc:
        raise SeedTrackValidationError(f"HH runtime seed dry-load failed: {type(exc).__name__}: {exc}") from exc
    return {
        "runtime_loadability_status": "hh_replay_dry_load_passed",
        "logical_operator_count": int(getattr(loaded.base_layout, "logical_parameter_count", 0)),
        "runtime_parameter_count": int(getattr(loaded.base_layout, "runtime_parameter_count", 0)),
        "selected_term_count": int(len(getattr(loaded, "selected_terms", ()))),
    }


def normalize_hh_seed_payload(
    payload: Mapping[str, Any],
    *,
    source_entry: Mapping[str, Any],
    source_path: str | Path,
    source_sha256: str,
    runtime_loadability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = coerce_hh_seed_payload(payload, source_entry=source_entry, source_path=source_path)
    summary = validate_hh_seed_source(payload, source_entry=source_entry, source_path=source_path)
    runtime_status = dict(runtime_loadability) if isinstance(runtime_loadability, Mapping) else {}
    if runtime_status:
        summary.update(runtime_status)
    out = copy.deepcopy(dict(payload))
    lock = dict(out.get("paper_ii_seed_lock", {})) if isinstance(out.get("paper_ii_seed_lock"), Mapping) else {}
    lock.update(
        {
            "seed_track": summary["seed_track"],
            "static_seed_display_label": summary["static_seed_display_label"],
            "static_algorithm_id": summary["static_algorithm_id"],
            "hh_regime_id": summary["hh_regime_id"],
            "hh_static_case_id": summary["hh_static_case_id"],
            "source_artifact_json": str(source_path),
            "source_artifact_sha256": str(source_sha256),
            "seed_selection_policy": summary["seed_selection_policy"],
            "static_abs_delta_e": summary["static_abs_delta_e"],
            "static_parameter_count": summary["static_parameter_count"],
            "runtime_loadability_status": summary["runtime_loadability_status"],
            "runtime_logical_operator_count": summary.get("logical_operator_count"),
            "runtime_parameter_count": summary.get("runtime_parameter_count"),
            "runtime_selected_term_count": summary.get("selected_term_count"),
            "same_cutoff": False,
            "paper_i_revised_hh_cartesian_grid": True,
        }
    )
    out["paper_ii_seed_lock"] = lock
    out["paper_ii_static_seed_export"] = {
        "schema": "paper_ii_hh_static_seed_export_v1",
        **summary,
        "source_artifact_json": str(source_path),
        "source_artifact_sha256": str(source_sha256),
    }
    out["replay_candidate_pool_mode"] = "complete"
    out["replay_candidate_pool_mode_reason"] = (
        "Paper-II seed-track dynamics includes adaptive pVQD/AVQDS comparators "
        "that require the benchmark-local candidate pool, not selected terms only."
    )
    out["source_artifact_json"] = str(source_path)
    out.setdefault("pipeline", "paper_ii_hh_seed_track_static_seed_export_v1")
    return out


def load_source_registry(
    path: str | Path,
    *,
    root: Path = ROOT,
    required_regime_ids: Sequence[str] | None = None,
    required_seed_tracks: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json(path, root=root)
    if payload.get("schema") != SOURCE_REGISTRY_SCHEMA:
        raise SeedTrackValidationError(
            f"Source registry {path} schema must be {SOURCE_REGISTRY_SCHEMA!r}; got {payload.get('schema')!r}."
        )
    sources = payload.get("sources", [])
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise SeedTrackValidationError("Source registry sources must be a list.")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise SeedTrackValidationError("Every source registry entry must be an object.")
        entry = dict(raw)
        regime_id = str(entry.get("hh_regime_id", entry.get("regime_id", ""))).strip()
        seed_track = str(entry.get("seed_track", "")).strip()
        key = (regime_id, seed_track)
        if key in out:
            raise SeedTrackValidationError(f"Duplicate source registry entry for {regime_id}/{seed_track}.")
        out[key] = entry
    if required_regime_ids is None:
        regimes = HH_REGIMES
    else:
        requested = {str(regime_id).strip() for regime_id in required_regime_ids}
        regimes = tuple(regime for regime in HH_ALL_REGIMES if regime.regime_id in requested)
        missing_regimes = sorted(requested.difference(regime.regime_id for regime in regimes))
        if missing_regimes:
            raise SeedTrackValidationError(f"Unknown required HH regime id(s): {missing_regimes}.")
    if required_seed_tracks is None:
        tracks = SEED_TRACK_SPECS
    else:
        requested_tracks = {str(track).strip() for track in required_seed_tracks}
        tracks = tuple(track for track in SEED_TRACK_SPECS if track.seed_track in requested_tracks)
        missing_tracks = sorted(requested_tracks.difference(track.seed_track for track in tracks))
        if missing_tracks:
            raise SeedTrackValidationError(f"Unknown required HH seed track(s): {missing_tracks}.")
    expected = {(regime.regime_id, track.seed_track) for regime in regimes for track in tracks}
    missing = sorted(expected.difference(out))
    extra = sorted(set(out).difference(expected)) if required_regime_ids is None else []
    if missing or extra:
        expected_count = len(expected)
        raise SeedTrackValidationError(
            f"Source registry must cover exactly {expected_count} HH sources; missing={missing}, extra={extra}."
        )
    if required_regime_ids is not None:
        return {key: value for key, value in out.items() if key in expected}
    return out


def seed_lock_for_case(
    *,
    regime: HHRegimeSpec,
    track: SeedTrackSpec,
    drive_label: str,
    seed_artifact_json: str,
    seed_artifact_sha256: str,
    source_artifact_json: str,
    source_artifact_sha256: str,
    normalized_seed_artifact_sha256: str,
    normalized_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "same_seed_comparator_group_id": hh_same_seed_group_id(regime.regime_id, track.seed_track, drive_label),
        "seed_track": track.seed_track,
        "static_seed_display_label": track.display_label,
        "static_algorithm_id": track.required_static_algorithm_id,
        "hh_regime_id": regime.regime_id,
        "hh_static_case_id": regime.static_case_id,
        "hh_u_over_t": regime.u_over_t,
        "hh_lambda": regime.lambda_ep,
        "n_ph_work": regime.n_ph_work,
        "n_ph_ref": regime.n_ph_ref,
        "normalized_seed_artifact_json": seed_artifact_json,
        "normalized_seed_artifact_sha256": normalized_seed_artifact_sha256,
        "seed_artifact_sha256": seed_artifact_sha256,
        "source_artifact_json": source_artifact_json,
        "selected_static_seed_source": source_artifact_json,
        "source_artifact_sha256": source_artifact_sha256,
        "seed_selection_policy": str(normalized_summary.get("seed_selection_policy", track.seed_selection_policy)),
        "static_abs_delta_e": normalized_summary.get("static_abs_delta_e"),
        "static_parameter_count": normalized_summary.get("static_parameter_count"),
        "runtime_loadability_status": normalized_summary.get("runtime_loadability_status"),
        "latest_phase3_source_artifact_missing_locally": False,
    }


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
