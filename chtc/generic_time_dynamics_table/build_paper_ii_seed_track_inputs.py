#!/usr/bin/env python3
"""Build Paper-II dynamics seed-track CHTC inputs.

This replaces the staged-seed Table-I batch with explicit static-ADAPT seed
tracks.  It keeps POS-GEO and SNAKE/Phase-3 seeds separate so downstream table
aggregation can decide which track is paper-facing instead of silently mixing
seed provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path("chtc/generic_time_dynamics_table/input")
SEED_DIR = INPUT_DIR / "seed_artifacts_paper_ii_seed_tracks_v2"
CASE_MANIFEST = INPUT_DIR / "paper_ii_seed_tracks_cases_v2.json"
RECORDS_TSV = INPUT_DIR / "paper_ii_seed_tracks_records_v2.tsv"
TABLE1_IDS = INPUT_DIR / "paper_ii_seed_tracks_table1_record_ids_v2.txt"
CONTROLLER_IDS = INPUT_DIR / "paper_ii_seed_tracks_controller_full_record_ids_v2.txt"
SMOKE_RECORDS_TSV = INPUT_DIR / "paper_ii_seed_tracks_smoke_records_v2.tsv"
SMOKE_IDS = INPUT_DIR / "paper_ii_seed_tracks_smoke_record_ids_v2.txt"
SEED_LEDGER = INPUT_DIR / "paper_ii_seed_tracks_seed_ledger_v2.json"

SUBMIT_SMOKE = Path("chtc/generic_time_dynamics_table/submit_paper_ii_seed_tracks_smoke_v2.sub")
SUBMIT_TABLE1 = Path("chtc/generic_time_dynamics_table/submit_paper_ii_seed_tracks_table1_v2.sub")
SUBMIT_CONTROLLER = Path("chtc/generic_time_dynamics_table/submit_paper_ii_seed_tracks_controller_full_v2.sub")

CLASS_SETTINGS = "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"

ARCHIVE_ROOT = Path("artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs")

FAMILIES: tuple[dict[str, str], ...] = (
    {"family": "hubbard", "case": "hubbard_L2", "table_class": "fermionic_lattice", "tuning_class": "fermionic"},
    {"family": "ionic_hubbard", "case": "ionic_hubbard_L2", "table_class": "fermionic_lattice", "tuning_class": "fermionic"},
    {"family": "extended_hubbard", "case": "extended_hubbard_L2", "table_class": "fermionic_lattice", "tuning_class": "fermionic"},
    {"family": "ttprime_hubbard", "case": "ttprime_hubbard_L2", "table_class": "fermionic_lattice", "tuning_class": "fermionic"},
    {"family": "spinless_tv", "case": "spinless_tv_L2", "table_class": "spinless_fermion_lattice", "tuning_class": "fermionic"},
    {"family": "bose_hubbard", "case": "bose_hubbard_L2", "table_class": "boson_chain", "tuning_class": "bosonic"},
    {"family": "harmonic_kerr_chain", "case": "harmonic_kerr_chain_L2", "table_class": "boson_chain", "tuning_class": "bosonic"},
    {"family": "spin_boson", "case": "spin_boson_L1", "table_class": "spin_boson", "tuning_class": "hybrid"},
    {"family": "hh", "case": "hh_L2", "table_class": "hubbard_holstein", "tuning_class": "hybrid"},
    {"family": "molecular_vibronic_h2", "case": "molecular_vibronic_h2_L2", "table_class": "molecular_vibronic", "tuning_class": "hybrid"},
)

ALGORITHMS: tuple[str, ...] = (
    # The exact trajectory is a diagnostic reference used inside each row's
    # error columns, not a Table-I competitor method.  Keeping it as a queued
    # CHTC row wastes runtime and can make smoke/full batches look hung.
    "dyn_fixed_mclachlan",
    "dyn_product_formula_envelope",
    "dyn_qdrift",
    "dyn_fixed_pvqd",
    "dyn_adaptive_pvqd",
    "dyn_avqds",
    "dyn_avqds_t",
)

DRIVES: tuple[tuple[str, float], ...] = (("A0p2", 0.2), ("A0p6", 0.6))

# Current-best locally available Phase-3/SNAKE-like static seeds, chosen by
# source static energy error while preserving a real runtime-loadable seed JSON.
SNAKE_SOURCE_OVERRIDES: dict[str, str] = {
    "hubbard": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0031/hubbard_L2/json/result.json",
    "ionic_hubbard": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0035/ionic_hubbard_L2/json/result.json",
    "extended_hubbard": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0031/extended_hubbard_L2/json/result.json",
    "ttprime_hubbard": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0031/ttprime_hubbard_L2/json/result.json",
    "spinless_tv": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/tripartite_spsa_A_current_collective_fermionic_smallrobust_target5e5_v1/run/trial_0033/spinless_tv_L2/json/result.json",
    "bose_hubbard": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/routeA_spsa_bosonic_L2_nph1_bh_hk_hkwarm_smallrobust_target5e5_v1/run/trial_0034/bose_hubbard_L2/json/result.json",
    "harmonic_kerr_chain": "artifacts/agent_runs/20260513_hk_l2_nph1_accuracy_ceiling_optuna/run/trial_0001/result.json",
    "spin_boson": "artifacts/chtc_archives/20260512_phase3_access_point_cleanup/raw_outputs/generic_static_table/static_table__spin_boson__spin_boson_L1__static_family_native_adapt_phase3/result/spin_boson_L1/json/result.json",
    "hh": "artifacts/agent_runs/20260504_hh_l2_nph1_spsa_cost_shot_optuna_remote_v1/hh_L2/trial_0012/hh_L2/json/result.json",
    "molecular_vibronic_h2": "raw_outputs/generic_static_table/static_table__molecular_vibronic_h2__molecular_vibronic_h2_L2__static_family_native_adapt_phase3/result/molecular_vibronic_h2_L2/json/result.json",
}


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _repo_path(path).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _parse_base_args(args: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
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
        "v-nn": "v_nn",
        "t-prime": "t_prime",
        "n-fermions": "n_fermions",
        "molecular-problem-json": "molecular_problem_json",
    }
    float_keys = {"t", "u", "dv", "omega0", "g_ep", "v_nn", "t_prime"}
    int_keys = {"L", "n_ph_max", "n_fermions"}
    while i < len(args):
        token = str(args[i])
        if not token.startswith("--"):
            i += 1
            continue
        raw_key = token[2:]
        mapped = key_map.get(raw_key)
        if mapped is None:
            i += 2
            continue
        if i + 1 >= len(args):
            raise ValueError(f"Missing value after {token}")
        raw_value = args[i + 1]
        if mapped in float_keys:
            value: Any = float(raw_value)
        elif mapped in int_keys:
            value = int(raw_value)
        else:
            value = str(raw_value)
        out[mapped] = value
        i += 2
    out.setdefault("t", 1.0)
    out.setdefault("u", 4.0)
    out.setdefault("dv", 0.0)
    out.setdefault("omega0", 1.0)
    out.setdefault("g_ep", 0.5)
    out.setdefault("n_ph_max", 1)
    out.setdefault("boson_encoding", "binary")
    out.setdefault("ordering", "blocked")
    out.setdefault("boundary", "open")
    out.setdefault("v_nn", 0.0)
    out.setdefault("t_prime", 0.0)
    out.setdefault("molecular_problem_json", None)
    return out


def _source_error(payload: Mapping[str, Any]) -> float | None:
    blocks = []
    if isinstance(payload.get("adapt_vqe"), Mapping):
        blocks.append(payload["adapt_vqe"])
    if isinstance(payload.get("result"), Mapping):
        blocks.append(payload["result"])
    blocks.append(payload)
    for block in blocks:
        for key in ("abs_delta_e", "delta_E_abs", "delta_e"):
            value = block.get(key) if isinstance(block, Mapping) else None
            if value is not None:
                return abs(float(value))
    return None


def _source_parameter_count(payload: Mapping[str, Any]) -> int | None:
    blocks = []
    if isinstance(payload.get("adapt_vqe"), Mapping):
        blocks.append(payload["adapt_vqe"])
    if isinstance(payload.get("result"), Mapping):
        blocks.append(payload["result"])
    blocks.append(payload)
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        for key in ("num_parameters", "logical_num_parameters", "selected_operator_count", "adapt_depth_reached"):
            value = block.get(key)
            if value is not None:
                return int(value)
    return None


def _convert_generic_static_wrapper(payload: Mapping[str, Any], *, source_path: str, family: str, case: str, track: str) -> dict[str, Any]:
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"Generic static wrapper missing result block: {source_path}")
    settings = _parse_base_args(list(payload.get("spec", {}).get("base_pipeline_args", [])) if isinstance(payload.get("spec"), Mapping) else [])
    settings.setdefault("problem", result.get("problem", family))
    settings.setdefault("L", int(result.get("L", 2)))
    selected = list(result.get("selected_operators") or result.get("operators") or [])
    theta = list(result.get("theta") or result.get("optimal_point") or [])
    if not selected or not theta:
        raise ValueError(f"Cannot export POS-GEO wrapper without selected operators/theta: {source_path}")
    if len(selected) != len(theta):
        raise ValueError(f"Selected operator/theta length mismatch in {source_path}: {len(selected)} vs {len(theta)}")
    exact = result.get("exact_gs_energy", result.get("exact_energy"))
    energy = result.get("energy", result.get("optimizer_reported_energy"))
    abs_delta = _source_error(payload)
    adapt_vqe = {
        "success": bool(result.get("success", True)),
        "method": str(payload.get("algorithm_id", payload.get("method_id", "static_pos_geo_adapt_vqe"))),
        "static_route_identity": f"Paper-II {track} static seed export",
        "energy": energy,
        "energy_source": "generic_static_table_payload",
        "exact_gs_energy": exact,
        "delta_e": None if energy is None or exact is None else float(energy) - float(exact),
        "abs_delta_e": abs_delta,
        "num_parameters": len(theta),
        "logical_num_parameters": len(theta),
        "ansatz_depth": int(result.get("adapt_depth_reached", len(theta))),
        "operators": selected,
        "optimal_point": theta,
        "logical_optimal_point": theta,
        "pool_type": str(result.get("pool_name", result.get("required_pool_key", "full_meta"))),
        "adapt_pool": str(result.get("pool_name", result.get("required_pool_key", "full_meta"))),
        "parameterization_mode": "legacy_logical_theta_runtime_expansion",
        "pool_pauli_labels_exyz": result.get("pool_pauli_labels_exyz", {}),
        "pool_qubit_supports": result.get("pool_qubit_supports", {}),
        "selected_operator_pauli_labels_exyz": result.get("selected_operator_pauli_labels_exyz", []),
        "source_generic_static_algorithm_id": str(payload.get("algorithm_id", payload.get("method_id", "unknown"))),
        "source_generic_static_case_id": str(payload.get("case_id", case)),
    }
    return {
        "generated_utc": "2026-05-17T00:00:00+00:00",
        "pipeline": "paper_ii_seed_track_static_seed_export_v2",
        "settings": settings,
        "ground_state": {
            "exact_energy": exact,
            "exact_energy_source": "generic_static_table_same_cutoff",
        },
        "adapt_vqe": adapt_vqe,
        "paper_ii_seed_lock": {
            "seed_track": track,
            "source_artifact_json": source_path,
            "source_case_id": case,
            "source_family": family,
            "static_abs_delta_e": abs_delta,
            "same_cutoff": True,
        },
        "source_artifact_json": source_path,
    }


def _normalize_seed(source_path: str, *, family: str, case: str, track: str) -> dict[str, Any]:
    payload = _read_json(source_path)
    if isinstance(payload.get("settings"), Mapping) and isinstance(payload.get("adapt_vqe"), Mapping):
        out = dict(payload)
        lock = dict(out.get("paper_ii_seed_lock", {})) if isinstance(out.get("paper_ii_seed_lock"), Mapping) else {}
        lock.update({
            "seed_track": track,
            "source_artifact_json": source_path,
            "source_case_id": case,
            "source_family": family,
            "static_abs_delta_e": _source_error(out),
            "same_cutoff": True,
        })
        out["paper_ii_seed_lock"] = lock
        out["source_artifact_json"] = source_path
        out.setdefault("pipeline", "paper_ii_seed_track_static_seed_export_v2")
        return out
    return _convert_generic_static_wrapper(payload, source_path=source_path, family=family, case=case, track=track)


def _posgeo_source(family: str, case: str) -> str:
    return str(Path("raw_outputs/generic_static_table") / f"static_table__{family}__{case}__static_pos_geo_adapt_vqe" / "result" / "generic_static_single.json")


def _seed_tracks_for_family(family: str, case: str) -> tuple[tuple[str, str, str], ...]:
    # HH legacy runtime loading requires a full replay payload with
    # ansatz_input_state.  The current POS-GEO HH artifact is a generic static
    # table wrapper only, so including it would make every HH POS-GEO dynamics
    # row fail.  Keep the loadable SNAKE/Phase-3 seed for HH until a proper
    # POS-GEO HH replay export exists.
    if family == "hh":
        return (("snake", SNAKE_SOURCE_OVERRIDES[family], "current_best_static_snake_or_phase3_seed_20260517"),)
    return (
        ("posgeo", _posgeo_source(family, case), "current_best_static_posgeo_adapt_seed_20260517"),
        ("snake", SNAKE_SOURCE_OVERRIDES[family], "current_best_static_snake_or_phase3_seed_20260517"),
    )


def _drive_block(amplitude: float) -> dict[str, Any]:
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


def _case_id(family: str, track: str, drive_label: str) -> str:
    return f"table1_{family}_{track}_{drive_label}_t8_dt321_seedtracks_v2"


def _record_id(case_id: str, algorithm_id: str) -> str:
    return f"paper_ii_seedtracks_v2_{case_id}_{algorithm_id}"


def build_inputs(*, smoke_only: bool = False) -> None:
    cases: list[dict[str, Any]] = []
    seed_ledger: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    controller_records: list[dict[str, str]] = []
    smoke_records: list[dict[str, str]] = []

    for spec in FAMILIES:
        family = spec["family"]
        case = spec["case"]
        for track, source, policy in _seed_tracks_for_family(family, case):
            if not _repo_path(source).exists():
                raise FileNotFoundError(f"Missing static seed source for {family}/{track}: {source}")
            seed_payload = _normalize_seed(source, family=family, case=case, track=track)
            rel_seed = SEED_DIR / f"{family}_{track}_seed.json"
            _write_json(rel_seed, seed_payload)
            seed_sha = _sha256_file(rel_seed)
            source_sha = _sha256_file(source)
            static_error = _source_error(seed_payload)
            seed_info = {
                "family": family,
                "benchmark_case": case,
                "seed_track": track,
                "seed_selection_policy": policy,
                "seed_artifact_json": str(rel_seed.relative_to(INPUT_DIR)),
                "seed_artifact_sha256": seed_sha,
                "source_artifact_json": source,
                "source_artifact_sha256": source_sha,
                "static_abs_delta_e": static_error,
                "static_parameter_count": _source_parameter_count(seed_payload),
                "latest_phase3_source_artifact_missing_locally": False,
            }
            seed_ledger.append(seed_info)
            for drive_label, amplitude in DRIVES:
                cid = _case_id(family, track, drive_label)
                same_seed_group_id = f"{family}_{track}_{drive_label}_t8_dt321_same_seed_v2"
                metadata = {
                    "canonical_case_manifest_id": "paper_ii_seed_tracks_cases_v2",
                    "paper_ii_seed_track_case_manifest": True,
                    "paper_ii_table_lock": True,
                    "controller_settings_scope": "coarse_hamiltonian_class",
                    "static_scaffold_scope": "benchmark_point",
                    "enable_drive": True,
                    "disable_drive": False,
                    "drive": _drive_block(amplitude),
                    "time_dependence": "driven_staggered_midpoint",
                    "latest_phase3_source_artifact_missing_locally": False,
                    "seed_lock": {
                        "same_seed_comparator_group_id": same_seed_group_id,
                        "seed_track": track,
                        "seed_artifact_sha256": seed_sha,
                        "source_artifact_sha256": source_sha,
                        "seed_selection_policy": policy,
                        "selected_static_seed_source": source,
                        "source_artifact_json": source,
                        "normalized_seed_artifact_json": str(rel_seed.relative_to(INPUT_DIR)),
                        "static_abs_delta_e": static_error,
                        "static_parameter_count": _source_parameter_count(seed_payload),
                    },
                    "qpu_faithful_controller_data_contract": "measurement_compatible_prepared_state_observables_only",
                    "diagnostic_exact_reference_mode": "benchmark_exact_reporting_only",
                }
                case_payload = {
                    "case_id": cid,
                    "family": family,
                    "table_class": spec["table_class"],
                    "tuning_class": spec["tuning_class"],
                    "artifact_json": str(rel_seed.relative_to(INPUT_DIR)),
                    "description": f"Paper-II Table-I {track} static-seed run for {family}, A={amplitude}",
                    "loader_mode": "replay_family",
                    "generator_family": "full_meta",
                    "fallback_family": "full_meta",
                    "append_pool_family": "full_meta",
                    "t_final": 8.0,
                    "num_times": 321,
                    "metadata": metadata,
                }
                cases.append(case_payload)
                for algorithm_id in ALGORITHMS:
                    rec = {
                        "record_id": _record_id(cid, algorithm_id),
                        "kind": "benchmark",
                        "family": family,
                        "tuning_class": spec["tuning_class"],
                        "case_id": cid,
                        "algorithm_id": algorithm_id,
                        "variants": "",
                        "case_manifest": str(CASE_MANIFEST),
                    }
                    records.append(rec)
                    if (family, track, drive_label, algorithm_id) in {
                        ("hubbard", "posgeo", "A0p2", "dyn_fixed_mclachlan"),
                        ("hubbard", "posgeo", "A0p2", "dyn_product_formula_envelope"),
                        ("harmonic_kerr_chain", "posgeo", "A0p2", "dyn_fixed_mclachlan"),
                        ("harmonic_kerr_chain", "snake", "A0p2", "dyn_fixed_mclachlan"),
                    }:
                        smoke_records.append(rec)
                ctl = {
                    "record_id": _record_id(cid, "dyn_controller_full"),
                    "kind": "ablation",
                    "family": family,
                    "tuning_class": spec["tuning_class"],
                    "case_id": cid,
                    "algorithm_id": "dyn_controller_ablation_matrix",
                    "variants": "full_controller",
                    "case_manifest": str(CASE_MANIFEST),
                }
                # The controller-only CHTC submit queues the explicit
                # dyn_controller_full record ids below.  Keep those exact ids in
                # the master TSV; otherwise run_task_apptainer cannot resolve
                # them even though CONTROLLER_IDS is correct.  The broader
                # ablation-matrix rows are submitted through a separate route,
                # not through this Paper-II Table-I same-seed controller pass.
                records.append(ctl)
                controller_records.append(ctl)
                if (family, track, drive_label) in {
                    ("hubbard", "posgeo", "A0p2"),
                    ("harmonic_kerr_chain", "posgeo", "A0p2"),
                    ("harmonic_kerr_chain", "snake", "A0p2"),
                }:
                    smoke_records.append(ctl)

    manifest = {
        "manifest_id": "paper_ii_seed_tracks_cases_v2",
        "schema": "paper_ii_time_dynamics_seed_track_case_manifest_v2",
        "generated_utc": "2026-05-17T00:00:00+00:00",
        "case_count": len(cases),
        "canonical_controller_policy_classes": ["fermionic", "bosonic", "hybrid"],
        "controller_settings_scope": "coarse_hamiltonian_class",
        "exact_reference_policy": "diagnostic_only_not_controller_input",
        "seed_tracks": ["posgeo", "snake"],
        "notes": [
            "Every method within a case shares the same normalized static seed artifact hash.",
            "POS-GEO and SNAKE/Phase-3 tracks are intentionally separate and must not be averaged together unless the manuscript explicitly selects that aggregation.",
            "Exact references are diagnostic/reporting-only; controller decisions must remain measurement-compatible.",
        ],
        "cases": cases,
        "seed_ledger_path": str(SEED_LEDGER),
    }
    _write_json(CASE_MANIFEST, manifest)
    # The calibration skill consumes selected-generator ledgers as a top-level
    # route-key -> entry mapping.  Keep that shape here, with one key per
    # seed-track, so the audit can verify that case seed sources come from
    # explicit static ADAPT artifacts rather than staged dynamics artifacts.
    audit_ledger: dict[str, dict[str, Any]] = {}
    for info in seed_ledger:
        fam = str(info["family"])
        case = str(info["benchmark_case"])
        track = str(info["seed_track"])
        settings_nph = None
        try:
            seed_payload = _read_json(str(INPUT_DIR / info["seed_artifact_json"]))
            settings = seed_payload.get("settings", {}) if isinstance(seed_payload.get("settings"), Mapping) else {}
            settings_nph = int(settings.get("n_ph_max", 1))
        except Exception:
            settings_nph = 1
        key = f"{fam}|case={case}|nph={settings_nph}|track={track}"
        audit_ledger[key] = {
            "problem": fam,
            "artifact_json": info["source_artifact_json"],
            "source_artifact_json": info["source_artifact_json"],
            "normalized_seed_artifact_json": str(INPUT_DIR / info["seed_artifact_json"]),
            "abs_delta_e": info.get("static_abs_delta_e"),
            "seed_track": track,
            "seed_selection_policy": info.get("seed_selection_policy"),
            "seed_artifact_sha256": info.get("seed_artifact_sha256"),
            "source_artifact_sha256": info.get("source_artifact_sha256"),
            "label_count": info.get("static_parameter_count"),
        }
    _write_json(SEED_LEDGER, audit_ledger)

    fieldnames = ["record_id", "kind", "family", "tuning_class", "case_id", "algorithm_id", "variants", "case_manifest"]
    for path, rows in ((RECORDS_TSV, records), (SMOKE_RECORDS_TSV, smoke_records)):
        _repo_path(path).parent.mkdir(parents=True, exist_ok=True)
        with _repo_path(path).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    _repo_path(TABLE1_IDS).write_text("\n".join(r["record_id"] for r in records if r["kind"] == "benchmark") + "\n", encoding="utf-8")
    _repo_path(CONTROLLER_IDS).write_text("\n".join(r["record_id"] for r in controller_records) + "\n", encoding="utf-8")
    _repo_path(SMOKE_IDS).write_text("\n".join(r["record_id"] for r in smoke_records) + "\n", encoding="utf-8")
    _write_submit_files()


def _submit_text(*, batch: str, records_path: Path, queue_path: Path, runtime: int = 28800) -> str:
    return f"""universe = vanilla
executable = chtc/generic_time_dynamics_table/run_task_apptainer.sh
arguments = $(record_id) {records_path}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, test_support, MATH/Math.md, run_guide.md, AGENTS.md, chtc/generic_time_dynamics_table, chtc/time_dynamics_optuna/image.sif
transfer_output_files = raw_outputs, logs
log = logs/{batch}.$(Cluster).$(Process).log
output = logs/{batch}.$(Cluster).$(Process).out
error = logs/{batch}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 1
request_memory = 12GB
request_disk = 40GB
+MaxRuntime = {int(runtime)}
+JobBatchName = \"holstein-{batch}\"
environment = \"GENERIC_TD_TABLE_RECORDS_PATH={records_path} GENERIC_TD_CLASS_SETTINGS_MANIFEST={CLASS_SETTINGS} GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1\"
queue record_id from {queue_path}
"""


def _write_submit_files() -> None:
    files = {
        SUBMIT_SMOKE: _submit_text(batch="paper-ii-seedtracks-v2-smoke", records_path=SMOKE_RECORDS_TSV, queue_path=SMOKE_IDS, runtime=7200),
        SUBMIT_TABLE1: _submit_text(batch="paper-ii-seedtracks-v2-table1", records_path=RECORDS_TSV, queue_path=TABLE1_IDS, runtime=28800),
        SUBMIT_CONTROLLER: _submit_text(batch="paper-ii-seedtracks-v2-controller-full", records_path=RECORDS_TSV, queue_path=CONTROLLER_IDS, runtime=28800),
    }
    for path, text in files.items():
        _repo_path(path).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true", help="accepted for interface stability; currently writes full v2 inputs plus smoke selectors")
    args = parser.parse_args()
    build_inputs(smoke_only=bool(args.smoke_only))
    print(f"wrote {CASE_MANIFEST}")
    print(f"wrote {RECORDS_TSV}")
    print(f"wrote {SMOKE_RECORDS_TSV}")
    print(f"wrote submit files: {SUBMIT_SMOKE}, {SUBMIT_TABLE1}, {SUBMIT_CONTROLLER}")


if __name__ == "__main__":
    main()
