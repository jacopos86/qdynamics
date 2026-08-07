#!/usr/bin/env python3
"""Generate Paper-I HH Table-III SNAKE shot-proxy repair records.

This is a source-locked CHTC replay bundle for the four visible HH Table-III
SNAKE rows.  It does not run Optuna and does not change paper-facing source maps;
it reruns the visible plateau prefixes with explicit deterministic shot-proxy
inputs so the SNAKE rows can emit comparator-compatible ``shots_total`` fields.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    table_i_canonical_spec_by_case_id,
)
from pipelines.static_adapt.optimization.phase3_policy_optuna import (  # noqa: E402
    AlgorithmPolicy,
    InnerOptimizerPolicy,
    PoolPolicy,
    StaticScaffoldPolicy,
    _apply_trial_param_overrides_to_policy,
)
from pipelines.static_adapt.route_identity import (  # noqa: E402
    ROUTE_ID_A,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "paper_i_hh_tableiii_snake_shot_proxy_repair_20260612_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID

SOURCE_MAP = Path("MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json")
RECORDS = "paper_i_hh_tableiii_snake_shot_proxy_repair_records.tsv"
RECORD_IDS = "paper_i_hh_tableiii_snake_shot_proxy_repair_record_ids.txt"
MANIFEST = "paper_i_hh_tableiii_snake_shot_proxy_repair_manifest.json"
SUBMIT_FILE = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
POLICY_DIRNAME = "phase3_policies"

RUN_CLASS = "paper_i_hh_tableiii_snake_shot_proxy_repair"
TABLE_LABEL = "tab:hh_first_plateau_prefix_costs"
PRIMARY_ENERGY_METRIC = "same_cutoff_plateau_prefix_abs_delta_e_with_higher_cutoff_diagnostic"
SAME_CUTOFF_ERROR_ROLE = "primary_tableiii_metric"
SHOTS_PER_PAULI_TERM_PROXY = 1024
PARALLEL_GRADIENT_WORKERS = 4
BEAM_PARENT_WORKERS = 2
TAU_PHYS = "2e-4"

FIELDNAMES = (
    "record_id",
    "family",
    "case_id",
    "algorithm_id",
    "suite_profile",
    "mode",
    "families",
    "benchmark_ids",
    "fixed_inner_optimizer",
    "phase2_novelty_mode",
    "phase3_policy_json",
    "phase3_adapt_max_depth",
    "phase3_adapt_maxiter",
    "phase3_refit_maxiter",
    "phase3_final_maxiter",
    "phase3_adapt_spsa_a",
    "phase3_adapt_spsa_c",
    "phase3_adapt_spsa_big_a",
    "phase3_adapt_spsa_alpha",
    "phase3_adapt_spsa_gamma",
    "phase3_adapt_spsa_eval_repeats",
    "phase3_adapt_spsa_avg_last",
    "phase3_adapt_allow_repeats",
    "phase3_adapt_parallel_gradient_workers",
    "phase3_adapt_beam_parent_workers",
    "static_route_id",
    "selected_logical_route",
    "selected_logical_source_json",
    "selected_logical_transfer_mode",
    "energy_stop_target",
    "first_hit_thresholds",
    "n_ph_work",
    "n_ph_ref",
    "exact_reference_n_ph_max",
    "same_cutoff_exact_gs_energy",
    "exact_reference_energy",
    "same_cutoff_reference_energy_key",
    "reference_cutoff_energy_key",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "tau_phys",
    "shots_per_pauli_term_proxy",
    "requires_deterministic_shot_proxy",
    "run_class",
    "table_label",
    "hh_tableiii_regime",
    "source_plateau_iteration",
    "source_map_json",
    "source_json",
    "source_sha256",
    "base_records_tsv",
    "base_record_id",
    "base_records_sha256",
    "trial_param_overrides_json",
    "source_trial_param_overrides_json",
    "source_trial_param_overrides_sha256",
    "source_policy_materialization_status",
    "paper_i_recovery_intent",
    "route_evidence_role",
    "promotion_status",
    "algorithm_variant",
)


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    benchmark_id: str
    base_records_tsv: str
    base_record_id: str
    n_ph_work: int
    n_ph_ref: int


REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec(
        key="weak_weak",
        benchmark_id="hh_L2_nph2_three_model_sym_weak_weak",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_weak_weak_snake_flatnovelty_nocost_bounded_20260530_v3/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_weak_weak_full_meta_flatnovelty_nocost_bounded_v3",
        n_ph_work=2,
        n_ph_ref=5,
    ),
    RegimeSpec(
        key="strong_weak",
        benchmark_id="hh_L2_nph2_three_model_sym_strong_weak",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6",
        n_ph_work=2,
        n_ph_ref=5,
    ),
    RegimeSpec(
        key="weak_strong",
        benchmark_id="hh_L2_nph4_three_model_sym_weak_strong",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph4_three_model_sym_weak_strong_new_full_meta_flatnovelty_nocost_bounded_longtrial_v4",
        n_ph_work=4,
        n_ph_ref=7,
    ),
    RegimeSpec(
        key="strong_strong",
        benchmark_id="hh_L2_nph4_three_model_sym_strong_strong",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph4_three_model_sym_strong_strong_new_full_meta_flatnovelty_nocost_bounded_longtrial_v4",
        n_ph_work=4,
        n_ph_ref=7,
    ),
)


def _repo_relative(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    return str(p.relative_to(REPO_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str | Path) -> Any:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_source_map() -> Mapping[str, Any]:
    payload = _read_json(SOURCE_MAP)
    if not isinstance(payload, Mapping):
        raise ValueError(f"source map is not a JSON object: {SOURCE_MAP}")
    return payload


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _load_row(path: str, record_id: str) -> dict[str, str]:
    with _resolve(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    matches = [dict(row) for row in rows if row.get("record_id") == record_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source row {record_id!r} in {path}; found {len(matches)}")
    return matches[0]


def _load_trial_param_overrides(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"trial-param override file is not a JSON object: {path}")
    raw = payload.get("trial_param_overrides", payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"trial_param_overrides is not a JSON object: {path}")
    return dict(raw)


def _policy_path(output_dir: Path, regime: RegimeSpec) -> Path:
    return output_dir / POLICY_DIRNAME / f"{regime.key}_source_locked_phase3_policy.json"


def _source_locked_policy_payload(
    *,
    regime: RegimeSpec,
    source_row: Mapping[str, str],
    trial_param_overrides: Mapping[str, Any],
    source_plateau_iteration: int,
) -> dict[str, Any]:
    base_policy = AlgorithmPolicy(
        pool=PoolPolicy(pool_key="full_meta"),
        static=StaticScaffoldPolicy(
            static_route_id=ROUTE_ID_A,
            static_meta_feature_profile=STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
            adapt_allow_repeats=True,
            hardware_resolution_mode="ideal",
        ),
        inner_optimizer=InnerOptimizerPolicy(inner_optimizer="SPSA", final_optimizer_type="SPSA"),
    )
    policy = _apply_trial_param_overrides_to_policy(base_policy, trial_param_overrides)
    policy = replace(
        policy,
        static=replace(
            policy.static,
            adapt_max_depth=int(source_plateau_iteration),
            adapt_parallel_gradient_workers=PARALLEL_GRADIENT_WORKERS,
            adapt_beam_parent_workers=BEAM_PARENT_WORKERS,
            static_route_id=ROUTE_ID_A,
            static_meta_feature_profile=STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
            adapt_allow_repeats=True,
            hardware_resolution_mode="ideal",
            phase3_oracle_gradient_mode="off",
            phase3_oracle_inner_objective_mode="exact",
            phase3_oracle_value_noise_model="off",
            phase3_oracle_value_noise_std=0.0,
        ),
        inner_optimizer=replace(
            policy.inner_optimizer,
            inner_optimizer="SPSA",
            final_optimizer_type="SPSA",
        ),
    )
    return {
        "schema": "phase3_algorithm_policy_source_locked_replay_v1",
        "batch_id": BATCH_ID,
        "record_id": f"{BATCH_ID}_{regime.key}",
        "run_class": RUN_CLASS,
        "regime": regime.key,
        "benchmark_id": regime.benchmark_id,
        "source_base_records_tsv": regime.base_records_tsv,
        "source_base_record_id": regime.base_record_id,
        "source_trial_param_overrides_json": str(source_row.get("trial_param_overrides_json") or ""),
        "source_plateau_iteration": int(source_plateau_iteration),
        "materialization_policy": (
            "source trial_param_overrides applied through phase3_policy_optuna, then capped at visible "
            "Table-III SNAKE plateau depth; no Optuna search is run"
        ),
        "policy": asdict(policy),
    }


def _reference_fields(regime: RegimeSpec) -> dict[str, str]:
    spec = table_i_canonical_spec_by_case_id(
        "hh",
        regime.benchmark_id,
        TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    )
    same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=int(regime.n_ph_work))
    ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=int(regime.n_ph_ref))
    return {
        "same_cutoff_exact_gs_energy": repr(float(same_energy)),
        "exact_reference_energy": repr(float(ref_energy)),
        "same_cutoff_reference_energy_key": str(same_key),
        "reference_cutoff_energy_key": str(ref_key),
    }


def _row_for(
    *,
    output_dir: Path,
    regime: RegimeSpec,
    source_row: Mapping[str, str],
    source_entry: Mapping[str, Any],
    source_plateau_iteration: int,
    policy_payload: Mapping[str, Any],
) -> dict[str, str]:
    policy = policy_payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("generated policy payload missing policy mapping")
    static = policy.get("static") if isinstance(policy.get("static"), Mapping) else {}
    inner = policy.get("inner_optimizer") if isinstance(policy.get("inner_optimizer"), Mapping) else {}
    source_trial = str(source_row.get("trial_param_overrides_json") or "").strip()
    row: dict[str, str] = {
        "record_id": f"{BATCH_ID}_{regime.key}",
        "family": "hh",
        "case_id": regime.benchmark_id,
        "algorithm_id": "static_family_native_adapt_phase3",
        "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "mode": "generic-static-source-locked-replay",
        "families": "hh",
        "benchmark_ids": regime.benchmark_id,
        "fixed_inner_optimizer": "SPSA",
        "phase2_novelty_mode": str(static.get("phase2_novelty_mode") or "collective_span_v1"),
        "phase3_policy_json": _repo_relative(_policy_path(output_dir, regime)),
        "phase3_adapt_max_depth": str(int(source_plateau_iteration)),
        "phase3_adapt_maxiter": str(int(static.get("adapt_maxiter") or 0) or ""),
        "phase3_refit_maxiter": str(int(inner.get("refit_maxiter") or 0) or ""),
        "phase3_final_maxiter": str(int(inner.get("final_maxiter") or 0) or ""),
        "phase3_adapt_spsa_a": str(inner.get("spsa_a") or ""),
        "phase3_adapt_spsa_c": str(inner.get("spsa_c") or ""),
        "phase3_adapt_spsa_big_a": str(inner.get("spsa_A") or ""),
        "phase3_adapt_spsa_alpha": str(inner.get("spsa_alpha") or ""),
        "phase3_adapt_spsa_gamma": str(inner.get("spsa_gamma") or ""),
        "phase3_adapt_spsa_eval_repeats": "1",
        "phase3_adapt_spsa_avg_last": "0",
        "phase3_adapt_allow_repeats": "true",
        "phase3_adapt_parallel_gradient_workers": str(PARALLEL_GRADIENT_WORKERS),
        "phase3_adapt_beam_parent_workers": str(BEAM_PARENT_WORKERS),
        "static_route_id": ROUTE_ID_A,
        "selected_logical_route": "",
        "selected_logical_source_json": "",
        "selected_logical_transfer_mode": "",
        "energy_stop_target": "",
        "first_hit_thresholds": "",
        "n_ph_work": str(int(regime.n_ph_work)),
        "n_ph_ref": str(int(regime.n_ph_ref)),
        "exact_reference_n_ph_max": str(int(regime.n_ph_ref)),
        "primary_energy_metric": PRIMARY_ENERGY_METRIC,
        "same_cutoff_error_role": SAME_CUTOFF_ERROR_ROLE,
        "tau_phys": TAU_PHYS,
        "shots_per_pauli_term_proxy": str(SHOTS_PER_PAULI_TERM_PROXY),
        "requires_deterministic_shot_proxy": "true",
        "run_class": RUN_CLASS,
        "table_label": TABLE_LABEL,
        "hh_tableiii_regime": regime.key,
        "source_plateau_iteration": str(int(source_plateau_iteration)),
        "source_map_json": _repo_relative(SOURCE_MAP),
        "source_json": str(source_entry.get("source_json") or ""),
        "source_sha256": str(source_entry.get("source_sha256") or ""),
        "base_records_tsv": regime.base_records_tsv,
        "base_record_id": regime.base_record_id,
        "base_records_sha256": _sha256(regime.base_records_tsv),
        "trial_param_overrides_json": source_trial,
        "source_trial_param_overrides_json": source_trial,
        "source_trial_param_overrides_sha256": _sha256(source_trial) if source_trial else "",
        "source_policy_materialization_status": "source_trial_overrides_materialized_to_phase3_policy_json",
        "paper_i_recovery_intent": "rerun_visible_hh_tableiii_snake_plateau_prefix_for_strict_s_alg_and_shots_total",
        "route_evidence_role": "paper_i_hh_tableiii_snake_shot_proxy_repair_candidate",
        "promotion_status": "local_prep_only_not_promoted",
        "algorithm_variant": "paper_i_hh_tableiii_snake_source_locked_shot_proxy_repair",
    }
    row.update(_reference_fields(regime))
    return {key: str(row.get(key, "")) for key in FIELDNAMES}


def build_rows_and_policies(output_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any], dict[Path, dict[str, Any]]]:
    source_map = _load_source_map()
    rows: list[dict[str, str]] = []
    policies: dict[Path, dict[str, Any]] = {}
    source_details: dict[str, Any] = {}
    for regime in REGIMES:
        source_row = _load_row(regime.base_records_tsv, regime.base_record_id)
        source_entry = _nested(source_map, "regimes", regime.key, "methods", "SNAKE")
        plateau_payload = _nested(source_map, "plateau_markers", regime.key, "SNAKE")
        if not isinstance(source_entry, Mapping):
            raise ValueError(f"missing SNAKE source-map entry for {regime.key}")
        if not isinstance(plateau_payload, Mapping) or plateau_payload.get("iteration") is None:
            raise ValueError(f"missing SNAKE plateau marker for {regime.key}")
        source_plateau_iteration = int(plateau_payload["iteration"])
        source_trial = str(source_row.get("trial_param_overrides_json") or "").strip()
        if not source_trial:
            raise ValueError(f"source row {regime.base_record_id!r} has no trial_param_overrides_json")
        trial_param_overrides = _load_trial_param_overrides(source_trial)
        policy_payload = _source_locked_policy_payload(
            regime=regime,
            source_row=source_row,
            trial_param_overrides=trial_param_overrides,
            source_plateau_iteration=source_plateau_iteration,
        )
        policy_path = _policy_path(output_dir, regime)
        policies[policy_path] = policy_payload
        rows.append(
            _row_for(
                output_dir=output_dir,
                regime=regime,
                source_row=source_row,
                source_entry=source_entry,
                source_plateau_iteration=source_plateau_iteration,
                policy_payload=policy_payload,
            )
        )
        source_details[regime.key] = {
            "benchmark_id": regime.benchmark_id,
            "base_records_tsv": regime.base_records_tsv,
            "base_record_id": regime.base_record_id,
            "base_records_sha256": _sha256(regime.base_records_tsv),
            "source_json": source_entry.get("source_json"),
            "source_sha256": source_entry.get("source_sha256"),
            "source_trial_param_overrides_json": source_trial,
            "source_trial_param_overrides_sha256": _sha256(source_trial),
            "generated_phase3_policy_json": _repo_relative(policy_path),
            "n_ph_work": regime.n_ph_work,
            "n_ph_ref": regime.n_ph_ref,
            "source_plateau_iteration": source_plateau_iteration,
            "shots_per_pauli_term_proxy": SHOTS_PER_PAULI_TERM_PROXY,
            "same_cutoff_error_at_visible_plateau": plateau_payload.get("error"),
        }
    return rows, source_details, policies


def _tsv_text(rows: Sequence[Mapping[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FIELDNAMES), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return buf.getvalue()


def _ids_text(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["record_id"]) for row in rows) + "\n"


def _submit_text(*, records_path: Path, record_ids_path: Path, job_batch_name: str) -> str:
    return f"""universe = vanilla
executable = chtc/phase3_optuna/run_generic_static_table_task_apptainer.sh
arguments = $(record_id) {_repo_relative(records_path)} raw_outputs/{BATCH_ID}/$(record_id)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna
transfer_output_files = raw_outputs, logs
log = logs/{job_batch_name}.$(Cluster).$(Process).log
output = logs/{job_batch_name}.$(Cluster).$(Process).out
error = logs/{job_batch_name}.$(Cluster).$(Process).err
stream_output = False
stream_error = False
requirements = TARGET.HasSIF
request_cpus = {PARALLEL_GRADIENT_WORKERS}
request_memory = 32GB
request_disk = 122880MB
+MaxRuntime = 172800
+JobBatchName = "holstein-{job_batch_name}"
queue record_id from {_repo_relative(record_ids_path)}
"""


def _manifest_payload(*, rows: Sequence[Mapping[str, str]], source_details: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_tableiii_snake_shot_proxy_repair_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_utc": "2026-06-12T00:00:00+00:00",
        "generated_by": "chtc/phase3_optuna/generate_paper_i_hh_tableiii_snake_shot_proxy_repair_records.py",
        "run_class": RUN_CLASS,
        "table_label": TABLE_LABEL,
        "records_tsv": _repo_relative(output_dir / RECORDS),
        "record_id_file": _repo_relative(output_dir / RECORD_IDS),
        "submit_file": _repo_relative(SUBMIT_FILE),
        "record_ids": [str(row["record_id"]) for row in rows],
        "record_count": len(rows),
        "regime_order": [regime.key for regime in REGIMES],
        "shots_per_pauli_term_proxy": SHOTS_PER_PAULI_TERM_PROXY,
        "shot_proxy_formula": (
            "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * "
            "(energy_eval_count_proxy + gradient_operator_probe_count_proxy + metric_operator_probe_count_proxy)"
        ),
        "execution_contract": {
            "runner": "chtc/phase3_optuna/run_generic_static_table_task_apptainer.sh",
            "arguments": "$(record_id) <records.tsv> raw_outputs/<batch>/$(record_id)",
            "optuna_search": False,
            "source_policy_materialization": "source trial_param_overrides applied once to generated phase3_policy_json",
            "depth_cap": "visible source-map SNAKE plateau iteration per regime",
            "submit_status": "not_submitted_by_generator",
        },
        "progress_contract": {
            "transfer_policy": "ON_EXIT_OR_EVICT",
            "transfer_output_files": ["raw_outputs", "logs"],
            "expected_result_json": "raw_outputs/<batch>/<record_id>/result/generic_static_single.json",
            "expected_raw_phase3_result_json": "raw_outputs/<batch>/<record_id>/result/result.json",
            "expected_effective_env_overlay": "raw_outputs/<batch>/<record_id>/effective_env_overlay.json",
        },
        "source_map": _repo_relative(SOURCE_MAP),
        "source_details": source_details,
    }


def render_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    records_path = output_dir / RECORDS
    ids_path = output_dir / RECORD_IDS
    manifest_path = output_dir / MANIFEST
    rows, source_details, policies = build_rows_and_policies(output_dir)
    manifest = _manifest_payload(rows=rows, source_details=source_details, output_dir=output_dir)
    files: dict[Path, str] = {
        records_path: _tsv_text(rows),
        ids_path: _ids_text(rows),
        manifest_path: json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        SUBMIT_FILE: _submit_text(records_path=records_path, record_ids_path=ids_path, job_batch_name=BATCH_ID),
    }
    for policy_path, policy_payload in policies.items():
        files[policy_path] = json.dumps(policy_payload, indent=2, sort_keys=True) + "\n"
    return {"rows": rows, "manifest": manifest, "files": files}


def write_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    rendered = render_artifacts(output_dir)
    for path, text in rendered["files"].items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(text), encoding="utf-8")
    return rendered


def check_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    rendered = render_artifacts(output_dir)
    mismatches: list[str] = []
    for path, expected_text in rendered["files"].items():
        p = Path(path)
        if not p.exists():
            mismatches.append(f"missing:{_repo_relative(p)}")
            continue
        actual = p.read_text(encoding="utf-8")
        if actual != expected_text:
            mismatches.append(f"content_mismatch:{_repo_relative(p)}")
    return {"ok": not mismatches, "mismatches": mismatches, **rendered}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write", action="store_true", help="write generated TSV/manifest/submit/policy files")
    parser.add_argument("--check", action="store_true", help="fail if generated files are absent or stale")
    args = parser.parse_args(argv)

    if args.write:
        rendered = write_artifacts(args.output_dir)
        print(json.dumps({"status": "written", "record_count": len(rendered["rows"]), "output_dir": _repo_relative(args.output_dir)}, indent=2))
        return 0
    if args.check:
        checked = check_artifacts(args.output_dir)
        if not checked["ok"]:
            print(json.dumps({"status": "stale", "mismatches": checked["mismatches"]}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps({"status": "ok", "record_count": len(checked["rows"])}, indent=2))
        return 0
    rendered = render_artifacts(args.output_dir)
    print(json.dumps(rendered["manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
