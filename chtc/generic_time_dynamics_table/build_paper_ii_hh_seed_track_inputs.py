#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# RESTORED 2026-09-01 -- B4 of the open-decisions queue (classification row 31).
# Source: origin/codex/wip-local-active-source-snapshot-20260706-120000 @
# 517eca4c, blob 13f42c99f2a25c02d5ad98a85c02baf19a6fc6b7, 2026-07-08.
# Content below this header is VERBATIM from that blob; nothing was
# forward-ported. It was never grafted onto trunk, which left
# test/chtc/test_generic_time_dynamics_table_inputs.py a hard collection ERROR
# that silently masked 33 tests.
# ---------------------------------------------------------------------------
"""Build Paper-II HH seed-track CHTC inputs.

This is the additive v1 Hubbard--Holstein surface for strict GeoAdapt, SNAKE,
and append-only ADAPT static seeds.  It intentionally does not mutate the
historical all-family v2 POS-GEO/SNAKE input surface.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping

from chtc.generic_time_dynamics_table.paper_ii_seed_track_common import (
    CLASS_SETTINGS_LOCK,
    DRIVES,
    FULL_BENCHMARK_ALGORITHMS,
    HH_REGIMES,
    INPUT_DIR,
    ROOT,
    SEED_TRACK_CASE_SCHEMA,
    SEED_TRACK_LEDGER_SCHEMA,
    SEED_TRACK_SPECS,
    VISIBLE_ALGORITHMS,
    drive_block,
    dry_load_hh_seed_payload,
    hh_record_id,
    hh_same_seed_group_id,
    hh_seed_artifact_relpath,
    hh_seed_track_case_id,
    load_source_registry,
    normalize_hh_seed_payload,
    read_json,
    repo_path,
    seed_lock_for_case,
    sha256_file,
    source_parameter_count,
    source_static_error,
    write_json,
    coerce_hh_seed_payload,
)

SEED_DIR = INPUT_DIR / "seed_artifacts_paper_ii_hh_seed_tracks_v1"
SOURCE_REGISTRY = INPUT_DIR / "paper_ii_hh_static_seed_sources_v1.json"
CASE_MANIFEST = INPUT_DIR / "paper_ii_hh_seed_tracks_cases_v1.json"
SEED_LEDGER = INPUT_DIR / "paper_ii_hh_seed_tracks_seed_ledger_v1.json"
AUDIT_COMPAT_SEED_LEDGER = INPUT_DIR / "paper_ii_hh_seed_tracks_seed_ledger_audit_compat_v1.json"
RECORDS_TSV = INPUT_DIR / "paper_ii_hh_seed_tracks_records_v1.tsv"
VISIBLE_IDS = INPUT_DIR / "paper_ii_hh_seed_tracks_visible_record_ids_v1.txt"
FULL_IDS = INPUT_DIR / "paper_ii_hh_seed_tracks_full_record_ids_v1.txt"
CONTROLLER_IDS = INPUT_DIR / "paper_ii_hh_seed_tracks_controller_full_record_ids_v1.txt"
SMOKE_RECORDS_TSV = INPUT_DIR / "paper_ii_hh_seed_tracks_smoke_records_v1.tsv"
SMOKE_IDS = INPUT_DIR / "paper_ii_hh_seed_tracks_smoke_record_ids_v1.txt"

SUBMIT_SMOKE = Path("chtc/generic_time_dynamics_table/submit_paper_ii_hh_seed_tracks_smoke_v1.sub")
SUBMIT_VISIBLE = Path("chtc/generic_time_dynamics_table/submit_paper_ii_hh_seed_tracks_visible_v1.sub")
SUBMIT_FULL = Path("chtc/generic_time_dynamics_table/submit_paper_ii_hh_seed_tracks_full_v1.sub")
SUBMIT_CONTROLLER = Path("chtc/generic_time_dynamics_table/submit_paper_ii_hh_seed_tracks_controller_full_v1.sub")


def _write_text(path: str | Path, text: str, *, root: Path) -> None:
    resolved = repo_path(path, root=root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def _rel_input_seed_path(regime_id: str, seed_track: str) -> Path:
    return SEED_DIR.relative_to(INPUT_DIR) / hh_seed_artifact_relpath(regime_id, seed_track).name


def _seed_summary(
    *,
    regime_id: str,
    seed_track: str,
    source_entry: Mapping[str, Any],
    source_path: str,
    source_sha: str,
    seed_rel_to_input: Path,
    seed_sha: str,
    seed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    export = seed_payload.get("paper_ii_static_seed_export", {})
    export_map = dict(export) if isinstance(export, Mapping) else {}
    return {
        "hh_regime_id": regime_id,
        "hh_static_case_id": str(source_entry.get("hh_static_case_id", source_entry.get("case_id", ""))),
        "seed_track": seed_track,
        "static_algorithm_id": export_map.get("static_algorithm_id", source_entry.get("static_algorithm_id")),
        "static_seed_display_label": export_map.get("static_seed_display_label"),
        "seed_selection_policy": export_map.get("seed_selection_policy"),
        "seed_artifact_json": str(seed_rel_to_input),
        "seed_artifact_sha256": seed_sha,
        "normalized_seed_artifact_json": str(INPUT_DIR / seed_rel_to_input),
        "normalized_seed_artifact_sha256": seed_sha,
        "source_artifact_json": source_path,
        "source_artifact_sha256": source_sha,
        "source_record_id": source_entry.get("source_record_id", source_entry.get("record_id", "")),
        "static_abs_delta_e": source_static_error(seed_payload),
        "static_parameter_count": source_parameter_count(seed_payload),
        "runtime_loadability_status": export_map.get("runtime_loadability_status"),
        "latest_phase3_source_artifact_missing_locally": False,
    }


def _audit_compat_seed_ledger(seed_ledger: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the legacy audit-script ledger shape for HH seed-track sources.

    The generic Paper-II audit helper indexes top-level ``family|...`` keys and
    validates case ``seed_lock.source_artifact_json`` against ``artifact_json``.
    The HH seed-track ledger keeps normalized seed artifacts and static source
    artifacts separate, so this compatibility view intentionally exposes the
    static source artifact as ``artifact_json`` while preserving the normalized
    runtime seed path in explicit ``seed_*`` fields.
    """

    out: dict[str, dict[str, Any]] = {}
    for entry in seed_ledger:
        regime_id = str(entry.get("hh_regime_id", ""))
        seed_track = str(entry.get("seed_track", ""))
        if not regime_id or not seed_track:
            continue
        key = f"hh|{regime_id}|{seed_track}"
        out[key] = {
            "problem": "hh",
            "artifact_json": str(entry.get("source_artifact_json", "")),
            "source_artifact_json": str(entry.get("source_artifact_json", "")),
            "source_artifact_sha256": entry.get("source_artifact_sha256"),
            "seed_artifact_json": entry.get("seed_artifact_json"),
            "seed_artifact_sha256": entry.get("seed_artifact_sha256"),
            "normalized_seed_artifact_json": entry.get("normalized_seed_artifact_json"),
            "normalized_seed_artifact_sha256": entry.get("normalized_seed_artifact_sha256"),
            "abs_delta_e": entry.get("static_abs_delta_e"),
            "static_abs_delta_e": entry.get("static_abs_delta_e"),
            "static_algorithm_id": entry.get("static_algorithm_id"),
            "static_parameter_count": entry.get("static_parameter_count"),
            "seed_selection_policy": entry.get("seed_selection_policy"),
            "runtime_loadability_status": entry.get("runtime_loadability_status"),
            "latest_phase3_source_artifact_missing_locally": entry.get(
                "latest_phase3_source_artifact_missing_locally"
            ),
            "hh_regime_id": regime_id,
            "seed_track": seed_track,
        }
    return out


def _case_payload(
    *,
    regime_id: str,
    seed_track: str,
    drive_label: str,
    amplitude: float,
    seed_rel_to_input: Path,
    seed_lock: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = hh_seed_track_case_id(regime_id, seed_track, drive_label)
    return {
        "case_id": case_id,
        "family": "hh",
        "table_class": "hubbard_holstein",
        "tuning_class": "hybrid",
        "artifact_json": str(seed_rel_to_input),
        "description": f"Paper-II HH {seed_track} static-seed run for {regime_id}, A={float(amplitude)}",
        "loader_mode": "replay_family",
        "generator_family": "full_meta",
        "fallback_family": "full_meta",
        "append_pool_family": "full_meta",
        "t_final": 8.0,
        "num_times": 321,
        "metadata": {
            "canonical_case_manifest_id": "paper_ii_hh_seed_tracks_cases_v1",
            "paper_ii_hh_seed_track_case_manifest": True,
            "paper_ii_table_lock": True,
            "controller_settings_scope": "coarse_hamiltonian_class",
            "static_scaffold_scope": "benchmark_point",
            "enable_drive": True,
            "disable_drive": False,
            "drive": drive_block(amplitude),
            "time_dependence": "driven_staggered_midpoint",
            "latest_phase3_source_artifact_missing_locally": False,
            "seed_lock": dict(seed_lock),
            "same_seed_comparator_group_id": hh_same_seed_group_id(regime_id, seed_track, drive_label),
            "qpu_faithful_controller_data_contract": "measurement_compatible_prepared_state_observables_only",
            "diagnostic_exact_reference_mode": "benchmark_exact_reporting_only",
        },
    }


def _benchmark_record(case: Mapping[str, Any], algorithm_id: str, *, visible: bool) -> dict[str, str]:
    case_id = str(case["case_id"])
    metadata = dict(case.get("metadata", {})) if isinstance(case.get("metadata"), Mapping) else {}
    seed_lock = dict(metadata.get("seed_lock", {})) if isinstance(metadata.get("seed_lock"), Mapping) else {}
    return {
        "record_id": hh_record_id(case_id, algorithm_id),
        "kind": "benchmark",
        "family": "hh",
        "tuning_class": "hybrid",
        "case_id": case_id,
        "algorithm_id": algorithm_id,
        "variants": "",
        "case_manifest": str(CASE_MANIFEST),
        "seed_track": str(seed_lock.get("seed_track", "")),
        "hh_regime_id": str(seed_lock.get("hh_regime_id", "")),
        "visible_table_method": "1" if visible else "0",
    }


def _controller_record(case: Mapping[str, Any]) -> dict[str, str]:
    case_id = str(case["case_id"])
    metadata = dict(case.get("metadata", {})) if isinstance(case.get("metadata"), Mapping) else {}
    seed_lock = dict(metadata.get("seed_lock", {})) if isinstance(metadata.get("seed_lock"), Mapping) else {}
    return {
        "record_id": hh_record_id(case_id, "dyn_controller_full"),
        "kind": "ablation",
        "family": "hh",
        "tuning_class": "hybrid",
        "case_id": case_id,
        "algorithm_id": "dyn_controller_ablation_matrix",
        "variants": "full_controller",
        "case_manifest": str(CASE_MANIFEST),
        "seed_track": str(seed_lock.get("seed_track", "")),
        "hh_regime_id": str(seed_lock.get("hh_regime_id", "")),
        "visible_table_method": "1",
    }


def _write_tsv(path: Path, rows: list[dict[str, str]], *, root: Path) -> None:
    fieldnames = [
        "record_id",
        "kind",
        "family",
        "tuning_class",
        "case_id",
        "algorithm_id",
        "variants",
        "case_manifest",
        "seed_track",
        "hh_regime_id",
        "visible_table_method",
    ]
    resolved = repo_path(path, root=root)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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
environment = \"GENERIC_TD_TABLE_RECORDS_PATH={records_path} GENERIC_TD_CLASS_SETTINGS_MANIFEST={CLASS_SETTINGS_LOCK} GENERIC_TD_REQUIRE_LOCKED_CLASS_SETTINGS=1\"
queue record_id from {queue_path}
"""


def _write_submit_files(*, root: Path) -> None:
    files = {
        SUBMIT_SMOKE: _submit_text(batch="paper-ii-hh-seedtracks-v1-smoke", records_path=SMOKE_RECORDS_TSV, queue_path=SMOKE_IDS, runtime=7200),
        SUBMIT_VISIBLE: _submit_text(batch="paper-ii-hh-seedtracks-v1-visible", records_path=RECORDS_TSV, queue_path=VISIBLE_IDS, runtime=28800),
        SUBMIT_FULL: _submit_text(batch="paper-ii-hh-seedtracks-v1-full", records_path=RECORDS_TSV, queue_path=FULL_IDS, runtime=28800),
        SUBMIT_CONTROLLER: _submit_text(batch="paper-ii-hh-seedtracks-v1-controller-full", records_path=RECORDS_TSV, queue_path=CONTROLLER_IDS, runtime=28800),
    }
    for path, text in files.items():
        _write_text(path, text, root=root)


def build_inputs(
    *,
    root: Path = ROOT,
    source_registry_path: str | Path = SOURCE_REGISTRY,
    write_submit_files: bool = True,
    require_runtime_dry_load: bool = True,
    regime_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_regime_ids = None if regime_ids is None else tuple(str(item).strip() for item in regime_ids if str(item).strip())
    regimes = [regime for regime in HH_REGIMES if selected_regime_ids is None or regime.regime_id in selected_regime_ids]
    if selected_regime_ids is not None:
        missing_requested = sorted(set(selected_regime_ids).difference(regime.regime_id for regime in regimes))
        if missing_requested:
            raise ValueError(f"Unknown HH regime-id filter(s): {missing_requested}")
    source_registry = load_source_registry(source_registry_path, root=root, required_regime_ids=selected_regime_ids)
    cases: list[dict[str, Any]] = []
    seed_ledger: list[dict[str, Any]] = []
    records: list[dict[str, str]] = []
    visible_records: list[dict[str, str]] = []
    controller_records: list[dict[str, str]] = []
    smoke_records: list[dict[str, str]] = []
    prepared_seeds: list[dict[str, Any]] = []

    for regime in regimes:
        for track in SEED_TRACK_SPECS:
            source_entry = source_registry[(regime.regime_id, track.seed_track)]
            source = str(source_entry.get("source_artifact_json", "")).strip()
            if not source:
                raise FileNotFoundError(f"Missing source_artifact_json for {regime.regime_id}/{track.seed_track}")
            if not repo_path(source, root=root).exists():
                raise FileNotFoundError(f"Missing HH static seed source for {regime.regime_id}/{track.seed_track}: {source}")
            source_sha = sha256_file(source, root=root)
            raw_payload = read_json(source, root=root)
            raw_payload = coerce_hh_seed_payload(raw_payload, source_entry=source_entry, source_path=source)
            runtime_loadability = None
            if require_runtime_dry_load:
                runtime_loadability = dry_load_hh_seed_payload(raw_payload, artifact_json=source)
            seed_payload = normalize_hh_seed_payload(
                raw_payload,
                source_entry=source_entry,
                source_path=source,
                source_sha256=source_sha,
                runtime_loadability=runtime_loadability,
            )
            seed_rel_to_input = _rel_input_seed_path(regime.regime_id, track.seed_track)
            prepared_seeds.append(
                {
                    "regime": regime,
                    "track": track,
                    "source_entry": source_entry,
                    "source": source,
                    "source_sha": source_sha,
                    "seed_payload": seed_payload,
                    "seed_rel_to_input": seed_rel_to_input,
                }
            )

    for prepared in prepared_seeds:
        regime = prepared["regime"]
        track = prepared["track"]
        source_entry = prepared["source_entry"]
        source = prepared["source"]
        source_sha = prepared["source_sha"]
        seed_payload = prepared["seed_payload"]
        seed_rel_to_input = prepared["seed_rel_to_input"]
        seed_path = INPUT_DIR / seed_rel_to_input
        write_json(seed_path, seed_payload, root=root)
        seed_sha = sha256_file(seed_path, root=root)
        export = seed_payload.get("paper_ii_static_seed_export", {})
        export_map = dict(export) if isinstance(export, Mapping) else {}
        summary = _seed_summary(
            regime_id=regime.regime_id,
            seed_track=track.seed_track,
            source_entry=source_entry,
            source_path=source,
            source_sha=source_sha,
            seed_rel_to_input=seed_rel_to_input,
            seed_sha=seed_sha,
            seed_payload=seed_payload,
        )
        seed_ledger.append(summary)
        for drive_label, amplitude in DRIVES:
            lock = seed_lock_for_case(
                regime=regime,
                track=track,
                drive_label=drive_label,
                seed_artifact_json=str(seed_rel_to_input),
                seed_artifact_sha256=seed_sha,
                source_artifact_json=source,
                source_artifact_sha256=source_sha,
                normalized_seed_artifact_sha256=seed_sha,
                normalized_summary=export_map,
            )
            case = _case_payload(
                regime_id=regime.regime_id,
                seed_track=track.seed_track,
                drive_label=drive_label,
                amplitude=amplitude,
                seed_rel_to_input=seed_rel_to_input,
                seed_lock=lock,
            )
            cases.append(case)
            for algorithm_id in FULL_BENCHMARK_ALGORITHMS:
                rec = _benchmark_record(case, algorithm_id, visible=(algorithm_id in VISIBLE_ALGORITHMS))
                records.append(rec)
                if rec["visible_table_method"] == "1":
                    visible_records.append(rec)
            ctl = _controller_record(case)
            records.append(ctl)
            visible_records.append(ctl)
            controller_records.append(ctl)
            if regime.regime_id == "weak_weak" and track.seed_track == "geo" and drive_label == "A0p2":
                smoke_records.extend([ctl])
                smoke_records.extend(
                    _benchmark_record(case, algorithm_id, visible=(algorithm_id in VISIBLE_ALGORITHMS))
                    for algorithm_id in ("dyn_fixed_mclachlan", "dyn_adaptive_pvqd", "dyn_avqds_t")
                )

    manifest = {
        "manifest_id": "paper_ii_hh_seed_tracks_cases_v1",
        "schema": SEED_TRACK_CASE_SCHEMA,
        "generated_utc": "2026-05-28T00:00:00+00:00",
        "case_count": len(cases),
        "seed_source_count": len(seed_ledger),
        "seed_tracks": [track.seed_track for track in SEED_TRACK_SPECS],
        "hh_regimes": [regime.regime_id for regime in regimes],
        "drives": [label for label, _ in DRIVES],
        "visible_algorithms": list(VISIBLE_ALGORITHMS),
        "full_benchmark_algorithms": list(FULL_BENCHMARK_ALGORITHMS),
        "controller_algorithm_record_id": "dyn_controller_full",
        "controller_settings_scope": "coarse_hamiltonian_class",
        "exact_reference_policy": "diagnostic_only_not_controller_input",
        "notes": [
            "Every method within a case shares the same normalized static seed artifact hash.",
            "GeoAdapt, SNAKE, and append-only ADAPT tracks are intentionally separate same-seed surfaces.",
            "GeoAdapt requires static_geo_adapt_vqe; PosGeo/static_pos_geo_adapt_vqe is rejected.",
            "Exact references are diagnostic/reporting-only; controller decisions must remain measurement-compatible.",
        ],
        "cases": cases,
        "seed_ledger_path": str(SEED_LEDGER),
        "source_registry_path": str(source_registry_path),
    }
    write_json(CASE_MANIFEST, manifest, root=root)
    ledger_payload = {
        "schema": SEED_TRACK_LEDGER_SCHEMA,
        "generated_utc": manifest["generated_utc"],
        "seed_source_count": len(seed_ledger),
        "sources": seed_ledger,
    }
    write_json(SEED_LEDGER, ledger_payload, root=root)
    write_json(AUDIT_COMPAT_SEED_LEDGER, _audit_compat_seed_ledger(seed_ledger), root=root)
    _write_tsv(RECORDS_TSV, records, root=root)
    _write_tsv(SMOKE_RECORDS_TSV, smoke_records, root=root)
    _write_text(VISIBLE_IDS, "\n".join(r["record_id"] for r in visible_records) + "\n", root=root)
    _write_text(FULL_IDS, "\n".join(r["record_id"] for r in records) + "\n", root=root)
    _write_text(CONTROLLER_IDS, "\n".join(r["record_id"] for r in controller_records) + "\n", root=root)
    _write_text(SMOKE_IDS, "\n".join(r["record_id"] for r in smoke_records) + "\n", root=root)
    if write_submit_files:
        _write_submit_files(root=root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", default=str(SOURCE_REGISTRY), help="Paper-II HH static seed source registry JSON")
    parser.add_argument("--no-submit-files", action="store_true", help="write manifests/records/queues but skip Condor submit files")
    parser.add_argument(
        "--no-runtime-dry-load",
        action="store_true",
        help="skip heavyweight runtime-loader dry-load during input generation; use local smoke runs for loader validation",
    )
    parser.add_argument(
        "--regime-id",
        action="append",
        default=None,
        help="restrict output to one HH regime id; may be repeated. Defaults to all four regimes.",
    )
    args = parser.parse_args()
    manifest = build_inputs(
        source_registry_path=args.source_registry,
        write_submit_files=not bool(args.no_submit_files),
        require_runtime_dry_load=not bool(args.no_runtime_dry_load),
        regime_ids=args.regime_id,
    )
    print(f"wrote {CASE_MANIFEST} ({manifest['case_count']} cases)")
    print(f"wrote {SEED_LEDGER}")
    print(f"wrote {RECORDS_TSV}")


if __name__ == "__main__":
    main()
