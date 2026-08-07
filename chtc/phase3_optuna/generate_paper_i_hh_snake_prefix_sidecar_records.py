#!/usr/bin/env python3
"""Generate CHTC records for HH SNAKE prefix-sidecar repair replays."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (  # noqa: E402
    FIELDNAMES as BASE_FIELDNAMES,
    configure_batch,
    output_paths,
    rel_or_abs,
    sha256_path,
)


DEFAULT_BATCH_ID = "paper_i_hh_snake_prefix_sidecar_replay_20260622_v1"
QISKIT_REPLAY_JSON = ROOT / "output" / "pdf" / "paper_i_hh_native200_qiskit_table_replay_20260621_v1.json"
NATIVE_DEPTH30_RECORDS = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_native200_depth30_20260619_v1"
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
PROTECT5_STRONG_WEAK_RECORDS = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_native200_snake_protect5_noearly_20260620_v1"
    / "paper_i_hh_native200_snake_protect5_noearly_records.tsv"
)
STRONG_STRONG_RESUME_SCAFFOLD = (
    "raw_outputs/chtc_retrievals/paper_i_u8_hh_strong_strong_snake_current_best/"
    "paper_i_u8_hh_ss_v2_7702629_2_20260614T180758Z/trial_0001_current.json"
)

FEATURE_FIELDNAMES = (
    "hh_feature_ablation_variant",
    "hh_feature_ablation_feature",
    "hh_feature_ablation_submit_group",
    "hh_feature_ablation_note",
    "hh_feature_ablation_overrides_json",
    "hh_feature_ablation_allowed_flags_json",
    "hh_feature_ablation_plateau_source_json",
    "hh_feature_ablation_plateau_source_sha256",
    "hh_feature_ablation_plateau_k",
    "hh_feature_ablation_plateau_abs_delta_e",
    "hh_feature_ablation_plateau_s_alg",
    "hh_feature_ablation_fanout_gate",
)
PREFIX_FIELDNAMES = (
    "prefix_k_pl",
    "prefix_expected_same_cutoff_abs_delta_e",
    "prefix_source_json",
    "prefix_source_sha256",
    "prefix_source_row_tsv",
    "prefix_source_row_record_id",
    "snake_prefix_algorithmic_work_rel",
)
FIELDNAMES = tuple(dict.fromkeys((*BASE_FIELDNAMES, *FEATURE_FIELDNAMES, *PREFIX_FIELDNAMES)))

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {str(k): "" if v is None else str(v) for k, v in row.items()}
            for row in csv.DictReader(fh, delimiter="\t")
        ]


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def qiskit_snake_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain rows")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("method")) != "SNAKE":
            continue
        regime = str(row.get("regime") or "")
        if regime:
            out[regime] = row
    missing = [regime for regime in REGIME_ORDER if regime not in out]
    if missing:
        raise ValueError(f"{path} is missing SNAKE rows for: {', '.join(missing)}")
    return out


def source_row_for_regime(regime: str) -> tuple[dict[str, str], Path]:
    source_tsv = PROTECT5_STRONG_WEAK_RECORDS if regime == "strong-weak" else NATIVE_DEPTH30_RECORDS
    matches = [
        row
        for row in read_records(source_tsv)
        if row.get("method_key") == "snake" and row.get("display_regime") == regime
    ]
    if len(matches) != 1:
        raise ValueError(f"{source_tsv}: expected one SNAKE row for {regime}, got {len(matches)}")
    return dict(matches[0]), source_tsv


def resolve_repo_path(raw: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return ROOT / path


def stage_source_json(*, batch_id: str, regime: str, row: Mapping[str, str]) -> tuple[str, str]:
    raw = str(row.get("source_json") or "").strip()
    if not raw:
        raise ValueError(f"{regime}: source row has no source_json")
    source = resolve_repo_path(raw)
    if not source.exists():
        raise FileNotFoundError(f"{regime}: source_json does not exist: {source}")
    sha = sha256_path(source)
    staged = (
        ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / batch_id
        / "sources"
        / regime.replace("-", "_")
        / "source_result.json"
    )
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    return rel_or_abs(staged), sha


def make_record(*, batch_id: str, regime: str, qrow: Mapping[str, Any]) -> dict[str, str]:
    source, source_tsv = source_row_for_regime(regime)
    staged_source, staged_sha = stage_source_json(batch_id=batch_id, regime=regime, row=source)
    record_id = f"{batch_id}__{regime.replace('-', '_')}__snake__prefix_sidecar"
    row = dict(source)
    row.update(
        {
            "record_id": record_id,
            "batch_id": batch_id,
            "run_class": "diagnostic_snake_prefix_sidecar_replay",
            "runnable": "true",
            "blocker": "",
            "method_key": "snake",
            "method_label": "SNAKE",
            "source_json": staged_source,
            "source_json_sha256": staged_sha,
            "schedule_source_policy": "paper_i_hh_snake_prefix_sidecar_replay_source_locked",
            "schedule_source_regime": regime,
            "schedule_source_method": "SNAKE",
            "schedule_source_json": staged_source,
            "schedule_source_note": (
                "Diagnostic prefix-sidecar replay. Preserve source command/settings; "
                "change output paths and metadata only, then compute canonical plateau-prefix S from result history."
            ),
            "prefix_k_pl": str(qrow.get("k_pl") or ""),
            "prefix_expected_same_cutoff_abs_delta_e": str(qrow.get("same_cutoff_abs_delta_e_at_k_pl") or ""),
            "prefix_source_json": str(qrow.get("source_json") or ""),
            "prefix_source_sha256": str(qrow.get("source_sha256") or ""),
            "prefix_source_row_tsv": rel_or_abs(source_tsv),
            "prefix_source_row_record_id": str(source.get("record_id") or ""),
            "changed_fields_vs_anchor": ",".join(
                sorted(
                    set(
                        filter(
                            None,
                            [
                                *(str(source.get("changed_fields_vs_anchor") or "").split(",")),
                                "output_paths",
                                "source_json_staged_copy",
                                "prefix_sidecar_metadata",
                            ],
                        )
                    )
                )
            ),
            "source_contract_note": (
                str(source.get("source_contract_note") or "")
                + " Prefix replay does not alter Hamiltonian, optimizer, SNAKE selector, pruning, batching, or SPSA settings."
            ).strip(),
        }
    )
    row.update(output_paths(record_id, "snake"))
    record_root = Path(row["record_output_dir"])
    row["snake_prefix_algorithmic_work_rel"] = str(record_root / "snake_prefix_algorithmic_work.json")
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def write_submit_file(*, batch_id: str, submit_path: Path, records_tsv: Path, record_ids: Path) -> None:
    output_root = f"raw_outputs/{batch_id}"
    logs_root = f"logs/{batch_id}"
    transfer_inputs = [
        "pipelines",
        "src",
        "docs",
        "test_support",
        "chtc/phase3_optuna",
        "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json",
        "MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json",
    ]
    resume_path = ROOT / STRONG_STRONG_RESUME_SCAFFOLD
    if resume_path.exists():
        transfer_inputs.append(STRONG_STRONG_RESUME_SCAFFOLD)
    lines = [
        "universe = vanilla",
        "executable = chtc/phase3_optuna/run_paper_i_hh_snake_prefix_sidecar_task_apptainer.sh",
        f"arguments = $(record_id) {rel_or_abs(records_tsv)} {output_root}/$(record_id)",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "transfer_executable = True",
        "preserve_relative_paths = True",
        "transfer_input_files = " + ", ".join(transfer_inputs),
        f"transfer_output_files = {output_root}, {logs_root}",
        "stream_output = False",
        "stream_error = False",
        f"log = logs/{batch_id}.$(Cluster).$(Process).log",
        f"output = logs/{batch_id}.$(Cluster).$(Process).out",
        f"error = logs/{batch_id}.$(Cluster).$(Process).err",
        "requirements = TARGET.HasSIF",
        "request_cpus = 4",
        "request_memory = 49152MB",
        "request_disk = 122880MB",
        "+MaxRuntime = 172800",
        f'+JobBatchName = "holstein-{batch_id.replace("_", "-")}"',
        f"queue record_id from {rel_or_abs(record_ids)}",
    ]
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records(batch_id: str, records: Sequence[dict[str, str]]) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_snake_prefix_sidecar_records.tsv"
    record_ids = input_dir / "paper_i_hh_snake_prefix_sidecar_record_ids.txt"
    manifest_json = input_dir / "paper_i_hh_snake_prefix_sidecar_manifest.json"
    preflight_json = input_dir / "paper_i_hh_snake_prefix_sidecar_preflight.json"
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    write_lines(record_ids, (row["record_id"] for row in records))
    write_submit_file(batch_id=batch_id, submit_path=submit_path, records_tsv=records_tsv, record_ids=record_ids)
    preflight = {
        "schema": "paper_i_hh_snake_prefix_sidecar_preflight_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "row_count": len(records),
        "status": "pass" if len(records) == 6 else "blocked",
        "allowed_row_changes": [
            "record_id",
            "batch_id",
            "run_class",
            "output paths",
            "source_json staged copy path/hash",
            "schedule/prefix metadata",
        ],
        "records": [
            {
                "record_id": row["record_id"],
                "regime": row["display_regime"],
                "source_row_tsv": row["prefix_source_row_tsv"],
                "source_row_record_id": row["prefix_source_row_record_id"],
                "staged_source_json": row["source_json"],
                "staged_source_sha256": row["source_json_sha256"],
                "k_pl": row["prefix_k_pl"],
                "expected_same_cutoff_abs_delta_e": row["prefix_expected_same_cutoff_abs_delta_e"],
                "uses_feature_overrides": bool(row.get("hh_feature_ablation_overrides_json")),
                "feature_variant": row.get("hh_feature_ablation_variant") or None,
            }
            for row in records
        ],
    }
    preflight_json.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "paper_i_hh_snake_prefix_sidecar_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "purpose": "SNAKE-only Paper-I HH prefix S/1-F sidecar repair diagnostic",
        "qiskit_replay_json": rel_or_abs(QISKIT_REPLAY_JSON),
        "records_tsv": rel_or_abs(records_tsv),
        "record_ids": rel_or_abs(record_ids),
        "submit_file": rel_or_abs(submit_path),
        "preflight_json": rel_or_abs(preflight_json),
        "row_count": len(records),
        "records": preflight["records"],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_records(batch_id: str) -> list[dict[str, str]]:
    configure_batch(batch_id)
    rows = qiskit_snake_rows(QISKIT_REPLAY_JSON)
    return [make_record(batch_id=batch_id, regime=regime, qrow=rows[regime]) for regime in REGIME_ORDER]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    records = build_records(batch_id)
    manifest = write_records(batch_id, records)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
