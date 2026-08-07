#!/usr/bin/env python3
"""Offload large Paper-I HH Optuna result JSONs while keeping DB priors local."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_OFFLOAD_INDEX = "paper_i_hh_optuna_result_json_offload_index_v1"
SCHEMA_OFFLOAD_SIDECAR = "paper_i_hh_optuna_result_json_offloaded_sidecar_v1"
SCHEMA_PRIOR_LEDGER = "paper_i_hh_optuna_prior_ledger_v1"
SCHEMA_PRIOR_SUGGESTIONS = "paper_i_hh_optuna_prior_suggestions_v1"

DEFAULT_ARTIFACT_ROOT = Path("raw_outputs/local_hh_optuna_supervisor")
DEFAULT_DB_ROOT = Path("raw_outputs/optuna_studies/local_hh_optuna_supervisor")
DEFAULT_MANIFEST_ROOT = DEFAULT_ARTIFACT_ROOT / "_offload_manifests"
DEFAULT_LEDGER_ROOT = DEFAULT_DB_ROOT / "_prior_ledgers"

ATTR_KEYS = (
    "abs_delta_e",
    "adapt_iteration_count",
    "feasible",
    "graph_count_2q",
    "graph_depth",
    "graph_count_1q",
    "graph_theta_count",
    "paper_i_table_s_alg",
    "paper_i_table_shots_total",
    "paper_i_table_shots_status",
    "graph_hardware_objective_scalar",
    "paper_i_shot_cost_scalar",
    "graph_plus_shot_objective_scalar",
    "invalid_reasons",
    "objective_mode",
    "result_json",
    "case_dir",
)


@dataclass(frozen=True)
class TrialRecord:
    db_path: Path
    study_name: str
    study_slug: str
    regime: str
    trial_id: int
    trial_number: int
    state: str
    objective_value: float | None
    params: dict[str, Any]
    attrs: dict[str, Any]
    result_json: Path | None
    case_dir: Path | None
    sidecar_path: Path | None
    offloaded: bool
    offload_path: str | None
    result_size_bytes: int | None
    keep_reason: str | None = None


@dataclass(frozen=True)
class UnindexedResultJson:
    result_json: Path
    study_slug: str
    regime: str
    cycle_label: str
    trial_number: int | None
    result_size_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def parse_jsonish(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return raw


def maybe_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def table_names(con: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in con.execute("select name from sqlite_master where type='table'")}


def db_trial_records(db_path: Path, repo_root: Path) -> list[TrialRecord]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        names = table_names(con)
        if "trials" not in names or "studies" not in names:
            return []
        trial_values = "trial_values" in names
        rows = con.execute(
            """
            select t.trial_id, t.number, t.state, s.study_name
            from trials t
            join studies s on s.study_id = t.study_id
            order by t.trial_id
            """
        ).fetchall()
        records: list[TrialRecord] = []
        for row in rows:
            trial_id = int(row["trial_id"])
            params = {
                str(param_row["param_name"]): parse_jsonish(param_row["param_value"])
                for param_row in con.execute(
                    "select param_name, param_value from trial_params where trial_id=? order by param_name",
                    (trial_id,),
                )
            }
            attrs = {
                str(attr_row["key"]): parse_jsonish(attr_row["value_json"])
                for attr_row in con.execute(
                    "select key, value_json from trial_user_attributes where trial_id=? order by key",
                    (trial_id,),
                )
            }
            objective_value = None
            if trial_values:
                value_row = con.execute(
                    "select value from trial_values where trial_id=? and objective=0",
                    (trial_id,),
                ).fetchone()
                if value_row is not None:
                    objective_value = maybe_float(value_row["value"])
            result_json = resolve_path(attrs.get("result_json"), repo_root)
            case_dir = resolve_path(attrs.get("case_dir"), repo_root)
            if result_json is None and case_dir is not None:
                result_json = case_dir / "json" / "result.json"
            sidecar_path = None if result_json is None else result_json.with_name("result.offloaded.json")
            offloaded = False
            offload_path = None
            if sidecar_path is not None and sidecar_path.exists():
                try:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    offloaded = True
                    offload_path = str(sidecar.get("offload_path") or "")
                except Exception:
                    offloaded = True
            result_size_bytes = None
            if result_json is not None and result_json.exists():
                try:
                    result_size_bytes = int(result_json.stat().st_size)
                except OSError:
                    result_size_bytes = None
            records.append(
                TrialRecord(
                    db_path=db_path,
                    study_name=str(row["study_name"]),
                    study_slug=db_path.parent.name,
                    regime=db_path.stem,
                    trial_id=trial_id,
                    trial_number=int(row["number"]),
                    state=str(row["state"]),
                    objective_value=objective_value,
                    params=params,
                    attrs={key: attrs.get(key) for key in ATTR_KEYS if key in attrs},
                    result_json=result_json,
                    case_dir=case_dir,
                    sidecar_path=sidecar_path,
                    offloaded=offloaded,
                    offload_path=offload_path,
                    result_size_bytes=result_size_bytes,
                )
            )
        return records
    finally:
        con.close()


def resolve_path(raw: Any, repo_root: Path) -> Path | None:
    if raw in {None, ""}:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def writable_probe(path: Path, *, create: bool = False) -> bool:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            return False
        probe = path / f".write-test-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def choose_offload_root(args: argparse.Namespace) -> tuple[Path | None, str]:
    create = bool(args.execute)
    if args.offload_root:
        root = Path(args.offload_root).expanduser()
        if create and not writable_probe(root, create=True):
            return (None, "explicit_not_writable")
        return (root, "explicit")
    stamp = args.timestamp
    preferred = Path(args.preferred_volume)
    if preferred.is_dir() and os.access(str(preferred), os.W_OK) and writable_probe(preferred):
        root = preferred / "Holstein_optuna_json_offload" / stamp
        if not create or writable_probe(root, create=True):
            return (root, "preferred")
    fallback = Path(args.fallback_volume)
    if fallback.is_dir() and os.access(str(fallback), os.W_OK) and writable_probe(fallback):
        root = fallback / "Holstein_optuna_json_offload" / stamp
        if not create or writable_probe(root, create=True):
            return (root, "fallback")
    return (None, "no_writable_volume")


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_active_process_text() -> str:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid,stat,etime,command"],
            check=False,
            text=True,
            capture_output=True,
        )
    except Exception:
        return ""
    lines = []
    for line in proc.stdout.splitlines():
        if any(token in line for token in ("paper_i_hh_local_optuna_supervisor", "hh_cost_energy_optuna", "adapt_pipeline")):
            lines.append(line)
    return "\n".join(lines)


def active_status_paths(artifact_root: Path) -> set[Path]:
    active: set[Path] = set()
    for status_path in artifact_root.glob("*/supervisor_status.json"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        supervisor_pid = payload.get("supervisor_pid") or payload.get("pid")
        supervisor_live = is_pid_live(supervisor_pid)
        rows = payload.get("active_rows") or payload.get("active") or []
        if not isinstance(rows, Sequence):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_live = supervisor_live or any(
                is_pid_live(row.get(key))
                for key in ("pid", "wrapper_pid", "adapt_child_pid", "child_pid", "process_pid")
            )
            if not row_live:
                continue
            for key in ("case_dir", "trial_dir", "output_dir", "result_json", "current_json", "path"):
                raw = row.get(key)
                if raw:
                    active.add(Path(str(raw)).expanduser())
    return active


def is_pid_live(raw: Any) -> bool:
    if raw in {None, ""}:
        return False
    try:
        pid = int(raw)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def scalar_sort_key(record: TrialRecord, key: str) -> tuple[int, float, int]:
    value = None
    if key == "objective_value":
        value = record.objective_value
    else:
        value = maybe_float(record.attrs.get(key))
    if value is None:
        return (1, float("inf"), record.trial_number)
    return (0, float(value), record.trial_number)


def top_keep_keys(records: Sequence[TrialRecord], keep_top_per_regime: int) -> set[tuple[Path, int]]:
    keep: set[tuple[Path, int]] = set()
    if keep_top_per_regime <= 0:
        return keep
    complete = [record for record in records if record.state == "COMPLETE"]
    by_regime: dict[str, list[TrialRecord]] = {}
    for record in complete:
        by_regime.setdefault(record.regime, []).append(record)
    ranking_keys = (
        "abs_delta_e",
        "objective_value",
        "graph_count_2q",
        "graph_depth",
        "paper_i_table_s_alg",
        "graph_plus_shot_objective_scalar",
    )
    for regime_records in by_regime.values():
        for key in ranking_keys:
            count = 0
            for record in sorted(regime_records, key=lambda item: scalar_sort_key(item, key)):
                value = record.objective_value if key == "objective_value" else record.attrs.get(key)
                if maybe_float(value) is None:
                    continue
                keep.add((record.db_path, record.trial_id))
                count += 1
                if count >= keep_top_per_regime:
                    break
    return keep


def classify_records(
    records: Sequence[TrialRecord],
    *,
    repo_root: Path,
    artifact_root: Path,
    min_size_bytes: int,
    keep_top_per_regime: int,
    keep_paths: Sequence[Path],
    keep_study_slug_substrings: Sequence[str],
    include_failed_result_jsons: bool,
) -> tuple[list[TrialRecord], list[dict[str, Any]]]:
    active_text = load_active_process_text()
    active_paths = active_status_paths(artifact_root)
    keep_keys = top_keep_keys(records, keep_top_per_regime)
    candidates: list[TrialRecord] = []
    decisions: list[dict[str, Any]] = []
    for record in records:
        reason: str | None = None
        path = record.result_json
        if record.state == "RUNNING":
            reason = "trial_state_running"
        elif record.state != "COMPLETE" and not (include_failed_result_jsons and record.state == "FAIL"):
            reason = f"trial_state_{record.state.lower()}"
        elif path is None:
            reason = "missing_result_json_attr"
        elif not path.exists():
            reason = "result_json_missing_or_already_removed"
        elif record.offloaded:
            reason = "already_offloaded"
        elif any(token and token in record.study_slug for token in keep_study_slug_substrings):
            reason = "study_slug_kept"
        elif record.result_size_bytes is None or record.result_size_bytes < min_size_bytes:
            reason = "below_min_size"
        elif (record.db_path, record.trial_id) in keep_keys:
            reason = "top_candidate_kept"
        elif any(path_contains(keep_path, path) or path_contains(path, keep_path) for keep_path in keep_paths):
            reason = "explicit_keep_path"
        elif str(path) in active_text or (record.case_dir is not None and str(record.case_dir) in active_text):
            reason = "active_process_path"
        elif any(path_contains(active, path) or path_contains(path, active) for active in active_paths):
            reason = "active_supervisor_status_path"
        if reason is None:
            candidates.append(record)
            reason = "candidate"
        decisions.append(
            {
                "db_path": repo_rel(record.db_path, repo_root),
                "study_name": record.study_name,
                "study_slug": record.study_slug,
                "regime": record.regime,
                "trial_id": record.trial_id,
                "trial_number": record.trial_number,
                "state": record.state,
                "result_json": None if path is None else repo_rel(path, repo_root),
                "size_bytes": record.result_size_bytes,
                "decision": reason,
            }
        )
    candidates.sort(key=lambda record: int(record.result_size_bytes or 0), reverse=True)
    return candidates, decisions


def ledger_row(record: TrialRecord, repo_root: Path, decisions_by_key: Mapping[tuple[Path, int], str]) -> dict[str, Any]:
    decision = decisions_by_key.get((record.db_path, record.trial_id))
    return {
        "schema": SCHEMA_PRIOR_LEDGER,
        "generated_utc": utc_now(),
        "db_path": repo_rel(record.db_path, repo_root),
        "study_name": record.study_name,
        "study_slug": record.study_slug,
        "regime": record.regime,
        "trial_id": record.trial_id,
        "trial_number": record.trial_number,
        "state": record.state,
        "objective_value": record.objective_value,
        "params": record.params,
        "attrs": record.attrs,
        "result_json": None if record.result_json is None else repo_rel(record.result_json, repo_root),
        "case_dir": None if record.case_dir is None else repo_rel(record.case_dir, repo_root),
        "result_size_bytes": record.result_size_bytes,
        "offloaded": bool(record.offloaded),
        "offload_path": record.offload_path,
        "offload_decision": decision,
    }


def write_prior_ledger(records: Sequence[TrialRecord], decisions: Sequence[Mapping[str, Any]], ledger_path: Path, repo_root: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_by_key: dict[tuple[Path, int], str] = {}
    for decision in decisions:
        db_path = resolve_path(decision.get("db_path"), repo_root)
        if db_path is None:
            continue
        decisions_by_key[(db_path, int(decision["trial_id"]))] = str(decision["decision"])
    with ledger_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(ledger_row(record, repo_root, decisions_by_key), sort_keys=True) + "\n")


def compact_trial_ref(record: TrialRecord, repo_root: Path, basis: str) -> dict[str, Any]:
    return {
        "basis": basis,
        "db_path": repo_rel(record.db_path, repo_root),
        "study_name": record.study_name,
        "study_slug": record.study_slug,
        "regime": record.regime,
        "trial_id": record.trial_id,
        "trial_number": record.trial_number,
        "objective_value": record.objective_value,
        "abs_delta_e": maybe_float(record.attrs.get("abs_delta_e")),
        "graph_count_2q": maybe_float(record.attrs.get("graph_count_2q")),
        "graph_depth": maybe_float(record.attrs.get("graph_depth")),
        "paper_i_table_s_alg": maybe_float(record.attrs.get("paper_i_table_s_alg")),
        "graph_plus_shot_objective_scalar": maybe_float(record.attrs.get("graph_plus_shot_objective_scalar")),
        "params": record.params,
        "result_json": None if record.result_json is None else repo_rel(record.result_json, repo_root),
        "offloaded": bool(record.offloaded),
        "offload_path": record.offload_path,
    }


def top_records_for_key(records: Sequence[TrialRecord], key: str, limit: int) -> list[TrialRecord]:
    rows: list[TrialRecord] = []
    for record in records:
        if record.state != "COMPLETE" or not record.params:
            continue
        value = record.objective_value if key == "objective_value" else record.attrs.get(key)
        if maybe_float(value) is None:
            continue
        rows.append(record)
    return sorted(rows, key=lambda item: scalar_sort_key(item, key))[:limit]


def parameter_bands(records: Sequence[TrialRecord]) -> dict[str, dict[str, Any]]:
    values: dict[str, list[float]] = {}
    for record in records:
        for key, raw in record.params.items():
            value = maybe_float(raw)
            if value is None:
                continue
            values.setdefault(key, []).append(float(value))
    return {
        key: {
            "min": min(vals),
            "max": max(vals),
            "count": len(vals),
            "unique_values": sorted(set(vals))[:25],
        }
        for key, vals in sorted(values.items())
        if vals
    }


def write_prior_suggestions(
    records: Sequence[TrialRecord],
    suggestions_path: Path,
    repo_root: Path,
    *,
    source_ledger: Path | None,
    per_basis_limit: int,
) -> None:
    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    complete = [record for record in records if record.state == "COMPLETE"]
    by_regime: dict[str, list[TrialRecord]] = {}
    for record in complete:
        by_regime.setdefault(record.regime, []).append(record)
    bases = {
        "energy_dominant": "abs_delta_e",
        "objective_dominant": "objective_value",
        "graph_count_dominant": "graph_count_2q",
        "graph_depth_dominant": "graph_depth",
        "shot_salg_dominant": "paper_i_table_s_alg",
        "graph_plus_shot_dominant": "graph_plus_shot_objective_scalar",
    }
    regimes: dict[str, Any] = {}
    for regime, regime_records in sorted(by_regime.items()):
        selected_union: dict[tuple[Path, int], TrialRecord] = {}
        basis_payload: dict[str, list[dict[str, Any]]] = {}
        for basis, key in bases.items():
            top = top_records_for_key(regime_records, key, per_basis_limit)
            basis_payload[basis] = [compact_trial_ref(record, repo_root, basis) for record in top]
            for record in top:
                selected_union[(record.db_path, record.trial_id)] = record
        regimes[regime] = {
            "enqueue_trials": basis_payload,
            "parameter_bands_from_selected": parameter_bands(list(selected_union.values())),
        }
    payload = {
        "schema": SCHEMA_PRIOR_SUGGESTIONS,
        "generated_utc": utc_now(),
        "source_ledger": None if source_ledger is None else repo_rel(source_ledger, repo_root),
        "per_basis_limit": int(per_basis_limit),
        "regimes": regimes,
    }
    suggestions_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_streaming(source: Path, dest_tmp: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    with source.open("rb") as src, dest_tmp.open("wb") as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)


def copy_and_remove(record: TrialRecord, offload_root: Path, repo_root: Path) -> dict[str, Any]:
    if record.result_json is None or not record.result_json.exists():
        raise FileNotFoundError(f"missing result_json for trial {record.trial_number}: {record.result_json}")
    source = record.result_json
    source_size = source.stat().st_size
    source_sha = sha256_path(source)
    rel_source = repo_rel(source, repo_root)
    dest = offload_root / record.study_slug / record.regime / f"trial_{record.trial_number:04d}" / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"offload destination already exists: {dest}")
    dest_tmp = dest.with_name(f"{dest.name}.copying-{os.getpid()}.partial")
    if dest_tmp.exists():
        dest_tmp.unlink()
    copy_streaming(source, dest_tmp)
    try:
        shutil.copystat(source, dest_tmp)
    except OSError:
        pass
    dest_size = dest_tmp.stat().st_size
    dest_sha = sha256_path(dest_tmp)
    if dest_size != source_size or dest_sha != source_sha:
        try:
            dest_tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(f"checksum/size mismatch for {source}")
    dest_tmp.rename(dest)
    sidecar = {
        "schema": SCHEMA_OFFLOAD_SIDECAR,
        "generated_utc": utc_now(),
        "source_path": rel_source,
        "source_abs_path": str(source),
        "offload_path": str(dest),
        "offload_size_bytes": int(dest_size),
        "sha256": dest_sha,
        "db_path": repo_rel(record.db_path, repo_root),
        "study_name": record.study_name,
        "study_slug": record.study_slug,
        "regime": record.regime,
        "trial_id": record.trial_id,
        "trial_number": record.trial_number,
        "case_dir": None if record.case_dir is None else repo_rel(record.case_dir, repo_root),
    }
    sidecar_path = source.with_name("result.offloaded.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source.unlink()
    return sidecar


def infer_unindexed_result_json(path: Path, artifact_root: Path) -> UnindexedResultJson:
    try:
        rel_parts = path.resolve().relative_to(artifact_root.resolve()).parts
    except Exception:
        rel_parts = path.parts
    study_slug = rel_parts[0] if rel_parts else "unknown_study"
    cycle_label = "unknown_cycle"
    regime = "unknown_regime"
    trial_number: int | None = None
    for idx, part in enumerate(rel_parts):
        if part.startswith("cycle_"):
            cycle_label = part
            if idx + 1 < len(rel_parts):
                regime = rel_parts[idx + 1]
        if part.startswith("trial_"):
            try:
                trial_number = int(part.removeprefix("trial_"))
            except ValueError:
                trial_number = None
    return UnindexedResultJson(
        result_json=path,
        study_slug=study_slug,
        regime=regime,
        cycle_label=cycle_label,
        trial_number=trial_number,
        result_size_bytes=int(path.stat().st_size),
    )


def discover_unindexed_result_jsons(
    *,
    records: Sequence[TrialRecord],
    repo_root: Path,
    artifact_root: Path,
    min_size_bytes: int,
    keep_paths: Sequence[Path],
) -> tuple[list[UnindexedResultJson], list[dict[str, Any]]]:
    active_text = load_active_process_text()
    active_paths = active_status_paths(artifact_root)
    indexed_paths: set[Path] = set()
    for record in records:
        if record.result_json is None:
            continue
        try:
            indexed_paths.add(record.result_json.resolve())
        except OSError:
            indexed_paths.add(record.result_json)
    candidates: list[UnindexedResultJson] = []
    decisions: list[dict[str, Any]] = []
    for path in sorted(artifact_root.rglob("result.json")):
        reason: str | None = None
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in indexed_paths:
            reason = "db_indexed"
        elif path.with_name("result.offloaded.json").exists():
            reason = "already_offloaded"
        else:
            try:
                size = int(path.stat().st_size)
            except OSError:
                size = 0
            if size < min_size_bytes:
                reason = "below_min_size"
            elif any(path_contains(keep_path, path) or path_contains(path, keep_path) for keep_path in keep_paths):
                reason = "explicit_keep_path"
            elif str(path) in active_text:
                reason = "active_process_path"
            elif any(path_contains(active, path) or path_contains(path, active) for active in active_paths):
                reason = "active_supervisor_status_path"
        inferred = infer_unindexed_result_json(path, artifact_root)
        if reason is None:
            candidates.append(inferred)
            reason = "unindexed_candidate"
        decisions.append(
            {
                "decision": reason,
                "result_json": repo_rel(path, repo_root),
                "size_bytes": inferred.result_size_bytes,
                "study_slug": inferred.study_slug,
                "regime": inferred.regime,
                "cycle_label": inferred.cycle_label,
                "trial_number": inferred.trial_number,
            }
        )
    candidates.sort(key=lambda item: int(item.result_size_bytes), reverse=True)
    return candidates, decisions


def copy_unindexed_and_remove(record: UnindexedResultJson, offload_root: Path, repo_root: Path) -> dict[str, Any]:
    source = record.result_json
    if not source.exists():
        raise FileNotFoundError(f"missing unindexed result_json: {source}")
    source_size = source.stat().st_size
    source_sha = sha256_path(source)
    trial_label = "trial_unknown" if record.trial_number is None else f"trial_{record.trial_number:04d}"
    dest = offload_root / record.study_slug / record.regime / f"{record.cycle_label}_{trial_label}" / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"offload destination already exists: {dest}")
    dest_tmp = dest.with_name(f"{dest.name}.copying-{os.getpid()}.partial")
    if dest_tmp.exists():
        dest_tmp.unlink()
    copy_streaming(source, dest_tmp)
    try:
        shutil.copystat(source, dest_tmp)
    except OSError:
        pass
    dest_size = dest_tmp.stat().st_size
    dest_sha = sha256_path(dest_tmp)
    if dest_size != source_size or dest_sha != source_sha:
        try:
            dest_tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(f"checksum/size mismatch for {source}")
    dest_tmp.rename(dest)
    sidecar = {
        "schema": SCHEMA_OFFLOAD_SIDECAR,
        "generated_utc": utc_now(),
        "offload_kind": "unindexed_result_json",
        "source_path": repo_rel(source, repo_root),
        "source_abs_path": str(source),
        "offload_path": str(dest),
        "offload_size_bytes": int(dest_size),
        "sha256": dest_sha,
        "db_path": None,
        "study_name": None,
        "study_slug": record.study_slug,
        "regime": record.regime,
        "trial_id": None,
        "trial_number": record.trial_number,
        "cycle_label": record.cycle_label,
        "case_dir": repo_rel(source.parents[1], repo_root) if len(source.parents) > 1 else None,
    }
    source.with_name("result.offloaded.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source.unlink()
    return sidecar


def write_index(
    *,
    index_path: Path,
    repo_root: Path,
    offload_root: Path | None,
    volume_choice: str,
    records: Sequence[TrialRecord],
    candidates: Sequence[TrialRecord],
    decisions: Sequence[Mapping[str, Any]],
    unindexed_candidates: Sequence[UnindexedResultJson],
    unindexed_decisions: Sequence[Mapping[str, Any]],
    copied: Sequence[Mapping[str, Any]],
    execute: bool,
    limit: int | None,
    ledger_path: Path | None,
    suggestions_path: Path | None,
    error: Mapping[str, Any] | None = None,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    db_candidate_bytes = sum(int(record.result_size_bytes or 0) for record in candidates)
    unindexed_candidate_bytes = sum(int(record.result_size_bytes or 0) for record in unindexed_candidates)
    total_candidate_bytes = db_candidate_bytes + unindexed_candidate_bytes
    payload = {
        "schema": SCHEMA_OFFLOAD_INDEX,
        "generated_utc": utc_now(),
        "repo_root": str(repo_root),
        "execute": bool(execute),
        "limit": limit,
        "offload_root": None if offload_root is None else str(offload_root),
        "volume_choice": volume_choice,
        "ledger_path": None if ledger_path is None else repo_rel(ledger_path, repo_root),
        "suggestions_path": None if suggestions_path is None else repo_rel(suggestions_path, repo_root),
        "records_seen": len(records),
        "db_candidate_count": len(candidates),
        "db_candidate_bytes": db_candidate_bytes,
        "unindexed_candidate_count": len(unindexed_candidates),
        "unindexed_candidate_bytes": unindexed_candidate_bytes,
        "candidate_count": len(candidates) + len(unindexed_candidates),
        "candidate_bytes": total_candidate_bytes,
        "copied_count": len(copied),
        "copied_bytes": sum(int(item.get("offload_size_bytes") or 0) for item in copied),
        "error": None if error is None else dict(error),
        "copied": list(copied),
        "decisions": list(decisions),
        "unindexed_decisions": list(unindexed_decisions),
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def existing_keep_paths(repo_root: Path) -> list[Path]:
    raw_paths = [
        "raw_outputs/chtc_retrievals/paper_i_u8_hh_strong_strong_snake_current_best/paper_i_u8_hh_ss_v2_7702629_2_20260614T180758Z/trial_0001_current.json",
    ]
    paths: list[Path] = []
    for raw in raw_paths:
        path = repo_root / raw
        if path.exists():
            paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT)
    parser.add_argument("--preferred-volume", default="/Volumes/HolsteinOffload")
    parser.add_argument("--fallback-volume", default="/Volumes/Memorex USB")
    parser.add_argument("--offload-root", default=None)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--min-size-mib", type=float, default=16.0)
    parser.add_argument("--keep-top-per-regime", type=int, default=6)
    parser.add_argument("--keep-study-slug-substring", action="append", default=None)
    parser.add_argument(
        "--include-unindexed-result-jsons",
        action="store_true",
        help="Also offload large artifact-root json/result.json files that are not referenced by Optuna DB attrs.",
    )
    parser.add_argument(
        "--include-failed-result-jsons",
        action="store_true",
        help="Also offload large DB-linked result JSONs from failed trials. RUNNING trials are still excluded.",
    )
    parser.add_argument("--prior-suggestion-limit", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-prior-ledger", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="Run repeated offload passes until interrupted.")
    parser.add_argument("--poll-interval-s", type=float, default=600.0)
    parser.add_argument("--monitor-max-passes", type=int, default=None)
    parser.add_argument("--monitor-min-free-gb", type=float, default=20.0)
    parser.add_argument("--monitor-execute-always", action="store_true")
    return parser


def run_once(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    repo_root = args.repo_root.expanduser().resolve()
    artifact_root = (repo_root / args.artifact_root).resolve() if not args.artifact_root.is_absolute() else args.artifact_root.resolve()
    db_root = (repo_root / args.db_root).resolve() if not args.db_root.is_absolute() else args.db_root.resolve()
    manifest_root = (repo_root / args.manifest_root).resolve() if not args.manifest_root.is_absolute() else args.manifest_root.resolve()
    ledger_root = (repo_root / args.ledger_root).resolve() if not args.ledger_root.is_absolute() else args.ledger_root.resolve()
    if not args.dry_run and not args.write_prior_ledger and not args.execute:
        args.dry_run = True
        args.write_prior_ledger = True
    db_paths = sorted(db_root.rglob("*.sqlite3"))
    records: list[TrialRecord] = []
    for db_path in db_paths:
        records.extend(db_trial_records(db_path, repo_root))
    min_size_bytes = int(float(args.min_size_mib) * 1024 * 1024)
    keep_paths = existing_keep_paths(repo_root)
    candidates, decisions = classify_records(
        records,
        repo_root=repo_root,
        artifact_root=artifact_root,
        min_size_bytes=min_size_bytes,
        keep_top_per_regime=int(args.keep_top_per_regime),
        keep_paths=keep_paths,
        keep_study_slug_substrings=list(args.keep_study_slug_substring or []),
        include_failed_result_jsons=bool(getattr(args, "include_failed_result_jsons", False)),
    )
    unindexed_candidates: list[UnindexedResultJson] = []
    unindexed_decisions: list[dict[str, Any]] = []
    if bool(getattr(args, "include_unindexed_result_jsons", False)):
        unindexed_candidates, unindexed_decisions = discover_unindexed_result_jsons(
            records=records,
            repo_root=repo_root,
            artifact_root=artifact_root,
            min_size_bytes=min_size_bytes,
            keep_paths=keep_paths,
        )
    offload_root, volume_choice = choose_offload_root(args)
    if args.execute and offload_root is None:
        summary = {
            "schema": SCHEMA_OFFLOAD_INDEX,
            "error": {"type": "NoWritableVolume", "message": "no writable offload root available"},
            "offload_root": None,
            "volume_choice": volume_choice,
        }
        print("ERROR: no writable offload root available", file=sys.stderr, flush=True)
        return 2, summary
    ledger_path: Path | None = None
    suggestions_path: Path | None = None
    if args.write_prior_ledger:
        ledger_path = ledger_root / f"hh_optuna_prior_ledger_{args.timestamp}.jsonl"
        write_prior_ledger(records, decisions, ledger_path, repo_root)
        suggestions_path = ledger_root / f"hh_optuna_prior_suggestions_{args.timestamp}.json"
        write_prior_suggestions(
            records,
            suggestions_path,
            repo_root,
            source_ledger=ledger_path,
            per_basis_limit=int(args.prior_suggestion_limit),
        )
    selected = candidates[: args.limit] if args.limit is not None else candidates
    remaining_limit = None if args.limit is None else max(0, int(args.limit) - len(selected))
    selected_unindexed = (
        unindexed_candidates[:remaining_limit] if remaining_limit is not None else unindexed_candidates
    )
    copied: list[Mapping[str, Any]] = []
    error: Mapping[str, Any] | None = None
    if args.execute:
        assert offload_root is not None
        free = disk_free_bytes(offload_root)
        need = sum(int(record.result_size_bytes or 0) for record in selected) + sum(
            int(record.result_size_bytes or 0) for record in selected_unindexed
        )
        if need > free:
            print(f"ERROR: selected offload needs {need} bytes but target has {free} bytes free", file=sys.stderr, flush=True)
            return 3, {
                "schema": SCHEMA_OFFLOAD_INDEX,
                "error": {"type": "InsufficientTargetSpace", "message": f"selected offload needs {need} bytes but target has {free} bytes free"},
                "candidate_count": len(candidates) + len(unindexed_candidates),
                "candidate_bytes": sum(int(record.result_size_bytes or 0) for record in candidates)
                + sum(int(record.result_size_bytes or 0) for record in unindexed_candidates),
                "selected_count": len(selected) + len(selected_unindexed),
                "offload_root": str(offload_root),
                "volume_choice": volume_choice,
            }
        try:
            for record in selected:
                copied.append(copy_and_remove(record, offload_root, repo_root))
            for record in selected_unindexed:
                copied.append(copy_unindexed_and_remove(record, offload_root, repo_root))
        except KeyboardInterrupt:
            error = {"type": "KeyboardInterrupt", "message": "offload interrupted by user"}
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    index_path = manifest_root / f"offload_index_{args.timestamp}.json"
    write_index(
        index_path=index_path,
        repo_root=repo_root,
        offload_root=offload_root,
        volume_choice=volume_choice,
        records=records,
        candidates=candidates,
        decisions=decisions,
        unindexed_candidates=unindexed_candidates,
        unindexed_decisions=unindexed_decisions,
        copied=copied,
        execute=bool(args.execute),
        limit=args.limit,
        ledger_path=ledger_path,
        suggestions_path=suggestions_path,
        error=error,
    )
    summary = {
        "schema": SCHEMA_OFFLOAD_INDEX,
        "index_path": repo_rel(index_path, repo_root),
        "ledger_path": None if ledger_path is None else repo_rel(ledger_path, repo_root),
        "suggestions_path": None if suggestions_path is None else repo_rel(suggestions_path, repo_root),
        "records_seen": len(records),
        "db_candidate_count": len(candidates),
        "db_candidate_bytes": sum(int(record.result_size_bytes or 0) for record in candidates),
        "unindexed_candidate_count": len(unindexed_candidates),
        "unindexed_candidate_bytes": sum(int(record.result_size_bytes or 0) for record in unindexed_candidates),
        "candidate_count": len(candidates) + len(unindexed_candidates),
        "candidate_bytes": sum(int(record.result_size_bytes or 0) for record in candidates)
        + sum(int(record.result_size_bytes or 0) for record in unindexed_candidates),
        "selected_count": len(selected) + len(selected_unindexed),
        "copied_count": len(copied),
        "copied_bytes": sum(int(item.get("offload_size_bytes") or 0) for item in copied),
        "error": error,
        "offload_root": None if offload_root is None else str(offload_root),
        "volume_choice": volume_choice,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if error is not None:
        return (130 if error.get("type") == "KeyboardInterrupt" else 4), summary
    return 0, summary


def next_monitor_timestamp(base: str, pass_index: int) -> str:
    return f"{base}_pass{pass_index:04d}_{datetime.now().strftime('%H%M%S')}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not args.monitor:
        code, _summary = run_once(args)
        return code
    base_timestamp = args.timestamp
    pass_index = 0
    try:
        while args.monitor_max_passes is None or pass_index < int(args.monitor_max_passes):
            pass_index += 1
            loop_args = argparse.Namespace(**vars(args))
            loop_args.timestamp = next_monitor_timestamp(base_timestamp, pass_index)
            repo_root = loop_args.repo_root.expanduser().resolve()
            internal_free_gb = shutil.disk_usage(repo_root).free / (1024.0**3)
            loop_args.write_prior_ledger = True
            loop_args.dry_run = False
            loop_args.execute = bool(args.monitor_execute_always or internal_free_gb < float(args.monitor_min_free_gb))
            if not loop_args.execute:
                loop_args.dry_run = True
            code, summary = run_once(loop_args)
            if code not in {0, 130}:
                print(f"monitor pass {pass_index} exited with code {code}", file=sys.stderr, flush=True)
            print(
                f"monitor pass {pass_index}: free_gb={internal_free_gb:.2f} "
                f"execute={loop_args.execute} candidates={summary.get('candidate_count')} "
                f"copied={summary.get('copied_count')}",
                flush=True,
            )
            time.sleep(float(args.poll_interval_s))
    except KeyboardInterrupt:
        print("monitor interrupted", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
