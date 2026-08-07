#!/usr/bin/env python3
"""Restartable local supervisor for Paper-I HH SNAKE Optuna cycles.

Runs two Hubbard-Holstein regimes at a time through the canonical
``paper_i_hh_speed_optuna`` launcher, persists each regime's Optuna study in
SQLite, and repeats the pair cycle until stopped. This script is intentionally
model-free: RepoPrompt/Codex can inspect and steer it, but the run state lives in
SQLite, JSON manifests, and logs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_geo_targets_20260615.json"
DEFAULT_GEO_GRAPH_PROXY_TARGET_MANIFEST = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_geo_graph_proxy_targets_20260617.json"
DEFAULT_GRAPH_PROXY_TARGET_MANIFEST = REPO_ROOT / "output/pdf/paper_i_hh_snake_graph_proxy_targets_from_overlay_review_20260616.json"
DEFAULT_CANDIDATE_PRIOR_MANIFEST = REPO_ROOT / "output/pdf/paper_i_hh_shot_focus_candidate_priors_from_overlay_20260616.json"
DEFAULT_SHOT_FOCUSED_REGIMES = "weak-weak,weak-strong,intermediate-strong,strong-weak-u8,strong-strong-u8"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "raw_outputs/local_hh_optuna_supervisor"
DEFAULT_STORAGE_ROOT = REPO_ROOT / "raw_outputs/optuna_studies/local_hh_optuna_supervisor"
PIPELINE = "paper_i_hh_local_optuna_supervisor_v1"


@dataclass(frozen=True)
class RegimeLaunch:
    regime: str
    cycle: int
    pair_index: int
    output_dir: str
    storage_path: str
    command: tuple[str, ...]
    stdout_log: str
    stderr_log: str
    command_sh: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(keep).strip("_")
    return slug or "unnamed"


def _default_slug(args: argparse.Namespace) -> str:
    stem = "paper_i_hh_local_geo_energy_shotcost"
    if str(getattr(args, "objective_mode", "")) == "shot_then_energy_graph_cost":
        stem = "paper_i_hh_local_shotfirst_energy_graph"
    elif str(getattr(args, "objective_mode", "")) == "geo_energy_gate_then_shot_energy_graph_cost":
        stem = "paper_i_hh_local_egate_shotfirst_energy_graph"
    elif str(getattr(args, "objective_mode", "")) == "geo_energy_then_graph_shot_cost":
        stem = "paper_i_hh_local_geo_energy_graphcost"
    if str(getattr(args, "speed_surface_profile", "")) == "shortlist_refine":
        stem = f"{stem}_shortlist_refine"
    return f"{stem}_{_timestamp_slug()}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _load_adapt_resume_scaffold_map(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional per-regime structural ADAPT resume settings."""
    if path is None:
        return {}
    payload = _load_json(Path(path))
    raw_map = payload.get("regimes", payload)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--adapt-resume-scaffold-map-json must be a JSON object or contain a 'regimes' object.")
    out: dict[str, dict[str, Any]] = {}
    for regime, raw_entry in raw_map.items():
        if isinstance(raw_entry, str):
            entry: dict[str, Any] = {"adapt_resume_scaffold_json": raw_entry}
        elif isinstance(raw_entry, Mapping):
            entry = dict(raw_entry)
        else:
            raise ValueError(f"Invalid scaffold-map entry for regime {regime!r}: expected string or object.")
        if entry.get("enabled", True) is False:
            continue
        scaffold = entry.get("adapt_resume_scaffold_json", entry.get("scaffold_json"))
        if scaffold in {None, ""}:
            raise ValueError(f"Scaffold-map entry for regime {regime!r} is missing adapt_resume_scaffold_json.")
        scaffold_path = Path(str(scaffold))
        if not scaffold_path.is_absolute():
            scaffold_path = REPO_ROOT / scaffold_path
        if not scaffold_path.exists():
            raise FileNotFoundError(f"Scaffold-map entry for regime {regime!r} does not exist: {scaffold_path}")
        entry["adapt_resume_scaffold_json"] = str(scaffold_path)
        out[str(regime)] = entry
    return out


def _load_regime_launch_settings_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(Path(path))
    raw_map = payload.get("regimes", payload)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--regime-launch-settings-json must be a JSON object or contain a 'regimes' object.")
    out: dict[str, dict[str, Any]] = {}
    for regime, raw_entry in raw_map.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Invalid launch-settings entry for regime {regime!r}: expected object.")
        out[str(regime)] = dict(raw_entry)
    return out


def _load_regime_storage_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = _load_json(Path(path))
    raw_map = payload.get("regimes", payload)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--regime-storage-map-json must be a JSON object or contain a 'regimes' object.")
    out: dict[str, Path] = {}
    for regime, raw_entry in raw_map.items():
        if isinstance(raw_entry, str):
            storage = raw_entry
        elif isinstance(raw_entry, Mapping):
            storage = raw_entry.get("storage_path") or raw_entry.get("sqlite")
        else:
            raise ValueError(f"Invalid storage-map entry for regime {regime!r}: expected string or object.")
        if storage in {None, ""}:
            raise ValueError(f"Storage-map entry for regime {regime!r} is missing storage_path.")
        storage_path = Path(str(storage))
        if not storage_path.is_absolute():
            storage_path = REPO_ROOT / storage_path
        out[str(regime)] = storage_path
    return out


def _load_regime_study_prefix_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _load_json(Path(path))
    raw_map = payload.get("regimes", payload)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--regime-study-prefix-map-json must be a JSON object or contain a 'regimes' object.")
    out: dict[str, str] = {}
    for regime, raw_entry in raw_map.items():
        if isinstance(raw_entry, str):
            prefix = raw_entry
        elif isinstance(raw_entry, Mapping):
            prefix = raw_entry.get("study_name_prefix") or raw_entry.get("study_prefix")
        else:
            raise ValueError(f"Invalid study-prefix entry for regime {regime!r}: expected string or object.")
        if prefix in {None, ""}:
            raise ValueError(f"Study-prefix entry for regime {regime!r} is missing study_name_prefix.")
        out[str(regime)] = str(prefix)
    return out


def _load_graph_proxy_target_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = REPO_ROOT / target_path
    if not target_path.exists():
        return {}
    payload = _load_json(target_path)
    raw_map = payload.get("regimes", payload)
    if not isinstance(raw_map, Mapping):
        raise ValueError("--graph-proxy-target-manifest must be a JSON object or contain a 'regimes' object.")
    out: dict[str, dict[str, Any]] = {}
    for regime, raw_entry in raw_map.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Invalid graph-proxy target entry for regime {regime!r}: expected object.")
        entry = dict(raw_entry)
        if str(entry.get("cost_surface", entry.get("surface", ""))) not in {"graph_proxy", "marrakesh_graph_span_v1"}:
            continue
        n2q = None
        d2q = None
        for key in (
            "graph_proxy_N2q",
            "snake_graph_proxy_N2q",
            "geo_graph_proxy_N2q",
            "geo_graph_count_2q",
            "graph_count_2q",
            "N2Q_proxy",
            "geo_N2q",
        ):
            if entry.get(key) is not None:
                n2q = float(entry[key])
                break
        for key in (
            "graph_proxy_D2q",
            "snake_graph_proxy_D2q",
            "geo_graph_proxy_D2q",
            "geo_graph_depth",
            "graph_depth",
            "D2Q_proxy",
            "geo_D2q",
        ):
            if entry.get(key) is not None:
                d2q = float(entry[key])
                break
        if n2q is not None and d2q is not None:
            entry["geo_graph_proxy_N2q"] = float(n2q)
            entry["geo_graph_proxy_D2q"] = float(d2q)
            entry["graph_proxy_N2q"] = float(n2q)
            entry["graph_proxy_D2q"] = float(d2q)
            out[str(regime)] = entry
    return out


def _launch_settings_for_regime(args: argparse.Namespace, regime: str) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "trials_per_chunk": args.trials_per_chunk,
        "n_startup_trials": args.n_startup_trials,
        "objective_mode": args.objective_mode,
        "speed_surface_profile": args.speed_surface_profile,
        "max_depth": args.max_depth,
        "maxiter": args.maxiter,
        "final_refit_maxiter": args.final_refit_maxiter,
        "gradient_workers": args.gradient_workers,
        "beam_parent_workers": args.beam_parent_workers,
        "spsa_parallel_evaluations": args.spsa_parallel_evaluations,
        "spsa_profile": args.spsa_profile,
        "phase2_w_shot_profile_space": args.phase2_w_shot_profile_space,
        "enable_hva_generators": args.enable_hva_generators,
        "hva_aggressive_screening": args.hva_aggressive_screening,
    }
    launch_map = getattr(args, "_regime_launch_settings_map", {}) or {}
    override = launch_map.get(str(regime))
    if isinstance(override, Mapping):
        for key in settings:
            if key in override:
                settings[key] = override[key]
    if bool(settings.get("hva_aggressive_screening")) and not bool(settings.get("enable_hva_generators")):
        raise ValueError(
            f"Regime {regime!r} requested hva_aggressive_screening without enable_hva_generators."
        )
    return settings


def _storage_path_for_regime(args: argparse.Namespace, slug: str, regime: str) -> Path:
    storage_map = getattr(args, "_regime_storage_map", {}) or {}
    mapped = storage_map.get(str(regime))
    if mapped is not None:
        return Path(mapped)
    return Path(args.storage_root) / slug / f"{regime}.sqlite3"


def _study_prefix_for_regime(args: argparse.Namespace, slug: str, regime: str, study_suffix: str) -> str:
    study_prefix_map = getattr(args, "_regime_study_prefix_map", {}) or {}
    mapped = study_prefix_map.get(str(regime))
    if mapped not in {None, ""}:
        return str(mapped)
    return f"{slug}_{regime.replace('-', '_')}_{study_suffix}"


def _resume_settings_for_regime(args: argparse.Namespace, regime: str) -> dict[str, Any]:
    """Resolve global structural resume flags plus optional per-regime overrides."""
    settings: dict[str, Any] = {
        "adapt_resume_scaffold_json": None if args.adapt_resume_scaffold_json is None else str(Path(args.adapt_resume_scaffold_json)),
        "adapt_resume_mode": args.adapt_resume_mode,
        "adapt_segment_id": args.adapt_segment_id,
        "adapt_segment_target_depth": args.adapt_segment_target_depth,
        "adapt_segment_max_new_admissions": args.adapt_segment_max_new_admissions,
        "adapt_segment_wallclock_cap_s": args.adapt_segment_wallclock_cap_s,
        "adapt_resume_compile_smoke": args.adapt_resume_compile_smoke,
        "adapt_resume_smoke_backend": args.adapt_resume_smoke_backend,
    }
    scaffold_map = getattr(args, "_adapt_resume_scaffold_map", {}) or {}
    override = scaffold_map.get(str(regime))
    if override:
        alias_pairs = {
            "scaffold_json": "adapt_resume_scaffold_json",
            "resume_mode": "adapt_resume_mode",
            "segment_id": "adapt_segment_id",
            "segment_target_depth": "adapt_segment_target_depth",
            "segment_max_new_admissions": "adapt_segment_max_new_admissions",
            "segment_wallclock_cap_s": "adapt_segment_wallclock_cap_s",
            "resume_compile_smoke": "adapt_resume_compile_smoke",
            "resume_smoke_backend": "adapt_resume_smoke_backend",
        }
        normalized = dict(override)
        for src, dst in alias_pairs.items():
            if src in normalized and dst not in normalized:
                normalized[dst] = normalized[src]
        for key in settings:
            if key in normalized:
                settings[key] = normalized[key]
    return settings


def _disk_free_gb(path: Path) -> float:
    target = Path(path)
    # Do not mkdir here.  The supervisor calls this in the hot scheduling loop,
    # including for external-volume/offload paths.  A slow or wedged /Volumes
    # mount can block inside mkdir and freeze rolling launch/refill.  Disk usage
    # only needs an existing filesystem anchor, so walk upward to one.
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(str(target))
    return float(usage.free) / float(1024**3)


def _mkdir_with_timeout(path: Path, *, timeout_s: float = 5.0) -> tuple[bool, str | None]:
    """Create a directory without allowing a stuck external mount to freeze scheduling."""
    target = Path(path)
    if target.exists():
        return True, None
    try:
        subprocess.run(
            ["/bin/mkdir", "-p", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=True,
        )
        return True, None
    except subprocess.TimeoutExpired:
        return False, f"mkdir_timeout_after_{float(timeout_s):g}s"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        return False, f"mkdir_failed:{detail or exc.returncode}"
    except Exception as exc:
        return False, f"mkdir_error:{type(exc).__name__}:{exc}"


def _first_writable_external_volume() -> Path | None:
    volumes = Path("/Volumes")
    if not volumes.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for child in volumes.iterdir():
        if child.name in {"Macintosh HD", "Home"}:
            continue
        if not child.is_dir() or not os.access(str(child), os.W_OK):
            continue
        try:
            free = _disk_free_gb(child)
        except Exception:
            continue
        candidates.append((free, child))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates[0][1]


def _resolve_offload_dir(raw: str | None) -> Path | None:
    if raw in {None, "", "none", "off", "false"}:
        return None
    if str(raw).lower() == "auto":
        volume = _first_writable_external_volume()
        if volume is None:
            return None
        return volume / "Holstein_optuna_offload"
    return Path(str(raw)).expanduser().resolve()


def _tree_size_bytes(path: Path) -> int:
    total = 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _offload_completed_cycles(
    *,
    supervisor_root: Path,
    offload_dir: Path | None,
    min_free_gb: float,
    active_cycle: int,
    active_cycles: set[int] | None = None,
) -> list[dict[str, Any]]:
    if offload_dir is None:
        return []
    if _disk_free_gb(supervisor_root) >= float(min_free_gb):
        return []
    offloaded: list[dict[str, Any]] = []
    ok, reason = _mkdir_with_timeout(offload_dir, timeout_s=5.0)
    if not ok:
        _write_json(
            supervisor_root / "offload_skipped.json",
            {
                "schema": "paper_i_hh_local_optuna_offload_skipped_v1",
                "generated_utc": _utc_now(),
                "offload_dir": str(offload_dir),
                "reason": reason,
                "min_free_gb": float(min_free_gb),
                "supervisor_free_gb": _disk_free_gb(supervisor_root),
            },
        )
        return []
    for cycle_dir in sorted(supervisor_root.glob("cycle_*")):
        if not cycle_dir.is_dir():
            continue
        if (cycle_dir / "OFFLOADED.json").exists():
            continue
        try:
            cycle_number = int(cycle_dir.name.split("_", 1)[1])
        except Exception:
            cycle_number = -1
        # Do not offload a cycle while any launch in that cycle is still active.
        if cycle_dir.name == f"cycle_{int(active_cycle):04d}" or (
            active_cycles is not None and cycle_number in active_cycles
        ):
            continue
        dest = offload_dir / supervisor_root.name / cycle_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        size_bytes = _tree_size_bytes(cycle_dir)
        shutil.copytree(cycle_dir, dest)
        shutil.rmtree(cycle_dir)
        cycle_dir.mkdir(parents=True, exist_ok=True)
        stub = {
            "schema": "paper_i_hh_local_optuna_offload_stub_v1",
            "generated_utc": _utc_now(),
            "source_path": str(cycle_dir),
            "offload_path": str(dest),
            "size_bytes": int(size_bytes),
        }
        _write_json(cycle_dir / "OFFLOADED.json", stub)
        offloaded.append(stub)
        if _disk_free_gb(supervisor_root) >= float(min_free_gb):
            break
    return offloaded


def _parse_sqlite_datetime(raw: str | None) -> datetime | None:
    if raw in {None, ""}:
        return None
    text = str(raw).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _mark_incompatible_table_shot_trials(storage_path: Path) -> list[dict[str, Any]]:
    """Preserve completed trials even when shot telemetry is blocked.

    Earlier versions demoted every COMPLETE trial whose
    ``paper_i_table_shots_status`` was not ``"ok"``.  That is too destructive
    for the corrected Geo graph-proxy comparison: a blocked ``S_alg`` means the
    shot/work comparison is unavailable, not that the energy and graph-proxy
    result should disappear from the Optuna study.
    """
    return []


def _mark_stale_running_trials(storage_path: Path, *, stale_after_hours: float) -> list[dict[str, Any]]:
    if not storage_path.exists() or float(stale_after_hours) <= 0:
        return []
    con = sqlite3.connect(str(storage_path))
    con.row_factory = sqlite3.Row
    try:
        rows = list(con.execute("select trial_id, number, state, datetime_start, datetime_complete from trials where state='RUNNING'"))
    except sqlite3.Error:
        con.close()
        return []
    now = datetime.now(timezone.utc)
    stale: list[sqlite3.Row] = []
    for row in rows:
        started = _parse_sqlite_datetime(row["datetime_start"])
        if started is None:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_hours = (now - started).total_seconds() / 3600.0
        if age_hours >= float(stale_after_hours):
            stale.append(row)
    marked: list[dict[str, Any]] = []
    if stale:
        complete_text = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        con.executemany(
            "update trials set state='FAIL', datetime_complete=? where trial_id=? and state='RUNNING'",
            [(complete_text, int(row["trial_id"])) for row in stale],
        )
        try:
            cols = [r["name"] for r in con.execute("pragma table_info(trial_system_attributes)")]
            if {"trial_id", "key", "value_json"}.issubset(cols):
                for row in stale:
                    con.execute(
                        "insert into trial_system_attributes (trial_id, key, value_json) values (?, ?, ?)",
                        (
                            int(row["trial_id"]),
                            "paper_i_local_supervisor_stale_stop_note",
                            json.dumps(f"Marked FAIL by {PIPELINE}: stale RUNNING trial exceeded {stale_after_hours} h before relaunch."),
                        ),
                    )
        except sqlite3.Error:
            pass
        con.commit()
        marked = [dict(row) for row in stale]
    con.close()
    return marked


def _cache_root_for_args(args: argparse.Namespace) -> Path:
    raw = getattr(args, "cache_root", None)
    if raw is not None:
        return Path(raw)
    return Path(args.output_root).parent / "cache"


def _hh_pool_cache_dir(args: argparse.Namespace) -> Path:
    return _cache_root_for_args(args) / "hh_pool_cache_v1"


def _candidate_record_cache_dir(args: argparse.Namespace) -> Path:
    return _cache_root_for_args(args) / "static_adapt_candidate_records_v1"


def _base_env(worker_cap: int, *, cache_root: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    cache_root = Path(cache_root) if cache_root is not None else REPO_ROOT / "raw_outputs" / "cache"
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "STATIC_ADAPT_AUTO_WORKER_CAP": str(int(max(1, worker_cap))),
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR": str(cache_root / "static_adapt_candidate_records_v1"),
            "STATIC_ADAPT_HH_POOL_CACHE": "disk",
            "STATIC_ADAPT_HH_POOL_CACHE_DIR": str(cache_root / "hh_pool_cache_v1"),
            "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "paper_i_holstein_sector",
        }
    )
    return env


def _build_launch(
    *,
    args: argparse.Namespace,
    slug: str,
    regime: str,
    target: Mapping[str, Any],
    cycle: int,
    pair_index: int,
    write_files: bool = True,
) -> RegimeLaunch:
    output_dir = Path(args.output_root) / slug / f"cycle_{cycle:04d}" / regime
    storage_path = _storage_path_for_regime(args, slug, str(regime))
    storage_dir = storage_path.parent
    logs_dir = output_dir / "supervisor_logs"
    stdout_log = logs_dir / "stdout.log"
    stderr_log = logs_dir / "stderr.log"
    command_sh = logs_dir / "command.sh"
    graph_proxy_targets = (getattr(args, "_graph_proxy_targets", {}) or {}).get(str(regime), {})
    has_graph_targets = graph_proxy_targets.get("geo_graph_proxy_N2q") is not None and graph_proxy_targets.get("geo_graph_proxy_D2q") is not None
    has_shot_target = target.get("geo_S_alg") is not None
    launch_settings = _launch_settings_for_regime(args, str(regime))
    objective_mode = str(launch_settings["objective_mode"])
    if objective_mode in {
        "geo_energy_then_shot_graph_cost",
        "shot_then_energy_graph_cost",
        "geo_energy_gate_then_shot_energy_graph_cost",
    } and not has_shot_target:
        raise ValueError(f"Shot-focused objective for {regime} requires geo_S_alg in the target manifest.")
    if objective_mode == "shot_then_energy_graph_cost":
        study_suffix = "shotfirst_salg_v7"
    elif objective_mode == "geo_energy_gate_then_shot_energy_graph_cost":
        study_suffix = "egate_shotfirst_salg_v8"
    elif objective_mode == "geo_energy_then_shot_graph_cost":
        study_suffix = "shotdom_salg_v6"
    else:
        study_suffix = "graphdomshot_salg_v5" if (has_graph_targets and has_shot_target) else ("graphdom_v3" if has_graph_targets else "graphshot_v2")
    study_prefix = _study_prefix_for_regime(args, slug, str(regime), study_suffix)
    tag = f"{slug}_{regime.replace('-', '_')}_cycle{cycle:04d}"
    resume_settings = _resume_settings_for_regime(args, str(regime))
    command = [
        sys.executable,
        "-u",
        "-m",
        "pipelines.exact_bench.paper_i_hh_speed_optuna",
        "--regime",
        str(regime),
        "--tag",
        tag,
        "--output-dir",
        str(output_dir),
        "--n-trials",
        str(int(launch_settings["trials_per_chunk"])),
        "--n-startup-trials",
        str(int(launch_settings["n_startup_trials"])),
        "--lanes",
        "canonical",
        "--epsilon-bands",
        "1e9",
        "--optuna-storage",
        str(storage_path),
        "--study-name-prefix",
        study_prefix,
        "--load-if-exists",
        "--objective-mode",
        objective_mode,
        "--geo-target-abs-delta-e",
        str(float(target["geo_abs_delta_e"])),
        "--graph-proxy-target-manifest",
        str(Path(args.graph_proxy_target_manifest)),
        "--geo-graph-proxy-target-manifest",
        str(Path(args.geo_graph_proxy_target_manifest)),
        "--speed-surface-profile",
        str(launch_settings["speed_surface_profile"]),
        "--phase2-w-shot-profile-space",
        str(launch_settings["phase2_w_shot_profile_space"]),
        "--max-depth",
        str(int(launch_settings["max_depth"])),
        "--maxiter",
        str(int(launch_settings["maxiter"])),
        "--final-refit-maxiter",
        str(int(launch_settings["final_refit_maxiter"])),
        "--gradient-workers",
        str(int(launch_settings["gradient_workers"])),
        "--beam-parent-workers",
        str(int(launch_settings["beam_parent_workers"])),
        "--spsa-parallel-evaluations",
        str(int(launch_settings["spsa_parallel_evaluations"])),
        "--runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--symmetry-mode",
        "off",
        "--pool-cache-mode",
        "disk",
        "--pool-cache-dir",
        str(_hh_pool_cache_dir(args)),
        "--pool-cache-scope",
        "paper_i_holstein_sector",
        "--candidate-record-cache-mode",
        "disk",
        "--candidate-record-cache-dir",
        str(_candidate_record_cache_dir(args)),
        "--do-not-force-phase2-w-shot",
    ]
    if bool(launch_settings.get("enable_hva_generators")):
        command.append("--enable-hva-generators")
    if bool(launch_settings.get("hva_aggressive_screening")):
        command.append("--hva-aggressive-screening")
    if args.enqueue_params_json is not None and Path(args.enqueue_params_json).exists():
        command.extend(["--enqueue-params-json", str(Path(args.enqueue_params_json))])
    resume_scaffold_json = resume_settings.get("adapt_resume_scaffold_json")
    if resume_scaffold_json is not None:
        command.extend(["--adapt-resume-scaffold-json", str(Path(str(resume_settings["adapt_resume_scaffold_json"])))])
        if resume_settings.get("adapt_resume_mode") not in {None, ""}:
            command.extend(["--adapt-resume-mode", str(resume_settings["adapt_resume_mode"])])
        if resume_settings.get("adapt_segment_id") not in {None, ""}:
            command.extend(["--adapt-segment-id", str(resume_settings["adapt_segment_id"])])
        if resume_settings.get("adapt_segment_target_depth") is not None:
            command.extend(["--adapt-segment-target-depth", str(int(resume_settings["adapt_segment_target_depth"]))])
        if resume_settings.get("adapt_segment_max_new_admissions") is not None:
            command.extend(["--adapt-segment-max-new-admissions", str(int(resume_settings["adapt_segment_max_new_admissions"]))])
        if resume_settings.get("adapt_segment_wallclock_cap_s") is not None:
            command.extend(["--adapt-segment-wallclock-cap-s", str(float(resume_settings["adapt_segment_wallclock_cap_s"]))])
        if resume_settings.get("adapt_resume_compile_smoke") not in {None, ""}:
            command.extend(["--adapt-resume-compile-smoke", str(resume_settings["adapt_resume_compile_smoke"])])
        if resume_settings.get("adapt_resume_smoke_backend") not in {None, ""}:
            command.extend(["--adapt-resume-smoke-backend", str(resume_settings["adapt_resume_smoke_backend"])])
    if has_graph_targets:
        command.extend(
            [
                "--geo-target-graph-count-2q",
                str(float(graph_proxy_targets["geo_graph_proxy_N2q"])),
                "--geo-target-graph-depth",
                str(float(graph_proxy_targets["geo_graph_proxy_D2q"])),
            ]
        )
    if has_shot_target:
        command.extend(["--geo-target-s-alg", str(float(target["geo_S_alg"]))])
    if launch_settings.get("spsa_profile") not in {None, "", "none"}:
        command.extend(["--spsa-profile", str(launch_settings["spsa_profile"])])
    if args.use_default_warm_starts:
        command.append("--use-default-warm-starts")
    if bool(write_files):
        quoted = " ".join(shlex.quote(str(token)) for token in command)
        _write_text(command_sh, f"#!/usr/bin/env bash\nset -euo pipefail\ncd {shlex.quote(str(REPO_ROOT))}\nexec {quoted}\n", executable=True)
        _write_json(
            logs_dir / "launch.json",
            {
                "schema": "paper_i_hh_local_optuna_regime_launch_v1",
                "generated_utc": _utc_now(),
                "pipeline": PIPELINE,
                "regime": str(regime),
                "cycle": int(cycle),
                "pair_index": int(pair_index),
                "target": dict(target),
                "graph_proxy_target": dict(graph_proxy_targets),
                "graph_cost_target_policy": (
                    "same_surface_graph_proxy"
                    if has_graph_targets
                    else "no_graph_cost_dominance_without_same_surface_proxy_target"
                ),
                "graph_proxy_target_manifest": str(Path(args.graph_proxy_target_manifest)),
                "enqueue_params_json": (
                    None if args.enqueue_params_json is None else str(Path(args.enqueue_params_json))
                ),
                "output_dir": str(output_dir),
                "storage_path": str(storage_path),
                "study_name_prefix": study_prefix,
                "launch_settings": dict(launch_settings),
                "pool_policy": (
                    "full_meta_hva_enabled"
                    if bool(launch_settings.get("enable_hva_generators"))
                    else "full_meta_minus_hva"
                ),
                "screening_policy": (
                    "manual_fixed_hva_aggressive_screening"
                    if bool(launch_settings.get("hva_aggressive_screening"))
                    else "optuna_sampled_staged_shot_shortlist_profiles"
                ),
                "adapt_resume": dict(resume_settings),
                "command": [str(token) for token in command],
            },
        )
    return RegimeLaunch(
        regime=str(regime),
        cycle=int(cycle),
        pair_index=int(pair_index),
        output_dir=str(output_dir),
        storage_path=str(storage_path),
        command=tuple(str(token) for token in command),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command_sh=str(command_sh),
    )


def _start_launch(launch: RegimeLaunch, env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    Path(launch.stdout_log).parent.mkdir(parents=True, exist_ok=True)
    stdout = open(launch.stdout_log, "ab")
    stderr = open(launch.stderr_log, "ab")
    proc = subprocess.Popen(
        list(launch.command),
        cwd=str(REPO_ROOT),
        stdout=stdout,
        stderr=stderr,
        env=dict(env),
    )
    # Keep the handles alive with the Popen object.  Closing them immediately can
    # leave child Python with invalid standard streams under the macOS runtime.
    proc._paper_i_stdout_handle = stdout  # type: ignore[attr-defined]
    proc._paper_i_stderr_handle = stderr  # type: ignore[attr-defined]
    return proc


def _close_launch_stdio(proc: subprocess.Popen[bytes]) -> None:
    for attr in ("_paper_i_stdout_handle", "_paper_i_stderr_handle"):
        handle = getattr(proc, attr, None)
        if handle is None:
            continue
        try:
            handle.close()
        except Exception:
            pass
        try:
            delattr(proc, attr)
        except Exception:
            pass


def _read_current_best(output_dir: Path) -> dict[str, Any] | None:
    for name in ("current_best.json", "live_current_best.json"):
        path = Path(output_dir) / name
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        payload.setdefault("current_best_source", str(path))
        return payload
    return None


def _status_row_for_launch(launch: RegimeLaunch, proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    rc = proc.poll()
    best = _read_current_best(Path(launch.output_dir))
    return {
        "regime": launch.regime,
        "pid": proc.pid,
        "returncode": rc,
        "state": "running" if rc is None else "done",
        "output_dir": launch.output_dir,
        "storage_path": launch.storage_path,
        "stdout_log": launch.stdout_log,
        "stderr_log": launch.stderr_log,
        "current_best": best,
    }


def _write_status(status_path: Path, *, active: bool, rows: Sequence[Mapping[str, Any]], scheduler_mode: str) -> dict[str, Any]:
    generated_utc = _utc_now()
    payload = {
        "schema": "paper_i_hh_local_optuna_supervisor_status_v1",
        "generated_utc": generated_utc,
        "last_update_utc": generated_utc,
        "pipeline": PIPELINE,
        "scheduler_mode": str(scheduler_mode),
        "active": bool(active),
        "rows": [dict(row) for row in rows],
    }
    _write_json(status_path, payload)
    return payload


def _wait_pair(
    *,
    launches: Sequence[RegimeLaunch],
    procs: Sequence[subprocess.Popen[bytes]],
    status_path: Path,
    poll_interval_s: float,
) -> dict[str, Any]:
    assert len(launches) == len(procs)
    while True:
        rows = []
        all_done = True
        for launch, proc in zip(launches, procs):
            row = _status_row_for_launch(launch, proc)
            if row["returncode"] is None:
                all_done = False
            rows.append(row)
        payload = _write_status(status_path, active=not all_done, rows=rows, scheduler_mode="paired")
        if all_done:
            for proc in procs:
                _close_launch_stdio(proc)
            return payload
        time.sleep(float(poll_interval_s))


def _flatten_regime_order(pairs: Sequence[Sequence[str]], targets: Mapping[str, Any]) -> tuple[list[str], dict[str, int]]:
    ordered: list[str] = []
    pair_index_by_regime: dict[str, int] = {}
    for pair_index, pair in enumerate(pairs):
        for regime in pair:
            key = str(regime)
            if key not in targets:
                continue
            pair_index_by_regime.setdefault(key, int(pair_index))
            if key not in ordered:
                ordered.append(key)
    fallback_pair_index = len(pairs)
    for regime in targets:
        key = str(regime)
        if key not in ordered:
            ordered.append(key)
            pair_index_by_regime.setdefault(key, fallback_pair_index)
    return ordered, pair_index_by_regime


def _parse_regime_subset(raw: str | None, targets: Mapping[str, Any]) -> list[str] | None:
    if raw in {None, "", "all"}:
        return None
    aliases = {
        "strong-weak": "intermediate-weak",
        "strong_weak": "intermediate-weak",
        "strong-strong": "intermediate-strong",
        "strong_strong": "intermediate-strong",
        "u8-strong-weak": "strong-weak-u8",
        "u8_strong_weak": "strong-weak-u8",
        "u8-strong-strong": "strong-strong-u8",
        "u8_strong_strong": "strong-strong-u8",
    }
    valid = set(str(key) for key in targets)
    selected: list[str] = []
    for token in str(raw).replace(";", ",").split(","):
        key = token.strip()
        if not key:
            continue
        normalized = key.lower().replace("_", "-")
        normalized = aliases.get(normalized, aliases.get(key.lower(), normalized))
        if normalized not in valid:
            raise ValueError(f"Unknown --regimes entry {key!r}; valid regimes: {', '.join(sorted(valid))}")
        if normalized not in selected:
            selected.append(normalized)
    if not selected:
        raise ValueError("--regimes did not contain any valid regimes.")
    return selected


def _filter_pairs_for_regimes(pairs: Sequence[Sequence[str]], subset: Sequence[str] | None) -> list[list[str]]:
    if subset is None:
        return [[str(regime) for regime in pair] for pair in pairs]
    wanted = set(str(regime) for regime in subset)
    filtered: list[list[str]] = []
    for pair in pairs:
        filtered_pair = [str(regime) for regime in pair if str(regime) in wanted]
        if filtered_pair:
            filtered.append(filtered_pair)
    present = {regime for pair in filtered for regime in pair}
    for regime in subset:
        if str(regime) not in present:
            filtered.append([str(regime)])
    return filtered


def _supervisor_launch_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Return the scientific/throughput settings shared by supervisor launches."""
    return {
        "objective_mode": str(args.objective_mode),
        "speed_surface_profile": str(args.speed_surface_profile),
        "max_depth": int(args.max_depth),
        "maxiter": int(args.maxiter),
        "final_refit_maxiter": int(args.final_refit_maxiter),
        "gradient_workers": int(args.gradient_workers),
        "beam_parent_workers": int(args.beam_parent_workers),
        "spsa_parallel_evaluations": int(args.spsa_parallel_evaluations),
        "spsa_profile": str(args.spsa_profile),
        "phase2_w_shot_profile_space": str(args.phase2_w_shot_profile_space),
        "n_startup_trials": int(args.n_startup_trials),
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "symmetry_mode": "off",
        "pool_cache_mode": "disk",
        "pool_cache_dir": str(_hh_pool_cache_dir(args)),
        "pool_cache_scope": "paper_i_holstein_sector",
        "candidate_record_cache_mode": "disk",
        "candidate_record_cache_dir": str(_candidate_record_cache_dir(args)),
        "cache_root": str(_cache_root_for_args(args)),
        "force_phase2_w_shot": False,
        "adapt_resume_global": _resume_settings_for_regime(args, "__global__"),
        "adapt_resume_scaffold_map_json": (
            None if args.adapt_resume_scaffold_map_json is None else str(Path(args.adapt_resume_scaffold_map_json))
        ),
        "adapt_resume_scaffold_map_regimes": sorted((getattr(args, "_adapt_resume_scaffold_map", {}) or {}).keys()),
        "regime_launch_settings_json": (
            None if args.regime_launch_settings_json is None else str(Path(args.regime_launch_settings_json))
        ),
        "regime_launch_settings_regimes": sorted((getattr(args, "_regime_launch_settings_map", {}) or {}).keys()),
        "regime_storage_map_json": (
            None if args.regime_storage_map_json is None else str(Path(args.regime_storage_map_json))
        ),
        "regime_storage_map_regimes": sorted((getattr(args, "_regime_storage_map", {}) or {}).keys()),
        "regime_study_prefix_map_json": (
            None if args.regime_study_prefix_map_json is None else str(Path(args.regime_study_prefix_map_json))
        ),
        "regime_study_prefix_map_regimes": sorted((getattr(args, "_regime_study_prefix_map", {}) or {}).keys()),
        "graph_proxy_target_manifest": str(Path(args.graph_proxy_target_manifest)),
        "graph_proxy_target_regimes": sorted((getattr(args, "_graph_proxy_targets", {}) or {}).keys()),
        "geo_graph_proxy_target_manifest": str(Path(args.geo_graph_proxy_target_manifest)),
        "enqueue_params_json": (None if args.enqueue_params_json is None else str(Path(args.enqueue_params_json))),
    }


def _run_rolling_supervisor(args: argparse.Namespace) -> int:
    target_manifest = _load_json(Path(args.target_manifest))
    targets = target_manifest["regimes"]
    pairs = target_manifest.get("cycle_pairs") or []
    regime_order, pair_index_by_regime = _flatten_regime_order(pairs, targets)
    regime_subset = _parse_regime_subset(getattr(args, "regimes", None), targets)
    if regime_subset is not None:
        regime_order = list(regime_subset)
    if not regime_order:
        raise ValueError("No regimes found in target manifest for rolling supervisor.")
    slug = _safe_slug(str(args.slug or _default_slug(args)))
    supervisor_root = Path(args.output_root) / slug
    storage_root = Path(args.storage_root) / slug
    offload_dir = _resolve_offload_dir(args.offload_dir)
    cache_root = _cache_root_for_args(args)
    env = _base_env(
        max(args.gradient_workers, args.beam_parent_workers, args.spsa_parallel_evaluations),
        cache_root=cache_root,
    )
    max_concurrent = max(1, int(args.max_concurrent_regimes))
    manifest = {
        "schema": "paper_i_hh_local_optuna_supervisor_manifest_v1",
        "generated_utc": _utc_now(),
        "pipeline": PIPELINE,
        "scheduler_mode": "rolling",
        "slug": slug,
        "target_manifest": str(Path(args.target_manifest)),
        "output_root": str(supervisor_root),
        "storage_root": str(storage_root),
        "cache_root": str(cache_root),
        "offload_dir": None if offload_dir is None else str(offload_dir),
        "cycles_requested": int(args.cycles),
        "trials_per_chunk": int(args.trials_per_chunk),
        "max_concurrent_regimes": int(max_concurrent),
        "objective_mode": str(args.objective_mode),
        "speed_surface_profile": str(args.speed_surface_profile),
        "launch_settings": _supervisor_launch_settings(args),
        "regime_order": list(regime_order),
        "regime_subset": None if regime_subset is None else list(regime_subset),
        "pairs": pairs,
        "worker_env": {key: env[key] for key in sorted(env) if key.startswith("STATIC_ADAPT") or key.endswith("THREADS")},
    }
    if not bool(args.dry_run):
        _write_json(supervisor_root / "supervisor_manifest.json", manifest)
    planned: list[dict[str, Any]] = []
    for position, regime in enumerate(regime_order[:max_concurrent]):
        launch = _build_launch(
            args=args,
            slug=slug,
            regime=str(regime),
            target=targets[str(regime)],
            cycle=1,
            pair_index=int(pair_index_by_regime[str(regime)]),
            write_files=False,
        )
        planned.append(asdict(launch) | {"rolling_position": int(position)})
    if bool(args.dry_run):
        print(json.dumps({"manifest": manifest, "planned_initial_launches": planned}, indent=2))
        return 0

    active: list[tuple[RegimeLaunch, subprocess.Popen[bytes]]] = []
    next_cycle = 1
    next_position = 0
    status_path = supervisor_root / "supervisor_status.json"
    last_rows: list[dict[str, Any]] = []

    def has_more_launches() -> bool:
        return int(args.cycles) == 0 or int(next_cycle) <= int(args.cycles)

    def launch_next() -> None:
        nonlocal next_cycle, next_position
        regime = regime_order[int(next_position)]
        launch = _build_launch(
            args=args,
            slug=slug,
            regime=str(regime),
            target=targets[str(regime)],
            cycle=int(next_cycle),
            pair_index=int(pair_index_by_regime[str(regime)]),
        )
        storage_path = Path(launch.storage_path)
        stale_marked = _mark_stale_running_trials(storage_path, stale_after_hours=float(args.stale_running_hours))
        incompatible_marked = _mark_incompatible_table_shot_trials(storage_path)
        if incompatible_marked:
            stale_best = Path(launch.output_dir) / "current_best.json"
            if stale_best.exists():
                stale_best.unlink()
        _write_json(
            supervisor_root / "last_rolling_launch.json",
            {
                "schema": "paper_i_hh_local_optuna_rolling_launch_v1",
                "generated_utc": _utc_now(),
                "cycle": int(next_cycle),
                "position": int(next_position),
                "regime": str(regime),
                "stale_marked": stale_marked,
                "incompatible_marked": incompatible_marked,
                "launch": asdict(launch),
            },
        )
        proc = _start_launch(launch, env)
        active.append((launch, proc))
        next_position += 1
        if next_position >= len(regime_order):
            next_position = 0
            next_cycle += 1

    while active or has_more_launches():
        _offload_completed_cycles(
            supervisor_root=supervisor_root,
            offload_dir=offload_dir,
            min_free_gb=float(args.min_free_gb),
            active_cycle=int(next_cycle),
            active_cycles={int(launch.cycle) for launch, _proc in active},
        )
        while len(active) < max_concurrent and has_more_launches():
            launch_next()
        rows: list[dict[str, Any]] = []
        still_active: list[tuple[RegimeLaunch, subprocess.Popen[bytes]]] = []
        bad_rows: list[dict[str, Any]] = []
        for launch, proc in active:
            row = _status_row_for_launch(launch, proc)
            rows.append(row)
            if row["returncode"] is None:
                still_active.append((launch, proc))
            elif row.get("returncode") not in {0, None}:
                bad_rows.append(row)
                _close_launch_stdio(proc)
            else:
                _close_launch_stdio(proc)
        active = still_active
        last_rows = rows
        _write_status(
            status_path,
            active=bool(active or has_more_launches()),
            rows=rows,
            scheduler_mode="rolling",
        )
        if bad_rows and not bool(args.continue_on_failure) and not active:
            _write_json(
                supervisor_root / "supervisor_failed.json",
                {
                    "schema": "paper_i_hh_local_optuna_supervisor_failed_v1",
                    "generated_utc": _utc_now(),
                    "scheduler_mode": "rolling",
                    "bad_rows": bad_rows,
                },
            )
            return 2
        if not active and not has_more_launches():
            break
        time.sleep(float(args.poll_interval_s))

    _offload_completed_cycles(
        supervisor_root=supervisor_root,
        offload_dir=offload_dir,
        min_free_gb=float(args.min_free_gb),
        active_cycle=-1,
    )
    _write_status(status_path, active=False, rows=last_rows, scheduler_mode="rolling")
    _write_json(
        supervisor_root / "supervisor_done.json",
        {
            "schema": "paper_i_hh_local_optuna_supervisor_done_v1",
            "generated_utc": _utc_now(),
            "scheduler_mode": "rolling",
            "cycles_completed": int(next_cycle - 1 if next_position == 0 else next_cycle),
        },
    )
    return 0


def _run_supervisor(args: argparse.Namespace) -> int:
    if str(getattr(args, "scheduler_mode", "paired")) == "rolling":
        return _run_rolling_supervisor(args)
    target_manifest = _load_json(Path(args.target_manifest))
    targets = target_manifest["regimes"]
    pairs = target_manifest.get("cycle_pairs") or []
    regime_subset = _parse_regime_subset(getattr(args, "regimes", None), targets)
    filtered_pairs = _filter_pairs_for_regimes(pairs, regime_subset)
    slug = _safe_slug(str(args.slug or _default_slug(args)))
    supervisor_root = Path(args.output_root) / slug
    storage_root = Path(args.storage_root) / slug
    offload_dir = _resolve_offload_dir(args.offload_dir)
    cache_root = _cache_root_for_args(args)
    env = _base_env(
        max(args.gradient_workers, args.beam_parent_workers, args.spsa_parallel_evaluations),
        cache_root=cache_root,
    )
    manifest = {
        "schema": "paper_i_hh_local_optuna_supervisor_manifest_v1",
        "generated_utc": _utc_now(),
        "pipeline": PIPELINE,
        "slug": slug,
        "target_manifest": str(Path(args.target_manifest)),
        "output_root": str(supervisor_root),
        "storage_root": str(storage_root),
        "cache_root": str(cache_root),
        "offload_dir": None if offload_dir is None else str(offload_dir),
        "cycles_requested": int(args.cycles),
        "trials_per_chunk": int(args.trials_per_chunk),
        "objective_mode": str(args.objective_mode),
        "speed_surface_profile": str(args.speed_surface_profile),
        "launch_settings": _supervisor_launch_settings(args),
        "pairs": filtered_pairs,
        "regime_subset": None if regime_subset is None else list(regime_subset),
        "worker_env": {key: env[key] for key in sorted(env) if key.startswith("STATIC_ADAPT") or key.endswith("THREADS")},
    }
    if not bool(args.dry_run):
        _write_json(supervisor_root / "supervisor_manifest.json", manifest)
    planned: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(filtered_pairs):
        for regime in pair:
            launch = _build_launch(args=args, slug=slug, regime=str(regime), target=targets[str(regime)], cycle=1, pair_index=pair_index, write_files=False)
            planned.append(asdict(launch))
    if bool(args.dry_run):
        print(json.dumps({"manifest": manifest, "planned_first_cycle": planned}, indent=2))
        return 0

    cycle = 1
    while True:
        if int(args.cycles) > 0 and cycle > int(args.cycles):
            break
        for pair_index, pair in enumerate(filtered_pairs):
            _offload_completed_cycles(
                supervisor_root=supervisor_root,
                offload_dir=offload_dir,
                min_free_gb=float(args.min_free_gb),
                active_cycle=int(cycle),
            )
            launches: list[RegimeLaunch] = []
            stale_marked: dict[str, list[dict[str, Any]]] = {}
            incompatible_marked: dict[str, list[dict[str, Any]]] = {}
            for regime in pair:
                target = targets[str(regime)]
                launch = _build_launch(args=args, slug=slug, regime=str(regime), target=target, cycle=cycle, pair_index=pair_index)
                storage_path = Path(launch.storage_path)
                stale_marked[str(regime)] = _mark_stale_running_trials(
                    storage_path,
                    stale_after_hours=float(args.stale_running_hours),
                )
                incompatible_marked[str(regime)] = _mark_incompatible_table_shot_trials(storage_path)
                if incompatible_marked[str(regime)]:
                    stale_best = Path(launch.output_dir) / "current_best.json"
                    if stale_best.exists():
                        stale_best.unlink()
                launches.append(launch)
            _write_json(
                supervisor_root / "last_pair_launch.json",
                {
                    "schema": "paper_i_hh_local_optuna_pair_launch_v1",
                    "generated_utc": _utc_now(),
                    "cycle": int(cycle),
                    "pair_index": int(pair_index),
                    "regimes": list(pair),
                    "stale_marked": stale_marked,
                    "incompatible_marked": incompatible_marked,
                    "launches": [asdict(x) for x in launches],
                },
            )
            procs = [_start_launch(launch, env) for launch in launches]
            status = _wait_pair(
                launches=launches,
                procs=procs,
                status_path=supervisor_root / "supervisor_status.json",
                poll_interval_s=float(args.poll_interval_s),
            )
            bad = [row for row in status["rows"] if row.get("returncode") not in {0, None}]
            _offload_completed_cycles(
                supervisor_root=supervisor_root,
                offload_dir=offload_dir,
                min_free_gb=float(args.min_free_gb),
                active_cycle=-1,
            )
            if bad and not bool(args.continue_on_failure):
                _write_json(
                    supervisor_root / "supervisor_failed.json",
                    {
                        "schema": "paper_i_hh_local_optuna_supervisor_failed_v1",
                        "generated_utc": _utc_now(),
                        "cycle": int(cycle),
                        "pair_index": int(pair_index),
                        "bad_rows": bad,
                    },
                )
                return 2
        cycle += 1
    _write_json(
        supervisor_root / "supervisor_done.json",
        {"schema": "paper_i_hh_local_optuna_supervisor_done_v1", "generated_utc": _utc_now(), "cycles_completed": int(cycle - 1)},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", type=str, default=None)
    p.add_argument("--target-manifest", type=Path, default=DEFAULT_TARGET_MANIFEST)
    p.add_argument(
        "--graph-proxy-target-manifest",
        type=Path,
        default=DEFAULT_GRAPH_PROXY_TARGET_MANIFEST,
        help=(
            "Optional same-surface graph-proxy target manifest. The default is the current "
            "SNAKE review-candidate graph-proxy bests, not Paper-I Qiskit/table costs."
        ),
    )
    p.add_argument(
        "--geo-graph-proxy-target-manifest",
        type=Path,
        default=DEFAULT_GEO_GRAPH_PROXY_TARGET_MANIFEST,
        help=(
            "Optional same-surface Geo graph-proxy target manifest. The main target manifest "
            "keeps energy/S_alg/table metadata; its Qiskit/table N2q/D2q values are not used for Optuna graph dominance."
        ),
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help=(
            "Shared cache root for delegated HH pool and candidate-record caches. "
            "Defaults to the sibling raw_outputs/cache directory implied by --output-root."
        ),
    )
    p.add_argument(
        "--regimes",
        type=str,
        default=DEFAULT_SHOT_FOCUSED_REGIMES,
        help=(
            "Optional comma-separated subset/order of target-manifest regimes to cycle. The default "
            "shot-focused queue skips intermediate-weak."
        ),
    )
    p.add_argument("--trials-per-chunk", type=int, default=20)
    p.add_argument("--n-startup-trials", type=int, default=6)
    p.add_argument("--cycles", type=int, default=0, help="0 means repeat indefinitely until stopped.")
    p.add_argument("--scheduler-mode", choices=["paired", "rolling"], default="paired", help="paired waits for each manifest pair; rolling keeps up to --max-concurrent-regimes active slots filled.")
    p.add_argument("--max-concurrent-regimes", type=int, default=2, help="Number of concurrent regime chunks for --scheduler-mode rolling.")
    p.add_argument(
        "--objective-mode",
        choices=[
            "geo_energy_then_graph_shot_cost",
            "geo_energy_then_shot_graph_cost",
            "shot_then_energy_graph_cost",
            "geo_energy_gate_then_shot_energy_graph_cost",
        ],
        default="geo_energy_then_shot_graph_cost",
        help=(
            "Objective passed to the HH speed wrapper. Use shot_then_energy_graph_cost for S_alg first, then energy, then graph; "
            "use geo_energy_gate_then_shot_energy_graph_cost for a hard Geo energy gate followed by S_alg, energy, and graph; "
            "the default is shot-first after the Geo energy gate and shot-target gate; "
            "use geo_energy_then_graph_shot_cost to continue the older graph-first v5 studies."
        ),
    )
    p.add_argument(
        "--speed-surface-profile",
        choices=["staged_graph", "staged_shot", "shortlist_refine", "energy_discovery", "standard"],
        default="staged_shot",
        help=(
            "Search surface passed to the HH speed wrapper; shortlist_refine anchors non-shortlist "
            "settings to per-regime candidate priors and samples shortlist/window/threshold knobs "
            "with maturity shots off."
        ),
    )
    p.add_argument("--max-depth", type=int, default=13)
    p.add_argument("--maxiter", type=int, default=800)
    p.add_argument("--final-refit-maxiter", type=int, default=800)
    p.add_argument("--gradient-workers", type=int, default=8)
    p.add_argument("--beam-parent-workers", type=int, default=8)
    p.add_argument("--spsa-parallel-evaluations", type=int, default=8)
    p.add_argument("--spsa-profile", type=str, default="current")
    p.add_argument(
        "--phase2-w-shot-profile-space",
        choices=["default", "legacy_with_zero"],
        default="default",
        help="Default phase2_w_shot_profile categorical menu passed to the speed wrapper.",
    )
    p.set_defaults(enable_hva_generators=True, hva_aggressive_screening=False)
    p.add_argument(
        "--enable-hva-generators",
        dest="enable_hva_generators",
        action="store_true",
        help=(
            "Pass --enable-hva-generators to each speed-wrapper launch, using HVA-enabled full_meta "
            "instead of the default full_meta_minus_hva surface."
        ),
    )
    p.add_argument(
        "--disable-hva-generators",
        dest="enable_hva_generators",
        action="store_false",
        help="Use the older full_meta_minus_hva surface.",
    )
    p.add_argument(
        "--hva-aggressive-screening",
        dest="hva_aggressive_screening",
        action="store_true",
        help=(
            "Optional manual override that fixes early shortlist sizes. The default leaves "
            "shortlisting to Optuna's sampled staged_shot surface."
        ),
    )
    p.add_argument(
        "--no-hva-aggressive-screening",
        dest="hva_aggressive_screening",
        action="store_false",
        help="Leave shortlist settings to Optuna sampling.",
    )
    p.add_argument("--adapt-resume-scaffold-json", type=Path, default=None)
    p.add_argument(
        "--adapt-resume-scaffold-map-json",
        type=Path,
        default=None,
        help="Optional JSON mapping regime names to structural ADAPT resume scaffold settings.",
    )
    p.add_argument(
        "--regime-launch-settings-json",
        type=Path,
        default=None,
        help="Optional JSON mapping regime names to per-regime launch settings such as max_depth or trials_per_chunk.",
    )
    p.add_argument(
        "--regime-storage-map-json",
        type=Path,
        default=None,
        help="Optional JSON mapping regime names to existing Optuna SQLite storage paths.",
    )
    p.add_argument(
        "--regime-study-prefix-map-json",
        type=Path,
        default=None,
        help="Optional JSON mapping regime names to existing Optuna study-name prefixes.",
    )
    p.add_argument(
        "--enqueue-params-json",
        type=Path,
        default=DEFAULT_CANDIDATE_PRIOR_MANIFEST,
        help="Optional JSON manifest of per-regime Optuna parameter rows to enqueue as candidate priors.",
    )
    p.add_argument("--adapt-resume-mode", choices=["scaffold_v1"], default="scaffold_v1")
    p.add_argument("--adapt-segment-id", type=str, default=None)
    p.add_argument("--adapt-segment-target-depth", type=int, default=None)
    p.add_argument("--adapt-segment-max-new-admissions", type=int, default=None)
    p.add_argument("--adapt-segment-wallclock-cap-s", type=float, default=None)
    p.add_argument("--adapt-resume-compile-smoke", choices=["required", "auto", "off"], default=None)
    p.add_argument("--adapt-resume-smoke-backend", type=str, default=None)
    p.add_argument("--poll-interval-s", type=float, default=60.0)
    p.add_argument("--stale-running-hours", type=float, default=12.0)
    p.add_argument("--min-free-gb", type=float, default=40.0)
    p.add_argument("--offload-dir", type=str, default="auto", help="auto uses the writable external /Volumes device with most free space; none disables offload.")
    p.add_argument("--use-default-warm-starts", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    args._adapt_resume_scaffold_map = _load_adapt_resume_scaffold_map(args.adapt_resume_scaffold_map_json)
    args._regime_launch_settings_map = _load_regime_launch_settings_map(args.regime_launch_settings_json)
    args._regime_storage_map = _load_regime_storage_map(args.regime_storage_map_json)
    args._regime_study_prefix_map = _load_regime_study_prefix_map(args.regime_study_prefix_map_json)
    args._graph_proxy_targets = _load_graph_proxy_target_map(args.graph_proxy_target_manifest)
    if not args._graph_proxy_targets:
        args._graph_proxy_targets = _load_graph_proxy_target_map(args.geo_graph_proxy_target_manifest)
    return _run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
