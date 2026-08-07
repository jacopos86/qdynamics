"""Persistent local six-regime launcher for the canonical FM+SR profile.

This module owns only unattended process orchestration.  Scientific behavior
continues to live in :mod:`pipelines.static_adapt.adapt_pipeline` and the typed
formal-manifold route profile.  The queue is intentionally serial so one local
FM scientific process exists at a time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from pipelines.static_adapt.formal_manifold_route_profile import (
    FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
    FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE,
)
from pipelines.static_adapt.formal_manifold_warm_start import FormalManifoldConfig


SCHEMA = "paper_i_hh_fm_sr_local_campaign_v1"
CAMPAIGN_ID = "paper_i_hh_fm_sr_no_n2_local_20260714"
VISIBLE_SOURCE_LOCK = Path(
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_visible_snake_historical_accounting_audit_20260712.json"
)
CANONICAL_PROFILE_DOC = Path(
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_fm_snake_canonical_runtime_settings_20260714.md"
)
DISK_LAUNCH_FLOOR_GIB = 10.0

# Preserve the visible Paper-I regime order and same-cutoff exact references.
REGIMES: tuple[dict[str, Any], ...] = (
    {
        "id": "weak-weak",
        "u": 0.25,
        "lambda": 0.25,
        "g_ep": 0.353553390593,
        "n_ph_max": 2,
        "exact_energy": -0.9183531194991743,
    },
    {
        "id": "intermediate-weak",
        "u": 1.25,
        "lambda": 0.25,
        "g_ep": 0.353553390593,
        "n_ph_max": 2,
        "exact_energy": -0.49499563910866023,
    },
    {
        "id": "strong-weak-u8",
        "u": 8.0,
        "lambda": 0.25,
        "g_ep": 0.353553390593,
        "n_ph_max": 2,
        "exact_energy": 0.5264587007998404,
    },
    {
        "id": "weak-strong",
        "u": 0.25,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_max": 4,
        "exact_energy": -1.138579200359263,
    },
    {
        "id": "intermediate-strong",
        "u": 1.25,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_max": 4,
        "exact_energy": -0.6239104048313422,
    },
    {
        "id": "strong-strong-u8",
        "u": 8.0,
        "lambda": 1.25,
        "g_ep": 0.790569415042,
        "n_ph_max": 4,
        "exact_energy": 0.5205762777107088,
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(raw)


def _cell_paths(campaign_dir: Path, regime_id: str) -> dict[str, Path]:
    root = campaign_dir / str(regime_id)
    return {
        "root": root,
        "result": root / "result.json",
        "current": root / "current.json",
        "estimator_ledger": root / "estimator_call_ledger.json",
        "stdout": root / "stdout.log",
        "stderr": root / "stderr.log",
    }


def build_cell_argv(
    *,
    repo_root: Path,
    campaign_dir: Path,
    regime: Mapping[str, Any],
    formal_config_json: Path,
) -> list[str]:
    """Return one explicit, source-locked direct-pipeline invocation."""

    paths = _cell_paths(campaign_dir, str(regime["id"]))
    return [
        sys.executable,
        "-m",
        "pipelines.static_adapt.adapt_pipeline",
        "--problem",
        "hh",
        "--L",
        "2",
        "--ordering",
        "blocked",
        "--boundary",
        "open",
        "--t",
        "1.0",
        "--dv",
        "0.0",
        "--omega0",
        "1.0",
        "--boson-encoding",
        "binary",
        "--adapt-pool",
        "full_meta",
        "--adapt-continuation-mode",
        "phase3_v1",
        "--static-route-id",
        "route_a",
        "--static-meta-feature-profile",
        "paper_i_production_v1",
        "--static-lane-route",
        "physical_operator_type",
        "--physical-lane-shortlist-aggressiveness",
        "3",
        "--adapt-inner-optimizer",
        "POWELL",
        "--adapt-maxiter",
        "200",
        "--adapt-scipy-maxfev",
        "0",
        "--adapt-seed",
        "7",
        "--adapt-max-depth",
        "30",
        "--adapt-eps-grad",
        "1e-4",
        "--adapt-eps-energy",
        "1e-8",
        "--adapt-reopt-policy",
        "full",
        "--adapt-window-size",
        "99",
        "--adapt-window-topk",
        "0",
        "--adapt-full-refit-every",
        "8",
        "--adapt-final-full-refit",
        "true",
        "--adapt-final-refit-maxiter",
        "200",
        "--adapt-insertion-mode",
        "append_only",
        "--adapt-allow-repeats",
        "--adapt-no-finite-angle-fallback",
        "--adapt-state-backend",
        "compiled",
        "--adapt-reoptimization-route",
        FORMAL_MANIFOLD_REOPTIMIZATION_ROUTE,
        "--adapt-formal-manifold-route-profile",
        FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
        "--adapt-formal-manifold-config-json",
        str(formal_config_json),
        "--phase0-no-pilot",
        "--phase0-pilot-max-records",
        "0",
        "--phase0-pilot-max-operators",
        "0",
        "--phase1-shortlist-size",
        "24",
        "--phase1-probe-max-positions",
        "999999",
        "--phase1-trough-margin-ratio",
        "1.0",
        "--phase2-shortlist-size",
        "12",
        "--phase2-shortlist-fraction",
        "0.25",
        "--phase3-shortlist-size",
        "12",
        "--phase3-geometry-window-size",
        "99",
        "--phase2-no-batching",
        "--phase3-no-batching",
        "--phase3-runtime-split-mode",
        "shortlist_pauli_children_v1",
        "--allow-archival-phase3-runtime-split",
        "--phase3-runtime-split-selection-mode",
        "archival_child_set_forward_v1",
        "--phase3-runtime-split-max-subset-size",
        "1",
        "--phase3-runtime-split-subset-sizes",
        "1",
        "--phase3-runtime-split-child-set-symmetry-policy",
        "hard_guard",
        "--phase3-runtime-split-child-padding-policy",
        "exact_projected_grouped_v1",
        "--historical-singleton-coordinate-solve-policy",
        "supported_metric_whitened_eigh_v1",
        "--historical-singleton-coordinate-solve-scope",
        "phase2_and_phase3_v1",
        "--historical-singleton-trust-region-update-policy",
        "displacement_calibrated_unbounded_v2",
        "--sr-powell-coordinate-chart-policy",
        "expanded_runtime_projected_logical_v1",
        "--phase2-novelty-mode",
        "collective_span_v1",
        "--phase3-novelty-ablation-mode",
        "no_phase2",
        "--phase1-prune-enabled",
        "--phase1-prune-policy",
        "recoverability_ladder_v1",
        "--phase1-prune-mode",
        "both",
        "--phase1-prune-max-regression",
        "1e-8",
        "--phase1-prune-tolerance-screen-coeff",
        "0.01",
        "--phase1-prune-retained-gain-ratio",
        "0.5",
        "--phase1-prune-protect-steps",
        "2",
        "--phase1-prune-small-theta-abs",
        "0.001",
        "--phase1-prune-small-theta-relative",
        "0.5",
        "--phase1-prune-cooldown-steps",
        "2",
        "--phase1-prune-local-window-size",
        "4",
        "--phase1-prune-checkpoint-period",
        "3",
        "--phase1-prune-schur-nomination-route",
        "metric_regularized_v1",
        "--phase1-prune-metric-schur-mu",
        "0.01",
        "--phase1-prune-metric-schur-solve-mode",
        "stationary_gw_zero_v1",
        "--phase1-prune-metric-schur-cost-weighting",
        "ansatz_entry_denominator_v1",
        "--adapt-beam-live-branches",
        "3",
        "--adapt-beam-children-per-parent",
        "2",
        "--adapt-beam-lambda",
        "0.005",
        "--phase3-backend-cost-mode",
        "marrakesh_graph_span_v1",
        "--phase3-selector-policy",
        "algebraic_nested_v1",
        "--phase3-selector-geometry-mode",
        "reduced",
        "--adapt-exact-gs-override",
        repr(float(regime["exact_energy"])),
        "--u",
        repr(float(regime["u"])),
        "--g-ep",
        repr(float(regime["g_ep"])),
        "--n-ph-max",
        str(int(regime["n_ph_max"])),
        "--adapt-current-json",
        str(paths["current"]),
        "--adapt-current-json-every-depth",
        "1",
        "--adapt-current-json-keep-history-tail",
        "100",
        "--adapt-estimator-call-ledger-json",
        str(paths["estimator_ledger"]),
        "--output-json",
        str(paths["result"]),
        "--skip-pdf",
        "--skip-trajectory",
    ]


def _manifest_path(campaign_dir: Path) -> Path:
    return campaign_dir / "campaign_manifest.json"


def initialize_campaign(*, repo_root: Path, campaign_dir: Path) -> dict[str, Any]:
    """Materialize an immutable no-clobber six-cell launch plan."""

    manifest_path = _manifest_path(campaign_dir)
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("schema") != SCHEMA:
            raise ValueError(f"Refusing to reuse incompatible campaign: {manifest_path}")
        return existing
    if campaign_dir.exists() and any(campaign_dir.iterdir()):
        raise FileExistsError(f"Refusing to clobber nonempty campaign directory: {campaign_dir}")

    source_lock = repo_root / VISIBLE_SOURCE_LOCK
    profile_doc = repo_root / CANONICAL_PROFILE_DOC
    for required in (source_lock, profile_doc):
        if not required.is_file():
            raise FileNotFoundError(required)

    campaign_dir.mkdir(parents=True, exist_ok=True)
    formal_config_path = campaign_dir / "formal_manifold_config.json"
    _write_json(formal_config_path, FormalManifoldConfig().as_dict())

    cells = []
    for order, regime in enumerate(REGIMES):
        paths = _cell_paths(campaign_dir, str(regime["id"]))
        paths["root"].mkdir(parents=True, exist_ok=True)
        argv = build_cell_argv(
            repo_root=repo_root,
            campaign_dir=campaign_dir,
            regime=regime,
            formal_config_json=formal_config_path,
        )
        cells.append(
            {
                "order": int(order),
                "regime": dict(regime),
                "status": "queued",
                "argv": argv,
                "paths": {name: str(path) for name, path in paths.items()},
                "started_utc": None,
                "finished_utc": None,
                "returncode": None,
                "validation": None,
            }
        )

    payload = {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "status": "queued",
        "scientific_concurrency": 1,
        "repo_root": str(repo_root),
        "route_family": "formal_manifold_snake",
        "route_profile": FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
        "candidate_selector_family": "singleton_response_snake",
        "source_lock": {
            "path": str(source_lock),
            "sha256": _sha256(source_lock),
        },
        "profile_doc": {
            "path": str(profile_doc),
            "sha256": _sha256(profile_doc),
        },
        "formal_manifold_config": {
            "path": str(formal_config_path),
            "sha256": _sha256(formal_config_path),
            "qbroyd_enabled": True,
            "qbang_momentum_active": False,
        },
        "settings_reused": {
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "scipy_maxfev": None,
            "max_depth": 30,
            "seed": 7,
            "pool": "full_meta",
            "phase1_shortlist_size": 24,
            "phase2_shortlist_size": 12,
            "phase2_shortlist_fraction": 0.25,
            "beam_live_branches": 3,
            "beam_children_per_parent": 2,
            "beam_lambda": 0.005,
            "same_cutoff": True,
        },
        "settings_changed": {
            "adapt_reopt_policy": "full",
            "formal_manifold_route_profile": FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
            "phase0_pilot_enabled": False,
            "phase2_batching_enabled": False,
            "phase3_batching_enabled": False,
            "coordinate_solve_scope": "phase2_and_phase3_v1",
            "phase3_geometry_window_size": 99,
            "trust_region_update_policy": "displacement_calibrated_unbounded_v2",
            "powell_coordinate_chart": "expanded_runtime_projected_logical_v1",
            "phase2_novelty_multiplier_active": False,
            "phase3_novelty_multiplier_active": False,
            "structural_rollback_enabled": False,
            "adapt_insertion_mode": "append_only",
        },
        "cells": cells,
    }
    _write_json(manifest_path, payload)
    return payload


def _validate_cell_result(cell: Mapping[str, Any]) -> dict[str, Any]:
    paths = {name: Path(value) for name, value in dict(cell["paths"]).items()}
    result = _read_json(paths["result"])
    adapt = result.get("adapt_vqe")
    if not isinstance(adapt, Mapping):
        raise ValueError("Terminal result is missing adapt_vqe.")
    if not bool(adapt.get("success")):
        raise ValueError(f"ADAPT did not succeed: {adapt.get('error')}")
    identity = adapt.get("static_route_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Terminal result is missing static_route_identity.")
    expected = {
        "route_family": "formal_manifold_snake",
        "route_profile": FM_ROUTE_PROFILE_SR_PHASE2_PHASE3_WHITENED_ADAPTIVE_TRUST_NO_N2_V1,
        "candidate_selector_family": "singleton_response_snake",
        "coordinate_solve_scope": "phase2_and_phase3_v1",
        "powell_coordinate_chart_policy": "expanded_runtime_projected_logical_v1",
    }
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    closure = adapt.get("formal_manifold_query_closure")
    if not isinstance(closure, Mapping):
        mismatches["formal_manifold_query_closure"] = {
            "expected": "object",
            "actual": type(closure).__name__,
        }
    else:
        for key, value in {
            "joint_response_selector_invoked": False,
            "formal_combinatorial_selector_invoked": False,
            "singleton_response_selector_invoked": True,
        }.items():
            if closure.get(key) is not value:
                mismatches[f"query_closure.{key}"] = {
                    "expected": value,
                    "actual": closure.get(key),
                }
    if mismatches:
        raise ValueError(f"FM+SR terminal provenance mismatch: {mismatches}")
    for required in (paths["current"], paths["estimator_ledger"]):
        if not required.is_file() or required.stat().st_size <= 0:
            raise ValueError(f"Required sidecar missing or empty: {required}")
    return {
        "status": "validated",
        "energy": adapt.get("energy"),
        "abs_delta_e": adapt.get("abs_delta_e"),
        "controller_round_count": adapt.get("controller_round_count"),
        "ansatz_depth": len(adapt.get("operators", [])),
        "result_sha256": _sha256(paths["result"]),
        "current_sha256": _sha256(paths["current"]),
        "estimator_ledger_sha256": _sha256(paths["estimator_ledger"]),
    }


def _disk_preflight(repo_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(repo_root)
    free_gib = float(usage.free / (1024**3))
    return {
        "free_gib": free_gib,
        "launch_floor_gib": DISK_LAUNCH_FLOOR_GIB,
        "status": "ok" if free_gib >= DISK_LAUNCH_FLOOR_GIB else "blocked",
    }


def run_campaign(*, repo_root: Path, campaign_dir: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(campaign_dir)
    manifest = initialize_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
    lock_path = campaign_dir / ".campaign.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Campaign lock already exists: {lock_path}") from exc
    with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted_utc={_utc_now()}\n")

    try:
        manifest["status"] = "running"
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        for cell in manifest["cells"]:
            if cell.get("status") == "complete":
                continue
            paths = {name: Path(value) for name, value in dict(cell["paths"]).items()}
            if paths["result"].exists() or paths["current"].exists():
                raise RuntimeError(
                    "Refusing an implicit restart/continuation for nonterminal cell "
                    f"{cell['regime']['id']}; inspect {paths['root']} first."
                )
            disk = _disk_preflight(repo_root)
            if disk["status"] != "ok":
                raise RuntimeError(f"Disk preflight blocked launch: {disk}")

            cell["status"] = "running"
            cell["started_utc"] = _utc_now()
            cell["disk_preflight"] = disk
            manifest["active_regime"] = str(cell["regime"]["id"])
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("STATIC_ADAPT_HH_POOL_CACHE", "disk")
            env.setdefault("STATIC_ADAPT_HH_POOL_CACHE_SCOPE", "paper_i_holstein_sector")
            env.setdefault("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "disk")
            with paths["stdout"].open("ab") as stdout_handle, paths["stderr"].open(
                "ab"
            ) as stderr_handle:
                completed = subprocess.run(
                    [str(value) for value in cell["argv"]],
                    cwd=repo_root,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
            cell["returncode"] = int(completed.returncode)
            cell["finished_utc"] = _utc_now()
            if int(completed.returncode) != 0:
                cell["status"] = "failed"
                manifest["status"] = "failed"
                manifest["updated_utc"] = _utc_now()
                _write_json(manifest_path, manifest)
                raise RuntimeError(
                    f"FM cell {cell['regime']['id']} exited {completed.returncode}."
                )
            cell["validation"] = _validate_cell_result(cell)
            cell["status"] = "complete"
            manifest["updated_utc"] = _utc_now()
            _write_json(manifest_path, manifest)

        manifest["status"] = "complete"
        manifest["active_regime"] = None
        manifest["finished_utc"] = _utc_now()
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["updated_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "run"),
        help="Materialize the campaign, or run every queued cell serially.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    if args.command == "plan":
        payload = initialize_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
    else:
        payload = run_campaign(repo_root=repo_root, campaign_dir=campaign_dir)
    print(json.dumps({
        "campaign_id": payload["campaign_id"],
        "status": payload["status"],
        "manifest": str(_manifest_path(campaign_dir)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
