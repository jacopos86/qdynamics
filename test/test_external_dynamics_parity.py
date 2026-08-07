#!/usr/bin/env python3
"""Tests for diagnostic-only external dynamics parity helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pipelines.exact_bench.external_dynamics.adapter import run_dynamics_parity_checks
from pipelines.exact_bench.external_dynamics.local_reference_manager import resolve_external_dynamics_reference
from pipelines.exact_bench.external_dynamics.native_diagnostic_inputs import run_avqds_t_rhs_limit_diagnostic


def _git_init_commit(path: Path) -> str:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "add", "."], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _write_manifest(path: Path, *, name: str, url: str, checkout: Path, commit: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "external_dynamics_parity_refs_v1",
                "repos": [
                    {
                        "name": name,
                        "url": url,
                        "path": str(checkout),
                        "commit": commit,
                        "commit_date": "2026-05-26T00:00:00+00:00",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_missing_manifest_skips_external_check(tmp_path: Path) -> None:
    payload = run_dynamics_parity_checks(
        checks=("adaptive_pvqd_external",),
        manifest_path=tmp_path / "missing.json",
        output_dir=tmp_path / "out",
    )

    result = payload["results"][0]
    assert result["status"] == "skipped_missing_manifest"
    assert result["passed"] is None
    assert result["guardrails"]["paper_facing_table_evidence"] is False
    assert (tmp_path / "out" / "external_dynamics_parity_summary.json").exists()


def test_reference_resolution_skips_missing_checkout_unpinned_and_mismatched_commits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    missing_checkout = tmp_path / "missing-adaptive-pvqd"
    _write_manifest(
        manifest,
        name="adaptive-pvqd",
        url="https://github.com/dalin27/adaptive-pvqd.git",
        checkout=missing_checkout,
        commit="abc123",
    )
    assert (
        resolve_external_dynamics_reference("dalin27_adaptive_pvqd", manifest_path=manifest).status
        == "skipped_missing_checkout"
    )

    checkout = tmp_path / "adaptive-pvqd"
    checkout.mkdir()
    (checkout / "adaptive_pvqd.py").write_text("class AdaptivePVQD: pass\n", encoding="utf-8")
    commit = _git_init_commit(checkout)
    _write_manifest(
        manifest,
        name="adaptive-pvqd",
        url="https://github.com/dalin27/adaptive-pvqd.git",
        checkout=checkout,
        commit="badcommit",
    )
    assert (
        resolve_external_dynamics_reference("dalin27_adaptive_pvqd", manifest_path=manifest).status
        == "skipped_provenance_mismatch"
    )

    manifest.write_text(
        json.dumps(
            {
                "schema": "external_dynamics_parity_refs_v1",
                "repos": [
                    {
                        "name": "adaptive-pvqd",
                        "url": "https://github.com/dalin27/adaptive-pvqd.git",
                        "path": str(checkout),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        resolve_external_dynamics_reference("dalin27_adaptive_pvqd", manifest_path=manifest).status
        == "skipped_unpinned_reference"
    )
    assert (
        resolve_external_dynamics_reference(
            "dalin27_adaptive_pvqd",
            manifest_path=manifest,
            allow_unpinned_local_reference=True,
        ).status
        == "available"
    )
    assert commit



def test_adaptive_pvqd_source_probe_uses_pinned_local_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "adaptive-pvqd"
    checkout.mkdir()
    (checkout / "adaptive_pvqd.py").write_text(
        "class AdaptivePVQD:\n"
        "    def get_loss(self):\n"
        "        return SuzukiTrotter, PauliEvolutionGate\n"
        "    def adaptive_step(self):\n"
        "        # Method 1: Tetris-like\n"
        "        return set().isdisjoint(set())\n"
        "    def minimization_routine(self): pass\n"
        "    def one_time_step(self): pass\n"
        "    def evolve(self): pass\n",
        encoding="utf-8",
    )
    commit = _git_init_commit(checkout)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        name="adaptive-pvqd",
        url="https://github.com/dalin27/adaptive-pvqd.git",
        checkout=checkout,
        commit=commit,
    )

    resolved = resolve_external_dynamics_reference("dalin27_adaptive_pvqd", manifest_path=manifest)
    assert resolved.status == "available"
    assert resolved.provenance_verified is True

    payload = run_dynamics_parity_checks(
        checks=("adaptive_pvqd_external",),
        manifest_path=manifest,
        output_dir=tmp_path / "out",
    )
    result = payload["results"][0]
    assert result["status"] == "completed_source_probe"
    assert result["provenance_status"] == "verified_pinned_checkout"
    assert result["passed"] is None
    worker = result["details"]["worker"]
    assert worker["source_conformance_passed"] is True
    assert worker["numeric_parity_executed"] is False


def test_avqds_source_probe_reports_reference_features(tmp_path: Path) -> None:
    checkout = tmp_path / "avqds"
    source_dir = checkout / "avqds"
    source_dir.mkdir(parents=True)
    (source_dir / "ansatz.py").write_text(
        "class ansatz:\n"
        "    def one_step(self):\n"
        "        # McLachlan distance\n"
        "        self._rcut = 1e-3\n"
        "    def add_ops_dyn(self):\n"
        "        self._ansatz = []\n"
        "        self._ansatz.append(1)\n"
        "    def get_dist(self): pass\n"
        "    def set_par_states(self): pass\n",
        encoding="utf-8",
    )
    (source_dir / "avaridyn.py").write_text(
        "class avaridynBase:\n"
        "    def run(self): pass\n"
        "    def set_initial_state(self): pass\n"
        "    def init_records(self): pass\n"
        "    def update_records(self): pass\n",
        encoding="utf-8",
    )
    commit = _git_init_commit(checkout)
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        name="avqds",
        url="https://gitlab.com/gqce/avqds.git",
        checkout=checkout,
        commit=commit,
    )

    payload = run_dynamics_parity_checks(
        checks=("avqds_external",),
        manifest_path=manifest,
        output_dir=tmp_path / "out",
    )
    result = payload["results"][0]
    assert result["status"] == "completed_source_probe"
    worker = result["details"]["worker"]
    assert worker["source_conformance_passed"] is True
    assert worker["features"]["uses_mclachlan_distance_cutoff"] is True


def test_avqds_t_rhs_limit_diagnostic_passes_and_is_non_promotional() -> None:
    payload = run_avqds_t_rhs_limit_diagnostic(dt_scales=(0.05, 0.2, 0.1))

    assert payload["status"] == "completed"
    assert payload["passed"] is True
    assert payload["diagnostic_only_not_paper_evidence"] is True
    assert payload["normal_avqds_t_semantics_modified"] is False
    deltas = [row["target_tangent_l2_delta"] for row in payload["checks"]]
    assert deltas == sorted(deltas, reverse=True)


def test_external_dynamics_cli_summary_records_source_probes_and_native_pass(tmp_path: Path) -> None:
    payload = run_dynamics_parity_checks(
        checks=("avqds_t_rhs_limit",),
        output_dir=tmp_path / "out",
    )

    summary = payload["summary"]
    assert summary["status"] == "completed"
    assert summary["passed"] is True
    assert summary["paper_facing_table_evidence"] is False
    assert (tmp_path / "out" / "avqds_t_rhs_limit" / "native_diagnostic.json").exists()
