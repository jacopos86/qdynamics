"""Materialize the source-locked L=4 Qiskit/proxy CHTC campaign."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMPAIGN_ID = "paper_i_l4_qiskit_proxy_extensions_20260825"
HORIZON = 50
SOURCE_ROOTS = (
    Path("pipelines"),
    Path("src"),
    Path("docs"),
    Path("chtc/ra_adapt_l3_manuscript_cost_20260824/run_arm.py"),
    Path("chtc/ra_adapt_l4_route_matrix_20260825/run_l4_arm.py"),
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    files: list[Path] = []
    for relative in SOURCE_ROOTS:
        path = REPO / relative
        if path.is_file():
            files.append(path)
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            rel = candidate.relative_to(REPO)
            if any(part in EXCLUDED_PARTS for part in rel.parts):
                continue
            if candidate.suffix in {".pyc", ".pyo"}:
                continue
            files.append(candidate)
    return sorted(set(files), key=lambda item: item.relative_to(REPO).as_posix())


def _write_source_archive(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(REPO).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)


def _cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = [
        {
            "execution_id": "l4__avqe__k50",
            "arm": "avqe",
            "selector_cost": None,
            "request_cpus": 4,
            "memory_mb": 16384,
            "disk_mb": 40960,
            "max_runtime_seconds": 259200,
        }
    ]
    arm_specs = (
        ("baseline", 24576, 61440),
        ("batching", 32768, 102400),
        ("beam_escalated", 49152, 153600),
        ("metric_ablation_gated", 32768, 102400),
    )
    for selector_cost in ("qiskit", "proxy"):
        for arm, memory_mb, disk_mb in arm_specs:
            label = (
                "prune_gated_protect0"
                if arm == "metric_ablation_gated"
                else arm
            )
            cells.append(
                {
                    "execution_id": f"l4__ra_{selector_cost}__{label}__k50",
                    "arm": arm,
                    "selector_cost": selector_cost,
                    "request_cpus": 4,
                    "memory_mb": memory_mb,
                    "disk_mb": disk_mb,
                    "max_runtime_seconds": 259200,
                }
            )
    return cells


def main() -> None:
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_archive = HERE / "source/source_locked.tar.gz"
    files = _source_files()
    _write_source_archive(files, source_archive)
    inventory = [
        {
            "path": path.relative_to(REPO).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    source_manifest = {
        "schema": "paper_i_l4_source_archive_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": created_at,
        "git_commit": "618b3bf605720cd248ea7ea96af16b1c283d949b",
        "dirty_source_lock": True,
        "files": inventory,
        "inventory_sha256": _canonical_sha256(inventory),
        "archive_size_bytes": source_archive.stat().st_size,
        "archive_sha256": _sha256_file(source_archive),
    }
    (HERE / "source/source_archive_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    cells = _cells()
    plan_body = {
        "schema": "paper_i_l4_qiskit_proxy_campaign_plan_v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": created_at,
        "run_class": "candidate",
        "paper_adoption_authorized": False,
        "execution_authorized": False,
        "target": "pure_hubbard_l4_route_comparison",
        "physics": {
            "model": "Hubbard",
            "num_sites": 4,
            "boundary": "open",
            "ordering": "blocked",
            "t": 1.0,
            "u": 1.25,
            "half_filled_sector": [2, 2],
            "qubits": 8,
            "exact_energy": -3.3827681604170525,
        },
        "route": {
            "candidate_representation": "intact_macro",
            "phase0_gradient_screen": True,
            "adaptive_phase_shortlists": True,
            "adaptive_frontier_ratio": 0.9,
            "phase3_candidate_gain_policy": "joint_total_gain_v1",
            "active_gradient_policy": "stationary",
            "accepted_refit_parameterization": (
                "fixed_fs_whitened_affine_coordinate_chart_v1"
            ),
            "powell": {"xtol": 1e-5, "ftol": 1e-12, "maxiter": 200},
            "horizon": HORIZON,
            "qiskit_reporting": {
                "backend": "FakeMarrakesh",
                "optimization_level": 1,
                "seed_transpiler": 7,
            },
            "prune_variant": {
                "arm": "metric_ablation_gated",
                "small_theta_abs": 1e-3,
                "protect_steps": 0,
                "max_regression": 1e-8,
                "max_predicted_loss_per_cost": 1e-8,
                "delete_and_complete_refit": True,
            },
            "beam_variant": "beam_escalated_branch_compiled_cost_v1",
            "batch_variant": "combinatorial_size3_window6_v1",
        },
        "matrix": {
            "shared_avqe_cell": True,
            "ra_selector_costs": ["qiskit", "proxy"],
            "ra_arms": [
                "baseline",
                "batching",
                "beam_escalated",
                "metric_ablation_gated_protect0",
            ],
            "cells": cells,
        },
        "source_archive": {
            "path": "source/source_locked.tar.gz",
            "sha256": source_manifest["archive_sha256"],
            "inventory_sha256": source_manifest["inventory_sha256"],
        },
        "expected_artifacts": [
            "run/result.json",
            "run/summary.json",
            "run/current.json",
            "run/estimator_ledger.json",
            "execution_receipt.json",
        ],
        "execution_target": "CHTC",
    }
    plan = {**plan_body, "sha256": _canonical_sha256(plan_body)}
    (HERE / "execution_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    authorization_body = {
        "schema": "paper_i_l4_qiskit_proxy_authorization_v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at": created_at,
        "authority_basis": (
            "explicit_user_directive_20260825_submit_l4_avqe_ra_append_batch_beam_prune_"
            "qiskit_and_proxy_k50_in_parallel"
        ),
        "execution_authorized": True,
        "submission_authorized": True,
        "paper_adoption_authorized": False,
        "plan_sha256": plan["sha256"],
        "source_archive_sha256": source_manifest["archive_sha256"],
        "execution_ids": [cell["execution_id"] for cell in cells],
    }
    authorization = {
        **authorization_body,
        "sha256": _canonical_sha256(authorization_body),
    }
    (HERE / "execution_authorization.json").write_text(
        json.dumps(authorization, indent=2) + "\n", encoding="utf-8"
    )
    queue_lines: list[str] = []
    for cell in cells:
        queue_lines.append(
            "\t".join(
                str(cell[key] if cell[key] is not None else "none")
                for key in (
                    "execution_id",
                    "arm",
                    "selector_cost",
                    "request_cpus",
                    "memory_mb",
                    "disk_mb",
                    "max_runtime_seconds",
                )
            )
        )
    (HERE / "queue.tsv").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "cells": len(cells),
                "plan_sha256": plan["sha256"],
                "source_archive_sha256": source_manifest["archive_sha256"],
                "source_archive_size_bytes": source_manifest["archive_size_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
