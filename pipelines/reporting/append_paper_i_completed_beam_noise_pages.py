#!/usr/bin/env python3
"""Append authenticated Page-14 beam and Page-15 noise progress pages.

The evolving Paper-I report already owns Pages 1--13.  This updater preserves
those pages byte-for-byte at the PDF content-stream level, validates completed
CHTC archives against their sealed package jobs, reconstructs the shared
Paper-I Qiskit five-tuple, and appends (or atomically replaces) two pages:

* Page 14: Page-13 macro Phase-0 route with 3-branch beam and metric pruning.
* Page 15: two pure-Hubbard Page-12 plots with the three noise levels overlaid.

Incomplete cells remain explicit and never replace completed evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import uuid
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
TARGET_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
TARGET_PROVENANCE = TARGET_PDF.with_name(
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)

PAGE13_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_proxy_no_lanes_page13_adapter.json"
)
PAGE14_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_beam3x2_metric_page14"
)
PAGE15_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "pure_hubbard_page12_fullnoise_page15"
)
PAGE14_PDF = REPORT_DIR / f"{PAGE14_STEM}.pdf"
PAGE14_PNG = REPORT_DIR / f"{PAGE14_STEM}.png"
PAGE14_ADAPTER = REPORT_DIR / f"{PAGE14_STEM}_adapter.json"
PAGE15_PDF = REPORT_DIR / f"{PAGE15_STEM}.pdf"
PAGE15_PNG = REPORT_DIR / f"{PAGE15_STEM}.png"
PAGE15_ADAPTER = REPORT_DIR / f"{PAGE15_STEM}_adapter.json"

PAGE14_ID = "macro_phase0_beam3x2_metric_prune_partial_v1"
PAGE15_ID = "pure_hubbard_page12_fullnoise_two_panel_overlay_v2"
PAGE15_PRIOR_ID = "pure_hubbard_page12_fullnoise_partial_v1"

BEAM_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "beam3x2_metric_prune_cap24_tau1em4_r20_20260811_v4_chtc"
)
BEAM_RETRIEVED = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260812_page13_macro_beam_v4"
)
NOISE_R50_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3_chtc"
)
NOISE_R50_RETRIEVED = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260811_pure_hubbard_noise_v3"
)
NOISE_R20_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1_chtc"
)
NOISE_R20_RETRIEVED = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260812_pure_hubbard_noise_r20_v1"
)
PENDING_NOISE_R30_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r30_20260812_v1_chtc"
)

BEAM_PACKAGE_SHA256 = (
    "125f4857de21ce348c6ad8e437cd8b1a728bc6b4c1be780e749fc10361380658"
)
NOISE_R50_PACKAGE_SHA256 = (
    "ed280fa195757fe9b3363b36dc85fdab851d9411a7b5e77f356242fe3944f38b"
)
NOISE_R20_PACKAGE_SHA256 = (
    "494971753470c7a83093c849d4d35d7dba48424a3fe2ca61b9fbd8b136cd3a8b"
)
PENDING_NOISE_R30_PACKAGE_SHA256 = (
    "0b30d0314caa44047cff1af850bc84b065c3a68d628346ffbad9cd2959214dce"
)

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
NPH = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
NOISE_U_ORDER = ("u1p5", "u8")
NOISE_LEVEL_ORDER = ("low", "high", "extreme")
NOISE_U_LABELS = {"u1p5": r"$U/t=1.5$", "u8": r"$U/t=8$"}

PLOT_FLOOR = 1.0e-16
BLUE = "#4C78A8"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"


BEAM_CLUSTER_ID = 9638417
NOISE_R50_CLUSTER_ID = 9634547
NOISE_R20_CLUSTER_ID = 9636601
PENDING_NOISE_R30_CLUSTER_ID = 9644468
SUPERSEDED_BEAM_CLUSTER_ID = 9631689
SUPERSEDED_BEAM_PACKAGE_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "beam3x2_metric_prune_cap24_tau1em4_r20_20260811_v3_chtc"
)
SUPERSEDED_BEAM_PACKAGE_SHA256 = (
    "a182ffcbbdb4763d5ffbeeb160152c0de49e7d254d1740686493050890b340b3"
)

BEAM_ARCHIVES = {
    "weak_weak": {
        "proc_id": 0,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__weak_weak__"
            "nph3__ra_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
            "plateau_beam3x2_metric_prune__9638417__0.tar.gz"
        ),
        "size_bytes": 290_332_669,
        "sha256": "8d8e88d1942cb4ceb43f75f1705dd5d7171c0bc8bf991b5773c1f0426e2e4770",
    },
    "intermediate_weak": {
        "proc_id": 1,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__"
            "intermediate_weak__nph3__ra_macro_gradient_phase0_macro_phase123_"
            "proxy_no_lanes_plateau_beam3x2_metric_prune__9638417__1.tar.gz"
        ),
        "size_bytes": 339_489_014,
        "sha256": "2766ba8cd6a878d7a9fb77504d6133754a18d2f92c108c59426aafb7b83fb727",
    },
    "strong_weak_u8": {
        "proc_id": 2,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__"
            "strong_weak_u8__nph3__ra_macro_gradient_phase0_macro_phase123_"
            "proxy_no_lanes_plateau_beam3x2_metric_prune__9638417__2.tar.gz"
        ),
        "size_bytes": 265_280_978,
        "sha256": "8dd2d2298240c0dd05083e5f81bc9516fa6dedf1cc065da405cc14b40abbfba2",
    },
    "weak_strong": {
        "proc_id": 3,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__weak_strong__"
            "nph7__ra_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
            "plateau_beam3x2_metric_prune__9638417__3.tar.gz"
        ),
        "size_bytes": 303_811_642,
        "sha256": "70cdb2b96b36c5fa8d617d58b7d43f588dbdcbd21efb7d3aa37e81f4ee572882",
    },
    "intermediate_strong": {
        "proc_id": 4,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__"
            "intermediate_strong__nph7__ra_macro_gradient_phase0_macro_phase123_"
            "proxy_no_lanes_plateau_beam3x2_metric_prune__9638417__4.tar.gz"
        ),
        "size_bytes": 234_589_358,
        "sha256": "69b76b4cfc25db24746a9bbbc58d4dc451b8104b2791f2a224654229b9a829b0",
    },
    "strong_strong_u8": {
        "proc_id": 5,
        "filename": (
            "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric__"
            "strong_strong_u8__nph7__ra_macro_gradient_phase0_macro_phase123_"
            "proxy_no_lanes_plateau_beam3x2_metric_prune__9638417__5.tar.gz"
        ),
        "size_bytes": 275_161_207,
        "sha256": "8d8e069987ec657e2f413223f2d44a83b931738c5aedb79f83a59154f9e64fa3",
    },
}

NOISE_ARCHIVES = {
    ("u1p5", "low"): {
        "proc_id": 0,
        "target_horizon": 20,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1/"
            "pure_hubbard_page12_fullnoise__u1p5__low__9636601__0.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u1p5__low__9636601__0.tar.gz"
        ),
        "size_bytes": 48_822_613,
        "sha256": "bda143927e4eab8e37f396b5ca9343831ad500c8d80e1ad5a8bb20b7a8bf721f",
    },
    ("u1p5", "high"): {
        "proc_id": 1,
        "target_horizon": 20,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1/"
            "pure_hubbard_page12_fullnoise__u1p5__high__9636601__1.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u1p5__high__9636601__1.tar.gz"
        ),
        "size_bytes": 44_276_774,
        "sha256": "c083cbfedea4ebe9a49acec25ca1b48f63c203f761a281b3211612466622c759",
    },
    ("u1p5", "extreme"): {
        "proc_id": 2,
        "target_horizon": 50,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3/"
            "pure_hubbard_page12_fullnoise__u1p5__extreme__9634547__2.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u1p5__extreme__9634547__2.tar.gz"
        ),
        "size_bytes": 205_692_040,
        "sha256": "b7e2d35e33e9c195f1ab7eb3282f10b873d217e2b4cb4d5345e3429b5f38b078",
    },
    ("u8", "low"): {
        "proc_id": 2,
        "target_horizon": 20,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1/"
            "pure_hubbard_page12_fullnoise__u8__low__9636601__2.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u8__low__9636601__2.tar.gz"
        ),
        "size_bytes": 49_470_933,
        "sha256": "8e204d137817e2295e1b67568f8b38399e687c631cccd36ab105f5a759606a6c",
    },
    ("u8", "high"): {
        "proc_id": 3,
        "target_horizon": 20,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1/"
            "pure_hubbard_page12_fullnoise__u8__high__9636601__3.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u8__high__9636601__3.tar.gz"
        ),
        "size_bytes": 49_639_473,
        "sha256": "c027d590bf146016b9db0373c15ede2aa4160abc592e49f77cc14e9cd28c44db",
    },
    ("u8", "extreme"): {
        "proc_id": 5,
        "target_horizon": 50,
        "remote_path": (
            "/staging/jsstrobel/"
            "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3/"
            "pure_hubbard_page12_fullnoise__u8__extreme__9634547__5.tar.gz"
        ),
        "filename": (
            "pure_hubbard_page12_fullnoise__u8__extreme__9634547__5.tar.gz"
        ),
        "size_bytes": 216_361_821,
        "sha256": "e845fdafc56eefaf155923dc5de8c78921fcde46bbb8234c8f39619700ff6fba",
    },
}


class UpdateError(ValueError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise UpdateError(f"unsafe or missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpdateError(f"JSON object required: {path}")
    return value


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if claimed != _canonical_sha256(unsigned):
        raise UpdateError(f"{label}: self digest drifted")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_member_name(name: str) -> str:
    raw = name.removeprefix("./")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise UpdateError(f"unsafe archive member: {name}")
    return path.as_posix()


def _package_jobs(package: Path, expected_manifest_sha256: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    manifest_path = package / "package_manifest.json"
    manifest = load(manifest_path)
    verify_self_digest(manifest, label=f"{package.name} package manifest")
    if manifest.get("sha256") != expected_manifest_sha256:
        raise UpdateError(f"package manifest identity drifted: {package}")
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((package / "jobs").glob("*.json")):
        job = load(path)
        verify_self_digest(job, label=f"job {path.name}")
        execution_id = str(job.get("execution_id"))
        if not execution_id or execution_id in result:
            raise UpdateError(f"duplicate/invalid job identity: {path}")
        result[execution_id] = (path, job)
    return result


def _compile_cost_tuple(summary: Mapping[str, Any], *, round_index: int) -> tuple[dict[str, int], dict[str, Any]]:
    from pipelines.reporting.ingest_paper_i_phase0_completed_archive import (
        _prefix_compile_input,
    )
    from pipelines.reporting.paper_i_run_summary import (
        compile_paper_i_prefix_qiskit_payload,
    )

    requested = [
        row
        for row in summary.get("requested_rounds", [])
        if row.get("controller_round") == round_index
    ]
    if len(requested) != 1 or requested[0].get("status") != "available":
        raise UpdateError(f"requested round {round_index} compilation is unavailable")
    row = requested[0]
    prefix = _prefix_compile_input(row.get("prefix"))
    if prefix.controller_round != round_index:
        raise UpdateError("requested prefix round drifted")
    payload = compile_paper_i_prefix_qiskit_payload(prefix)
    if (
        payload.get("compile_convention") != "table_i_basis_gate_transpile_v1"
        or payload.get("qiskit_basis_work_status") != "ok"
    ):
        raise UpdateError("shared Paper-I Qiskit compiler contract drifted")
    costs = {
        "N2q": int(payload["compiled_count_2q_total"]),
        "D2q": int(payload["compiled_depth_2q_total"]),
        "Dc": int(payload["compiled_depth_total"]),
        "W1q": int(payload["qiskit_pretranspile_pauli_1q_work_total"]),
        "S_alg": int(prefix.algorithmic_work.s_alg),
    }
    serialized = row.get("resources")
    expected = {
        "N2q": serialized.get("compiled_two_qubit_count") if isinstance(serialized, Mapping) else None,
        "D2q": serialized.get("compiled_two_qubit_depth") if isinstance(serialized, Mapping) else None,
        "Dc": serialized.get("compiled_total_depth") if isinstance(serialized, Mapping) else None,
    }
    if any(costs[key] != expected[key] for key in expected):
        raise UpdateError("shared Qiskit recompile differs from serialized triplet")
    return costs, {
        "compile_convention": payload.get("compile_convention"),
        "qiskit_version": payload.get("qiskit_version"),
        "compiled_basis_gates": payload.get("compiled_basis_gates"),
        "qiskit_transpile_optimization_level": payload.get(
            "qiskit_transpile_optimization_level"
        ),
        "qiskit_transpile_seed": payload.get("qiskit_transpile_seed"),
        "qiskit_basis_work_status": payload.get("qiskit_basis_work_status"),
        "source": "PaperIPrefixCompileInput_shared_locked_compiler_cross_checked_v1",
    }


def _archive_result(
    *,
    path: Path,
    expected: Mapping[str, Any],
    cluster_id: int,
    job_path: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    archive_binding = binding(path)
    if (
        archive_binding["sha256"] != expected["sha256"]
        or archive_binding["size_bytes"] != expected["size_bytes"]
    ):
        raise UpdateError(f"archive identity drifted: {path}")
    with tarfile.open(path, "r:gz") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            relative = _safe_member_name(member.name)
            if relative in members:
                raise UpdateError(f"duplicate archive member: {relative}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise UpdateError(f"unsafe archive member type: {relative}")
            members[relative] = member

        def read_json(suffix: str) -> tuple[str, bytes, dict[str, Any]]:
            matches = [name for name in members if name.endswith(suffix)]
            if len(matches) != 1 or not members[matches[0]].isfile():
                raise UpdateError(f"archive requires one {suffix}: {path}")
            stream = archive.extractfile(members[matches[0]])
            if stream is None:
                raise UpdateError(f"archive member is unreadable: {matches[0]}")
            raw = stream.read()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise UpdateError(f"archive JSON object required: {matches[0]}")
            return matches[0], raw, value

        worker_name, worker_raw, worker = read_json("worker_receipt.json")
        manifest_name, manifest_raw, manifest = read_json("execution_manifest.json")
        summary_name, summary_raw, summary = read_json("summary/summary.json")

        verify_self_digest(worker, label=f"{path.name} worker receipt")
        verify_self_digest(manifest, label=f"{path.name} execution manifest")
        execution_id = str(job["execution_id"])
        target_horizon = int(job["target_horizon"])
        if (
            worker.get("status") != "passed"
            or worker.get("execution_id") != execution_id
            or worker.get("job_spec_sha256") != job["sha256"]
            or worker.get("package_id") != job["package_id"]
            or worker.get("controller_rounds_completed") != target_horizon
            or manifest.get("status") != "passed"
            or manifest.get("execution_id") != execution_id
            or manifest.get("job_spec_sha256") != job["sha256"]
            or manifest.get("package_id") != job["package_id"]
            or manifest.get("controller_rounds_completed") != target_horizon
            or manifest.get("target_horizon") != target_horizon
            or manifest.get("sha256") != worker.get("execution_manifest_sha256")
        ):
            raise UpdateError(f"completed execution closure drifted: {execution_id}")
        output_payloads = manifest.get("output_payloads")
        if not isinstance(output_payloads, Mapping):
            raise UpdateError("execution manifest output inventory is absent")
        for role in ("checkpoint", "result", "estimator_ledger", "summary"):
            row = output_payloads.get(role)
            if not isinstance(row, Mapping):
                raise UpdateError(f"execution manifest lacks {role}")
            member_name = str(row.get("path", "")).removeprefix("./")
            member = members.get(member_name)
            if member is None or not member.isfile() or member.size != row.get("size_bytes"):
                raise UpdateError(f"archive {role} binding drifted: {execution_id}")
        summary_binding = output_payloads["summary"]
        if (
            summary_name != str(summary_binding["path"])
            or hashlib.sha256(summary_raw).hexdigest() != summary_binding["sha256"]
        ):
            raise UpdateError(f"summary bytes drifted: {execution_id}")
        trace = summary.get("accepted_error_trace")
        if (
            summary.get("schema") != "paper_i_run_summary_v1"
            or not isinstance(trace, list)
            or [row.get("controller_round") for row in trace]
            != list(range(1, target_horizon + 1))
        ):
            raise UpdateError(f"accepted trajectory drifted: {execution_id}")
        exact = float(summary["provenance"]["exact_same_cutoff_energy"])
        points = []
        for row in trace:
            if not math.isclose(
                float(row["exact_same_cutoff_energy"]),
                exact,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise UpdateError(f"same-cutoff reference drifted: {execution_id}")
            points.append(
                {
                    "k": int(row["controller_round"]),
                    "energy": float(row["accepted_energy"]),
                    "error": float(row["absolute_energy_error"]),
                    "active_ansatz_depth": int(row["active_ansatz_depth"]),
                }
            )
        costs, compile_receipt = _compile_cost_tuple(
            summary,
            round_index=target_horizon,
        )
        return {
            "status": "completed_authenticated_chtc_archive",
            "cluster_id": cluster_id,
            "proc_id": int(expected["proc_id"]),
            "execution_id": execution_id,
            "target_horizon": target_horizon,
            "points": points,
            "terminal": copy.deepcopy(points[-1]),
            "costs": costs,
            "compile": compile_receipt,
            "sources": {
                "archive": archive_binding,
                "retrieval_identity": {
                    "expected_sha256": str(expected["sha256"]),
                    "expected_size_bytes": int(expected["size_bytes"]),
                    "remote_path": expected.get("remote_path"),
                    "remote_state": (
                        "preserved_after_exact_size_sha256_verified_fetch"
                        if expected.get("remote_path")
                        else "prior_local_fetch_remote_digest_not_recorded"
                    ),
                    "local_final_rename_completed": True,
                    "gzip_integrity": "passed_via_tarfile_full_member_scan",
                },
                "worker_receipt": {
                    "member": worker_name,
                    "sha256": hashlib.sha256(worker_raw).hexdigest(),
                    "canonical_sha256": worker["sha256"],
                },
                "execution_manifest": {
                    "member": manifest_name,
                    "sha256": hashlib.sha256(manifest_raw).hexdigest(),
                    "canonical_sha256": manifest["sha256"],
                },
                "summary": {
                    "member": summary_name,
                    "sha256": hashlib.sha256(summary_raw).hexdigest(),
                },
                "job": binding(job_path),
            },
        }


def _base_page13() -> dict[str, Any]:
    adapter = load(PAGE13_ADAPTER)
    verify_self_digest(adapter, label="Page-13 adapter")
    cells = adapter.get("cells")
    if (
        adapter.get("page_id")
        != "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"
        or not isinstance(cells, list)
        or [row.get("regime_id") for row in cells] != list(REGIME_ORDER)
        or any(
            not isinstance(row.get("macro_phase0_route"), Mapping)
            or row["macro_phase0_route"].get("status")
            != "completed_authenticated_local_summary"
            for row in cells
        )
    ):
        raise UpdateError("Page-13 completed route authority drifted")
    return adapter


def build_beam_adapter() -> dict[str, Any]:
    jobs = _package_jobs(BEAM_PACKAGE, BEAM_PACKAGE_SHA256)
    base = _base_page13()
    base_cells = {row["regime_id"]: row for row in base["cells"]}
    cells = []
    for regime in REGIME_ORDER:
        expected_id_fragment = f"__{regime}__nph{NPH[regime]}__"
        matches = [
            value
            for key, value in jobs.items()
            if expected_id_fragment in key
        ]
        if len(matches) != 1:
            raise UpdateError(f"beam job coverage drifted: {regime}")
        job_path, job = matches[0]
        if (
            job.get("regime_id") != regime
            or job.get("nph") != NPH[regime]
            or job.get("target_horizon") != 20
            or job.get("candidate_representation") != "macro_generator_v1"
        ):
            raise UpdateError(f"beam job identity drifted: {regime}")
        completed = None
        archive_spec = BEAM_ARCHIVES.get(regime)
        if archive_spec is not None:
            completed = _archive_result(
                path=BEAM_RETRIEVED / str(archive_spec["filename"]),
                expected=archive_spec,
                cluster_id=BEAM_CLUSTER_ID,
                job_path=job_path,
                job=job,
            )
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "conventional_unwhitened_adapt": copy.deepcopy(
                    base_cells[regime]["conventional_unwhitened_adapt"]
                ),
                "page13_unpruned_route": copy.deepcopy(
                    base_cells[regime]["macro_phase0_route"]
                ),
                "beam_metric_route": completed,
                "status": (
                    "completed_authenticated_chtc_archive"
                    if completed is not None
                    else "pending_no_completed_archive"
                ),
                "job": binding(job_path),
            }
        )
    completed_count = sum(
        cell["beam_metric_route"] is not None for cell in cells
    )
    pending_count = len(REGIME_ORDER) - completed_count
    limitations = [
        "the beam run ends at controller round 20 and is not a round-50 replacement",
        "the worker append-matched observation is unavailable because the source-locked worker omitted the canonical Append registry; trajectory and requested-prefix resources remain closed",
        (
            "the defective v3 cluster 9631689 is superseded by the corrected "
            "source-locked v4 cluster 9638417"
        ),
    ]
    if pending_count:
        limitations.insert(
            0,
            f"{pending_count} of 6 corrected v4 cells lack completed archives",
        )
    unsigned = {
        "schema": "paper_i_macro_phase0_beam_metric_page14_adapter_v1",
        "page_id": PAGE14_ID,
        "status": (
            "completed_6_of_6"
            if completed_count == len(REGIME_ORDER)
            else f"partial_{completed_count}_of_6_completed"
        ),
        "paper_evidence_adopted": False,
        "source_package": binding(BEAM_PACKAGE / "package_manifest.json"),
        "source_page13_adapter": binding(PAGE13_ADAPTER),
        "cluster_id": BEAM_CLUSTER_ID,
        "completed_regime_count": completed_count,
        "pending_regime_count": pending_count,
        "supersedes": {
            "package_id": SUPERSEDED_BEAM_PACKAGE_ID,
            "package_manifest_canonical_sha256": (
                SUPERSEDED_BEAM_PACKAGE_SHA256
            ),
            "cluster_id": SUPERSEDED_BEAM_CLUSTER_ID,
            "reason": "beam_pool_contraction_defect_corrected_in_v4",
            "prior_completed_archives_preserved": True,
            "prior_page_evidence_state": "superseded_defective_v3",
        },
        "route_delta_from_page13": {
            "beam": "fork_local_three_branches_keep_two_v1",
            "pruning": "metric_pruning_v1",
            "target_horizon": 20,
            "unchanged": (
                "macro gradient Phase 0; macro Phase I/II/III; proxy cost; "
                "no lanes; stationary source response; commutation-reduced "
                "relative plateau insertion tau=1e-4; Powell-200; seed 7"
            ),
        },
        "cells": cells,
        "limitations": limitations,
    }
    unsigned["sha256"] = _canonical_sha256(unsigned)
    return unsigned


def build_noise_adapter() -> dict[str, Any]:
    jobs_by_horizon = {
        20: _package_jobs(NOISE_R20_PACKAGE, NOISE_R20_PACKAGE_SHA256),
        50: _package_jobs(NOISE_R50_PACKAGE, NOISE_R50_PACKAGE_SHA256),
    }
    pending_r30_jobs = _package_jobs(
        PENDING_NOISE_R30_PACKAGE,
        PENDING_NOISE_R30_PACKAGE_SHA256,
    )
    for u_key in NOISE_U_ORDER:
        expected_u = 1.5 if u_key == "u1p5" else 8.0
        for level in ("low", "high"):
            execution_id = f"pure_hubbard_page12_fullnoise__{u_key}__{level}"
            pending = pending_r30_jobs.get(execution_id)
            if pending is None:
                raise UpdateError(
                    f"pending r30 noise job coverage drifted: {execution_id}"
                )
            _, pending_job = pending
            if (
                float(pending_job.get("u_over_t")) != expected_u
                or pending_job.get("noise_level_id") != level
                or pending_job.get("target_horizon") != 30
                or pending_job.get("candidate_representation")
                != "single_pauli_word_v1"
            ):
                raise UpdateError(
                    f"pending r30 noise job identity drifted: {execution_id}"
                )
    cells = []
    for row_index, u_key in enumerate(NOISE_U_ORDER):
        expected_u = 1.5 if u_key == "u1p5" else 8.0
        for column_index, level in enumerate(NOISE_LEVEL_ORDER):
            execution_id = f"pure_hubbard_page12_fullnoise__{u_key}__{level}"
            target_horizon = 50 if level == "extreme" else 20
            jobs = jobs_by_horizon[target_horizon]
            if execution_id not in jobs:
                raise UpdateError(f"noise job coverage drifted: {execution_id}")
            job_path, job = jobs[execution_id]
            if (
                float(job.get("u_over_t")) != expected_u
                or job.get("noise_level_id") != level
                or job.get("target_horizon") != target_horizon
                or job.get("candidate_representation") != "single_pauli_word_v1"
            ):
                raise UpdateError(f"noise job identity drifted: {execution_id}")
            archive_spec = NOISE_ARCHIVES[(u_key, level)]
            if int(archive_spec["target_horizon"]) != target_horizon:
                raise UpdateError(f"noise archive horizon drifted: {execution_id}")
            cluster_id = (
                NOISE_R50_CLUSTER_ID
                if target_horizon == 50
                else NOISE_R20_CLUSTER_ID
            )
            retrieval_root = (
                NOISE_R50_RETRIEVED
                if target_horizon == 50
                else NOISE_R20_RETRIEVED
            )
            completed = _archive_result(
                path=retrieval_root / str(archive_spec["filename"]),
                expected=archive_spec,
                cluster_id=cluster_id,
                job_path=job_path,
                job=job,
            )
            cells.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "u_key": u_key,
                    "u_over_t": expected_u,
                    "noise_level_id": level,
                    "noise_tuple": copy.deepcopy(job["noise_tuple"]),
                    "noise_tuple_order": copy.deepcopy(job["noise_tuple_order"]),
                    "target_horizon": target_horizon,
                    "source_cluster_id": cluster_id,
                    "result": completed,
                    "status": "completed_authenticated_chtc_archive",
                    "job": binding(job_path),
                }
            )
    unsigned = {
        "schema": "paper_i_pure_hubbard_page12_noise_page15_adapter_v2",
        "page_id": PAGE15_ID,
        "status": "completed_6_of_6_mixed_horizon",
        "paper_evidence_adopted": False,
        "source_packages": {
            "low_high_r20": {
                "package_manifest": binding(
                    NOISE_R20_PACKAGE / "package_manifest.json"
                ),
                "package_manifest_canonical_sha256": NOISE_R20_PACKAGE_SHA256,
                "cluster_id": NOISE_R20_CLUSTER_ID,
                "target_horizon": 20,
                "noise_levels": ["low", "high"],
            },
            "extreme_r50": {
                "package_manifest": binding(
                    NOISE_R50_PACKAGE / "package_manifest.json"
                ),
                "package_manifest_canonical_sha256": NOISE_R50_PACKAGE_SHA256,
                "cluster_id": NOISE_R50_CLUSTER_ID,
                "target_horizon": 50,
                "noise_levels": ["extreme"],
            },
        },
        "terminal_horizon_by_noise_level": {
            "low": 20,
            "high": 20,
            "extreme": 50,
        },
        "pending_low_high_extension": {
            "status": "canceled_r30_not_required_after_completed_r20_review",
            "package_manifest": binding(
                PENDING_NOISE_R30_PACKAGE / "package_manifest.json"
            ),
            "package_manifest_canonical_sha256": (
                PENDING_NOISE_R30_PACKAGE_SHA256
            ),
            "cluster_id": PENDING_NOISE_R30_CLUSTER_ID,
            "target_horizon": 30,
            "noise_levels": ["low", "high"],
            "completed_page15_cells_replaced": False,
            "scheduler_disposition": "removed_20260812",
            "reason": (
                "user_declared_completed_k20_low_high_and_k50_extreme_"
                "evidence_sufficient"
            ),
        },
        "cells": cells,
        "limitations": [
            (
                "the displayed matrix has mixed terminal horizons: low/high "
                "cells end at controller round 20 and extreme cells end at "
                "round 50"
            ),
            (
                "the low/high round-30 extension on cluster 9644468 was "
                "canceled after the completed round-20 evidence was accepted "
                "as sufficient"
            ),
            (
                "this pure-Hubbard noise matrix is a Page-12 route application "
                "and is not a Hubbard--Holstein six-regime replacement"
            ),
        ],
    }
    unsigned["sha256"] = _canonical_sha256(unsigned)
    return unsigned


def format_error(value: float) -> str:
    return f"{value:.2e}"


def format_s_alg(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_cost_tuple(value: Mapping[str, Any]) -> str:
    return "(" + ",".join(
        format_s_alg(int(value[field])) if field == "S_alg" else str(int(value[field]))
        for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
    ) + ")"


def _save_page(fig: Any, *, png_path: Path, pdf_path: Path) -> None:
    from PIL import Image

    token = uuid.uuid4().hex
    temporary_png = png_path.with_name(f".{png_path.name}.{token}.tmp.png")
    temporary_pdf = pdf_path.with_name(f".{pdf_path.name}.{token}.tmp.pdf")
    try:
        fig.savefig(temporary_png, dpi=240, bbox_inches="tight")
        with Image.open(temporary_png) as source:
            source.convert("RGB").save(
                temporary_pdf,
                format="PDF",
                resolution=240.0,
            )
        os.replace(temporary_png, png_path)
        os.replace(temporary_pdf, pdf_path)
    except BaseException:
        temporary_png.unlink(missing_ok=True)
        temporary_pdf.unlink(missing_ok=True)
        raise


def render_beam_page(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _atomic_json(PAGE14_ADAPTER, adapter)
    mpl.rcParams.update({"font.family": "serif", "font.size": 7.2})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.58),
        hspace=0.34,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        for source, color, width in (
            (cell["conventional_unwhitened_adapt"], BLUE, 1.25),
            (cell["page13_unpruned_route"], GREEN, 1.55),
        ):
            points = [row for row in source["points"] if int(row["k"]) <= 20]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=color,
                lw=width,
            )
            axis.scatter(
                [points[-1]["k"]],
                [max(float(points[-1]["error"]), PLOT_FLOOR)],
                color=color,
                s=20,
                zorder=4,
            )
        beam = cell["beam_metric_route"]
        if beam is not None:
            points = beam["points"]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=ORANGE,
                lw=1.9,
            )
            axis.scatter(
                [beam["terminal"]["k"]],
                [max(float(beam["terminal"]["error"]), PLOT_FLOOR)],
                color=ORANGE,
                marker="D",
                s=30,
                zorder=5,
            )
            axis.text(
                0.97,
                0.07,
                rf"beam complete: $k=20$" "\n" + rf"$|\Delta E|={format_error(beam['terminal']['error'])}$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                color=ORANGE,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        else:
            axis.text(
                0.5,
                0.12,
                "beam/pruning cell not completed",
                transform=axis.transAxes,
                ha="center",
                fontsize=6.7,
                color=ORANGE,
                bbox={"facecolor": "white", "edgecolor": ORANGE, "alpha": 0.82},
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 20)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)",
            fontsize=8.3,
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.25, marker="o", markersize=4, label="Conventional unwhitened ADAPT"),
            Line2D([0], [0], color=GREEN, lw=1.55, marker="o", markersize=4, label="Page-13 route (no beam/pruning)"),
            Line2D([0], [0], color=ORANGE, lw=1.9, marker="D", markersize=4, label="3-branch beam + metric pruning"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Page-13 macro Phase-0 route: 3-branch beam and metric pruning (round 20)",
        fontsize=10.6,
        fontweight="bold",
        y=0.988,
    )
    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        beam = cell["beam_metric_route"]
        if beam is None:
            continue
        base20 = cell["page13_unpruned_route"]["points"][19]
        adapt20 = cell["conventional_unwhitened_adapt"]["points"][19]
        rows.append(
            [
                cell["regime_label"],
                format_error(float(beam["terminal"]["error"])),
                format_error(float(base20["error"])),
                format_error(float(adapt20["error"])),
                str(int(beam["terminal"]["active_ansatz_depth"])),
                format_cost_tuple(beam["costs"]),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            r"beam $|\Delta E_{20}|$",
            r"Page-13 $|\Delta E_{20}|$",
            r"ADAPT $|\Delta E_{20}|$",
            "beam depth",
            r"beam $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.14, 0.13, 0.14, 0.13, 0.10, 0.27),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    table.scale(1.0, 0.9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.text(
        0.5,
        0.017,
        r"Tuple uses the shared locked Table-I Qiskit compiler; $S_{alg}$ uses X.YeZ notation.",
        ha="center",
        fontsize=6.5,
    )
    _save_page(fig, png_path=PAGE14_PNG, pdf_path=PAGE14_PDF)
    plt.close(fig)


def _noise_tuple_text(values: Sequence[Any]) -> str:
    return "(" + ", ".join(f"{float(value):.0e}" for value in values) + ")"


def render_noise_page(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _atomic_json(PAGE15_ADAPTER, adapter)
    mpl.rcParams.update({"font.family": "serif", "font.size": 7.4})
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.80,
        bottom=0.11,
        wspace=0.17,
    )

    style_by_level = {
        "low": {"color": GREEN, "marker": "o", "label": r"low noise ($k=20$)"},
        "high": {"color": ORANGE, "marker": "s", "label": r"high noise ($k=20$)"},
        "extreme": {
            "color": PURPLE,
            "marker": "D",
            "label": r"extreme noise ($k=20$)",
        },
    }
    cells_by_key = {
        (str(cell["u_key"]), str(cell["noise_level_id"])): cell
        for cell in adapter["cells"]
    }
    for column, u_key in enumerate(NOISE_U_ORDER):
        axis = axes[column]
        for level in NOISE_LEVEL_ORDER:
            cell = cells_by_key[(u_key, level)]
            result = cell["result"]
            style = style_by_level[level]
            if result is None:
                raise UpdateError(
                    f"two-panel noise overlay requires a completed result: {u_key}/{level}"
                )
            points = [row for row in result["points"] if int(row["k"]) <= 20]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=style["color"],
                lw=1.9,
                label=style["label"],
            )
            axis.scatter(
                [points[-1]["k"]],
                [max(float(points[-1]["error"]), PLOT_FLOOR)],
                color=style["color"],
                marker=style["marker"],
                s=34,
                zorder=5,
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 20)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(NOISE_U_LABELS[u_key], fontsize=9.2)
        axis.set_xlabel("ADAPT iteration")
    axes[0].set_ylabel(r"same-cutoff $|\Delta E|$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style_by_level[level]["color"],
            lw=1.9,
            marker=style_by_level[level]["marker"],
            markersize=4.5,
            label=style_by_level[level]["label"],
        )
        for level in NOISE_LEVEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Pure-Hubbard Page-12 route under synthetic noise",
        fontsize=10.8,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.855,
        r"Noise tuple order $(\sigma_E,p_1,p_2,\epsilon_1,\epsilon_2)$: "
        + r"low $(1e{-}6,1e{-}8,1e{-}7,2e{-}4,6e{-}4)$; "
        + r"high $(7e{-}5,1e{-}6,1e{-}5,2e{-}3,6e{-}3)$; "
        + r"extreme $(1e{-}2,1e{-}3,1e{-}2,6e{-}2,6e{-}2)$.",
        ha="center",
        fontsize=6.8,
    )

    fig.text(
        0.5,
        0.045,
        r"All three noise levels are displayed through the common prefix $k=20$.",
        ha="center",
        fontsize=6.5,
    )
    _save_page(fig, png_path=PAGE15_PNG, pdf_path=PAGE15_PDF)
    plt.close(fig)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def append_or_replace_pages(
    beam_adapter: Mapping[str, Any],
    noise_adapter: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    declared = provenance.get("outputs", {}).get("partial_progress_pdf")
    layout = provenance.get("layout")
    if not isinstance(declared, Mapping) or not isinstance(layout, Mapping):
        raise UpdateError("report provenance is incomplete")
    page_count = int(layout.get("page_count", -1))
    if (
        current["sha256"] != declared.get("sha256")
        or current["size_bytes"] != declared.get("size_bytes")
        or page_count < 15
        or layout.get("page_13")
        != "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"
        or (
            layout.get("page_14") != PAGE14_ID
            or layout.get("page_15") not in (PAGE15_PRIOR_ID, PAGE15_ID)
        )
        or any(f"page_{page_number}" not in layout for page_number in range(16, page_count + 1))
    ):
        raise UpdateError(
            "target PDF/provenance is not a supported page-15 replacement state"
        )
    original = PdfReader(str(TARGET_PDF), strict=False)
    page14 = PdfReader(str(PAGE14_PDF), strict=False)
    page15 = PdfReader(str(PAGE15_PDF), strict=False)
    if (
        len(original.pages) != page_count
        or len(page14.pages) != 1
        or len(page15.pages) != 1
    ):
        raise UpdateError(
            "combined report requires its declared one-page replacement inputs"
        )
    preserved_prefix_hashes = [
        _page_content_sha256(page) for page in original.pages[:13]
    ]
    preserved_suffix_hashes = [
        _page_content_sha256(page) for page in original.pages[15:]
    ]
    writer = PdfWriter()
    for page in original.pages[:13]:
        writer.add_page(page)
    writer.add_page(page14.pages[0])
    writer.add_page(page15.pages[0])
    for page in original.pages[15:]:
        writer.add_page(page)
    page_count_after = page_count

    token = uuid.uuid4().hex
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.{token}.tmp"
    )
    rollback_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.rollback")
    rollback_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.{token}.rollback"
    )
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        combined_reader = PdfReader(str(temporary_pdf), strict=False)
        if len(combined_reader.pages) != page_count_after:
            raise UpdateError("combined report page count drifted")
        if [
            _page_content_sha256(page) for page in combined_reader.pages[:13]
        ] != preserved_prefix_hashes:
            raise UpdateError("Page-14 update changed a preserved leading page")
        if [
            _page_content_sha256(page) for page in combined_reader.pages[15:]
        ] != preserved_suffix_hashes:
            raise UpdateError("Page-15 update changed a later report page")

        updated = copy.deepcopy(dict(provenance))
        updated["layout"]["page_14"] = PAGE14_ID
        updated["layout"]["page_count"] = page_count_after
        updated["macro_phase0_beam_metric_progress"] = {
            "schema": "paper_i_macro_phase0_beam_metric_progress_report_v1",
            "page_id": PAGE14_ID,
            "status": beam_adapter["status"],
            "paper_evidence_adopted": False,
            "supersedes": copy.deepcopy(beam_adapter["supersedes"]),
            "adapter": {
                **binding(PAGE14_ADAPTER),
                "canonical_sha256": beam_adapter["sha256"],
            },
            "cells": copy.deepcopy(beam_adapter["cells"]),
            "limitations": copy.deepcopy(beam_adapter["limitations"]),
            "outputs": {
                "page_pdf": binding(PAGE14_PDF),
                "page_png": binding(PAGE14_PNG),
            },
        }
        updated["layout"]["page_15"] = PAGE15_ID
        updated["pure_hubbard_page12_noise_progress"] = {
            "schema": (
                "paper_i_pure_hubbard_page12_noise_progress_report_v2"
            ),
            "page_id": PAGE15_ID,
            "status": noise_adapter["status"],
            "paper_evidence_adopted": False,
            "source_packages": copy.deepcopy(noise_adapter["source_packages"]),
            "terminal_horizon_by_noise_level": copy.deepcopy(
                noise_adapter["terminal_horizon_by_noise_level"]
            ),
            "pending_low_high_extension": copy.deepcopy(
                noise_adapter["pending_low_high_extension"]
            ),
            "adapter": {
                **binding(PAGE15_ADAPTER),
                "canonical_sha256": noise_adapter["sha256"],
            },
            "cells": copy.deepcopy(noise_adapter["cells"]),
            "limitations": copy.deepcopy(noise_adapter["limitations"]),
            "outputs": {
                "page_pdf": binding(PAGE15_PDF),
                "page_png": binding(PAGE15_PNG),
            },
        }
        for key, value in (
            ("macro_phase0_beam_metric_page14_pdf", PAGE14_PDF),
            ("macro_phase0_beam_metric_page14_png", PAGE14_PNG),
            ("macro_phase0_beam_metric_page14_adapter", PAGE14_ADAPTER),
        ):
            updated["outputs"][key] = binding(value)
        for key, value in (
            ("pure_hubbard_page12_noise_page15_pdf", PAGE15_PDF),
            ("pure_hubbard_page12_noise_page15_png", PAGE15_PNG),
            ("pure_hubbard_page12_noise_page15_adapter", PAGE15_ADAPTER),
        ):
            updated["outputs"][key] = binding(value)
        combined = binding(temporary_pdf)
        combined["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(TARGET_PDF, rollback_pdf)
        os.link(TARGET_PROVENANCE, rollback_provenance)
        os.replace(temporary_pdf, TARGET_PDF)
        try:
            os.replace(temporary_provenance, TARGET_PROVENANCE)
        except BaseException:
            os.replace(rollback_pdf, TARGET_PDF)
            os.replace(rollback_provenance, TARGET_PROVENANCE)
            raise
        rollback_pdf.unlink(missing_ok=True)
        rollback_provenance.unlink(missing_ok=True)
    except BaseException:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        rollback_provenance.unlink(missing_ok=True)
        raise
    return {
        "status": "updated_existing_report_in_place",
        "page_count": page_count_after,
        "pdf": binding(TARGET_PDF),
        "provenance": binding(TARGET_PROVENANCE),
        "page_14_completed_cells": sum(
            cell["beam_metric_route"] is not None for cell in beam_adapter["cells"]
        ),
        "page_15_completed_cells": sum(
            cell["result"] is not None for cell in noise_adapter["cells"]
        ),
    }


def main() -> int:
    provenance = load(TARGET_PROVENANCE)
    beam = build_beam_adapter()
    render_beam_page(beam)
    noise = build_noise_adapter()
    render_noise_page(noise)
    result = append_or_replace_pages(beam, noise, provenance)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
