#!/usr/bin/env python3
"""Add the authenticated tau=1e-6 weak--weak confirmation to page 8.

This updater is intentionally narrow.  It preserves pages 1--7 of the
existing evolving report, keeps the six-regime tau=1e-4 comparison intact,
and replaces page 8 with a three-curve weak--weak panel plus unchanged panels
for the other five regimes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import add_paper_i_phase3_on_plateau_singleton_page as base


EXPECTED_ARCHIVE_SHA256 = (
    "586e331c824c8c3dfa0f4160f46f3e6e049f37fe4962854926bfc11e0e011886"
)
EXPECTED_PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_1em6_"
    "weak_weak_r50_20260804_v1_chtc"
)
EXPECTED_CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_"
    "sensitivity_weak_weak_r50_20260804_v1"
)
EXPECTED_EXECUTION_ID = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau"
)
EXPECTED_ROUTE_CONTRACT_SHA256 = (
    "1579b15c50eaa4daa5b7c8ab9343488634452775fb04f13bc52b5cd0dd5e2ff2"
)
EXPECTED_ACTIVATION_SHA256 = (
    "bcebe6f4c792c3669598d671f1c3f6768a769d3b529baee795b839daac958109"
)
EXPECTED_WORKER_RECEIPT_SHA256 = (
    "0047e0136e1e414baf5651abc42018f686038f1a1244cfc22e25fcade4a6c096"
)
THRESHOLD = 1.0e-6
TARGET_ROUND = 50
ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_page8_"
    "threshold_confirmation_v1"
)
PAGE_ID = (
    "ra_singleton_phase3_population_on_insertion_plateau_vs_append_"
    "r50_threshold_confirmation_v3"
)
REPORT_SCHEMA = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_"
    "page8_threshold_confirmation_report_v1"
)
OUTPUT_PREFIX = "phase3_on_plateau_singleton_page8_threshold_confirmation"


class ConfirmationError(ValueError):
    """The selected confirmation or report state is not admissible."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfirmationError(f"{label} is unavailable or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationError(f"{label} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _self_digest(value: Mapping[str, Any], *, label: str) -> str:
    return base.verify_self_digest(value, label=label)


def _artifact_binding(worker: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = base._sequence(worker.get("artifacts"), label="worker artifacts")
    matches = [
        base._mapping(row, label="worker artifact")
        for row in rows
        if base._mapping(row, label="worker artifact").get("path") == name
    ]
    if len(matches) != 1:
        raise ConfirmationError(f"worker artifact binding is not unique: {name}")
    return matches[0]


def _check_file_binding(path: Path, expected: Mapping[str, Any], *, label: str) -> None:
    observed = _binding(path)
    if (
        observed["sha256"] != expected.get("sha256")
        or observed["size_bytes"] != expected.get("size_bytes")
    ):
        raise ConfirmationError(f"{label} byte binding drifted")


def load_confirmation(*, archive: Path, extracted: Path) -> dict[str, Any]:
    if _sha256(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ConfirmationError("confirmation archive SHA-256 drifted")
    attempt = _load(extracted / "worker_attempt_receipt.json", label="attempt receipt")
    job = _load(extracted / "authority/job.json", label="job")
    authorization = _load(
        extracted / "authority/execution_authorization.json",
        label="execution authorization",
    )
    activation = _load(
        extracted / "authority/activation_manifest.json", label="activation manifest"
    )
    worker = _load(
        extracted / "worker_outputs/worker_receipt.json", label="worker receipt"
    )
    artifacts = extracted / "worker_outputs/artifacts"
    execution = _load(artifacts / "execution_manifest.json", label="execution manifest")
    summary = _load(artifacts / "paper_i_summary.json", label="Paper-I summary")
    result = _load(artifacts / "result.json", label="result")

    attempt_sha = _self_digest(attempt, label="attempt receipt")
    job_sha = _self_digest(job, label="job")
    authorization_sha = _self_digest(authorization, label="authorization")
    activation_sha = _self_digest(activation, label="activation")
    worker_sha = _self_digest(worker, label="worker receipt")
    execution_sha = _self_digest(execution, label="execution manifest")
    if (
        attempt.get("execution_id") != EXPECTED_EXECUTION_ID
        or attempt.get("cluster_id") != 9576591
        or attempt.get("proc_id") != 0
        or attempt.get("attempt_ordinal") != 1
        or attempt.get("worker_exit_status") != 0
        or attempt.get("science_evidence_state") != "success_payload_closed_v2"
        or job.get("package_id") != EXPECTED_PACKAGE_ID
        or job.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or job.get("execution_id") != EXPECTED_EXECUTION_ID
        or job.get("regime_id") != "weak_weak"
        or job.get("target_horizon") != TARGET_ROUND
        or job.get("route_contract_sha256") != EXPECTED_ROUTE_CONTRACT_SHA256
        or job.get("plateau_prior_mean_decrease_ratio_threshold") != THRESHOLD
        or job.get("plateau_threshold_comparison")
        != "marginal_to_prior_mean_strictly_below_v2"
        or job.get("insertion_policy") != "plateau_commutation"
        or activation_sha != EXPECTED_ACTIVATION_SHA256
        or activation.get("package_id") != EXPECTED_PACKAGE_ID
        or activation.get("submission_authorized") is not True
        or authorization.get("status") != "passed"
        or authorization.get("job_spec_sha256") != job_sha
        or worker_sha != EXPECTED_WORKER_RECEIPT_SHA256
        or worker.get("status") != "passed"
        or worker.get("controller_rounds_completed") != TARGET_ROUND
        or worker.get("job_spec_sha256") != job_sha
        or worker.get("authorization_sha256") != authorization_sha
        or worker.get("execution_manifest_sha256") != execution_sha
        or execution.get("status") != "passed"
        or execution.get("controller_rounds_completed") != TARGET_ROUND
        or execution.get("fresh_start") is not True
        or execution.get("source_checkpoint_consumed") is not False
    ):
        raise ConfirmationError("confirmation authority closure drifted")

    attempt_rows = {
        str(base._mapping(row, label="attempt file").get("path")): base._mapping(
            row, label="attempt file"
        )
        for row in base._sequence(attempt.get("worker_files"), label="attempt files")
    }
    for relative in (
        "worker_receipt.json",
        "artifacts/execution_manifest.json",
        "artifacts/paper_i_summary.json",
        "artifacts/result.json",
    ):
        expected = attempt_rows.get(relative)
        if expected is None:
            raise ConfirmationError(f"attempt binding is absent: {relative}")
        _check_file_binding(
            extracted / "worker_outputs" / relative,
            expected,
            label=relative,
        )
    for name in ("execution_manifest.json", "paper_i_summary.json", "result.json"):
        _check_file_binding(
            artifacts / name,
            _artifact_binding(worker, name),
            label=name,
        )

    run = base._mapping(result.get("run"), label="result run")
    transitions = base._sequence(
        run.get("accepted_transitions"), label="accepted transitions"
    )
    if len(transitions) != TARGET_ROUND:
        raise ConfirmationError("result does not contain 50 accepted transitions")
    initial_energy = base._finite(
        base._mapping(transitions[0], label="first transition").get("energy_before"),
        label="initial energy",
    )
    provenance = base._mapping(summary.get("provenance"), label="summary provenance")
    exact = base._finite(
        provenance.get("exact_same_cutoff_energy"), label="exact energy"
    )
    if (
        summary.get("schema") != base.SUMMARY_SCHEMA
        or summary.get("available_controller_rounds") != TARGET_ROUND
        or provenance.get("route_contract_sha256")
        != EXPECTED_ROUTE_CONTRACT_SHA256
        or provenance.get("candidate_representation") != "single_pauli_word_v1"
    ):
        raise ConfirmationError("summary identity drifted")
    trace = base._sequence(summary.get("accepted_error_trace"), label="error trace")
    if len(trace) != TARGET_ROUND:
        raise ConfirmationError("confirmation trace is not 50 rounds")
    points = [{"k": 0, "error": abs(initial_energy - exact)}]
    for expected_round, raw in enumerate(trace, start=1):
        row = base._mapping(raw, label="trace row")
        error = base._finite(
            row.get("absolute_energy_error"), label="trace error", minimum=0.0
        )
        energy = base._finite(row.get("accepted_energy"), label="accepted energy")
        if row.get("controller_round") != expected_round or not math.isclose(
            error, abs(energy - exact), rel_tol=1.0e-11, abs_tol=1.0e-12
        ):
            raise ConfirmationError("confirmation trace math drifted")
        points.append({"k": expected_round, "error": error})

    requested = base._sequence(summary.get("requested_rounds"), label="requested rounds")
    if len(requested) != 1:
        raise ConfirmationError("round-50 observation is not unique")
    round_50 = base._mapping(requested[0], label="round-50 observation")
    prefix = base._mapping(round_50.get("prefix"), label="round-50 prefix")
    resources = base._mapping(round_50.get("resources"), label="round-50 resources")
    compiled_payload = dict(base._compile_prefix_mapping(prefix))
    compiled = base._normalize_compiled_cost(compiled_payload)
    if (
        round_50.get("controller_round") != TARGET_ROUND
        or round_50.get("status") != "available"
        or compiled["N2q"] != resources.get("compiled_two_qubit_count")
        or compiled["D2q"] != resources.get("compiled_two_qubit_depth")
        or compiled["Dc"] != resources.get("compiled_total_depth")
    ):
        raise ConfirmationError("round-50 recompilation disagrees with the run")
    all_work = base._work(summary.get("canonical_all_work"), label="canonical work")
    plateau = base._mapping(summary.get("effective_plateau"), label="effective plateau")
    marker_round = base._integer(
        plateau.get("controller_round"), label="effective plateau round", minimum=1
    )
    marker_error = base._finite(
        plateau.get("absolute_energy_error"), label="effective plateau error", minimum=0.0
    )
    if not math.isclose(
        marker_error,
        float(points[marker_round]["error"]),
        rel_tol=1.0e-11,
        abs_tol=1.0e-12,
    ):
        raise ConfirmationError("effective plateau marker drifted")

    source_files = {
        "archive": _binding(archive),
        "attempt_receipt": {
            **_binding(extracted / "worker_attempt_receipt.json"),
            "canonical_sha256": attempt_sha,
        },
        "job": {
            **_binding(extracted / "authority/job.json"),
            "canonical_sha256": job_sha,
        },
        "authorization": {
            **_binding(extracted / "authority/execution_authorization.json"),
            "canonical_sha256": authorization_sha,
        },
        "activation_manifest": {
            **_binding(extracted / "authority/activation_manifest.json"),
            "canonical_sha256": activation_sha,
        },
        "worker_receipt": {
            **_binding(extracted / "worker_outputs/worker_receipt.json"),
            "canonical_sha256": worker_sha,
        },
        "execution_manifest": {
            **_binding(artifacts / "execution_manifest.json"),
            "canonical_sha256": execution_sha,
        },
        "summary": _binding(artifacts / "paper_i_summary.json"),
        "result": _binding(artifacts / "result.json"),
    }
    return {
        "regime_id": "weak_weak",
        "execution_id": EXPECTED_EXECUTION_ID,
        "package_id": EXPECTED_PACKAGE_ID,
        "campaign_id": EXPECTED_CAMPAIGN_ID,
        "cluster_id": 9576591,
        "proc_id": 0,
        "attempt_ordinal": 1,
        "plateau_prior_mean_decrease_ratio_threshold": THRESHOLD,
        "route_contract_sha256": EXPECTED_ROUTE_CONTRACT_SHA256,
        "points": points,
        "marker": {
            "k": marker_round,
            "error": marker_error,
            "policy": "first_effective_plateau_prefix",
        },
        "terminal": {
            "k": TARGET_ROUND,
            "error": float(points[-1]["error"]),
            **compiled,
            "S_alg": all_work["s_alg"],
            "status": "complete",
        },
        "compile_payload_sha256": hashlib.sha256(
            base.canonical_json_bytes(compiled_payload)
        ).hexdigest(),
        "exact_same_cutoff_energy": exact,
        "source_bindings": source_files,
    }


def build_adapter(
    *, base_adapter_path: Path, confirmation: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    current = base.validate_adapter(base_adapter_path)
    weak = base._mapping(current["cells"][0], label="weak--weak base cell")
    if weak.get("regime_id") != "weak_weak" or not math.isclose(
        float(weak["exact_same_cutoff_energy"]),
        float(confirmation["exact_same_cutoff_energy"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ConfirmationError("confirmation comparison space drifted")
    cells = copy.deepcopy(current["cells"])
    cells[0]["threshold_confirmation"] = copy.deepcopy(dict(confirmation))
    updated = copy.deepcopy(current)
    updated.pop("sha256", None)
    updated.update(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed_six_cells_with_append_and_weak_weak_threshold_confirmation",
            "page_id": PAGE_ID,
            "comparison_method": "Append-ADAPT and weak--weak tau=1e-6 confirmation",
            "threshold_confirmation": {
                "regime_id": "weak_weak",
                "threshold": THRESHOLD,
                "package_id": confirmation["package_id"],
                "campaign_id": confirmation["campaign_id"],
                "execution_id": confirmation["execution_id"],
                "archive": copy.deepcopy(confirmation["source_bindings"]["archive"]),
            },
            "cells": cells,
        }
    )
    updated = base.digested(updated)
    base._atomic_write_json(output, updated)
    return updated


def render_plot(adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.6,
            "font.family": "serif",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.1), constrained_layout=True)
    for index, (ax, raw_cell) in enumerate(zip(axes.flat, adapter["cells"], strict=True)):
        cell = base._mapping(raw_cell, label="plot cell")
        append = base._mapping(cell["append_adapt"], label="Append cell")
        append_points = [base._mapping(row, label="Append point") for row in append["points"]]
        ra_points = [base._mapping(row, label="RA point") for row in cell["points"]]
        ax.plot(
            [int(row["k"]) for row in append_points],
            [max(float(row["error"]), base.PLOT_FLOOR) for row in append_points],
            color="#4C78A8",
            linewidth=1.6,
        )
        ax.plot(
            [int(row["k"]) for row in ra_points],
            [max(float(row["error"]), base.PLOT_FLOOR) for row in ra_points],
            color="#009E73",
            linewidth=1.8,
        )
        marker = base._mapping(cell["marker"], label="RA marker")
        ax.scatter(
            [int(marker["k"])],
            [max(float(marker["error"]), base.PLOT_FLOOR)],
            color="#009E73",
            marker="P",
            s=40,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        confirmation = cell.get("threshold_confirmation")
        if isinstance(confirmation, Mapping):
            confirmation_points = [
                base._mapping(row, label="confirmation point")
                for row in confirmation["points"]
            ]
            ax.plot(
                [int(row["k"]) for row in confirmation_points],
                [max(float(row["error"]), base.PLOT_FLOOR) for row in confirmation_points],
                color="#D55E00",
                linewidth=1.8,
                linestyle="--",
            )
            confirmation_marker = base._mapping(
                confirmation["marker"], label="confirmation marker"
            )
            ax.scatter(
                [int(confirmation_marker["k"])],
                [max(float(confirmation_marker["error"]), base.PLOT_FLOOR)],
                color="#D55E00",
                marker="*",
                s=58,
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
        ax.set_yscale("log")
        ax.set_xlim(0, TARGET_ROUND)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(True, which="major", alpha=0.22, linewidth=0.55)
        ax.set_title(f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)")
        if index // 3 == 1:
            ax.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            ax.set_ylabel(r"same-cutoff $|\Delta E|$")
    legend = [
        Line2D([0], [0], color="#4C78A8", lw=1.6, label="Append-ADAPT"),
        Line2D([0], [0], color="#009E73", lw=1.8, marker="P", markersize=5,
               label=r"RA Phase III on plateau ($\tau=10^{-4}$)"),
        Line2D([0], [0], color="#D55E00", lw=1.8, ls="--", marker="*", markersize=7,
               label=r"Weak--weak confirmation ($\tau=10^{-6}$)"),
    ]
    fig.suptitle(
        "Singleton Phase-III-on-plateau comparison at the common round-50 horizon",
        fontsize=11.1,
        fontweight="bold",
    )
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_page_tex(adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path) -> None:
    rows: list[str] = []
    for raw in adapter["cells"]:
        cell = base._mapping(raw, label="table cell")
        terminal = base._mapping(cell["terminal"], label="RA terminal")
        append_terminal = base._mapping(
            base._mapping(cell["append_adapt"], label="Append cell")["terminal"],
            label="Append terminal",
        )
        rows.append(
            f"{base._latex_escape(str(cell['regime_label']))} ($10^{{-4}}$) & "
            f"{cell['marker']['k']} & {base._format_sci(float(terminal['error']))} & "
            f"{base._latex_escape(base._format_cost(terminal))} & "
            f"{base._format_sci(float(append_terminal['error']))} & "
            f"{base._latex_escape(base._format_cost(append_terminal))} \\\\"
        )
        confirmation = cell.get("threshold_confirmation")
        if isinstance(confirmation, Mapping):
            confirmation_terminal = base._mapping(
                confirmation["terminal"], label="confirmation terminal"
            )
            rows.append(
                f"{base._latex_escape(str(cell['regime_label']))} ($10^{{-6}}$ confirmation) & "
                f"{confirmation['marker']['k']} & "
                f"{base._format_sci(float(confirmation_terminal['error']))} & "
                f"{base._latex_escape(base._format_cost(confirmation_terminal))} & "
                f"{base._format_sci(float(append_terminal['error']))} & "
                f"{base._latex_escape(base._format_cost(append_terminal))} \\\\"
            )
    tex = rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.16in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.92\textwidth,height=3.62in,keepaspectratio]{{{plot_pdf.as_posix()}}}
\vspace{{0.08em}}

\tiny
\setlength{{\tabcolsep}}{{2.2pt}}
\resizebox{{0.985\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Regime / RA threshold & $k_{{pl}}^{{\rm RA}}$ & $|\Delta E_{{50}}^{{\rm RA}}|$ &
$C_{{50}}^{{\rm RA}}$ & $|\Delta E_{{50}}^{{\rm Append}}|$ &
$C_{{50}}^{{\rm Append}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.30em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ at controller round 50;
all errors use exact diagonalization at the identical phonon cutoff.  The green
six-regime route uses $\tau=10^{{-4}}$; the orange weak--weak confirmation is
the completed source-locked CHTC run at $\tau=10^{{-6}}$.  Both use stationary
active gradients, commutation-reduced plateau insertion, and Phase III opening
on the authenticated insertion plateau.  This page is diagnostic and is not
adopted Paper-I evidence.
\end{{document}}
""".strip()
    tex_path.write_text(tex + "\n", encoding="utf-8")


def build_assets(
    adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(adapter, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
    base._compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def replace_page8(
    *,
    target_pdf: Path,
    target_provenance: Path,
    adapter_path: Path,
    adapter: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    assets: Mapping[str, Path],
) -> dict[str, Any]:
    provenance = _load(target_provenance, label="target provenance")
    current_pdf = _binding(target_pdf)
    expected_pdf = base._mapping(
        base._mapping(provenance.get("outputs"), label="outputs").get(
            "partial_progress_pdf"
        ),
        label="PDF binding",
    )
    if (
        current_pdf["sha256"] != expected_pdf.get("sha256")
        or current_pdf["size_bytes"] != expected_pdf.get("size_bytes")
        or base._mapping(provenance.get("layout"), label="layout").get("page_count")
        != 8
    ):
        raise ConfirmationError("target PDF/provenance binding drifted")
    before_hashes = base._page_content_hashes(target_pdf)
    if len(before_hashes) != 8:
        raise ConfirmationError("target is not the supported eight-page report")
    from pypdf import PdfReader, PdfWriter

    new_page = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(new_page.pages) != 1:
        raise ConfirmationError("replacement page is not one page")
    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.threshold.tmp")
    temporary_provenance = target_provenance.with_name(
        f".{target_provenance.name}.threshold.tmp"
    )
    rollback_pdf = target_pdf.with_name(f".{target_pdf.name}.threshold.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback_pdf):
        if path.exists() or path.is_symlink():
            raise ConfirmationError(f"stale temporary exists: {path}")
    writer = PdfWriter()
    old_reader = PdfReader(str(target_pdf), strict=False)
    for page in old_reader.pages[:7]:
        writer.add_page(page)
    writer.add_page(new_page.pages[0])
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        after_hashes = base._page_content_hashes(temporary_pdf)
        if len(after_hashes) != 8 or after_hashes[:7] != before_hashes[:7]:
            raise ConfirmationError("page-8 replacement altered a preserved page")
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_8"] = PAGE_ID
        adapter_binding = {
            **_binding(adapter_path),
            "canonical_sha256": adapter["sha256"],
        }
        asset_bindings = {role: _binding(path) for role, path in assets.items()}
        report = copy.deepcopy(updated[base.REPORT_KEY])
        report.update(
            {
                "schema": REPORT_SCHEMA,
                "page_id": PAGE_ID,
                "adapter": adapter_binding,
                "comparison_method": adapter["comparison_method"],
                "threshold_confirmation": copy.deepcopy(dict(confirmation)),
                "cells": copy.deepcopy(adapter["cells"]),
                "outputs": asset_bindings,
                "structural_validation": {
                    "pages_before": 8,
                    "pages_after": 8,
                    "preserved_page_content_sha256": before_hashes[:7],
                    "previous_page_8_content_sha256": before_hashes[7],
                    "new_page_8_content_sha256": after_hashes[7],
                },
            }
        )
        updated[base.REPORT_KEY] = report
        combined = _binding(temporary_pdf)
        combined["path"] = str(target_pdf.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
        updated["outputs"][f"{OUTPUT_PREFIX}_adapter"] = adapter_binding
        for role, binding in asset_bindings.items():
            updated["outputs"][f"{OUTPUT_PREFIX}_{role}"] = binding
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(target_pdf, rollback_pdf)
        os.replace(temporary_pdf, target_pdf)
        try:
            os.replace(temporary_provenance, target_provenance)
        except Exception:
            os.replace(rollback_pdf, target_pdf)
            raise
        rollback_pdf.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        raise
    return {
        "status": "replaced_page_8_with_threshold_confirmation",
        "pages": 8,
        "preserved_pages": 7,
        "terminal_delta_e": confirmation["terminal"]["error"],
        "terminal_cost": {
            key: confirmation["terminal"][key]
            for key in ("N2q", "D2q", "Dc", "W1q", "S_alg")
        },
        "pdf_sha256": _sha256(target_pdf),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--extracted", type=Path, required=True)
    result.add_argument("--base-adapter", type=Path, required=True)
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--target-pdf", type=Path, required=True)
    result.add_argument("--target-provenance", type=Path, required=True)
    result.add_argument("--asset-dir", type=Path, required=True)
    result.add_argument("--asset-stem", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        confirmation = load_confirmation(
            archive=args.archive.resolve(), extracted=args.extracted.resolve()
        )
        adapter = build_adapter(
            base_adapter_path=args.base_adapter.resolve(),
            confirmation=confirmation,
            output=args.adapter.resolve(),
        )
        assets = build_assets(
            adapter,
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
        result = replace_page8(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            adapter=adapter,
            confirmation=confirmation,
            assets=assets,
        )
    except (ConfirmationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
