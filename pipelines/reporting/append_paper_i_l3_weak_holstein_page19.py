#!/usr/bin/env python3
"""Append the authenticated L=3 weak-Holstein progress page to the master PDF.

The three completed conventional-Append summaries are represented by a compact
projection authenticated to their remote CHTC archive hashes.  The matching
Page-12 RA cells remain explicitly pending.  This reporting-only command does
not contact CHTC or change scheduler state.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
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
STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "l3_weak_holstein_append_k50_page19"
)
PLOT_PNG = REPORT_DIR / f"{STEM}_plot.png"
PAGE_TEX = REPORT_DIR / f"{STEM}.tex"
ADAPTER_PATH = REPORT_DIR / f"{STEM}_adapter.json"
APPEND_RECEIPT = REPORT_DIR / f"{STEM}_append_receipt.json"
MUTATION_LOCK = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_mutation.lock"
)

PACKAGE_ID = "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4_chtc"
PACKAGE_DIR = (
    REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727" / PACKAGE_ID
)
SUBMISSION_RECEIPT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_"
    "v4_chtc_submission_receipt_9651011.json"
)
CLUSTER_ID = 9651011
PAGE_ID = "l3_weak_holstein_page12_vs_append_k50_progress_v1"
PAGE_SCHEMA = "paper_i_l3_weak_holstein_append_progress_adapter_v1"
REPORT_SCHEMA = "paper_i_l3_weak_holstein_append_page19_master_append_v1"
RECEIPT_SCHEMA = "paper_i_l3_weak_holstein_append_page19_append_receipt_v1"

REGIME_ORDER = ("weak_weak", "intermediate_weak", "strong_weak_u8")
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
}
REGIME_U = {"weak_weak": 0.25, "intermediate_weak": 1.25, "strong_weak_u8": 8.0}
EXPECTED_ARCHIVES = {
    "weak_weak": {
        "proc_id": 1,
        "sha256": "ae0626ca311c0e88d988ea903dfd2717cd2d9ef742c2115bb391e645a5100565",
        "size_bytes": 16_525_550,
    },
    "intermediate_weak": {
        "proc_id": 3,
        "sha256": "eb6aca02112ea4eb6bef2cbdd5107639faccc275dd06ce86f6cdc0943b501fab",
        "size_bytes": 16_464_581,
    },
    "strong_weak_u8": {
        "proc_id": 5,
        "sha256": "febaed1712381b4df03aba88aade0cecea70535f45647920267fe24ecbcf552b",
        "size_bytes": 17_368_623,
    },
}
RA_PENDING_PROCS = {"weak_weak": 0, "intermediate_weak": 2, "strong_weak_u8": 4}
SCHEDULER_OBSERVED_AT = "2026-08-13T19:13:10-05:00"
PLOT_FLOOR = 1.0e-16

# Gzip-compressed canonical JSON.  Each row is a lossless projection of one
# source-locked summary: all k=0..50 same-cutoff errors, terminal Qiskit tuple,
# S_alg, source member path, and the complete remote archive binding.
_PROJECTED_DATA_B64 = (
    "H4sIAAAAAAAC/62XTVPbSBBA/4vOXjHTHzM9nHPeyx72QFGqkTQGJyBY2yRLUvnv25IhgOmhILUn20+y5GdX9XOf/WjydrjcfC3NaXOF3beSv3SXN1e7fdlMXbeZ9mV7XcZN3pfDsW66vcSuy7e3ZRq74Wb6Wqb95mbKV93d9O1ysy9TGbsuBfbO+67Ddp+37cX3ZvV4o253mYGD3q/0IQ/ZgfdQMumrvqxh6MdR3xsDpnUehgEij6MLQ5GwDsM4uETYs/Pr3D+/5uZ76fr7fdk1pz5QIBa/asq/edh327Iu2zINpdPPtr24b07/cG2I5CNzCJQkREer5vZGbfXtZ2duhS3B8+PnqzOvEL0Xhx6dOBA9qhgU+6SfyYU0f2wnXinOVCKCUIqSgBzO16AZRwYI6FyIMQLPmBdMERk9eQB/uHQ4YCBC/RJCCHpAcbSxvMZKk0m9s7G3MdgYbUw2ZhsHG0cb25LetgTbEmxLsC3BtgTbEmxLsC3BtgTbEmxLtC3RtkTbEm1LtC3RtkTbEm1LtC3RtiTbkmxLsi3JtiTbkmxLsi3JtiTbkmxLtizPV822XGyu5wn9ah7rCNzdXV/n7X13Xa77stWT2pPt3bQ7+d+m+cnDHR4f28+7m0lvPF9so6c2pz+aT/BPcwqr5tOg89ezPvtzJjpW/+ry1YU+Y4hh1fzt5/NY+Viu9rlTqVfGOrcf5jS07HRQSgxex2XiVfOlOWX38+fq7Wzt9tub6eJA7+TDzeI3mrUufS6jjx5QfE/j2mHus0jOY3FDGUqOjpHXxIFiAgchrgtQGfphzQzVZkUMEnTYVJvl2picjzFhEqe3gKNkSRtdEgAX3NyjIEuzpOUUiVIAYochLoNEKQm7GL2IuChRlmQpjaBnp+Q5oU+yFEtpcBIhauOisD8Ea6H6DcxvB0gpLL1SykmSE/DqTcRLro4oLLE6gp6XWB3TQ6uOqJNDq0wMNkYbk43ZxsHG0cZi42RisC3BtgTbEmxLsC3BtgTbEmxLsC3BtkTbEm1LtC3RtkTbEm1LtC3RtkTbEm1Lsi3JtiTbkmxLsi3JtiTbkmxLsi3JtmTL8nmrXg7hD4Xqd+b3uysl4TFTc5CWTCV41qnk0kOnUH/3p069sn3qVGqBn3WK3tup5eVvrVX+jUTl4gKEIeu2MrgiMiaRkpPDcT1C9HEYYUxlrZN40NWL+x6TL7oxZb2248DVtYqBWX/2+lrlW9BVR1KMIpBAF5qXjYI2xRfHcYkUtBK0qBFJdyukKGGpFLTaItR4uajpCUJLpZRiBFIRB6Br1GGvUgrsybEuZ05Qt7ElUzPWZTAxhPkPiP6FWDq1YM/C2tIUdR0LS6iOMR/2qlcYllaZ2LsK9xUOFY4VThXOFR4qPFZ4RddXfKHiCxVfqPhCxRcqvlDxhYovVHyh4gsVX6z4YsUXK75Y8cWKL1Z8seKLFV+s+GLFlyq+VPGlii9VfKniSxVfqvhSxZcqvlTxZdP3edZ+zewPFe2Dk/7dMUvuIWY6Co2YoYNfMdOf5ilmryWfauaPti6dFY89O/8PpJGiozAUAAA="
)


class Page19Error(ValueError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Page19Error(f"unsafe or missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Page19Error(f"cannot load JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Page19Error(f"JSON object required: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise Page19Error(f"{label}: self digest drifted")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _projected_rows() -> list[dict[str, Any]]:
    payload = gzip.decompress(base64.b64decode(_PROJECTED_DATA_B64))
    rows = json.loads(payload)
    if not isinstance(rows, list) or len(rows) != 3:
        raise Page19Error("embedded summary projection is malformed")
    return rows


def build_adapter() -> dict[str, Any]:
    package_manifest = _load(PACKAGE_DIR / "package_manifest.json")
    submission = _load(SUBMISSION_RECEIPT)
    _verify_self_digest(package_manifest, label="package manifest")
    _verify_self_digest(submission, label="submission receipt")
    if (
        package_manifest.get("package_id") != PACKAGE_ID
        or submission.get("package_id") != PACKAGE_ID
        or submission.get("submission", {}).get("cluster_id") != CLUSTER_ID
    ):
        raise Page19Error("L=3 package/submission identity drifted")

    rows_by_regime = {str(row.get("regime")): row for row in _projected_rows()}
    if set(rows_by_regime) != set(REGIME_ORDER):
        raise Page19Error("summary projection regime set drifted")
    cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        row = rows_by_regime[regime]
        expected_archive = EXPECTED_ARCHIVES[regime]
        if (
            row.get("archive_sha256") != expected_archive["sha256"]
            or row.get("archive_size_bytes") != expected_archive["size_bytes"]
        ):
            raise Page19Error(f"{regime}: archive binding drifted")
        points = row.get("points")
        if (
            not isinstance(points, list)
            or len(points) != 51
            or [point[0] for point in points] != list(range(51))
            or not all(
                isinstance(point, list)
                and len(point) == 2
                and math.isfinite(float(point[1]))
                and float(point[1]) >= 0.0
                for point in points
            )
        ):
            raise Page19Error(f"{regime}: k=0..50 trajectory is malformed")
        terminal = row.get("terminal")
        if not isinstance(terminal, dict) or terminal.get("k") != 50:
            raise Page19Error(f"{regime}: terminal record is malformed")
        if not math.isclose(
            float(points[-1][1]),
            float(terminal.get("delta_e")),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise Page19Error(f"{regime}: trajectory/terminal mismatch")
        source_contract_path = (
            PACKAGE_DIR / "source_authority" / f"{regime}_application_source_contract.json"
        )
        source_contract = _load(source_contract_path)
        _verify_self_digest(source_contract, label=f"{regime} application source")
        exact = source_contract.get("same_cutoff_exact_reference")
        if (
            not isinstance(exact, dict)
            or exact.get("n_ph_max") != 3
            or not math.isclose(
                float(exact.get("energy")),
                float(row.get("exact_reference_energy")),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ):
            raise Page19Error(f"{regime}: same-cutoff exact reference drifted")
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "u": REGIME_U[regime],
                "nph": 3,
                "exact_reference": copy.deepcopy(exact),
                "append": {
                    "method": "conventional_unwhitened_adapt",
                    "status": "completed_exact_k50_exit_0",
                    "cluster_id": CLUSTER_ID,
                    "proc_id": expected_archive["proc_id"],
                    "archive": {
                        "remote_directory": (
                            "/staging/jsstrobel/"
                            "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4"
                        ),
                        "name": row["archive"],
                        "sha256": row["archive_sha256"],
                        "size_bytes": row["archive_size_bytes"],
                        "summary_member": row["summary_member"],
                    },
                    "points": [
                        {"k": int(point[0]), "error": float(point[1])}
                        for point in points
                    ],
                    "terminal": copy.deepcopy(terminal),
                },
                "page12_ra": {
                    "method": "global_singleton_gradient_phase0_page12",
                    "status": "idle_zero_starts",
                    "cluster_id": CLUSTER_ID,
                    "proc_id": RA_PENDING_PROCS[regime],
                    "trajectory_available": False,
                },
                "application_source_contract": _binding(source_contract_path),
            }
        )

    unsigned: dict[str, Any] = {
        "schema": PAGE_SCHEMA,
        "page_id": PAGE_ID,
        "status": "append_completed_3_of_3_ra_pending_3_of_3",
        "classification": "partial_progress_not_adopted_paper_evidence",
        "paper_evidence_adopted": False,
        "model_manifest": {
            "family": "Hubbard--Holstein",
            "num_sites": 3,
            "boundary": "open",
            "n_ph_max": 3,
            "boson_encoding": "binary",
            "fermion_ordering": "blocked",
            "sector_num_particles": [2, 1],
            "omega0": 1.0,
            "g_ep": 0.3535533905932738,
            "t": 1.0,
            "candidate_representation": "single_pauli_word_v1",
            "executable_pool_count": 314,
            "append_selector": "largest_absolute_commutator_gradient_v1",
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "adapt_seed": 7,
            "transpiler_seed": 7,
            "target_horizon": 50,
        },
        "package": {
            **_binding(PACKAGE_DIR / "package_manifest.json"),
            "canonical_sha256": package_manifest["sha256"],
            "package_id": PACKAGE_ID,
        },
        "submission_receipt": {
            **_binding(SUBMISSION_RECEIPT),
            "canonical_sha256": submission["sha256"],
        },
        "scheduler_observation": {
            "observed_at": SCHEDULER_OBSERVED_AT,
            "cluster_id": CLUSTER_ID,
            "append": {"completed": 3, "exit_zero": 3},
            "page12_ra": {"idle": 3, "num_job_starts_zero": 3},
        },
        "cost_tuple_order": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
        "cells": cells,
        "limitations": [
            "The Page-12 RA cells have zero starts, so this page reports no RA trajectory.",
            "The complete archives remain source-locked on CHTC; this page uses the authenticated compact summary projections and records each full archive SHA-256 and size.",
            "This L=3 candidate comparison is partial progress and is not adopted Paper-I evidence.",
        ],
    }
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _format_error_tex(value: float) -> str:
    if value == 0.0:
        return r"$0.0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"


def _format_s_alg(value: int) -> str:
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.1f}e{exponent}"


def _render_plot(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"font.family": "serif", "font.size": 8.0})
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))
    for axis, cell in zip(axes, adapter["cells"], strict=True):
        points = cell["append"]["points"]
        x = [int(point["k"]) for point in points]
        y = [max(float(point["error"]), PLOT_FLOOR) for point in points]
        axis.plot(x, y, color="#4C78A8", linewidth=2.0)
        axis.scatter([50], [y[-1]], color="#4C78A8", marker="o", s=28, zorder=4)
        axis.set_yscale("log")
        axis.set_xlim(0, 50)
        axis.grid(True, which="major", alpha=0.24, linewidth=0.55)
        axis.set_title(f"{cell['regime_label']} ($U={cell['u']:g}$)", fontsize=9.2)
        axis.set_xlabel("ADAPT controller round $k$")
        axis.text(
            0.5,
            0.08,
            "Page-12 RA: idle (0 starts)",
            transform=axis.transAxes,
            ha="center",
            color="#D55E00",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "#D55E00", "alpha": 0.9},
        )
    axes[0].set_ylabel(r"same-cutoff $|\Delta E|$")
    figure.suptitle(
        "L=3 weak-Holstein singleton pool: conventional unwhitened ADAPT",
        fontsize=11.0,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    temporary = PLOT_PNG.with_name(f".{PLOT_PNG.name}.tmp.png")
    figure.savefig(temporary, dpi=220, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, PLOT_PNG)


def _write_tex(adapter: Mapping[str, Any]) -> None:
    rows = []
    for cell in adapter["cells"]:
        terminal = cell["append"]["terminal"]
        cost_tuple = (
            f"({terminal['N2q']},{terminal['D2q']},{terminal['Dc']},"
            f"{terminal['W1q']},{_format_s_alg(int(terminal['S_alg']))})"
        )
        rows.append(
            " & ".join(
                (
                    cell["regime_label"],
                    f"${cell['u']:g}$",
                    f"${cell['exact_reference']['energy']:.9f}$",
                    f"${terminal['energy']:.9f}$",
                    _format_error_tex(float(terminal["delta_e"])),
                    rf"$\mathrm{{{cost_tuple}}}$",
                    f"proc {cell['page12_ra']['proc_id']}: idle, 0 starts",
                )
            )
            + r" \\"
        )
    plot_path = PLOT_PNG.resolve().as_posix()
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.30in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\tabcolsep}}{{3.1pt}}
\begin{{document}}
\begin{{center}}
{{\large\bfseries L=3 weak-Holstein progress: conventional Append complete; Page-12 RA pending}}\\[-0.1em]
{{\scriptsize Hubbard--Holstein; $L=3$; open boundary; $n_{{ph}}=3$; binary bosons; blocked fermions;
$(N_\alpha,N_\beta)=(2,1)$; $t=\omega_0=1$; $g=0.353553391$; 314 guarded singleton Pauli words;
Powell-200; ADAPT/transpiler seed 7; target $k=50$.}}

\includegraphics[width=0.985\textwidth,height=4.05in,keepaspectratio]{{\detokenize{{{plot_path}}}}}
\vspace{{-0.55em}}

\scriptsize
\resizebox{{0.99\textwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrll@{{}}}}
\toprule
Regime & $U$ & same-cutoff $E_{{\rm exact}}$ & $E_{{50}}$ & $|\Delta E_{{50}}|$ &
$C_{{50}}=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$ & matched Page-12 RA \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{center}}
\vspace{{-0.35em}}
\footnotesize
Blue trajectories are the three completed conventional unwhitened Append-ADAPT cells in CHTC cluster
{CLUSTER_ID} (procs 1, 3, and 5; exit 0). The tuple is compiled through the shared locked Table-I
Qiskit path; $S_{{\rm alg}}$ uses X.YeZ notation. Orange status labels are not curves: matching Page-12
RA procs 0, 2, and 4 were still idle with zero starts at {SCHEDULER_OBSERVED_AT}.
This is partial L=3 candidate evidence, not an adopted Paper-I result.
\end{{document}}
"""
    temporary = PAGE_TEX.with_name(f".{PAGE_TEX.name}.tmp")
    temporary.write_text(tex, encoding="utf-8")
    os.replace(temporary, PAGE_TEX)


def _compile_page(destination: Path) -> None:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise Page19Error("latexmk is required for the paper-facing Page-19 build")
    scratch_root = REPO_ROOT / "tmp/pdfs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paper_i_l3_page19_", dir=scratch_root) as raw:
        build = Path(raw)
        completed = subprocess.run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build}",
                PAGE_TEX.name,
            ],
            cwd=PAGE_TEX.parent,
            text=True,
            capture_output=True,
            env={**os.environ, "FORCE_SOURCE_DATE": "1", "TZ": "UTC"},
        )
        if completed.returncode != 0:
            raise Page19Error(
                "Page-19 LaTeX build failed:\n"
                + completed.stdout[-3000:]
                + completed.stderr[-3000:]
            )
        compiled = build / f"{PAGE_TEX.stem}.pdf"
        if not compiled.is_file():
            raise Page19Error("LaTeX completed without a Page-19 PDF")
        shutil.copyfile(compiled, destination)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    payload = b"" if contents is None else contents.get_data()
    return hashlib.sha256(payload).hexdigest()


def append_page(adapter: Mapping[str, Any]) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    _render_plot(adapter)
    _write_tex(adapter)
    _atomic_json(ADAPTER_PATH, adapter)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MUTATION_LOCK.touch(exist_ok=True)
    with MUTATION_LOCK.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        provenance = _load(TARGET_PROVENANCE)
        current_pdf = _binding(TARGET_PDF)
        declared_pdf = provenance.get("outputs", {}).get("partial_progress_pdf")
        layout = provenance.get("layout")
        if not isinstance(declared_pdf, dict) or not isinstance(layout, dict):
            raise Page19Error("master provenance is incomplete")
        if (
            declared_pdf.get("sha256") != current_pdf["sha256"]
            or declared_pdf.get("size_bytes") != current_pdf["size_bytes"]
        ):
            raise Page19Error("master PDF/provenance binding drifted")
        if layout.get("page_count") == 19 and layout.get("page_19") == PAGE_ID:
            report = provenance.get("l3_weak_holstein_append_page19")
            if (
                not isinstance(report, dict)
                or report.get("adapter", {}).get("canonical_sha256")
                != adapter["sha256"]
            ):
                raise Page19Error("Page 19 already exists for different evidence")
            result = {
                "status": "already_current",
                "page_count": 19,
                "pdf": current_pdf,
                "provenance": _binding(TARGET_PROVENANCE),
            }
            if APPEND_RECEIPT.is_file() and not APPEND_RECEIPT.is_symlink():
                receipt = _load(APPEND_RECEIPT)
                _verify_self_digest(receipt, label="Page-19 append receipt")
                if (
                    receipt.get("page_id") != PAGE_ID
                    or receipt.get("pdf") != current_pdf
                    or receipt.get("adapter", {}).get("canonical_sha256")
                    != adapter["sha256"]
                ):
                    raise Page19Error("existing Page-19 append receipt drifted")
            return result
        if layout.get("page_count") != 18 or "page_19" in layout:
            raise Page19Error("master report is not the supported 18-page state")

        original = PdfReader(str(TARGET_PDF), strict=False)
        if len(original.pages) != 18:
            raise Page19Error("master report is not exactly 18 pages")
        prior_hashes = [_page_content_sha256(page) for page in original.pages]
        token = uuid.uuid4().hex
        temporary_page = REPO_ROOT / "tmp/pdfs" / f"page19-{token}.pdf"
        temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.tmp")
        temporary_provenance = TARGET_PROVENANCE.with_name(
            f".{TARGET_PROVENANCE.name}.{token}.tmp"
        )
        rollback_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.rollback")
        rollback_provenance = TARGET_PROVENANCE.with_name(
            f".{TARGET_PROVENANCE.name}.{token}.rollback"
        )
        try:
            _compile_page(temporary_page)
            page_reader = PdfReader(str(temporary_page), strict=False)
            if len(page_reader.pages) != 1:
                raise Page19Error("compiled Page 19 is not exactly one page")
            writer = PdfWriter()
            for page in original.pages:
                writer.add_page(page)
            writer.add_page(page_reader.pages[0])
            with temporary_pdf.open("xb") as stream:
                writer.write(stream)
                stream.flush()
                os.fsync(stream.fileno())
            combined = PdfReader(str(temporary_pdf), strict=False)
            combined_hashes = [_page_content_sha256(page) for page in combined.pages]
            if len(combined.pages) != 19 or combined_hashes[:18] != prior_hashes:
                raise Page19Error("Page-19 append did not preserve Pages 1--18")

            updated = copy.deepcopy(provenance)
            updated["layout"]["page_count"] = 19
            updated["layout"]["page_19"] = PAGE_ID
            updated["l3_weak_holstein_append_page19"] = {
                "schema": REPORT_SCHEMA,
                "page_id": PAGE_ID,
                "status": adapter["status"],
                "paper_evidence_adopted": False,
                "adapter": {
                    **_binding(ADAPTER_PATH),
                    "canonical_sha256": adapter["sha256"],
                },
                "model_manifest": copy.deepcopy(adapter["model_manifest"]),
                "scheduler_observation": copy.deepcopy(
                    adapter["scheduler_observation"]
                ),
                "cells": copy.deepcopy(adapter["cells"]),
                "limitations": copy.deepcopy(adapter["limitations"]),
                "structural_validation": {
                    "pages_before": 18,
                    "pages_after": 19,
                    "preserved_page_content_sha256": prior_hashes,
                    "new_page_content_sha256": combined_hashes[18],
                },
                "outputs": {
                    "plot_png": _binding(PLOT_PNG),
                    "page_tex": _binding(PAGE_TEX),
                },
            }
            updated["outputs"]["l3_weak_holstein_append_page19_adapter"] = {
                **_binding(ADAPTER_PATH),
                "canonical_sha256": adapter["sha256"],
            }
            updated["outputs"]["l3_weak_holstein_append_page19_plot_png"] = (
                _binding(PLOT_PNG)
            )
            updated["outputs"]["l3_weak_holstein_append_page19_tex"] = _binding(
                PAGE_TEX
            )
            combined_binding = _binding(temporary_pdf)
            combined_binding["path"] = str(TARGET_PDF.resolve())
            updated["outputs"]["partial_progress_pdf"] = combined_binding
            limitation = (
                "Page 19 is partial L=3 weak-Holstein progress: three conventional "
                "Append cells completed, while the three matched Page-12 RA cells "
                "had zero starts at the recorded scheduler observation."
            )
            if limitation not in updated.setdefault("limitations", []):
                updated["limitations"].append(limitation)
            with temporary_provenance.open("xb") as stream:
                stream.write(
                    json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode()
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            if _binding(TARGET_PDF) != current_pdf:
                raise Page19Error("master PDF changed during Page-19 preparation")
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
        finally:
            temporary_page.unlink(missing_ok=True)
            temporary_pdf.unlink(missing_ok=True)
            temporary_provenance.unlink(missing_ok=True)
            rollback_pdf.unlink(missing_ok=True)
            rollback_provenance.unlink(missing_ok=True)

    unsigned_receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "appended_page19_to_existing_master",
        "page_id": PAGE_ID,
        "page_count": 19,
        "preserved_page_count": 18,
        "adapter": {**_binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
        "pdf": _binding(TARGET_PDF),
        "provenance": _binding(TARGET_PROVENANCE),
        "paper_evidence_adopted": False,
    }
    receipt = {
        **unsigned_receipt,
        "sha256": _canonical_sha256(unsigned_receipt),
    }
    _atomic_json(APPEND_RECEIPT, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = append_page(build_adapter())
    except (OSError, Page19Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
