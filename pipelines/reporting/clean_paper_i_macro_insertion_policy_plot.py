#!/usr/bin/env python3
"""Build the Paper-I intact-macro four-policy appendix comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
PAGE16 = SOURCE_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_phase23_qiskit_no_lanes_page16_adapter.json"
)
POLICIES = SOURCE_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "insertion_comparator_live_snapshot_adapter.json"
)
OVERRIDE_ROOT = ROOT / (
    "output/local_runs/"
    "paper_i_page16_weak_append_user_authorized_k30_to_k50_20260813_v3"
)
OUT_DIR = ROOT / "MATH/paper_details/figures/paper_i_macro_insertion_policies_20260813"
OUTPUT = OUT_DIR / "paper_i_macro_insertion_policies.pdf"
PROVENANCE = OUT_DIR / "paper_i_macro_insertion_policies_provenance.json"

REGIMES = (
    ("weak_weak", "Weak--weak", 3),
    ("intermediate_weak", "Intermediate--weak", 3),
    ("strong_weak_u8", "Strong--weak", 3),
    ("weak_strong", "Weak--strong", 7),
    ("intermediate_strong", "Intermediate--strong", 7),
    ("strong_strong_u8", "Strong--strong", 7),
)
STRONG_HOLSTEIN = {"weak_strong", "intermediate_strong", "strong_strong_u8"}
APPEND_ARCHIVES = ROOT / (
    "raw_outputs/paper_i_ra_adapt_stationary_core_v7_partial_report_20260729"
)
CURVE_CACHE = ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/"
    "paper_i_hh_macro_common_accuracy_20260723_stationary_page1_macro_curve_cache.json"
)
APPEND_PROCS = {
    "weak_strong": 24,
    "intermediate_strong": 32,
    "strong_strong_u8": 40,
}
OVERRIDE_IDS = {
    "weak_weak": (
        "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
        "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_append_only"
    ),
    "intermediate_weak": (
        "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__nph3__"
        "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_append_only"
    ),
}


class BuildError(RuntimeError):
    """The authenticated policy-comparison inputs do not close."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _points(rows: list[dict[str, Any]], *, round_key: str = "k", error_key: str = "error") -> list[dict[str, float | int]]:
    points = [
        {"k": int(row[round_key]), "error": float(row[error_key])}
        for row in rows
    ]
    if [point["k"] for point in points] != list(range(1, len(points) + 1)):
        raise BuildError("trajectory rounds are not contiguous from one")
    if any(not math.isfinite(float(point["error"])) or float(point["error"]) < 0.0 for point in points):
        raise BuildError("trajectory contains an invalid energy error")
    return points


def _costs_from_requested_round(summary: dict[str, Any]) -> dict[str, int]:
    from pipelines.reporting.append_paper_i_completed_beam_noise_pages import (
        _compile_cost_tuple,
    )

    costs, _receipt = _compile_cost_tuple(summary, round_index=50)
    return {key: int(costs[key]) for key in ("N2q", "D2q", "Dc", "S_alg")}


def _strong_append_k30(regime: str) -> tuple[dict[str, int], dict[str, str]]:
    from pipelines.reporting.build_paper_i_hh_insertion_policy_overlays import (
        _compile_append_cost,
    )

    proc = APPEND_PROCS[regime]
    archive = APPEND_ARCHIVES / f"proc{proc}_validation_input" / (
        f"core__{regime}__nph7__append_macro__cluster_9392883__proc_{proc}.tar.gz"
    )
    if not archive.is_file() or archive.is_symlink():
        raise BuildError(f"authenticated Append archive is absent: {archive}")
    costs = _compile_append_cost(archive, k=30)
    return (
        dict(zip(("N2q", "D2q", "Dc", "W1q", "S_alg"), costs, strict=True)),
        {"path": str(archive.relative_to(ROOT)), "sha256": _sha256(archive)},
    )


def _override_append(regime: str, source: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    execution_id = OVERRIDE_IDS[regime]
    runtime_dir = OVERRIDE_ROOT / execution_id
    run_root = runtime_dir / "runs" / execution_id
    summary_path = run_root / "summary/summary.json"
    manifest_path = run_root / "execution_manifest.json"
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    authority_path = runtime_dir / "provenance/resume_authorization.json"
    for path in (summary_path, manifest_path, receipt_path, authority_path):
        if not path.is_file() or path.is_symlink():
            raise BuildError(f"closed user-authorized continuation is absent: {path}")
    summary = _load(summary_path)
    manifest = _load(manifest_path)
    receipt = _load(receipt_path)
    authority = _load(authority_path)
    artifacts = {row["path"]: row for row in receipt.get("artifacts", [])}
    summary_relative = str(summary_path.relative_to(runtime_dir))
    manifest_relative = str(manifest_path.relative_to(runtime_dir))
    if (
        receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != 50
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("target_horizon") != 50
        or authority.get("status") != "authorized_authenticated_resume_to_k50"
        or authority.get("execution_id") != execution_id
        or authority.get("paper_evidence_adoption_authorized") is not True
        or authority.get("authorization_basis") != "direct_user_instruction_2026-08-13"
        or authority.get("scientific_setting_changes") != []
        or receipt.get("resume_authorization_sha256") != authority.get("sha256")
        or summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("available_controller_rounds") != 50
        or artifacts.get(summary_relative, {}).get("sha256") != _sha256(summary_path)
        or artifacts.get(manifest_relative, {}).get("sha256") != _sha256(manifest_path)
    ):
        raise BuildError(f"user-authorized continuation closure drifted: {regime}")
    trace = summary.get("accepted_error_trace")
    requested = summary.get("requested_rounds")
    if not isinstance(trace, list) or not isinstance(requested, list) or len(requested) != 1:
        raise BuildError(f"user-authorized continuation summary is incomplete: {regime}")
    points = _points(trace, round_key="controller_round", error_key="absolute_energy_error")
    if len(points) != 50 or int(requested[0].get("controller_round", -1)) != 50:
        raise BuildError(f"user-authorized continuation did not close at k=50: {regime}")
    source_points = _points(source["points"])
    if len(source_points) != 30:
        raise BuildError(f"authenticated source prefix is not k=30: {regime}")
    for old, new in zip(source_points, points[:30], strict=True):
        if old["k"] != new["k"] or not math.isclose(
            float(old["error"]), float(new["error"]), rel_tol=0.0, abs_tol=2.0e-12
        ):
            raise BuildError(f"continued append-only prefix drifted: {regime}")
    result = {
        "status": "completed_authenticated_local_k50_user_authorized_continuation",
        "execution_id": execution_id,
        "points": points,
        "terminal": points[-1],
        "costs": _costs_from_requested_round(summary),
    }
    bindings = [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in (summary_path, manifest_path, receipt_path, authority_path)
    ]
    return result, bindings


def main() -> None:
    page16 = _load(PAGE16)
    policies = _load(POLICIES)
    curve_cache = _load(CURVE_CACHE)
    page16_cells = {cell["regime_id"]: cell for cell in page16["cells"]}
    policy_cells = policies["completed_comparators"]
    if set(page16_cells) != {regime for regime, _, _ in REGIMES}:
        raise BuildError("Page-16 regime inventory drifted")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    override_bindings: list[dict[str, str]] = []
    for regime, _label, _nph in REGIMES:
        base = page16_cells[regime]
        plateau = base["page16_qiskit_route"]
        conventional = base["conventional_unwhitened_adapt"]
        always = policy_cells[regime]["always_commutation_reduced"]
        append_ra = policy_cells[regime]["append_only"]
        if regime in OVERRIDE_IDS:
            append_ra, bindings = _override_append(regime, append_ra)
            override_bindings.extend(bindings)
        conventional_points = _points(conventional["points"])
        conventional_costs = {
            "N2q": int(conventional["terminal"]["N2q"]),
            "D2q": int(conventional["terminal"]["D2q"]),
            "Dc": int(conventional["terminal"]["Dc"]),
            "S_alg": int(conventional["terminal"]["S_alg"]),
        }
        if regime in STRONG_HOLSTEIN:
            conventional_points = conventional_points[:30]
            cached_points = _points(curve_cache["curves"][regime]["append"])
            if conventional_points != cached_points[:30]:
                raise BuildError(f"strong-Holstein Append k=30 prefix drifted: {regime}")
            k30_costs, archive_binding = _strong_append_k30(regime)
            conventional_costs = {
                key: int(k30_costs[key]) for key in ("N2q", "D2q", "Dc", "S_alg")
            }
            override_bindings.append(archive_binding)
        records[regime] = {
            "plateau": {
                "points": _points(plateau["points"]),
                "terminal": {
                    "k": int(plateau["terminal"]["k"]),
                    "error": float(plateau["terminal"]["error"]),
                },
                "costs": {key: int(plateau["costs"][key]) for key in ("N2q", "D2q", "Dc", "S_alg")},
            },
            "conventional_append": {
                "points": conventional_points,
                "terminal": {
                    "k": int(conventional_points[-1]["k"]),
                    "error": float(conventional_points[-1]["error"]),
                },
                "costs": conventional_costs,
            },
            "always": {
                "points": _points(always["points"]),
                "terminal": {"k": int(always["terminal"]["k"]), "error": float(always["terminal"]["error"])},
                "costs": {key: int(always["costs"][key]) for key in ("N2q", "D2q", "Dc", "S_alg")},
            },
            "ra_append": {
                "points": _points(append_ra["points"]),
                "terminal": {"k": int(append_ra["terminal"]["k"]), "error": float(append_ra["terminal"]["error"])},
                "costs": {key: int(append_ra["costs"][key]) for key in ("N2q", "D2q", "Dc", "S_alg")},
            },
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = {
        "plateau": ("#d62728", 1.8, "o", "RA, plateau insertion"),
        "conventional_append": ("#1f77b4", 1.4, "^", "Append-ADAPT VQE"),
        "always": ("#e69f00", 1.45, "D", "RA, always insertion"),
        "ra_append": ("#9467bd", 1.35, "s", "RA, append only"),
    }
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.15), sharex=True)
    floor = 1.0e-16
    for axis, (regime, label, nph) in zip(axes.flat, REGIMES, strict=True):
        for policy in ("plateau", "conventional_append", "always", "ra_append"):
            record = records[regime][policy]
            color, width, marker, _ = styles[policy]
            axis.plot(
                [point["k"] for point in record["points"]],
                [max(float(point["error"]), floor) for point in record["points"]],
                color=color,
                lw=width,
            )
            axis.scatter(
                int(record["terminal"]["k"]),
                max(float(record["terminal"]["error"]), floor),
                color=color,
                marker=marker,
                s=30,
                zorder=6,
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 30 if regime in STRONG_HOLSTEIN else 50)
        axis.grid(True, which="major", alpha=0.25, lw=0.55)
        axis.set_title(rf"{label} ($n_{{\rm ph}}={nph}$)", fontsize=9.4)
        axis.set_xlabel("ADAPT iteration")
    axes[0, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    axes[1, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D([0], [0], color=color, lw=width, marker=marker, label=label)
            for color, width, marker, label in styles.values()
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.085, top=0.88, hspace=0.39, wspace=0.24)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)

    record = {
        "schema": "paper_i_intact_macro_four_policy_comparison_v1",
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
        "display_horizon": 50,
        "route_horizons": {
            "weak_holstein_ra_policies": 50,
            "strong_holstein_all_policies": 30,
            "weak_holstein_conventional_append": 50,
        },
        "cost_tuple": ["N2q", "D2q", "Dc", "S_alg"],
        "records": records,
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in (PAGE16, POLICIES, CURVE_CACHE)
        ]
        + override_bindings,
    }
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
