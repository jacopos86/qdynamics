#!/usr/bin/env python3
"""Build the reader-facing k<=50 matched singleton plot for Paper I."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RA_DIR = ROOT / "chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809"
APPEND = ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "append_singleton_r70_all6_adapter.json"
)
MATCHED_REPORT = ROOT / "output/pdf/paper_i_ra_vs_append_matched_comparisons_20260812.tex"
POLICY_ADAPTER = ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "page12_singleton_insertion_comparator_snapshot_adapter.json"
)
LOCAL_POLICY_CAMPAIGN = ROOT / (
    "output/local_runs/"
    "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1"
)
LOCAL_POLICY_STATUS = LOCAL_POLICY_CAMPAIGN / "status/campaign.json"
LOCAL_POLICY_TERMINAL = LOCAL_POLICY_CAMPAIGN / "terminal_receipt.json"
OUT_DIR = ROOT / "MATH/paper_details/figures/paper_i_ra_vs_append_matched_singleton_plateau_20260812"
OUTPUT = OUT_DIR / "paper_i_ra_vs_append_matched_singleton_plateau.pdf"
PROVENANCE = OUT_DIR / "paper_i_ra_vs_append_matched_singleton_plateau_provenance.json"

REGIMES = [
    ("weak_weak", "Weak--weak", 3, 1.00e-9, 37, 37, 2.61, 45, 27),
    ("intermediate_weak", "Intermediate--weak", 3, 5.59e-9, 34, 32, 3.07, 42, 25),
    ("strong_weak_u8", "Strong--weak", 3, 1.41e-6, 11, 11, 0.94, 11, 11),
    ("weak_strong", "Weak--strong", 7, 6.06e-4, 35, 50, 3.80, 50, 34),
    ("intermediate_strong", "Intermediate--strong", 7, 1.41e-4, 39, 49, 3.42, 49, 35),
    ("strong_strong_u8", "Strong--strong", 7, 3.21e-8, 45, 45, 2.13, 45, 28),
]

LOCAL_POLICY_EXECUTIONS = {
    "weak_strong": {
        "always_commutation_reduced": (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            "weak_strong__nph7__"
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "always_commutation_reduced"
        ),
    },
    "intermediate_strong": {
        "always_commutation_reduced": (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            "intermediate_strong__nph7__"
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "always_commutation_reduced"
        ),
        "append_only": (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            "intermediate_strong__nph7__"
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "append_only"
        ),
    },
    "strong_strong_u8": {
        "always_commutation_reduced": (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            "strong_strong_u8__nph7__"
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "always_commutation_reduced"
        ),
        "append_only": (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            "strong_strong_u8__nph7__"
            "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "append_only"
        ),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_digested_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if not isinstance(claimed, str) or claimed != canonical_sha256(unsigned):
        raise ValueError(f"{label} self digest drifted: {path}")
    return value


def validate_file_binding(
    binding: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any] | None]:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is absent")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe: {raw_path}")
    path = LOCAL_POLICY_CAMPAIGN / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != binding.get("size_bytes")
        or sha256(path) != binding.get("sha256")
    ):
        raise ValueError(f"{label} file binding drifted: {path}")
    canonical = binding.get("canonical_sha256")
    if canonical is None:
        return path, None
    value = load_digested_json(path, label=label)
    if value.get("sha256") != canonical:
        raise ValueError(f"{label} canonical digest drifted: {path}")
    return path, value


def load_local_completed_policies() -> tuple[
    dict[str, dict[str, dict]], list[Path], dict[str, Any]
]:
    """Load receipt-validated local k=50 policy trajectories."""

    expected_ids = [
        execution_id
        for policies in LOCAL_POLICY_EXECUTIONS.values()
        for execution_id in policies.values()
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("local policy execution IDs are not unique")

    status = load_digested_json(LOCAL_POLICY_STATUS, label="local campaign status")
    terminal = load_digested_json(
        LOCAL_POLICY_TERMINAL, label="local campaign terminal receipt"
    )
    expected_id_set = set(expected_ids)
    if (
        terminal.get("schema")
        != "paper_i_page12_strong_sector5_local_terminal_receipt_v1"
        or terminal.get("status") != "passed_all_five_cells_immutable_closure"
        or set(terminal.get("completed_execution_ids", ())) != expected_id_set
        or terminal.get("execution_authorized") is not True
        or terminal.get("submission_authorized") is not False
        or terminal.get("paper_evidence_adoption_authorized") is not False
        or status.get("schema") != "paper_i_page12_strong_sector5_local_status_v1"
        or status.get("status") != "passed_all_five_cells"
        or set(status.get("completed_execution_ids", ())) != expected_id_set
        or status.get("current_execution_id") is not None
        or status.get("child_pid") is not None
        or status.get("failure") is not None
        or status.get("terminal_receipt_sha256") != terminal.get("sha256")
    ):
        raise ValueError("local five-cell campaign closure drifted")
    terminal_cells = terminal.get("cells")
    if not isinstance(terminal_cells, list):
        raise ValueError("local campaign terminal cell bindings are absent")
    cells_by_id = {
        cell.get("execution_id"): cell
        for cell in terminal_cells
        if isinstance(cell, Mapping)
    }
    if set(cells_by_id) != expected_id_set or len(cells_by_id) != len(terminal_cells):
        raise ValueError("local campaign terminal cell inventory drifted")

    completed: dict[str, dict[str, dict]] = {}
    sources: list[Path] = [LOCAL_POLICY_STATUS, LOCAL_POLICY_TERMINAL]
    for regime, policies in LOCAL_POLICY_EXECUTIONS.items():
        for policy, execution_id in policies.items():
            run_dir = LOCAL_POLICY_CAMPAIGN / "runs" / execution_id
            terminal_cell = cells_by_id[execution_id]
            manifest_path, manifest = validate_file_binding(
                terminal_cell["execution_manifest"],
                label=f"{execution_id} execution manifest",
            )
            receipt_path, receipt = validate_file_binding(
                terminal_cell["worker_receipt"],
                label=f"{execution_id} worker receipt",
            )
            guard_path, guard = validate_file_binding(
                terminal_cell["guard_receipt"],
                label=f"{execution_id} guard receipt",
            )
            assert manifest is not None and receipt is not None and guard is not None
            summary_binding = manifest.get("output_payloads", {}).get("summary")
            if not isinstance(summary_binding, Mapping):
                raise ValueError(f"local policy summary binding is absent: {execution_id}")
            summary_path, _ = validate_file_binding(
                summary_binding, label=f"{execution_id} summary"
            )
            if summary_path.resolve() != (run_dir / "summary/summary.json").resolve():
                raise ValueError(f"local policy summary path drifted: {execution_id}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            if (
                manifest.get("status") != "passed"
                or manifest.get("execution_id") != execution_id
                or manifest.get("comparator_policy") != policy
                or manifest.get("controller_rounds_completed") != 50
                or manifest.get("paper_evidence_adoption_authorized") is not False
                or receipt.get("status") != "passed"
                or receipt.get("execution_id") != execution_id
                or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
                or guard.get("status") != "passed"
                or guard.get("execution_id") != execution_id
                or guard.get("execution_manifest_sha256") != manifest.get("sha256")
                or guard.get("child_returncode") != 0
                or guard.get("guard_stop_reason") is not None
            ):
                raise ValueError(f"local policy receipt is not terminally passed: {receipt_path}")
            if receipt.get("controller_rounds_completed") != 50:
                raise ValueError(f"local policy receipt does not close k=50: {receipt_path}")
            trace = summary.get("accepted_error_trace")
            if not isinstance(trace, list) or len(trace) != 50:
                raise ValueError(f"local policy summary does not contain 50 accepted rounds: {summary_path}")
            rounds = [point.get("controller_round") for point in trace]
            if rounds != list(range(1, 51)):
                raise ValueError(f"local policy summary has a noncontiguous round trace: {summary_path}")

            points = [
                {
                    "k": point["controller_round"],
                    "active_ansatz_depth": point["active_ansatz_depth"],
                    "energy": point["accepted_energy"],
                    "error": point["absolute_energy_error"],
                }
                for point in trace
            ]
            terminal_point = points[-1]
            completed.setdefault(regime, {})[policy] = {
                "comparator_policy": policy,
                "controller_rounds_completed": 50,
                "exact_same_cutoff_energy": trace[-1]["exact_same_cutoff_energy"],
                "points": points,
                "terminal": {
                    "k": terminal_point["k"],
                    "energy": terminal_point["energy"],
                    "error": terminal_point["error"],
                },
            }
            sources.extend([manifest_path, receipt_path, guard_path, summary_path])
    return completed, sources, {
        "status": terminal["status"],
        "terminal_receipt_sha256": terminal["sha256"],
        "campaign_status_sha256": status["sha256"],
        "completed_execution_ids": terminal["completed_execution_ids"],
        "paper_evidence_adoption_authorized": False,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    append = json.loads(APPEND.read_text(encoding="utf-8"))
    append_cells = {cell["regime_id"]: cell for cell in append["cells"]}
    policies = json.loads(POLICY_ADAPTER.read_text(encoding="utf-8"))
    completed_policies = policies["completed_comparators"]
    (
        local_completed_policies,
        local_policy_sources,
        local_policy_campaign_closure,
    ) = load_local_completed_policies()
    for regime, local_policies in local_completed_policies.items():
        completed_policies.setdefault(regime, {}).update(local_policies)
    ra_cells = {}
    for path in sorted(RA_DIR.glob("*_completed_report_adapter.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        ra_cells[cell["regime_id"]] = (path, cell)

    plt.rcParams.update({"font.family": "serif", "font.size": 8.5})
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.15), sharex=True)
    ra_color = "#d62728"
    append_color = "#1f77b4"
    always_color = "#e69f00"
    ra_append_color = "#9467bd"

    for ax, (regime, title, nph, target, kra, kap, sadv, bra, bap) in zip(axes.flat, REGIMES):
        _, ra = ra_cells[regime]
        ap = append_cells[regime]
        ap_points = [point for point in ap["points"] if point["round"] <= 50]
        ap_x = [point["round"] for point in ap_points]
        ap_y = [point["delta_e"] for point in ap_points]
        ra_x = [0] + [point["k"] for point in ra["points"] if point["k"] <= 50]
        exact = ap["exact_same_cutoff_energy"]
        ra_y = [ap_y[0]] + [abs(point["energy"] - exact) for point in ra["points"] if point["k"] <= 50]

        ax.plot(ra_x, ra_y, color=ra_color, lw=1.8)
        ax.plot(ap_x, ap_y, color=append_color, lw=1.4)
        available_policies = completed_policies.get(regime, {})
        if "always_commutation_reduced" in available_policies:
            always = available_policies["always_commutation_reduced"]
            always_x = [0] + [point["k"] for point in always["points"]]
            always_y = [ap_y[0]] + [point["error"] for point in always["points"]]
            ax.plot(always_x, always_y, color=always_color, lw=1.45)
            ax.scatter(
                always["terminal"]["k"],
                always["terminal"]["error"],
                s=36,
                marker="D",
                color=always_color,
                zorder=6,
            )
        if "append_only" in available_policies:
            ra_append = available_policies["append_only"]
            ra_append_x = [0] + [point["k"] for point in ra_append["points"]]
            ra_append_y = [ap_y[0]] + [point["error"] for point in ra_append["points"]]
            ax.plot(ra_append_x, ra_append_y, color=ra_append_color, lw=1.35)
            ax.scatter(
                ra_append["terminal"]["k"],
                ra_append["terminal"]["error"],
                s=39,
                marker="s",
                color=ra_append_color,
                zorder=6,
            )
        ax.scatter(kra, ra_y[kra], s=28, facecolors="white", edgecolors=ra_color, linewidths=1.4, zorder=5)
        ax.scatter(kap, ap_y[kap], s=28, facecolors="white", edgecolors=append_color, linewidths=1.4, zorder=5)
        ax.scatter(bra, ra_y[bra], s=38, marker="^", color=ra_color, zorder=5)
        ax.scatter(bap, ap_y[bap], s=38, marker="^", color=append_color, zorder=5)
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.grid(True, which="major", alpha=0.25, lw=0.55)
        ax.set_title(rf"{title} ($n_{{\rm ph}}={nph}$)", fontsize=9.4)
        ax.set_xlabel("ADAPT iteration")

    axes[0, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    axes[1, 0].set_ylabel(r"same-cutoff $|\Delta E|$")
    legend = [
        Line2D([0], [0], color=ra_color, lw=1.8, label="RA, plateau insertion"),
        Line2D([0], [0], color=append_color, lw=1.4, label="Append-ADAPT VQE"),
        Line2D(
            [0],
            [0],
            color=always_color,
            marker="D",
            markevery=[1],
            lw=1.45,
            label="RA, always insertion",
        ),
        Line2D(
            [0],
            [0],
            color=ra_append_color,
            marker="s",
            markevery=[1],
            lw=1.35,
            label="RA, append only",
        ),
        Line2D([0], [0], marker="o", color="0.25", markerfacecolor="white", lw=0, label="first common-accuracy crossing"),
        Line2D([0], [0], marker="^", color="0.25", lw=0, label=r"best error within shared $\mathcal{E}$"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, fontsize=7.2)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.085, top=0.855, hspace=0.39, wspace=0.24)
    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    source_paths = [APPEND, MATCHED_REPORT, POLICY_ADAPTER] + [
        ra_cells[r][0] for r, *_ in REGIMES
    ] + local_policy_sources
    record = {
        "schema": "paper_i_reader_facing_matched_singleton_plot_v8",
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "horizon": 50,
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in source_paths
        ],
        "matched_markers": "locked to the matched-comparison report dated 2026-08-12",
        "common_accuracy_lines": "omitted by author direction; first-crossing markers retained",
        "common_accuracy_panel_text": "omitted by author direction",
        "legend_layout": "two_rows_by_three_columns",
        "policy_curve_coverage": {
            "always_commutation_reduced": [
                regime
                for regime, *_ in REGIMES
                if "always_commutation_reduced" in completed_policies.get(regime, {})
            ],
            "append_only": [
                regime
                for regime, *_ in REGIMES
                if "append_only" in completed_policies.get(regime, {})
            ],
            "markers": "terminal k=50 only",
        },
        "local_policy_campaign_closure": local_policy_campaign_closure,
        "evidence_class": "preliminary_local_candidate_overlay_not_adopted",
        "paper_evidence_adoption_authorized": False,
        "cost_tuple": ["N2q", "D2q", "Dc", "S_alg"],
    }
    PROVENANCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
