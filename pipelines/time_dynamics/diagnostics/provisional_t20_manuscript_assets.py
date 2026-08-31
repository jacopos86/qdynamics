"""Build the full-horizon Paper-II trajectory and frontier assets.

The reader-facing table contains one compact row per verified run.  It reports
full-horizon mean infidelity, three mean observable errors, accepted integration
steps, terminal active coordinates, and the routing-independent terminal
graph-cost tuple.  Every row uses a stitched t=0..20 trajectory and one verified
shared exact reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    ExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.physical_errors import physical_errors
from pipelines.time_dynamics.diagnostics.promotion_contract import (
    manuscript_rows_from_results_pdf,
    verified_results_pdf_tuples,
)
from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
    load_terminal_resources,
)


SCHEMA = "paper_ii_t20_manuscript_assets_v4"
AUTHOR_OVERRIDE = (
    "Author explicitly authorized immediate manuscript admission of the "
    "artifact-verified t20 frontier. Full-horizon mean infidelity, accepted "
    "integration steps, and terminal graph costs are reported numerically."
)

DRIVE_ORDER = {
    "weakslow": 0,
    "fastweak": 1,
    "midres": 2,
    "slowstrong": 3,
    "strongfast": 4,
    "fastfast": 5,
}
CONTROLLER_ORDER = {
    "delta_theta_5e-5": 0,
    "state_motion_1e-3_plus_parameter_5e-5": 1,
    "state_motion_5e-5": 2,
}
CONTROLLER_TEX = {
    "delta_theta_5e-5": r"$\delta\theta$",
    "state_motion_1e-3_plus_parameter_5e-5": "combined",
    "state_motion_5e-5": "state motion",
}
METHOD_TEX = {
    "avqds": "AVQDS",
    "apm": "full APM",
    "append_only": "insertion McLachlan (IM)",
}
DRIVE_TEX = {
    "weakslow": "weak--slow",
    "fastweak": "fast--weak",
    "midres": "mid-resonant",
    "slowstrong": "slow--strong",
    "strongfast": "strong--fast",
    "fastfast": "fast--fast",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_label(label: str) -> dict[str, str]:
    parts = label.split("__")
    if len(parts) != 7 or parts[5] != "t20":
        raise ValueError(f"unexpected staged t20 label: {label}")
    if parts[0] == "promote_avqds_t20":
        method = "avqds"
    elif parts[0] == "promote_apm_t20" and parts[2] == "append_only":
        method = "append_only"
    elif parts[0] == "promote_apm_t20":
        method = "apm"
    else:
        raise ValueError(f"unexpected staged t20 method: {label}")
    return {
        "method": method,
        "seed": parts[1],
        "route": parts[2],
        "controller": parts[3],
        "drive": parts[4],
        "horizon": parts[5],
        "cut": parts[6].removeprefix("cut"),
    }


def _action_sources(actions_jsonl: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for line in actions_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        action_id = str(payload["action_id"])
        source = Path(payload["resume_receipt"]["source_path"])
        if action_id in sources:
            raise ValueError(f"duplicate action id in {actions_jsonl}: {action_id}")
        sources[action_id] = source
    return sources


def _frontier_rows(
    t20_root: Path,
    *,
    actions_jsonl: Path,
    seed_path: Path,
    reference_store: ExactReferenceStore,
    reference_substeps: int,
) -> list[dict[str, Any]]:
    screen_sources = _action_sources(actions_jsonl)
    rows: list[dict[str, Any]] = []
    for run_path in sorted(t20_root.glob("*/run.json")):
        run = _read_json(run_path)
        resources = load_terminal_resources(run_path)
        metadata = _parse_label(run_path.parent.name)
        summary = run["summary"]
        screen_path = screen_sources.get(run_path.parent.name)
        if screen_path is None or not screen_path.is_file():
            raise FileNotFoundError(
                f"missing t=5 source for {run_path.parent.name}: {screen_path}"
            )
        observed = _full_trajectory(screen_path, run_path)
        screen_summary = _read_json(screen_path)["summary"]
        reference = reference_store.get_or_build(
            seed_path=seed_path,
            run_path=run_path,
            times=[float(row["time"]) for row in observed],
            substeps=int(reference_substeps),
        )
        errors = physical_errors(observed, reference.rows)
        rows.append(
            {
                **metadata,
                "run_path": str(run_path),
                "run_sha256": _sha256(run_path),
                "cost_path": str(resources.artifact_path),
                "cost_sha256": resources.artifact_sha256,
                "cost_source_kind": resources.source_kind,
                "screen_run_path": str(screen_path),
                "screen_run_sha256": _sha256(screen_path),
                "n_theta": resources.n_theta,
                "graph": list(resources.graph),
                "mean_abs_energy_error": errors.mean_abs_energy_error,
                "mean_abs_doublon_error": errors.mean_abs_doublon_error,
                "mean_max_abs_site_occupation_error": (
                    errors.mean_max_abs_site_occupation_error
                ),
                "mean_infidelity": errors.mean_infidelity,
                "time_step_count": (
                    int(screen_summary["accepted_internal_substep_count"])
                    + int(summary["accepted_internal_substep_count"])
                ),
                "reference_key": reference.key,
                "reference_manifest_path": str(reference.manifest_path),
                "reference_manifest_sha256": _sha256(reference.manifest_path),
                "reference_data_sha256": reference.data_sha256,
                "parity": "passed",
                "accuracy_status": "full_horizon_errors_computed",
            }
        )
    rows.sort(
        key=lambda row: (
            DRIVE_ORDER.get(row["drive"], 99),
            CONTROLLER_ORDER.get(row["controller"], 99),
            0 if row["method"] == "apm" else 1,
        )
    )
    return rows


def _full_trajectory(t5_path: Path, t20_path: Path) -> list[dict[str, Any]]:
    t5 = _read_json(t5_path)["plot_rows"]
    t20 = _read_json(t20_path)["plot_rows"]
    if not t5 or not t20:
        raise ValueError("trajectory legs must be nonempty")
    if abs(float(t5[-1]["time"]) - float(t20[0]["time"])) > 1.0e-12:
        raise ValueError("t5 and t20 legs do not share a resume boundary")
    return [*t5, *t20[1:]]


def _build_figure(
    *,
    apm_row: dict[str, Any],
    avqds_row: dict[str, Any],
    seed_path: Path,
    reference_store: ExactReferenceStore,
    reference_substeps: int,
    output: Path,
) -> list[dict[str, str]]:
    apm_t5 = Path(apm_row["screen_run_path"])
    apm_t20 = Path(apm_row["run_path"])
    avqds_t5 = Path(avqds_row["screen_run_path"])
    avqds_t20 = Path(avqds_row["run_path"])
    sources = [apm_t5, apm_t20, avqds_t5, avqds_t20]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
    apm = _full_trajectory(apm_t5, apm_t20)
    avqds = _full_trajectory(avqds_t5, avqds_t20)
    reference = reference_store.get_or_build(
        seed_path=seed_path,
        run_path=apm_t20,
        times=[float(row["time"]) for row in apm],
        substeps=int(reference_substeps),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.18), constrained_layout=True)
    trajectories = (
        ("AP-McLachlan", apm, "#7F0000", "-", 1.25),
        ("AVQDS", avqds, "#2166AC", "--", 1.15),
    )
    for label, rows, color, linestyle, linewidth in trajectories:
        time = [float(row["time"]) for row in rows]
        energy = [float(row["energy_expectation"]) for row in rows]
        doublon = [float(row["doublon"]) for row in rows]
        imbalance = [
            float(row["site_occupations"][0])
            - float(row["site_occupations"][1])
            for row in rows
        ]
        for axis, values in zip(axes, (energy, doublon, imbalance), strict=True):
            axis.plot(
                time,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
    exact_values = (
        [float(row["energy"]) for row in reference.rows],
        [float(row["doublon"]) for row in reference.rows],
        [
            float(row["site_occupations"][0])
            - float(row["site_occupations"][1])
            for row in reference.rows
        ],
    )
    exact_time = [float(row["time"]) for row in reference.rows]
    for axis, values in zip(axes, exact_values, strict=True):
        axis.plot(
            exact_time,
            values,
            color="black",
            linestyle=":",
            linewidth=1.05,
            label="exact",
        )
    ylabels = (r"$E(t)$", r"$D(t)$", r"$\Delta n(t)$")
    for axis, ylabel in zip(axes, ylabels, strict=True):
        axis.set_xlabel(r"time $t$")
        axis.set_ylabel(ylabel)
        axis.axvline(5.0, color="0.72", linewidth=0.60, linestyle="-.")
        axis.grid(alpha=0.18, linewidth=0.45)
        axis.tick_params(labelsize=7)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    result = [{"path": str(path), "sha256": _sha256(path)} for path in sources]
    result.append(
        {
            "path": str(reference.manifest_path),
            "sha256": _sha256(reference.manifest_path),
        }
    )
    return result


def _table_tex(rows: list[dict[str, Any]]) -> str:
    result_body: list[str] = []
    for row in rows:
        prefix = (
            f"{DRIVE_TEX[row['drive']]} & {CONTROLLER_TEX[row['controller']]} & "
            f"{METHOD_TEX[row['method']]}"
        )
        paper_status = row.get("paper_status", "qualified")
        if paper_status in {"pending", "unresolved"} and "mean_infidelity" not in row:
            reason = {
                "t20_not_completed": "run not completed",
                "safeguard_provenance_only": "safeguard",
                "ladder_or_safeguard_exhausted": "unresolved: ladder/resource exhausted",
            }.get(
                str(row.get("state_reason") or row.get("pending_reason") or ""),
                paper_status,
            )
            result_body.append(
                f"{prefix} & "
                f"\\multicolumn{{7}}{{c}}{{\\textit{{{reason}}}}} \\\\"
            )
            continue
        graph = ",".join(str(value) for value in row["graph"])
        result_body.append(
            f"{prefix} & ${_tex_scientific(float(row['mean_infidelity']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_energy_error']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_doublon_error']))}$ & "
            f"${_tex_scientific(float(row['mean_abs_charge_imbalance_error']))}$ & "
            f"{row['time_step_count']:,} & {row['n_theta']} & $({graph})$ \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{Planned $t=20$ APM/AVQDS comparison matrix, evaluated against one shared exact trajectory for each physical drive on the complete 501-point grid. Each cell is defined by the physical regime, drive, and time-step controller. Every completed arm reports mean infidelity $\overline{1-F}=\operatorname{mean}_t[1-|\langle\psi_{\rm ex}(t)|\psi(t)\rangle|^2]$, mean absolute errors in energy, doublon occupation, and charge imbalance, total accepted integration steps $N_{\rm step}$, terminal active-coordinate count $N_\theta$, and routing-independent terminal graph tuple $\widehat{\boldsymbol C}_G=(\widehat C_{2q},\widehat C_D,\widehat C_{1q})$. Observable accuracy is normalized by the RMS exact response induced by the corresponding drive. Each arm targets five-percent relative response error and $\overline{1-F}\leq10^{-4}$; a pair may also support a cost comparison when both arms remain inside the fifteen-percent response guard, obey the absolute safety ceilings, and both their aggregate accuracy burdens and every deadbanded primitive $q$ component agree within a factor of two. For this fixed-number two-site benchmark, $\overline{|\Delta(\Delta n)|}=2\,\overline{\max_i|\Delta n_i|}$. Only unfinished arms remain pending placeholders. Additional observable plots remain in the campaign results PDF pending author selection for publication.}",
            r"\label{tab:t20_provisional_frontier}",
            r"\centering\tiny",
            r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lllrrrrrrr}",
            r"\toprule",
            r"drive & controller & method & $\overline{1-F}$ & $\overline{|\Delta E|}$ & $\overline{|\Delta D|}$ & $\overline{|\Delta(\Delta n)|}$ & $N_{\rm step}$ & $N_\theta$ & $\widehat{\boldsymbol C}_G$ \\",
            r"\midrule",
            *result_body,
            r"\addlinespace[2pt]",
            r"\multicolumn{3}{l}{insertion McLachlan (IM)} & \multicolumn{7}{c}{\textit{$t=20$ matrix pending}} \\",
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )


def write_table_from_results_pdf(
    *,
    preview_receipt_path: Path,
    report_manifest_path: Path,
    table_path: Path,
    receipt_path: Path,
    workspace_root: Path = Path("."),
) -> dict[str, Any]:
    """Write the full planned matrix already verified in the results PDF."""

    tuples = verified_results_pdf_tuples(
        preview_receipt_path=preview_receipt_path,
        report_manifest_path=report_manifest_path,
        workspace_root=workspace_root,
    )
    rows = manuscript_rows_from_results_pdf(tuples)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(_table_tex(rows), encoding="utf-8")
    receipt = {
        "schema": "paper_ii_results_pdf_first_table_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "results_pdf_tuple_coverage_verified_before_manuscript_table",
        "observable_plot_policy": (
            "campaign-results-PDF only pending author publication selection"
        ),
        "results_pdf": {
            "report_manifest": str(report_manifest_path),
            "report_manifest_sha256": _sha256(report_manifest_path),
            "preview_receipt": str(preview_receipt_path),
            "preview_receipt_sha256": _sha256(preview_receipt_path),
        },
        "tuple_count": len(tuples),
        "tuple_ids": [item.tuple_id for item in tuples],
        "row_count": len(rows),
        "qualified_row_count": sum(
            row["paper_status"] == "qualified" for row in rows
        ),
        "candidate_row_count": sum(
            row["paper_status"] == "candidate" for row in rows
        ),
        "pending_row_count": sum(row["paper_status"] == "pending" for row in rows),
        "unresolved_row_count": sum(
            row["paper_status"] == "unresolved" for row in rows
        ),
        "table": {"path": str(table_path), "sha256": _sha256(table_path)},
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _tex_scientific(value: float) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t20-root", type=Path)
    parser.add_argument("--actions-jsonl", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-substeps", type=int, default=40)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--results-preview-receipt", type=Path)
    parser.add_argument("--campaign-report-manifest", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--finalize-manuscript-source", type=Path)
    parser.add_argument("--finalize-manuscript-pdf", type=Path)
    args = parser.parse_args()

    if args.results_preview_receipt or args.campaign_report_manifest:
        if not (
            args.results_preview_receipt
            and args.campaign_report_manifest
            and args.table
        ):
            parser.error(
                "results-PDF-first table generation requires preview receipt, "
                "campaign report manifest, and table output"
            )
        write_table_from_results_pdf(
            preview_receipt_path=args.results_preview_receipt,
            report_manifest_path=args.campaign_report_manifest,
            table_path=args.table,
            receipt_path=args.receipt,
            workspace_root=args.workspace_root,
        )
        return

    if args.finalize_manuscript_source or args.finalize_manuscript_pdf:
        if not (args.finalize_manuscript_source and args.finalize_manuscript_pdf):
            parser.error("both manuscript finalization paths are required")
        receipt = _read_json(args.receipt)
        for key in ("figure", "table"):
            artifact = Path(receipt[key]["path"])
            if _sha256(artifact) != receipt[key]["sha256"]:
                raise ValueError(f"{key} digest changed before manuscript finalization")
        receipt["promotion"] = {
            "status": "author_override_full_horizon_infidelity_promoted",
            "manuscript_source": {
                "path": str(args.finalize_manuscript_source),
                "sha256": _sha256(args.finalize_manuscript_source),
            },
            "manuscript_pdf": {
                "path": str(args.finalize_manuscript_pdf),
                "sha256": _sha256(args.finalize_manuscript_pdf),
            },
            "admitted_claims": [
                "artifact-verified t20 trajectories exist",
                "terminal support and resource observations passed parity",
                "reported mean infidelity uses all t=0..20 points",
                "reported step count spans the stitched t=0..20 trajectory",
                "exact reference is shared across methods for each drive",
            ],
            "excluded_claims": [
                "complete drive-controller-method matrix",
                "pruning attribution without the insertion-face ablation",
            ],
        }
        receipt["generated_utc"] = datetime.now(timezone.utc).isoformat()
        args.receipt.write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        return

    if args.table is not None:
        parser.error(
            "direct manuscript-table generation from raw t20 outputs is retired; "
            "provide --results-preview-receipt and --campaign-report-manifest"
        )

    required = (
        args.t20_root,
        args.actions_jsonl,
        args.seed,
        args.reference_root,
        args.figure,
        args.table,
    )
    if any(value is None for value in required):
        parser.error(
            "asset generation requires t20 root, action provenance, seed, "
            "reference store, figure, and table"
        )
    reference_store = ExactReferenceStore(args.reference_root)
    rows = _frontier_rows(
        args.t20_root,
        actions_jsonl=args.actions_jsonl,
        seed_path=args.seed,
        reference_store=reference_store,
        reference_substeps=int(args.reference_substeps),
    )
    if len(rows) != 12:
        raise ValueError(f"expected twelve verified t20 rows, found {len(rows)}")
    by_key = {(row["method"], row["drive"], row["controller"]): row for row in rows}
    apm_row = by_key[("apm", "weakslow", "delta_theta_5e-5")]
    avqds_row = by_key[("avqds", "weakslow", "delta_theta_5e-5")]
    figure_sources = _build_figure(
        apm_row=apm_row,
        avqds_row=avqds_row,
        seed_path=args.seed,
        reference_store=reference_store,
        reference_substeps=int(args.reference_substeps),
        output=args.figure,
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table_tex(rows), encoding="utf-8")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "admission": "explicit_author_override_full_horizon_infidelity_computed",
        "author_override": AUTHOR_OVERRIDE,
        "frontier_row_count": len(rows),
        "rows": rows,
        "figure": {"path": str(args.figure), "sha256": _sha256(args.figure)},
        "figure_sources": figure_sources,
        "table": {"path": str(args.table), "sha256": _sha256(args.table)},
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
