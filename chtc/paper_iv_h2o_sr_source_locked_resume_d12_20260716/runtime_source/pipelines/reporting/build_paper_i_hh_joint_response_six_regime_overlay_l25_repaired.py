#!/usr/bin/env python3
"""Append the repaired fixed-policy L25 six-regime comparison as page three."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import build_paper_i_hh_joint_response_six_regime_overlay as base


SCHEMA = "paper_i_hh_joint_response_six_regime_overlay_l25_repaired_v1"
STEM = "paper_i_hh_joint_response_six_regime_overlay_l25_repaired_20260712"
DEFAULT_OUTPUT_DIR = base.REPO_ROOT / f"output/pdf/{STEM}"
ORIGINAL_DIR = base.DEFAULT_OUTPUT_DIR
ORIGINAL_TEX = ORIGINAL_DIR / f"{base.STEM}.tex"
RETRIEVAL_ROOT = base.REPO_ROOT / (
    "output/chtc_retrievals/paper_i_hh_wave11_legal_six_regime_20260712"
)
RESULT_ROOT = RETRIEVAL_ROOT / (
    "raw_outputs/paper_i_hh_wave11_legal_six_regime_20260711_v3/trial_000000"
)
TRIAL_MANIFEST = RETRIEVAL_ROOT / (
    "chtc/phase3_optuna/input/paper_i_hh_wave11_legal_six_regime_20260711_v2/"
    "trial_manifests/trial_000000.json"
)
REPAIR_MANIFEST = RETRIEVAL_ROOT / (
    "chtc/phase3_optuna/input/paper_i_hh_wave11_legal_six_regime_20260711_v3_repair/"
    "repair_manifest.json"
)
EXPECTED_CONTRACT_HASH = "a142d1c4f3c7a40d36505cba34012397bb8026dca9b28f4cfe903c8455299b8b"
REPAIRED_STYLE = {
    "label": "Repaired L25 fixed-policy SNAKE",
    "color": "#8F63A8",
    "marker": "v",
    "width": 1.8,
}


def _load_evidence(output_dir: Path) -> dict[str, Any]:
    trial_manifest = base._read_json(TRIAL_MANIFEST)
    repair_manifest = base._read_json(REPAIR_MANIFEST)
    policy = trial_manifest.get("policy") or {}
    checks = {
        "execution_profile": trial_manifest.get("execution_profile")
        == "wave11_legal_fixed_policy_v1",
        "scientific_contract_hash": trial_manifest.get("scientific_contract_hash")
        == EXPECTED_CONTRACT_HASH,
        "repair_contract_hash": repair_manifest.get("scientific_contract_hash")
        == EXPECTED_CONTRACT_HASH,
        "repair_science_unchanged": repair_manifest.get("scientific_contract_change") is False,
        "cluster": int(repair_manifest.get("submitted_cluster", -1)) == 8775444,
        "policy": policy
        == {
            "batch_mode": "combinatorial",
            "batch_search_pool_size": 25,
            "batch_size_cap": 2,
            "child_phase1_cap": 128,
            "child_phase2_cap": 64,
            "lambda_add": 0.0,
            "macro_phase1_cap": 64,
            "macro_phase2_cap": 48,
        },
    }
    if not all(checks.values()):
        raise ValueError(f"Repaired-L25 campaign contract failed: {checks}")

    references = base._paper_reference_rows(base._read_json(base.PAPER_I_REFERENCE_JSON))
    resource_rows = base._resource_rows_by_regime(base._read_json(base.QISKIT_TABLE_JSON))
    qiskit_dir = output_dir / "supplemental_selected_prefix_qiskit"
    regimes: list[dict[str, Any]] = []
    for regime in base.REGIME_ORDER:
        directory = RESULT_ROOT / base.CAMPAIGN_DIR[regime]
        result_path = directory / "result.json"
        summary_path = directory / "compact_summary.json"
        query_path = directory / "query_work_sidecar.json"
        wrapper_path = directory / "wrapper_status.json"
        result = base._read_json(result_path)
        summary = base._read_json(summary_path)
        query = base._read_json(query_path)
        wrapper = base._read_json(wrapper_path)
        row_checks = {
            "result_complete": result.get("status") == "complete",
            "summary_complete": summary.get("status") == "complete",
            "query_complete": query.get("status") == "complete",
            "query_ok": query.get("query_work_status") == "ok",
            "query_scope": query.get("query_work_scope") == "winner_lineage_terminal",
            "wrapper_ok": wrapper.get("status") == "wrapper_complete"
            and int(wrapper.get("exit_code", -1)) == 0,
        }
        if not all(row_checks.values()):
            raise ValueError(f"Incomplete repaired-L25 evidence for {regime}: {row_checks}")
        rounds = int(summary["summary"]["controller_round_count"])
        error = float(summary["summary"]["abs_delta_e"])
        if rounds <= 0 or not math.isfinite(error) or error < 0.0:
            raise ValueError(f"Invalid repaired-L25 endpoint for {regime}")
        curve = base._history_curve(
            result_path,
            role="repaired_l25",
            marker_k=rounds,
            marker_error=error,
        )
        repaired_resource = base._supplemental_resource_row(
            regime=regime,
            method="repaired_l25_snake",
            source_json=result_path,
            history_position=rounds,
            expected_error=error,
            sidecar_json=qiskit_dir / f"repaired-l25-{regime}.json",
            s_override=float(query["query_work_total"]),
            s_source_override="query_work_sidecar.query_work_total",
            query_work_sidecar=query_path,
        )
        paper_curves = {
            method: base._paper_curve(references[(regime, method)])
            for method in base.PAPER_METHODS
        }
        paper_resources = [
            next(row for row in resource_rows[regime] if row["method"] == method)
            for method in base.PAPER_METHODS
        ]
        regimes.append(
            {
                "regime": regime,
                "display": base.REGIME_DISPLAY[regime],
                "curves": {
                    "repaired_l25": {
                        "points": [{"k": k, "error": value} for k, value in curve.points],
                        "marker_k": curve.marker_k,
                        "marker_error": curve.marker_error,
                        "source_json": curve.source_json,
                        "source_sha256": curve.source_sha256,
                    },
                    **{
                        method: {
                            "points": [{"k": k, "error": value} for k, value in item.points],
                            "marker_k": item.marker_k,
                            "marker_error": item.marker_error,
                            "source_json": item.source_json,
                            "source_sha256": item.source_sha256,
                        }
                        for method, item in paper_curves.items()
                    },
                },
                "resource_table_rows": [repaired_resource, *paper_resources],
                "completion_checks": row_checks,
                "source_files": {
                    "result": {"path": base._rel(result_path), "sha256": base._sha256(result_path)},
                    "summary": {"path": base._rel(summary_path), "sha256": base._sha256(summary_path)},
                    "query_work": {"path": base._rel(query_path), "sha256": base._sha256(query_path)},
                    "wrapper": {"path": base._rel(wrapper_path), "sha256": base._sha256(wrapper_path)},
                },
            }
        )
    return {
        "schema": SCHEMA,
        "status": "retrieved_6_of_6_r15_capped",
        "completed_regime_count": len(regimes),
        "expected_regime_count": len(base.REGIME_ORDER),
        "cluster": 8775444,
        "scientific_contract_hash": EXPECTED_CONTRACT_HASH,
        "execution_profile": trial_manifest["execution_profile"],
        "policy": policy,
        "contract_checks": checks,
        "trial_manifest": {"path": base._rel(TRIAL_MANIFEST), "sha256": base._sha256(TRIAL_MANIFEST)},
        "repair_manifest": {"path": base._rel(REPAIR_MANIFEST), "sha256": base._sha256(REPAIR_MANIFEST)},
        "paper_i_reference": {
            "path": base._rel(base.PAPER_I_REFERENCE_JSON),
            "sha256": base._sha256(base.PAPER_I_REFERENCE_JSON),
        },
        "paper_i_qiskit_resources": {
            "path": base._rel(base.QISKIT_TABLE_JSON),
            "sha256": base._sha256(base.QISKIT_TABLE_JSON),
        },
        "regimes": regimes,
    }


def _plot(evidence: Mapping[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {method: base.STYLE[method] for method in base.PAPER_METHODS}
    styles["repaired_l25"] = REPAIRED_STYLE
    fig = plt.figure(figsize=(13.6, 8.25), constrained_layout=False)
    outer = fig.add_gridspec(
        2, 3, left=0.055, right=0.992, bottom=0.055, top=0.91, wspace=0.19, hspace=0.28
    )
    for index, regime_row in enumerate(evidence["regimes"]):
        inner = outer[index // 3, index % 3].subgridspec(
            2, 1, height_ratios=(2.45, 1.05), hspace=0.115
        )
        error_ax = fig.add_subplot(inner[0])
        table_ax = fig.add_subplot(inner[1])
        for role in ("append", "geo", "snake", "repaired_l25"):
            curve = regime_row["curves"][role]
            style = styles[role]
            error_ax.plot(
                [point["k"] for point in curve["points"]],
                [point["error"] for point in curve["points"]],
                color=style["color"], linewidth=style["width"], alpha=0.97,
            )
            error_ax.scatter(
                [curve["marker_k"]], [curve["marker_error"]], color=style["color"],
                marker=style["marker"], s=50 if role == "snake" else 35,
                edgecolor="white", linewidth=0.35, zorder=5,
            )
        error_ax.set_yscale("log")
        error_ax.set_xlim(left=0)
        error_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
        error_ax.grid(True, which="major", alpha=0.2, linewidth=0.45)
        error_ax.tick_params(axis="both", labelsize=6.4)
        error_ax.set_ylabel(r"$|\Delta E|$", fontsize=7.2)
        error_ax.set_title(str(regime_row["display"]), fontsize=8.4, pad=2.5)
        table_rows = []
        for row in regime_row["resource_table_rows"]:
            table_rows.append(
                [
                    "Repaired L25" if row["method"] == "repaired_l25_snake" else base.RESOURCE_METHOD_DISPLAY[row["method"]],
                    f"{int(row['k_pl'])}", f"{float(row['abs_delta_e']):.2e}",
                    f"{int(row['N2q']):,}", f"{int(row['D2q']):,}",
                    f"{int(row['Dc']):,}", f"{int(row['S']):,}",
                ]
            )
        table_ax.set_axis_off()
        table = table_ax.table(
            cellText=table_rows,
            colLabels=("Method", r"$k$", r"$|\Delta E|$", r"$N_{2q}$", r"$D_{2q}$", r"$D_c$", r"$S$"),
            cellLoc="right", colLoc="right",
            colWidths=(0.27, 0.07, 0.15, 0.12, 0.12, 0.12, 0.15),
            bbox=(0.0, 0.02, 1.0, 0.95),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(4.45)
        for (row_index, column_index), cell in table.get_celld().items():
            cell.visible_edges = "horizontal"
            cell.set_edgecolor("#A8A8A8")
            cell.set_linewidth(0.28)
            cell.PAD = 0.025
            if row_index == 0:
                cell.set_text_props(weight="semibold", color="#222222")
                cell.set_facecolor("#F7F7F7")
            elif column_index == 0:
                role = ("repaired_l25", "snake", "geo", "append")[row_index - 1]
                cell.set_text_props(ha="left", color=styles[role]["color"], weight="medium")
    handles = [
        Line2D([0], [0], color=styles[role]["color"], linewidth=styles[role]["width"],
               marker=styles[role]["marker"], markersize=5.2, label=styles[role]["label"])
        for role in ("repaired_l25", "snake", "geo", "append")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=7.0, bbox_to_anchor=(0.52, 0.985))
    fig.text(
        0.5, 0.018,
        "Retrieved L25 trajectories used a 15-controller-round cap; Paper-I comparators retain the locked page-1 selected-prefix/plateau convention",
        ha="center", fontsize=6.7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_tex(path: Path, *, image_path: Path, provenance_path: Path, evidence: Mapping[str, Any]) -> None:
    original = ORIGINAL_TEX.read_text(encoding="utf-8")
    marker = r"\end{document}"
    if original.count(marker) != 1:
        raise ValueError("Original two-page TeX has an unexpected document terminator")
    machine_comment = json.dumps(
        {
            "schema": SCHEMA,
            "provenance_json": base._rel(provenance_path),
            "cluster": evidence["cluster"],
            "status": evidence["status"],
            "scientific_contract_hash": evidence["scientific_contract_hash"],
            "execution_profile": evidence["execution_profile"],
            "policy": evidence["policy"],
            "manuscript_edited": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    page_three = rf"""
\newpage
% BEGIN_MACHINE_READABLE_REPAIRED_L25_PAGE
% {machine_comment}
% END_MACHINE_READABLE_REPAIRED_L25_PAGE
\begin{{center}}
{{\large\bfseries Retrieved Early-L25 fixed policy: R15-capped six-regime comparison}}\\[-0.15ex]
{{\fontsize{{5.45}}{{6.1}}\selectfont \textbf{{Manifest:}} CHTC cluster 8775444; 6/6 jobs complete; controller cap $=15$ (three selector-exhausted earlier); profile \texttt{{wave11\_legal\_fixed\_policy\_v1}}; combinatorial $B_{{\max}}=2$, $L_{{\rm search}}=25$; macro P1/P2 $=64/48$; child P1/P2 $=128/64$; $\lambda_{{\rm add}}=0$; Powell inner refit.}}\\[-0.1ex]
\includegraphics[width=0.998\linewidth,height=7.55in,keepaspectratio]{{{image_path.resolve().as_posix()}}}
\end{{center}}
{marker}
"""
    path.write_text(original.replace(marker, page_three), encoding="utf-8")


def build(output_dir: Path = DEFAULT_OUTPUT_DIR, stem: str = STEM) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _load_evidence(output_dir)
    image_path = output_dir / f"{stem}-page3.png"
    provenance_path = output_dir / f"{stem}.json"
    tex_path = output_dir / f"{stem}.tex"
    _plot(evidence, image_path)
    evidence["artifacts"] = {
        "page3_png": base._rel(image_path),
        "tex": base._rel(tex_path),
        "pdf": base._rel(tex_path.with_suffix(".pdf")),
    }
    provenance_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tex(tex_path, image_path=image_path, provenance_path=provenance_path, evidence=evidence)
    pdf_path = base._compile_latex(tex_path)
    if base._page_count(pdf_path) != 3:
        raise ValueError("Three-page original-plus-repaired-L25 PDF contract failed")
    return {
        "pdf": str(pdf_path),
        "page3_png": str(image_path),
        "provenance": str(provenance_path),
        "tex": str(tex_path),
        "status": evidence["status"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=STEM)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.output_dir, args.stem), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
