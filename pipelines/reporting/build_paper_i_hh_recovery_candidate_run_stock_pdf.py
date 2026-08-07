#!/usr/bin/env python3
"""Build the Paper-I HH recovery/candidate run-stock PDF.

The report starts as a pre-submit/run-stock manifest and is intended to be
refreshed as CHTC results arrive.  It deliberately reports evidence status only;
promotion decisions remain user-owned.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf" / "paper_i_hh_recovery_candidate_run_stock_20260705"
DEFAULT_STEM = "paper_i_hh_recovery_candidate_run_stock_20260705"
DEFAULT_AUDIT_JSON = REPO_ROOT / "output" / "preflight" / "paper_i_hh_recovery_candidate_20260705_tsv_audit.json"

INPUT_BATCHES = (
    "paper_i_hh_recovery_candidate_20260705_powell_nobatch_wave0",
    "paper_i_hh_recovery_candidate_20260705_powell_nobatch_wave1",
    "paper_i_hh_recovery_candidate_20260705_powell_nobatch_wave2",
    "paper_i_hh_recovery_candidate_20260705_spsa_nobatch_wave0",
    "paper_i_hh_recovery_candidate_20260705_spsa_nobatch_wave1",
    "paper_i_hh_recovery_candidate_20260705_spsa_nobatch_wave2",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_nobatch_wave0",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_nobatch_wave1",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_nobatch_wave2",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_comparators_wave0",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_comparators_wave1",
    "paper_i_hh_recovery_candidate_20260705_rotosolve_comparators_wave2",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _short_batch(batch_id: str) -> str:
    prefix = "paper_i_hh_recovery_candidate_20260705_"
    text = batch_id.replace(prefix, "")
    replacements = {
        "powell_nobatch_wave": "powell nb w",
        "spsa_nobatch_wave": "spsa nb w",
        "rotosolve_nobatch_wave": "roto nb w",
        "rotosolve_comparators_wave": "roto comp w",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _short_matrix(label: str) -> str:
    return {
        "A_native_staged_singleton_hard_guard": "A1 phase3 cap3",
        "B_common_phase0_singleton_hard_guard": "B1 append child",
        "C_macro_only": "C macro",
    }.get(label, label)


def _short_route(route: str) -> str:
    return {
        "nobatch_anchor_cap3_metricprune_beam0p005": "anchor nb",
        "rotosolve_historical_comparator": "hist comp",
    }.get(route, route)


def _short_child(row: "ReportRow") -> str:
    if row.child_policy == "native_phase3_singleton":
        return f"phase3 cap {row.subset_size or '--'}"
    if row.child_policy == "common_phase0_singleton":
        return f"phase0 cap {row.subset_size or '1'}"
    if row.child_policy == "macro_only":
        return "macro"
    return row.child_policy or "--"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@dataclass
class ReportRow:
    batch_id: str
    record_id: str
    optimizer: str
    wave: str
    regime: str
    method: str
    matrix_label: str
    route_variant: str
    maxiter: str
    depth_cap: str
    pool_contract: str
    child_policy: str
    subset_size: str
    metric_prune: str
    beam_lambda: str
    batching: str
    status: str
    result_json: str
    records_tsv: str
    manifest_json: str


def _setting_changed(row: Mapping[str, str], key: str) -> str:
    try:
        payload = json.loads(row.get("settings_changed_json") or "{}")
    except Exception:
        payload = {}
    return str(payload.get(key) or "")


def _row_status(row: Mapping[str, str], result_path: Path) -> str:
    if (row.get("runnable") or "true").lower() == "false":
        return "blocked"
    if result_path.exists():
        return "fetched"
    return "ready-for-submit"


def load_rows(input_root: Path) -> tuple[list[ReportRow], list[dict[str, Any]]]:
    rows: list[ReportRow] = []
    manifests: list[dict[str, Any]] = []
    for batch_id in INPUT_BATCHES:
        batch_dir = input_root / batch_id
        records_tsv = batch_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
        manifest_json = batch_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
        if not records_tsv.exists() or not manifest_json.exists():
            continue
        manifest = _read_json(manifest_json)
        manifests.append(
            {
                "batch_id": batch_id,
                "manifest_json": _rel(manifest_json),
                "record_count": manifest.get("record_count"),
                "runnable_record_count": manifest.get("runnable_record_count"),
                "blocked_record_count": manifest.get("blocked_record_count"),
                "source_contract": manifest.get("source_contract"),
                "run_stock": manifest.get("run_stock"),
            }
        )
        for row in _load_tsv(records_tsv):
            result_json = REPO_ROOT / str(row.get("result_json_rel") or "")
            route_variant = row.get("route_variant") or ""
            batching = "disabled" if route_variant.startswith("nobatch") else str(row.get("phase2_batch_selection_mode") or "--")
            rows.append(
                ReportRow(
                    batch_id=batch_id,
                    record_id=row.get("record_id") or "",
                    optimizer=row.get("optimizer_overlay_id") or row.get("adapt_optimizer_kind") or "",
                    wave=row.get("regime_wave_label") or row.get("regime_wave_index") or "",
                    regime=row.get("display_regime") or "",
                    method=row.get("method_key") or "",
                    matrix_label=row.get("matrix_label") or "",
                    route_variant=route_variant,
                    maxiter=row.get("budget") or "",
                    depth_cap=row.get("max_depth") or "",
                    pool_contract=row.get("pool_contract") or "",
                    child_policy=row.get("child_policy") or "",
                    subset_size=row.get("child_subset_size") or row.get("snake_phase3_runtime_split_max_subset_size") or "",
                    metric_prune=_setting_changed(row, "--phase1-prune-schur-nomination-route"),
                    beam_lambda=row.get("adapt_beam_lambda") or _setting_changed(row, "--adapt-beam-lambda"),
                    batching=batching,
                    status=_row_status(row, result_json),
                    result_json=_rel(result_json),
                    records_tsv=_rel(records_tsv),
                    manifest_json=_rel(manifest_json),
                )
            )
    return rows, manifests


def write_sidecars(
    rows: Sequence[ReportRow],
    manifests: Sequence[Mapping[str, Any]],
    *,
    audit_json: Path,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _read_json(audit_json) if audit_json.exists() else {"ok": False, "missing": _rel(audit_json)}
    payload = {
        "schema": "paper_i_hh_recovery_candidate_run_stock_report_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_submit_or_partial_results",
        "promotion_status": "not_promoted_user_decides",
        "audit_json": _rel(audit_json),
        "audit": audit,
        "manifests": list(manifests),
        "rows": [asdict(row) for row in rows],
    }
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    md_lines = [
        "# Paper-I HH Recovery/Candidate Run Stock",
        "",
        f"Generated UTC: `{payload['generated_utc']}`",
        f"Audit: `{_rel(audit_json)}` (`ok={audit.get('ok')}`)",
        "",
        "This is evidence status only. Promotion decisions are user-owned.",
        "",
        f"Rows: `{len(rows)}`",
        f"Fetched results: `{sum(1 for row in rows if row.status == 'fetched')}`",
        f"Ready for submit: `{sum(1 for row in rows if row.status == 'ready-for-submit')}`",
        "",
        "Sidecars:",
        f"- `{_rel(json_path)}`",
        f"- `{_rel(csv_path)}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def write_tex(
    rows: Sequence[ReportRow],
    *,
    audit_json: Path,
    json_path: Path,
    csv_path: Path,
    md_path: Path,
    output_dir: Path,
    stem: str,
) -> Path:
    audit = _read_json(audit_json) if audit_json.exists() else {"ok": False}
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    lines = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=0.55in,landscape]{geometry}",
        r"\usepackage{booktabs,longtable,xcolor,hyperref,url}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\LTpre}{4pt}",
        r"\setlength{\LTpost}{4pt}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Paper-I HH Recovery/Candidate Run Stock}\\",
        r"{\small Source-locked no-batch anchors and ROTOSOLVE historical-pool comparators}\\",
        r"\end{center}",
        r"\textbf{Generated UTC:} " + _tex_escape(datetime.now(timezone.utc).isoformat()) + r"\\",
        r"\textbf{Run class:} candidate pending completed evidence; not promoted.\\",
        r"\textbf{Audit:} " + _tex_escape(_rel(audit_json)) + r" (ok=" + _tex_escape(audit.get("ok")) + r", errors=" + _tex_escape(audit.get("error_count")) + r")\\",
        r"\textbf{Sidecars:}\\",
        r"\footnotesize\path{" + _rel(json_path) + r"}\\",
        r"\path{" + _rel(csv_path) + r"}\\",
        r"\path{" + _rel(md_path) + r"}\\",
        r"\normalsize",
        r"\textbf{Scope:} SNAKE across all six HH regimes and POWELL/SPSA/ROTOSOLVE; ROTOSOLVE-only Geo/append comparators.\\",
        r"\textbf{User-approved changes from visible-row baseline:} Phase-III archival child-set cap 3, metric-prune route, adapt beam lambda 0.005, worker parallelism. Batching remains disabled for these anchors.\\",
        r"\textbf{PDF status semantics:} ready-for-submit means the local CHTC records are audited but not yet uploaded/submitted or fetched. fetched means the expected local result JSON path exists.\\",
        "",
        r"\vspace{3pt}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Status & Count \\",
        r"\midrule",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(_tex_escape(status) + " & " + str(count) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{6pt}",
            r"\scriptsize",
            r"\begin{longtable}{p{0.075\linewidth}p{0.06\linewidth}p{0.095\linewidth}p{0.055\linewidth}p{0.085\linewidth}p{0.075\linewidth}p{0.085\linewidth}p{0.055\linewidth}p{0.075\linewidth}p{0.10\linewidth}}",
            r"\toprule",
            r"Batch & Opt & Regime & Method & Matrix & Route & Child & Beam & Batch & Status \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"Batch & Opt & Regime & Method & Matrix & Route & Child & Beam & Batch & Status \\",
            r"\midrule",
            r"\endhead",
        ]
    )
    for row in rows:
        lines.append(
            " & ".join(
                _tex_escape(value)
                for value in (
                    _short_batch(row.batch_id),
                    row.optimizer,
                    row.regime,
                    row.method,
                    _short_matrix(row.matrix_label),
                    _short_route(row.route_variant),
                    _short_child(row),
                    row.beam_lambda or "--",
                    row.batching,
                    row.status,
                )
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\normalsize",
            r"\newpage",
            r"\section*{Provenance Notes}",
            r"\begin{itemize}",
            r"\item Qiskit cost columns and iteration plots are intentionally absent until result JSONs are fetched; this pre-submit report records the run-stock contract.",
            r"\item Later refreshes of this report family must preserve Qiskit compiled costs as terminal compiled circuit costs and S-work as algorithmic work decomposition.",
            r"\item Existing manuscript/table evidence is preserved; this report does not promote or replace visible values.",
            r"\end{itemize}",
            r"\end{document}",
        ]
    )
    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def build_report(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stem: str = DEFAULT_STEM,
    audit_json: Path = DEFAULT_AUDIT_JSON,
) -> dict[str, str]:
    input_root = REPO_ROOT / "chtc" / "phase3_optuna" / "input"
    rows, manifests = load_rows(input_root)
    if not rows:
        raise RuntimeError("No run-stock rows found.")
    json_path, csv_path, md_path = write_sidecars(rows, manifests, audit_json=audit_json, output_dir=output_dir, stem=stem)
    tex_path = write_tex(rows, audit_json=audit_json, json_path=json_path, csv_path=csv_path, md_path=md_path, output_dir=output_dir, stem=stem)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "pdf": str(output_dir / f"{stem}.pdf"),
        "tex": str(tex_path),
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "rows": str(len(rows)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(build_report(output_dir=args.output_dir, stem=args.stem, audit_json=args.audit_json), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
