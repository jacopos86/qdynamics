#!/usr/bin/env python3
"""Paper III QSE results PDF builder (source-map-audited, fail-closed).

RECONSTRUCTION (2026-08-19): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its test without it. This
implementation is reconstructed against the committed behavioral spec in
``test/test_paper_iii_qse_results_pdf.py``.

The builder consumes one or more audited ``paper_iii_qse_source_map_v1``
files, re-validates every referenced source (hash, schema, controller
boundary, production gates — fail closed on any drift), and renders a LaTeX
results report with newest-first sections, machine-readable provenance
comment blocks, and a parameter/provenance manifest page. Sidecars: a
report manifest JSON, an audit JSON, an agent run manifest declaring the
read-only scope, and a Markdown command log. Compilation is policy-gated:
``skip`` never compiles, ``auto`` compiles when a LaTeX engine exists and
records a graceful skip otherwise, ``require`` fails without one. This
results PDF is deliberately separate from the Paper III manuscript.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.source_maps import (
    load_paper_iii_qse_source_map,
    sha256_file,
    validate_paper_iii_qse_source_map,
)

REPORT_SCHEMA_VERSION = "paper_iii_qse_results_report_v1"
REPORT_AUDIT_SCHEMA_VERSION = "paper_iii_qse_results_report_audit_v1"

COMPILE_AUTO = "auto"
COMPILE_REQUIRE = "require"
COMPILE_SKIP = "skip"
_COMPILE_MODES = (COMPILE_AUTO, COMPILE_REQUIRE, COMPILE_SKIP)

DEFAULT_WORK_DIR = Path("output/paper_iii_qse_results")
DEFAULT_TEX = DEFAULT_WORK_DIR / "paper_iii_qse_results.tex"
DEFAULT_PDF = Path("output/pdf/paper_iii_qse_results.pdf")
DEFAULT_REPORT_MANIFEST_JSON = DEFAULT_WORK_DIR / "paper_iii_qse_results.manifest.json"
DEFAULT_AUDIT_JSON = DEFAULT_WORK_DIR / "paper_iii_qse_results.audit.json"
DEFAULT_RUN_MANIFEST_JSON = DEFAULT_WORK_DIR / "run_manifest.json"
DEFAULT_COMMAND_LOG_MD = DEFAULT_WORK_DIR / "command_log.md"

_LATEX_ENGINES = ("latexmk", "pdflatex", "tectonic")


class PaperIIIQSEResultsReportError(RuntimeError):
    """Raised when the results report cannot be built fail-closed."""


def default_output_paths() -> dict[str, Path]:
    return {
        "work_dir": DEFAULT_WORK_DIR,
        "tex": DEFAULT_TEX,
        "pdf": DEFAULT_PDF,
        "report_manifest_json": DEFAULT_REPORT_MANIFEST_JSON,
        "audit_json": DEFAULT_AUDIT_JSON,
        "run_manifest_json": DEFAULT_RUN_MANIFEST_JSON,
        "command_log_md": DEFAULT_COMMAND_LOG_MD,
    }


@dataclass(frozen=True)
class PaperIIIQSEResultsReportConfig:
    """Inputs and output locations for one results-report build."""

    source_map_jsons: tuple[Path, ...]
    base_dir: Path
    output_tex: Path = DEFAULT_TEX
    output_pdf: Path = DEFAULT_PDF
    report_manifest_json: Path = DEFAULT_REPORT_MANIFEST_JSON
    audit_json: Path = DEFAULT_AUDIT_JSON
    run_manifest_json: Path = DEFAULT_RUN_MANIFEST_JSON
    command_log_md: Path = DEFAULT_COMMAND_LOG_MD
    compile_mode: str = COMPILE_AUTO
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not tuple(self.source_map_jsons):
            raise PaperIIIQSEResultsReportError("At least one source map JSON is required.")
        if str(self.compile_mode) not in _COMPILE_MODES:
            raise PaperIIIQSEResultsReportError(
                f"compile_mode must be one of {list(_COMPILE_MODES)!r}; got {self.compile_mode!r}."
            )


def _escape(text: Any) -> str:
    out = str(text)
    for raw, escaped in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        out = out.replace(raw, escaped)
    return out


def _machine_block(name: str, payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, indent=1, sort_keys=True)
    commented = "\n".join(f"% {line}" if line else "%" for line in body.splitlines())
    return (
        f"% BEGIN_MACHINE_READABLE_{name}\n{commented}\n% END_MACHINE_READABLE_{name}\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PaperIIIQSEResultsReportError(f"Source payload must be a JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_line(command: Any) -> str:
    if command is None:
        return ""
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command)


def _source_sections(source: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    """Human-readable body lines for one audited source, by role."""

    role = str(source.get("source_map_role"))
    lines: list[str] = []
    case_id = payload.get("case_id") or payload.get("regime") or payload.get("run_tag") or source.get(
        "source_path"
    )
    lines.append(rf"\subsection*{{{_escape(case_id)} ({_escape(role)})}}")
    run_class = source.get("run_class")
    approval = source.get("approval_status")
    if run_class is not None or approval is not None:
        lines.append(
            f"Run class: {_escape(run_class)}; approval: {_escape(approval)}."
        )
    tier = source.get("compatibility_tier")
    if tier:
        lines.append(f"Compatibility tier: {_escape(tier)}.")
    if role == "qse_manifest":
        conductivity = payload.get("qse_conductivity_response_v1")
        if isinstance(conductivity, Mapping):
            channels = conductivity.get("channels") or []
            lines.append(
                f"Reported current/conductivity channels: {len(list(channels))} "
                f"(policy {_escape((conductivity.get('contact_policy') or {}).get('name'))})."
            )
        green = payload.get("qse_green_function_v1")
        if isinstance(green, Mapping):
            summary = green.get("summary") or {}
            lines.append(
                f"Green-function modes: {_escape(summary.get('mode_count'))} across "
                f"{_escape(summary.get('sector_count'))} sectors "
                f"({_escape(summary.get('solved_sector_count'))} solved)."
            )
        diagnostics = payload.get("diagnostics") or {}
        lines.append(
            f"Basis size {_escape(diagnostics.get('basis_size'))}, retained rank "
            f"{_escape(diagnostics.get('retained_rank'))}."
        )
    if role == "aggregate":
        summary = payload.get("summary") or {}
        lines.append(f"Aggregate rows: {_escape(summary.get('row_count'))}.")
        lines.append(
            f"rows with conductivity/current payloads: "
            f"{_escape(summary.get('rows_with_conductivity_payload'))}; "
            f"rows with Green functions: "
            f"{_escape(summary.get('rows_with_green_function_payload'))}; "
            f"rows with neutral response payloads: "
            f"{_escape(summary.get('rows_with_response_payload'))}."
        )
    generated = payload.get("generated_utc")
    if generated:
        lines.append(f"Generated (UTC): {_escape(generated)}.")
    return lines


def _resolve_compile(
    config: PaperIIIQSEResultsReportConfig,
) -> tuple[str, str | None]:
    mode = str(config.compile_mode)
    if mode == COMPILE_SKIP:
        return "skipped_disabled", None
    engine = next((name for name in _LATEX_ENGINES if shutil.which(name)), None)
    if engine is None:
        if mode == COMPILE_REQUIRE:
            raise PaperIIIQSEResultsReportError(
                "No LaTeX engine available (searched "
                + ", ".join(_LATEX_ENGINES)
                + ") and compile_mode=require."
            )
        return "skipped_tex_unavailable", None
    return "pending", engine


def _compile_pdf(
    *, engine: str, tex_path: Path, pdf_path: Path
) -> tuple[str, list[str]]:
    work_dir = tex_path.parent
    if engine == "latexmk":
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path.name]
    elif engine == "tectonic":
        command = ["tectonic", tex_path.name]
    else:
        command = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    log: list[str] = [" ".join(command)]
    completed = subprocess.run(
        command, cwd=str(work_dir), capture_output=True, text=True, timeout=600
    )
    if completed.returncode != 0:
        raise PaperIIIQSEResultsReportError(
            f"LaTeX compile failed with {engine} (exit {completed.returncode}); "
            f"see {work_dir} for logs."
        )
    built = work_dir / (tex_path.stem + ".pdf")
    if not built.is_file():
        raise PaperIIIQSEResultsReportError(f"LaTeX reported success but {built} is missing.")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, pdf_path)
    return "compiled", log


def build_paper_iii_qse_results_report(
    config: PaperIIIQSEResultsReportConfig,
    *,
    command: Any = None,
) -> dict[str, Any]:
    """Audit the source maps, render the tex + sidecars, optionally compile."""

    command_line = _command_line(command)
    generated_utc = (
        _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    base_dir = Path(config.base_dir)

    source_map_references: list[dict[str, Any]] = []
    audited_sources: list[dict[str, Any]] = []
    for map_path in config.source_map_jsons:
        map_path = Path(map_path)
        source_map = load_paper_iii_qse_source_map(map_path)
        # Fail closed before rendering anything: hash/schema/boundary/gate
        # drift in any referenced source aborts the report.
        validate_paper_iii_qse_source_map(source_map, base_dir=base_dir)
        map_sha = sha256_file(map_path)
        source_map_references.append(
            {
                "path": str(map_path),
                "map_id": source_map.get("map_id"),
                "source_map_sha256": map_sha,
                "source_count": int(source_map.get("source_count") or 0),
            }
        )
        for record in source_map.get("sources") or []:
            source_path = str(record.get("source_path"))
            payload = _load_json(base_dir / source_path)
            audited_sources.append(
                {
                    "source_id": f"{record.get('source_map_role')}:{source_path}",
                    "source_map_role": str(record.get("source_map_role")),
                    "source_path": source_path,
                    "source_sha256": str(record.get("source_sha256")),
                    "schema_version": record.get("schema_version"),
                    "run_class": record.get("run_class"),
                    "approval_status": record.get("approval_status"),
                    "compatibility_tier": record.get("compatibility_tier"),
                    "generated_utc": payload.get("generated_utc"),
                    "source_map_sha256": map_sha,
                    "map_id": source_map.get("map_id"),
                    "_payload": payload,
                }
            )

    audited_sources.sort(key=lambda item: str(item.get("generated_utc") or ""), reverse=True)
    role_counts: dict[str, int] = {}
    for source in audited_sources:
        role_counts[source["source_map_role"]] = role_counts.get(source["source_map_role"], 0) + 1

    report_block = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "command_line": command_line,
        "source_maps": [
            {key: value for key, value in reference.items()} for reference in source_map_references
        ],
        "sources": [
            {key: value for key, value in source.items() if key != "_payload"}
            for source in audited_sources
        ],
    }

    tex_lines: list[str] = []
    tex_lines.append(_machine_block("PAPER_III_QSE_RESULTS_REPORT", report_block))
    tex_lines.append(r"\documentclass[11pt]{article}")
    tex_lines.append(r"\usepackage[margin=2.2cm]{geometry}")
    tex_lines.append(r"\begin{document}")
    tex_lines.append(r"\section*{Paper III QSE results report}")
    tex_lines.append(
        "This results PDF is separate from the Paper III manuscript; it renders "
        "audited evidence records only, newest first, and never edits manuscript "
        "or run state."
    )
    tex_lines.append(
        "Approval policy: candidate and user-review-required records are not promoted "
        "by this report; promotion requires explicit user approval outside the builder."
    )
    tex_lines.append("")
    for source in audited_sources:
        section_block = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source": {
                key: value for key, value in source.items() if key != "_payload"
            },
            "source_map": {
                "map_id": source.get("map_id"),
                "sha256": source.get("source_map_sha256"),
            },
        }
        tex_lines.append(_machine_block("PAPER_III_QSE_RESULT_SECTION", section_block))
        tex_lines.extend(_source_sections(source, source["_payload"]))
        tex_lines.append("")

    tex_lines.append(r"\clearpage")
    tex_lines.append(r"\section*{PARAMETER / PROVENANCE MANIFEST}")
    tex_lines.append(r"\begin{itemize}")
    tex_lines.append(rf"\item Source-map count: {len(source_map_references)}")
    tex_lines.append(rf"\item Audited source count: {len(audited_sources)}")
    tex_lines.append(
        r"\item Approval policy: candidate and user-review-required records are "
        r"rendered as evidence only and are not promoted."
    )
    tex_lines.append(
        r"\item Exact/reference data policy: exact and ED references are "
        r"diagnostic reporting only and never feed controller decisions."
    )
    tex_lines.append(
        r"\item Newest-first sort: sections are ordered by source generated\_utc, "
        r"descending."
    )
    tex_lines.append(rf"\item Report generated (UTC): {_escape(generated_utc)}")
    tex_lines.append(rf"\item Command line: {_escape(command_line) or '(none recorded)'}")
    tex_lines.append(r"\end{itemize}")
    tex_lines.append(r"\end{document}")

    config.output_tex.parent.mkdir(parents=True, exist_ok=True)
    config.output_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    compile_status, engine = _resolve_compile(config)
    compile_log: list[str] = []
    if compile_status == "pending":
        compile_status, compile_log = _compile_pdf(
            engine=str(engine), tex_path=config.output_tex, pdf_path=config.output_pdf
        )

    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "command_line": command_line,
        "compile": {"mode": str(config.compile_mode), "status": compile_status, "engine": engine},
        "outputs": {
            "tex": str(config.output_tex),
            "pdf": str(config.output_pdf),
            "report_manifest_json": str(config.report_manifest_json),
            "audit_json": str(config.audit_json),
            "run_manifest_json": str(config.run_manifest_json),
            "command_log_md": str(config.command_log_md),
        },
        "source_maps": report_block["source_maps"],
        "summary": {
            "source_role_counts": role_counts,
            "source_count": len(audited_sources),
        },
    }
    _write_json(config.report_manifest_json, manifest)

    audit = {
        "schema_version": REPORT_AUDIT_SCHEMA_VERSION,
        "ok": True,
        "generated_utc": generated_utc,
        "source_map_references": report_block["source_maps"],
        "source_count": len(audited_sources),
    }
    _write_json(config.audit_json, audit)

    run_manifest = {
        "schema_version": "agent_run_manifest_v1",
        "generated_utc": generated_utc,
        "command_line": command_line,
        "read_only": True,
        "no_run_or_chtc_work": True,
        "no_pdf_build": compile_status not in {"compiled"},
        "no_manuscript_edit": True,
        "compile_status": compile_status,
    }
    _write_json(config.run_manifest_json, run_manifest)

    config.command_log_md.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        "# Paper III QSE results report command log",
        "",
        f"- generated_utc: {generated_utc}",
        f"- command: `{command_line or '(none recorded)'}`",
        f"- compile: {compile_status}" + (f" ({engine})" if engine else ""),
    ]
    log_lines.extend(f"- compile step: `{line}`" for line in compile_log)
    config.command_log_md.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the audited Paper III QSE results report (tex + sidecars, optional PDF)."
    )
    parser.add_argument("--source-map-json", action="append", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--report-manifest-json", type=Path, default=DEFAULT_REPORT_MANIFEST_JSON)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--run-manifest-json", type=Path, default=DEFAULT_RUN_MANIFEST_JSON)
    parser.add_argument("--command-log-md", type=Path, default=DEFAULT_COMMAND_LOG_MD)
    parser.add_argument(
        "--compile-mode", choices=list(_COMPILE_MODES), default=COMPILE_AUTO
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Shorthand for --compile-mode skip.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)
    args = build_parser().parse_args(raw_argv)
    compile_mode = COMPILE_SKIP if args.no_compile else str(args.compile_mode)
    config = PaperIIIQSEResultsReportConfig(
        source_map_jsons=tuple(args.source_map_json),
        base_dir=args.base_dir,
        output_tex=args.output_tex,
        output_pdf=args.output_pdf,
        report_manifest_json=args.report_manifest_json,
        audit_json=args.audit_json,
        run_manifest_json=args.run_manifest_json,
        command_log_md=args.command_log_md,
        compile_mode=compile_mode,
    )
    command = [Path(sys.argv[0]).name, *raw_argv] if argv is None else list(raw_argv)
    build_paper_iii_qse_results_report(config, command=command)
    return 0


__all__ = [
    "COMPILE_AUTO",
    "COMPILE_REQUIRE",
    "COMPILE_SKIP",
    "DEFAULT_AUDIT_JSON",
    "DEFAULT_COMMAND_LOG_MD",
    "DEFAULT_PDF",
    "DEFAULT_REPORT_MANIFEST_JSON",
    "DEFAULT_RUN_MANIFEST_JSON",
    "DEFAULT_TEX",
    "DEFAULT_WORK_DIR",
    "REPORT_AUDIT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "PaperIIIQSEResultsReportConfig",
    "PaperIIIQSEResultsReportError",
    "build_paper_iii_qse_results_report",
    "build_parser",
    "default_output_paths",
    "main",
]
