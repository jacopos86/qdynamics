"""Reporting-facing audit wrapper for Paper-III QSE source maps.

This module reads source maps and their referenced JSON sources only.  It does
not launch runs, mutate CHTC inputs, build PDFs, or edit manuscript files.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.qse_spectra.source_maps import (
    PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION,
    PaperIIIQSESourceMapError,
    audit_paper_iii_qse_source_map,
    load_paper_iii_qse_source_map,
    validate_paper_iii_qse_source_map,
)


class PaperIIIQSEAuditError(RuntimeError):
    """Raised when a Paper-III QSE source-map audit fails closed."""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = dict(report or {})


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def run_paper_iii_qse_source_map_audit(
    source_map_json: Path,
    *,
    base_dir: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Audit one ``paper_iii_qse_source_map_v1`` file and fail on errors."""

    source_map = load_paper_iii_qse_source_map(Path(source_map_json))
    report = audit_paper_iii_qse_source_map(source_map, base_dir=base_dir)
    report = {
        **report,
        "schema_version": PAPER_III_QSE_SOURCE_MAP_AUDIT_SCHEMA_VERSION,
        "mode": {
            "read_only": True,
            "no_run_or_chtc_work": True,
            "no_pdf_build": True,
            "no_manuscript_edit": True,
        },
        "inputs": {"source_map_json": str(source_map_json)},
    }
    if output_json is not None:
        _atomic_write_json(Path(output_json), report)
    if not report.get("ok"):
        failures = report.get("failures") if isinstance(report.get("failures"), list) else []
        rendered = "; ".join(
            f"{item.get('code')}: {item.get('message')}" for item in failures[:5] if isinstance(item, Mapping)
        )
        raise PaperIIIQSEAuditError(
            "Paper III QSE source-map audit failed" + (f": {rendered}" if rendered else ""),
            report=report,
        )
    # Keep this call for fail-closed parity with the lower-level validator and
    # to attach the richer failures to PaperIIIQSESourceMapError if invariants
    # ever diverge between audit and validate.
    try:
        validate_paper_iii_qse_source_map(source_map, base_dir=base_dir)
    except PaperIIIQSESourceMapError as exc:  # pragma: no cover - defensive parity guard
        raise PaperIIIQSEAuditError(str(exc), report=report) from exc
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Paper III QSE source maps without running jobs or building PDFs.")
    parser.add_argument("--source-map-json", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_paper_iii_qse_source_map_audit(
            args.source_map_json,
            base_dir=args.base_dir,
            output_json=args.output_json,
        )
    except PaperIIIQSEAuditError as exc:
        report = exc.report
        if not report and args.output_json is not None and Path(args.output_json).exists():
            report = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_count": report.get("error_count") if isinstance(report, Mapping) else None,
                    "output_json": str(args.output_json) if args.output_json else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "source_count": report["source_count"],
                "output_json": str(args.output_json) if args.output_json else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "PaperIIIQSEAuditError",
    "build_parser",
    "main",
    "run_paper_iii_qse_source_map_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
