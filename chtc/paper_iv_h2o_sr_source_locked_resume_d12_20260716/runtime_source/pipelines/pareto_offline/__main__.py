#!/usr/bin/env python3
"""CLI entrypoint for offline Pareto scaffold diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from pipelines.pareto_offline.analysis import ParetoAxis, analyze_continuation_artifact
from pipelines.pareto_offline.artifacts import (
    build_candidate_ledger_entry,
    build_default_candidate_ledger_entries,
    render_candidate_ledger_markdown,
)
from pipelines.pareto_offline.compare import (
    DEFAULT_TARGET_CLASS,
    build_proposal_packet,
    decompose_admission_signal_gap,
)


ANALYSIS_ONLY_FLAGS = (
    "--surface-key",
    "--overrides-json",
    "--selector-geometry-mode",
    "--pareto-x-key",
    "--pareto-x-direction",
    "--pareto-y-key",
    "--pareto-y-direction",
)
SEAM_ONLY_FLAGS = (
    "--retained-surface-key",
    "--admitted-surface-key",
    "--target-class",
    "--comparator-class",
    "--comparator-label",
)
LEDGER_ONLY_FLAGS = (
    "--candidate-root",
    "--output-markdown",
    "--ledger-title",
)
LEDGER_DISALLOWED_FLAGS = ("--output-json",)


def _load_json_mapping(raw: str | None) -> dict[str, Any]:
    if raw in {None, ""}:
        return {}
    candidate_path = Path(str(raw))
    if candidate_path.exists():
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError("Overrides JSON must decode to an object.")
    return dict(payload)


def _flag_was_supplied(argv_tokens: Sequence[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(token == flag or str(token).startswith(prefix) for token in argv_tokens)


def _validate_mode_flags(parser: argparse.ArgumentParser, args: argparse.Namespace, argv_tokens: Sequence[str]) -> None:
    report = str(args.report)
    if report == "analysis":
        if args.artifact_json is None:
            parser.error("artifact_json is required for --report=analysis")
        for flag in SEAM_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=admission_signal_gap or --report=proposal_packet")
        for flag in LEDGER_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=ledger")
        return
    if report in {"admission_signal_gap", "proposal_packet"}:
        if args.artifact_json is None:
            parser.error(f"artifact_json is required for --report={report}")
        for flag in ANALYSIS_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=analysis")
        for flag in LEDGER_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=ledger")
        return
    if report == "ledger":
        for flag in ANALYSIS_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=analysis")
        for flag in SEAM_ONLY_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is only valid for --report=admission_signal_gap or --report=proposal_packet")
        for flag in LEDGER_DISALLOWED_FLAGS:
            if _flag_was_supplied(argv_tokens, flag):
                parser.error(f"{flag} is not valid for --report=ledger; use --output-markdown")
        return
    parser.error(f"Unsupported report mode: {report}")


def _analysis_output_path(artifact_json: Path, output_json: Path | None) -> Path:
    return output_json or artifact_json.with_name(f"{artifact_json.stem}_pareto_offline.json")


def _admission_output_path(artifact_json: Path, output_json: Path | None) -> Path:
    return output_json or artifact_json.with_name(
        f"{artifact_json.stem}_pareto_offline_admission_signal_gap.json"
    )


def _ledger_output_path(output_markdown: Path | None) -> Path:
    return output_markdown or Path("pipelines/pareto_offline/CANDIDATE_LEDGER.md")


def _proposal_packet_output_path(artifact_json: Path, output_json: Path | None) -> Path:
    return output_json or artifact_json.with_name(
        f"{artifact_json.stem}_pareto_offline_proposal_packet.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline Pareto scaffold diagnostics for saved HH continuation artifacts.",
    )
    parser.add_argument(
        "artifact_json",
        nargs="?",
        type=Path,
        default=None,
        help="Path to saved continuation artifact JSON (required for analysis/admission/proposal_packet modes).",
    )
    parser.add_argument(
        "--report",
        default="analysis",
        choices=["analysis", "admission_signal_gap", "ledger", "proposal_packet"],
        help="Offline report mode (default: analysis).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output path (default: <artifact stem>_pareto_offline.json for analysis, "
            "<artifact stem>_pareto_offline_admission_signal_gap.json for admission_signal_gap, "
            "or <artifact stem>_pareto_offline_proposal_packet.json for proposal_packet)."
        ),
    )

    analysis_group = parser.add_argument_group("analysis options")
    analysis_group.add_argument(
        "--surface-key",
        default=None,
        help="Continuation surface key to analyze (default: best available retained/admitted/shortlist order).",
    )
    analysis_group.add_argument(
        "--overrides-json",
        default=None,
        help="JSON object or path to JSON file with FullScoreConfig overrides.",
    )
    analysis_group.add_argument(
        "--selector-geometry-mode",
        default=None,
        choices=["reduced", "raw_exact"],
        help="Optional convenience override for phase3_selector_geometry_mode.",
    )
    analysis_group.add_argument("--pareto-x-key", default=None, help="Optional first Pareto axis key.")
    analysis_group.add_argument(
        "--pareto-x-direction",
        default="min",
        choices=["min", "max"],
        help="Direction for the first Pareto axis.",
    )
    analysis_group.add_argument("--pareto-y-key", default=None, help="Optional second Pareto axis key.")
    analysis_group.add_argument(
        "--pareto-y-direction",
        default="max",
        choices=["min", "max"],
        help="Direction for the second Pareto axis.",
    )

    admission_group = parser.add_argument_group("admission-signal-gap options")
    admission_group.add_argument(
        "--retained-surface-key",
        default="phase2_retained_shortlist_rows",
        help="Retained surface key for the comparator search.",
    )
    admission_group.add_argument(
        "--admitted-surface-key",
        default="phase2_admitted_rows",
        help="Admitted surface key for the winner selection.",
    )
    admission_group.add_argument(
        "--target-class",
        default=DEFAULT_TARGET_CLASS,
        help="Target class prefix used for comparator fallback.",
    )
    admission_group.add_argument(
        "--comparator-class",
        default=None,
        help="Explicit comparator class prefix override.",
    )
    admission_group.add_argument(
        "--comparator-label",
        default=None,
        help="Explicit comparator label override.",
    )

    ledger_group = parser.add_argument_group("ledger options")
    ledger_group.add_argument(
        "--candidate-root",
        action="append",
        default=[],
        help=(
            "Candidate run root, or a JSON/log path inside one, to include in ledger mode. "
            "Repeatable. Defaults to the built-in Path B seed set when omitted."
        ),
    )
    ledger_group.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Output markdown path for ledger mode (default: pipelines/pareto_offline/CANDIDATE_LEDGER.md).",
    )
    ledger_group.add_argument(
        "--ledger-title",
        default="Path B Candidate Ledger",
        help="Title for ledger markdown output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_tokens)
    _validate_mode_flags(parser, args, argv_tokens)

    if str(args.report) == "analysis":
        overrides = _load_json_mapping(args.overrides_json)
        pareto_axes = None
        if args.pareto_x_key and args.pareto_y_key:
            pareto_axes = (
                ParetoAxis(key=str(args.pareto_x_key), direction=str(args.pareto_x_direction)),
                ParetoAxis(key=str(args.pareto_y_key), direction=str(args.pareto_y_direction)),
            )
        payload = analyze_continuation_artifact(
            args.artifact_json,
            surface_key=args.surface_key,
            overrides=overrides,
            selector_geometry_mode=args.selector_geometry_mode,
            pareto_axes=pareto_axes,
        )
        output_json = _analysis_output_path(args.artifact_json, args.output_json)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        selected = payload.get("selected_candidate") or {}
        inverse = payload.get("inverse_ranking") or {}
        print(f"artifact_json: {args.artifact_json}")
        print(f"surface_key: {payload.get('surface_key')}")
        print(f"selected_candidate: {selected.get('candidate_label')}")
        print(f"observed_rank: {inverse.get('observed_rank')}")
        print(f"reranked_rank: {inverse.get('reranked_rank')}")
        print(f"observed_frontier_count: {payload.get('observed_frontier', {}).get('frontier_count')}")
        print(f"reranked_frontier_count: {payload.get('reranked_frontier', {}).get('frontier_count')}")
        print(f"output_json: {output_json}")
        return

    if str(args.report) == "ledger":
        candidate_roots = [Path(path) for path in args.candidate_root]
        if args.artifact_json is not None:
            candidate_roots = [args.artifact_json, *candidate_roots]
        if candidate_roots:
            entries = [
                build_candidate_ledger_entry(
                    candidate_root,
                    provenance_note="manual candidate-root input",
                )
                for candidate_root in candidate_roots
            ]
        else:
            entries = build_default_candidate_ledger_entries()
        output_markdown = _ledger_output_path(args.output_markdown)
        markdown = render_candidate_ledger_markdown(entries, title=str(args.ledger_title))
        output_markdown.write_text(markdown, encoding="utf-8")
        print(f"ledger_entries: {len(entries)}")
        print(f"output_markdown: {output_markdown}")
        return

    if str(args.report) == "proposal_packet":
        payload = build_proposal_packet(
            args.artifact_json,
            retained_surface_key=str(args.retained_surface_key),
            admitted_surface_key=str(args.admitted_surface_key),
            target_class=str(args.target_class),
            comparator_class=(str(args.comparator_class) if args.comparator_class is not None else None),
            comparator_label=(str(args.comparator_label) if args.comparator_label is not None else None),
        )
        output_json = _proposal_packet_output_path(args.artifact_json, args.output_json)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"artifact_json: {args.artifact_json}")
        print(f"admitted_label: {payload.get('admitted_label')}")
        print(f"comparator_label: {payload.get('comparator_label')}")
        print(f"proposal_candidate_label: {payload.get('proposal_candidate_label')}")
        print(f"proposal_candidate_role: {payload.get('proposal_candidate_role')}")
        print(f"dominant_gap_source: {payload.get('dominant_gap_source')}")
        print(f"recommended_action: {payload.get('recommended_action')}")
        print(f"output_json: {output_json}")
        return

    payload = decompose_admission_signal_gap(
        args.artifact_json,
        retained_surface_key=str(args.retained_surface_key),
        admitted_surface_key=str(args.admitted_surface_key),
        target_class=str(args.target_class),
        comparator_class=(str(args.comparator_class) if args.comparator_class is not None else None),
        comparator_label=(str(args.comparator_label) if args.comparator_label is not None else None),
    )
    output_json = _admission_output_path(args.artifact_json, args.output_json)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"artifact_json: {args.artifact_json}")
    print(f"retained_surface_key: {payload.get('retained_surface_key')}")
    print(f"admitted_surface_key: {payload.get('admitted_surface_key')}")
    print(f"admitted_label: {payload.get('admitted_label')}")
    print(f"comparator_label: {payload.get('comparator_label')}")
    print(f"comparator_resolution_source: {payload.get('comparator_resolution_source')}")
    print(f"dominant_gap_source: {payload.get('dominant_gap_source')}")
    print(f"output_json: {output_json}")


if __name__ == "__main__":
    main()
