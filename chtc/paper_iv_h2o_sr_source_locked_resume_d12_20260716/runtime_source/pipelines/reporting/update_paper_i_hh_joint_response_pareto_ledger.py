#!/usr/bin/env python3
"""Insert completed joint-response evidence into the Paper-I Pareto ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_snake_pareto_candidate_ledger_20260711.md"
)
BEGIN_MARKER = "<!-- BEGIN AUTO: joint-response-long-horizon-20260712 -->"
END_MARKER = "<!-- END AUTO: joint-response-long-horizon-20260712 -->"
INSERT_BEFORE = "## Candidate policies"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload is not an object: {path}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_tuple(row: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(row[key])
        for key in ("abs_delta_e", "N2q", "D2q", "Dc", "S")
    )


def _relation(current: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    current_values = _metric_tuple(current)
    reference_values = _metric_tuple(reference)
    current_no_worse = all(left <= right for left, right in zip(current_values, reference_values))
    reference_no_worse = all(right <= left for left, right in zip(current_values, reference_values))
    if current_no_worse and any(left < right for left, right in zip(current_values, reference_values)):
        return "dominates"
    if reference_no_worse and any(right < left for left, right in zip(current_values, reference_values)):
        return "dominated"
    error_relation = (
        "lower error"
        if current_values[0] < reference_values[0]
        else "higher error"
        if current_values[0] > reference_values[0]
        else "equal error"
    )
    return f"tradeoff ({error_relation})"


def _format_error(value: Any) -> str:
    return f"{float(value):.16e}"


def render_block(report: Mapping[str, Any], *, report_path: Path) -> str:
    rows = report.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("report is missing rows")
    by_key = {
        (str(row.get("regime")), str(row.get("method"))): row
        for row in rows
        if isinstance(row, Mapping)
    }
    regimes = ("weak-weak", "intermediate-weak", "strong-weak")
    methods = ("joint_response_snake", "snake", "geo", "append")
    missing = [
        (regime, method)
        for regime in regimes
        for method in methods
        if (regime, method) not in by_key
    ]
    if missing:
        raise ValueError(f"report is missing regime/method rows: {missing}")

    lines = [
        BEGIN_MARKER,
        "## Joint-step warm-start long-horizon evidence (2026-07-12)",
        "",
        "Status: completed candidate evidence only. This block does not promote a policy or edit the manuscript.",
        "",
        f"Source report: `{report_path}` (SHA-256 `{_sha256(report_path)}`).",
        "The current row uses the report's first-prefix-within-10%-of-trajectory-minimum policy. "
        "Circuit and winning-branch `S_alg` values refer to that same prefix.",
        "",
        "| Regime | Selected round / ansatz depth | New `abs(Delta E)` | New `N2q / D2q / Dc` | New winning-branch `S_alg` | Versus Paper-I SNAKE | Versus Geo | Versus Append |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for regime in regimes:
        current = by_key[(regime, "joint_response_snake")]
        references = {
            method: by_key[(regime, method)]
            for method in ("snake", "geo", "append")
        }
        lines.append(
            "| "
            + " | ".join(
                (
                    regime,
                    f"{int(current['k_pl'])} / {int(current['ansatz_depth'])}",
                    _format_error(current["abs_delta_e"]),
                    f"{int(current['N2q']):,} / {int(current['D2q']):,} / {int(current['Dc']):,}",
                    f"{int(current['S']):,}",
                    _relation(current, references["snake"]),
                    _relation(current, references["geo"]),
                    _relation(current, references["append"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "The complete Paper-I-style table, selected-prefix Qiskit sidecars, settings, and hashes remain in the source report bundle.",
            END_MARKER,
        )
    )
    return "\n".join(lines)


def update_ledger(*, report_json: Path, ledger_md: Path) -> str:
    report_json = report_json.resolve()
    ledger_md = ledger_md.resolve()
    report = _read_json(report_json)
    text = ledger_md.read_text(encoding="utf-8")
    block = render_block(report, report_path=report_json)
    if BEGIN_MARKER in text or END_MARKER in text:
        if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
            raise ValueError("ledger contains malformed auto-update markers")
        start = text.index(BEGIN_MARKER)
        end = text.index(END_MARKER, start) + len(END_MARKER)
        updated = text[:start] + block + text[end:]
    else:
        anchor = text.find(INSERT_BEFORE)
        if anchor < 0:
            raise ValueError(f"ledger is missing insertion anchor: {INSERT_BEFORE}")
        updated = text[:anchor].rstrip() + "\n\n" + block + "\n\n" + text[anchor:]
    ledger_md.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return block


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    block = update_ledger(
        report_json=args.report_json,
        ledger_md=args.ledger_md,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "report_json": str(args.report_json.resolve()),
                "ledger_md": str(args.ledger_md.resolve()),
                "rendered_line_count": len(block.splitlines()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
