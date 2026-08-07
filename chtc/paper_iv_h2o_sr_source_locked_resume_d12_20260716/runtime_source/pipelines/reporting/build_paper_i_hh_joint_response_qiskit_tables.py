#!/usr/bin/env python3
"""Build table-only Paper-I HH joint-response Qiskit comparisons.

The report compares fresh joint-response SNAKE results with the locked Paper-I
SNAKE, Geo-ADAPT, and Append-ADAPT rows.  Each current row uses the first
history prefix within 10 percent of that trajectory's minimum error.  Circuit
costs and algorithmic work are compiled for that exact prefix.

Optional Qiskit and LaTeX work is performed only from ``build``/``main``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "paper_i_hh_joint_response_qiskit_tables_v1"
PREFIX_POLICY_ID = "first_history_row_within_1p10_of_trajectory_minimum_v1"
PREFIX_TOLERANCE_FACTOR = 1.10
DEFAULT_REFERENCE_JSON = REPO_ROOT / (
    "output/pdf/paper_i_hh_corrected_vs_current_20260710/"
    "paper_i_hh_corrected_vs_current_onepage_20260710.json"
)
DEFAULT_STEM = "paper_i_hh_joint_response_qiskit_tables"
QISKIT_COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
S_CONVENTION = "paper_i_winning_branch_s_alg_v1"


@dataclass(frozen=True)
class RegimeSpec:
    regime: str
    campaign_dir: str
    display: str


REGIME_SPECS = (
    RegimeSpec("weak-weak", "weak-weak", "Weak Hubbard / weak Holstein"),
    RegimeSpec(
        "intermediate-weak",
        "intermediate-weak",
        "Intermediate Hubbard / weak Holstein",
    ),
    RegimeSpec("strong-weak", "strong-weak-u8", "Strong Hubbard / weak Holstein"),
    RegimeSpec("weak-strong", "weak-strong", "Weak Hubbard / strong Holstein"),
    RegimeSpec(
        "intermediate-strong",
        "intermediate-strong",
        "Intermediate Hubbard / strong Holstein",
    ),
    RegimeSpec("strong-strong", "strong-strong-u8", "Strong Hubbard / strong Holstein"),
)

CURRENT_METHOD = "joint_response_snake"
REFERENCE_METHOD_ORDER = ("snake", "geo", "append")
METHOD_ORDER = (CURRENT_METHOD, *REFERENCE_METHOD_ORDER)
METHOD_DISPLAY = {
    CURRENT_METHOD: "Joint-response SNAKE",
    "snake": "Paper-I SNAKE",
    "geo": "Geo-ADAPT",
    "append": "Append-ADAPT",
}


class SelectedPrefixCompilationError(RuntimeError):
    """The exact selected prefix could not produce valid Qiskit/S evidence."""


@dataclass(frozen=True)
class PrefixSelection:
    history_position: int
    k_pl: int
    error: float
    error_field: str
    trajectory_minimum: float
    threshold: float
    valid_history_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": PREFIX_POLICY_ID,
            "factor": PREFIX_TOLERANCE_FACTOR,
            "history_position": self.history_position,
            "k_pl": self.k_pl,
            "abs_delta_e": self.error,
            "error_field": self.error_field,
            "trajectory_minimum_abs_delta_e": self.trajectory_minimum,
            "selection_threshold_abs_delta_e": self.threshold,
            "valid_history_rows": self.valid_history_rows,
        }


@dataclass(frozen=True)
class TableRow:
    regime: str
    campaign_regime: str
    method: str
    method_display: str
    role: str
    k_pl: int
    ansatz_depth: int
    abs_delta_e: float
    n2q: int
    d2q: int
    dc: int
    s_value: int
    s_source: str
    source_json: str
    source_sha256: str
    qiskit_sidecar: str | None = None
    qiskit_sidecar_sha256: str | None = None
    reference_collection: str | None = None
    reference_row_exact: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "regime": self.regime,
            "campaign_regime": self.campaign_regime,
            "method": self.method,
            "method_display": self.method_display,
            "role": self.role,
            "k_pl": self.k_pl,
            "ansatz_depth": self.ansatz_depth,
            "abs_delta_e": self.abs_delta_e,
            "N2q": self.n2q,
            "D2q": self.d2q,
            "Dc": self.dc,
            "S": self.s_value,
            "S_source": self.s_source,
            "source_json": self.source_json,
            "source_sha256": self.source_sha256,
            "qiskit_sidecar": self.qiskit_sidecar,
            "qiskit_sidecar_sha256": self.qiskit_sidecar_sha256,
            "reference_collection": self.reference_collection,
        }
        if self.reference_row_exact is not None:
            payload["reference_row_exact"] = copy.deepcopy(dict(self.reference_row_exact))
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected JSON object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("adapt_vqe")
    return nested if isinstance(nested, Mapping) else payload


def _history_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history = _adapt_payload(payload).get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("result is missing adapt_vqe.history")
    if not history:
        raise ValueError("adapt_vqe.history is empty")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(history, start=1):
        if not isinstance(row, Mapping):
            raise TypeError(f"adapt_vqe.history[{index - 1}] is not an object")
        rows.append(row)
    return rows


def _validate_completed_result(payload: Mapping[str, Any], *, source: Path) -> None:
    adapt = _adapt_payload(payload)
    summary = payload.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    success = summary.get("success", adapt.get("success"))
    history_complete = adapt.get("history_checkpoint_complete")
    if success is not True or history_complete is not True:
        raise ValueError(
            "joint-response report requires a successful complete result: "
            f"source={source}, success={success!r}, "
            f"history_checkpoint_complete={history_complete!r}"
        )


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _history_error(row: Mapping[str, Any]) -> tuple[float, str]:
    for key in ("delta_abs_current", "benchmark_target_abs_delta_current", "abs_delta_e"):
        value = _finite_nonnegative(row.get(key))
        if value is not None:
            return value, key
    raise ValueError("history row has no finite nonnegative error field")


def select_plateau_prefix(
    payload: Mapping[str, Any],
    *,
    factor: float = PREFIX_TOLERANCE_FACTOR,
) -> PrefixSelection:
    """Select the first history row within ``factor`` of trajectory minimum."""

    if not math.isfinite(float(factor)) or float(factor) < 1.0:
        raise ValueError("prefix factor must be finite and at least 1")
    rows = _history_rows(payload)
    errors: list[tuple[float, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            errors.append(_history_error(row))
        except ValueError as exc:
            raise ValueError(f"history row {index} cannot participate in prefix selection: {exc}") from exc
    terminal_error = _finite_nonnegative(_adapt_payload(payload).get("abs_delta_e"))
    if terminal_error is not None:
        # Terminal pruning/final refit can improve the winner after the final
        # history row. The exact circuit sidecar compiles that terminal winner,
        # so prefix selection must use the same error value at the same k.
        errors[-1] = (terminal_error, "adapt_vqe.abs_delta_e_terminal_winner")
    minimum = min(value for value, _ in errors)
    threshold = float(factor) * minimum
    for index, (value, field) in enumerate(errors, start=1):
        if value <= threshold:
            return PrefixSelection(
                history_position=index,
                k_pl=index,
                error=value,
                error_field=field,
                trajectory_minimum=minimum,
                threshold=threshold,
                valid_history_rows=len(errors),
            )
    raise AssertionError("prefix selection failed despite a finite trajectory minimum")


def build_selected_prefix_sidecar(
    *,
    result_json: Path,
    history_position: int,
    output_json: Path,
    threshold: float,
) -> Mapping[str, Any]:
    """Lazily invoke the existing selected-prefix Qiskit compiler."""

    from pipelines.reporting.build_paper_i_selected_prefix_qiskit_sidecar import (
        build_sidecar,
    )

    return build_sidecar(
        result_json=result_json,
        history_position=history_position,
        output_json=output_json,
        threshold=threshold,
    )


def _count(value: Any, *, label: str) -> int:
    parsed = _finite_nonnegative(value)
    if parsed is None or not math.isclose(parsed, round(parsed), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} is missing or not a nonnegative integer count")
    return int(round(parsed))


def _status_is_ok(value: Any) -> bool:
    return str(value or "").strip().lower().startswith("ok")


def _selected_s(sidecar: Mapping[str, Any]) -> tuple[int, str]:
    mechanism = _finite_nonnegative(sidecar.get("mechanism_formula_S"))
    if mechanism is not None and _status_is_ok(sidecar.get("mechanism_formula_status")):
        return _count(mechanism, label="mechanism_formula_S"), "mechanism_formula_S"
    instrumented = _finite_nonnegative(sidecar.get("instrumented_runtime_S"))
    if instrumented is not None and _status_is_ok(sidecar.get("instrumented_runtime_status")):
        return _count(instrumented, label="instrumented_runtime_S"), "instrumented_runtime_S"
    raise ValueError("selected-prefix sidecar has no valid mechanism-formula or instrumented S")


def _validate_sidecar(
    sidecar: Mapping[str, Any],
    *,
    selection: PrefixSelection,
) -> tuple[int, int, int, int, int, str, dict[str, bool]]:
    checks = {
        "history_position_matches": int(sidecar.get("history_position", -1))
        == selection.history_position,
        "k_pl_matches": int(sidecar.get("k_pl", -1)) == selection.k_pl,
        "error_matches": math.isclose(
            float(sidecar.get("primary_error_at_prefix", math.nan)),
            selection.error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        "qiskit_validated": sidecar.get("compiled_resource_qiskit_validated") is True,
        "compiled_status_ok": str(sidecar.get("compiled_circuit_stats_status")) == "ok",
        "compile_convention_matches": str(sidecar.get("compile_convention"))
        == QISKIT_COMPILE_CONVENTION,
        "compile_convention_expected_matches": str(sidecar.get("compile_convention_expected"))
        == QISKIT_COMPILE_CONVENTION,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError("selected-prefix circuit alignment failed: " + ", ".join(failed))
    n2q = _count(sidecar.get("compiled_count_2q_total"), label="compiled_count_2q_total")
    d2q = _count(sidecar.get("compiled_depth_2q_total"), label="compiled_depth_2q_total")
    dc = _count(sidecar.get("compiled_depth_total"), label="compiled_depth_total")
    replay = sidecar.get("replay")
    replay = replay if isinstance(replay, Mapping) else {}
    ansatz_depth = _count(
        replay.get("replayed_operator_count"),
        label="replay.replayed_operator_count",
    )
    s_value, s_source = _selected_s(sidecar)
    return n2q, d2q, dc, ansatz_depth, s_value, s_source, checks


def _load_reference_rows(
    reference_payload: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    snake_rows = reference_payload.get("corrected_and_snake_rows")
    comparator_rows = reference_payload.get("current_paper_i_comparator_rows")
    if not isinstance(snake_rows, Sequence) or isinstance(snake_rows, (str, bytes)):
        raise ValueError("reference JSON is missing corrected_and_snake_rows")
    if not isinstance(comparator_rows, Sequence) or isinstance(comparator_rows, (str, bytes)):
        raise ValueError("reference JSON is missing current_paper_i_comparator_rows")

    selected: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    exact_rows: list[dict[str, Any]] = []
    corrected_rows = [row for row in snake_rows if isinstance(row, Mapping)]
    fallback_rows = [row for row in comparator_rows if isinstance(row, Mapping)]
    for spec in REGIME_SPECS:
        for method in REFERENCE_METHOD_ORDER:
            candidates = [
                row
                for row in corrected_rows
                if str(row.get("regime")) == spec.regime
                and str(row.get("method")) == method
            ]
            collection = "corrected_and_snake_rows"
            if not candidates:
                candidates = [
                    row
                    for row in fallback_rows
                    if str(row.get("regime")) == spec.regime
                    and str(row.get("method")) == method
                ]
                collection = "current_paper_i_comparator_rows"
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one locked Paper-I {method} row for {spec.regime}, "
                    f"found {len(candidates)}"
                )
            raw = copy.deepcopy(dict(candidates[0]))
            selected[(spec.regime, method)] = (raw, collection)
            exact_rows.append(copy.deepcopy(raw))
    return selected, exact_rows


def _reference_table_row(
    spec: RegimeSpec,
    method: str,
    raw: Mapping[str, Any],
    collection: str,
) -> TableRow:
    error_key = "abs_delta_e"
    error = _finite_nonnegative(raw.get(error_key))
    if error is None:
        error_key = "table_abs_delta_e"
        error = _finite_nonnegative(raw.get(error_key))
    if error is None:
        raise ValueError(f"reference {spec.regime}/{method} is missing {error_key}")
    k_pl = _count(raw.get("k_pl"), label=f"{spec.regime}/{method}.k_pl")
    raw_depth = raw.get("logical_depth")
    ansatz_depth = (
        _count(raw_depth, label=f"{spec.regime}/{method}.logical_depth")
        if raw_depth is not None
        else k_pl
    )
    return TableRow(
        regime=spec.regime,
        campaign_regime=spec.campaign_dir,
        method=method,
        method_display=METHOD_DISPLAY[method],
        role="paper_i_reference",
        k_pl=k_pl,
        ansatz_depth=ansatz_depth,
        abs_delta_e=error,
        n2q=_count(raw.get("N2q"), label=f"{spec.regime}/{method}.N2q"),
        d2q=_count(raw.get("D2q"), label=f"{spec.regime}/{method}.D2q"),
        dc=_count(raw.get("Dc"), label=f"{spec.regime}/{method}.Dc"),
        s_value=_count(raw.get("S_alg"), label=f"{spec.regime}/{method}.S_alg"),
        s_source="reference_json.S_alg",
        source_json=str(raw.get("source_json") or ""),
        source_sha256=str(raw.get("source_sha256") or ""),
        reference_collection=collection,
        reference_row_exact=copy.deepcopy(dict(raw)),
    )


def _route_settings(
    result_payload: Mapping[str, Any],
    plan_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    settings = result_payload.get("settings")
    adapt = _adapt_payload(result_payload)
    route_invocation: Mapping[str, Any] = {}
    plan_regime: Mapping[str, Any] = {}
    scientific_settings_hash = None
    if isinstance(plan_payload, Mapping):
        scientific = plan_payload.get("scientific_settings")
        if isinstance(scientific, Mapping):
            candidate = scientific.get("route_a_invocation")
            if isinstance(candidate, Mapping):
                route_invocation = candidate
            candidate_regime = scientific.get("regime")
            if isinstance(candidate_regime, Mapping):
                plan_regime = candidate_regime
        scientific_settings_hash = plan_payload.get("scientific_settings_hash")
    return {
        "result_settings": copy.deepcopy(dict(settings)) if isinstance(settings, Mapping) else {},
        "result_adapt_summary": {
            key: adapt.get(key)
            for key in (
                "method",
                "continuation",
                "pool_type",
                "pool_size",
                "adapt_beam_enabled",
                "stop_reason",
            )
            if key in adapt
        },
        "plan_scientific_settings_hash": scientific_settings_hash,
        "plan_regime": copy.deepcopy(dict(plan_regime)),
        "plan_route_a_invocation": copy.deepcopy(dict(route_invocation)),
    }


def _resolve_plan_json(campaign_dir: Path) -> Path | None:
    for filename in ("plan.json", "plan_new_route.json"):
        candidate = campaign_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _compact_mode(value: Any) -> str:
    text = str(value or "unspecified")
    for suffix in ("_v1", "_v2"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.replace("_", " ")


def _route_summary(route_settings: Mapping[str, Any]) -> str:
    invocation = route_settings.get("plan_route_a_invocation")
    if isinstance(invocation, Mapping) and invocation:
        mechanisms = invocation.get("mechanisms")
        optimizer = invocation.get("optimizer")
        shortlists = invocation.get("shortlists")
        mechanisms = mechanisms if isinstance(mechanisms, Mapping) else {}
        optimizer = optimizer if isinstance(optimizer, Mapping) else {}
        shortlists = shortlists if isinstance(shortlists, Mapping) else {}
        route_id = str(invocation.get("route_id") or "unspecified")
        route_label = "Route A" if route_id == "route_a" else _compact_mode(route_id)
        return (
            f"{route_label}; profile={_compact_mode(invocation.get('profile'))}; "
            f"funnel={_compact_mode(mechanisms.get('phase3_candidate_population'))}; "
            f"batch={_compact_mode(mechanisms.get('batch_selection_mode'))}; "
            f"Bmax={mechanisms.get('batch_size_cap')}; "
            f"Lsearch={mechanisms.get('batch_search_pool_size')}; "
            f"context={_compact_mode(mechanisms.get('joint_batch_context_mode'))}; "
            f"optimizer={_compact_mode(optimizer.get('inner_optimizer'))}; "
            f"maxiter={optimizer.get('maxiter')}; maxfev={optimizer.get('scipy_maxfev')}; "
            f"M1/M2={shortlists.get('phase1_size')}/{shortlists.get('phase2_size')}; "
            f"C1/C2={shortlists.get('child_phase1_size')}/{shortlists.get('child_phase2_size')}"
        )
    settings = route_settings.get("result_settings")
    settings = settings if isinstance(settings, Mapping) else {}
    route_id = str(settings.get("static_route_id") or "unspecified")
    route_label = "Route A" if route_id == "route_a" else _compact_mode(route_id)
    return (
        f"{route_label}; profile={_compact_mode(settings.get('static_meta_feature_profile'))}; "
        f"continuation={_compact_mode(settings.get('continuation_mode'))}; "
        f"pool={_compact_mode(settings.get('adapt_pool'))}"
    )


def _physics_summary(route_settings: Mapping[str, Any]) -> str:
    plan_regime = route_settings.get("plan_regime")
    if isinstance(plan_regime, Mapping) and plan_regime:
        return ", ".join(
            f"{key}={plan_regime.get(key)}"
            for key in ("u", "lambda", "g_ep", "omega0", "n_ph_work", "n_ph_ref")
            if plan_regime.get(key) is not None
        )
    settings = route_settings.get("result_settings")
    settings = settings if isinstance(settings, Mapping) else {}
    return ", ".join(
        f"{key}={settings.get(key)}"
        for key in ("u", "g_ep", "omega0", "n_ph_max", "boson_encoding")
        if settings.get(key) is not None
    )


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _format_error(value: float) -> str:
    return f"{float(value):.3e}"


def _table_tex(regime: str, rows: Sequence[TableRow]) -> str:
    selected = [row for row in rows if row.regime == regime]
    if [row.method for row in selected] != list(METHOD_ORDER):
        raise ValueError(f"unexpected row order for {regime}")
    lines = [
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        r"Method & $k_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S$\\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{_latex_escape(row.method_display)} & {row.k_pl} & {_format_error(row.abs_delta_e)} & "
            f"{row.n2q:,} & {row.d2q:,} & {row.dc:,} & {row.s_value:,} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular*}"))
    return "\n".join(lines)


def _write_tex(
    path: Path,
    *,
    generated_utc: str,
    campaign_root: Path,
    reference_json: Path,
    reference_sha256: str,
    current_evidence: Mapping[str, Mapping[str, Any]],
    rows: Sequence[TableRow],
    report_json: Path,
    report_csv: Path,
) -> None:
    first_route = current_evidence[REGIME_SPECS[0].regime]["route_settings"]
    manifest_rows = [
        ("Schema", SCHEMA),
        ("Generated UTC", generated_utc),
        ("Campaign", f"{campaign_root.name}; full paths and hashes are in the JSON sidecar"),
        ("Prefix policy", "first prefix within 1.10 times the trajectory minimum"),
        ("Qiskit / S", "basis-gate transpile; Paper-I winning-branch S"),
        ("Current route", _route_summary(first_route)),
        ("Reference", f"{reference_json.name}; sha256={reference_sha256[:12]}..."),
        ("Sidecars", f"{report_json.name} and {report_csv.name}"),
        ("Manuscript", "Paper_I.tex not edited"),
    ]
    for spec in REGIME_SPECS:
        evidence = current_evidence[spec.regime]
        selection = evidence["prefix_selection"]
        manifest_rows.append(
            (
                spec.regime,
                f"{_physics_summary(evidence['route_settings'])}; "
                f"result hash={evidence['result_sha256'][:12]}...; "
                f"k_pl={selection['k_pl']}; ansatz_depth={evidence['ansatz_depth']}; "
                f"min_error={selection['trajectory_minimum_abs_delta_e']:.6g}",
            )
        )
    manifest_tex = "\n".join(
        f"\\textbf{{{_latex_escape(label)}}} & {_latex_escape(value)} \\\\"
        for label, value in manifest_rows
    )
    sections = []
    for spec in REGIME_SPECS:
        sections.extend(
            (
                f"\\subsection*{{{_latex_escape(spec.display)}}}",
                _table_tex(spec.regime, rows),
                r"\vspace{1.0em}",
            )
        )
    machine_comment = json.dumps(
        {
            "schema": SCHEMA,
            "campaign_root": _display_path(campaign_root),
            "reference_json": _display_path(reference_json),
            "prefix_policy": PREFIX_POLICY_ID,
            "report_json": _display_path(report_json),
            "report_csv": _display_path(report_csv),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=0.55in]{{geometry}}
\usepackage{{booktabs,microtype}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\setlength{{\parindent}}{{0pt}}
\pagestyle{{plain}}
\begin{{document}}
\section*{{Normalized parameter and provenance manifest}}
% BEGIN_MACHINE_READABLE_JOINT_RESPONSE_QISKIT_TABLE_REPORT
% {machine_comment}
% END_MACHINE_READABLE_JOINT_RESPONSE_QISKIT_TABLE_REPORT
\small
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular*}}{{\linewidth}}{{@{{}}p{{0.19\linewidth}}@{{\extracolsep{{\fill}}}}p{{0.76\linewidth}}@{{}}}}
\toprule
Field & Normalized value\\
\midrule
{manifest_tex}
\bottomrule
\end{{tabular*}}

\section*{{Paper-I-style selected-prefix resource tables}}
Each current row uses the first history prefix satisfying
$|\Delta E_k|\leq 1.10\min_j |\Delta E_j|$. Circuit resources and $S$ are
evaluated at that same prefix. Reference rows are copied from the locked
Paper-I comparison JSON.

{chr(10).join(sections)}
\end{{document}}
"""
    if "\\includegraphics" in tex:
        raise AssertionError("table-only report unexpectedly contains graphics")
    path.write_text(tex, encoding="ascii")


def compile_latex(tex_path: Path) -> Path:
    executable = shutil.which("latexmk")
    if executable:
        command = [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    else:
        executable = shutil.which("tectonic")
        if not executable:
            raise RuntimeError("Neither latexmk nor tectonic is available")
        command = [
            executable,
            "--keep-logs",
            "--reruns",
            "2",
            "--outdir",
            str(tex_path.parent),
            tex_path.name,
        ]
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"LaTeX build failed:\n{completed.stdout}\n{completed.stderr}")
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    return pdf_path


def _write_csv(path: Path, rows: Sequence[TableRow]) -> None:
    fieldnames = (
        "regime",
        "campaign_regime",
        "role",
        "method",
        "method_display",
        "k_pl",
        "ansatz_depth",
        "abs_delta_e",
        "N2q",
        "D2q",
        "Dc",
        "S",
        "S_source",
        "source_json",
        "source_sha256",
        "qiskit_sidecar",
        "qiskit_sidecar_sha256",
        "reference_collection",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.as_dict()
            writer.writerow({key: payload.get(key) for key in fieldnames})


def _validate_rows(
    rows: Sequence[TableRow],
    *,
    current_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    expected = [(spec.regime, method) for spec in REGIME_SPECS for method in METHOD_ORDER]
    actual = [(row.regime, row.method) for row in rows]
    current_by_regime = {row.regime: row for row in rows if row.method == CURRENT_METHOD}
    checks = {
        "six_regimes_four_rows_each": actual == expected,
        "all_counts_nonnegative": all(min(row.n2q, row.d2q, row.dc, row.s_value) >= 0 for row in rows),
        "all_errors_finite_nonnegative": all(
            math.isfinite(row.abs_delta_e) and row.abs_delta_e >= 0.0 for row in rows
        ),
        "current_prefix_error_aligned": all(
            math.isclose(
                current_by_regime[spec.regime].abs_delta_e,
                float(current_evidence[spec.regime]["prefix_selection"]["abs_delta_e"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for spec in REGIME_SPECS
        ),
        "current_prefix_k_aligned": all(
            current_by_regime[spec.regime].k_pl
            == int(current_evidence[spec.regime]["prefix_selection"]["k_pl"])
            for spec in REGIME_SPECS
        ),
        "current_ansatz_depth_aligned": all(
            current_by_regime[spec.regime].ansatz_depth
            == int(current_evidence[spec.regime]["ansatz_depth"])
            for spec in REGIME_SPECS
        ),
        "reference_rows_retain_exact_objects": all(
            row.reference_row_exact is not None for row in rows if row.role == "paper_i_reference"
        ),
        "current_uses_selected_prefix_sidecar": all(
            row.qiskit_sidecar is not None for row in rows if row.method == CURRENT_METHOD
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError("report row validation failed: " + ", ".join(failed))
    return checks


def build(
    *,
    campaign_root: Path,
    output_dir: Path,
    stem: str = DEFAULT_STEM,
    reference_json: Path = DEFAULT_REFERENCE_JSON,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    output_dir = output_dir.resolve()
    reference_json = reference_json.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_utc = _utc_now()

    reference_payload = _read_json(reference_json)
    references, exact_reference_rows = _load_reference_rows(reference_payload)
    reference_hash = _sha256(reference_json)
    rows: list[TableRow] = []
    current_evidence: dict[str, dict[str, Any]] = {}
    prefix_dir = output_dir / "selected_prefix_qiskit"

    for spec in REGIME_SPECS:
        result_json = campaign_root / spec.campaign_dir / "result.json"
        result_payload = _read_json(result_json)
        _validate_completed_result(result_payload, source=result_json)
        selection = select_plateau_prefix(result_payload)
        sidecar_json = prefix_dir / f"{spec.campaign_dir}_selected_prefix_qiskit.json"
        try:
            build_selected_prefix_sidecar(
                result_json=result_json,
                history_position=selection.history_position,
                output_json=sidecar_json,
                threshold=selection.threshold,
            )
        except Exception as exc:
            raise SelectedPrefixCompilationError(
                "selected-prefix replay/Qiskit compilation failed for "
                f"regime={spec.regime}, campaign_regime={spec.campaign_dir}, "
                f"history_position={selection.history_position}, result={result_json}: {exc}"
            ) from exc
        if not sidecar_json.is_file():
            raise SelectedPrefixCompilationError(
                f"selected-prefix compiler did not write {sidecar_json} for {spec.regime}"
            )
        sidecar = _read_json(sidecar_json)
        try:
            (
                n2q,
                d2q,
                dc,
                ansatz_depth,
                s_value,
                s_source,
                sidecar_checks,
            ) = _validate_sidecar(
                sidecar,
                selection=selection,
            )
        except Exception as exc:
            raise SelectedPrefixCompilationError(
                "selected-prefix evidence validation failed for "
                f"regime={spec.regime}, history_position={selection.history_position}, "
                f"sidecar={sidecar_json}: {exc}"
            ) from exc
        plan_json = _resolve_plan_json(campaign_root / spec.campaign_dir)
        plan_payload = _read_json(plan_json) if plan_json is not None else None
        route_settings = _route_settings(result_payload, plan_payload)
        result_hash = _sha256(result_json)
        sidecar_hash = _sha256(sidecar_json)
        rows.append(
            TableRow(
                regime=spec.regime,
                campaign_regime=spec.campaign_dir,
                method=CURRENT_METHOD,
                method_display=METHOD_DISPLAY[CURRENT_METHOD],
                role="current_joint_response",
                k_pl=selection.k_pl,
                ansatz_depth=ansatz_depth,
                abs_delta_e=selection.error,
                n2q=n2q,
                d2q=d2q,
                dc=dc,
                s_value=s_value,
                s_source=s_source,
                source_json=_display_path(result_json),
                source_sha256=result_hash,
                qiskit_sidecar=_display_path(sidecar_json),
                qiskit_sidecar_sha256=sidecar_hash,
            )
        )
        for method in REFERENCE_METHOD_ORDER:
            raw, collection = references[(spec.regime, method)]
            rows.append(_reference_table_row(spec, method, raw, collection))
        current_evidence[spec.regime] = {
            "campaign_regime": spec.campaign_dir,
            "result_json": _display_path(result_json),
            "result_sha256": result_hash,
            "plan_json": _display_path(plan_json) if plan_json is not None else None,
            "plan_sha256": _sha256(plan_json) if plan_json is not None else None,
            "prefix_selection": selection.as_dict(),
            "ansatz_depth": int(ansatz_depth),
            "selected_prefix_qiskit_sidecar": _display_path(sidecar_json),
            "selected_prefix_qiskit_sidecar_sha256": sidecar_hash,
            "selected_prefix_validation": sidecar_checks,
            "S_source": s_source,
            "route_settings": route_settings,
        }

    validation = _validate_rows(rows, current_evidence=current_evidence)
    report_json = output_dir / f"{stem}.json"
    report_csv = output_dir / f"{stem}.csv"
    report_tex = output_dir / f"{stem}.tex"
    _write_csv(report_csv, rows)
    _write_tex(
        report_tex,
        generated_utc=generated_utc,
        campaign_root=campaign_root,
        reference_json=reference_json,
        reference_sha256=reference_hash,
        current_evidence=current_evidence,
        rows=rows,
        report_json=report_json,
        report_csv=report_csv,
    )
    report_pdf = compile_latex(report_tex)
    if not report_pdf.is_file():
        raise FileNotFoundError(report_pdf)

    payload = {
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "run_class": "candidate_comparison_report",
        "manuscript_edited": False,
        "table_only": True,
        "campaign_root": _display_path(campaign_root),
        "prefix_policy": {
            "id": PREFIX_POLICY_ID,
            "factor": PREFIX_TOLERANCE_FACTOR,
            "definition": "first history row with abs error <= factor times trajectory minimum",
        },
        "qiskit_compile_convention": QISKIT_COMPILE_CONVENTION,
        "S_convention": S_CONVENTION,
        "S_preference_order": ["mechanism_formula_S", "instrumented_runtime_S"],
        "reference_source": {
            "path": _display_path(reference_json),
            "sha256": reference_hash,
            "preferred_collection": "corrected_and_snake_rows",
            "fallback_collection": "current_paper_i_comparator_rows",
        },
        "paper_i_reference_rows_exact": exact_reference_rows,
        "current_evidence": current_evidence,
        "rows": [row.as_dict() for row in rows],
        "validation": validation,
        "artifacts": {
            "json": _display_path(report_json),
            "csv": _display_path(report_csv),
            "csv_sha256": _sha256(report_csv),
            "tex": _display_path(report_tex),
            "tex_sha256": _sha256(report_tex),
            "pdf": _display_path(report_pdf),
            "pdf_sha256": _sha256(report_pdf),
        },
    }
    _write_json(report_json, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(
        campaign_root=args.campaign_root,
        output_dir=args.output_dir,
        stem=str(args.stem),
    )
    print(json.dumps({"status": "ok", **payload["artifacts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
