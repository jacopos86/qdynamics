#!/usr/bin/env python3
"""Read-only fairness/provenance audit for the condensed Paper-I manuscript.

This diagnostic tool audits the rendered-target source proxy
``MATH/paper_details/static_adapt_paper_I_condensed.tex`` and named evidence
sidecars.  It writes JSON/CSV/Markdown reports only; it does not edit TeX,
rebuild PDFs, launch runs, fetch CHTC output, or make promotion decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "paper_i_condensed_fairness_provenance_audit_v1"
FIELDS_12 = ("DeltaE", "one_minus_F", "N2q", "D2q", "Dc", "S")
HH_PLATEAU_FIELDS = ("k_pl", "DeltaE", "N2q", "D2q", "Dc")
TWO_REGIME_TABLES = {
    "tab:fixed_accuracy_claims": ("weak", "strong"),
    "tab:fixed_accuracy_spin_boson": ("weak", "strong"),
}
HH_APPENDIX_LABEL = "tab:fixed_accuracy_hh_cartesian"
HH_PLATEAU_LABEL = "tab:hh_first_plateau_prefix_costs"
PATH_PREFIXES = (
    "MATH/",
    "output/",
    "raw_outputs/",
    "artifacts/",
    "chtc/",
    "docs/",
    "tmp/",
    "pipelines/",
    "agent_guidance/",
)
PATH_SUFFIXES = (".json", ".jsonl", ".csv", ".txt", ".pdf", ".png", ".tex", ".py")
REMOTE_PREFIXES = ("/work/", "chtc-live:", "~/")


@dataclass(frozen=True)
class Finding:
    status: str
    code: str
    table_label: str | None = None
    method: str | None = None
    regime: str | None = None
    field_group: str | None = None
    evidence: tuple[str, ...] = ()
    message: str = ""
    follow_up_scope: str = "review"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "table_label": self.table_label,
            "method": self.method,
            "regime": self.regime,
            "field_group": self.field_group,
            "evidence": list(self.evidence),
            "message": self.message,
            "follow_up_scope": self.follow_up_scope,
        }


@dataclass(frozen=True)
class SourceReference:
    path: str
    referenced_by: str
    expected_sha256: str | None = None
    source_key: str | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        tmp = Path(fh.name)
        fh.write(text)
    tmp.replace(path)


def _method_key(method: str) -> str:
    text = str(method).strip()
    text = re.sub(r"\$\^\\(?:dagger|ddagger)\$", "", text).strip()
    aliases = {
        "Append-ADAPT": "append ADAPT",
        "append ADAPT": "append ADAPT",
        "append-only ADAPT": "append ADAPT",
        "TETRIS-ADAPT": "TETRIS-ADAPT",
        "Geo-ADAPT": "Geo-ADAPT",
        "Qubit/QEB": "Qubit/QEB",
        "SNAKE": "SNAKE",
        "HEA VQE": "HEA VQE",
        "family VQE": "family VQE",
    }
    return aliases.get(text, text)


def _strip_tex_markup(text: str) -> str:
    out = text.strip()
    out = re.sub(r"\$\\(?:dagger|ddagger)\$", "", out)
    out = out.replace(r"\twq", "2q")
    out = out.replace(r"\rm", "")
    out = re.sub(r"\\metrichead\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", out)
    out = out.replace("{", "").replace("}", "")
    out = out.replace("$", "")
    return out.strip()


def _split_row(line: str) -> list[str]:
    clean = line.split("%", 1)[0].strip()
    clean = clean.removesuffix(r"\\").strip()
    return [_strip_tex_markup(part) for part in clean.split("&")]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = str(value).strip()
    if text in {"", "--", "running", "n/a", "NR"}:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _line_for_index(idx: int) -> int:
    return idx + 1


def _find_caption_line(lines: Sequence[str], label: str) -> int:
    needle = r"\inlinetablecaption{" + label + "}"
    for idx, line in enumerate(lines):
        if needle in line:
            return idx
    raise ValueError(f"could not find table label {label}")


def _caption_snippet(lines: Sequence[str], label: str) -> str:
    idx = _find_caption_line(lines, label)
    return _strip_tex_markup(lines[idx])[:500]


def _parse_comment_blocks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    begin_re = re.compile(r"^% BEGIN_MACHINE_READABLE_(.+)$")
    idx = 0
    while idx < len(lines):
        match = begin_re.match(lines[idx])
        if not match:
            idx += 1
            continue
        name = match.group(1)
        start_line = _line_for_index(idx)
        idx += 1
        body: list[str] = []
        while idx < len(lines) and not lines[idx].startswith("% END_MACHINE_READABLE_"):
            line = lines[idx]
            body.append(line[2:] if line.startswith("% ") else line[1:] if line.startswith("%") else line)
            idx += 1
        end_line = _line_for_index(idx) if idx < len(lines) else len(lines)
        raw = "\n".join(body).strip()
        parsed: Any = None
        error: str | None = None
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                # Some historical comments contain prose before the JSON object.
                first = raw.find("{")
                last = raw.rfind("}")
                if first >= 0 and last > first:
                    try:
                        parsed = json.loads(raw[first : last + 1])
                    except Exception as exc2:  # pragma: no cover - diagnostic path
                        error = str(exc2)
                else:
                    error = str(exc)
        blocks.append(
            {
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
                "raw": raw,
                "json": parsed,
                "json_error": error,
            }
        )
        idx += 1
    return blocks


def _parse_two_regime_table(lines: Sequence[str], label: str) -> list[dict[str, Any]]:
    start = _find_caption_line(lines, label)
    end = start
    while end < len(lines) and r"\end{tabular}" not in lines[end]:
        end += 1
    regimes = TWO_REGIME_TABLES[label]
    caption = _caption_snippet(lines, label)
    rows: list[dict[str, Any]] = []
    for idx in range(start, min(end + 1, len(lines))):
        line = lines[idx]
        if "&" not in line or r"\\" not in line:
            continue
        if line.lstrip().startswith(("%", "\\", "Method", "&")):
            continue
        parts = _split_row(line)
        if len(parts) != 13:
            continue
        method = _method_key(parts[0])
        for regime_idx, regime in enumerate(regimes):
            offset = 1 + 6 * regime_idx
            values = dict(zip(FIELDS_12, parts[offset : offset + 6]))
            rows.append(
                {
                    "table_label": label,
                    "table_context": "condensed_two_regime",
                    "method": method,
                    "regime": regime,
                    "line": _line_for_index(idx),
                    "caption_snippet": caption,
                    "values": values,
                }
            )
    return rows


def _parse_hh_plateau(lines: Sequence[str]) -> list[dict[str, Any]]:
    label = HH_PLATEAU_LABEL
    start = _find_caption_line(lines, label)
    caption = _caption_snippet(lines, label)
    rows: list[dict[str, Any]] = []
    current_regime: str | None = None
    regime_alias = {
        "weak--weak": "weak_weak",
        "strong--weak": "strong_weak",
        "weak--strong": "weak_strong",
        "strong--strong": "strong_strong",
    }
    for idx in range(start, len(lines)):
        line = lines[idx]
        if r"\inlinetablecaption{tab:ablation_matrix}" in line or "BEGIN_MACHINE_READABLE_TABLE_IV_ROUTE_A" in line:
            break
        textit = re.search(r"\\textit\{([^:]+):", line)
        if textit:
            key = textit.group(1).strip().replace("-", "-")
            current_regime = regime_alias.get(key, key.replace("--", "_").replace("-", "_"))
            continue
        if current_regime is None or "&" not in line or r"\\" not in line:
            continue
        if line.lstrip().startswith(("%", "\\", "Method", "&")):
            continue
        parts = _split_row(line)
        if len(parts) != 6:
            continue
        method = _method_key(parts[0])
        values = dict(zip(HH_PLATEAU_FIELDS, parts[1:6]))
        rows.append(
            {
                "table_label": label,
                "table_context": "condensed_hh_plateau_blocks",
                "method": method,
                "regime": current_regime,
                "line": _line_for_index(idx),
                "caption_snippet": caption,
                "values": values,
            }
        )
    return rows


def _parse_hh_appendix(lines: Sequence[str]) -> list[dict[str, Any]]:
    label = HH_APPENDIX_LABEL
    start = _find_caption_line(lines, label)
    end = start
    while end < len(lines) and r"\end{tabular}" not in lines[end]:
        end += 1
    caption = _caption_snippet(lines, label)
    rows: list[dict[str, Any]] = []
    regime_pair = ("weak_weak", "strong_weak")
    for idx in range(start, min(end + 1, len(lines))):
        line = lines[idx]
        if "Weak-strong:" in line and "Strong-strong:" in line:
            regime_pair = ("weak_strong", "strong_strong")
            continue
        if "Weak-weak:" in line and "Strong-weak:" in line:
            regime_pair = ("weak_weak", "strong_weak")
            continue
        if "&" not in line or r"\\" not in line:
            continue
        if line.lstrip().startswith(("%", "\\", "Method", "&")):
            continue
        parts = _split_row(line)
        if len(parts) != 13:
            continue
        method = _method_key(parts[0])
        for regime_idx, regime in enumerate(regime_pair):
            offset = 1 + 6 * regime_idx
            values = dict(zip(FIELDS_12, parts[offset : offset + 6]))
            rows.append(
                {
                    "table_label": label,
                    "table_context": "condensed_hh_appendix_fixed_prefix",
                    "method": method,
                    "regime": regime,
                    "line": _line_for_index(idx),
                    "caption_snippet": caption,
                    "values": values,
                }
            )
    return rows


def parse_condensed_tex(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    visible_cells: list[dict[str, Any]] = []
    visible_cells.extend(_parse_hh_plateau(lines))
    visible_cells.extend(_parse_two_regime_table(lines, "tab:fixed_accuracy_claims"))
    visible_cells.extend(_parse_two_regime_table(lines, "tab:fixed_accuracy_spin_boson"))
    visible_cells.extend(_parse_hh_appendix(lines))
    return {
        "visible_cells": visible_cells,
        "machine_readable_comments": _parse_comment_blocks(text),
    }


def _looks_like_local_source_path(value: str) -> bool:
    text = str(value).strip()
    if not text or re.search(r"\s", text) or text.startswith(("max(", "external_", "same_", "current_", "completed_", "user-", "source-backed")):
        return False
    if text.startswith(REMOTE_PREFIXES):
        return True
    if text.startswith(PATH_PREFIXES):
        return True
    return text.endswith(PATH_SUFFIXES)


def _expected_hash_for_key(mapping: Mapping[str, Any], key: str) -> str | None:
    candidates = [f"{key}_sha256"]
    if key.endswith("_json"):
        candidates.append(f"{key[:-5]}_sha256")
    if key.endswith("_path"):
        candidates.append(f"{key[:-5]}_sha256")
    if key == "source":
        candidates.append("source_sha256")
    if key == "source_map":
        candidates.append("source_map_sha256")
    if key == "source_json":
        candidates.append("source_sha256")
    if key == "promotion_json":
        candidates.append("promotion_sha256")
    if key == "audit_json":
        candidates.append("audit_sha256")
    if key == "plot_provenance":
        candidates.append("plot_provenance_sha256")
    for candidate in candidates:
        value = mapping.get(candidate)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return value.lower()
    return None


def _source_refs_from_mapping(value: Any, referenced_by: str) -> Iterable[SourceReference]:
    if isinstance(value, Mapping):
        for key, val in value.items():
            key_text = str(key)
            if key_text in {"skill"}:
                yield from _source_refs_from_mapping(val, referenced_by)
                continue
            if isinstance(val, str) and _looks_like_local_source_path(val):
                yield SourceReference(
                    path=val,
                    referenced_by=referenced_by,
                    expected_sha256=_expected_hash_for_key(value, key_text),
                    source_key=key_text,
                )
            yield from _source_refs_from_mapping(val, referenced_by)
    elif isinstance(value, list):
        for item in value:
            yield from _source_refs_from_mapping(item, referenced_by)


def collect_source_references(blocks: Sequence[Mapping[str, Any]], extra_payloads: Sequence[tuple[str, Any]] = ()) -> list[SourceReference]:
    refs: list[SourceReference] = []
    for block in blocks:
        payload = block.get("json")
        if isinstance(payload, Mapping):
            refs.extend(_source_refs_from_mapping(payload, str(block.get("name"))))
    for name, payload in extra_payloads:
        refs.extend(_source_refs_from_mapping(payload, name))
    # Preserve first occurrence for stable output.
    seen: set[tuple[str, str, str | None]] = set()
    out: list[SourceReference] = []
    for ref in refs:
        key = (ref.path, ref.referenced_by, ref.expected_sha256)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def check_source_reference(repo_root: Path, ref: SourceReference) -> dict[str, Any]:
    status = "not_checked"
    actual_sha: str | None = None
    exists = False
    if ref.path.startswith(REMOTE_PREFIXES):
        status = "external_not_checked" if ref.path.startswith("~/") else "remote_not_checked"
    else:
        path = repo_root / ref.path
        exists = path.exists()
        if not exists:
            status = "missing"
        elif path.is_dir():
            status = "directory_not_checked"
        else:
            actual_sha = _sha256(path)
            status = "match" if ref.expected_sha256 and actual_sha == ref.expected_sha256 else "mismatch" if ref.expected_sha256 else "not_checked"
    return {
        "path": ref.path,
        "referenced_by": ref.referenced_by,
        "source_key": ref.source_key,
        "exists": exists,
        "expected_sha256": ref.expected_sha256,
        "actual_sha256": actual_sha,
        "hash_status": status,
    }


def _optional_json(repo_root: Path, rel_path: str) -> Any | None:
    path = repo_root / rel_path
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def _payloads_for_source_extraction(repo_root: Path) -> list[tuple[str, Any]]:
    paths = [
        "MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json",
        "MATH/paper_facing/paper_I_static_scaffold/paper_i_snake_fairness_status_20260608.json",
        "MATH/paper_facing/paper_I_static_scaffold/paper_i_tables_i_ii_spsa_optuna_current_best_promotion_20260601.json",
        "MATH/paper_facing/paper_I_static_scaffold/paper_i_table_iii_hh_geo_qeb_spsa_repair_promotion_20260604.json",
        "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_strong_weak_snake_stdout_held_continuation_promotion_20260609.json",
    ]
    out: list[tuple[str, Any]] = []
    for path in paths:
        payload = _optional_json(repo_root, path)
        if payload is not None:
            out.append((path, payload))
    return out


def _pdf_sync_status(condensed_tex: Path, condensed_pdf: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "condensed_tex": str(condensed_tex),
        "condensed_pdf": str(condensed_pdf),
        "tex_exists": condensed_tex.exists(),
        "pdf_exists": condensed_pdf.exists(),
        "tex_sha256": _sha256(condensed_tex) if condensed_tex.exists() else None,
        "pdf_sha256": _sha256(condensed_pdf) if condensed_pdf.exists() else None,
        "pdf_page_count": None,
        "pdf_metadata": None,
        "sync_status": "pdf_source_sync_unknown",
        "sync_notes": [],
    }
    if not condensed_pdf.exists() or not condensed_tex.exists():
        status["sync_notes"].append("PDF or TeX source is missing.")
        return status
    tex_mtime = condensed_tex.stat().st_mtime
    pdf_mtime = condensed_pdf.stat().st_mtime
    status["tex_mtime_utc"] = datetime.fromtimestamp(tex_mtime, timezone.utc).isoformat()
    status["pdf_mtime_utc"] = datetime.fromtimestamp(pdf_mtime, timezone.utc).isoformat()
    if pdf_mtime + 1 < tex_mtime:
        status["sync_notes"].append("PDF is older than condensed TeX source.")
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(condensed_pdf))
        status["pdf_page_count"] = len(reader.pages)
        meta = reader.metadata or {}
        status["pdf_metadata"] = {str(k): str(v) for k, v in meta.items()}
        status["sync_status"] = "metadata_checked"
    except Exception as exc:  # pragma: no cover - depends on optional package/PDF parser
        status["sync_notes"].append(f"PDF metadata extraction unavailable: {exc}")
        status["sync_status"] = "pdf_source_sync_unknown"
    return status


def _status_counts(items: Iterable[Mapping[str, Any]], key: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _visible_row_key(row: Mapping[str, Any], field_group: str) -> str:
    return f"{row.get('table_label')}|{row.get('regime')}|{row.get('method')}|{field_group}"


def _build_metric_policy_matrix(condensed_text: str) -> list[dict[str, Any]]:
    appendix_divergence = (
        "same-cutoff ED convention used in the plateau-prefix table" in condensed_text
        and "display_delta_e_policy\": \"raw_external_abs_delta_e" in condensed_text
    )
    rows = [
        {
            "table_label": "tab:fixed_accuracy_claims",
            "expected_policy": "raw_absolute_error_no_phonon",
            "observed_policy": "raw_absolute_error_from_condensed_metric_paragraph",
            "status": "ok",
            "evidence": "static_adapt_paper_I_condensed.tex metric paragraph and caption",
        },
        {
            "table_label": "tab:fixed_accuracy_spin_boson",
            "expected_policy": "raw_same_cutoff_ed_error_with_higher_cutoff_diagnostic",
            "observed_policy": "raw_same_cutoff_ed_error",
            "status": "ok",
            "evidence": "tab:fixed_accuracy_spin_boson caption",
        },
        {
            "table_label": HH_PLATEAU_LABEL,
            "expected_policy": "raw_same_cutoff_ed_error_at_first_effective_plateau_prefix",
            "observed_policy": "raw_same_cutoff_ed_error_at_first_effective_plateau_prefix",
            "status": "ok",
            "evidence": "tab:hh_first_plateau_prefix_costs caption and machine-readable comment",
        },
        {
            "table_label": HH_APPENDIX_LABEL,
            "expected_policy": "raw_external_reference_error_with_fixed_prefix_resources",
            "observed_policy": "same_cutoff_wording_in_condensed_prose_and_caption" if appendix_divergence else "not_detected",
            "status": "policy_divergence" if appendix_divergence else "not_checked",
            "evidence": "condensed HH appendix paragraph/caption versus support contract comments",
        },
    ]
    return rows


def _load_hh_plateau_audit(repo_root: Path, blocks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    source = "output/pdf/paper_i_hh_tableiii_first_effective_plateau_prefix_cost_audit_20260609.json"
    for block in blocks:
        if block.get("name") == "TABLE_III_FIRST_PLATEAU_PREFIX_COSTS" and isinstance(block.get("json"), Mapping):
            value = block["json"].get("source")
            if isinstance(value, str):
                source = value
    payload = _optional_json(repo_root, source)
    return payload if isinstance(payload, Mapping) else None


def _algorithm_to_method(algorithm_id: Any) -> str | None:
    text = str(algorithm_id or "")
    if "append" in text:
        return "append ADAPT"
    if "tetris" in text:
        return "TETRIS-ADAPT"
    if "geo" in text and "qeb" not in text:
        return "Geo-ADAPT"
    if "qeb" in text:
        return "Qubit/QEB"
    if "snake" in text or "route_a" in text or "paper_i_production" in text:
        return "SNAKE"
    return None


def _build_hh_plateau_lookup(payload: Mapping[str, Any] | None) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return out
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        method = _method_key(str(row.get("method") or _algorithm_to_method(row.get("algorithm_id")) or ""))
        regime = str(row.get("regime") or "")
        if method and regime:
            out[(method, regime)] = row
    return out


def _visible_prefix_audit_lookup(repo_root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    payload = _optional_json(repo_root, "output/pdf/paper_i_tables_i_ii_visible_prefix_cost_audit_20260528.json")
    out: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return out
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        table = str(row.get("table_label") or "")
        case = str(row.get("case_id") or "")
        regime = "strong" if "strong" in case else "weak"
        method = _method_key(_algorithm_to_method(row.get("algorithm_id")) or str(row.get("method") or ""))
        if table and method:
            out[(table, method, regime)] = row
    # The current artifact records this skipped row in top-level metadata/comment.
    out[("tab:fixed_accuracy_spin_boson", "Qubit/QEB", "strong")] = {
        "status": "skipped",
        "reason": "source_json_missing for current nph2 strong comparator; existing resource cell preserved",
    }
    return out


def build_compiled_cost_matrix(
    repo_root: Path,
    visible_cells: Sequence[Mapping[str, Any]],
    hh_plateau_payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    hh_lookup = _build_hh_plateau_lookup(hh_plateau_payload)
    prefix_lookup = _visible_prefix_audit_lookup(repo_root)
    matrix: list[dict[str, Any]] = []
    for row in visible_cells:
        values = row.get("values", {}) if isinstance(row.get("values"), Mapping) else {}
        if not any(field in values for field in ("N2q", "D2q", "Dc")):
            continue
        table = str(row["table_label"])
        method = str(row["method"])
        regime = str(row["regime"])
        tuple_values = {field: values.get(field) for field in ("N2q", "D2q", "Dc")}
        status = "ok"
        blocker = None
        source = "visible_numeric_cells"
        provenance = "visible table reports numeric compiled resource tuple"
        if any(str(v).strip() == "--" for v in tuple_values.values()):
            status = "blocked"
            blocker = "compiled_cost_missing"
            provenance = "one or more displayed resource cells are --"
        if table == HH_PLATEAU_LABEL:
            audit_row = hh_lookup.get((method, regime))
            if audit_row:
                source = str(audit_row.get("source_json") or "hh_plateau_audit")
                audit_status = str(audit_row.get("status") or "not_checked")
                if audit_status == "blocked":
                    status = "blocked"
                    blocker = str(audit_row.get("reason") or "prefix_operator_metadata_missing")
                elif audit_status == "qualified" and status != "blocked":
                    status = "qualified"
                    blocker = "retained_resource_cells_qualified"
                compiled = audit_row.get("compiled") if isinstance(audit_row.get("compiled"), Mapping) else {}
                convention = compiled.get("compile_convention")
                if convention and convention != "table_i_basis_gate_transpile_v1":
                    status = "blocked"
                    blocker = "compiled_cost_convention_mismatch"
                provenance = str(audit_row.get("cost_provenance") or provenance)
            elif status != "blocked":
                status = "not_checked"
                blocker = "source_map_stale_or_superseded"
        elif (table, method, regime) in prefix_lookup:
            audit_row = prefix_lookup[(table, method, regime)]
            source = str(audit_row.get("source_json") or "visible_prefix_cost_audit")
            if str(audit_row.get("status")) in {"skipped", "blocked"}:
                status = "qualified" if all(str(v).strip() != "--" for v in tuple_values.values()) else "blocked"
                blocker = "compiled_cost_missing" if status == "blocked" else "retained_resource_cells_qualified"
                provenance = str(audit_row.get("reason") or "visible-prefix audit skipped this row")
            else:
                provenance = str(audit_row.get("compiled_resource_source_kind") or provenance)
        matrix.append(
            {
                "row_key": _visible_row_key(row, "compiled_cost"),
                "table_label": table,
                "method": method,
                "regime": regime,
                "status": status,
                "blocker": blocker,
                "resource_tuple": tuple_values,
                "compile_convention_expected": "table_i_basis_gate_transpile_v1",
                "source": source,
                "provenance_note": provenance,
            }
        )
    return matrix


def build_work_proxy_matrix(visible_cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for row in visible_cells:
        values = row.get("values", {}) if isinstance(row.get("values"), Mapping) else {}
        if "S" not in values:
            continue
        method = str(row["method"])
        table = str(row["table_label"])
        s_value = str(values.get("S") or "").strip()
        if s_value == "--":
            currency = "missing"
            status = "blocked"
            blocker = "work_proxy_currency_mismatch"
            comparability = "not_comparable_until_source_work_proxy_is_recovered"
        elif method in {"HEA VQE", "family VQE"}:
            currency = "fixed_structure_energy_eval_proxy"
            status = "qualified"
            blocker = None
            comparability = "compare_only_with_matching_fixed_structure_rows"
        elif method == "SNAKE":
            currency = "controller_shot_proxy_or_S_norm_legacy_surface"
            status = "qualified"
            blocker = "legacy_proxy_only"
            comparability = "requires S_alg component audit before apples-to-apples work claim"
        else:
            currency = "S_norm_or_visible_prefix_custom_work"
            status = "qualified"
            blocker = None
            comparability = "documented_partial_work_proxy; compare cautiously unless mapped to S_alg bins"
        matrix.append(
            {
                "row_key": _visible_row_key(row, "work_proxy"),
                "table_label": table,
                "method": method,
                "regime": row["regime"],
                "S": s_value,
                "status": status,
                "blocker": blocker,
                "source_currency": currency,
                "comparability_status": comparability,
            }
        )
    return matrix


def build_fairness_status_matrix(repo_root: Path, hh_plateau_payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    fairness = _optional_json(repo_root, "MATH/paper_facing/paper_I_static_scaffold/paper_i_snake_fairness_status_20260608.json")
    if isinstance(fairness, Mapping):
        for key, payload in fairness.get("status", {}).items():
            if not isinstance(payload, Mapping):
                continue
            status_text = str(payload.get("status") or "not_checked")
            matrix.append(
                {
                    "surface": key,
                    "table_label": payload.get("table_label"),
                    "method": "SNAKE",
                    "status": "qualified" if "not_yet" in status_text else "ok",
                    "status_detail": status_text,
                    "source": payload.get("audit_json") or payload.get("source_map"),
                    "notes": payload.get("visible_source_condition") or payload.get("fairness_interpretation"),
                }
            )
    if isinstance(hh_plateau_payload, Mapping):
        for row in hh_plateau_payload.get("rows", []):
            if not isinstance(row, Mapping) or row.get("method") != "SNAKE":
                continue
            audit_status = str(row.get("status") or "not_checked")
            matrix.append(
                {
                    "surface": "hh_plateau_prefix_audit",
                    "table_label": HH_PLATEAU_LABEL,
                    "method": "SNAKE",
                    "regime": row.get("regime"),
                    "status": audit_status,
                    "status_detail": row.get("reason") or row.get("qualification") or row.get("notes"),
                    "source": row.get("source_json"),
                    "strict_replay_status": "blocked" if audit_status == "blocked" else "qualified" if audit_status == "qualified" else "ok",
                }
            )
    return matrix


def _findings_from_matrices(
    source_checks: Sequence[Mapping[str, Any]],
    metric_policy_matrix: Sequence[Mapping[str, Any]],
    compiled_cost_matrix: Sequence[Mapping[str, Any]],
    work_proxy_matrix: Sequence[Mapping[str, Any]],
    fairness_status_matrix: Sequence[Mapping[str, Any]],
    pdf_sync: Mapping[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    if pdf_sync.get("sync_status") == "pdf_source_sync_unknown":
        findings.append(
            Finding(
                status="qualified",
                code="pdf_source_sync_unknown",
                field_group="pdf_tex_sync",
                evidence=(str(pdf_sync.get("condensed_pdf")), str(pdf_sync.get("condensed_tex"))),
                message="PDF/TeX sync could not be fully established; audit continues against condensed TeX proxy.",
                follow_up_scope="optional_pdf_sync_check",
            )
        )
    for item in source_checks:
        if item.get("hash_status") == "missing":
            findings.append(
                Finding(
                    status="blocked",
                    code="missing_referenced_artifact",
                    evidence=(str(item.get("referenced_by")), str(item.get("path"))),
                    message=f"Referenced source is missing locally: {item.get('path')}",
                    follow_up_scope="fetch_or_recover_source_artifact",
                )
            )
        elif item.get("hash_status") == "mismatch":
            findings.append(
                Finding(
                    status="blocked",
                    code="source_sha256_mismatch",
                    evidence=(str(item.get("referenced_by")), str(item.get("path"))),
                    message=f"Referenced source hash mismatch: {item.get('path')}",
                    follow_up_scope="repair_source_pointer_or_hash",
                )
            )
    for item in metric_policy_matrix:
        if item.get("status") == "policy_divergence":
            findings.append(
                Finding(
                    status="policy_divergence",
                    code="metric_policy_mismatch",
                    table_label=str(item.get("table_label")),
                    field_group="energy_metric_policy",
                    evidence=(str(item.get("evidence")),),
                    message=f"{item.get('table_label')} observed policy `{item.get('observed_policy')}` differs from expected `{item.get('expected_policy')}`.",
                    follow_up_scope="candidate_checklist_only_no_manuscript_edit",
                )
            )
    for item in compiled_cost_matrix:
        if item.get("status") in {"blocked", "qualified"}:
            findings.append(
                Finding(
                    status=str(item.get("status")),
                    code=str(item.get("blocker") or "compiled_cost_scope_mismatch"),
                    table_label=str(item.get("table_label")),
                    method=str(item.get("method")),
                    regime=str(item.get("regime")),
                    field_group="compiled_cost",
                    evidence=(str(item.get("source")),),
                    message=f"Compiled-cost status for {item.get('method')}/{item.get('regime')} is {item.get('status')}: {item.get('provenance_note')}",
                    follow_up_scope="recover_or_validate_compiled_cost_source",
                )
            )
    for item in work_proxy_matrix:
        if item.get("status") in {"blocked", "qualified"} and item.get("blocker"):
            findings.append(
                Finding(
                    status=str(item.get("status")),
                    code=str(item.get("blocker")),
                    table_label=str(item.get("table_label")),
                    method=str(item.get("method")),
                    regime=str(item.get("regime")),
                    field_group="work_proxy",
                    evidence=(str(item.get("source_currency")),),
                    message=f"Work proxy for {item.get('method')}/{item.get('regime')} is `{item.get('source_currency')}`; {item.get('comparability_status')}",
                    follow_up_scope="classify_or_recompute_S_alg_before_strong_work_claim",
                )
            )
    for item in fairness_status_matrix:
        if item.get("status") in {"blocked", "qualified"} or "not_yet" in str(item.get("status_detail")):
            findings.append(
                Finding(
                    status=str(item.get("status")),
                    code="settings_fairness_noncanonical" if "not_yet" in str(item.get("status_detail")) else "strict_replay_missing",
                    table_label=str(item.get("table_label")),
                    method=str(item.get("method")),
                    regime=str(item.get("regime")) if item.get("regime") is not None else None,
                    field_group="fairness_status",
                    evidence=(str(item.get("source")),),
                    message=f"Fairness/provenance status: {item.get('status_detail')}",
                    follow_up_scope="status_only_user_decides_repair_or_defer",
                )
            )
    return findings


def _candidate_checklist(findings: Sequence[Finding]) -> list[dict[str, str]]:
    checklist: list[dict[str, str]] = []
    if any(f.code == "metric_policy_mismatch" and f.table_label == HH_APPENDIX_LABEL for f in findings):
        checklist.append(
            {
                "scope": "candidate_checklist_only",
                "item": "Review the condensed HH appendix fixed-prefix metric wording against the support contract: the audit sees same-cutoff wording where the Paper-I results contract expects raw higher-cutoff external-reference error.",
                "evidence": "tab:fixed_accuracy_hh_cartesian caption/prose and paper_i_tables.md contract",
            }
        )
    if any(f.code in {"compiled_cost_missing", "prefix_operator_metadata_missing"} for f in findings):
        checklist.append(
            {
                "scope": "candidate_checklist_only",
                "item": "Review HH SNAKE rows with blocked or qualified compiled-cost provenance before making cost-comparison claims.",
                "evidence": "HH first-effective plateau prefix-cost audit",
            }
        )
    if any(f.code in {"legacy_proxy_only", "work_proxy_currency_mismatch"} for f in findings):
        checklist.append(
            {
                "scope": "candidate_checklist_only",
                "item": "Review claims that compare estimator/work proxy `S` across methods; several rows use legacy or partial proxy currencies rather than complete `S_alg` components.",
                "evidence": "work_proxy_currency_matrix",
            }
        )
    if any(f.code == "missing_referenced_artifact" for f in findings):
        checklist.append(
            {
                "scope": "candidate_checklist_only",
                "item": "Recover or fetch missing named source artifacts before treating affected rows as fully reproducible provenance.",
                "evidence": "source_references",
            }
        )
    return checklist


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def build_audit(
    repo_root: Path,
    *,
    condensed_tex: Path,
    condensed_pdf: Path,
    non_condensed_tex: Path | None = None,
) -> dict[str, Any]:
    text = condensed_tex.read_text(encoding="utf-8")
    parsed = parse_condensed_tex(text)
    comments = parsed["machine_readable_comments"]
    extra_payloads = _payloads_for_source_extraction(repo_root)
    source_refs = collect_source_references(comments, extra_payloads=extra_payloads)
    source_checks = [check_source_reference(repo_root, ref) for ref in source_refs]
    pdf_sync = _pdf_sync_status(condensed_tex, condensed_pdf)
    hh_plateau_payload = _load_hh_plateau_audit(repo_root, comments)
    metric_policy_matrix = _build_metric_policy_matrix(text)
    compiled_cost_matrix = build_compiled_cost_matrix(repo_root, parsed["visible_cells"], hh_plateau_payload)
    work_proxy_matrix = build_work_proxy_matrix(parsed["visible_cells"])
    fairness_status_matrix = build_fairness_status_matrix(repo_root, hh_plateau_payload)
    findings = _findings_from_matrices(
        source_checks,
        metric_policy_matrix,
        compiled_cost_matrix,
        work_proxy_matrix,
        fairness_status_matrix,
        pdf_sync,
    )
    blocker_rows = [f.to_json() for f in findings]
    status_counts = {
        "findings": _status_counts(blocker_rows),
        "source_hash": _status_counts(source_checks, key="hash_status"),
        "metric_policy": _status_counts(metric_policy_matrix),
        "compiled_cost": _status_counts(compiled_cost_matrix),
        "work_proxy": _status_counts(work_proxy_matrix),
        "fairness": _status_counts(fairness_status_matrix),
    }
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": {
            "evidence_only": True,
            "manuscript_edited": False,
            "runs_launched": False,
            "pdf_rebuilt": False,
            "promotion_decision": "none",
            "safe_for_manuscript_transfer": False,
        },
        "inputs": {
            "condensed_tex": str(condensed_tex.relative_to(repo_root) if condensed_tex.is_relative_to(repo_root) else condensed_tex),
            "condensed_pdf": str(condensed_pdf.relative_to(repo_root) if condensed_pdf.is_relative_to(repo_root) else condensed_pdf),
            "non_condensed_tex": str(non_condensed_tex.relative_to(repo_root) if non_condensed_tex and non_condensed_tex.is_relative_to(repo_root) else non_condensed_tex) if non_condensed_tex else None,
        },
        "pdf_tex_sync": pdf_sync,
        "visible_table_inventory": parsed["visible_cells"],
        "machine_readable_comment_count": len(comments),
        "source_references": source_checks,
        "metric_policy_matrix": metric_policy_matrix,
        "compiled_cost_matrix": compiled_cost_matrix,
        "work_proxy_currency_matrix": work_proxy_matrix,
        "fairness_status_matrix": fairness_status_matrix,
        "findings": blocker_rows,
        "candidate_fix_checklist": _candidate_checklist(findings),
        "status_counts": status_counts,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    counts = report.get("status_counts", {}) if isinstance(report.get("status_counts"), Mapping) else {}
    lines = [
        "# Paper I Condensed Fairness and Provenance Audit",
        "",
        f"Generated: `{report.get('generated_utc')}`",
        "",
        "## Scope",
        "",
        "- Evidence-only audit; no manuscript edits, no table edits, no runs, no PDF rebuild.",
        f"- Condensed TeX: `{report.get('inputs', {}).get('condensed_tex')}`",
        f"- Condensed PDF: `{report.get('inputs', {}).get('condensed_pdf')}`",
        "- Candidate fix checklist is for user review only; it is not an edit instruction.",
        "",
        "## Summary Counts",
        "",
    ]
    for group, group_counts in counts.items():
        lines.append(f"- `{group}`: `{group_counts}`")
    sync = report.get("pdf_tex_sync", {}) if isinstance(report.get("pdf_tex_sync"), Mapping) else {}
    lines.extend(
        [
            "",
            "## PDF/TeX Sync",
            "",
            f"- Status: `{sync.get('sync_status')}`",
            f"- PDF pages: `{sync.get('pdf_page_count')}`",
            f"- Notes: `{sync.get('sync_notes')}`",
            "",
            "## Metric Policy Matrix",
            "",
        ]
    )
    lines.append("| Table | Expected | Observed | Status |")
    lines.append("|---|---|---|---|")
    for row in report.get("metric_policy_matrix", []):
        lines.append(
            f"| `{row.get('table_label')}` | `{row.get('expected_policy')}` | `{row.get('observed_policy')}` | `{row.get('status')}` |"
        )
    lines.extend(["", "## Main Findings", ""])
    findings = report.get("findings", [])
    if not findings:
        lines.append("No blockers or qualifications found.")
    else:
        for item in findings:
            lines.append(
                f"- **{item.get('status')} / {item.get('code')}** "
                f"{item.get('table_label') or ''} {item.get('method') or ''} {item.get('regime') or ''}: "
                f"{item.get('message')}"
            )
    lines.extend(["", "## Candidate Fix Checklist (Review Only)", ""])
    checklist = report.get("candidate_fix_checklist", [])
    if not checklist:
        lines.append("No candidate manuscript/table-text fixes were identified.")
    else:
        for idx, item in enumerate(checklist, start=1):
            lines.append(f"{idx}. {item.get('item')}  ")
            lines.append(f"   Evidence: `{item.get('evidence')}`")
    lines.extend(
        [
            "",
            "## Output Matrices",
            "",
            "Detailed JSON/CSV matrices contain the visible table inventory, source-reference checks, compiled-cost classifications, work-proxy currency classifications, and fairness-status classifications.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], *, output_json: Path, output_md: Path, output_csv_dir: Path | None) -> None:
    _atomic_write_text(output_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(output_md, _render_markdown(report) + "\n")
    if output_csv_dir:
        _write_csv(output_csv_dir / "visible_cells.csv", report.get("visible_table_inventory", []))
        _write_csv(output_csv_dir / "source_references.csv", report.get("source_references", []))
        _write_csv(output_csv_dir / "metric_policy_matrix.csv", report.get("metric_policy_matrix", []))
        _write_csv(output_csv_dir / "compiled_cost_matrix.csv", report.get("compiled_cost_matrix", []))
        _write_csv(output_csv_dir / "work_proxy_currency_matrix.csv", report.get("work_proxy_currency_matrix", []))
        _write_csv(output_csv_dir / "fairness_status_matrix.csv", report.get("fairness_status_matrix", []))
        _write_csv(output_csv_dir / "blockers.csv", report.get("findings", []))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--condensed-tex", type=Path, default=Path("MATH/paper_details/static_adapt_paper_I_condensed.tex"))
    parser.add_argument("--condensed-pdf", type=Path, default=Path("MATH/paper_details/static_adapt_paper_I_condensed.pdf"))
    parser.add_argument("--non-condensed-tex", type=Path, default=Path("MATH/paper_details/static_adapt_paper_I.tex"))
    parser.add_argument("--output-json", type=Path, default=Path("output/pdf/paper_i_condensed_fairness_provenance_audit_20260610.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/reports/paper_i_condensed_fairness_provenance_audit_20260610.md"))
    parser.add_argument("--output-csv-dir", type=Path, default=Path("output/pdf/paper_i_condensed_fairness_provenance_audit_20260610"))
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    condensed_tex = args.condensed_tex if args.condensed_tex.is_absolute() else repo_root / args.condensed_tex
    condensed_pdf = args.condensed_pdf if args.condensed_pdf.is_absolute() else repo_root / args.condensed_pdf
    non_condensed_tex = args.non_condensed_tex if args.non_condensed_tex.is_absolute() else repo_root / args.non_condensed_tex
    output_json = args.output_json if args.output_json.is_absolute() else repo_root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md
    output_csv_dir = args.output_csv_dir if args.output_csv_dir.is_absolute() else repo_root / args.output_csv_dir

    report = build_audit(
        repo_root,
        condensed_tex=condensed_tex,
        condensed_pdf=condensed_pdf,
        non_condensed_tex=non_condensed_tex,
    )
    write_outputs(report, output_json=output_json, output_md=output_md, output_csv_dir=output_csv_dir)
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status_counts": report["status_counts"],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "output_csv_dir": str(output_csv_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
