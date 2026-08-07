#!/usr/bin/env python3
"""Build an all-regime HH candidate/Qiskit overlay review PDF.

This is a report-only helper. It does not mutate Optuna storage, source maps,
manuscript tables, or run artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "paper-i-hh-candidate-qiskit-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output/pdf"
DEFAULT_REVIEW_JSON = OUT_DIR / "paper_i_hh_snake_optuna_overlay_review_20260616.json"
DEFAULT_SOURCE_MAP = ROOT / "MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json"
BASE_REPORT_SCRIPT = ROOT / "agent_guidance/skills/paper-i-hh-replay-overlay/scripts/build_hh_replay_overlay_report.py"
U8_ITERATION_REPORT_JSON = OUT_DIR / "paper_i_true_strong_replacement_20260613.json"

REGIME_ORDER = [
    "weak-weak",
    "intermediate-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-weak-u8",
    "strong-strong-u8",
]
SOURCE_MAP_REGIME = {
    "weak-weak": "weak_weak",
    "intermediate-weak": "strong_weak",
    "weak-strong": "weak_strong",
    "intermediate-strong": "strong_strong",
}
METHOD_COLORS = {
    "SNAKE": "#E45756",
    "Paper-I SNAKE": "#E45756",
    "Geo-ADAPT": "#54A24B",
    "Append-ADAPT": "#4C78A8",
}
PLOT_X_LIMITS = {
    "weak-weak": 20,
    "intermediate-weak": 20,
    "weak-strong": 20,
    "intermediate-strong": 25,
}
GEO_FALLBACK_SOURCES = {
    "weak_weak": ROOT
    / "artifacts/chtc_fetch/paper_i_hh_table3_incremental_20260527/raw_outputs/paper_i_hh_symmetric_residual_full_meta_depth500_20260527_v1/static_table__hh__hh_L2_nph2_three_model_sym_weak_weak__static_geo_adapt_vqe/result/result.json",
    "strong_weak": ROOT
    / "artifacts/chtc_fetch/paper_i_hh_residual_proc4_6953576_20260527/raw_outputs/paper_i_hh_symmetric_residual_full_meta_depth500_20260527_v1/static_table__hh__hh_L2_nph2_three_model_sym_strong_weak__static_geo_adapt_vqe/result/result.json",
    "weak_strong": ROOT
    / "artifacts/chtc_fetch/paper_i_hh_residual_geo_nph4_6953576_6_7_20260527/raw_outputs/paper_i_hh_symmetric_residual_full_meta_depth500_20260527_v1/static_table__hh__hh_L2_nph4_three_model_sym_weak_strong__static_geo_adapt_vqe/result/result.json",
    "strong_strong": ROOT
    / "artifacts/chtc_fetch/paper_i_hh_residual_geo_nph4_6953576_6_7_20260527/raw_outputs/paper_i_hh_symmetric_residual_full_meta_depth500_20260527_v1/static_table__hh__hh_L2_nph4_three_model_sym_strong_strong__static_geo_adapt_vqe/result/result.json",
}
U8_ITERATION_REPORT_SECTIONS = {
    "strong-weak-u8": "hh_u8_strong_weak",
    "strong-strong-u8": "hh_u8_strong_strong",
}


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("paper_i_hh_replay_base", BASE_REPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base report helper from {BASE_REPORT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = load_base_module()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_path(raw: Any) -> Path | None:
    if raw in {None, ""}:
        return None
    p = Path(str(raw))
    return p if p.is_absolute() else ROOT / p


def repo_rel(raw: Any) -> str | None:
    if raw in {None, ""}:
        return None
    p = Path(str(raw))
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def offloaded_result_for_original(original: Path | None) -> Path | None:
    if original is None:
        return None
    sidecar = original.with_name("result.offloaded.json")
    if not sidecar.exists():
        return None
    try:
        payload = load_json(sidecar)
    except Exception:
        return None
    for key in ("external_path", "offload_path", "copied_to"):
        raw = payload.get(key) if isinstance(payload, Mapping) else None
        if raw in {None, ""}:
            continue
        path = Path(str(raw))
        if path.exists():
            return path
    return None


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def fmt_num(value: Any, *, missing: str = "--") -> str:
    val = maybe_float(value)
    if val is None:
        return missing
    if val != 0 and abs(val) < 1e-3:
        return f"{val:.3e}"
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:.5g}"


def fmt_cost_triplet(n2q: Any, d2q: Any, dc: Any | None = None) -> str:
    if dc is None:
        return f"{fmt_num(n2q)}/{fmt_num(d2q)}"
    return f"{fmt_num(n2q)}/{fmt_num(d2q)}/{fmt_num(dc)}"


def first_present(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def first_metric(*sources_and_keys: tuple[Mapping[str, Any] | None, Sequence[str]]) -> Any:
    missing_text = {"", "--", "n/a", "na", "none", "null"}
    for source, keys in sources_and_keys:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value.strip().lower() in missing_text:
                continue
            return value
    return None


def fmt_status_num(value: Any, status: Any = None) -> str:
    if maybe_float(value) is not None:
        return fmt_num(value)
    text = str(status or "")
    if "blocked" in text:
        return "blocked"
    if text and text not in {"ok", "None"}:
        return tex_escape(short_text(text, limit=18))
    return "--"


def tex_escape(value: Any) -> str:
    s = "" if value is None else str(value)
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in s)


def short_text(value: Any, *, limit: int = 54) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def replay_keys(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    regime = str(row.get("regime"))
    keys: list[tuple[str, str]] = []
    for prefix, raw in (
        ("id", row.get("trial_id")),
        ("id", row.get("source_trial_id")),
        ("num", row.get("trial_number")),
        ("num", row.get("source_trial_number")),
        ("num", row.get("trial")),
    ):
        if raw in {None, ""}:
            continue
        try:
            value = str(int(raw))
        except Exception:
            value = str(raw)
        key = (regime, f"{prefix}:{value}")
        if key not in keys:
            keys.append(key)
    return keys


def _store_replay(out: dict[tuple[str, str], dict[str, Any]], row: Mapping[str, Any]) -> None:
    for key in replay_keys(row):
        out[key] = dict(row)


def load_source_locked_replays(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        manifest_semantic = (
            payload.get("semantic_overrides")
            if isinstance(payload.get("semantic_overrides"), Mapping)
            else {}
        )
        for row in payload.get("replayed_candidate_rows") or []:
            if not isinstance(row, Mapping):
                continue
            result = repo_path(row.get("replay_result_json"))
            compile_json = repo_path(row.get("compile_json"))
            if result is None or not result.exists():
                continue
            replay_ok = row.get("returncode") == 0 or row.get("status") in {"complete", "completed", "ok", "replay_compile_ok"}
            compile_ok = compile_json is not None and compile_json.exists() and (
                row.get("compile_returncode") in {0, None} or replay_ok
            )
            merged = dict(row)
            merged["source_locked_manifest"] = repo_rel(path)
            merged["source_locked_replay_ok"] = bool(replay_ok)
            merged["source_locked_compile_ok"] = bool(compile_ok)
            if manifest_semantic and not isinstance(merged.get("semantic_overrides"), Mapping):
                merged["semantic_overrides"] = dict(manifest_semantic)
            _store_replay(out, merged)
    return out


def load_source_locked_roots(paths: Sequence[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for root in paths:
        if not root.exists():
            continue
        for result in sorted(root.glob("*/*/trial_*/json/result.json")):
            case_dir = result.parent.parent
            trial_token = case_dir.name.removeprefix("trial_")
            try:
                trial_id = int(trial_token)
            except Exception:
                continue
            regime = case_dir.parent.parent.name
            compile_json = result.parent / "compile_scout_fake_marrakesh.json"
            if not compile_json.exists():
                continue
            row = {
                "regime": regime,
                "source_trial_id": trial_id,
                "replay_result_json": repo_rel(result),
                "compile_json": repo_rel(compile_json),
                "source_locked_root": repo_rel(root),
                "status": "complete",
                "returncode": 0,
                "compile_returncode": 0,
                "source_locked_replay_ok": True,
                "source_locked_compile_ok": True,
            }
            _store_replay(out, row)
    return out


def prior_overlay_candidates(payload: Mapping[str, Any], regime: str, *, limit: int) -> list[dict[str, Any]]:
    regimes = payload.get("regimes")
    block: Mapping[str, Any] | None = None
    if isinstance(regimes, Mapping) and isinstance(regimes.get(regime), Mapping):
        block = regimes.get(regime)
    elif isinstance(regimes, list):
        for item in regimes:
            if isinstance(item, Mapping) and item.get("regime") == regime:
                block = item
                break
    rows = block.get("candidates") if isinstance(block, Mapping) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        item = dict(row)
        item["regime"] = regime
        out.append(item)
        if len(out) >= int(limit):
            break
    return out


def select_report_candidates(payload: Mapping[str, Any], regime: str, *, limit: int) -> list[dict[str, Any]]:
    selected = BASE.select_candidates(payload, regime, limit=limit)
    if selected:
        return selected
    return prior_overlay_candidates(payload, regime, limit=limit)


def visible_rows(review: Mapping[str, Any], regime: str) -> list[dict[str, Any]]:
    for block_name in ("visible_rows", "u8_visible_rows"):
        block = (review.get(block_name) or {}).get(regime)
        if isinstance(block, Mapping) and isinstance(block.get("rows"), list):
            return [dict(r) for r in block["rows"] if isinstance(r, Mapping)]
    regimes = review.get("regimes")
    if isinstance(regimes, Mapping):
        block = regimes.get(regime)
        if isinstance(block, Mapping) and isinstance(block.get("visible_rows"), list):
            return [dict(r) for r in block["visible_rows"] if isinstance(r, Mapping)]
    if isinstance(regimes, list):
        for block in regimes:
            if not isinstance(block, Mapping) or block.get("regime") != regime:
                continue
            if isinstance(block.get("visible_rows"), list):
                return [dict(r) for r in block["visible_rows"] if isinstance(r, Mapping)]
    return []


def normalize_method_text(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").replace("_", " ").split())


def visible_method_row(review: Mapping[str, Any], regime: str, method_substr: str) -> dict[str, Any] | None:
    needle = normalize_method_text(method_substr)
    compact_needle = needle.replace(" ", "")
    for row in visible_rows(review, regime):
        haystack = normalize_method_text(row.get("method", ""))
        if needle in haystack or compact_needle in haystack.replace(" ", ""):
            return row
    return None


def source_map_visible_method_rows(source_map: Mapping[str, Any], review: Mapping[str, Any], regime: str) -> list[dict[str, Any]]:
    source_regime = SOURCE_MAP_REGIME.get(regime)
    if source_regime is None:
        return visible_rows(review, regime)
    reg_cfg = (source_map.get("regimes") or {}).get(source_regime)
    if not isinstance(reg_cfg, Mapping):
        return visible_rows(review, regime)
    methods = reg_cfg.get("methods") if isinstance(reg_cfg.get("methods"), Mapping) else {}
    if not methods:
        return visible_rows(review, regime)
    marker_cfg = ((source_map.get("plateau_markers") or {}).get(source_regime) or {})
    order = source_map.get("method_order")
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        order = ["Append-ADAPT", "TETRIS-ADAPT", "SNAKE", "Geo-ADAPT"]
    rows: list[dict[str, Any]] = []
    for method in order:
        cfg = methods.get(method)
        if not isinstance(cfg, Mapping):
            old = visible_method_row(review, regime, str(method))
            if old is not None:
                rows.append(old)
            continue
        old = visible_method_row(review, regime, str(method)) or {}
        cells = cfg.get("visible_cells") if isinstance(cfg.get("visible_cells"), Mapping) else {}
        marker = marker_cfg.get(method) if isinstance(marker_cfg.get(method), Mapping) else {}
        table_display = cfg.get("table_display_prefix") if isinstance(cfg.get("table_display_prefix"), Mapping) else {}
        table_resource = (
            table_display.get("compiled_resource_cells")
            if isinstance(table_display.get("compiled_resource_cells"), Mapping)
            else {}
        )
        compiled_resource = cfg.get("compiled_resource_cells") if isinstance(cfg.get("compiled_resource_cells"), Mapping) else {}
        retained_resource = (
            cfg.get("resource_cells_retained_from_previous_json_backed_prefix")
            if isinstance(cfg.get("resource_cells_retained_from_previous_json_backed_prefix"), Mapping)
            else {}
        )
        shot_diag = cfg.get("shot_proxy_diagnostic") if isinstance(cfg.get("shot_proxy_diagnostic"), Mapping) else {}
        rows.append(
            {
                "method": method,
                "dE": first_metric(
                    (cells, ("DeltaE", "deltaE", "dE")),
                    (cfg, ("same_cutoff_plateau_abs_delta_e", "display_delta_e_from_promotion")),
                    (old, ("dE",)),
                ),
                "k": first_metric(
                    (table_display, ("k_pl", "prefix_k")),
                    (cfg, ("compiled_cost_prefix_k",)),
                    (marker, ("iteration", "k")),
                    (old, ("k",)),
                ),
                "N2q": first_metric(
                    (cells, ("N2q", "compiled_count_2q_total")),
                    (table_resource, ("N2q", "compiled_count_2q_total")),
                    (compiled_resource, ("N2q", "compiled_count_2q_total")),
                    (retained_resource, ("N2q", "compiled_count_2q_total")),
                    (old, ("N2q",)),
                ),
                "D2q": first_metric(
                    (cells, ("D2q", "compiled_depth_2q_total")),
                    (table_resource, ("D2q", "compiled_depth_2q_total")),
                    (compiled_resource, ("D2q", "compiled_depth_2q_total")),
                    (retained_resource, ("D2q", "compiled_depth_2q_total")),
                    (old, ("D2q",)),
                ),
                "Dc": first_metric(
                    (cells, ("Dc", "D_circ", "compiled_depth_total")),
                    (table_resource, ("Dc", "D_circ", "compiled_depth_total")),
                    (compiled_resource, ("Dc", "D_circ", "compiled_depth_total")),
                    (retained_resource, ("Dc", "D_circ", "compiled_depth_total")),
                    (old, ("Dc",)),
                ),
                "S_alg": first_metric(
                    (cells, ("S", "S_alg", "Salg")),
                    (shot_diag, ("S_alg", "S_norm")),
                    (old, ("S_alg",)),
                ),
                "paper_i_visible_row_source": "source_map_visible_cells",
                "source_map_regime": source_regime,
                "source_json": repo_rel(cfg.get("source_json")),
            }
        )
    return rows


def path_candidates_for_compile(candidate: Mapping[str, Any], result_path: Path | None, replay: Mapping[str, Any] | None) -> list[Path]:
    paths: list[Path] = []
    for raw in (
        None if replay is None else replay.get("compile_json"),
        candidate.get("compile_json"),
    ):
        p = repo_path(raw)
        if p is not None:
            paths.append(p)
    if result_path is not None:
        paths.extend(
            [
                result_path.parent / "compile_scout_fake_marrakesh.json",
                result_path.parent / "qiskit_cost_fake_marrakesh.json",
                result_path.parent / "compile_scout.json",
            ]
        )
    case_dir = repo_path(candidate.get("case_dir"))
    if case_dir is not None:
        paths.extend(
            [
                case_dir / "json/compile_scout_fake_marrakesh.json",
                case_dir / "json/qiskit_cost_fake_marrakesh.json",
                case_dir / "compile_scout_fake_marrakesh.json",
            ]
        )
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def qiskit_cost(candidate: Mapping[str, Any], result_path: Path | None, replay: Mapping[str, Any] | None) -> dict[str, Any]:
    checked: list[str] = []
    for path in path_candidates_for_compile(candidate, result_path, replay):
        checked.append(repo_rel(path) or str(path))
        cost = BASE.qiskit_cost_from_sidecar(path)
        if cost.get("status") == "ok":
            cost["checked_compile_jsons"] = checked
            return cost
    return {"status": "missing", "checked_compile_jsons": checked}


def prefer_source_locked_replay(replay: Mapping[str, Any] | None) -> bool:
    if not isinstance(replay, Mapping):
        return False
    if not replay.get("replay_result_json"):
        return False
    semantic = replay.get("semantic_overrides") if isinstance(replay.get("semantic_overrides"), Mapping) else {}
    if semantic.get("pool_policy") == "full_meta_hva_enabled":
        return True
    return bool(replay.get("source_locked_replay_ok") and replay.get("source_locked_compile_ok"))


def replay_pool_policy(replay: Mapping[str, Any] | None) -> str | None:
    if not isinstance(replay, Mapping):
        return None
    semantic = replay.get("semantic_overrides") if isinstance(replay.get("semantic_overrides"), Mapping) else {}
    raw = semantic.get("pool_policy")
    return None if raw in {None, ""} else str(raw)


def curve_from_result_path(path: Path, reference_energy: float | None = None) -> dict[str, Any]:
    curve = BASE.curve_from_json(path, reference_energy)
    try:
        payload = load_json(path)
    except Exception:
        return curve
    if not isinstance(payload, Mapping):
        return curve
    segment = payload.get("adapt_segment") if isinstance(payload.get("adapt_segment"), Mapping) else {}
    resume_import = payload.get("adapt_resume_import") if isinstance(payload.get("adapt_resume_import"), Mapping) else {}
    base_depth = maybe_float(segment.get("base_depth") if isinstance(segment, Mapping) else None)
    if base_depth is None:
        base_depth = maybe_float(resume_import.get("source_ansatz_depth") if isinstance(resume_import, Mapping) else None)
    if not (isinstance(segment, Mapping) and segment.get("resume_enabled") and base_depth is not None):
        return curve
    base_path = repo_path(resume_import.get("path")) if isinstance(resume_import, Mapping) else None
    if base_path is None or not base_path.exists():
        shifted = dict(curve)
        shifted["depth"] = [int(base_depth) + int(d) for d in curve.get("depth") or []]
        shifted["status"] = "ok_structural_resume_segment_shifted"
        shifted["resume_base_depth"] = int(base_depth)
        return shifted
    try:
        base_curve = BASE.curve_from_json(base_path, reference_energy)
    except Exception:
        base_curve = {"depth": [], "dE": []}
    depths: list[int] = []
    errors: list[float] = []
    for d, e in zip(base_curve.get("depth") or [], base_curve.get("dE") or []):
        dv = maybe_float(d)
        ev = maybe_float(e)
        if dv is None or ev is None or dv > base_depth:
            continue
        depths.append(int(dv))
        errors.append(ev)
    for d, e in zip(curve.get("depth") or [], curve.get("dE") or []):
        dv = maybe_float(d)
        ev = maybe_float(e)
        if dv is None or ev is None:
            continue
        shifted_depth = int(base_depth) + int(dv)
        if depths and shifted_depth == depths[-1] and abs(ev - errors[-1]) <= 1e-14:
            continue
        depths.append(shifted_depth)
        errors.append(ev)
    out = dict(curve)
    out.update(
        {
            "source_json": repo_rel(path),
            "source_sha256": BASE.sha256(path),
            "depth": depths,
            "dE": errors,
            "point_count": len(errors),
            "status": "ok_structural_resume_stitched" if errors else "missing_error_history",
            "resume_base_depth": int(base_depth),
            "resume_base_source_json": repo_rel(base_path),
        }
    )
    return out


def infer_pool_policy(result_path: Path | None, candidate: Mapping[str, Any]) -> str:
    raw_filter = None
    raw_pool = None
    if result_path is not None and result_path.exists():
        try:
            payload = load_json(result_path)
            settings = payload.get("settings") if isinstance(payload, Mapping) else {}
            adapt = payload.get("adapt_vqe") if isinstance(payload, Mapping) else {}
            if isinstance(settings, Mapping):
                raw_filter = settings.get("adapt_pool_class_filter_json")
                raw_pool = settings.get("adapt_pool")
            if raw_filter in {None, ""} and isinstance(adapt, Mapping):
                raw_filter = adapt.get("adapt_pool_class_filter_json")
            if raw_pool in {None, ""} and isinstance(adapt, Mapping):
                raw_pool = adapt.get("adapt_pool")
        except Exception:
            raw_filter = None
    if raw_filter in {None, ""}:
        raw_filter = candidate.get("force_adapt_pool_class_filter_json") or candidate.get("adapt_pool_class_filter_json")
    if raw_pool in {None, ""}:
        raw_pool = candidate.get("force_adapt_pool") or candidate.get("adapt_pool")
    if raw_filter and "minus_hva" in str(raw_filter):
        return "full_meta_minus_hva"
    if raw_filter in {None, ""} and raw_pool == "full_meta":
        return "full_meta_hva_enabled"
    if raw_filter in {None, ""}:
        return "unknown_or_hva_enabled"
    return "filtered_pool"


def resolve_candidate(candidate: Mapping[str, Any], replays: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    original = repo_path(candidate.get("result_json"))
    replay = None
    for key in replay_keys(candidate):
        replay = replays.get(key)
        if replay is not None:
            break
    if replay is None and isinstance(candidate.get("source_locked_replay"), Mapping):
        replay = candidate.get("source_locked_replay")
    replay_result = repo_path(replay.get("replay_result_json")) if replay is not None else None
    if prefer_source_locked_replay(replay) and replay_result is not None and replay_result.exists():
        result_path = replay_result
        evidence = "source_locked_rerun_result"
    elif original is not None and original.exists():
        result_path = original
        evidence = "local_original_result"
    elif (offloaded_result := offloaded_result_for_original(original)) is not None:
        result_path = offloaded_result
        evidence = "offloaded_original_result"
    elif replay_result is not None and replay_result.exists():
        result_path = replay_result
        evidence = "source_locked_rerun_result"
    else:
        result_path = None
        evidence = "missing_result_json"
    cost = qiskit_cost(candidate, result_path, replay)
    curve = None
    if result_path is not None and result_path.exists():
        try:
            curve = curve_from_result_path(result_path, None)
        except Exception as exc:
            curve = {"status": f"read_failed:{type(exc).__name__}", "dE": [], "depth": []}
    trajectory_status = "ok" if isinstance(curve, Mapping) and curve.get("dE") else "missing_error_history"
    if result_path is None:
        trajectory_status = "missing_result_json"
    out = dict(candidate)
    replay_delta_e = replay.get("dE") if isinstance(replay, Mapping) else None
    replay_k = replay.get("k") if isinstance(replay, Mapping) else None
    if maybe_float(replay_delta_e) is None and evidence == "source_locked_rerun_result" and isinstance(curve, Mapping):
        if curve.get("dE"):
            replay_delta_e = curve.get("dE")[-1]
        if maybe_float(replay_k) is None and curve.get("depth"):
            replay_k = curve.get("depth")[-1]
    review_salg = first_present(candidate, "Salg", "S_alg", "paper_i_table_s_alg")
    review_salg_status = first_present(candidate, "Salg_status", "S_alg_status")
    replay_salg = first_present(replay, "Salg", "S_alg")
    replay_salg_status = first_present(replay, "Salg_status", "S_alg_status")
    out.update(
        {
            "original_result_json": repo_rel(candidate.get("result_json")),
            "resolved_result_json": repo_rel(result_path) if result_path is not None else None,
            "evidence_status": evidence,
            "review_dE": candidate.get("dE"),
            "replay_dE": replay_delta_e,
            "display_dE": replay_delta_e if maybe_float(replay_delta_e) is not None else candidate.get("dE"),
            "review_k": candidate.get("k") or candidate.get("adapt_iteration_count"),
            "replay_k": replay_k,
            "display_k": replay_k if maybe_float(replay_k) is not None else candidate.get("k") or candidate.get("adapt_iteration_count"),
            "review_Salg": review_salg,
            "review_Salg_status": review_salg_status,
            "replay_Salg": replay_salg,
            "replay_Salg_status": replay_salg_status,
            "trajectory_status": trajectory_status,
            "trajectory_point_count": 0 if not isinstance(curve, Mapping) else len(curve.get("dE") or []),
            "qiskit_cost": cost,
            "pool_policy_observed": replay_pool_policy(replay) or infer_pool_policy(result_path, candidate),
            "source_locked_replay": None if replay is None else dict(replay),
        }
    )
    return out


def has_complete_candidate_evidence(row: Mapping[str, Any]) -> bool:
    q = row.get("qiskit_cost") if isinstance(row.get("qiskit_cost"), Mapping) else {}
    return bool(row.get("trajectory_status") == "ok" and q.get("status") == "ok" and row.get("resolved_result_json"))


def apply_same_trial_equivalent_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    donors: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not has_complete_candidate_evidence(row):
            continue
        for key in replay_keys(row):
            donors.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        if not has_complete_candidate_evidence(updated):
            donor = None
            for key in replay_keys(updated):
                candidate = donors.get(key)
                if candidate is not None and candidate is not row:
                    donor = candidate
                    break
            if donor is not None:
                updated["resolved_result_json"] = donor.get("resolved_result_json")
                updated["evidence_status"] = "equivalent_same_trial_result"
                updated["trajectory_status"] = donor.get("trajectory_status")
                updated["trajectory_point_count"] = donor.get("trajectory_point_count")
                updated["qiskit_cost"] = dict(donor.get("qiskit_cost") or {})
                updated["pool_policy_observed"] = donor.get("pool_policy_observed")
                if maybe_float(updated.get("display_dE")) is None:
                    updated["display_dE"] = donor.get("display_dE")
                if maybe_float(updated.get("display_k")) is None:
                    updated["display_k"] = donor.get("display_k")
                updated["equivalent_evidence_source"] = {
                    "candidate_rank": donor.get("candidate_rank"),
                    "trial_id": donor.get("trial_id"),
                    "trial_number": donor.get("trial_number"),
                    "trial": donor.get("trial"),
                    "resolved_result_json": donor.get("resolved_result_json"),
                    "basis": "same trial key",
                }
        out.append(updated)
    return out


def method_source_paths(cfg: Mapping[str, Any], method: str) -> list[tuple[str, Path]]:
    keys = ["history_source_json", "previous_slim_history_source_json", "source_json", "previous_source_json"]
    if method == "SNAKE":
        keys = [
            "history_source_json",
            "previous_slim_history_source_json",
            "source_json",
            "base_source_json_before_continuation",
            "previous_source_json",
        ]
    out: list[tuple[str, Path]] = []
    for key in keys:
        p = repo_path(cfg.get(key))
        if p is not None:
            out.append((key, p))
    return out


def visible_marker_curve(review: Mapping[str, Any], regime: str, method: str, *, status: str, checked: Sequence[str] | None = None) -> dict[str, Any] | None:
    vis = visible_method_row(review, regime, method)
    if vis is None:
        return None
    x = maybe_float(vis.get("k"))
    y = maybe_float(vis.get("dE"))
    if x is None or y is None:
        return None
    return {
        "method": method,
        "source_key": "visible_table_plateau_marker",
        "source_json": None,
        "depth": [x],
        "dE": [max(abs(y), 1e-14)],
        "point_count": 1,
        "status": status,
        "checked_sources": list(checked or []),
    }


def u8_iteration_report_curves(review: Mapping[str, Any], regime: str) -> tuple[dict[str, Any], list[str]]:
    section_name = U8_ITERATION_REPORT_SECTIONS.get(regime)
    if section_name is None:
        return {}, []
    blockers: list[str] = []
    if not U8_ITERATION_REPORT_JSON.exists():
        return {}, [f"{regime}:u8_iteration_report_json_missing"]
    try:
        report = load_json(U8_ITERATION_REPORT_JSON)
    except Exception as exc:
        return {}, [f"{regime}:u8_iteration_report_json_read_failed:{type(exc).__name__}"]
    sections = report.get("sections") if isinstance(report, Mapping) else {}
    rows = sections.get(section_name) if isinstance(sections, Mapping) else None
    if not isinstance(rows, list):
        return {}, [f"{regime}:u8_iteration_report_section_missing"]
    curves: dict[str, Any] = {}
    for method in ("SNAKE", "Geo-ADAPT", "Append-ADAPT"):
        wanted = normalize_method_text(method)
        matched = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if normalize_method_text(row.get("method")) == wanted:
                matched = row
                break
        if matched is None:
            marker = visible_marker_curve(
                review,
                regime,
                method,
                status="marker_only_missing_u8_iteration_report_row",
                checked=[repo_rel(U8_ITERATION_REPORT_JSON) or str(U8_ITERATION_REPORT_JSON)],
            )
            if marker is not None:
                curves[method] = marker
            blockers.append(f"{regime}:{method}:u8_iteration_report_row_missing")
            continue
        xs = [maybe_float(x) for x in (matched.get("x") or [])]
        ys = [maybe_float(y) for y in (matched.get("y") or [])]
        points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if points:
            source = repo_path(matched.get("source_json"))
            curves[method] = {
                "method": method,
                "source_key": "u8_iteration_report_history",
                "source_json": repo_rel(source) if source is not None else repo_rel(matched.get("source_json")),
                "history_report_json": repo_rel(U8_ITERATION_REPORT_JSON),
                "source_sha256": BASE.sha256(source) if source is not None and source.exists() else None,
                "history_report_sha256": BASE.sha256(U8_ITERATION_REPORT_JSON),
                "depth": [x for x, _ in points],
                "dE": [max(abs(y), 1e-14) for _, y in points],
                "point_count": len(points),
                "status": "ok_u8_iteration_report_history",
                "source_note": matched.get("source_note"),
                "checked_sources": [repo_rel(U8_ITERATION_REPORT_JSON) or str(U8_ITERATION_REPORT_JSON)],
            }
            continue
        x = maybe_float(matched.get("k"))
        y = maybe_float(matched.get("error"))
        if x is None or y is None:
            marker = visible_marker_curve(
                review,
                regime,
                method,
                status="marker_only_missing_u8_history_but_visible_k",
                checked=[repo_rel(U8_ITERATION_REPORT_JSON) or str(U8_ITERATION_REPORT_JSON)],
            )
            if marker is not None:
                curves[method] = marker
            blockers.append(f"{regime}:{method}:u8_iteration_history_missing")
            continue
        curves[method] = {
            "method": method,
            "source_key": "u8_iteration_report_plateau_marker",
            "source_json": repo_rel(matched.get("source_json")),
            "history_report_json": repo_rel(U8_ITERATION_REPORT_JSON),
            "history_report_sha256": BASE.sha256(U8_ITERATION_REPORT_JSON),
            "depth": [x],
            "dE": [max(abs(y), 1e-14)],
            "point_count": 1,
            "status": "marker_only_u8_iteration_report_no_history",
            "source_note": matched.get("source_note"),
            "checked_sources": [repo_rel(U8_ITERATION_REPORT_JSON) or str(U8_ITERATION_REPORT_JSON)],
        }
        blockers.append(f"{regime}:{method}:u8_iteration_history_missing_marker_only")
    return curves, blockers


def comparator_curves(source_map: Mapping[str, Any], review: Mapping[str, Any], regime: str) -> tuple[dict[str, Any], list[str]]:
    source_regime = SOURCE_MAP_REGIME.get(regime)
    if source_regime is None:
        curves, blockers = u8_iteration_report_curves(review, regime)
        for method in ("SNAKE", "Geo-ADAPT", "Append-ADAPT"):
            if method in curves:
                continue
            marker = visible_marker_curve(
                review,
                regime,
                method,
                status="marker_only_missing_source_map_history_json",
            )
            if marker is not None:
                curves[method] = marker
                blockers.append(f"{regime}:{method}:no_source_map_history_json_marker_at_visible_k")
                continue
            blockers.append(f"{regime}:{method}:no_source_map_history_json_or_visible_k")
        return curves, blockers
    reg_cfg = (source_map.get("regimes") or {}).get(source_regime)
    if not isinstance(reg_cfg, Mapping):
        return {}, [f"{regime}:source_map_regime_missing"]
    reference = maybe_float(reg_cfg.get("reference_energy_same_cutoff") or reg_cfg.get("reference_energy"))
    methods = reg_cfg.get("methods") if isinstance(reg_cfg.get("methods"), Mapping) else {}
    curves: dict[str, Any] = {}
    blockers: list[str] = []
    for method in ("SNAKE", "Geo-ADAPT", "Append-ADAPT"):
        cfg = methods.get(method)
        if not isinstance(cfg, Mapping):
            blockers.append(f"{regime}:{method}:source_block_missing")
            continue
        loaded = None
        checked: list[str] = []
        for key, path in method_source_paths(cfg, method):
            checked.append(f"{key}:{repo_rel(path)}")
            if not path.exists():
                continue
            try:
                curve = curve_from_result_path(path, reference)
                if curve.get("dE"):
                    curve["method"] = method
                    curve["source_key"] = key
                    loaded = curve
                    break
            except Exception as exc:
                checked.append(f"{key}:read_failed:{type(exc).__name__}")
        if loaded is not None:
            curves[method] = loaded
            continue
        if method == "Geo-ADAPT":
            fallback = GEO_FALLBACK_SOURCES.get(source_regime)
            if fallback is not None and fallback.exists():
                checked.append(f"local_full_geo_fallback:{repo_rel(fallback)}")
                try:
                    curve = curve_from_result_path(fallback, reference)
                    if curve.get("dE"):
                        curve["method"] = method
                        curve["source_key"] = "local_full_geo_fallback_plot_only"
                        curve["source_note"] = "Older local full Geo trajectory used for plot shape because current repair source is absent locally; table values still come from current Paper-I rows."
                        curves[method] = curve
                        loaded = curve
                except Exception as exc:
                    checked.append(f"local_full_geo_fallback:read_failed:{type(exc).__name__}")
                if loaded is not None:
                    continue
        marker = ((source_map.get("plateau_markers") or {}).get(source_regime) or {}).get(method)
        vis = visible_method_row(review, regime, method)
        if isinstance(marker, Mapping) or vis is not None:
            x = maybe_float((marker or {}).get("iteration"))
            y = maybe_float((marker or {}).get("error"))
            if x is None and vis is not None:
                x = maybe_float(vis.get("k"))
            if y is None and vis is not None:
                y = maybe_float(vis.get("dE"))
            if x is not None and y is not None:
                curves[method] = {
                    "method": method,
                    "source_key": "visible_plateau_marker_only",
                    "source_json": None,
                    "depth": [x],
                    "dE": [max(abs(y), 1e-14)],
                    "point_count": 1,
                    "status": "marker_only_missing_history_json",
                    "checked_sources": checked,
                }
            elif y is not None:
                blockers.append(f"{regime}:{method}:marker_missing_iteration")
        blockers.append(f"{regime}:{method}:missing_local_history_json")
    return curves, blockers


def candidate_curve(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    p = repo_path(candidate.get("resolved_result_json"))
    if p is None or not p.exists():
        return None
    try:
        curve = curve_from_result_path(p, None)
    except Exception:
        return None
    if not curve.get("dE"):
        return None
    display_y = maybe_float(candidate.get("display_dE", candidate.get("replay_dE")))
    if display_y is None:
        return curve
    xs = list(curve.get("depth") or [])
    ys = list(curve.get("dE") or [])
    display_x = maybe_float(candidate.get("display_k"))
    if display_x is None and xs:
        display_x = maybe_float(xs[-1])
    if display_x is None:
        return curve
    display_y = max(abs(display_y), 1e-14)
    last_x = maybe_float(xs[-1]) if xs else None
    last_y = maybe_float(ys[-1]) if ys else None
    tol = max(5e-10, 1e-3 * max(abs(display_y), abs(last_y or 0.0), 1e-12))
    if last_x is None or last_y is None or int(round(last_x)) != int(round(display_x)) or abs(last_y - display_y) > tol:
        out = dict(curve)
        out["depth"] = xs + [int(display_x) if float(display_x).is_integer() else display_x]
        out["dE"] = ys + [display_y]
        out["point_count"] = len(out["dE"])
        out["status"] = f"{curve.get('status', 'ok')}_with_reported_terminal"
        out["reported_terminal_dE"] = display_y
        out["reported_terminal_k"] = display_x
        if last_y is not None:
            out["history_terminal_dE"] = last_y
        return out
    return curve


def candidate_plot_id(candidate: Mapping[str, Any], fallback: int | None = None) -> str:
    for key in ("candidate_plot_id", "candidate_rank"):
        raw = candidate.get(key)
        if raw not in {None, ""}:
            try:
                return str(int(raw))
            except Exception:
                return str(raw)
    return str(fallback) if fallback is not None else "--"


def plot_regime(
    *,
    out_dir: Path,
    stem: str,
    regime: str,
    candidates: Sequence[Mapping[str, Any]],
    comp_curves: Mapping[str, Any],
) -> str | None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=170)
    plotted = False
    for method in ("Append-ADAPT", "Geo-ADAPT"):
        curve = comp_curves.get(method)
        if not isinstance(curve, Mapping) or not curve.get("dE"):
            continue
        marker_only = "marker_only" in str(curve.get("status") or "")
        ax.plot(
            curve["depth"],
            curve["dE"],
            color=METHOD_COLORS[method],
            marker="^" if method == "Geo-ADAPT" else "o",
            linestyle="None" if marker_only else "-",
            linewidth=1.6,
            markersize=8 if marker_only else 4,
            label=f"{method} {'marker' if marker_only else 'curve'}",
        )
        plotted = True
    snake = comp_curves.get("SNAKE")
    if isinstance(snake, Mapping) and snake.get("dE"):
        marker_only = "marker_only" in str(snake.get("status") or "")
        ax.plot(
            snake["depth"],
            snake["dE"],
            color=METHOD_COLORS["Paper-I SNAKE"],
            marker="*",
            linestyle="None" if marker_only else "--",
            linewidth=2.0,
            markersize=9 if marker_only else 5,
            label=f"Paper-I SNAKE {'marker' if marker_only else 'curve'}",
        )
        plotted = True
    candidate_curve_count = 0
    for cand in candidates:
        curve = candidate_curve(cand)
        if curve is None:
            continue
        xs = list(curve["depth"])
        ys = list(curve["dE"])
        label = "Optuna SNAKE candidates (# = table Plot #)" if candidate_curve_count == 0 else None
        ax.plot(
            xs,
            ys,
            color="#C43C39",
            alpha=0.85 if candidate_curve_count == 0 else 0.34,
            linewidth=2.0 if candidate_curve_count == 0 else 1.1,
            marker=".",
            markersize=3.0,
            label=label,
        )
        if xs and ys:
            backstep = candidate_curve_count % min(4, len(xs))
            pos_idx = max(0, len(xs) - 1 - backstep)
            offset = ((candidate_curve_count % 5) - 2) * 5
            ax.annotate(
                candidate_plot_id(cand, candidate_curve_count + 1),
                xy=(xs[pos_idx], ys[pos_idx]),
                xytext=(6, offset),
                textcoords="offset points",
                color="#8C1D18",
                fontsize=7.5,
                fontweight="bold",
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#C43C39", "alpha": 0.82, "linewidth": 0.6},
            )
        candidate_curve_count += 1
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_yscale("log")
    ax.set_xlabel("ADAPT iteration / prefix")
    ax.set_ylabel(r"$|\Delta E|$")
    ax.set_title(regime)
    ax.margins(x=0.08, y=0.12)
    if regime in PLOT_X_LIMITS:
        ax.set_xlim(left=0, right=PLOT_X_LIMITS[regime])
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    fig_dir = out_dir / f"{stem}_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{stem}_{regime.replace('-', '_')}_overlay.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return repo_rel(png)


def clean_method_name(value: Any) -> str:
    raw = str(value or "")
    return (
        raw.replace("SNAKE (visible Paper I)", "Paper-I SNAKE")
        .replace("SNAKE", "Paper-I SNAKE" if raw.strip() == "SNAKE" else "SNAKE")
        .replace("append ADAPT", "Append-ADAPT")
    )


def candidate_trial_label(row: Mapping[str, Any]) -> str:
    for key in ("trial_number", "trial", "trial_id", "source_trial_number", "source_trial_id"):
        raw = row.get(key)
        if raw not in {None, ""}:
            try:
                return f"{int(raw):04d}"
            except Exception:
                return str(raw)
    return "--"


def candidate_basis_label(row: Mapping[str, Any]) -> str:
    labels = row.get("selection_bases") or row.get("source_labels") or row.get("selection_labels") or []
    if not labels:
        return "--"
    return "; ".join(str(x) for x in labels)


def candidate_tex_row(row: Mapping[str, Any]) -> str:
    q = row.get("qiskit_cost") if isinstance(row.get("qiskit_cost"), Mapping) else {}
    return (
        f"{tex_escape(candidate_plot_id(row))} & "
        f"{tex_escape(candidate_trial_label(row))} & "
        f"{tex_escape(short_text(candidate_basis_label(row), limit=42))} & "
        f"{fmt_num(row.get('review_dE', row.get('dE')))} & "
        f"{fmt_num(row.get('display_dE', row.get('dE')))} & "
        f"{fmt_num(row.get('display_k'))} & "
        f"{fmt_cost_triplet(row.get('N2Q_proxy') or row.get('graph_count_2q'), row.get('D2Q_proxy') or row.get('graph_depth'))} & "
        f"{fmt_num(row.get('Salg') or row.get('S_alg') or row.get('paper_i_table_s_alg'))} & "
        f"{fmt_cost_triplet(q.get('N2q'), q.get('D2q'), q.get('Dc'))} \\\\"
    )


def visible_tex_row(row: Mapping[str, Any]) -> str:
    return (
        f"{tex_escape(clean_method_name(row.get('method')))} & {fmt_num(row.get('dE'))} & -- & "
        f"{fmt_num(row.get('k'))} & "
        f"{fmt_cost_triplet(row.get('N2q'), row.get('D2q'), row.get('Dc'))} & "
        f"{fmt_num(row.get('S_alg'))} & -- \\\\"
    )


def replay_delta_e_match(row: Mapping[str, Any]) -> str:
    review = maybe_float(row.get("review_dE", row.get("dE")))
    replay = maybe_float(row.get("display_dE", row.get("dE")))
    if review is None or replay is None:
        return "--"
    tol = max(5e-10, 1e-3 * max(abs(review), abs(replay), 1e-12))
    return "same" if abs(review - replay) <= tol else "changed"


def extended_candidate_tex_row(row: Mapping[str, Any]) -> str:
    q = row.get("qiskit_cost") if isinstance(row.get("qiskit_cost"), Mapping) else {}
    plot_id = candidate_plot_id(row)
    return (
        f"Optuna SNAKE {tex_escape(plot_id)} & "
        f"{fmt_num(row.get('review_dE', row.get('dE')))} & "
        f"{fmt_num(row.get('display_dE', row.get('dE')))} & "
        f"{fmt_num(row.get('display_k'))} & "
        f"{fmt_cost_triplet(q.get('N2q'), q.get('D2q'), q.get('Dc'))} & "
        f"{fmt_status_num(row.get('review_Salg', first_present(row, 'Salg', 'S_alg', 'paper_i_table_s_alg')), row.get('review_Salg_status', first_present(row, 'Salg_status', 'S_alg_status')))} & "
        f"{fmt_status_num(row.get('replay_Salg'), row.get('replay_Salg_status'))} \\\\"
    )


def best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in candidates:
        q = row.get("qiskit_cost") if isinstance(row.get("qiskit_cost"), Mapping) else {}
        if row.get("trajectory_status") == "ok" and q.get("status") == "ok":
            return row
    return candidates[0] if candidates else None


def visible_priority_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    priority = [
        ("Paper-I SNAKE", ("snake",)),
        ("Geo-ADAPT", ("geo",)),
        ("Append-ADAPT", ("append",)),
    ]
    out: list[Mapping[str, Any]] = []
    used: set[int] = set()
    for _, needles in priority:
        for idx, row in enumerate(rows):
            if idx in used:
                continue
            text = normalize_method_text(row.get("method"))
            if all(n in text for n in needles):
                out.append(row)
                used.add(idx)
                break
    return out


def comparison_tex_row(label: str, delta_e: Any, k: Any, n2q: Any, d2q: Any, dc: Any, s_alg: Any, note: str = "") -> str:
    return (
        f"{tex_escape(label)} & {fmt_num(delta_e)} & {fmt_num(k)} & "
        f"{fmt_cost_triplet(n2q, d2q, dc)} & {fmt_num(s_alg)} & "
        f"{tex_escape(short_text(note, limit=52))} \\\\"
    )


def candidate_comparison_tex_row(row: Mapping[str, Any]) -> str:
    q = row.get("qiskit_cost") if isinstance(row.get("qiskit_cost"), Mapping) else {}
    note = f"plot {candidate_plot_id(row)}, trial {candidate_trial_label(row)}"
    return comparison_tex_row(
        "Best Optuna SNAKE candidate",
        row.get("display_dE", row.get("dE")),
        row.get("display_k"),
        q.get("N2q"),
        q.get("D2q"),
        q.get("Dc"),
        row.get("Salg") or row.get("S_alg") or row.get("paper_i_table_s_alg"),
        note,
    )


def make_tex(payload: Mapping[str, Any], out_dir: Path, stem: str) -> str:
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.6in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Paper-I HH Extended Candidate Tables with Qiskit Costs}\\",
        rf"Generated: {tex_escape(payload['generated_utc'])}",
        r"\end{center}",
        r"\paragraph{Scope.} Each regime keeps the current Paper-I row order and appends the selected Optuna SNAKE rows. Qiskit columns come only from compile sidecars; graph proxy columns are omitted. Candidate review/replay scalars do not certify identical generator selection.",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Summary & Count & & \\",
        r"\midrule",
        rf"Candidate rows & {fmt_num(payload['summary']['candidate_count'])} & & \\",
        rf"Rows with trajectories & {fmt_num(payload['summary']['trajectory_ok_count'])} & & \\",
        rf"Rows with Qiskit sidecars & {fmt_num(payload['summary']['qiskit_ok_count'])} & & \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    for block in payload["regimes"]:
        lines.extend([r"\clearpage", rf"\section*{{{tex_escape(block['regime'])}}}"])
        vis = block.get("visible_rows") or []
        lines.append(r"\subsection*{Current Paper-I Rows Plus Optuna SNAKE Candidates}")
        if vis:
            lines.extend(
                [
                    r"\begin{table}[h!]",
                    r"\centering",
                    r"\small",
                    r"\setlength{\tabcolsep}{4pt}",
                    r"\resizebox{\linewidth}{!}{%",
                    r"\begin{tabular}{@{}lrrrrrr@{}}",
                    r"\toprule",
                    r"Method & Review $|\Delta E|$ & Replay $|\Delta E|$ & k & Qiskit N2q/D2q/Dc & Review $S_{\rm alg}$ & Replay $S_{\rm alg}$ \\",
                    r"\midrule",
                ]
            )
            for row in vis:
                lines.append(visible_tex_row(row))
            candidates = block.get("candidates") or []
            if candidates:
                lines.append(r"\midrule")
                for row in candidates:
                    lines.append(extended_candidate_tex_row(row))
            lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
        else:
            lines.append(r"No visible Paper-I rows were found in the review JSON.")
        plot = block.get("plot_png")
        if plot:
            rel = os.path.relpath(str(ROOT / plot), out_dir)
            lines.extend(
                [
                    r"\subsection*{Energy Error vs Iteration}",
                    rf"\includegraphics[width=0.95\linewidth]{{{tex_escape(rel)}}}",
                ]
            )
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def compile_tex(tex: Path) -> Path:
    if shutil := __import__("shutil"):
        latexmk = shutil.which("latexmk")
    else:
        latexmk = None
    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name]
        subprocess.run(cmd, cwd=tex.parent, check=True)
    else:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=tex.parent, check=True)
    return tex.with_suffix(".pdf")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    p.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    p.add_argument("--output-stem", default=f"paper_i_hh_all_candidate_best_qiskit_overlay_{datetime.now(timezone.utc).strftime('%Y%m%d')}")
    p.add_argument("--limit-candidates", type=int, default=6)
    p.add_argument("--source-locked-manifest", action="append", type=Path, default=[])
    p.add_argument("--source-locked-root", action="append", type=Path, default=[])
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    review_json = args.review_json if args.review_json.is_absolute() else ROOT / args.review_json
    source_map_path = args.source_map if args.source_map.is_absolute() else ROOT / args.source_map
    review = load_json(review_json)
    source_map = load_json(source_map_path)
    manifest_paths = [
        p if p.is_absolute() else ROOT / p
        for p in args.source_locked_manifest
    ]
    if not manifest_paths:
        manifest_paths = sorted(OUT_DIR.glob("paper_i_hh_review_source_locked_reruns*.json"))
    root_paths = [
        p if p.is_absolute() else ROOT / p
        for p in args.source_locked_root
    ]
    replays = load_source_locked_roots(root_paths)
    replays.update(load_source_locked_replays(manifest_paths))
    regimes_payload: list[dict[str, Any]] = []
    all_blockers: list[str] = []
    candidate_count = 0
    trajectory_ok_count = 0
    qiskit_ok_count = 0
    hva_enabled_count = 0
    for regime in REGIME_ORDER:
        selected = select_report_candidates(review, regime, limit=int(args.limit_candidates))
        resolved = apply_same_trial_equivalent_evidence([resolve_candidate(c, replays) for c in selected])
        for idx, row in enumerate(resolved, start=1):
            row["candidate_plot_id"] = idx
        candidate_count += len(resolved)
        trajectory_ok_count += sum(1 for c in resolved if c.get("trajectory_status") == "ok")
        qiskit_ok_count += sum(1 for c in resolved if (c.get("qiskit_cost") or {}).get("status") == "ok")
        hva_enabled_count += sum(1 for c in resolved if c.get("pool_policy_observed") == "full_meta_hva_enabled")
        comp, blockers = comparator_curves(source_map, review, regime)
        for c in resolved:
            if c.get("evidence_status") == "missing_result_json":
                blockers.append(f"{regime}:candidate_rank_{c.get('candidate_rank')}:missing_result_json")
            if (c.get("qiskit_cost") or {}).get("status") != "ok":
                blockers.append(f"{regime}:candidate_rank_{c.get('candidate_rank')}:missing_qiskit_cost")
        plot = plot_regime(out_dir=OUT_DIR, stem=args.output_stem, regime=regime, candidates=resolved, comp_curves=comp)
        visible = source_map_visible_method_rows(source_map, review, regime)
        block = {
            "regime": regime,
            "visible_rows": visible,
            "candidates": resolved,
            "comparator_curves": comp,
            "blockers": blockers,
            "plot_png": plot,
        }
        all_blockers.extend(blockers)
        regimes_payload.append(block)
    payload = {
        "schema": "paper_i_hh_all_candidate_best_qiskit_overlay_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "review_json": repo_rel(review_json),
            "source_map": repo_rel(source_map_path),
            "source_locked_manifests": [repo_rel(p) for p in manifest_paths if p.exists()],
            "source_locked_roots": [repo_rel(p) for p in root_paths if p.exists()],
        },
        "settings": {"candidate_limit": int(args.limit_candidates)},
        "summary": {
            "candidate_count": candidate_count,
            "trajectory_ok_count": trajectory_ok_count,
            "qiskit_ok_count": qiskit_ok_count,
            "hva_enabled_count": hva_enabled_count,
            "blocker_count": len(all_blockers),
            "pool_note": "Reviewed Optuna rows may originate from full_meta_minus_hva; source-locked HVA promotion reruns are reported separately as full_meta_hva_enabled when the result artifact uses unfiltered full_meta.",
        },
        "regimes": regimes_payload,
        "blockers": all_blockers,
    }
    json_path = OUT_DIR / f"{args.output_stem}.json"
    tex_path = OUT_DIR / f"{args.output_stem}.tex"
    write_json(json_path, payload)
    tex_path.write_text(make_tex(payload, OUT_DIR, args.output_stem), encoding="utf-8")
    pdf_path = compile_tex(tex_path)
    print(f"json: {json_path}")
    print(f"tex: {tex_path}")
    print(f"pdf: {pdf_path}")
    print(f"summary: candidates={candidate_count} trajectory_ok={trajectory_ok_count} qiskit_ok={qiskit_ok_count} blockers={len(all_blockers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
