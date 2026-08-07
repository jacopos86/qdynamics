#!/usr/bin/env python3
"""Interpretable meta-analysis for Paper-I HH SNAKE Route-A feature rows.

Consumes the broad combined feature NDJSON produced by
``hh_snake_shallow_feature_extract.py`` plus fixed Paper-I anchors. Emits simple,
auditable summaries: anchor-relative labels, binned knob effects, robust suggested
ranges, and a Markdown report. No sklearn dependency, no run launching.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

PAPER_REGIMES = ("weak-weak", "strong-weak", "weak-strong", "strong-strong")
TUNABLES = (
    "phase1_prune_fraction",
    "batch_near_degenerate_ratio_shared",
    "batch_rank_rel_tol_shared",
    "batch_additivity_slack_scale",
)
ANCHOR_SETS = ("snake_incumbent", "paper_i_geo_useful")
EPS_LOG10 = 0.05
MATERIAL_WIN_LOG10 = 0.05


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _log10(value: float | None) -> float | None:
    if value is None or value <= 0 or not math.isfinite(float(value)):
        return None
    return math.log10(max(float(value), 1.0e-16))


def _quantile(vals: Sequence[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    if len(xs) == 1:
        return xs[0]
    pos = float(q) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_anchor_sets(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sets = payload.get("anchor_sets", {}) if isinstance(payload, Mapping) else {}
    out: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in ANCHOR_SETS}
    for set_name in ANCHOR_SETS:
        rows = sets.get(set_name, []) if isinstance(sets, Mapping) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping) and row.get("regime"):
                out[set_name][str(row["regime"])] = dict(row)
    return out


def _row_delta(row: Mapping[str, Any]) -> float | None:
    return _finite_float(row.get("delta_E_best_proxy") if row.get("delta_E_best_proxy") is not None else row.get("delta_E_best"))


def _row_k(row: Mapping[str, Any]) -> int | None:
    return _finite_int(row.get("k_best_energy_proxy") if row.get("k_best_energy_proxy") is not None else row.get("k_best_energy"))


def _effective_params(row: Mapping[str, Any]) -> Mapping[str, Any]:
    params = row.get("effective_params")
    return params if isinstance(params, Mapping) else {}


def label_row(row: Mapping[str, Any], anchors: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    regime = str(row.get("regime"))
    delta = _row_delta(row)
    k = _row_k(row)
    labels: list[str] = []
    metrics: dict[str, Any] = {}
    for set_name in ANCHOR_SETS:
        anchor = anchors.get(set_name, {}).get(regime)
        if not anchor:
            labels.append(f"missing_{set_name}_anchor")
            continue
        a_delta = _finite_float(anchor.get("delta_E"))
        a_k = _finite_int(anchor.get("k_useful"))
        y = _log10(delta)
        ay = _log10(a_delta)
        prefix = "snake" if set_name == "snake_incumbent" else "geo"
        if delta is not None and a_delta is not None and a_delta > 0:
            ratio = delta / a_delta
            metrics[f"{prefix}_delta_ratio"] = ratio
            if y is not None and ay is not None and y <= ay + EPS_LOG10:
                labels.append(f"{prefix}_parity")
            if y is not None and ay is not None and y <= ay - MATERIAL_WIN_LOG10:
                labels.append(f"{prefix}_material_win")
        if k is not None and a_k not in {None, 0}:
            kratio = float(k) / float(a_k)
            metrics[f"{prefix}_k_ratio"] = kratio
            if kratio <= 1.25:
                labels.append(f"{prefix}_early_or_comparable_k")
            elif kratio >= 2.0:
                labels.append(f"{prefix}_late_only")
    if row.get("include_for_strict_bound_training"):
        labels.append("strict_feature_row")
    for flag in row.get("quality_flags", []) or []:
        labels.append(f"q:{flag}")
    return {"labels": sorted(set(labels)), **metrics}


def _is_strict(row: Mapping[str, Any]) -> bool:
    return bool(row.get("include_for_strict_bound_training")) and str(row.get("regime")) in PAPER_REGIMES


def _is_positive(labeled: Mapping[str, Any]) -> bool:
    labels = set(labeled.get("labels", []))
    return bool({"snake_material_win", "snake_parity"} & labels) and "geo_parity" in labels


def _is_strong_positive(labeled: Mapping[str, Any]) -> bool:
    labels = set(labeled.get("labels", []))
    return "snake_material_win" in labels and "geo_parity" in labels and "snake_late_only" not in labels


def _is_bad(labeled: Mapping[str, Any]) -> bool:
    labels = set(labeled.get("labels", []))
    return "snake_late_only" in labels or ("snake_parity" not in labels and "snake_material_win" not in labels)


def _bin_values(values: Sequence[float], *, log: bool = False, bins: int = 4) -> list[tuple[float, float]]:
    vals = [math.log10(v) if log else v for v in values if v is not None and math.isfinite(float(v)) and (not log or v > 0)]
    if not vals:
        return []
    qs = [_quantile(vals, i / bins) for i in range(bins + 1)]
    edges = [q for q in qs if q is not None]
    out: list[tuple[float, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if log:
            lo, hi = 10.0 ** lo, 10.0 ** hi
        if not out or abs(float(lo) - float(hi)) > 1.0e-18:
            out.append((float(lo), float(hi)))
    if not out:
        v = 10.0 ** vals[0] if log else vals[0]
        out.append((float(v), float(v)))
    return out


def _in_bin(value: float, lo: float, hi: float, *, last: bool) -> bool:
    if last:
        return lo <= value <= hi
    return lo <= value < hi


def binned_effects(rows: Sequence[Mapping[str, Any]], labeled_by_idx: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    strict_rows = [row for row in rows if _is_strict(row)]
    effects: list[dict[str, Any]] = []
    for name in TUNABLES:
        log = name in {"batch_rank_rel_tol_shared", "batch_additivity_slack_scale"}
        vals = [_finite_float(_effective_params(row).get(name)) for row in strict_rows]
        vals = [v for v in vals if v is not None]
        bins = _bin_values(vals, log=log, bins=4)
        for idx_bin, (lo, hi) in enumerate(bins):
            members = []
            for row in strict_rows:
                val = _finite_float(_effective_params(row).get(name))
                if val is None:
                    continue
                if _in_bin(val, lo, hi, last=idx_bin == len(bins) - 1):
                    members.append(row)
            if not members:
                continue
            deltas = [_row_delta(row) for row in members]
            deltas = [d for d in deltas if d is not None]
            labels = [labeled_by_idx[int(row["ml_row_id"])] for row in members if row.get("ml_row_id") is not None and int(row["ml_row_id"]) in labeled_by_idx]
            effects.append(
                {
                    "parameter": name,
                    "bin_index": idx_bin,
                    "range": [lo, hi],
                    "n": len(members),
                    "positive_count": sum(1 for lab in labels if _is_positive(lab)),
                    "strong_positive_count": sum(1 for lab in labels if _is_strong_positive(lab)),
                    "bad_count": sum(1 for lab in labels if _is_bad(lab)),
                    "median_delta_E": median(deltas) if deltas else None,
                    "regime_counts": _counts(str(row.get("regime")) for row in members),
                }
            )
    return effects


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def suggested_ranges(rows: Sequence[Mapping[str, Any]], labeled_by_idx: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    strict_rows = [row for row in rows if _is_strict(row)]
    positives = [row for row in strict_rows if _is_strong_positive(labeled_by_idx.get(int(row.get("ml_row_id", -1)), {}))]
    parity = [row for row in strict_rows if _is_positive(labeled_by_idx.get(int(row.get("ml_row_id", -1)), {}))]
    out: list[dict[str, Any]] = []
    for name in TUNABLES:
        source_rows = positives if len(positives) >= 10 else parity
        vals = [_finite_float(_effective_params(row).get(name)) for row in source_rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 10:
            out.append(
                {
                    "parameter": name,
                    "status": "inconclusive",
                    "support_n": len(vals),
                    "rule": "insufficient_positive_rows_for_robust_interval",
                    "suggested_range": None,
                }
            )
            continue
        out.append(
            {
                "parameter": name,
                "status": "candidate_range",
                "support_n": len(vals),
                "rule": "p10_p90_of_positive_rows_strong_positive_preferred",
                "suggested_range": [_quantile(vals, 0.10), _quantile(vals, 0.90)],
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field)) if isinstance(row.get(field), (dict, list)) else row.get(field) for field in fieldnames})


def write_markdown(path: Path, summary: Mapping[str, Any], ranges: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Paper-I HH SNAKE Route-A interpretable meta-analysis")
    lines.append("")
    lines.append("This is a local ML/meta-analysis artifact. It is not a paper-promotion decision.")
    lines.append("")
    lines.append("## Funnel")
    for key in ["row_count", "strict_row_count", "positive_count", "strong_positive_count", "bad_count"]:
        lines.append(f"- `{key}`: {summary.get(key)}")
    lines.append(f"- by regime: `{json.dumps(summary.get('by_regime', {}), sort_keys=True)}`")
    lines.append("")
    lines.append("## Candidate ranges")
    lines.append("| parameter | status | support | suggested range | median |")
    lines.append("|---|---:|---:|---|---:|")
    for row in ranges:
        lines.append(f"| `{row['parameter']}` | {row['status']} | {row.get('support_n')} | `{row.get('suggested_range')}` | {row.get('median')} |")
    lines.append("")
    lines.append("## Strongest binned effects")
    ranked = sorted(effects, key=lambda r: (-(r.get("strong_positive_count") or 0), -(r.get("positive_count") or 0), r.get("bad_count") or 0, -(r.get("n") or 0)))[:20]
    lines.append("| parameter | bin | n | strong+ | positive | bad | median ΔE | regimes |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in ranked:
        lines.append(
            f"| `{row['parameter']}` | `{row['range']}` | {row['n']} | {row['strong_positive_count']} | {row['positive_count']} | {row['bad_count']} | {row['median_delta_E']} | `{json.dumps(row['regime_counts'], sort_keys=True)}` |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Treat ranges as priors/guardrails, not final Optuna bounds.")
    lines.append("- `lambda_K_scale` and `prune_recoverability_slack_scale` remain mostly missing in recovered rows; do not infer them from this pass.")
    lines.append("- Scheduled/non-flat novelty rows are retained with flags; filter to `phase2_gamma_N_schedule_mode=fixed` for the Paper-I-flat first pass.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(features_path: Path, anchors_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = _load_ndjson(features_path)
    # Ensure stable row ids if caller input lacks them.
    for idx, row in enumerate(rows):
        row.setdefault("ml_row_id", idx)
    anchors = _load_anchor_sets(anchors_path)
    labeled_by_idx: dict[int, dict[str, Any]] = {}
    labeled_rows: list[dict[str, Any]] = []
    for row in rows:
        labels = label_row(row, anchors)
        idx = int(row.get("ml_row_id", len(labeled_rows)))
        labeled_by_idx[idx] = labels
        labeled_rows.append({"ml_row_id": idx, "path": row.get("path"), "regime": row.get("regime"), "delta_E": _row_delta(row), "k": _row_k(row), **labels})
    effects = binned_effects(rows, labeled_by_idx)
    ranges = suggested_ranges(rows, labeled_by_idx)
    strict_rows = [row for row in rows if _is_strict(row)]
    strict_labels = [labeled_by_idx[int(row["ml_row_id"])] for row in strict_rows if int(row["ml_row_id"]) in labeled_by_idx]
    summary = {
        "schema": "paper_i_hh_snake_interpretable_ml_analysis_v1",
        "features_path": str(features_path),
        "anchors_path": str(anchors_path),
        "row_count": len(rows),
        "strict_row_count": len(strict_rows),
        "positive_count": sum(1 for lab in strict_labels if _is_positive(lab)),
        "strong_positive_count": sum(1 for lab in strict_labels if _is_strong_positive(lab)),
        "bad_count": sum(1 for lab in strict_labels if _is_bad(lab)),
        "by_regime": _counts(str(row.get("regime")) for row in strict_rows),
        "candidate_ranges": ranges,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "candidate_ranges.json").write_text(json.dumps(ranges, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "binned_effects.json").write_text(json.dumps(effects, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "labeled_rows.ndjson").open("w", encoding="utf-8") as fh:
        for row in labeled_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(output_dir / "binned_effects.csv", effects, ["parameter", "bin_index", "range", "n", "positive_count", "strong_positive_count", "bad_count", "median_delta_E", "regime_counts"])
    write_markdown(output_dir / "analysis_report.md", summary, ranges, effects)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="Combined feature NDJSON.")
    p.add_argument("--anchors", required=True, help="Fixed baseline anchors JSON.")
    p.add_argument("--output-dir", required=True, help="Directory for analysis outputs.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = analyze(Path(args.features), Path(args.anchors), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
