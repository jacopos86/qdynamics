#!/usr/bin/env python3
"""Summarize Paper-I HH scalar-noise robustness in S-proxy units.

This is a diagnostic/reporting helper.  It does not launch runs, edit manuscript
files, or promote table cells.  It reads completed or stopped SNAKE result
payloads, reconstructs normalized measurement-work telemetry with the existing
SNAKE Table-I helper, and attaches the scalar value-noise contract used by the
HH noise route.

Important convention: ``N_eff`` here is the post-expectation scalar-noise model
knob from ``std_E = sigma0_abs / sqrt(N_eff)``.  The script labels products such
as ``S_norm * N_eff`` as scalar-noise model units, not physical hardware shots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.exact_bench.snake_table_i_measurement_work import (
    normalize_snake_measurement_work_row,
    snake_controller_shot_proxy_from_payload,
)

SCHEMA = "paper_i_hh_noise_proxy_summary_v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("adapt_vqe")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _mapping_at(root: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any] | None:
    current: Any = root
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current if isinstance(current, Mapping) else None


def _value_at(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = root
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first_finite(root: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> tuple[float | None, str | None]:
    for path in paths:
        value = _finite_float(_value_at(root, path))
        if value is not None:
            return value, ".".join(path)
    return None, None


def _first_value(root: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> tuple[Any, str | None]:
    for path in paths:
        value = _value_at(root, path)
        if value is not None:
            return value, ".".join(path)
    return None, None


def extract_value_noise_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the scalar value-noise contract from a result/current payload."""

    adapt = _adapt_payload(payload)
    continuation = adapt.get("continuation") if isinstance(adapt.get("continuation"), Mapping) else {}
    candidate_roots: list[tuple[str, Mapping[str, Any]]] = []
    exact_inner = _mapping_at(adapt, ("continuation", "oracle_inner_exact_structure_value_noise", "last_draw"))
    if exact_inner is not None:
        candidate_roots.append(("adapt_vqe.continuation.oracle_inner_exact_structure_value_noise.last_draw", exact_inner))
    exact_inner_top = _mapping_at(adapt, ("phase3_oracle_inner_exact_structure_value_noise", "last_draw"))
    if exact_inner_top is not None:
        candidate_roots.append(("adapt_vqe.phase3_oracle_inner_exact_structure_value_noise.last_draw", exact_inner_top))
    gradient_cfg = _mapping_at(adapt, ("continuation", "oracle_gradient_config", "value_noise"))
    if gradient_cfg is not None:
        candidate_roots.append(("adapt_vqe.continuation.oracle_gradient_config.value_noise", gradient_cfg))

    for source, root in candidate_roots:
        n_eff = _finite_float(root.get("n_eff", root.get("N_eff")))
        sigma0_abs = _finite_float(root.get("sigma0_abs"))
        if n_eff is not None and n_eff > 0.0 and sigma0_abs is not None and sigma0_abs >= 0.0:
            return {
                "status": "ok",
                "model": "gaussian_iid_scalar_value_noise_v1",
                "source": source,
                "N_eff": float(n_eff),
                "sigma0_abs": float(sigma0_abs),
                "std_abs": float(sigma0_abs / math.sqrt(n_eff)),
                "post_expectation_value_noise_not_physical_shots": bool(
                    root.get("post_expectation_value_noise_not_physical_shots", True)
                ),
            }

    mode, mode_source = _first_value(
        adapt,
        (
            ("continuation", "oracle_inner_objective_mode"),
            ("phase3_oracle_inner_objective_mode",),
            ("settings", "phase3_oracle_inner_objective_mode"),
        ),
    )
    return {
        "status": "off_or_missing",
        "model": str(mode) if mode is not None else None,
        "model_source": mode_source,
        "N_eff": None,
        "sigma0_abs": None,
        "std_abs": 0.0 if str(mode or "").strip().lower() in {"off", "exact"} else None,
        "post_expectation_value_noise_not_physical_shots": True,
    }


def reconstruct_measurement_work(payload: Mapping[str, Any], *, source_label: str) -> dict[str, Any]:
    """Reconstruct SNAKE normalized measurement-work fields from runtime telemetry."""

    enriched = normalize_snake_measurement_work_row(
        {},
        source_payload=payload,
        source_label=source_label,
        allow_runtime_reconstruction=True,
    )
    controller = snake_controller_shot_proxy_from_payload(payload, source_label=source_label)
    measurement_work = enriched.get("measurement_work") if isinstance(enriched.get("measurement_work"), Mapping) else {}
    algorithmic_work = (
        enriched.get("algorithmic_measurement_work")
        if isinstance(enriched.get("algorithmic_measurement_work"), Mapping)
        else {}
    )
    return {
        "S_norm_status": enriched.get("S_norm_status"),
        "S_norm": enriched.get("S_norm"),
        "S_norm_components": measurement_work.get("components") if isinstance(measurement_work, Mapping) else None,
        "S_alg_status": enriched.get("S_alg_status"),
        "S_alg": enriched.get("S_alg"),
        "S_alg_components": algorithmic_work.get("components") if isinstance(algorithmic_work, Mapping) else None,
        "controller_proxy_status": controller.get("status"),
        "controller_shot_proxy": controller.get("controller_shot_proxy"),
        "controller_phase_records_with_group_keys": controller.get("controller_phase_records_with_group_keys"),
        "controller_phase_shots_new": controller.get("controller_phase_shots_new"),
    }


def summarize_payload(path: Path, *, role: str, baseline_s_norm: float | None = None) -> dict[str, Any]:
    payload = _load_json(path)
    adapt = _adapt_payload(payload)
    work = reconstruct_measurement_work(payload, source_label=str(path))
    noise = extract_value_noise_contract(payload)
    s_norm = _finite_float(work.get("S_norm"))
    n_eff = _finite_float(noise.get("N_eff"))
    std_abs = _finite_float(noise.get("std_abs"))
    sigma0_abs = _finite_float(noise.get("sigma0_abs"))
    target_abs, target_source = _first_finite(
        adapt,
        (
            ("benchmark_target_abs_delta_e",),
            ("target_abs_delta_e",),
            ("history", "0", "benchmark_target_abs_delta_e"),
        ),
    )
    # history is a list, so handle the common first-row target manually.
    history = adapt.get("history")
    if target_abs is None and isinstance(history, list) and history and isinstance(history[0], Mapping):
        target_abs = _finite_float(history[0].get("benchmark_target_abs_delta_e"))
        target_source = "adapt_vqe.history[0].benchmark_target_abs_delta_e" if target_abs is not None else target_source

    scalar_model_units = float(s_norm * n_eff) if s_norm is not None and n_eff is not None else None
    baseline_factor = (
        float(s_norm / baseline_s_norm)
        if s_norm is not None and baseline_s_norm is not None and baseline_s_norm > 0.0
        else None
    )
    model_units_factor_vs_baseline_events = (
        float(scalar_model_units / baseline_s_norm)
        if scalar_model_units is not None and baseline_s_norm is not None and baseline_s_norm > 0.0
        else None
    )
    target_std_ratio = (
        float(target_abs / std_abs)
        if target_abs is not None and std_abs is not None and std_abs > 0.0
        else None
    )

    return {
        "role": role,
        "path": str(path),
        "artifact_kind": "result" if path.name == "result.json" else path.name.removesuffix(".json"),
        "ansatz_depth": adapt.get("ansatz_depth"),
        "stop_reason": adapt.get("stop_reason"),
        "benchmark_target_hit_success": adapt.get("benchmark_target_hit_success"),
        "energy": adapt.get("energy"),
        "abs_delta_e": adapt.get("abs_delta_e"),
        "benchmark_target_abs_delta_e_current": adapt.get("benchmark_target_abs_delta_e_current"),
        "target_abs_delta_e": target_abs,
        "target_abs_delta_e_source": target_source,
        "measurement_work": work,
        "value_noise": noise,
        "derived_mapping": {
            "schema": "scalar_noise_to_s_proxy_mapping_v1",
            "interpretation": "S_norm counts measurement-bearing estimator/probe/refit events; N_eff is a scalar post-expectation noise model knob, not physical shots.",
            "S_norm": s_norm,
            "S_norm_factor_vs_baseline_S_norm": baseline_factor,
            "scalar_noise_model_units_per_event": n_eff,
            "scalar_noise_model_units_total": scalar_model_units,
            "scalar_noise_model_units_factor_vs_baseline_S_norm": model_units_factor_vs_baseline_events,
            "std_abs": std_abs,
            "sigma0_abs": sigma0_abs,
            "target_abs_delta_e_over_std_abs": target_std_ratio,
            "target_abs_delta_e_over_std_abs_squared": (None if target_std_ratio is None else float(target_std_ratio * target_std_ratio)),
        },
    }


def build_summary(
    *,
    baseline: Path,
    noise_results: Sequence[Path],
    diagnostic_currents: Sequence[Path],
) -> dict[str, Any]:
    baseline_payload = _load_json(baseline)
    baseline_work = reconstruct_measurement_work(baseline_payload, source_label=str(baseline))
    baseline_s_norm = _finite_float(baseline_work.get("S_norm"))
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "Relate HH SNAKE scalar value-noise tolerance to the Paper-I normalized measurement-work proxy without claiming physical shot counts.",
        "conventions": {
            "S_norm": "N_H_outer_eval + N_grad + N_metric + N_H_refit_eval reconstructed by pipelines.exact_bench.snake_table_i_measurement_work.",
            "N_eff": "Scalar value-noise model knob in std_E = sigma0_abs / sqrt(N_eff). It is not a hardware shot count.",
            "scalar_noise_model_units_total": "S_norm * N_eff. This is a model-unit bridge for comparing noise tolerance to event work, not a physical runtime estimate.",
        },
        "baseline": summarize_payload(baseline, role="baseline", baseline_s_norm=baseline_s_norm),
        "noise_results": [
            summarize_payload(path, role="noise_result", baseline_s_norm=baseline_s_norm)
            for path in noise_results
        ],
        "diagnostic_currents": [
            summarize_payload(path, role="diagnostic_current", baseline_s_norm=baseline_s_norm)
            for path in diagnostic_currents
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        tmp = Path(fh.name)
        fh.write(text)
    tmp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Noiseless/local baseline result.json")
    parser.add_argument(
        "--noise-result",
        type=Path,
        action="append",
        default=[],
        help="Completed noisy result.json; may be supplied multiple times",
    )
    parser.add_argument(
        "--diagnostic-current",
        type=Path,
        action="append",
        default=[],
        help="Stopped diagnostic current.json; may be supplied multiple times",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_summary(
        baseline=args.baseline,
        noise_results=args.noise_result,
        diagnostic_currents=args.diagnostic_current,
    )
    if args.output is not None:
        _atomic_write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
