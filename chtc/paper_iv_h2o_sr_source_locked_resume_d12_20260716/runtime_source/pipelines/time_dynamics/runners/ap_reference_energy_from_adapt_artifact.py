"""Generate seed-specific reference energy caches from ADAPT artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.runtime_loader import (
    load_scaffold_runtime_input,
    load_scaffold_runtime_input_from_payload,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.reference_energy_generation import (
    REFERENCE_KIND_EXACT_INITIAL_STATE_V1,
    REFERENCE_KIND_SEED_PREPARED_STATE_V1,
    REFERENCE_METHOD_AUTO,
    REFERENCE_METHOD_SOLVE_IVP_DOP853,
    REFERENCE_METHOD_STATIC_SPECTRAL,
    ReferenceEnergyGenerationConfig,
    generate_reference_energy_trajectory,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    AP_SUPPORTED_PARAMETERIZATION_MODES,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact import (
    _drive_config_from_args,
    _parse_times,
)


RUNNER_SCHEMA_V1 = "ap_reference_energy_from_adapt_artifact_v1"


def generate_reference_energy_from_runtime_input(
    runtime_input: Any,
    *,
    times: Sequence[float],
    enable_drive: bool = False,
    drive_config: Any | None = None,
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
    config: ReferenceEnergyGenerationConfig = ReferenceEnergyGenerationConfig(),
    runner_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bool(enable_drive) and drive_config is None:
        raise ValueError("enable_drive=True requires a drive_config.")
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=str(parameterization_mode),
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=(drive_config if bool(enable_drive) else None),
    )
    reference = generate_reference_energy_trajectory(
        state=state,
        hamiltonian=hamiltonian,
        times=tuple(float(t) for t in times),
        config=config,
        metadata={
            "runner_schema": RUNNER_SCHEMA_V1,
            **dict(runner_metadata or {}),
        },
    )
    return {
        **reference.to_json_dict(),
        "runner_schema": RUNNER_SCHEMA_V1,
        "state": state.to_json_dict(),
        "hamiltonian": hamiltonian.to_json_dict(),
        "decision_data_flow": {
            "reference_energy_cache_scope": "post_run_reporting",
            "uses_reference_for_controller_decision": False,
            "uses_exact_reference_for_controller_decision": False,
        },
    }


def _load_runtime_input_or_raise(
    artifact_path: Path,
    *,
    loader_mode: str | None,
    tag: str | None,
    generator_family: str,
    fallback_family: str,
    replay_candidate_pool_mode: str | None,
) -> Any:
    try:
        if replay_candidate_pool_mode not in {None, ""}:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"Expected JSON object payload at {artifact_path}.")
            payload = dict(payload)
            payload["replay_candidate_pool_mode"] = str(replay_candidate_pool_mode)
            return load_scaffold_runtime_input_from_payload(
                payload,
                artifact_json=artifact_path,
                loader_mode=loader_mode,
                tag=tag,
                generator_family=str(generator_family),
                fallback_family=str(fallback_family),
            )
        return load_scaffold_runtime_input(
            artifact_path,
            loader_mode=loader_mode,
            tag=tag,
            generator_family=str(generator_family),
            fallback_family=str(fallback_family),
        )
    except ValueError as exc:
        if "missing settings object" in str(exc):
            raise ValueError(
                "AP reference-energy runner requires a raw scaffold artifact JSON with "
                "`settings` and `adapt_vqe`/parameterization data. The provided "
                f"path looks like a wrapper/provenance JSON, not a runnable seed artifact: {artifact_path}"
            ) from exc
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a reference energy cache for reporting."
    )
    parser.add_argument("--artifact-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--loader-mode", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--generator-family", default="match_adapt")
    parser.add_argument("--fallback-family", default="full_meta")
    parser.add_argument("--replay-candidate-pool-mode", default=None)
    parser.add_argument("--times", default=None, help="Comma-separated time grid. Overrides --t-final/--num-times.")
    parser.add_argument("--t-final", type=float, default=0.2)
    parser.add_argument("--num-times", type=int, default=3)
    parser.add_argument("--enable-drive", action="store_true")
    parser.add_argument("--drive-A", type=float, default=0.0)
    parser.add_argument("--drive-omega", type=float, default=1.0)
    parser.add_argument("--drive-tbar", type=float, default=1.0)
    parser.add_argument("--drive-phi", type=float, default=0.0)
    parser.add_argument("--drive-pattern", default="staggered")
    parser.add_argument("--drive-custom-weights", default=None)
    parser.add_argument("--drive-include-identity", action="store_true")
    parser.add_argument("--drive-time-sampling", default="midpoint")
    parser.add_argument("--drive-t0", type=float, default=0.0)
    parser.add_argument("--drive-n-sites", type=int, default=None)
    parser.add_argument("--drive-ordering", default=None)
    parser.add_argument(
        "--parameterization-mode",
        choices=AP_SUPPORTED_PARAMETERIZATION_MODES,
        default=AP_PARAMETERIZATION_PER_PAULI_TERM,
        help=(
            "AP variational coordinate mode: per_pauli_term is per Pauli/polynomial "
            "term; logical_shared is per logical/macro generator."
        ),
    )
    parser.add_argument(
        "--reference-kind",
        choices=(REFERENCE_KIND_EXACT_INITIAL_STATE_V1, REFERENCE_KIND_SEED_PREPARED_STATE_V1),
        default=REFERENCE_KIND_EXACT_INITIAL_STATE_V1,
    )
    parser.add_argument(
        "--method",
        choices=(REFERENCE_METHOD_AUTO, REFERENCE_METHOD_STATIC_SPECTRAL, REFERENCE_METHOD_SOLVE_IVP_DOP853),
        default=REFERENCE_METHOD_AUTO,
    )
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--max-internal-step", type=float, default=None)
    parser.add_argument("--norm-drift-tolerance", type=float, default=1.0e-8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    artifact_path = Path(args.artifact_json)
    try:
        runtime_input = _load_runtime_input_or_raise(
            artifact_path,
            loader_mode=args.loader_mode,
            tag=args.tag,
            generator_family=str(args.generator_family),
            fallback_family=str(args.fallback_family),
            replay_candidate_pool_mode=args.replay_candidate_pool_mode,
        )
        drive_config = _drive_config_from_args(args, runtime_input)
        payload = generate_reference_energy_from_runtime_input(
            runtime_input,
            times=_parse_times(args),
            enable_drive=bool(args.enable_drive),
            drive_config=drive_config,
            parameterization_mode=str(args.parameterization_mode),
            config=ReferenceEnergyGenerationConfig(
                reference_kind=str(args.reference_kind),
                method=str(args.method),
                rtol=float(args.rtol),
                atol=float(args.atol),
                max_internal_step=args.max_internal_step,
                norm_drift_tolerance=float(args.norm_drift_tolerance),
            ),
            runner_metadata={
                "artifact_json": str(artifact_path),
                "loader_mode": args.loader_mode,
                "tag": args.tag,
                "generator_family": str(args.generator_family),
                "fallback_family": str(args.fallback_family),
                "replay_candidate_pool_mode": args.replay_candidate_pool_mode,
                "parameterization_mode": str(args.parameterization_mode),
            },
        )
        payload["source_artifact_json"] = str(artifact_path)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNNER_SCHEMA_V1",
    "generate_reference_energy_from_runtime_input",
    "main",
]
