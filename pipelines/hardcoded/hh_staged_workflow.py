#!/usr/bin/env python3
"""Shared staged Hubbard-Holstein noiseless workflow orchestration.

This module keeps the stage-chain logic out of the existing monolithic
entrypoints. It reuses the production hardcoded primitives instead of
re-implementing warm-start VQE, ADAPT, replay, or time dynamics.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    current_command_string,
    get_PdfPages,
    get_plt,
    render_command_page,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)
from pipelines.hardcoded import adapt_pipeline as adapt_mod
from pipelines.hardcoded.hh_pareto_tracking import (
    extract_staged_hh_pareto_rows,
    write_pareto_tracking,
)
from pipelines.hardcoded.hh_time_dynamics_spectra import (
    build_pair_difference_signal,
    build_staggered_signal,
    compute_one_sided_amplitude_spectrum,
)
from pipelines.hardcoded import hh_vqe_from_adapt_family as replay_mod
from pipelines.hardcoded import hubbard_pipeline as hc_pipeline
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    ControllerDriveConfig,
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    MeasurementTierConfig,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    validate_scaffold_acceptance,
)
from pipelines.hardcoded.handoff_state_bundle import (
    HandoffStateBundleConfig,
    write_handoff_state_bundle,
)
from src.quantum.drives_time_potential import (
    build_gaussian_sinusoid_density_drive,
    reference_method_name,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_hamiltonian,
)


_ALLOWED_NOISELESS_METHODS = ("suzuki2", "cfqm4", "cfqm6", "piecewise_exact")


@dataclass(frozen=True)
class PhysicsConfig:
    L: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int


@dataclass(frozen=True)
class WarmStartConfig:
    ansatz_name: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    progress_every_s: float
    energy_backend: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str


@dataclass(frozen=True)
class AdaptConfig:
    pool: str | None
    continuation_mode: str
    max_depth: int
    maxiter: int
    eps_grad: float
    eps_energy: float
    seed: int
    inner_optimizer: str
    allow_repeats: bool
    finite_angle_fallback: bool
    finite_angle: float
    finite_angle_min_improvement: float
    disable_hh_seed: bool
    reopt_policy: str
    window_size: int
    window_topk: int
    full_refit_every: int
    final_full_refit: bool
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    spsa_callback_every: int
    spsa_progress_every_s: float
    phase1_lambda_F: float
    phase1_lambda_compile: float
    phase1_lambda_measure: float
    phase1_lambda_leak: float
    phase1_score_z_alpha: float
    phase1_probe_max_positions: int
    phase1_plateau_patience: int
    phase1_trough_margin_ratio: float
    phase1_prune_enabled: bool
    phase1_prune_fraction: float
    phase1_prune_max_candidates: int
    phase1_prune_max_regression: float
    phase3_motif_source_json: Path | None
    phase3_symmetry_mitigation_mode: str
    phase3_enable_rescue: bool
    phase3_lifetime_cost_mode: str
    phase3_runtime_split_mode: str
    phase3_oracle_gradient_config: adapt_mod.Phase3OracleGradientConfig | None


@dataclass(frozen=True)
class ReplayConfig:
    generator_family: str
    fallback_family: str
    legacy_paop_key: str
    replay_seed_policy: str
    continuation_mode: str
    reps: int
    restarts: int
    maxiter: int
    method: str
    seed: int
    energy_backend: str
    progress_every_s: float
    wallclock_cap_s: int
    paop_r: int
    paop_split_paulis: bool
    paop_prune_eps: float
    paop_normalization: str
    spsa_a: float
    spsa_c: float
    spsa_alpha: float
    spsa_gamma: float
    spsa_A: float
    spsa_avg_last: int
    spsa_eval_repeats: int
    spsa_eval_agg: str
    replay_freeze_fraction: float
    replay_unfreeze_fraction: float
    replay_full_fraction: float
    replay_qn_spsa_refresh_every: int
    replay_qn_spsa_refresh_mode: str
    phase3_symmetry_mitigation_mode: str


@dataclass(frozen=True)
class DynamicsConfig:
    methods: tuple[str, ...]
    t_final: float
    num_times: int
    trotter_steps: int
    exact_steps_multiplier: int
    fidelity_subspace_energy_tol: float
    cfqm_stage_exp: str
    cfqm_coeff_drop_abs_tol: float
    cfqm_normalize: bool
    enable_drive: bool
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float
    drive_pattern: str
    drive_custom_s: str | None
    drive_include_identity: bool
    drive_time_sampling: str
    drive_t0: float


@dataclass(frozen=True)
class ArtifactConfig:
    tag: str
    output_json: Path
    output_pdf: Path
    handoff_json: Path
    replay_output_json: Path
    replay_output_csv: Path
    replay_output_md: Path
    replay_output_log: Path
    skip_pdf: bool


@dataclass(frozen=True)
class GateConfig:
    ecut_1: float
    ecut_2: float


@dataclass(frozen=True)
class SpectralReportConfig:
    target_observable: str = "auto"
    target_pair: tuple[int, int] | None = None
    detrend: str = "constant"
    window: str = "hann"
    max_harmonic: int = 3


@dataclass(frozen=True)
class StagedHHConfig:
    physics: PhysicsConfig
    warm_start: WarmStartConfig
    adapt: AdaptConfig
    replay: ReplayConfig
    dynamics: DynamicsConfig
    artifacts: ArtifactConfig
    gates: GateConfig
    spectral_report: SpectralReportConfig = field(default_factory=SpectralReportConfig)
    realtime_checkpoint: RealtimeCheckpointConfig = field(default_factory=RealtimeCheckpointConfig)
    smoke_test_intentionally_weak: bool = False
    default_provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class StageExecutionResult:
    h_poly: Any
    hmat: np.ndarray
    ordered_labels_exyz: list[str]
    coeff_map_exyz: dict[str, complex]
    nq_total: int
    psi_hf: np.ndarray
    psi_warm: np.ndarray
    psi_adapt: np.ndarray
    psi_final: np.ndarray
    warm_payload: dict[str, Any]
    adapt_payload: dict[str, Any]
    replay_payload: dict[str, Any]


"""
Δ_rel(E, E_ref) = |E - E_ref| / max(|E_ref|, 1e-14)
"""
def _relative_error_abs(value: float, reference: float) -> float:
    return float(abs(float(value) - float(reference)) / max(abs(float(reference)), 1e-14))


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _bool_flag(raw: Any) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Could not interpret boolean flag value {raw!r}.")


def _parse_noiseless_methods(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    else:
        parts = [str(x).strip().lower() for x in raw if str(x).strip()]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in _ALLOWED_NOISELESS_METHODS:
            raise ValueError(
                f"Unsupported noiseless method '{part}'. Expected subset of {_ALLOWED_NOISELESS_METHODS}."
            )
        if part in seen:
            continue
        seen.add(part)
        out.append(part)
    if not out:
        raise ValueError("At least one noiseless propagation method is required.")
    return tuple(out)


def _parse_drive_custom_weights(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    if text.startswith("["):
        vals = json.loads(text)
    else:
        vals = [float(x) for x in text.split(",") if x.strip()]
    return [float(x) for x in vals]


def _parse_target_pair(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "":
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Spectral target pair must have the form i,j.")
    left = int(parts[0])
    right = int(parts[1])
    if left == right:
        raise ValueError("Spectral target pair must use distinct site indices.")
    return (left, right)


def _parse_checkpoint_controller_step_scales(
    raw: str | Sequence[float] | None,
) -> tuple[float, ...]:
    if raw is None:
        return (1.0,)
    if isinstance(raw, str):
        vals = [float(x) for x in raw.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    out: list[float] = []
    seen: set[float] = set()
    for value in vals:
        scale = float(value)
        if (not math.isfinite(scale)) or scale <= 0.0:
            raise ValueError(
                f"Checkpoint-controller candidate step scales must be finite and positive; got {value!r}."
            )
        rounded = round(scale, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(scale)
    if not out:
        raise ValueError("At least one checkpoint-controller candidate step scale is required.")
    return tuple(out)


def _parse_checkpoint_controller_horizon_weights(
    raw: str | Sequence[float] | None,
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = str(raw).strip()
        if text == "":
            return ()
        vals = [float(x) for x in text.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    out: list[float] = []
    for value in vals:
        weight = float(value)
        if (not math.isfinite(weight)) or weight <= 0.0:
            raise ValueError(
                f"Checkpoint-controller exact forecast horizon weights must be finite and positive; got {value!r}."
            )
        out.append(weight)
    return tuple(out)


def _parse_checkpoint_controller_blend_weights(
    raw: str | Sequence[float] | None,
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = str(raw).strip()
        if text == "":
            return ()
        vals = [float(x) for x in text.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    out: list[float] = []
    seen: set[float] = set()
    for value in vals:
        weight = float(value)
        if (not math.isfinite(weight)) or weight < -1.0 or weight > 1.0:
            raise ValueError(
                f"Checkpoint-controller exact baseline blend weights must be finite and lie in [-1, 1]; got {value!r}."
            )
        rounded = round(weight, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(weight)
    return tuple(out)


def _parse_checkpoint_controller_gain_scales(
    raw: str | Sequence[float] | None,
) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = str(raw).strip()
        if text == "":
            return ()
        vals = [float(x) for x in text.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    out: list[float] = []
    seen: set[float] = set()
    for value in vals:
        scale = float(value)
        if (not math.isfinite(scale)) or scale <= 0.0:
            raise ValueError(
                f"Checkpoint-controller exact baseline gain scales must be finite and positive; got {value!r}."
            )
        rounded = round(scale, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(scale)
    return tuple(out)


def _half_filled_particles(L: int) -> tuple[int, int]:
    n_up, n_dn = hc_pipeline._half_filled_particles(int(L))
    return int(n_up), int(n_dn)


def _hh_nq_total(L: int, n_ph_max: int, boson_encoding: str) -> int:
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    return int(2 * int(L) + int(L) * qpb)


def _default_output_tag(
    *,
    L: int,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    ordering: str,
    boundary: str,
    sector_n_up: int,
    sector_n_dn: int,
    drive_enabled: bool,
    drive_pattern: str,
    drive_A: float,
    drive_omega: float,
    drive_tbar: float,
    drive_phi: float,
    drive_time_sampling: str,
    noiseless_methods: str,
    adapt_continuation_mode: str,
) -> str:
    drive_label = "drive" if bool(drive_enabled) else "static"
    spec = {
        "L": int(L),
        "t": float(t),
        "u": float(u),
        "dv": float(dv),
        "omega0": float(omega0),
        "g_ep": float(g_ep),
        "n_ph_max": int(n_ph_max),
        "ordering": str(ordering),
        "boundary": str(boundary),
        "sector_n_up": int(sector_n_up),
        "sector_n_dn": int(sector_n_dn),
        "drive_enabled": bool(drive_enabled),
        "drive_pattern": str(drive_pattern),
        "drive_A": float(drive_A),
        "drive_omega": float(drive_omega),
        "drive_tbar": float(drive_tbar),
        "drive_phi": float(drive_phi),
        "drive_time_sampling": str(drive_time_sampling),
        "noiseless_methods": str(noiseless_methods),
        "adapt_continuation_mode": str(adapt_continuation_mode),
    }
    digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return (
        f"hh_staged_L{int(L)}_{drive_label}_"
        f"t{float(t):g}_U{float(u):g}_dv{float(dv):g}_w{float(omega0):g}_g{float(g_ep):g}_nph{int(n_ph_max)}_{digest}"
    )


"""
ws_reps(L) = L
ws_restarts(L) = ceil(5L/3)
ws_maxiter(L) = round(4000 L^2 / 9)
adapt_max_depth(L) = 40L
adapt_maxiter(L) = round(5000 L^2 / 9)
final_reps(L) = L
final_restarts(L) := ws_restarts(L)   [workflow inference]
final_maxiter(L) := ws_maxiter(L)     [workflow inference]
t_final(L) = 5L
trotter_steps(L) = 64L
num_times(L) = 1 + ceil(200L/3)
exact_steps_multiplier(L) = ceil((L + 1)/2)
"""
def _scaled_defaults(L: int) -> dict[str, Any]:
    L_int = int(L)
    return {
        "warm_reps": int(max(1, L_int)),
        "warm_restarts": int(max(1, math.ceil((5.0 * L_int) / 3.0))),
        "warm_maxiter": int(max(200, round((4000.0 * L_int * L_int) / 9.0))),
        "adapt_max_depth": int(max(15, 40 * L_int)),
        "adapt_maxiter": int(max(300, round((5000.0 * L_int * L_int) / 9.0))),
        "adapt_eps_grad": 5e-7,
        "adapt_eps_energy": 1e-9,
        "final_reps": int(max(1, L_int)),
        "final_restarts": int(max(1, math.ceil((5.0 * L_int) / 3.0))),
        "final_maxiter": int(max(200, round((4000.0 * L_int * L_int) / 9.0))),
        "t_final": float(5.0 * L_int),
        "trotter_steps": int(64 * L_int),
        "num_times": int(1 + math.ceil((200.0 * L_int) / 3.0)),
        "exact_steps_multiplier": int(math.ceil((L_int + 1) / 2.0)),
    }


def _resolve_with_default(
    *,
    name: str,
    raw: Any,
    default: Any,
    provenance: dict[str, str],
    default_source: str,
) -> Any:
    if raw is None:
        provenance[name] = str(default_source)
        return default
    provenance[name] = "cli"
    return raw


def _enforce_not_weaker(
    *,
    cfg_values: Mapping[str, Any],
    baseline: Mapping[str, Any],
    smoke_test_intentionally_weak: bool,
) -> None:
    if bool(smoke_test_intentionally_weak):
        return
    checks = {
        "warm_reps": int(cfg_values["warm_reps"]) >= int(baseline["warm_reps"]),
        "warm_restarts": int(cfg_values["warm_restarts"]) >= int(baseline["warm_restarts"]),
        "warm_maxiter": int(cfg_values["warm_maxiter"]) >= int(baseline["warm_maxiter"]),
        "adapt_max_depth": int(cfg_values["adapt_max_depth"]) >= int(baseline["adapt_max_depth"]),
        "adapt_maxiter": int(cfg_values["adapt_maxiter"]) >= int(baseline["adapt_maxiter"]),
        "final_reps": int(cfg_values["final_reps"]) >= int(baseline["final_reps"]),
        "final_restarts": int(cfg_values["final_restarts"]) >= int(baseline["final_restarts"]),
        "final_maxiter": int(cfg_values["final_maxiter"]) >= int(baseline["final_maxiter"]),
        "trotter_steps": int(cfg_values["trotter_steps"]) >= int(baseline["trotter_steps"]),
    }
    failed = [key for key, ok in checks.items() if not bool(ok)]
    if failed:
        raise ValueError(
            "Under-parameterized staged HH run rejected. "
            f"Failed fields: {failed}. Baseline defaults: {dict(baseline)}. "
            "Use --smoke-test-intentionally-weak only for explicit smoke tests."
        )


def resolve_staged_hh_config(args: Any) -> StagedHHConfig:
    L = int(getattr(args, "L"))
    defaults = _scaled_defaults(L)
    provenance: dict[str, str] = {}

    sector_n_up_raw = getattr(args, "sector_n_up", None)
    sector_n_dn_raw = getattr(args, "sector_n_dn", None)
    sector_n_up_default, sector_n_dn_default = _half_filled_particles(L)

    sector_n_up = int(sector_n_up_default if sector_n_up_raw is None else sector_n_up_raw)
    sector_n_dn = int(sector_n_dn_default if sector_n_dn_raw is None else sector_n_dn_raw)
    if (int(sector_n_up), int(sector_n_dn)) != (int(sector_n_up_default), int(sector_n_dn_default)):
        raise ValueError(
            "hh_staged_noiseless currently supports only the half-filled sector across all stages. "
            "Non-default --sector-n-up/--sector-n-dn overrides are not yet plumbed through warm-start and ADAPT."
        )

    tag = _resolve_with_default(
        name="tag",
        raw=getattr(args, "tag", None),
        default=_default_output_tag(
            L=L,
            t=float(getattr(args, "t")),
            u=float(getattr(args, "u")),
            dv=float(getattr(args, "dv")),
            omega0=float(getattr(args, "omega0")),
            g_ep=float(getattr(args, "g_ep")),
            n_ph_max=int(getattr(args, "n_ph_max")),
            ordering=str(getattr(args, "ordering")),
            boundary=str(getattr(args, "boundary")),
            sector_n_up=int(sector_n_up),
            sector_n_dn=int(sector_n_dn),
            drive_enabled=bool(getattr(args, "enable_drive")),
            drive_pattern=str(getattr(args, "drive_pattern")),
            drive_A=float(getattr(args, "drive_A")),
            drive_omega=float(getattr(args, "drive_omega")),
            drive_tbar=float(getattr(args, "drive_tbar")),
            drive_phi=float(getattr(args, "drive_phi")),
            drive_time_sampling=str(getattr(args, "drive_time_sampling")),
            noiseless_methods=str(getattr(args, "noiseless_methods")),
            adapt_continuation_mode=str(getattr(args, "adapt_continuation_mode")),
        ),
        provenance=provenance,
        default_source="workflow.tag.default",
    )

    output_json = Path(
        _resolve_with_default(
            name="output_json",
            raw=getattr(args, "output_json", None),
            default=REPO_ROOT / "artifacts" / "json" / f"{tag}.json",
            provenance=provenance,
            default_source="artifacts/json/<tag>.json",
        )
    )
    output_pdf = Path(
        _resolve_with_default(
            name="output_pdf",
            raw=getattr(args, "output_pdf", None),
            default=REPO_ROOT / "artifacts" / "pdf" / f"{tag}.pdf",
            provenance=provenance,
            default_source="artifacts/pdf/<tag>.pdf",
        )
    )

    handoff_json = REPO_ROOT / "artifacts" / "json" / f"{tag}_adapt_handoff.json"
    replay_output_json = REPO_ROOT / "artifacts" / "json" / f"{tag}_replay.json"
    replay_output_csv = REPO_ROOT / "artifacts" / "json" / f"{tag}_replay.csv"
    replay_output_md = REPO_ROOT / "artifacts" / "useful" / f"L{L}" / f"{tag}_replay.md"
    replay_output_log = REPO_ROOT / "artifacts" / "logs" / f"{tag}_replay.log"

    cfg_values = {
        "warm_reps": _resolve_with_default(
            name="warm_reps",
            raw=getattr(args, "warm_reps", None),
            default=defaults["warm_reps"],
            provenance=provenance,
            default_source="run_guide.ws_reps(L)=L",
        ),
        "warm_restarts": _resolve_with_default(
            name="warm_restarts",
            raw=getattr(args, "warm_restarts", None),
            default=defaults["warm_restarts"],
            provenance=provenance,
            default_source="run_guide.ws_restarts(L)=ceil(5L/3)",
        ),
        "warm_maxiter": _resolve_with_default(
            name="warm_maxiter",
            raw=getattr(args, "warm_maxiter", None),
            default=defaults["warm_maxiter"],
            provenance=provenance,
            default_source="run_guide.ws_maxiter(L)=round(4000L^2/9)",
        ),
        "adapt_max_depth": _resolve_with_default(
            name="adapt_max_depth",
            raw=getattr(args, "adapt_max_depth", None),
            default=defaults["adapt_max_depth"],
            provenance=provenance,
            default_source="run_guide.adapt_max_depth(L)=40L",
        ),
        "adapt_maxiter": _resolve_with_default(
            name="adapt_maxiter",
            raw=getattr(args, "adapt_maxiter", None),
            default=defaults["adapt_maxiter"],
            provenance=provenance,
            default_source="run_guide.adapt_maxiter(L)=round(5000L^2/9)",
        ),
        "adapt_eps_grad": _resolve_with_default(
            name="adapt_eps_grad",
            raw=getattr(args, "adapt_eps_grad", None),
            default=defaults["adapt_eps_grad"],
            provenance=provenance,
            default_source="run_guide.adapt_eps_grad=5e-7",
        ),
        "adapt_eps_energy": _resolve_with_default(
            name="adapt_eps_energy",
            raw=getattr(args, "adapt_eps_energy", None),
            default=defaults["adapt_eps_energy"],
            provenance=provenance,
            default_source="run_guide.adapt_eps_energy=1e-9",
        ),
        "final_reps": _resolve_with_default(
            name="final_reps",
            raw=getattr(args, "final_reps", None),
            default=defaults["final_reps"],
            provenance=provenance,
            default_source="run_guide.vqe_reps(L)=L",
        ),
        "final_restarts": _resolve_with_default(
            name="final_restarts",
            raw=getattr(args, "final_restarts", None),
            default=defaults["final_restarts"],
            provenance=provenance,
            default_source="workflow.final_restarts := warm_restarts(L)",
        ),
        "final_maxiter": _resolve_with_default(
            name="final_maxiter",
            raw=getattr(args, "final_maxiter", None),
            default=defaults["final_maxiter"],
            provenance=provenance,
            default_source="workflow.final_maxiter := warm_maxiter(L)",
        ),
        "t_final": _resolve_with_default(
            name="t_final",
            raw=getattr(args, "t_final", None),
            default=defaults["t_final"],
            provenance=provenance,
            default_source="run_guide.t_final(L)=5L",
        ),
        "trotter_steps": _resolve_with_default(
            name="trotter_steps",
            raw=getattr(args, "trotter_steps", None),
            default=defaults["trotter_steps"],
            provenance=provenance,
            default_source="run_guide.trotter_steps(L)=64L",
        ),
        "num_times": _resolve_with_default(
            name="num_times",
            raw=getattr(args, "num_times", None),
            default=defaults["num_times"],
            provenance=provenance,
            default_source="run_guide.num_times(L)=1+ceil(200L/3)",
        ),
        "exact_steps_multiplier": _resolve_with_default(
            name="exact_steps_multiplier",
            raw=getattr(args, "exact_steps_multiplier", None),
            default=defaults["exact_steps_multiplier"],
            provenance=provenance,
            default_source="run_guide.exact_steps_multiplier(L)=ceil((L+1)/2)",
        ),
    }
    _enforce_not_weaker(
        cfg_values=cfg_values,
        baseline=defaults,
        smoke_test_intentionally_weak=bool(getattr(args, "smoke_test_intentionally_weak", False)),
    )

    physics = PhysicsConfig(
        L=L,
        t=float(getattr(args, "t")),
        u=float(getattr(args, "u")),
        dv=float(getattr(args, "dv")),
        omega0=float(getattr(args, "omega0")),
        g_ep=float(getattr(args, "g_ep")),
        n_ph_max=int(getattr(args, "n_ph_max")),
        boson_encoding=str(getattr(args, "boson_encoding")),
        ordering=str(getattr(args, "ordering")),
        boundary=str(getattr(args, "boundary")),
        sector_n_up=int(sector_n_up),
        sector_n_dn=int(sector_n_dn),
    )
    warm_start = WarmStartConfig(
        ansatz_name="hh_hva_ptw",
        reps=int(cfg_values["warm_reps"]),
        restarts=int(cfg_values["warm_restarts"]),
        maxiter=int(cfg_values["warm_maxiter"]),
        method=str(getattr(args, "warm_method")),
        seed=int(getattr(args, "warm_seed")),
        progress_every_s=float(getattr(args, "warm_progress_every_s")),
        energy_backend=str(getattr(args, "vqe_energy_backend")),
        spsa_a=float(getattr(args, "vqe_spsa_a")),
        spsa_c=float(getattr(args, "vqe_spsa_c")),
        spsa_alpha=float(getattr(args, "vqe_spsa_alpha")),
        spsa_gamma=float(getattr(args, "vqe_spsa_gamma")),
        spsa_A=float(getattr(args, "vqe_spsa_A")),
        spsa_avg_last=int(getattr(args, "vqe_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "vqe_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "vqe_spsa_eval_agg")),
    )
    adapt_mode = str(getattr(args, "adapt_continuation_mode"))
    phase3_oracle_gradient_config: adapt_mod.Phase3OracleGradientConfig | None = None
    phase3_oracle_gradient_mode_key = str(getattr(args, "phase3_oracle_gradient_mode", "off")).strip().lower()
    if phase3_oracle_gradient_mode_key != "off":
        phase3_oracle_gradient_config = adapt_mod._resolve_phase3_oracle_gradient_config(
            adapt_mod.Phase3OracleGradientConfig(
                noise_mode=str(phase3_oracle_gradient_mode_key),
                shots=int(getattr(args, "phase3_oracle_shots")),
                oracle_repeats=int(getattr(args, "phase3_oracle_repeats")),
                oracle_aggregate=str(getattr(args, "phase3_oracle_aggregate")),
                backend_name=(
                    None
                    if getattr(args, "phase3_oracle_backend_name", None) in {None, ""}
                    else str(getattr(args, "phase3_oracle_backend_name"))
                ),
                use_fake_backend=bool(getattr(args, "phase3_oracle_use_fake_backend")),
                seed=int(getattr(args, "phase3_oracle_seed")),
                gradient_step=(
                    float(getattr(args, "phase3_oracle_gradient_step"))
                    if getattr(args, "phase3_oracle_gradient_step", None) is not None
                    else float(getattr(args, "adapt_finite_angle"))
                ),
                mitigation_mode=str(getattr(args, "phase3_oracle_mitigation")),
                local_readout_strategy=(
                    None
                    if getattr(args, "phase3_oracle_local_readout_strategy", None) in {None, ""}
                    else str(getattr(args, "phase3_oracle_local_readout_strategy"))
                ),
                execution_surface_requested=str(getattr(args, "phase3_oracle_execution_surface")),
                raw_transport=str(getattr(args, "phase3_oracle_raw_transport")),
                raw_store_memory=bool(getattr(args, "phase3_oracle_raw_store_memory")),
                raw_artifact_path=(
                    None
                    if getattr(args, "phase3_oracle_raw_artifact_path", None) in {None, ""}
                    else str(getattr(args, "phase3_oracle_raw_artifact_path"))
                ),
                seed_transpiler=(
                    None
                    if getattr(args, "phase3_oracle_seed_transpiler", None) is None
                    else int(getattr(args, "phase3_oracle_seed_transpiler"))
                ),
                transpile_optimization_level=int(
                    getattr(args, "phase3_oracle_transpile_optimization_level")
                ),
            )
        )
        adapt_mod._validate_phase3_oracle_gradient_config(
            config=phase3_oracle_gradient_config,
            problem="hh",
            continuation_mode=str(adapt_mode),
        )
    adapt = AdaptConfig(
        pool=(None if getattr(args, "adapt_pool", None) in {None, "", "none"} else str(getattr(args, "adapt_pool"))),
        continuation_mode=adapt_mode,
        max_depth=int(cfg_values["adapt_max_depth"]),
        maxiter=int(cfg_values["adapt_maxiter"]),
        eps_grad=float(cfg_values["adapt_eps_grad"]),
        eps_energy=float(cfg_values["adapt_eps_energy"]),
        seed=int(getattr(args, "adapt_seed")),
        inner_optimizer=str(getattr(args, "adapt_inner_optimizer")),
        allow_repeats=bool(getattr(args, "adapt_allow_repeats")),
        finite_angle_fallback=bool(getattr(args, "adapt_finite_angle_fallback")),
        finite_angle=float(getattr(args, "adapt_finite_angle")),
        finite_angle_min_improvement=float(getattr(args, "adapt_finite_angle_min_improvement")),
        disable_hh_seed=bool(getattr(args, "adapt_disable_hh_seed")),
        reopt_policy=str(getattr(args, "adapt_reopt_policy")),
        window_size=int(getattr(args, "adapt_window_size")),
        window_topk=int(getattr(args, "adapt_window_topk")),
        full_refit_every=int(getattr(args, "adapt_full_refit_every")),
        final_full_refit=bool(getattr(args, "adapt_final_full_refit")),
        paop_r=int(getattr(args, "paop_r")),
        paop_split_paulis=bool(getattr(args, "paop_split_paulis")),
        paop_prune_eps=float(getattr(args, "paop_prune_eps")),
        paop_normalization=str(getattr(args, "paop_normalization")),
        spsa_a=float(getattr(args, "adapt_spsa_a")),
        spsa_c=float(getattr(args, "adapt_spsa_c")),
        spsa_alpha=float(getattr(args, "adapt_spsa_alpha")),
        spsa_gamma=float(getattr(args, "adapt_spsa_gamma")),
        spsa_A=float(getattr(args, "adapt_spsa_A")),
        spsa_avg_last=int(getattr(args, "adapt_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "adapt_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "adapt_spsa_eval_agg")),
        spsa_callback_every=int(getattr(args, "adapt_spsa_callback_every")),
        spsa_progress_every_s=float(getattr(args, "adapt_spsa_progress_every_s")),
        phase1_lambda_F=float(getattr(args, "phase1_lambda_F")),
        phase1_lambda_compile=float(getattr(args, "phase1_lambda_compile")),
        phase1_lambda_measure=float(getattr(args, "phase1_lambda_measure")),
        phase1_lambda_leak=float(getattr(args, "phase1_lambda_leak")),
        phase1_score_z_alpha=float(getattr(args, "phase1_score_z_alpha")),
        phase1_probe_max_positions=int(getattr(args, "phase1_probe_max_positions")),
        phase1_plateau_patience=int(getattr(args, "phase1_plateau_patience")),
        phase1_trough_margin_ratio=float(getattr(args, "phase1_trough_margin_ratio")),
        phase1_prune_enabled=bool(getattr(args, "phase1_prune_enabled")),
        phase1_prune_fraction=float(getattr(args, "phase1_prune_fraction")),
        phase1_prune_max_candidates=int(getattr(args, "phase1_prune_max_candidates")),
        phase1_prune_max_regression=float(getattr(args, "phase1_prune_max_regression")),
        phase3_motif_source_json=(
            None
            if getattr(args, "phase3_motif_source_json", None) is None
            else Path(getattr(args, "phase3_motif_source_json"))
        ),
        phase3_symmetry_mitigation_mode=str(getattr(args, "phase3_symmetry_mitigation_mode")),
        phase3_enable_rescue=bool(getattr(args, "phase3_enable_rescue")),
        phase3_lifetime_cost_mode=str(getattr(args, "phase3_lifetime_cost_mode")),
        phase3_runtime_split_mode=str(getattr(args, "phase3_runtime_split_mode")),
        phase3_oracle_gradient_config=phase3_oracle_gradient_config,
    )
    replay_mode_raw = getattr(args, "replay_continuation_mode", None)
    replay_mode = adapt_mode if replay_mode_raw in {None, "", "auto"} else str(replay_mode_raw)
    provenance["replay_continuation_mode"] = (
        "workflow.replay_mode := adapt_continuation_mode"
        if replay_mode_raw in {None, "", "auto"}
        else "cli"
    )
    replay = ReplayConfig(
        generator_family="match_adapt",
        fallback_family="full_meta",
        legacy_paop_key=str(getattr(args, "legacy_paop_key")),
        replay_seed_policy=str(getattr(args, "replay_seed_policy")),
        continuation_mode=str(replay_mode),
        reps=int(cfg_values["final_reps"]),
        restarts=int(cfg_values["final_restarts"]),
        maxiter=int(cfg_values["final_maxiter"]),
        method=str(getattr(args, "final_method")),
        seed=int(getattr(args, "final_seed")),
        energy_backend=str(getattr(args, "vqe_energy_backend")),
        progress_every_s=float(getattr(args, "final_progress_every_s")),
        wallclock_cap_s=int(getattr(args, "replay_wallclock_cap_s")),
        paop_r=int(getattr(args, "paop_r")),
        paop_split_paulis=bool(getattr(args, "paop_split_paulis")),
        paop_prune_eps=float(getattr(args, "paop_prune_eps")),
        paop_normalization=str(getattr(args, "paop_normalization")),
        spsa_a=float(getattr(args, "vqe_spsa_a")),
        spsa_c=float(getattr(args, "vqe_spsa_c")),
        spsa_alpha=float(getattr(args, "vqe_spsa_alpha")),
        spsa_gamma=float(getattr(args, "vqe_spsa_gamma")),
        spsa_A=float(getattr(args, "vqe_spsa_A")),
        spsa_avg_last=int(getattr(args, "vqe_spsa_avg_last")),
        spsa_eval_repeats=int(getattr(args, "vqe_spsa_eval_repeats")),
        spsa_eval_agg=str(getattr(args, "vqe_spsa_eval_agg")),
        replay_freeze_fraction=float(getattr(args, "replay_freeze_fraction")),
        replay_unfreeze_fraction=float(getattr(args, "replay_unfreeze_fraction")),
        replay_full_fraction=float(getattr(args, "replay_full_fraction")),
        replay_qn_spsa_refresh_every=int(getattr(args, "replay_qn_spsa_refresh_every")),
        replay_qn_spsa_refresh_mode=str(getattr(args, "replay_qn_spsa_refresh_mode")),
        phase3_symmetry_mitigation_mode=str(getattr(args, "phase3_symmetry_mitigation_mode")),
    )
    dynamics = DynamicsConfig(
        methods=_parse_noiseless_methods(getattr(args, "noiseless_methods")),
        t_final=float(cfg_values["t_final"]),
        num_times=int(cfg_values["num_times"]),
        trotter_steps=int(cfg_values["trotter_steps"]),
        exact_steps_multiplier=int(cfg_values["exact_steps_multiplier"]),
        fidelity_subspace_energy_tol=float(getattr(args, "fidelity_subspace_energy_tol")),
        cfqm_stage_exp=str(getattr(args, "cfqm_stage_exp")),
        cfqm_coeff_drop_abs_tol=float(getattr(args, "cfqm_coeff_drop_abs_tol")),
        cfqm_normalize=bool(getattr(args, "cfqm_normalize")),
        enable_drive=bool(getattr(args, "enable_drive")),
        drive_A=float(getattr(args, "drive_A")),
        drive_omega=float(getattr(args, "drive_omega")),
        drive_tbar=float(getattr(args, "drive_tbar")),
        drive_phi=float(getattr(args, "drive_phi")),
        drive_pattern=str(getattr(args, "drive_pattern")),
        drive_custom_s=getattr(args, "drive_custom_s", None),
        drive_include_identity=bool(getattr(args, "drive_include_identity")),
        drive_time_sampling=str(getattr(args, "drive_time_sampling")),
        drive_t0=float(getattr(args, "drive_t0")),
    )
    artifacts = ArtifactConfig(
        tag=str(tag),
        output_json=Path(output_json),
        output_pdf=Path(output_pdf),
        handoff_json=Path(handoff_json),
        replay_output_json=Path(replay_output_json),
        replay_output_csv=Path(replay_output_csv),
        replay_output_md=Path(replay_output_md),
        replay_output_log=Path(replay_output_log),
        skip_pdf=bool(getattr(args, "skip_pdf", False)),
    )
    gates = GateConfig(
        ecut_1=float(
            _resolve_with_default(
                name="ecut_1",
                raw=getattr(args, "ecut_1", None),
                default=1e-1,
                provenance=provenance,
                default_source="run_guide.ecut_1=1e-1",
            )
        ),
        ecut_2=float(
            _resolve_with_default(
                name="ecut_2",
                raw=getattr(args, "ecut_2", None),
                default=1e-4,
                provenance=provenance,
                default_source="run_guide.ecut_2=1e-4",
            )
        ),
    )
    spectral_report = SpectralReportConfig(
        target_observable=str(getattr(args, "spectral_target_observable", "auto")),
        target_pair=_parse_target_pair(getattr(args, "spectral_target_pair", "")),
        detrend=str(getattr(args, "spectral_detrend", "constant")),
        window=str(getattr(args, "spectral_window", "hann")),
        max_harmonic=int(getattr(args, "spectral_max_harmonic", 3)),
    )
    realtime_checkpoint = RealtimeCheckpointConfig(
        mode=str(getattr(args, "checkpoint_controller_mode", "off")),
        oracle_selection_policy=str(
            getattr(
                args,
                "checkpoint_controller_oracle_selection_policy",
                "measured_gain_commit_veto",
            )
        ),
        candidate_step_scales=_parse_checkpoint_controller_step_scales(
            getattr(args, "checkpoint_controller_candidate_step_scales", "1.0")
        ),
        exact_forecast_baseline_step_refine_rounds=int(
            getattr(args, "checkpoint_controller_exact_forecast_baseline_step_refine_rounds", 0)
        ),
        exact_forecast_baseline_proposal_mode=str(
            getattr(
                args,
                "checkpoint_controller_exact_forecast_baseline_proposal_mode",
                "norm_locked_blend_v1",
            )
        ),
        exact_forecast_baseline_blend_weights=_parse_checkpoint_controller_blend_weights(
            getattr(args, "checkpoint_controller_exact_forecast_baseline_blend_weights", "")
        ),
        exact_forecast_baseline_gain_scales=_parse_checkpoint_controller_gain_scales(
            getattr(args, "checkpoint_controller_exact_forecast_baseline_gain_scales", "")
        ),
        exact_forecast_include_tangent_secant_proposal=bool(
            getattr(
                args,
                "checkpoint_controller_exact_forecast_include_tangent_secant_proposal",
                False,
            )
        ),
        exact_forecast_tangent_secant_trust_radius=float(
            getattr(
                args,
                "checkpoint_controller_exact_forecast_tangent_secant_trust_radius",
                0.0,
            )
        ),
        exact_forecast_tangent_secant_signed_energy_lead_limit=float(
            getattr(
                args,
                "checkpoint_controller_exact_forecast_tangent_secant_signed_energy_lead_limit",
                0.0,
            )
        ),
        exact_forecast_tracking_horizon_steps=int(
            getattr(args, "checkpoint_controller_exact_forecast_horizon_steps", 1)
        ),
        exact_forecast_tracking_horizon_weights=_parse_checkpoint_controller_horizon_weights(
            getattr(args, "checkpoint_controller_exact_forecast_horizon_weights", "")
        ),
        exact_forecast_energy_slope_weight=float(
            getattr(args, "checkpoint_controller_exact_forecast_energy_slope_weight", 0.0)
        ),
        exact_forecast_energy_curvature_weight=float(
            getattr(args, "checkpoint_controller_exact_forecast_energy_curvature_weight", 0.0)
        ),
        exact_forecast_energy_excursion_under_weight=float(
            getattr(args, "checkpoint_controller_exact_forecast_energy_excursion_under_weight", 0.0)
        ),
        exact_forecast_energy_excursion_over_weight=float(
            getattr(args, "checkpoint_controller_exact_forecast_energy_excursion_over_weight", 0.0)
        ),
        exact_forecast_energy_excursion_rel_tolerance=float(
            getattr(args, "checkpoint_controller_exact_forecast_energy_excursion_rel_tolerance", 0.0)
        ),
        exact_forecast_guardrail_mode=str(
            getattr(args, "checkpoint_controller_exact_forecast_guardrail_mode", "off")
        ),
        exact_forecast_fidelity_loss_tol=float(
            getattr(args, "checkpoint_controller_exact_forecast_fidelity_loss_tol", 0.0)
        ),
        exact_forecast_abs_energy_error_increase_tol=float(
            getattr(
                args,
                "checkpoint_controller_exact_forecast_abs_energy_error_increase_tol",
                0.0,
            )
        ),
        confirm_score_mode=str(
            getattr(args, "checkpoint_controller_confirm_score_mode", "compressed_whitened_v1")
        ),
        confirm_compress_fraction=float(
            getattr(args, "checkpoint_controller_confirm_compress_fraction", 0.5)
        ),
        confirm_compress_min_modes=int(
            getattr(args, "checkpoint_controller_confirm_compress_min_modes", 1)
        ),
        confirm_compress_max_modes=int(
            getattr(args, "checkpoint_controller_confirm_compress_max_modes", 8)
        ),
        prune_mode=str(getattr(args, "checkpoint_controller_prune_mode", "off")),
        prune_miss_threshold=float(
            getattr(args, "checkpoint_controller_prune_miss_threshold", 0.02)
        ),
        prune_protection_steps=int(
            getattr(args, "checkpoint_controller_prune_protection_steps", 2)
        ),
        prune_stagnation_window=int(
            getattr(args, "checkpoint_controller_prune_stagnation_window", 3)
        ),
        prune_stagnation_alpha=float(
            getattr(args, "checkpoint_controller_prune_stagnation_alpha", 0.5)
        ),
        prune_stale_score_threshold=float(
            getattr(args, "checkpoint_controller_prune_stale_score_threshold", 0.75)
        ),
        prune_loss_threshold=float(
            getattr(args, "checkpoint_controller_prune_loss_threshold", 0.01)
        ),
        prune_max_candidates=int(
            getattr(args, "checkpoint_controller_prune_max_candidates", 2)
        ),
        prune_cooldown_steps=int(
            getattr(args, "checkpoint_controller_prune_cooldown_steps", 2)
        ),
        prune_safe_miss_increase_tol=float(
            getattr(args, "checkpoint_controller_prune_safe_miss_increase_tol", 0.01)
        ),
        prune_state_jump_l2_tol=float(
            getattr(args, "checkpoint_controller_prune_state_jump_l2_tol", 0.05)
        ),
        prune_theta_block_tol=float(
            getattr(args, "checkpoint_controller_prune_theta_block_tol", 0.05)
        ),
        miss_threshold=float(getattr(args, "checkpoint_controller_miss_threshold")),
        gain_ratio_threshold=float(getattr(args, "checkpoint_controller_gain_ratio_threshold")),
        append_margin_abs=float(getattr(args, "checkpoint_controller_append_margin_abs")),
        shortlist_size=int(getattr(args, "checkpoint_controller_shortlist_size")),
        shortlist_fraction=float(getattr(args, "checkpoint_controller_shortlist_fraction")),
        active_window_size=int(getattr(args, "checkpoint_controller_active_window_size")),
        max_probe_positions=int(getattr(args, "checkpoint_controller_max_probe_positions")),
        regularization_lambda=float(getattr(args, "checkpoint_controller_regularization_lambda")),
        candidate_regularization_lambda=float(
            getattr(args, "checkpoint_controller_candidate_regularization_lambda")
        ),
        pinv_rcond=float(getattr(args, "checkpoint_controller_pinv_rcond")),
        compile_penalty_weight=float(getattr(args, "checkpoint_controller_compile_penalty_weight")),
        measurement_penalty_weight=float(
            getattr(args, "checkpoint_controller_measurement_penalty_weight")
        ),
        directional_penalty_weight=float(
            getattr(args, "checkpoint_controller_directional_penalty_weight")
        ),
        motion_calm_direction_cosine_threshold=float(
            getattr(args, "checkpoint_controller_motion_calm_direction_cosine_threshold")
        ),
        motion_calm_rate_change_ratio_threshold=float(
            getattr(args, "checkpoint_controller_motion_calm_rate_change_ratio_threshold")
        ),
        motion_direction_reversal_cosine_threshold=float(
            getattr(args, "checkpoint_controller_motion_direction_reversal_cosine_threshold")
        ),
        motion_curvature_flip_cosine_threshold=float(
            getattr(args, "checkpoint_controller_motion_curvature_flip_cosine_threshold")
        ),
        motion_acceleration_l2_threshold=float(
            getattr(args, "checkpoint_controller_motion_acceleration_l2_threshold")
        ),
        motion_kink_rate_change_ratio_threshold=float(
            getattr(args, "checkpoint_controller_motion_kink_rate_change_ratio_threshold")
        ),
        motion_calm_shortlist_scale=float(
            getattr(args, "checkpoint_controller_motion_calm_shortlist_scale")
        ),
        motion_kink_shortlist_bonus=int(
            getattr(args, "checkpoint_controller_motion_kink_shortlist_bonus")
        ),
        motion_calm_oracle_budget_scale=float(
            getattr(args, "checkpoint_controller_motion_calm_oracle_budget_scale")
        ),
        motion_kink_oracle_budget_scale=float(
            getattr(args, "checkpoint_controller_motion_kink_oracle_budget_scale")
        ),
        position_jump_tie_margin_abs=float(
            getattr(args, "checkpoint_controller_position_jump_tie_margin_abs")
        ),
        reconstruction_tol=float(getattr(args, "checkpoint_controller_reconstruction_tol")),
        grouping_mode=str(getattr(args, "checkpoint_controller_grouping_mode")),
        tiers=(
            MeasurementTierConfig(
                tier_name="scout",
                exact_mode_behavior="proxy_only",
                oracle_shots=getattr(args, "checkpoint_controller_scout_shots", None),
                oracle_repeats=getattr(args, "checkpoint_controller_scout_repeats", None),
                oracle_aggregate="mean",
            ),
            MeasurementTierConfig(
                tier_name="confirm",
                exact_mode_behavior="incremental_exact",
                oracle_shots=getattr(args, "checkpoint_controller_confirm_shots", None),
                oracle_repeats=getattr(args, "checkpoint_controller_confirm_repeats", None),
                oracle_aggregate="mean",
            ),
            MeasurementTierConfig(
                tier_name="commit",
                exact_mode_behavior="commit_exact",
                oracle_shots=getattr(args, "checkpoint_controller_commit_shots", None),
                oracle_repeats=getattr(args, "checkpoint_controller_commit_repeats", None),
                oracle_aggregate="mean",
            ),
        ),
    )
    return StagedHHConfig(
        physics=physics,
        warm_start=warm_start,
        adapt=adapt,
        replay=replay,
        dynamics=dynamics,
        artifacts=artifacts,
        gates=gates,
        spectral_report=spectral_report,
        realtime_checkpoint=realtime_checkpoint,
        smoke_test_intentionally_weak=bool(getattr(args, "smoke_test_intentionally_weak", False)),
        default_provenance=dict(provenance),
    )


def _build_hh_context(cfg: StagedHHConfig) -> tuple[Any, np.ndarray, list[str], dict[str, complex], np.ndarray]:
    physics = cfg.physics
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=int(physics.L),
        J=float(physics.t),
        U=float(physics.u),
        omega0=float(physics.omega0),
        g=float(physics.g_ep),
        n_ph_max=int(physics.n_ph_max),
        boson_encoding=str(physics.boson_encoding),
        v_t=None,
        v0=float(physics.dv),
        t_eval=None,
        include_zero_point=True,
        repr_mode="JW",
        indexing=str(physics.ordering),
        pbc=(str(physics.boundary).strip().lower() == "periodic"),
    )
    ordered_labels_exyz, coeff_map_exyz = hc_pipeline._collect_hardcoded_terms_exyz(h_poly)
    hmat = hc_pipeline._build_hamiltonian_matrix(coeff_map_exyz)
    psi_hf = hc_pipeline._normalize_state(
        np.asarray(
            hubbard_holstein_reference_state(
                dims=int(physics.L),
                num_particles=(int(physics.sector_n_up), int(physics.sector_n_dn)),
                n_ph_max=int(physics.n_ph_max),
                boson_encoding=str(physics.boson_encoding),
                indexing=str(physics.ordering),
            ),
            dtype=complex,
        ).reshape(-1)
    )
    return h_poly, np.asarray(hmat, dtype=complex), list(ordered_labels_exyz), dict(coeff_map_exyz), psi_hf


def _handoff_continuation_meta(adapt_payload: Mapping[str, Any]) -> dict[str, Any]:
    continuation = adapt_payload.get("continuation", {})
    if not isinstance(continuation, Mapping):
        continuation = {}
    reserved_keys = {
        "mode",
        "scaffold",
        "optimizer_memory",
        "selected_generator_metadata",
        "generator_split_events",
        "motif_library",
        "motif_usage",
        "symmetry_mitigation",
        "rescue_history",
        "replay_contract_hint",
    }
    excluded_keys = {"phase1_feature_rows", "phase2_shortlist_rows"}
    continuation_details = {
        str(key): value
        for key, value in continuation.items()
        if str(key) not in reserved_keys and str(key) not in excluded_keys
    }
    return {
        "continuation_mode": str(adapt_payload.get("continuation_mode", continuation.get("mode", "legacy"))),
        "continuation_scaffold": (
            dict(adapt_payload.get("scaffold_fingerprint_lite", {}))
            if isinstance(adapt_payload.get("scaffold_fingerprint_lite", {}), Mapping)
            else None
        ),
        "optimizer_memory": (
            dict(continuation.get("optimizer_memory", {}))
            if isinstance(continuation.get("optimizer_memory", {}), Mapping)
            else None
        ),
        "selected_generator_metadata": (
            [dict(x) for x in continuation.get("selected_generator_metadata", [])]
            if isinstance(continuation.get("selected_generator_metadata", []), Sequence)
            else None
        ),
        "generator_split_events": (
            [dict(x) for x in continuation.get("generator_split_events", [])]
            if isinstance(continuation.get("generator_split_events", []), Sequence)
            else None
        ),
        "motif_library": (
            dict(continuation.get("motif_library", {}))
            if isinstance(continuation.get("motif_library", {}), Mapping)
            else None
        ),
        "motif_usage": (
            dict(continuation.get("motif_usage", {}))
            if isinstance(continuation.get("motif_usage", {}), Mapping)
            else None
        ),
        "symmetry_mitigation": (
            dict(continuation.get("symmetry_mitigation", {}))
            if isinstance(continuation.get("symmetry_mitigation", {}), Mapping)
            else None
        ),
        "rescue_history": (
            [dict(x) for x in continuation.get("rescue_history", [])]
            if isinstance(continuation.get("rescue_history", []), Sequence)
            else None
        ),
        "continuation_details": (
            dict(continuation_details)
            if continuation_details
            else None
        ),
        "prune_summary": (
            dict(adapt_payload.get("prune_summary", {}))
            if isinstance(adapt_payload.get("prune_summary", {}), Mapping)
            else None
        ),
        "pre_prune_scaffold": (
            dict(adapt_payload.get("pre_prune_scaffold", {}))
            if isinstance(adapt_payload.get("pre_prune_scaffold", {}), Mapping)
            else None
        ),
    }


def _write_adapt_handoff(
    cfg: StagedHHConfig,
    adapt_payload: Mapping[str, Any],
    psi_adapt: np.ndarray,
    psi_ansatz_input: np.ndarray,
) -> None:
    exact_energy = float(adapt_payload.get("exact_gs_energy", float("nan")))
    energy = float(adapt_payload.get("energy", float("nan")))
    continuation_meta = _handoff_continuation_meta(adapt_payload)
    handoff_cfg = HandoffStateBundleConfig(
        L=int(cfg.physics.L),
        t=float(cfg.physics.t),
        U=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        sector_n_up=int(cfg.physics.sector_n_up),
        sector_n_dn=int(cfg.physics.sector_n_dn),
    )
    write_handoff_state_bundle(
        path=cfg.artifacts.handoff_json,
        psi_state=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        cfg=handoff_cfg,
        source="adapt_vqe",
        exact_energy=float(exact_energy),
        energy=float(energy),
        delta_E_abs=float(adapt_payload.get("abs_delta_e", abs(energy - exact_energy))),
        relative_error_abs=float(_relative_error_abs(energy, exact_energy)),
        meta={
            "pipeline": "hh_staged_noiseless",
            "workflow_tag": str(cfg.artifacts.tag),
            "stage_chain": ["hf_reference", "warm_start_hva", "adapt_vqe", "matched_family_replay"],
        },
        adapt_operators=[str(x) for x in adapt_payload.get("operators", [])],
        adapt_optimal_point=[float(x) for x in adapt_payload.get("optimal_point", [])],
        adapt_logical_optimal_point=[float(x) for x in adapt_payload.get("logical_optimal_point", [])],
        adapt_parameterization=(
            dict(adapt_payload.get("parameterization", {}))
            if isinstance(adapt_payload.get("parameterization", {}), Mapping)
            else None
        ),
        adapt_logical_num_parameters=(
            int(adapt_payload.get("logical_num_parameters"))
            if adapt_payload.get("logical_num_parameters") is not None
            else None
        ),
        adapt_pool_type=(None if adapt_payload.get("pool_type") is None else str(adapt_payload.get("pool_type"))),
        handoff_state_kind="prepared_state",
        continuation_mode=str(continuation_meta.get("continuation_mode", cfg.adapt.continuation_mode)),
        continuation_scaffold=continuation_meta.get("continuation_scaffold"),
        continuation_details=continuation_meta.get("continuation_details"),
        optimizer_memory=continuation_meta.get("optimizer_memory"),
        selected_generator_metadata=continuation_meta.get("selected_generator_metadata"),
        generator_split_events=continuation_meta.get("generator_split_events"),
        motif_library=continuation_meta.get("motif_library"),
        motif_usage=continuation_meta.get("motif_usage"),
        symmetry_mitigation=continuation_meta.get("symmetry_mitigation"),
        rescue_history=continuation_meta.get("rescue_history"),
        prune_summary=continuation_meta.get("prune_summary"),
        pre_prune_scaffold=continuation_meta.get("pre_prune_scaffold"),
        replay_contract_hint={
            "generator_family": str(cfg.replay.generator_family),
            "fallback_family": str(cfg.replay.fallback_family),
            "replay_seed_policy": str(cfg.replay.replay_seed_policy),
            "replay_continuation_mode": str(cfg.replay.continuation_mode),
        },
        ansatz_input_state=np.asarray(psi_ansatz_input, dtype=complex).reshape(-1),
        ansatz_input_state_source="warm_start_hva",
        ansatz_input_state_handoff_state_kind="prepared_state",
    )


def _build_replay_run_config(cfg: StagedHHConfig) -> replay_mod.RunConfig:
    return replay_mod.RunConfig(
        adapt_input_json=Path(cfg.artifacts.handoff_json),
        output_json=Path(cfg.artifacts.replay_output_json),
        output_csv=Path(cfg.artifacts.replay_output_csv),
        output_md=Path(cfg.artifacts.replay_output_md),
        output_log=Path(cfg.artifacts.replay_output_log),
        tag=f"{cfg.artifacts.tag}_replay",
        generator_family=str(cfg.replay.generator_family),
        fallback_family=str(cfg.replay.fallback_family),
        legacy_paop_key=str(cfg.replay.legacy_paop_key),
        replay_seed_policy=str(cfg.replay.replay_seed_policy),
        replay_continuation_mode=str(cfg.replay.continuation_mode),
        L=int(cfg.physics.L),
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        sector_n_up=int(cfg.physics.sector_n_up),
        sector_n_dn=int(cfg.physics.sector_n_dn),
        reps=int(cfg.replay.reps),
        restarts=int(cfg.replay.restarts),
        maxiter=int(cfg.replay.maxiter),
        method=str(cfg.replay.method),
        seed=int(cfg.replay.seed),
        energy_backend=str(cfg.replay.energy_backend),
        progress_every_s=float(cfg.replay.progress_every_s),
        wallclock_cap_s=int(cfg.replay.wallclock_cap_s),
        paop_r=int(cfg.replay.paop_r),
        paop_split_paulis=bool(cfg.replay.paop_split_paulis),
        paop_prune_eps=float(cfg.replay.paop_prune_eps),
        paop_normalization=str(cfg.replay.paop_normalization),
        spsa_a=float(cfg.replay.spsa_a),
        spsa_c=float(cfg.replay.spsa_c),
        spsa_alpha=float(cfg.replay.spsa_alpha),
        spsa_gamma=float(cfg.replay.spsa_gamma),
        spsa_A=float(cfg.replay.spsa_A),
        spsa_avg_last=int(cfg.replay.spsa_avg_last),
        spsa_eval_repeats=int(cfg.replay.spsa_eval_repeats),
        spsa_eval_agg=str(cfg.replay.spsa_eval_agg),
        replay_freeze_fraction=float(cfg.replay.replay_freeze_fraction),
        replay_unfreeze_fraction=float(cfg.replay.replay_unfreeze_fraction),
        replay_full_fraction=float(cfg.replay.replay_full_fraction),
        replay_qn_spsa_refresh_every=int(cfg.replay.replay_qn_spsa_refresh_every),
        replay_qn_spsa_refresh_mode=str(cfg.replay.replay_qn_spsa_refresh_mode),
        phase3_symmetry_mitigation_mode=str(cfg.replay.phase3_symmetry_mitigation_mode),
    )


def run_stage_pipeline(cfg: StagedHHConfig) -> StageExecutionResult:
    h_poly, hmat, ordered_labels_exyz, coeff_map_exyz, psi_hf = _build_hh_context(cfg)

    warm_payload, psi_warm = hc_pipeline._run_hardcoded_vqe(
        num_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        boundary=str(cfg.physics.boundary),
        hopping_t=float(cfg.physics.t),
        onsite_u=float(cfg.physics.u),
        potential_dv=float(cfg.physics.dv),
        h_poly=h_poly,
        reps=int(cfg.warm_start.reps),
        restarts=int(cfg.warm_start.restarts),
        seed=int(cfg.warm_start.seed),
        maxiter=int(cfg.warm_start.maxiter),
        method=str(cfg.warm_start.method),
        energy_backend=str(cfg.warm_start.energy_backend),
        vqe_progress_every_s=float(cfg.warm_start.progress_every_s),
        ansatz_name=str(cfg.warm_start.ansatz_name),
        spsa_a=float(cfg.warm_start.spsa_a),
        spsa_c=float(cfg.warm_start.spsa_c),
        spsa_alpha=float(cfg.warm_start.spsa_alpha),
        spsa_gamma=float(cfg.warm_start.spsa_gamma),
        spsa_A=float(cfg.warm_start.spsa_A),
        spsa_avg_last=int(cfg.warm_start.spsa_avg_last),
        spsa_eval_repeats=int(cfg.warm_start.spsa_eval_repeats),
        spsa_eval_agg=str(cfg.warm_start.spsa_eval_agg),
        problem="hh",
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
    )

    adapt_payload, psi_adapt = adapt_mod._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        problem="hh",
        adapt_pool=cfg.adapt.pool,
        t=float(cfg.physics.t),
        u=float(cfg.physics.u),
        dv=float(cfg.physics.dv),
        boundary=str(cfg.physics.boundary),
        omega0=float(cfg.physics.omega0),
        g_ep=float(cfg.physics.g_ep),
        n_ph_max=int(cfg.physics.n_ph_max),
        boson_encoding=str(cfg.physics.boson_encoding),
        max_depth=int(cfg.adapt.max_depth),
        eps_grad=float(cfg.adapt.eps_grad),
        eps_energy=float(cfg.adapt.eps_energy),
        maxiter=int(cfg.adapt.maxiter),
        seed=int(cfg.adapt.seed),
        adapt_inner_optimizer=str(cfg.adapt.inner_optimizer),
        adapt_spsa_a=float(cfg.adapt.spsa_a),
        adapt_spsa_c=float(cfg.adapt.spsa_c),
        adapt_spsa_alpha=float(cfg.adapt.spsa_alpha),
        adapt_spsa_gamma=float(cfg.adapt.spsa_gamma),
        adapt_spsa_A=float(cfg.adapt.spsa_A),
        adapt_spsa_avg_last=int(cfg.adapt.spsa_avg_last),
        adapt_spsa_eval_repeats=int(cfg.adapt.spsa_eval_repeats),
        adapt_spsa_eval_agg=str(cfg.adapt.spsa_eval_agg),
        adapt_spsa_callback_every=int(cfg.adapt.spsa_callback_every),
        adapt_spsa_progress_every_s=float(cfg.adapt.spsa_progress_every_s),
        allow_repeats=bool(cfg.adapt.allow_repeats),
        finite_angle_fallback=bool(cfg.adapt.finite_angle_fallback),
        finite_angle=float(cfg.adapt.finite_angle),
        finite_angle_min_improvement=float(cfg.adapt.finite_angle_min_improvement),
        paop_r=int(cfg.adapt.paop_r),
        paop_split_paulis=bool(cfg.adapt.paop_split_paulis),
        paop_prune_eps=float(cfg.adapt.paop_prune_eps),
        paop_normalization=str(cfg.adapt.paop_normalization),
        disable_hh_seed=bool(cfg.adapt.disable_hh_seed),
        psi_ref_override=np.asarray(psi_warm, dtype=complex).reshape(-1),
        adapt_reopt_policy=str(cfg.adapt.reopt_policy),
        adapt_window_size=int(cfg.adapt.window_size),
        adapt_window_topk=int(cfg.adapt.window_topk),
        adapt_full_refit_every=int(cfg.adapt.full_refit_every),
        adapt_final_full_refit=bool(cfg.adapt.final_full_refit),
        adapt_continuation_mode=str(cfg.adapt.continuation_mode),
        phase1_lambda_F=float(cfg.adapt.phase1_lambda_F),
        phase1_lambda_compile=float(cfg.adapt.phase1_lambda_compile),
        phase1_lambda_measure=float(cfg.adapt.phase1_lambda_measure),
        phase1_lambda_leak=float(cfg.adapt.phase1_lambda_leak),
        phase1_score_z_alpha=float(cfg.adapt.phase1_score_z_alpha),
        phase1_probe_max_positions=int(cfg.adapt.phase1_probe_max_positions),
        phase1_plateau_patience=int(cfg.adapt.phase1_plateau_patience),
        phase1_trough_margin_ratio=float(cfg.adapt.phase1_trough_margin_ratio),
        phase1_prune_enabled=bool(cfg.adapt.phase1_prune_enabled),
        phase1_prune_fraction=float(cfg.adapt.phase1_prune_fraction),
        phase1_prune_max_candidates=int(cfg.adapt.phase1_prune_max_candidates),
        phase1_prune_max_regression=float(cfg.adapt.phase1_prune_max_regression),
        phase3_motif_source_json=cfg.adapt.phase3_motif_source_json,
        phase3_symmetry_mitigation_mode=str(cfg.adapt.phase3_symmetry_mitigation_mode),
        phase3_enable_rescue=bool(cfg.adapt.phase3_enable_rescue),
        phase3_lifetime_cost_mode=str(cfg.adapt.phase3_lifetime_cost_mode),
        phase3_runtime_split_mode=str(cfg.adapt.phase3_runtime_split_mode),
        phase3_oracle_gradient_config=cfg.adapt.phase3_oracle_gradient_config,
    )

    _write_adapt_handoff(
        cfg,
        adapt_payload,
        np.asarray(psi_adapt, dtype=complex).reshape(-1),
        np.asarray(psi_warm, dtype=complex).reshape(-1),
    )
    replay_cfg = _build_replay_run_config(cfg)
    replay_payload = replay_mod.run(replay_cfg)
    nq_total = _hh_nq_total(cfg.physics.L, cfg.physics.n_ph_max, cfg.physics.boson_encoding)
    best_state = replay_payload.get("best_state", {})
    if not isinstance(best_state, Mapping):
        raise ValueError("Replay payload missing best_state block.")
    amplitudes = best_state.get("amplitudes_qn_to_q0", None)
    if not isinstance(amplitudes, Mapping):
        raise ValueError("Replay payload missing best_state.amplitudes_qn_to_q0.")
    psi_final = hc_pipeline._state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    psi_final = hc_pipeline._normalize_state(np.asarray(psi_final, dtype=complex).reshape(-1))

    return StageExecutionResult(
        h_poly=h_poly,
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels_exyz),
        coeff_map_exyz=dict(coeff_map_exyz),
        nq_total=int(nq_total),
        psi_hf=np.asarray(psi_hf, dtype=complex).reshape(-1),
        psi_warm=np.asarray(psi_warm, dtype=complex).reshape(-1),
        psi_adapt=np.asarray(psi_adapt, dtype=complex).reshape(-1),
        psi_final=np.asarray(psi_final, dtype=complex).reshape(-1),
        warm_payload=dict(warm_payload),
        adapt_payload=dict(adapt_payload),
        replay_payload=dict(replay_payload),
    )


def _build_drive_provider(
    *,
    cfg: StagedHHConfig,
    nq_total: int,
    ordered_labels_exyz: Sequence[str],
) -> tuple[Any | None, dict[str, Any] | None, list[str], dict[str, Any] | None]:
    if not bool(cfg.dynamics.enable_drive):
        return None, None, list(ordered_labels_exyz), None
    custom_weights = None
    if str(cfg.dynamics.drive_pattern) == "custom":
        custom_weights = _parse_drive_custom_weights(cfg.dynamics.drive_custom_s)
        if custom_weights is None:
            raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
    drive = build_gaussian_sinusoid_density_drive(
        n_sites=int(cfg.physics.L),
        nq_total=int(nq_total),
        indexing=str(cfg.physics.ordering),
        A=float(cfg.dynamics.drive_A),
        omega=float(cfg.dynamics.drive_omega),
        tbar=float(cfg.dynamics.drive_tbar),
        phi=float(cfg.dynamics.drive_phi),
        pattern_mode=str(cfg.dynamics.drive_pattern),
        custom_weights=custom_weights,
        include_identity=bool(cfg.dynamics.drive_include_identity),
        coeff_tol=0.0,
    )
    drive_labels = set(drive.template.labels_exyz(include_identity=bool(drive.include_identity)))
    ordered = list(ordered_labels_exyz)
    missing = sorted(drive_labels.difference(ordered))
    ordered.extend(missing)
    profile = {
        "A": float(cfg.dynamics.drive_A),
        "omega": float(cfg.dynamics.drive_omega),
        "tbar": float(cfg.dynamics.drive_tbar),
        "phi": float(cfg.dynamics.drive_phi),
        "pattern": str(cfg.dynamics.drive_pattern),
        "custom_weights": custom_weights,
        "include_identity": bool(cfg.dynamics.drive_include_identity),
        "time_sampling": str(cfg.dynamics.drive_time_sampling),
        "t0": float(cfg.dynamics.drive_t0),
    }
    meta = {
        "reference_method": str(reference_method_name(str(cfg.dynamics.drive_time_sampling))),
        "missing_drive_labels_added": int(len(missing)),
        "drive_label_count": int(len(drive_labels)),
    }
    return drive.coeff_map_exyz, meta, ordered, profile


def _controller_drive_config_from_cfg(cfg: StagedHHConfig) -> ControllerDriveConfig | None:
    if not bool(cfg.dynamics.enable_drive):
        return None
    custom_weights = None
    if str(cfg.dynamics.drive_pattern) == "custom":
        custom_weights = _parse_drive_custom_weights(cfg.dynamics.drive_custom_s)
        if custom_weights is None:
            raise ValueError("--drive-custom-s is required when --drive-pattern custom.")
    return ControllerDriveConfig(
        enabled=True,
        n_sites=int(cfg.physics.L),
        ordering=str(cfg.physics.ordering),
        drive_A=float(cfg.dynamics.drive_A),
        drive_omega=float(cfg.dynamics.drive_omega),
        drive_tbar=float(cfg.dynamics.drive_tbar),
        drive_phi=float(cfg.dynamics.drive_phi),
        drive_pattern=str(cfg.dynamics.drive_pattern),
        drive_custom_weights=(
            None if custom_weights is None else tuple(float(x) for x in custom_weights)
        ),
        drive_include_identity=bool(cfg.dynamics.drive_include_identity),
        drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
        drive_t0=float(cfg.dynamics.drive_t0),
        exact_steps_multiplier=int(cfg.dynamics.exact_steps_multiplier),
    )


def _run_noiseless_profile(
    *,
    cfg: StagedHHConfig,
    psi_seed: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    drive_enabled: bool,
    ground_state_reference_energy: float,
) -> dict[str, Any]:
    drive_provider = None
    drive_meta = None
    drive_profile = None
    ordered_for_run = list(ordered_labels_exyz)
    if drive_enabled:
        drive_provider, drive_meta, ordered_for_run, drive_profile = _build_drive_provider(
            cfg=cfg,
            nq_total=int(round(math.log2(int(np.asarray(psi_seed).size)))),
            ordered_labels_exyz=ordered_labels_exyz,
        )

    method_payloads: dict[str, Any] = {}
    reference_rows: list[dict[str, Any]] | None = None
    psi_seed_arr = np.asarray(psi_seed, dtype=complex).reshape(-1)
    ground_state_energy = float(ground_state_reference_energy)

    for method in cfg.dynamics.methods:
        rows, _ = hc_pipeline._simulate_trajectory(
            num_sites=int(cfg.physics.L),
            ordering=str(cfg.physics.ordering),
            psi0_legacy_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_paop_trot=np.asarray(psi_seed_arr, dtype=complex),
            psi0_hva_trot=np.asarray(psi_seed_arr, dtype=complex),
            legacy_branch_label="replay",
            psi0_exact_ref=np.asarray(psi_seed_arr, dtype=complex),
            fidelity_subspace_basis_v0=np.asarray(psi_seed_arr, dtype=complex).reshape(-1, 1),
            fidelity_subspace_energy_tol=float(cfg.dynamics.fidelity_subspace_energy_tol),
            hmat=np.asarray(hmat, dtype=complex),
            ordered_labels_exyz=list(ordered_for_run),
            coeff_map_exyz=dict(coeff_map_exyz),
            trotter_steps=int(cfg.dynamics.trotter_steps),
            t_final=float(cfg.dynamics.t_final),
            num_times=int(cfg.dynamics.num_times),
            suzuki_order=2,
            drive_coeff_provider_exyz=drive_provider,
            drive_t0=float(cfg.dynamics.drive_t0 if drive_enabled else 0.0),
            drive_time_sampling=str(cfg.dynamics.drive_time_sampling),
            exact_steps_multiplier=(int(cfg.dynamics.exact_steps_multiplier) if drive_enabled else 1),
            propagator=str(method),
            cfqm_stage_exp=str(cfg.dynamics.cfqm_stage_exp),
            cfqm_coeff_drop_abs_tol=float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
            cfqm_normalize=bool(cfg.dynamics.cfqm_normalize),
        )
        rows_with_metrics: list[dict[str, Any]] = []
        for row in rows:
            row_out = dict(row)
            row_out["abs_energy_error_vs_ground_state"] = float(
                abs(float(row_out["energy_total_trotter"]) - ground_state_energy)
            )
            rows_with_metrics.append(row_out)
        reference_rows = rows_with_metrics if reference_rows is None else reference_rows
        final_row = rows_with_metrics[-1]
        final_reference_error = float(
            abs(float(final_row["energy_total_trotter"]) - float(final_row["energy_total_exact"]))
        )
        method_payloads[str(method)] = {
            "propagator": str(method),
            "trajectory": rows_with_metrics,
            "final": {
                "energy_total_trotter": float(final_row["energy_total_trotter"]),
                "energy_total_exact": float(final_row["energy_total_exact"]),
                "abs_energy_total_error": float(final_reference_error),
                "abs_energy_total_error_vs_reference": float(final_reference_error),
                "abs_energy_error_vs_ground_state": float(final_row["abs_energy_error_vs_ground_state"]),
                "fidelity": float(final_row["fidelity"]),
                "doublon_trotter": float(final_row["doublon_trotter"]),
                "doublon_exact": float(final_row["doublon_exact"]),
            },
            "settings": {
                "trotter_steps": int(cfg.dynamics.trotter_steps),
                "num_times": int(cfg.dynamics.num_times),
                "t_final": float(cfg.dynamics.t_final),
                "cfqm_stage_exp": str(cfg.dynamics.cfqm_stage_exp),
                "cfqm_coeff_drop_abs_tol": float(cfg.dynamics.cfqm_coeff_drop_abs_tol),
                "cfqm_normalize": bool(cfg.dynamics.cfqm_normalize),
            },
        }

    assert reference_rows is not None
    return {
        "drive_enabled": bool(drive_enabled),
        "drive_profile": drive_profile,
        "drive_meta": drive_meta,
        "times": [float(row["time"]) for row in reference_rows],
        "ground_state_reference": {
            "energy": float(ground_state_energy),
            "kind": "filtered_sector_ground_state_static",
            "source": "stage_pipeline.conventional_replay.exact_energy",
        },
        "reference": {
            "kind": "seeded_exact_reference",
            "initial_state": "psi_final",
            "method": (
                "eigendecomposition"
                if not drive_enabled
                else str(reference_method_name(str(cfg.dynamics.drive_time_sampling)))
            ),
            "energy_total_exact": [float(row["energy_total_exact"]) for row in reference_rows],
            "doublon_exact": [float(row["doublon_exact"]) for row in reference_rows],
        },
        "methods": method_payloads,
    }


def run_noiseless_profiles(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
    replay_exact = float(stage_result.replay_payload.get("exact", {}).get("E_exact_sector", float("nan")))
    profiles = {
        "static": _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=False,
            ground_state_reference_energy=replay_exact,
        )
    }
    if bool(cfg.dynamics.enable_drive):
        profiles["drive"] = _run_noiseless_profile(
            cfg=cfg,
            psi_seed=stage_result.psi_final,
            hmat=stage_result.hmat,
            ordered_labels_exyz=stage_result.ordered_labels_exyz,
            coeff_map_exyz=stage_result.coeff_map_exyz,
            drive_enabled=True,
            ground_state_reference_energy=replay_exact,
        )
    return {"profiles": profiles}


def _prepare_adaptive_realtime_checkpoint_inputs(
    stage_result: StageExecutionResult,
    cfg: StagedHHConfig,
) -> tuple[Any, Any, Sequence[float]] | None:
    mode = str(cfg.realtime_checkpoint.mode)
    if mode == "off":
        return None
    replay_cfg = _build_replay_run_config(cfg)
    replay_context = replay_mod.build_replay_scaffold_context(
        replay_cfg,
        h_poly=stage_result.h_poly,
    )
    acceptance = validate_scaffold_acceptance(replay_context.payload_in)
    if not bool(acceptance.accepted):
        raise ValueError(
            f"checkpoint controller rejected scaffold ownership: {acceptance.reason} ({acceptance.source_kind})."
        )
    best_state = stage_result.replay_payload.get("best_state", {})
    if not isinstance(best_state, Mapping):
        raise ValueError("Replay payload missing best_state block.")
    best_theta = best_state.get("best_theta", None)
    if not isinstance(best_theta, Sequence):
        raise ValueError(
            "Replay payload missing best_state.best_theta; checkpoint controller exact_v1 requires replay runtime parameters."
        )
    return replay_context, acceptance, best_theta


def run_adaptive_realtime_checkpoint_profile(
    stage_result: StageExecutionResult,
    cfg: StagedHHConfig,
) -> dict[str, Any] | None:
    if str(cfg.realtime_checkpoint.mode) == "oracle_v1":
        raise ValueError(
            "checkpoint controller oracle_v1 is only available through the staged noise workflow; use pipelines/hardcoded/hh_staged_noise.py."
        )
    prepared = _prepare_adaptive_realtime_checkpoint_inputs(stage_result, cfg)
    if prepared is None:
        return None
    replay_context, acceptance, best_theta = prepared
    controller = RealtimeCheckpointController(
        cfg=cfg.realtime_checkpoint,
        replay_context=replay_context,
        h_poly=stage_result.h_poly,
        hmat=stage_result.hmat,
        psi_initial=stage_result.psi_final,
        best_theta=best_theta,
        allow_repeats=bool(cfg.adapt.allow_repeats),
        t_final=float(cfg.dynamics.t_final),
        num_times=int(cfg.dynamics.num_times),
        drive_config=_controller_drive_config_from_cfg(cfg),
    )
    artifacts = controller.run()
    return {
        "mode": str(cfg.realtime_checkpoint.mode),
        "status": "completed",
        "scaffold_acceptance": dataclass_to_payload(acceptance),
        "reference": dict(artifacts.reference),
        "trajectory": list(artifacts.trajectory),
        "ledger": list(artifacts.ledger),
        "summary": dict(artifacts.summary),
    }


def _rows_numeric_series(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    values: list[float] = []
    for row in rows:
        if key not in row:
            return None
        values.append(float(row[key]))
    return np.asarray(values, dtype=float)


def _rows_site_matrix(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    out: list[list[float]] = []
    width: int | None = None
    for row in rows:
        raw = row.get(key)
        if not isinstance(raw, list):
            return None
        if width is None:
            width = len(raw)
        if len(raw) != width:
            return None
        out.append([float(x) for x in raw])
    return np.asarray(out, dtype=float)


def _resolve_spectral_target(
    site_occupations: np.ndarray,
    cfg: StagedHHConfig,
) -> tuple[str, str, tuple[int, int] | None]:
    explicit = str(cfg.spectral_report.target_observable).strip().lower()
    num_sites = int(site_occupations.shape[1])
    if explicit == "density_difference":
        pair = cfg.spectral_report.target_pair
        if pair is None:
            if num_sites != 2:
                raise ValueError(
                    "spectral_target_observable=density_difference requires --spectral-target-pair outside L=2."
                )
            pair = (0, 1)
        return "density_difference", f"d(t) = n_{pair[0]}(t) - n_{pair[1]}(t)", pair
    if explicit == "staggered":
        return "staggered", "m(t) = (1/L) sum_j (-1)^j n_j(t)", None
    if num_sites == 2:
        pair = cfg.spectral_report.target_pair if cfg.spectral_report.target_pair is not None else (0, 1)
        return "density_difference", f"d(t) = n_{pair[0]}(t) - n_{pair[1]}(t)", pair
    return "staggered", "m(t) = (1/L) sum_j (-1)^j n_j(t)", None


def _trace_for_spectral_target(
    site_occupations: np.ndarray,
    *,
    target_kind: str,
    target_pair: tuple[int, int] | None,
) -> np.ndarray:
    if str(target_kind) == "density_difference":
        if target_pair is None:
            raise ValueError("density_difference target requires a concrete site pair.")
        return np.asarray(build_pair_difference_signal(site_occupations, pair=target_pair), dtype=float)
    return np.asarray(build_staggered_signal(site_occupations), dtype=float)


def _harmonic_entry(
    entries: Sequence[Mapping[str, Any]],
    harmonic: int,
) -> dict[str, Any] | None:
    target = int(harmonic)
    for entry in entries:
        if int(round(float(entry.get("harmonic", -1)))) == target:
            return dict(entry)
    return None


def _safe_rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def _build_adaptive_realtime_spectral_trust(
    adaptive_rt: Mapping[str, Any],
    cfg: StagedHHConfig,
) -> dict[str, Any] | None:
    rows = adaptive_rt.get("trajectory")
    if not isinstance(rows, list) or not rows:
        return None
    if not all(isinstance(row, Mapping) for row in rows):
        return None
    mapped_rows = list(rows)
    time_key = "time" if "time" in mapped_rows[0] else "physical_time" if "physical_time" in mapped_rows[0] else None
    if time_key is None:
        return None
    times = _rows_numeric_series(mapped_rows, time_key)
    site_occ = _rows_site_matrix(mapped_rows, "site_occupations")
    site_occ_exact = _rows_site_matrix(mapped_rows, "site_occupations_exact")
    if times is None or site_occ is None:
        return None

    target_kind, display_label, target_pair = _resolve_spectral_target(site_occ, cfg)
    target_trace = _trace_for_spectral_target(
        site_occ,
        target_kind=target_kind,
        target_pair=target_pair,
    )
    target_exact = None
    if site_occ_exact is not None:
        target_exact = _trace_for_spectral_target(
            site_occ_exact,
            target_kind=target_kind,
            target_pair=target_pair,
        )

    drive_omega = float(cfg.dynamics.drive_omega) if bool(cfg.dynamics.enable_drive) else None
    controller_spec = compute_one_sided_amplitude_spectrum(
        times,
        target_trace,
        detrend=str(cfg.spectral_report.detrend),
        window=str(cfg.spectral_report.window),
        max_peaks=5,
        drive_omega=drive_omega,
        max_harmonic=int(cfg.spectral_report.max_harmonic),
    )
    exact_spec = None
    if target_exact is not None:
        exact_spec = compute_one_sided_amplitude_spectrum(
            times,
            target_exact,
            detrend=str(cfg.spectral_report.detrend),
            window=str(cfg.spectral_report.window),
            max_peaks=5,
            drive_omega=drive_omega,
            max_harmonic=int(cfg.spectral_report.max_harmonic),
        )

    target_error = None if target_exact is None else np.asarray(target_trace - target_exact, dtype=float)
    epsilon_osc = None
    mean_abs_error = None
    rms_error = None
    span_exact = None
    per_site_mae = None
    per_site_rms = None
    if target_exact is not None:
        ctrl_fluct = np.asarray(target_trace - np.mean(target_trace), dtype=float)
        exact_fluct = np.asarray(target_exact - np.mean(target_exact), dtype=float)
        epsilon_osc = float(
            np.linalg.norm(ctrl_fluct - exact_fluct) / max(float(np.linalg.norm(exact_fluct)), 1.0e-15)
        )
        mean_abs_error = float(np.mean(np.abs(target_error)))
        rms_error = _safe_rms(target_error)
        span_exact = float(np.max(target_exact) - np.min(target_exact))
        site_error = np.asarray(site_occ - site_occ_exact, dtype=float)
        per_site_mae = [float(x) for x in np.mean(np.abs(site_error), axis=0).tolist()]
        per_site_rms = [float(x) for x in np.sqrt(np.mean(np.square(site_error), axis=0)).tolist()]

    return {
        "target_observable": str(target_kind),
        "display_label": str(display_label),
        "target_pair": None if target_pair is None else [int(target_pair[0]), int(target_pair[1])],
        "time_key": str(time_key),
        "window": str(cfg.spectral_report.window),
        "detrend": str(cfg.spectral_report.detrend),
        "max_harmonic": int(cfg.spectral_report.max_harmonic),
        "times": [float(x) for x in times.tolist()],
        "target_trace": [float(x) for x in target_trace.tolist()],
        "target_trace_exact": None if target_exact is None else [float(x) for x in target_exact.tolist()],
        "target_error": None if target_error is None else [float(x) for x in target_error.tolist()],
        "site_occupations": [[float(x) for x in row] for row in site_occ.tolist()],
        "site_occupations_exact": None
        if site_occ_exact is None
        else [[float(x) for x in row] for row in site_occ_exact.tolist()],
        "oscillation_span_controller": float(np.max(target_trace) - np.min(target_trace)),
        "oscillation_span_exact": span_exact,
        "mean_abs_error": mean_abs_error,
        "rms_error": rms_error,
        "epsilon_osc": epsilon_osc,
        "per_site_mae": per_site_mae,
        "per_site_rms_error": per_site_rms,
        "drive_line_controller": _harmonic_entry(controller_spec.harmonic_fit, 1),
        "drive_line_exact": None if exact_spec is None else _harmonic_entry(exact_spec.harmonic_fit, 1),
        "harmonics_controller": [dict(entry) for entry in controller_spec.harmonic_fit],
        "harmonics_exact": None if exact_spec is None else [dict(entry) for entry in exact_spec.harmonic_fit],
        "spectrum_omega": [float(x) for x in controller_spec.omega.tolist()],
        "spectrum_amplitude_controller": [float(x) for x in controller_spec.amplitude.tolist()],
        "spectrum_amplitude_exact": None if exact_spec is None else [float(x) for x in exact_spec.amplitude.tolist()],
        "top_peaks_controller": list(controller_spec.top_peaks),
        "top_peaks_exact": None if exact_spec is None else list(exact_spec.top_peaks),
    }


def _stage_delta(payload: Mapping[str, Any], *, energy_key: str, exact_key: str) -> float:
    return float(abs(float(payload.get(energy_key, float("nan"))) - float(payload.get(exact_key, float("nan")))) )


def _stage_summary(stage_result: StageExecutionResult, cfg: StagedHHConfig) -> dict[str, Any]:
    warm_energy = float(stage_result.warm_payload.get("energy", float("nan")))
    warm_exact = float(stage_result.warm_payload.get("exact_filtered_energy", float("nan")))
    adapt_energy = float(stage_result.adapt_payload.get("energy", float("nan")))
    adapt_exact = float(stage_result.adapt_payload.get("exact_gs_energy", float("nan")))
    replay_vqe = stage_result.replay_payload.get("vqe", {})
    replay_exact = stage_result.replay_payload.get("exact", {})
    adapt_measurement = stage_result.adapt_payload.get("measurement_cache_summary", {})
    adapt_compile = stage_result.adapt_payload.get("compile_cost_proxy_summary", {})
    adapt_continuation = stage_result.adapt_payload.get("continuation", {})
    final_energy = float(replay_vqe.get("energy", float("nan")))
    final_exact = float(replay_exact.get("E_exact_sector", float("nan")))
    warm_delta = float(abs(warm_energy - warm_exact))
    adapt_delta = float(abs(adapt_energy - adapt_exact))
    final_delta = float(abs(final_energy - final_exact))
    return {
        "hf_reference": {
            "state_kind": "reference_state",
            "nq_total": int(stage_result.nq_total),
            "sector_n_up": int(cfg.physics.sector_n_up),
            "sector_n_dn": int(cfg.physics.sector_n_dn),
        },
        "warm_start": {
            "ansatz": str(stage_result.warm_payload.get("ansatz", cfg.warm_start.ansatz_name)),
            "energy": float(warm_energy),
            "exact_energy": float(warm_exact),
            "delta_abs": float(warm_delta),
            "ecut_1": {"threshold": float(cfg.gates.ecut_1), "pass": bool(warm_delta <= float(cfg.gates.ecut_1))},
            "optimizer_method": str(stage_result.warm_payload.get("optimizer_method", cfg.warm_start.method)),
            "reps": int(cfg.warm_start.reps),
            "restarts": int(cfg.warm_start.restarts),
            "maxiter": int(cfg.warm_start.maxiter),
            "message": str(stage_result.warm_payload.get("message", "")),
        },
        "adapt_vqe": {
            "energy": float(adapt_energy),
            "exact_energy": float(adapt_exact),
            "delta_abs": float(adapt_delta),
            "depth": int(stage_result.adapt_payload.get("ansatz_depth", 0)),
            "num_parameters": int(
                stage_result.adapt_payload.get(
                    "num_parameters",
                    stage_result.adapt_payload.get("ansatz_depth", 0),
                )
            ),
            "logical_num_parameters": int(
                stage_result.adapt_payload.get(
                    "logical_num_parameters",
                    stage_result.adapt_payload.get("ansatz_depth", 0),
                )
            ),
            "pool_type": str(stage_result.adapt_payload.get("pool_type", cfg.adapt.pool or cfg.adapt.continuation_mode)),
            "continuation_mode": str(stage_result.adapt_payload.get("continuation_mode", cfg.adapt.continuation_mode)),
            "stop_reason": str(stage_result.adapt_payload.get("stop_reason", "")),
            "measurement_cache_summary": (
                dict(adapt_measurement) if isinstance(adapt_measurement, Mapping) else None
            ),
            "compile_cost_proxy_summary": (
                dict(adapt_compile) if isinstance(adapt_compile, Mapping) else None
            ),
            "gradient_uncertainty_source": (
                str(adapt_continuation.get("gradient_uncertainty_source", "zero_default"))
                if isinstance(adapt_continuation, Mapping)
                else "zero_default"
            ),
            "oracle_gradient_scope": (
                str(adapt_continuation.get("oracle_gradient_scope", "off"))
                if isinstance(adapt_continuation, Mapping)
                else "off"
            ),
            "oracle_gradient_config": (
                dict(adapt_continuation.get("oracle_gradient_config", {}))
                if isinstance(adapt_continuation, Mapping)
                and isinstance(adapt_continuation.get("oracle_gradient_config"), Mapping)
                else None
            ),
            "oracle_execution_surface": (
                str(adapt_continuation.get("oracle_execution_surface", "off"))
                if isinstance(adapt_continuation, Mapping)
                else "off"
            ),
            "oracle_backend_info": (
                dict(adapt_continuation.get("oracle_backend_info", {}))
                if isinstance(adapt_continuation, Mapping)
                and isinstance(adapt_continuation.get("oracle_backend_info"), Mapping)
                else None
            ),
            "oracle_raw_transport": (
                None
                if not isinstance(adapt_continuation, Mapping)
                or adapt_continuation.get("oracle_raw_transport") in {None, ""}
                else str(adapt_continuation.get("oracle_raw_transport"))
            ),
            "oracle_gradient_raw_records_total": (
                int(adapt_continuation.get("oracle_gradient_raw_records_total", 0))
                if isinstance(adapt_continuation, Mapping)
                else 0
            ),
            "oracle_gradient_raw_artifact_path": (
                None
                if not isinstance(adapt_continuation, Mapping)
                or adapt_continuation.get("oracle_gradient_raw_artifact_path") in {None, ""}
                else str(adapt_continuation.get("oracle_gradient_raw_artifact_path"))
            ),
            "reoptimization_backend": (
                str(adapt_continuation.get("reoptimization_backend", "exact_statevector"))
                if isinstance(adapt_continuation, Mapping)
                else "exact_statevector"
            ),
            "runtime_split_summary": (
                dict(adapt_continuation.get("runtime_split_summary", {}))
                if isinstance(adapt_continuation, Mapping)
                else None
            ),
            "handoff_json": str(cfg.artifacts.handoff_json),
        },
        "conventional_replay": {
            "energy": float(final_energy),
            "exact_energy": float(final_exact),
            "delta_abs": float(final_delta),
            "ecut_2": {"threshold": float(cfg.gates.ecut_2), "pass": bool(final_delta <= float(cfg.gates.ecut_2))},
            "generator_family": dict(stage_result.replay_payload.get("generator_family", {})),
            "seed_baseline": dict(stage_result.replay_payload.get("seed_baseline", {})),
            "stop_reason": str(replay_vqe.get("stop_reason", replay_vqe.get("message", ""))),
            "replay_continuation_mode": str(stage_result.replay_payload.get("replay_contract", {}).get("continuation_mode", cfg.replay.continuation_mode)),
            "replay_output_json": str(cfg.artifacts.replay_output_json),
        },
    }


def _compute_comparisons(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "noiseless_vs_ground_state": {},
        "noiseless_vs_reference": {},
        "stage_gates": {},
    }
    stage_pipeline = payload.get("stage_pipeline", {})
    if isinstance(stage_pipeline, Mapping):
        warm = stage_pipeline.get("warm_start", {})
        final = stage_pipeline.get("conventional_replay", {})
        if isinstance(warm, Mapping):
            out["stage_gates"]["ecut_1"] = dict(warm.get("ecut_1", {}))
        if isinstance(final, Mapping):
            out["stage_gates"]["ecut_2"] = dict(final.get("ecut_2", {}))

    dynamics = payload.get("dynamics_noiseless", {})
    if isinstance(dynamics, Mapping):
        for profile_name, profile_payload in dynamics.get("profiles", {}).items():
            if not isinstance(profile_payload, Mapping):
                continue
            ground_state_cmp: dict[str, Any] = {}
            reference_cmp: dict[str, Any] = {}
            ground_state_ref = profile_payload.get("ground_state_reference", {})
            ground_state_energy = (
                float(ground_state_ref.get("energy", float("nan")))
                if isinstance(ground_state_ref, Mapping)
                else float("nan")
            )
            for method_name, method_payload in profile_payload.get("methods", {}).items():
                if not isinstance(method_payload, Mapping):
                    continue
                final = method_payload.get("final", {})
                ground_state_cmp[str(method_name)] = {
                    "ground_state_reference_energy": float(ground_state_energy),
                    "final_abs_energy_error": float(final.get("abs_energy_error_vs_ground_state", float("nan"))),
                }
                reference_cmp[str(method_name)] = {
                    "final_abs_energy_total_error": float(
                        final.get(
                            "abs_energy_total_error_vs_reference",
                            final.get("abs_energy_total_error", float("nan")),
                        )
                    ),
                    "final_fidelity": float(final.get("fidelity", float("nan"))),
                }
            out["noiseless_vs_ground_state"][str(profile_name)] = ground_state_cmp
            out["noiseless_vs_reference"][str(profile_name)] = reference_cmp
    return out


def _payload_artifacts(cfg: StagedHHConfig) -> dict[str, Any]:
    return {
        "workflow": {
            "output_json": str(cfg.artifacts.output_json),
            "output_pdf": str(cfg.artifacts.output_pdf),
        },
        "intermediate": {
            "adapt_handoff_json": str(cfg.artifacts.handoff_json),
            "replay_output_json": str(cfg.artifacts.replay_output_json),
            "replay_output_csv": str(cfg.artifacts.replay_output_csv),
            "replay_output_md": str(cfg.artifacts.replay_output_md),
            "replay_output_log": str(cfg.artifacts.replay_output_log),
        },
    }


def assemble_payload(
    *,
    cfg: StagedHHConfig,
    stage_result: StageExecutionResult,
    dynamics_noiseless: Mapping[str, Any],
    adaptive_realtime_checkpoint: Mapping[str, Any] | None = None,
    run_command: str,
) -> dict[str, Any]:
    adaptive_payload = None
    if adaptive_realtime_checkpoint is not None:
        adaptive_payload = dict(adaptive_realtime_checkpoint)
        spectral_trust = _build_adaptive_realtime_spectral_trust(adaptive_payload, cfg)
        if spectral_trust is not None:
            adaptive_payload["spectral_trust"] = spectral_trust
    payload = {
        "generated_utc": _now_utc(),
        "pipeline": "hh_staged_noiseless",
        "workflow_contract": {
            "stage_chain": [
                "hf_reference",
                "warm_start_hva",
                "adapt_vqe",
                "matched_family_replay",
                "final_only_noiseless_dynamics",
            ],
            "conventional_vqe_definition": "non-ADAPT matched-family replay from ADAPT handoff",
            "drive_default": "opt_in",
            "noiseless_energy_metric": "|E_method(t) - E_exact_sector_replay| with replay exact sector energy as baseline",
            "noiseless_fidelity_metric": "fidelity(method(t), exact-propagated psi_final)",
            "adaptive_realtime_checkpoint_mode": str(cfg.realtime_checkpoint.mode),
        },
        "settings": _jsonable(asdict(cfg)),
        "default_provenance": dict(cfg.default_provenance),
        "artifacts": _payload_artifacts(cfg),
        "command": str(run_command),
        "stage_pipeline": _stage_summary(stage_result, cfg),
        "dynamics_noiseless": dict(dynamics_noiseless),
    }
    if adaptive_payload is not None:
        payload["adaptive_realtime_checkpoint"] = adaptive_payload
    payload["comparisons"] = _compute_comparisons(payload)
    return payload


def _profile_plot_page(pdf: Any, profile_name: str, profile_payload: Mapping[str, Any]) -> None:
    require_matplotlib()
    plt = get_plt()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    times = [float(x) for x in profile_payload.get("times", [])]
    methods = profile_payload.get("methods", {})
    for method_name, method_payload in methods.items():
        if not isinstance(method_payload, Mapping):
            continue
        rows = method_payload.get("trajectory", [])
        if not rows:
            continue
        energy_err = [float(r.get("abs_energy_error_vs_ground_state", float("nan"))) for r in rows]
        fidelity = [float(r["fidelity"]) for r in rows]
        axes[0].plot(times, energy_err, label=str(method_name))
        axes[1].plot(times, fidelity, label=str(method_name))
    axes[0].set_title(f"{profile_name}: |E_method - E_GS|")
    axes[0].set_ylabel("abs energy error vs GS")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title(f"{profile_name}: fidelity to seeded exact reference")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("fidelity")
    axes[1].set_ylim(0.0, 1.01)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    pdf.savefig(fig)
    plt.close(fig)


def _append_spectral_harmonic_markers(
    ax: Any,
    *,
    drive_omega: float | None,
    max_harmonic: int,
    ymax: float,
) -> None:
    if drive_omega is None or float(drive_omega) <= 0.0:
        return
    for harmonic in range(1, int(max_harmonic) + 1):
        omega_n = float(harmonic) * float(drive_omega)
        ax.axvline(omega_n, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            omega_n,
            ymax,
            f"{harmonic}ωd",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#666666",
        )


def _append_adaptive_rt_spectral_pages(
    pdf: Any,
    *,
    spectral_trust: Mapping[str, Any],
    cfg: StagedHHConfig,
) -> None:
    require_matplotlib()
    plt = get_plt()

    target_label = str(spectral_trust.get("display_label", spectral_trust.get("target_observable", "target")))
    controller_span = float(spectral_trust.get("oscillation_span_controller", float("nan")))
    exact_span = spectral_trust.get("oscillation_span_exact", None)
    exact_span_text = "n/a" if exact_span is None else f"{float(exact_span):.6f}"
    drive_line_ctrl = spectral_trust.get("drive_line_controller", None)
    drive_line_exact = spectral_trust.get("drive_line_exact", None)
    spectral_lines = [
        "Adaptive realtime spectral-trust summary",
        "",
        f"Target observable: {target_label}",
        f"Window / detrend: {spectral_trust.get('window')} / {spectral_trust.get('detrend')}",
        f"Oscillation span (controller / exact): {controller_span:.6f} / {exact_span_text}",
        f"Mean |target error|: {spectral_trust.get('mean_abs_error', 'n/a')}",
        f"RMS target error: {spectral_trust.get('rms_error', 'n/a')}",
        f"epsilon_osc: {spectral_trust.get('epsilon_osc', 'n/a')}",
        (
            "Drive line amplitude (controller / exact): "
            f"{float(drive_line_ctrl.get('amplitude', float('nan'))):.6f} / "
            f"{float(drive_line_exact.get('amplitude', float('nan'))):.6f}"
            if isinstance(drive_line_ctrl, Mapping) and isinstance(drive_line_exact, Mapping)
            else "Drive line amplitude (controller / exact): n/a"
        ),
        (
            "Drive line phase (controller / exact): "
            f"{float(drive_line_ctrl.get('phase_radians', float('nan'))):.6f} / "
            f"{float(drive_line_exact.get('phase_radians', float('nan'))):.6f}"
            if isinstance(drive_line_ctrl, Mapping) and isinstance(drive_line_exact, Mapping)
            else "Drive line phase (controller / exact): n/a"
        ),
    ]
    per_site_mae = spectral_trust.get("per_site_mae", None)
    if isinstance(per_site_mae, list) and per_site_mae:
        spectral_lines.append(
            "Per-site MAE: " + ", ".join(f"n_{idx}={float(value):.4e}" for idx, value in enumerate(per_site_mae))
        )
    render_text_page(pdf, spectral_lines, fontsize=10, line_spacing=0.03)

    times = np.asarray(spectral_trust.get("times", []), dtype=float)
    target_trace = np.asarray(spectral_trust.get("target_trace", []), dtype=float)
    target_exact = (
        None
        if spectral_trust.get("target_trace_exact") is None
        else np.asarray(spectral_trust.get("target_trace_exact"), dtype=float)
    )
    site_occ = np.asarray(spectral_trust.get("site_occupations", []), dtype=float)
    site_occ_exact = (
        None
        if spectral_trust.get("site_occupations_exact") is None
        else np.asarray(spectral_trust.get("site_occupations_exact"), dtype=float)
    )

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True)
    axes[0].plot(times, target_trace, linewidth=2.2, color="#D55E00", label="controller")
    if target_exact is not None:
        axes[0].plot(times, target_exact, linewidth=1.7, color="black", linestyle="--", label="exact")
    axes[0].set_title(f"Target observable vs exact | {target_label}")
    axes[0].set_ylabel("observable")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    colors = plt.cm.tab10.colors
    if site_occ.ndim == 2:
        for site in range(site_occ.shape[1]):
            axes[1].plot(
                times,
                site_occ[:, site],
                color=colors[site % len(colors)],
                linewidth=1.8,
                label=f"n_{site}",
            )
            if site_occ_exact is not None:
                axes[1].plot(
                    times,
                    site_occ_exact[:, site],
                    color=colors[site % len(colors)],
                    linewidth=1.1,
                    linestyle="--",
                    label=f"n_{site} exact",
                )
    axes[1].set_title("Per-site occupations vs exact")
    axes[1].set_xlabel(str(spectral_trust.get("time_key", "time")))
    axes[1].set_ylabel("occupation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
    omega = np.asarray(spectral_trust.get("spectrum_omega", []), dtype=float)
    amp_ctrl = np.asarray(spectral_trust.get("spectrum_amplitude_controller", []), dtype=float)
    amp_exact = (
        None
        if spectral_trust.get("spectrum_amplitude_exact") is None
        else np.asarray(spectral_trust.get("spectrum_amplitude_exact"), dtype=float)
    )
    axes[0].plot(omega, amp_ctrl, linewidth=2.0, color="#D55E00", label="controller")
    ymax = float(np.max(amp_ctrl)) if amp_ctrl.size else 1.0e-12
    if amp_exact is not None:
        axes[0].plot(omega, amp_exact, linewidth=1.5, color="black", linestyle="--", label="exact")
        ymax = max(ymax, float(np.max(amp_exact)) if amp_exact.size else 1.0e-12)
    _append_spectral_harmonic_markers(
        axes[0],
        drive_omega=float(cfg.dynamics.drive_omega) if bool(cfg.dynamics.enable_drive) else None,
        max_harmonic=int(spectral_trust.get("max_harmonic", 3)),
        ymax=max(ymax, 1.0e-12),
    )
    axes[0].set_title(f"Target observable one-sided amplitude spectrum | {target_label}")
    axes[0].set_xlabel("angular frequency ω")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    harmonic_rows: list[list[str]] = []
    controller_harmonics = spectral_trust.get("harmonics_controller", [])
    exact_harmonics = spectral_trust.get("harmonics_exact", [])
    exact_by_h = {
        int(round(float(entry.get("harmonic", -1)))): entry
        for entry in exact_harmonics
        if isinstance(entry, Mapping)
    }
    for entry in controller_harmonics:
        if not isinstance(entry, Mapping):
            continue
        harmonic = int(round(float(entry.get("harmonic", -1))))
        exact_entry = exact_by_h.get(harmonic, {})
        harmonic_rows.append(
            [
                str(harmonic),
                f"{float(entry.get('amplitude', float('nan'))):.4e}",
                (
                    f"{float(exact_entry.get('amplitude', float('nan'))):.4e}"
                    if isinstance(exact_entry, Mapping) and exact_entry
                    else "n/a"
                ),
                f"{float(entry.get('phase_radians', float('nan'))):.4f}",
                (
                    f"{float(exact_entry.get('phase_radians', float('nan'))):.4f}"
                    if isinstance(exact_entry, Mapping) and exact_entry
                    else "n/a"
                ),
            ]
        )
    if not harmonic_rows:
        harmonic_rows = [["-", "n/a", "n/a", "n/a", "n/a"]]
    render_compact_table(
        axes[1],
        title="Drive-locked harmonic summary",
        col_labels=["harm", "amp ctrl", "amp exact", "phase ctrl", "phase exact"],
        rows=harmonic_rows,
        fontsize=8,
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_staged_hh_pdf(payload: Mapping[str, Any], cfg: StagedHHConfig, run_command: str) -> None:
    if bool(cfg.artifacts.skip_pdf):
        return
    require_matplotlib()
    pdf_path = Path(cfg.artifacts.output_pdf)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    PdfPages = get_PdfPages()
    plt = get_plt()

    stage_pipeline = payload.get("stage_pipeline", {})
    warm = stage_pipeline.get("warm_start", {}) if isinstance(stage_pipeline, Mapping) else {}
    adapt = stage_pipeline.get("adapt_vqe", {}) if isinstance(stage_pipeline, Mapping) else {}
    replay = stage_pipeline.get("conventional_replay", {}) if isinstance(stage_pipeline, Mapping) else {}
    adaptive_rt = payload.get("adaptive_realtime_checkpoint", {}) if isinstance(payload, Mapping) else {}
    spectral_trust = adaptive_rt.get("spectral_trust", {}) if isinstance(adaptive_rt, Mapping) else {}

    with PdfPages(pdf_path) as pdf:
        render_parameter_manifest(
            pdf,
            model="Hubbard-Holstein",
            ansatz="warm: hh_hva_ptw; ADAPT: staged HH; final: matched-family replay",
            drive_enabled=bool(cfg.dynamics.enable_drive),
            t=float(cfg.physics.t),
            U=float(cfg.physics.u),
            dv=float(cfg.physics.dv),
            extra={
                "L": int(cfg.physics.L),
                "omega0": float(cfg.physics.omega0),
                "g_ep": float(cfg.physics.g_ep),
                "n_ph_max": int(cfg.physics.n_ph_max),
                "boundary": str(cfg.physics.boundary),
                "ordering": str(cfg.physics.ordering),
                "warm_reps": int(cfg.warm_start.reps),
                "adapt_mode": str(cfg.adapt.continuation_mode),
                "replay_mode": str(cfg.replay.continuation_mode),
                "methods": ",".join(cfg.dynamics.methods),
                "t_final": float(cfg.dynamics.t_final),
                "trotter_steps": int(cfg.dynamics.trotter_steps),
                "num_times": int(cfg.dynamics.num_times),
                "spectral_target": str(cfg.spectral_report.target_observable),
                "spectral_window": str(cfg.spectral_report.window),
                "spectral_detrend": str(cfg.spectral_report.detrend),
            },
            command=str(run_command),
        )
        summary_lines = [
            "HH staged noiseless workflow summary",
            "",
            f"Warm-start: E={warm.get('energy')} exact={warm.get('exact_energy')} delta={warm.get('delta_abs')} ecut_1={warm.get('ecut_1')}",
            f"ADAPT: depth={adapt.get('depth')} pool={adapt.get('pool_type')} delta={adapt.get('delta_abs')} stop={adapt.get('stop_reason')}",
            f"Replay: E={replay.get('energy')} exact={replay.get('exact_energy')} delta={replay.get('delta_abs')} ecut_2={replay.get('ecut_2')}",
            (
                "Checkpoint controller: disabled"
                if not isinstance(adaptive_rt, Mapping)
                else (
                    f"Checkpoint controller: mode={adaptive_rt.get('mode')} append={adaptive_rt.get('summary', {}).get('append_count')} stay={adaptive_rt.get('summary', {}).get('stay_count')} final_fidelity={adaptive_rt.get('summary', {}).get('final_fidelity_exact')}"
                )
            ),
            (
                "Spectral target: unavailable"
                if not isinstance(spectral_trust, Mapping) or not spectral_trust
                else (
                    f"Spectral target: {spectral_trust.get('display_label')} | "
                    f"epsilon_osc={spectral_trust.get('epsilon_osc')} | "
                    f"mean_abs_error={spectral_trust.get('mean_abs_error')}"
                )
            ),
            "Dynamics metrics: energy uses replay exact-sector GS baseline; fidelity uses exact propagation from psi_final.",
            "",
            "Artifacts",
            f"- workflow_json: {cfg.artifacts.output_json}",
            f"- workflow_pdf: {cfg.artifacts.output_pdf}",
            f"- adapt_handoff_json: {cfg.artifacts.handoff_json}",
            f"- replay_json: {cfg.artifacts.replay_output_json}",
            f"- replay_csv: {cfg.artifacts.replay_output_csv}",
            f"- replay_md: {cfg.artifacts.replay_output_md}",
            f"- replay_log: {cfg.artifacts.replay_output_log}",
        ]
        render_text_page(pdf, summary_lines, fontsize=10, line_spacing=0.03)

        fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5))
        render_compact_table(
            axes[0],
            title="Stage metrics",
            col_labels=["Stage", "Energy", "Exact", "|ΔE|", "Gate/stop"],
            rows=[
                ["Warm", f"{warm.get('energy', float('nan')):.8f}", f"{warm.get('exact_energy', float('nan')):.8f}", f"{warm.get('delta_abs', float('nan')):.3e}", str(warm.get('ecut_1', {}))],
                ["ADAPT", f"{adapt.get('energy', float('nan')):.8f}", f"{adapt.get('exact_energy', float('nan')):.8f}", f"{adapt.get('delta_abs', float('nan')):.3e}", str(adapt.get('stop_reason', ''))],
                ["Replay", f"{replay.get('energy', float('nan')):.8f}", f"{replay.get('exact_energy', float('nan')):.8f}", f"{replay.get('delta_abs', float('nan')):.3e}", str(replay.get('ecut_2', {}))],
            ],
            fontsize=8,
        )
        cmp_rows: list[list[str]] = []
        gs_cmp = payload.get("comparisons", {}).get("noiseless_vs_ground_state", {})
        ref_cmp = payload.get("comparisons", {}).get("noiseless_vs_reference", {})
        for profile_name, profile_cmp in gs_cmp.items():
            reference_methods = ref_cmp.get(profile_name, {}) if isinstance(ref_cmp, Mapping) else {}
            for method_name, rec in profile_cmp.items():
                ref_rec = reference_methods.get(method_name, {}) if isinstance(reference_methods, Mapping) else {}
                cmp_rows.append([
                    str(profile_name),
                    str(method_name),
                    f"{float(rec.get('final_abs_energy_error', float('nan'))):.3e}",
                    f"{float(ref_rec.get('final_fidelity', float('nan'))):.6f}",
                ])
        if not cmp_rows:
            cmp_rows = [["(none)", "(none)", "nan", "nan"]]
        render_compact_table(
            axes[1],
            title="Noiseless dynamics: GS error + seeded-reference fidelity",
            col_labels=["Profile", "Method", "Final |E-E_GS|", "Final fidelity"],
            rows=cmp_rows,
            fontsize=8,
        )
        pdf.savefig(fig)
        plt.close(fig)

        for profile_name, profile_payload in payload.get("dynamics_noiseless", {}).get("profiles", {}).items():
            if isinstance(profile_payload, Mapping):
                _profile_plot_page(pdf, str(profile_name), profile_payload)

        if isinstance(spectral_trust, Mapping) and spectral_trust:
            _append_adaptive_rt_spectral_pages(pdf, spectral_trust=spectral_trust, cfg=cfg)

        render_command_page(
            pdf,
            str(run_command),
            script_name="pipelines/hardcoded/hh_staged_noiseless.py",
            extra_header_lines=[
                f"workflow_json: {cfg.artifacts.output_json}",
                f"replay_json: {cfg.artifacts.replay_output_json}",
            ],
        )


def run_staged_hh_noiseless(cfg: StagedHHConfig, *, run_command: str | None = None) -> dict[str, Any]:
    run_command_str = current_command_string() if run_command is None else str(run_command)
    cfg.artifacts.output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_md.parent.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.replay_output_log.parent.mkdir(parents=True, exist_ok=True)

    if str(cfg.realtime_checkpoint.mode) == "oracle_v1":
        raise ValueError(
            "checkpoint controller oracle_v1 is only available through the staged noise workflow; use pipelines/hardcoded/hh_staged_noise.py."
        )
    if cfg.adapt.phase3_oracle_gradient_config is not None:
        raise ValueError(
            "phase3 oracle-gradient staged runs are only available through the staged noise workflow; use pipelines/hardcoded/hh_staged_noise.py."
        )
    stage_result = run_stage_pipeline(cfg)
    if str(cfg.realtime_checkpoint.mode) != "off":
        _prepare_adaptive_realtime_checkpoint_inputs(stage_result, cfg)
    dynamics_noiseless = run_noiseless_profiles(stage_result, cfg)
    adaptive_realtime_checkpoint = run_adaptive_realtime_checkpoint_profile(stage_result, cfg)
    payload = assemble_payload(
        cfg=cfg,
        stage_result=stage_result,
        dynamics_noiseless=dynamics_noiseless,
        adaptive_realtime_checkpoint=adaptive_realtime_checkpoint,
        run_command=run_command_str,
    )
    pareto_rows = extract_staged_hh_pareto_rows(
        run_tag=str(cfg.artifacts.tag),
        physics=asdict(cfg.physics),
        warm_payload=stage_result.warm_payload,
        adapt_payload=stage_result.adapt_payload,
        replay_payload=stage_result.replay_payload,
    )
    pareto_tracking = write_pareto_tracking(
        rows=pareto_rows,
        output_json_path=cfg.artifacts.output_json,
        run_tag=str(cfg.artifacts.tag),
    )
    payload.setdefault("artifacts", {})
    payload["artifacts"]["pareto"] = {
        key: str(value) for key, value in pareto_tracking["paths"].items()
    }
    payload["pareto_tracking"] = {
        "schema": str(pareto_tracking.get("schema", "")),
        "objective_axes": list(pareto_tracking.get("objective_axes", [])),
        "diagnostic_axes": list(pareto_tracking.get("diagnostic_axes", [])),
        "current_run": dict(pareto_tracking.get("current_run", {})),
        "rolling": dict(pareto_tracking.get("rolling", {})),
    }
    _write_json(cfg.artifacts.output_json, payload)
    if not bool(cfg.artifacts.skip_pdf):
        write_staged_hh_pdf(payload, cfg, run_command_str)
    return payload
