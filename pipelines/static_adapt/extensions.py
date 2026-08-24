"""Optional Paper-I controller extensions.

The mathematical singleton route does not own batch, prune, or beam choices.
An enabled extension is resolved here from an authenticated route contract;
an absent extension carries no dormant policy values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
    PRUNE_METRIC_COST_WEIGHT_OFF,
    PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1,
    PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1,
    PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1,
    PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
    PruneConfig,
    resolve_prune_tolerance_mode,
)
from pipelines.static_adapt.prune_risk_dataset import (
    PRUNE_PREFILTER_MOTIF_RISK_V1,
    PRUNE_PREFILTER_OFF,
    load_prune_prefilter_profile,
)


PRUNING_SETTING_NAMES = (
    "phase1_prune_policy",
    "phase1_prune_mode",
    "phase1_prune_fraction",
    "phase1_prune_min_candidates",
    "phase1_prune_max_candidates",
    "phase1_prune_max_regression",
    "phase1_prune_tolerance_mode",
    "phase1_prune_tolerance_shot_coeff",
    "phase1_prune_tolerance_screen_coeff",
    "phase1_prune_tolerance_chem",
    "phase1_prune_tolerance_rel_coeff",
    "phase1_prune_tolerance_target_energy",
    "phase1_prune_retained_gain_ratio",
    "phase1_prune_protect_steps",
    "phase1_prune_cooldown_steps",
    "phase1_prune_local_window_size",
    "phase1_prune_recovery_trust_radius",
    "phase1_prune_schur_nomination_route",
    "phase1_prune_metric_schur_mu",
    "phase1_prune_metric_schur_solve_mode",
    "phase1_prune_metric_schur_cost_weighting",
    "phase1_prune_trust_update_policy",
    "phase1_prune_metric_mu_update_policy",
    "phase1_prune_endpoint_overlap_policy",
    "phase1_prune_old_fraction",
    "phase1_prune_checkpoint_period",
    "phase1_prune_live_min_depth",
    "phase1_prune_maturity_threshold",
    "phase1_prune_snr_threshold",
    "phase1_prune_prefilter_policy",
    "phase1_prune_prefilter_json",
    "phase1_prune_risk_threshold",
    "phase1_prune_prefilter_max_candidates",
)

PRUNING_RUNTIME_KEYS = frozenset(
    {"phase1_prune_enabled", *PRUNING_SETTING_NAMES}
)


@dataclass(frozen=True, slots=True)
class PruningExtension:
    """Complete choices for one enabled pruning policy.

    There are deliberately no field defaults.  The extension is either absent
    or its authenticated contract supplies every value used by the controller.
    """

    settings: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.settings.items()}
        missing = [name for name in PRUNING_SETTING_NAMES if name not in values]
        if missing:
            raise ValueError(
                "Enabled pruning is missing required choices: "
                + ", ".join(missing)
            )
        unknown = sorted(set(values).difference(PRUNING_SETTING_NAMES))
        if unknown:
            raise ValueError(
                "Enabled pruning received unknown choices: "
                + ", ".join(unknown)
            )
        object.__setattr__(self, "settings", MappingProxyType(values))

    def __getitem__(self, name: str) -> Any:
        return self.settings[str(name)]

    def to_runtime_dict(self) -> dict[str, Any]:
        return dict(self.settings)


@dataclass(frozen=True, slots=True)
class Extensions:
    """Resolved optional behavior composed after the singleton route."""

    pruning: PruningExtension | None = None

    def __iter__(self) -> Iterator[object]:
        if self.pruning is not None:
            yield self.pruning


NO_EXTENSIONS = Extensions()


@dataclass(frozen=True, slots=True)
class PruningRuntime:
    """Validated internal state for an enabled pruning extension."""

    config: PruneConfig
    mode: str
    checkpoint_period: int
    live_min_depth: int
    maturity_threshold: float
    snr_threshold: float
    prefilter_policy: str
    prefilter_path: Path | None
    prefilter_profile: Mapping[str, Any] | None
    risk_threshold: float
    prefilter_max_candidates: int
    trust_update_policy: str
    metric_mu_update_policy: str
    endpoint_overlap_policy: str

    def modes(self, *, phase1_enabled: bool) -> tuple[bool, bool]:
        enabled = bool(phase1_enabled)
        return (
            bool(enabled and self.mode in {"live", "both"}),
            bool(enabled and self.mode in {"final", "both"}),
        )


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if value is None:
        return 0.0
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


def resolve_pruning_runtime(
    extension: PruningExtension | None,
    *,
    repo_root: Path,
) -> PruningRuntime | None:
    """Validate the conditional pruning interview only when it is enabled."""

    if extension is None:
        return None
    values = extension.settings
    policy = str(values["phase1_prune_policy"]).strip().lower()
    mode = str(values["phase1_prune_mode"]).strip().lower()
    if mode not in {"live", "final", "both"}:
        raise ValueError(f"Unsupported phase1_prune_mode: {mode}")
    prefilter_policy = str(
        values["phase1_prune_prefilter_policy"]
    ).strip().lower()
    if prefilter_policy not in {
        PRUNE_PREFILTER_OFF,
        PRUNE_PREFILTER_MOTIF_RISK_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_prefilter_policy: "
            f"{prefilter_policy}"
        )
    tolerance_mode_requested = str(
        values["phase1_prune_tolerance_mode"]
    ).strip().lower()
    tolerance_mode = resolve_prune_tolerance_mode(
        mode=tolerance_mode_requested,
        prune_policy=policy,
    )
    nomination_route = str(
        values["phase1_prune_schur_nomination_route"]
    ).strip().lower()
    if nomination_route not in {
        PRUNE_SCHUR_ROUTE_HESSIAN_COUPLING_V1,
        PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
        PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_schur_nomination_route: "
            f"{nomination_route}"
        )
    solve_mode = str(
        values["phase1_prune_metric_schur_solve_mode"]
    ).strip().lower()
    if solve_mode not in {
        PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
        PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1,
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_schur_solve_mode: "
            f"{solve_mode}"
        )
    cost_weighting = str(
        values["phase1_prune_metric_schur_cost_weighting"]
    ).strip().lower()
    if cost_weighting not in {
        PRUNE_METRIC_COST_WEIGHT_ANSATZ_ENTRY_DENOMINATOR_V1,
        PRUNE_METRIC_COST_WEIGHT_OFF,
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_schur_cost_weighting: "
            f"{cost_weighting}"
        )
    trust_update = str(
        values["phase1_prune_trust_update_policy"]
    ).strip().lower()
    if trust_update not in {"off", "modeled_local_fs_conservative_v1"}:
        raise ValueError(
            "Unsupported phase1_prune_trust_update_policy: "
            f"{trust_update}"
        )
    metric_mu_update = str(
        values["phase1_prune_metric_mu_update_policy"]
    ).strip().lower()
    if metric_mu_update not in {
        "off",
        "same_trial_underprediction_monotone_v1",
    }:
        raise ValueError(
            "Unsupported phase1_prune_metric_mu_update_policy: "
            f"{metric_mu_update}"
        )
    endpoint_overlap = str(
        values["phase1_prune_endpoint_overlap_policy"]
    ).strip().lower()
    if endpoint_overlap not in {"off", "energy_safe_trial_only_v1"}:
        raise ValueError(
            "Unsupported phase1_prune_endpoint_overlap_policy: "
            f"{endpoint_overlap}"
        )
    local_window_size = max(
        0, int(values["phase1_prune_local_window_size"])
    )
    trust_radius = _nonnegative_finite(
        values["phase1_prune_recovery_trust_radius"],
        name="phase1_prune_recovery_trust_radius",
    )
    if nomination_route == PRUNE_SCHUR_ROUTE_FULL_LOGICAL_FS_TRUST_DELETE_REFIT_V1:
        if solve_mode != PRUNE_METRIC_SCHUR_SOLVE_AFFINE_DELETION_GLOBAL_TRUST_V1:
            raise ValueError(
                "Full-logical FS trust pruning requires "
                "affine_deletion_global_trust_v1."
            )
        if local_window_size != 0:
            raise ValueError(
                "Full-logical FS trust pruning requires local_window_size=0."
            )
        if trust_radius <= 0.0:
            raise ValueError(
                "Full-logical FS trust pruning requires a positive radius."
            )
        if endpoint_overlap != "off":
            raise ValueError(
                "The query-neutral prune route forbids endpoint-overlap probes."
            )

    target_energy = values["phase1_prune_tolerance_target_energy"]
    if target_energy is not None and not math.isfinite(float(target_energy)):
        raise ValueError(
            "phase1_prune_tolerance_target_energy must be finite."
        )
    prefilter_path: Path | None = None
    prefilter_profile: Mapping[str, Any] | None = None
    if prefilter_policy == PRUNE_PREFILTER_MOTIF_RISK_V1:
        raw_path = values["phase1_prune_prefilter_json"]
        if raw_path in {None, ""}:
            raise ValueError(
                "motif_risk_v1 pruning requires phase1_prune_prefilter_json."
            )
        prefilter_path = Path(str(raw_path))
        if not prefilter_path.is_absolute():
            prefilter_path = repo_root / prefilter_path
        prefilter_profile = load_prune_prefilter_profile(prefilter_path)

    recoverability = policy == PRUNE_POLICY_RECOVERABILITY_LADDER_V1
    maximum_candidates = max(
        1, int(values["phase1_prune_max_candidates"])
    )
    config = PruneConfig(
        policy=policy,
        max_candidates=maximum_candidates,
        min_candidates=max(1, int(values["phase1_prune_min_candidates"])),
        fraction_candidates=max(
            0.0, float(values["phase1_prune_fraction"])
        ),
        max_regression=_nonnegative_finite(
            values["phase1_prune_max_regression"],
            name="phase1_prune_max_regression",
        ),
        tolerance_mode_requested=tolerance_mode_requested,
        tolerance_mode=tolerance_mode,
        tolerance_shot_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_shot_coeff"],
            name="phase1_prune_tolerance_shot_coeff",
        ),
        tolerance_screen_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_screen_coeff"],
            name="phase1_prune_tolerance_screen_coeff",
        ),
        tolerance_chem=_nonnegative_finite(
            values["phase1_prune_tolerance_chem"],
            name="phase1_prune_tolerance_chem",
        ),
        tolerance_rel_coeff=_nonnegative_finite(
            values["phase1_prune_tolerance_rel_coeff"],
            name="phase1_prune_tolerance_rel_coeff",
        ),
        tolerance_target_energy=(
            None if target_energy is None else float(target_energy)
        ),
        retained_gain_ratio=max(
            0.0, float(values["phase1_prune_retained_gain_ratio"])
        ),
        protect_steps=max(0, int(values["phase1_prune_protect_steps"])),
        cooldown_steps=max(0, int(values["phase1_prune_cooldown_steps"])),
        local_window_size=local_window_size,
        surrogate_recovery_trust_radius=trust_radius,
        schur_nomination_route=nomination_route,
        metric_schur_mu=_nonnegative_finite(
            values["phase1_prune_metric_schur_mu"],
            name="phase1_prune_metric_schur_mu",
        ),
        metric_schur_solve_mode=solve_mode,
        metric_schur_cost_weighting=cost_weighting,
        old_fraction=min(
            1.0, max(0.0, float(values["phase1_prune_old_fraction"]))
        ),
        surrogate_enabled=recoverability,
        surrogate_nomination_gate_enabled=recoverability,
        surrogate_nomination_gate_factor=1.0,
        surrogate_exact_trial_cap=(1 if recoverability else maximum_candidates),
    )
    return PruningRuntime(
        config=config,
        mode=mode,
        checkpoint_period=max(
            1, int(values["phase1_prune_checkpoint_period"])
        ),
        live_min_depth=max(0, int(values["phase1_prune_live_min_depth"])),
        maturity_threshold=max(
            0.0, float(values["phase1_prune_maturity_threshold"])
        ),
        snr_threshold=max(0.0, float(values["phase1_prune_snr_threshold"])),
        prefilter_policy=prefilter_policy,
        prefilter_path=prefilter_path,
        prefilter_profile=prefilter_profile,
        risk_threshold=_nonnegative_finite(
            values["phase1_prune_risk_threshold"],
            name="phase1_prune_risk_threshold",
        ),
        prefilter_max_candidates=max(
            0, int(values["phase1_prune_prefilter_max_candidates"])
        ),
        trust_update_policy=trust_update,
        metric_mu_update_policy=metric_mu_update,
        endpoint_overlap_policy=endpoint_overlap,
    )


def extensions_from_route_contract(
    contract: Mapping[str, Any] | None,
) -> Extensions:
    """Resolve enabled extensions without retaining disabled-policy choices."""

    if contract is None:
        return NO_EXTENSIONS
    execution_settings = contract.get("execution_settings")
    if not isinstance(execution_settings, Mapping):
        raise ValueError("The route contract lacks execution settings.")
    if not bool(execution_settings.get("phase1_prune_enabled", False)):
        return NO_EXTENSIONS
    return Extensions(
        pruning=PruningExtension(
            {
                name: execution_settings[name]
                for name in PRUNING_SETTING_NAMES
                if name in execution_settings
            }
        )
    )


def without_extension_runtime_keys(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Project core runtime settings without optional-extension controls."""

    return {
        str(key): value
        for key, value in settings.items()
        if str(key) not in PRUNING_RUNTIME_KEYS
    }
