"""Declared Paper-III selected-QSE acquisition policies.

The generic selector keeps its historical defaults for behavior-lock and
backward-compatibility tests.  Paper-III producers must enter through this
module so a campaign cannot silently fall back to those legacy defaults.
"""

from __future__ import annotations

from pipelines.qse_spectra.record_selection import (
    GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
    StaticRecordSelectionConfig,
)


PAPER_III_LOCKED_SELECTOR_POLICY_ID = (
    "full_pencil_ky_fan_cost_ratio_novelty_floor_no_reward_v1"
)


def locked_paper_iii_selection_config(
    *,
    max_records: int,
    target_root_count: int,
    residual_stop: float | None,
) -> StaticRecordSelectionConfig:
    """Return the evidence-locked Paper-III acquisition configuration.

    Novelty remains a hard admissibility floor.  The additive novelty reward
    is disabled, and every admitted record must have a supported image.  The
    full proposed pencil supplies the multi-root Ky Fan gain used by the
    cost-normalized acquisition score.
    """

    return StaticRecordSelectionConfig(
        mode="geometry_selected",
        max_records=int(max_records),
        geometry_target_roots=int(target_root_count),
        geometry_score_mode=GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
        geometry_additive_novelty_reward=False,
        geometry_require_supported_image=True,
        geometry_min_metric_novelty=1.0e-12,
        geometry_cost_discount_alpha=1.0,
        geometry_residual_stop=(
            None if residual_stop is None else float(residual_stop)
        ),
    )


__all__ = [
    "PAPER_III_LOCKED_SELECTOR_POLICY_ID",
    "locked_paper_iii_selection_config",
]
