"""Contract tests for the locked Paper-III acquisition policy."""

from pipelines.qse_spectra.record_selection import (
    GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
)
from pipelines.qse_spectra.selection_policy import (
    PAPER_III_LOCKED_SELECTOR_POLICY_ID,
    locked_paper_iii_selection_config,
)


def test_locked_policy_is_full_pencil_cost_ratio_without_novelty_reward() -> None:
    config = locked_paper_iii_selection_config(
        max_records=64,
        target_root_count=6,
        residual_stop=1.0e-6,
    )

    assert PAPER_III_LOCKED_SELECTOR_POLICY_ID.endswith("_v1")
    assert config.geometry_score_mode == GEOMETRY_SCORE_FULL_PENCIL_KY_FAN
    assert config.geometry_additive_novelty_reward is False
    assert config.geometry_require_supported_image is True
    assert config.geometry_min_metric_novelty == 1.0e-12
    assert config.geometry_cost_discount_alpha == 1.0
    assert config.geometry_residual_stop == 1.0e-6
