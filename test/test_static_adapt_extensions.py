from __future__ import annotations

import pytest

from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.extensions import (
    BATCH_RUNTIME_KEYS,
    BatchExtension,
    NO_EXTENSIONS,
    PRUNING_RUNTIME_KEYS,
    batch_extension_from_admission,
    extensions_from_route_contract,
    resolve_pruning_runtime,
)
from pipelines.static_adapt.sr_snake.contracts import (
    CombinatorialBatchAdmission,
    SingletonAdmission,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract,
    canonical_sr_snake_v4_contract,
)


def test_absent_pruning_has_no_dormant_choices() -> None:
    extensions = extensions_from_route_contract(
        canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    )

    assert extensions == NO_EXTENSIONS
    assert tuple(extensions) == ()
    assert extensions.pruning is None


def test_enabled_pruning_resolves_only_from_authenticated_contract(
    tmp_path,
) -> None:
    extensions = extensions_from_route_contract(
        canonical_sr_snake_v4_contract()
    )

    assert extensions.pruning is not None
    runtime = resolve_pruning_runtime(
        extensions.pruning,
        repo_root=tmp_path,
    )
    assert runtime is not None
    assert runtime.modes(phase1_enabled=True) == (True, False)
    assert runtime.config.schur_nomination_route == (
        "full_logical_fs_trust_delete_refit_v1"
    )


def test_enabled_pruning_requires_its_complete_policy_interview() -> None:
    contract = canonical_sr_snake_v4_contract()
    del contract["execution_settings"]["phase1_prune_mode"]

    with pytest.raises(
        ValueError,
        match="phase1_prune_mode",
    ):
        extensions_from_route_contract(contract)


def test_pruning_is_absent_from_the_cli_default_surface() -> None:
    # The executor-signature half of this guarantee is now structural: the
    # legacy executor no longer exists, so no pruning key can appear on it.
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    args = parser.parse_args([])
    assert all(not hasattr(args, name) for name in PRUNING_RUNTIME_KEYS)
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["--phase1-prune-enabled"])


def test_enabled_batching_requires_a_complete_policy_interview() -> None:
    with pytest.raises(TypeError):
        BatchExtension()  # type: ignore[call-arg]

    extension = batch_extension_from_admission(
        CombinatorialBatchAdmission(
            maximum_size=3,
            search_window_size=6,
        )
    )

    assert extension == BatchExtension(
        strategy="combinatorial",
        maximum_size=3,
        search_window_size=6,
    )
    assert batch_extension_from_admission(SingletonAdmission()) is None


def test_batching_is_absent_from_the_cli_default_surface() -> None:
    # As above: structural now that the legacy executor is deleted.
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)
    args = parser.parse_args([])
    assert all(not hasattr(args, name) for name in BATCH_RUNTIME_KEYS)
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["--phase2-enable-batching"])
