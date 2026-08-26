from __future__ import annotations

import pytest

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
    canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256,
    canonical_sr_snake_v4_contract,
    canonical_sr_snake_v4_contract_sha256,
)
from test_support.route_contract_kwargs import (
    expected_flat_settings,
    route_pruning,
    route_runtime_kwargs,
)

# Three inert infrastructure policies (all pinned "off") live in
# _CANONICAL_SR_SNAKE_RUNTIME_INFRASTRUCTURE rather than in any contract's
# execution settings, so they legitimately appear in flat kwargs even though
# their names are members of PRUNING_RUNTIME_KEYS.
_INERT_INFRASTRUCTURE_PRUNE_POLICIES = frozenset(
    {
        "phase1_prune_endpoint_overlap_policy",
        "phase1_prune_metric_mu_update_policy",
        "phase1_prune_trust_update_policy",
    }
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


def test_pruning_never_appears_as_loose_runtime_settings() -> None:
    # The executor-signature and CLI-parser halves of this guarantee are now
    # structural: neither surface exists, so no pruning flag can reappear on
    # them. The live guarantee is that the runtime kwargs builder routes
    # pruning exclusively through the typed extensions value.
    no_prune_kwargs = route_runtime_kwargs(
        route_contract=canonical_sr_snake_no_prune_symmetric_cost_v1_contract(),
        route_contract_sha256=(
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
    )
    assert "phase1_prune_enabled" not in no_prune_kwargs
    assert (
        set(no_prune_kwargs) & set(PRUNING_RUNTIME_KEYS)
        == _INERT_INFRASTRUCTURE_PRUNE_POLICIES
    )
    assert route_pruning(no_prune_kwargs) is None

    prune_kwargs = route_runtime_kwargs(
        route_contract=canonical_sr_snake_v4_contract(),
        route_contract_sha256=canonical_sr_snake_v4_contract_sha256(),
    )
    assert "phase1_prune_enabled" not in prune_kwargs
    assert (
        set(prune_kwargs) & set(PRUNING_RUNTIME_KEYS)
        == _INERT_INFRASTRUCTURE_PRUNE_POLICIES
    )
    assert route_pruning(prune_kwargs) is not None


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


def test_batching_never_appears_as_loose_runtime_settings() -> None:
    # As above: the executor and parser halves are structural now that both
    # legacy surfaces are deleted. Batching has no inert infrastructure keys,
    # so the live disjointness is exact.
    kwargs = route_runtime_kwargs(
        route_contract=canonical_sr_snake_no_prune_symmetric_cost_v1_contract(),
        route_contract_sha256=(
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
    )
    assert BATCH_RUNTIME_KEYS.isdisjoint(kwargs)


def test_extension_owned_keys_are_projected_out_of_flat_kwargs() -> None:
    contract = canonical_sr_snake_no_prune_symmetric_cost_v1_contract()
    flat = expected_flat_settings(contract)
    extension_owned = set(PRUNING_RUNTIME_KEYS) | set(BATCH_RUNTIME_KEYS)

    # The projection strips every extension-owned execution setting.
    assert extension_owned.isdisjoint(flat)

    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=(
            canonical_sr_snake_no_prune_symmetric_cost_v1_contract_sha256()
        ),
    )
    # The projected settings appear verbatim in the flat runtime kwargs.
    assert all(kwargs[key] == value for key, value in flat.items())

    # No extension-owned execution setting leaks into flat kwargs: batching is
    # exactly disjoint, and the only pruning-named keys present are the inert
    # infrastructure policies, which no contract owns.
    assert BATCH_RUNTIME_KEYS.isdisjoint(kwargs)
    assert (
        set(kwargs) & set(PRUNING_RUNTIME_KEYS)
        == _INERT_INFRASTRUCTURE_PRUNE_POLICIES
    )

    # Present extension state lives on kwargs["extensions"], nowhere else.
    assert kwargs["extensions"] == extensions_from_route_contract(contract)
    assert kwargs["extensions"].pruning is None
