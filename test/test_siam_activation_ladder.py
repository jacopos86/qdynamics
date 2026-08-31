"""Versioned SIAM activation-ladder migration tests."""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.siam_activation_ladder import (
    SIAM_NB4_ACTIVATION_CUT_LADDER,
    SIAM_NB4_CANONICAL_LADDER_SHA256,
    SIAM_NB4_CANONICAL_LADDER_V2,
    SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER,
    SIAM_NB4_LEGACY_LADDER_SHA256,
    SIAM_NB4_LEGACY_LADDER_V1,
    canonical_registered_cut,
    ladder_contract_json,
    ladder_payload,
    resolve_embedded_ladder,
)


def test_canonical_ladder_is_the_frozen_coarse_to_floor_sequence() -> None:
    assert SIAM_NB4_ACTIVATION_CUT_LADDER == (
        1.0e-2,
        3.33333333333e-3,
        1.0e-3,
        3.33333333333e-4,
        1.0e-4,
        3.33333333333e-5,
        1.0e-5,
        3.33333333333e-6,
        1.0e-6,
        3.33333333333e-7,
        1.0e-7,
    )
    assert canonical_registered_cut(1.0e-3) == (1.0e-3, 2)


def test_legacy_r00_package_position_resolves_to_canonical_index_two() -> None:
    resolution = resolve_embedded_ladder(
        activation_cut=1.0e-3,
        package_index=0,
        action_ladder_sha256=SIAM_NB4_LEGACY_LADDER_SHA256,
        manifest_ladder_sha256=SIAM_NB4_LEGACY_LADDER_SHA256,
        manifest_ladder=list(
            ladder_payload(SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER)
        ),
    )
    assert resolution.package_contract == SIAM_NB4_LEGACY_LADDER_V1
    assert resolution.package_index == 0
    assert resolution.canonical_index == 2
    assert resolution.migrated is True


def test_new_package_contract_uses_canonical_position_and_digest() -> None:
    resolution = resolve_embedded_ladder(
        activation_cut=1.0e-3,
        package_index=2,
        action_ladder_sha256=SIAM_NB4_CANONICAL_LADDER_SHA256,
        manifest_ladder_sha256=SIAM_NB4_CANONICAL_LADDER_SHA256,
        manifest_ladder=list(ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER)),
    )
    assert resolution.package_contract == SIAM_NB4_CANONICAL_LADDER_V2
    assert resolution.package_index == resolution.canonical_index == 2
    assert resolution.migrated is False


def test_noncanonical_legacy_cut_remains_readable_but_covers_no_v2_rung() -> None:
    resolution = resolve_embedded_ladder(
        activation_cut=3.0e-4,
        package_index=1,
        action_ladder_sha256=SIAM_NB4_LEGACY_LADDER_SHA256,
        manifest_ladder_sha256=SIAM_NB4_LEGACY_LADDER_SHA256,
        manifest_ladder=list(
            ladder_payload(SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER)
        ),
    )
    assert resolution.package_contract == SIAM_NB4_LEGACY_LADDER_V1
    assert resolution.package_index == 1
    assert resolution.canonical_index is None
    assert resolution.has_canonical_equivalent is False
    assert resolution.migrated is True


def test_legacy_index_cannot_be_claimed_with_canonical_digest() -> None:
    with pytest.raises(ValueError, match="index/cut mismatch"):
        resolve_embedded_ladder(
            activation_cut=1.0e-3,
            package_index=0,
            action_ladder_sha256=SIAM_NB4_CANONICAL_LADDER_SHA256,
            manifest_ladder_sha256=SIAM_NB4_CANONICAL_LADDER_SHA256,
            manifest_ladder=list(ladder_payload(SIAM_NB4_ACTIVATION_CUT_LADDER)),
        )


def test_migration_contract_forbids_relabeling_legacy_action() -> None:
    contract = ladder_contract_json()
    alias = contract["legacy_compatibility"]["immutable_action_aliases"][
        "r00_cut1e-3"
    ]
    assert alias == {
        "activation_cut": 1.0e-3,
        "legacy_index": 0,
        "canonical_index": 2,
        "relabel": False,
    }
