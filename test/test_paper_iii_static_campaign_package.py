"""Focused tests for deterministic local static-QSE action packages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.qse_spectra.static_campaign import (
    DEFAULT_TARGET_ROOTS,
    GroundStateArtifactRef,
    METHOD_RECORD_FIRST,
    StaticCampaignError,
    StaticQSECampaign,
    StaticQSECellKey,
    canonical_digest,
    hh_physics_key,
)
from pipelines.qse_spectra.static_campaign_package import (
    ACTION_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    build_action_command,
    build_local_package,
    load_local_package,
)

_SPECTRAL_REQUEST = {
    "transition_observables": [
        {"name": "impurity_occupation", "builder": "siam_impurity_occupation"}
    ],
    "green_function_modes": ["d_up=0"],
    "grid": {"omega_min": -6.0, "omega_max": 6.0, "num_points": 121},
    "kernel": {"kernel": "lorentzian", "eta": 0.1},
    "windows": [],
}


def _context() -> dict:
    return {
        "problem": {"kind": "hh", "regime": "weak_weak", "u": 0.25,
                    "g_ep": 0.353553390593, "n_ph_max": 1},
        "ground_state": {"kind": "exact_diagnostic"},
        "trace": {"stride": 2},
        "spectral_request": dict(_SPECTRAL_REQUEST),
        "qualification_policy": {},
    }


def _cell() -> StaticQSECellKey:
    artifact = GroundStateArtifactRef(
        provider_id="provider::weak_weak_nph1",
        manifest_path="states/weak_weak_nph1.json",
        manifest_sha256="ab" * 32,
        state_key="initial_state",
        state_payload_sha256="cd" * 32,
        num_qubits=8,
        sector="half_filled_11",
        energy=-1.25,
        source_revision="e6b2c2ca",
        achieved_settings_digest="ef" * 32,
    )
    return StaticQSECellKey(
        physics=hh_physics_key("weak_weak", 0.25, 0.353553390593, 1),
        ground_state_artifact_id=artifact.artifact_id,
        method_id=METHOD_RECORD_FIRST,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest="pool" * 6,
        spectral_request_digest=canonical_digest(_SPECTRAL_REQUEST),
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
    )


def test_build_action_command_is_the_stable_producer_invocation(tmp_path: Path) -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    action = campaign.all_actions()[0]
    command = build_action_command(action, tmp_path / action.action_id)
    assert command[:3] == ("python", "-m", "pipelines.exact_bench.paper_iii_static_qse_cell")
    assert command[3] == "--action-json"
    assert command[4].endswith(f"{action.action_id}/{ACTION_FILENAME}")
    assert command[5] == "--output-dir"


def test_package_is_deterministic_and_reloadable(tmp_path: Path) -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    actions = campaign.all_actions()
    contexts = {cell.digest(): _context()}

    root = tmp_path / "package"
    package = build_local_package(
        campaign, actions, package_root=root, cell_contexts=contexts
    )
    first_manifest = (root / PACKAGE_MANIFEST_FILENAME).read_bytes()
    build_local_package(campaign, actions, package_root=root, cell_contexts=contexts)
    assert (root / PACKAGE_MANIFEST_FILENAME).read_bytes() == first_manifest

    assert len(package.actions) == len(actions)
    assert set(package.actions_by_id) == {a.action_id for a in actions}
    for action in package.actions:
        action_file = json.loads(
            (root / action.action_id / ACTION_FILENAME).read_text()
        )
        assert action_file["action_id"] == action.action_id
        assert action_file["context"]["spectral_request"] == _SPECTRAL_REQUEST

    reloaded = load_local_package(campaign, root)
    assert [a.action_id for a in reloaded.actions] == [
        a.action_id for a in package.actions
    ]
    assert reloaded.artifact_root == root


def test_package_rejects_context_spectral_digest_mismatch(tmp_path: Path) -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    bad = _context()
    bad["spectral_request"] = {**_SPECTRAL_REQUEST, "kernel": {"kernel": "lorentzian", "eta": 0.2}}
    with pytest.raises(StaticCampaignError, match="spectral_request digest"):
        build_local_package(
            campaign,
            campaign.all_actions(),
            package_root=tmp_path / "p",
            cell_contexts={cell.digest(): bad},
        )


def test_package_rejects_undeclared_actions_and_missing_context(tmp_path: Path) -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    with pytest.raises(StaticCampaignError, match="missing execution context"):
        build_local_package(
            campaign,
            campaign.all_actions(),
            package_root=tmp_path / "p",
            cell_contexts={},
        )
    with pytest.raises(StaticCampaignError, match="at least one action"):
        build_local_package(
            campaign, (), package_root=tmp_path / "q", cell_contexts={}
        )


def test_manifest_digest_drift_is_detected(tmp_path: Path) -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    root = tmp_path / "package"
    build_local_package(
        campaign,
        campaign.all_actions(),
        package_root=root,
        cell_contexts={cell.digest(): _context()},
    )
    manifest_path = root / PACKAGE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["action_count"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(StaticCampaignError, match="manifest digest drift"):
        load_local_package(campaign, root)
