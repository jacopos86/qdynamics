from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.exact_bench import table_i_route_cutoff_audit
from pipelines.exact_bench import table_i_routea_spsa_historical_pareto
from pipelines.exact_bench import generic_static_benchmark
from pipelines.static_adapt import historical_route_identity as historical
from pipelines.static_adapt import sr_snake_escape_controller
from pipelines.static_adapt import sr_snake_route_profile as historical_profiles


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _route_a_components(**overrides: object) -> dict[str, object]:
    components: dict[str, object] = {
        "base_pool_key": "full_meta",
        "continuation_mode": "phase3_v1",
        "phase2_novelty_mode": "collective_span_v1",
        "phase3_selector_policy": "algebraic_nested_v1",
        "phase3_selector_geometry_mode": "reduced",
        "algebraic_shortlisting_enabled": True,
        "hardware_resolution_schema": "gradient_resolution_v1",
        "hardware_resolution_mode": "ideal",
        "phase2_raw_score_formula": "DeltaE_TR_raw * N2 / (1 + K2)",
        "canonical_score_formula": "DeltaE_TR * N3 / (1 + K3)",
        "primary_selector_score_key": "full_v2_score",
        "auxiliary_terms_primary_mode": "tie_break_only",
        "phase3_novelty_ablation_mode": "off",
        "phase3_window_relaxation_mode": "reduced",
        "phase3_enable_batching": True,
        "phase3_batch_selection_mode": "reduced_plane",
        "phase3_batch_prefilter_mode": "off",
        "phase3_batch_order_selection_mode": "finite_step_v1",
        "phase3_nested_window_application": "composed_batch_window_v1",
        "phase1_prune_enabled": True,
        "phase1_prune_policy": "recoverability_ladder_v1",
        "phase1_prune_mode": "both",
    }
    components.update(overrides)
    return components


@pytest.mark.parametrize(
    (
        "route_id",
        "observed",
        "optimizer_lane",
        "evaluation_convention",
        "expected_digest",
    ),
    (
        (
            "unspecified",
            {},
            None,
            None,
            "738a81acd9fc34d78e7a19c1a8204474d6f040a9ee8c71fb43004a795a37f9b4",
        ),
        (
            "route_a",
            _route_a_components(),
            "spsa",
            "same_cutoff",
            "a67af67b5eb6fc589b37693a2e97399e96e306904278649e907e5ba88ac4b52d",
        ),
        (
            "route_b_legacy_pairwise",
            _route_a_components(
                phase2_novelty_mode="legacy_pairwise_v1",
                phase2_raw_score_formula=(
                    "DeltaE_TR_raw * N2_pairwise / (1 + K2)"
                ),
            ),
            "powell",
            "same_cutoff",
            "9f6bf603ef6262b52a54797bad33841079e354cfeb7987fe37b276754a42c66e",
        ),
        (
            "route_c",
            _route_a_components(
                phase3_plateau_acquisition_mode="novelty_cost_v1",
                phase3_plateau_acquisition_score="log_volume_v1",
                phase3_plateau_score_formula=(
                    "log(1 + sigma_perp_lambda / lambda_vol) / (1 + K3)"
                ),
                phase3_plateau_duplicate_policy="block_exact_position_v1",
            ),
            None,
            None,
            "738485134c9bcd98c45e08b53f28f254e05f25840e866fb7a6f90dfe2e1c4084",
        ),
        (
            "route_a",
            _route_a_components(
                static_meta_feature_profile="safe_core_v1",
                phase1_prune_enabled=False,
            ),
            "spsa",
            None,
            "bf15e63db8a1f81aeceedee237971852ddb7ecb0b90a9c5cebc5959158391bef",
        ),
        (
            "route_a",
            _route_a_components(
                static_meta_feature_profile="paper_i_production_v1",
            ),
            "spsa",
            None,
            "5c9a3cc042b2008c2b6bccc836b5a8cd74988795a7c0c095694438792f5d008f",
        ),
        (
            "route_a",
            _route_a_components(
                route_variant_id=(
                    "route_a_h2o_linear_fd_physical_operator_lanes_"
                    "v2_derivative_resolved"
                ),
                base_pool_key="full_meta_derivative_resolved_v2",
            ),
            "spsa",
            None,
            "36b67de246cb8064d934dbbae7fb96fde5b2e704f79e742ce5e8347e65bb7152",
        ),
    ),
)
def test_passive_reader_preserves_frozen_historical_payload_digest(
    route_id: str,
    observed: dict[str, object],
    optimizer_lane: str | None,
    evaluation_convention: str | None,
    expected_digest: str,
) -> None:
    actual = historical.read_historical_route_identity(
        observed,
        declared_route_id=route_id,
        optimizer_lane=optimizer_lane,
        evaluation_convention=evaluation_convention,
    )

    assert _canonical_digest(actual) == expected_digest


@pytest.mark.parametrize(
    ("row", "expected"),
    (
        ({"static_route_id": "route-a"}, historical.ROUTE_ID_A),
        (
            {"phase2_novelty_mode": "legacy_pairwise_v1"},
            historical.ROUTE_ID_B_LEGACY_PAIRWISE,
        ),
        (
            {"phase3_plateau_acquisition_mode": "novelty_cost_v1"},
            historical.ROUTE_ID_C,
        ),
        ({}, historical.ROUTE_ID_UNSPECIFIED),
    ),
)
def test_passive_record_reader_preserves_historical_route_inference(
    row: dict[str, object],
    expected: str,
) -> None:
    assert historical.read_historical_static_route_id(row) == expected


def test_passive_reader_fails_closed_on_mixed_historical_telemetry() -> None:
    mixed = {
        "phase2_novelty_mode": "legacy_pairwise_v1",
        "phase3_plateau_acquisition_mode": "novelty_cost_v1",
    }

    assert (
        historical.read_historical_static_route_id(mixed)
        == historical.ROUTE_ID_UNSPECIFIED
    )
    with pytest.raises(ValueError, match="mixes legacy Route-B"):
        historical.read_historical_static_route_id(
            mixed,
            record_id="mixed",
            fail_on_route_named_missing=True,
        )


def test_passive_reader_exposes_no_execution_registry_or_builder() -> None:
    forbidden = {
        "ROUTE_ID_CHOICES",
        "FIRST_CLASS_ROUTE_ID_CHOICES",
        "LEGACY_ROUTE_ID_CHOICES",
        "StaticRouteIdentityConfig",
        "build_static_route_identity_observed_components",
        "build_static_route_identity_payload",
        "validate_declared_static_route_identity",
    }

    assert forbidden.isdisjoint(historical.__all__)
    assert all(not hasattr(historical, name) for name in forbidden)
    source = Path(historical.__file__).read_text(encoding="utf-8")
    assert "from pipelines.static_adapt.route_identity import" not in source
    assert "import pipelines.static_adapt.route_identity" not in source


def test_passive_reporting_consumers_import_the_quarantined_reader() -> None:
    assert (
        table_i_route_cutoff_audit.read_historical_route_identity
        is historical.read_historical_route_identity
    )
    assert (
        table_i_routea_spsa_historical_pareto.ROUTE_ID_A
        == historical.ROUTE_ID_A
    )
    route_payload = historical.read_historical_route_identity(
        _route_a_components(),
        declared_route_id=historical.ROUTE_ID_A,
        optimizer_lane="SPSA",
    )
    assert (
        table_i_route_cutoff_audit._route_identity_class(
            route_payload=route_payload,
            selected_logical_route="standard",
            selected_logical_source=None,
            working_n_ph_max=1,
            algorithm_id=table_i_route_cutoff_audit.DEFAULT_ALGORITHM_ID,
        )
        == "canonical_route_a_matched"
    )
    assert table_i_routea_spsa_historical_pareto._has_route_a_provenance(
        {"static_route_id": "route_a"}
    )


def test_non_monolith_consumers_do_not_import_executable_route_registry() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    migrated_consumers = (
        repo_root / "pipelines/exact_bench/generic_static_benchmark.py",
        repo_root / "pipelines/static_adapt/cli_config.py",
        repo_root / "pipelines/static_adapt/paper_i_config.py",
    )
    forbidden = (
        "from pipelines.static_adapt.route_identity import",
        "import pipelines.static_adapt.route_identity",
    )

    for path in migrated_consumers:
        source = path.read_text(encoding="utf-8")
        assert not any(text in source for text in forbidden), path


def test_retired_route_and_legacy_cli_owners_are_inert_and_unreachable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    retired_sources = (
        repo_root / "pipelines/static_adapt/route_identity.py",
        repo_root
        / "pipelines/static_adapt/sr_snake/_cli_compatibility.py",
        repo_root / "pipelines/static_adapt/sr_snake_escape_campaign.py",
        repo_root
        / "pipelines/static_adapt/diagnostics/"
        "route_c_weak_regime_autopilot.py",
    )
    archived_snapshots = {
        (
            repo_root
            / "archive/paper_i_static_adapt_legacy_20260727/"
            "code/historical_profiles_and_cli_controls/route_identity.py.txt"
        ): "aa674ea7fdb5165908fd667f867e561bc0e94b9952745531ba39d0dc3d4acfb5",
        (
            repo_root
            / "archive/paper_i_static_adapt_legacy_20260727/"
            "code/historical_profiles_and_cli_controls/"
            "sr_snake_cli_compatibility.py.txt"
        ): "9acc32029e1812088674d5fb411ebb4bf5edd665e60df961e91006d49bcfa331",
        (
            repo_root
            / "archive/paper_i_static_adapt_legacy_20260727/"
            "code/historical_profiles_and_cli_controls/"
            "sr_snake_escape_campaign.py.txt"
        ): "0b363ed67bf67ab7ab5a0cda34d2d9daeef48002c2f69b1f419d2d50bfea0241",
        (
            repo_root
            / "archive/paper_i_static_adapt_legacy_20260727/"
            "code/historical_profiles_and_cli_controls/"
            "route_c_weak_regime_autopilot.py.txt"
        ): "1ae7046a93098de26affd0dbfde360f7de49d3d06975a55d40076329f508d037",
    }

    assert all(not path.exists() for path in retired_sources)
    for path, expected_sha256 in archived_snapshots.items():
        assert path.suffix == ".txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256

    manifest_path = (
        repo_root
        / "archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries_by_archive_path = {
        str(entry["archive_path"]): entry for entry in manifest["entries"]
    }
    route_archive_root = (
        repo_root
        / "archive/paper_i_static_adapt_legacy_20260727/"
        "code/historical_profiles_and_cli_controls"
    )
    for path in route_archive_root.rglob("*.txt"):
        relative = str(path.relative_to(repo_root))
        entry = entries_by_archive_path[relative]
        assert entry["family"] == "historical_profiles_and_cli_controls"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

    forbidden_imports = (
        "pipelines.static_adapt.route_identity",
        "pipelines.static_adapt.sr_snake._cli_compatibility",
        "pipelines.static_adapt.sr_snake_escape_campaign",
        "route_c_weak_regime_autopilot",
    )
    for active_root in ("pipelines", "src"):
        for path in (repo_root / active_root).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert not any(name in source for name in forbidden_imports), path

    assert sr_snake_escape_controller.SR_ESCAPE_DISABLED == "disabled"


def test_retired_route_controls_are_absent_from_the_canonical_parser() -> None:
    # Structural note: the CLI parser is retired, so the retired flags
    # (--static-route-id, --allow-legacy-static-route,
    # --static-meta-feature-profile, --phase0-lane-quota-pressure,
    # --phase0-algebraic-lane-mode, --algebraic-phase*-lane-*) are
    # unreachable as user inputs by construction.  The live claim that
    # survives: the physical lane-quota controls and the escape mode remain
    # first-class runtime settings, while their algebraic counterparts and
    # the legacy-route toggle never reach the flat runtime kwargs.  Note
    # static_route_id / static_meta_feature_profile remain present in the
    # kwargs as frozen keys of the canonical execution_settings; the
    # guarantee is "not settable by any caller", not "absent".
    from test_support.route_contract_kwargs import (
        route_identity,
        route_runtime_kwargs,
    )

    repo_root = Path(__file__).resolve().parents[1]
    resolved, contract, contract_sha256 = route_identity("sr_snake_v4")
    kwargs = route_runtime_kwargs(
        route_contract=contract,
        route_contract_sha256=contract_sha256,
        route_profile=resolved,
        route_profile_request="sr_snake_v4",
    )

    retired_kwargs = {
        "allow_legacy_static_route",
        "phase0_lane_quota_pressure",
        "phase0_algebraic_lane_mode",
        "algebraic_phase1_lane_quota_pressure",
        "algebraic_phase2_lane_quota_pressure",
        "algebraic_phase2_lane_rel_threshold",
    }
    retained_kwargs = {
        "physical_phase1_lane_quota_pressure",
        "physical_phase2_lane_quota_pressure",
        "physical_phase2_lane_rel_threshold",
        "sr_escape_mode",
    }

    assert retired_kwargs.isdisjoint(kwargs)
    assert retained_kwargs.issubset(kwargs)
    assert kwargs["sr_escape_mode"] == sr_snake_escape_controller.SR_ESCAPE_DISABLED
    benchmark_runtime_source = (
        repo_root
        / "pipelines/exact_bench/static_benchmark_runtime.py"
    ).read_text(encoding="utf-8")
    assert "--static-route-id" not in benchmark_runtime_source
    assert "--static-meta-feature-profile" not in benchmark_runtime_source


def test_historical_profile_option_names_are_passive_validation_metadata() -> None:
    source = Path(historical_profiles.__file__).read_text(encoding="utf-8")

    assert historical_profiles._DEST_OPTION_STRINGS["static_route_id"] == (
        "--static-route-id",
    )
    assert historical_profiles._DEST_OPTION_STRINGS[
        "static_meta_feature_profile"
    ] == ("--static-meta-feature-profile",)
    assert "_DEST_OPTION_STRINGS" not in historical_profiles.__all__
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "def _set_option(" not in source
    assert "def build_command(" not in source


@pytest.mark.parametrize("route_id", ("route_a", "unspecified"))
def test_generic_launcher_rejects_retired_static_route_env(
    monkeypatch: pytest.MonkeyPatch,
    route_id: str,
) -> None:
    monkeypatch.setenv("GENERIC_STATIC_TABLE_STATIC_ROUTE_ID", route_id)

    with pytest.raises(
        ValueError,
        match="Historical static-route execution controls are retired",
    ):
        generic_static_benchmark._static_route_policy_overrides_from_env()


def test_generic_launcher_rejects_historical_route_policy() -> None:
    from pipelines.exact_bench.static_benchmark_runtime import (
        AlgorithmPolicy,
        StaticScaffoldPolicy,
    )

    policy = AlgorithmPolicy(
        static=StaticScaffoldPolicy(static_route_id="route_a"),
    )
    with pytest.raises(
        ValueError,
        match="cannot execute historical static_route_id=.*must be 'unspecified'",
    ):
        generic_static_benchmark._enforce_phase3_policy_algorithm_route_contract(
            policy,
            algorithm_id="static_family_native_adapt_phase3",
        )
