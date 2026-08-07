from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    build_controller_config,
    build_oracle_config,
    build_parser,
)
from pipelines.time_dynamics.legacy.checkpoint_route_policy import (
    policy_for_family,
    strict_qpu_faithful_requested,
    validate_realtime_route_request,
)


def _minimal_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--artifact-json",
        str(tmp_path / "artifact.json"),
        "--output-json",
        str(tmp_path / "out.json"),
    ]


def test_generic_strict_qpu_parser_alias_drives_hh_config_and_oracle(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        _minimal_cli_args(tmp_path)
        + [
            "--checkpoint-controller-strict-qpu-faithful",
            "--checkpoint-controller-mode",
            "oracle_v1",
        ]
    )

    assert args.checkpoint_controller_strict_qpu_faithful is True
    assert args.checkpoint_controller_strict_qpu_hh is False
    assert strict_qpu_faithful_requested(args) is True

    cfg = build_controller_config(args)
    oracle_cfg = build_oracle_config(args)

    assert cfg.mode == "oracle_v1"
    assert cfg.reference_mode == "off"
    assert cfg.append_no_harm_guard_enabled is False
    assert oracle_cfg is not None
    assert oracle_cfg.noise_mode == "ideal"
    assert oracle_cfg.backend_name is None
    assert oracle_cfg.allow_aer_fallback is False
    assert oracle_cfg.mitigation == {"mode": "none"}


def test_generic_strict_observable_v1_parser_uses_no_oracle_config(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        _minimal_cli_args(tmp_path)
        + [
            "--checkpoint-controller-strict-qpu-faithful",
            "--checkpoint-controller-mode",
            "observable_v1",
        ]
    )

    cfg = build_controller_config(args)
    oracle_cfg = build_oracle_config(args)

    assert cfg.mode == "observable_v1"
    assert cfg.reference_mode == "off"
    assert oracle_cfg is None


def test_legacy_strict_qpu_hh_parser_alias_still_normalizes(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        _minimal_cli_args(tmp_path)
        + [
            "--checkpoint-controller-strict-qpu-hh",
            "--checkpoint-controller-mode",
            "oracle_v1",
        ]
    )

    assert args.checkpoint_controller_strict_qpu_hh is True
    assert args.checkpoint_controller_strict_qpu_faithful is False
    assert strict_qpu_faithful_requested(args) is True


def test_generic_strict_rejects_exact_input_alias_benchmark_mode(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        _minimal_cli_args(tmp_path)
        + [
            "--checkpoint-controller-strict-qpu-faithful",
            "--checkpoint-controller-mode",
            "oracle_v1",
            "--checkpoint-controller-exact-input-mode",
            "benchmark_exact",
        ]
    )

    with pytest.raises(ValueError, match="reference-mode off"):
        build_controller_config(args)


def test_policy_accepts_generic_strict_without_append_pool_promotion() -> None:
    route = validate_realtime_route_request(
        family_key="hubbard",
        controller_mode="observable_v1",
        reference_mode="off",
        drive_requested=False,
        strict_qpu_faithful=True,
        append_pool_family="match_replay",
    )

    assert route.family_key == "hubbard"
    assert route.strict_qpu_faithful is True
    assert route.effective_append_pool_family == "match_replay"
    assert route.controller_mode == "observable_v1"
    assert route.reference_mode == "off"


def test_policy_requires_controller_exact_inputs_off_for_strict() -> None:
    with pytest.raises(ValueError, match="exact inputs off"):
        validate_realtime_route_request(
            family_key="hubbard",
            controller_mode="oracle_v1",
            reference_mode="benchmark_exact",
            drive_requested=False,
            strict_qpu_faithful=True,
        )


def test_policy_preserves_exact_v1_append_pool_auto_promotion() -> None:
    route = validate_realtime_route_request(
        family_key="spinless_tv",
        controller_mode="exact_v1",
        reference_mode="benchmark_exact",
        drive_requested=False,
        strict_qpu_faithful=False,
        append_pool_family="match_replay",
    )

    assert route.strict_qpu_faithful is False
    assert route.effective_append_pool_family == "full_meta"


def test_policy_supports_molecular_vibronic_h2_driven_exact_and_strict_routes() -> None:
    exact_route = validate_realtime_route_request(
        family_key="molecular_vibronic_h2",
        controller_mode="exact_v1",
        reference_mode="benchmark_exact",
        drive_requested=True,
        strict_qpu_faithful=False,
        append_pool_family="match_replay",
    )
    strict_route = validate_realtime_route_request(
        family_key="molecular_vibronic_h2",
        controller_mode="observable_v1",
        reference_mode="off",
        drive_requested=True,
        strict_qpu_faithful=True,
        append_pool_family="full_meta",
    )

    assert exact_route.family_key == "molecular_vibronic_h2"
    assert exact_route.effective_append_pool_family == "full_meta"
    assert strict_route.strict_qpu_faithful is True
    assert strict_route.effective_append_pool_family == "full_meta"


def test_policy_rejects_static_boson_chain_exact_v1() -> None:
    with pytest.raises(ValueError, match="Static bose_hubbard realtime does not support exact_v1"):
        validate_realtime_route_request(
            family_key="bose_hubbard",
            controller_mode="exact_v1",
            reference_mode="benchmark_exact",
            drive_requested=False,
            strict_qpu_faithful=False,
        )


def test_policy_guards_spin_boson_driven_route_shape() -> None:
    with pytest.raises(ValueError, match="num_sites == 1"):
        validate_realtime_route_request(
            family_key="spin_boson",
            controller_mode="oracle_v1",
            reference_mode="off",
            drive_requested=True,
            strict_qpu_faithful=True,
            num_sites=2,
        )

    with pytest.raises(ValueError, match="drive-include-identity"):
        validate_realtime_route_request(
            family_key="spin_boson",
            controller_mode="oracle_v1",
            reference_mode="off",
            drive_requested=True,
            strict_qpu_faithful=True,
            num_sites=1,
            drive_include_identity=True,
        )


def test_policy_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="does not support problem family"):
        policy_for_family("chemistry")
