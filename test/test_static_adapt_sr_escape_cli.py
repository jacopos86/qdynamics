from __future__ import annotations

import pytest

from pipelines.static_adapt.cli_config import (
    _build_adapt_arg_parser,
    _build_run_hardcoded_adapt_vqe_kwargs,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_ESCAPE_DISABLED,
    SR_ESCAPE_MODE_CHOICES,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
)


def _parser():
    return _build_adapt_arg_parser(adapt_gradient_parity_rtol=1.0e-7)


def _runtime_kwargs(*, sr_escape_mode: str) -> dict[str, object]:
    args = _parser().parse_args(["--sr-escape-mode", sr_escape_mode])
    return _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=None,
        cli_adapt_continuation_mode="legacy",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )


def test_sr_escape_mode_cli_defaults_to_disabled() -> None:
    args = _parser().parse_args([])

    assert args.sr_escape_mode == SR_ESCAPE_DISABLED


@pytest.mark.parametrize("mode", SR_ESCAPE_MODE_CHOICES)
def test_sr_escape_mode_cli_choices_flow_to_runtime_kwargs(mode: str) -> None:
    args = _parser().parse_args(["--sr-escape-mode", mode])

    assert isinstance(args.sr_escape_mode, str)
    assert args.sr_escape_mode == mode
    assert _runtime_kwargs(sr_escape_mode=mode)["sr_escape_mode"] == mode


def test_sr_escape_mode_cli_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(["--sr-escape-mode", "uncertified_escape"])


def test_sr_powell_chart_cli_defaults_to_auto_and_flows_to_runtime_kwargs() -> None:
    args = _parser().parse_args([])
    assert args.sr_powell_coordinate_chart_policy == (
        SR_POWELL_COORDINATE_CHART_AUTO
    )
    kwargs = _build_run_hardcoded_adapt_vqe_kwargs(
        args,
        h_poly=None,
        resolved_problem_context=None,
        cli_adapt_continuation_mode="legacy",
        adapt_ref_base_depth=0,
        psi_ref_override=None,
        psi_ref_source=None,
        psi_ref_handoff_state_kind=None,
        exact_gs_override=0.0,
        phase3_oracle_gradient_config=None,
        final_noise_audit_config=None,
    )
    assert kwargs["sr_powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_AUTO
    )


@pytest.mark.parametrize("policy", SR_POWELL_COORDINATE_CHART_POLICY_CHOICES)
def test_sr_powell_chart_cli_accepts_explicit_policies(policy: str) -> None:
    args = _parser().parse_args(
        ["--sr-powell-coordinate-chart-policy", policy]
    )
    assert args.sr_powell_coordinate_chart_policy == policy


def test_sr_powell_chart_cli_rejects_unknown_policy() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parser().parse_args(
            ["--sr-powell-coordinate-chart-policy", "silent_fallback"]
        )


def test_sr_escape_cli_accepts_required_global_trust_solver() -> None:
    args = _parser().parse_args(
        [
            "--sr-escape-mode",
            "saddle_only",
            "--historical-singleton-coordinate-solve-policy",
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2,
        ]
    )

    assert args.historical_singleton_coordinate_solve_policy == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_GLOBAL_TRUST_EIGH_V2
    )
