"""CLI-free replacement for ``parse_args(argv)`` + ``_build_run_hardcoded_adapt_vqe_kwargs``.

The historical CLI surface built a 318-flag argparse namespace and projected it
onto the executor's keyword arguments.  ``adapt_pipeline.main()`` is retired and
that surface is being removed, but the assertions it carried are not: a route
profile must resolve to exactly one contract, that contract's execution settings
must appear verbatim in the flat runtime kwargs, and a profile disagreeing with
its contract must fail closed.

The live route resolves the same settings through
``adapt_pipeline._build_canonical_sr_snake_runtime_kwargs``, which takes a typed
``ResolvedProblemContext`` and a canonical route contract rather than CLI
strings.  This module is the one adapter between the two, so the nineteen test
modules that used the CLI do not each reinvent it.

**Where the fail-closed guarantee actually lives.**  The kwargs builder does
*not* reject a profile that disagrees with its contract -- it returns a 287-key
dict.  The guarantee is enforced one gate later, by
``adapt_pipeline._build_default_sr_controller_numerical_runtime``, and only for
profiles in that factory's authorized digest set.  Assertions about fail-closed
behaviour must therefore go through :func:`assert_route_binding_rejected`, not
through the kwargs dict.
"""

from __future__ import annotations

from typing import Any, Mapping

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.builders.problem_registry import (
    ProblemRequest,
    resolve_problem_context,
)
from pipelines.static_adapt.extensions import without_extension_runtime_keys
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
)


def hh_problem_context(
    *,
    n_ph_max: int = 1,
    u: float = 0.5,
    g_ep: float = 0.2,
) -> Any:
    """Return the shared canonical Hubbard--Holstein L=2 problem context."""

    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=u,
            dv=0.0,
            omega0=1.0,
            g_ep=g_ep,
            n_ph_max=n_ph_max,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def route_identity(profile_request: str) -> tuple[str, dict[str, Any], str]:
    """Resolve ``profile_request`` to ``(profile, contract, contract_sha256)``.

    Replaces the ``args.sr_route_profile_resolved`` / ``_contract`` /
    ``_contract_sha256`` triple the CLI namespace used to carry, including
    alias normalization.
    """

    resolved = normalize_sr_route_profile_request(profile_request)
    return (
        resolved,
        canonical_sr_snake_contract(resolved),
        canonical_sr_snake_contract_sha256(resolved),
    )


def route_runtime_kwargs(
    *,
    route_contract: Mapping[str, Any],
    route_contract_sha256: str,
    route_profile: str = "",
    route_profile_request: str | None = None,
    problem_context: Any = None,
    maximum_controller_rounds: int = 50,
    beam_extension: Any = None,
    exact_energy: float = -1.0,
    gradient_tolerance: float | None = None,
) -> dict[str, Any]:
    """Build the flat controller kwargs the live route uses.

    The checkpoint and exact-target parameters are defaulted away; tests that
    need them pass them explicitly.
    """

    return adapt_pipeline._build_canonical_sr_snake_runtime_kwargs(
        resolved_problem_context=(
            problem_context if problem_context is not None else hh_problem_context()
        ),
        maximum_controller_rounds=int(maximum_controller_rounds),
        exact_energy=float(exact_energy),
        exact_target_absolute_tolerance=None,
        exact_target_energy=None,
        checkpoint_path=None,
        checkpoint_every_controller_rounds=None,
        checkpoint_keep_history_tail=None,
        route_profile_request=str(
            route_profile_request
            if route_profile_request is not None
            else route_profile
        ),
        route_profile=str(route_profile),
        route_contract=dict(route_contract),
        route_contract_sha256=str(route_contract_sha256),
        beam_extension=beam_extension,
        gradient_tolerance=gradient_tolerance,
    )


def expected_flat_settings(route_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the execution settings that must appear verbatim in flat kwargs.

    Extension-owned keys are projected out (they move onto
    ``kwargs["extensions"]``), and the one boolean coercion the builder performs
    is applied, so the result compares equal key-for-key.
    """

    settings = dict(
        without_extension_runtime_keys(dict(route_contract["execution_settings"]))
    )
    if "adapt_final_full_refit" in settings:
        settings["adapt_final_full_refit"] = (
            str(settings["adapt_final_full_refit"]).strip().lower() == "true"
        )
    return settings


def route_pruning(kwargs: Mapping[str, Any]) -> Any:
    """Return the composed pruning extension, where phase1_prune_* settings live."""

    return kwargs["extensions"].pruning


def assert_route_binding_rejected(kwargs: Mapping[str, Any]) -> None:
    """Assert the runtime factory refuses this profile/contract binding.

    This is the live replacement for the CLI's fail-closed namespace check.  The
    kwargs builder itself does not validate the binding; the controller runtime
    factory does.
    """

    import pytest

    from pipelines.static_adapt.sr_snake.contracts import SRStopPolicy

    with pytest.raises(ValueError, match="requires the exact"):
        adapt_pipeline._build_default_sr_controller_numerical_runtime(
            stop_policy=SRStopPolicy(
                maximum_controller_rounds=int(kwargs["max_depth"]),
            ),
            executor_kwargs=dict(kwargs),
        )
