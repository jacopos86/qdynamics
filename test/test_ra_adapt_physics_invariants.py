"""Physics invariants for the RA-ADAPT route.

These assert only properties the *method* guarantees, never properties that
require knowing the answer during the run.  The exact sector ground energy is
used strictly as a reporting-side check on the recorded trajectory, matching
the repository invariant that exact/classical references are reporting inputs
and never online controller inputs.

Concretely: the variational principle bounds every accepted state from below by
E_0, and RA admits a candidate only on a positive Schur-reduced gain, so the
accepted energies must descend.  Neither statement assumes convergence to E_0,
which the method does not promise at bounded depth.
"""

from __future__ import annotations

import math
from typing import Any

from pipelines.static_adapt.ra_adapt import run_ra_adapt
from test_ra_adapt_facade import (
    _hh_problem,
    _validated_macro_protocol,
    _validated_singleton_protocol,
)


def _accepted_energies(result: Any) -> list[float]:
    return [float(row.energy) for row in result.run.accepted_trajectory]


def _exact_ground_energy(problem: Any) -> float:
    return float(problem.exact_target.resolve_energy(ai_log=None))


def _assert_variational(energies: list[float], exact: float) -> None:
    """<psi|H|psi> >= E_0 for every accepted state."""

    for index, energy in enumerate(energies):
        assert math.isfinite(energy), (
            f"accepted round {index} recorded a non-finite energy {energy!r}"
        )
        # Tolerance absorbs float accumulation in the energy estimate only;
        # a real violation is a broken ansatz or a broken sector projection.
        assert energy >= exact - 1.0e-9, (
            f"accepted round {index} energy {energy!r} lies below the exact "
            f"sector ground energy {exact!r}, violating the variational bound"
        )


def _assert_descent(energies: list[float]) -> None:
    """RA admits only on positive reduced gain, so accepted energy descends."""

    for index in range(1, len(energies)):
        assert energies[index] <= energies[index - 1] + 1.0e-9, (
            f"accepted round {index} energy {energies[index]!r} rose above "
            f"round {index - 1} energy {energies[index - 1]!r}; RA admits a "
            "candidate only on a positive Schur-reduced gain"
        )


def test_macro_append_only_respects_variational_bound() -> None:
    problem = _hh_problem()
    result = run_ra_adapt(problem, _validated_macro_protocol(problem, rounds=3))
    energies = _accepted_energies(result)
    assert energies, "bounded macro run recorded no accepted state"
    _assert_variational(energies, _exact_ground_energy(problem))


def test_macro_append_only_descends() -> None:
    problem = _hh_problem()
    result = run_ra_adapt(problem, _validated_macro_protocol(problem, rounds=3))
    _assert_descent(_accepted_energies(result))


def test_singleton_plateau_respects_variational_bound() -> None:
    problem = _hh_problem()
    result = run_ra_adapt(
        problem, _validated_singleton_protocol(problem, rounds=3)
    )
    energies = _accepted_energies(result)
    assert energies, "bounded singleton run recorded no accepted state"
    _assert_variational(energies, _exact_ground_energy(problem))


def test_singleton_plateau_descends() -> None:
    problem = _hh_problem()
    result = run_ra_adapt(
        problem, _validated_singleton_protocol(problem, rounds=3)
    )
    _assert_descent(_accepted_energies(result))
