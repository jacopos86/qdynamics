"""Post-run physical errors for Paper-II accuracy qualification.

The extractor consumes an already completed trajectory and an independently
constructed exact reference on the same grid.  None of these quantities are
available to either online structural method or time-step controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.staged_campaign import AccuracyMetrics


def _mean_abs(values: Sequence[float], *, name: str) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} series must be nonempty and finite")
    return float(np.mean(np.abs(array)))


def _recorded_state(row: Mapping[str, Any]) -> np.ndarray | None:
    real = row.get("psi_real")
    imag = row.get("psi_imag")
    if real is None and imag is None:
        return None
    if real is None or imag is None:
        raise ValueError("recorded statevector needs both psi_real and psi_imag")
    real_array = np.asarray(real, dtype=float).reshape(-1)
    imag_array = np.asarray(imag, dtype=float).reshape(-1)
    if real_array.size != imag_array.size or not real_array.size:
        raise ValueError("recorded statevector components have incompatible shapes")
    state = real_array + 1j * imag_array
    if not np.all(np.isfinite(state)):
        raise ValueError("recorded statevector must be finite")
    return state


def _infidelity(observed: np.ndarray, exact: np.ndarray) -> float:
    left = np.asarray(observed, dtype=complex).reshape(-1)
    right = np.asarray(exact, dtype=complex).reshape(-1)
    if left.size != right.size or not left.size:
        raise ValueError("observed and exact statevectors have incompatible shapes")
    left_norm_sq = float(np.real(np.vdot(left, left)))
    right_norm_sq = float(np.real(np.vdot(right, right)))
    if left_norm_sq <= 0.0 or right_norm_sq <= 0.0:
        raise ValueError("statevector norm must be positive")
    fidelity = float(abs(np.vdot(left, right)) ** 2 / (left_norm_sq * right_norm_sq))
    if not math.isfinite(fidelity):
        raise ValueError("state fidelity is not finite")
    return float(np.clip(1.0 - fidelity, 0.0, 1.0))


def physical_errors(
    plot_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> AccuracyMetrics:
    """Return the four componentwise errors declared by the campaign.

    The site-occupation quantity is first maximized over sites at each time,
    then averaged over time.  Infidelity is returned when *every* trajectory
    row carries a statevector; a partially recorded trajectory is rejected.
    """

    observed_rows = tuple(plot_rows)
    exact_rows = tuple(reference_rows)
    if not observed_rows or len(observed_rows) != len(exact_rows):
        raise ValueError("observed and exact trajectories need one nonempty common grid")

    energy_errors: list[float] = []
    doublon_errors: list[float] = []
    site_max_errors: list[float] = []
    infidelities: list[float] = []
    statevector_presence: list[bool] = []

    for observed, exact in zip(observed_rows, exact_rows):
        observed_time = float(observed["time"])
        exact_time = float(exact["time"])
        if not math.isclose(observed_time, exact_time, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"observed/exact time mismatch: {observed_time} != {exact_time}"
            )
        energy_errors.append(
            float(observed["energy_expectation"]) - float(exact["energy"])
        )
        doublon_errors.append(
            float(observed["doublon"]) - float(exact["doublon"])
        )
        observed_sites = np.asarray(observed["site_occupations"], dtype=float).reshape(-1)
        exact_sites = np.asarray(exact["site_occupations"], dtype=float).reshape(-1)
        if observed_sites.size != exact_sites.size or not observed_sites.size:
            raise ValueError("observed/exact site occupations have incompatible shapes")
        if not np.all(np.isfinite(observed_sites)) or not np.all(np.isfinite(exact_sites)):
            raise ValueError("site occupations must be finite")
        site_max_errors.append(float(np.max(np.abs(observed_sites - exact_sites))))

        observed_state = _recorded_state(observed)
        statevector_presence.append(observed_state is not None)
        if observed_state is not None:
            if exact.get("statevector") is None:
                raise ValueError("exact trajectory is missing a statevector")
            infidelities.append(
                _infidelity(observed_state, np.asarray(exact["statevector"], dtype=complex))
            )

    if any(statevector_presence) and not all(statevector_presence):
        raise ValueError("statevectors are present at only some trajectory checkpoints")

    return AccuracyMetrics(
        mean_abs_energy_error=_mean_abs(energy_errors, name="energy error"),
        mean_abs_doublon_error=_mean_abs(doublon_errors, name="doublon error"),
        mean_max_abs_site_occupation_error=_mean_abs(
            site_max_errors,
            name="maximum site-occupation error",
        ),
        mean_infidelity=(
            None
            if not infidelities
            else _mean_abs(infidelities, name="infidelity")
        ),
    )


def combine_resumed_rows(
    screen_rows: Sequence[Mapping[str, Any]],
    continuation_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Join t=0..5 and t=5..20 rows, retaining the boundary only once."""

    head = tuple(screen_rows)
    tail = tuple(continuation_rows)
    if not head or not tail:
        raise ValueError("resumed qualification needs both trajectory legs")
    boundary = float(head[-1]["time"])
    if not math.isclose(boundary, float(tail[0]["time"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("screen and continuation trajectories do not share a boundary")
    return (*head, *tail[1:])


def physical_errors_from_artifacts(
    *,
    run_path: str | Path,
    seed_path: str | Path,
    screen_run_path: str | Path | None = None,
    reference_substeps: int = 40,
    reference_store_root: str | Path | None = None,
) -> AccuracyMetrics:
    """Evaluate one ordinary run or one resumed t=0..20 trajectory.

    A resumed artifact stores only its continuation.  When ``screen_run_path``
    is supplied, the observed legs are joined at their common boundary and one
    exact reference is propagated from t=0 across the complete combined grid.
    """

    path = Path(run_path)
    run = json.loads(path.read_text(encoding="utf-8"))
    observed: tuple[Mapping[str, Any], ...] = tuple(run.get("plot_rows") or ())
    if screen_run_path is not None:
        screen = json.loads(Path(screen_run_path).read_text(encoding="utf-8"))
        observed = combine_resumed_rows(
            tuple(screen.get("plot_rows") or ()),
            observed,
        )
    times = [float(row["time"]) for row in observed]
    if reference_store_root is None:
        from pipelines.time_dynamics.diagnostics.exact_driven_reference import (
            exact_rows_on_times,
        )

        reference = exact_rows_on_times(
            seed_path,
            path,
            times=times,
            substeps=int(reference_substeps),
        )
    else:
        from pipelines.time_dynamics.diagnostics.exact_reference_store import (
            ExactReferenceStore,
        )

        reference = ExactReferenceStore(reference_store_root).get_or_build(
            seed_path=seed_path,
            run_path=path,
            times=times,
            substeps=int(reference_substeps),
        ).rows
    return physical_errors(observed, reference)


__all__ = [
    "combine_resumed_rows",
    "physical_errors",
    "physical_errors_from_artifacts",
]
