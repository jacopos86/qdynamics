"""Post-run exact trajectory for one recorded Paper-II drive.

This module is diagnostic-only.  It reconstructs the time-dependent
Hamiltonian from the trajectory's recorded drive profile and never feeds the
result back into support selection, numerical realization, or step control.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from scipy.linalg import expm

from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.observables import (
    build_site_doublon_observable_plan,
)


def exact_rows(
    seed_path: str | Path,
    run_path: str | Path,
    *,
    substeps: int = 40,
) -> list[dict[str, Any]]:
    """Return an exponential-midpoint reference on the recorded output grid."""

    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    return exact_rows_on_times(
        seed_path,
        run_path,
        times=[float(row["time"]) for row in run.get("plot_rows") or ()],
        substeps=int(substeps),
    )


def exact_rows_on_times(
    seed_path: str | Path,
    run_path: str | Path,
    *,
    times: Sequence[float],
    substeps: int = 40,
) -> list[dict[str, Any]]:
    """Return the exact seed trajectory on an explicit reporting grid.

    Propagation always begins at physical time zero.  This matters for resumed
    trajectories, whose recorded continuation begins at (for example) t=5 but
    whose exact state at that boundary is obtained by evolving the common seed
    from t=0, not by relabelling the seed state as the t=5 state.
    """

    if int(substeps) < 1:
        raise ValueError("substeps must be at least one")
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    profile = dict(run["hamiltonian"]["drive_profile"])
    runtime_input = load_scaffold_runtime_input(seed_path)
    request = getattr(runtime_input.resolved_problem, "request", None)
    drive_config = SimpleNamespace(
        enabled=True,
        drive_A=float(profile["A"]),
        drive_omega=float(profile["omega"]),
        drive_tbar=float(profile["tbar"]),
        drive_phi=float(profile["phi"]),
        drive_t0=float(profile["t0"]),
        drive_pattern=str(profile["pattern"]),
        drive_time_sampling=str(profile["time_sampling"]),
        drive_custom_weights=profile["custom_weights"],
        drive_include_identity=bool(profile["include_identity"]),
        drive_operator_label=str(profile["operator_label"]),
        operator_label=str(profile["operator_label"]),
        family_key=str(profile["family_key"]),
        n_sites=int(getattr(request, "num_sites", 2)),
        ordering=str(getattr(request, "ordering", "blocked")),
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_config=drive_config,
    )
    state = np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1)
    state /= np.linalg.norm(state)
    observable_plan = build_site_doublon_observable_plan(
        runtime_input.resolved_problem,
        dimension=state.size,
    )
    time_grid = [float(value) for value in times]
    if not time_grid:
        raise ValueError("trajectory has no plot rows")
    if any(not np.isfinite(value) for value in time_grid):
        raise ValueError("exact-reference times must be finite")
    if any(right < left for left, right in zip(time_grid, time_grid[1:])):
        raise ValueError("exact-reference times must be nondecreasing")
    if time_grid[0] < 0.0:
        raise ValueError("exact-reference trajectory cannot begin before t=0")

    rows: list[dict[str, Any]] = []
    current_time = 0.0
    for target_time in time_grid:
        interval = float(target_time - current_time)
        if interval > 0.0:
            dt = interval / int(substeps)
            for index in range(int(substeps)):
                midpoint = current_time + (index + 0.5) * dt
                matrix = np.asarray(
                    hamiltonian.matrix_at(midpoint),
                    dtype=complex,
                )
                state = expm(-1j * matrix * dt) @ state
                state /= np.linalg.norm(state)
            current_time = float(target_time)
        matrix = np.asarray(
            hamiltonian.matrix_at(target_time),
            dtype=complex,
        )
        snapshot = (
            observable_plan.evaluate(state)
            if observable_plan is not None
            else None
        )
        rows.append(
            {
                "time": float(target_time),
                "energy": float(np.real(np.vdot(state, matrix @ state))),
                "doublon": (
                    None if snapshot is None else float(snapshot.doublon)
                ),
                "site_occupations": (
                    None
                    if snapshot is None
                    else [
                        float(value)
                        for value in np.asarray(
                            snapshot.site_occupations, dtype=float
                        ).reshape(-1)
                    ]
                ),
                "primary_density": (
                    None
                    if snapshot is None
                    else float(snapshot.primary_density)
                ),
                "staggered": (
                    None if snapshot is None else float(snapshot.staggered)
                ),
                "statevector": state.copy(),
            }
        )
    return rows


__all__ = ["exact_rows", "exact_rows_on_times"]
