r"""Public configuration surface for finite-box SIAM impurity screening.

One call builds everything a run needs -- Hamiltonian, prepared state, logical
pool, observables, and the exact reference -- from a small declarative config,
so a caller never assembles a register layout or a Jordan-Wigner index by hand.

The physics lives in :mod:`src.quantum.siam_model`,
:mod:`src.quantum.siam_observables`, :mod:`src.quantum.siam_pool`, and
:mod:`src.quantum.siam_reference`; this module only composes them and is the
seam a runner or campaign should depend on.

The first pilot is :math:`N_b=4`, :math:`U=1.5`, :math:`V_f=1/\sqrt2`,
:math:`T=5` over 501 reporting points: 10 qubits, sector dimension 100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from src.quantum.siam_model import (
    DEFAULT_HOPPING,
    QUENCHED_HYBRIDIZATION,
    SIAMLayout,
    SIAM_MODEL_V1,
    build_initial_state,
    build_siam_hamiltonian,
    sector_indices,
)
from src.quantum.siam_observables import screening_observable_operators_v1
from src.quantum.siam_pool import build_siam_pool
from src.quantum.siam_reference import exact_reference

SIAM_SCREENING_CONFIG_V1 = "siam_screening_config_v1"


@dataclass(frozen=True)
class SIAMScreeningConfig:
    """One finite-box screening experiment."""

    n_bath: int = 4
    u: float = 1.5
    v_final: float = QUENCHED_HYBRIDIZATION
    hopping: float = DEFAULT_HOPPING
    t_final: float = 5.0
    num_times: int = 501
    label: str = "siam_pilot"

    def __post_init__(self) -> None:
        for name in ("n_bath", "num_times"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer; got {value!r}.")
        for name in ("u", "v_final", "hopping", "t_final"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number; got {value!r}.")
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite; got {value!r}.")
        if not isinstance(self.label, str):
            raise TypeError(f"label must be a string; got {self.label!r}.")
        if int(self.num_times) < 2:
            raise ValueError("num_times must be at least 2.")
        if float(self.t_final) <= 0.0:
            raise ValueError("t_final must be positive.")
        SIAMLayout(int(self.n_bath))  # validates the bath size

    @property
    def layout(self) -> SIAMLayout:
        return SIAMLayout(int(self.n_bath))

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(
            float(t)
            for t in np.linspace(0.0, float(self.t_final), int(self.num_times))
        )

    def provenance(self) -> dict[str, Any]:
        return {
            **self.physics_provenance(),
            "label": str(self.label),
            "t_final": float(self.t_final),
            "num_times": int(self.num_times),
        }

    def physics_provenance(self) -> dict[str, Any]:
        """Horizon-independent identity shared by every run of this model."""

        layout = self.layout
        n_up, n_dn = layout.electron_counts()
        return {
            "schema": SIAM_SCREENING_CONFIG_V1,
            "model": SIAM_MODEL_V1,
            "n_bath": int(self.n_bath),
            "u": float(self.u),
            "v_pre_quench": 0.0,
            "v_post_quench": float(self.v_final),
            "hopping": float(self.hopping),
            "n_qubits": layout.n_qubits,
            "electron_counts": {"up": n_up, "down": n_dn},
            "sector_dimension": len(sector_indices(int(self.n_bath))),
            "mode_order": "spin_blocked_jordan_wigner",
        }


@dataclass(frozen=True)
class SIAMScreeningProblem:
    """Everything a run needs, built once from a config."""

    config: SIAMScreeningConfig
    hamiltonian_pre_quench: Any
    hamiltonian: Any
    initial_state: np.ndarray
    pool: tuple
    observables: Mapping[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    def exact_reference(self, *, method: str = "eigh"):
        """Post-quench exact trajectory. Reporting only; never fed online."""

        return exact_reference(
            self.config.n_bath,
            u=self.config.u,
            v=self.config.v_final,
            hopping=self.config.hopping,
            times=self.config.times,
            method=method,
        )


def build_screening_problem(
    config: SIAMScreeningConfig | None = None,
) -> SIAMScreeningProblem:
    """Compose the full problem from a config."""

    cfg = config or SIAMScreeningConfig()
    layout = cfg.layout
    return SIAMScreeningProblem(
        config=cfg,
        hamiltonian_pre_quench=build_siam_hamiltonian(
            cfg.n_bath, u=cfg.u, v=0.0, hopping=cfg.hopping
        ),
        hamiltonian=build_siam_hamiltonian(
            cfg.n_bath, u=cfg.u, v=cfg.v_final, hopping=cfg.hopping
        ),
        initial_state=build_initial_state(cfg.n_bath, hopping=cfg.hopping),
        pool=tuple(build_siam_pool(cfg.n_bath)),
        # Preserve the already-submitted raw-run observable bundle exactly.
        # Sigma-v2 is reconstructed only at the versioned offline-reporting
        # boundary from recorded statevectors.
        observables=screening_observable_operators_v1(layout, v=cfg.v_final),
        provenance=cfg.provenance(),
    )


PILOT_NB4 = SIAMScreeningConfig(
    n_bath=4, u=1.5, t_final=5.0, num_times=501, label="siam_pilot_nb4_u1p5_t5"
)
FIXTURE_NB2 = SIAMScreeningConfig(
    n_bath=2, u=1.5, t_final=5.0, num_times=126, label="siam_fixture_nb2"
)


__all__ = [
    "FIXTURE_NB2",
    "PILOT_NB4",
    "SIAMScreeningConfig",
    "SIAMScreeningProblem",
    "SIAM_SCREENING_CONFIG_V1",
    "build_screening_problem",
]
