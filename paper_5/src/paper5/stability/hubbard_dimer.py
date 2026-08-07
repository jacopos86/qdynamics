"""Scalar Hubbard--Holstein dimer stability models.

The five-variable Ehrenfest system follows Eqs. (78)--(82) of the working
document ``Dynamics_on_the_Hubbard_DIMER.pdf``. The thirteen-variable
Fan--Migdal system follows Eqs. (87)--(99). These equations are a diagnostic
reproduction seam; their component-level derivation from the primary matrix
Eqs. (14a)--(14e) still needs an independent source-faithful check.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sin, sqrt
from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RhsFunction = Callable[[float, FloatArray], FloatArray]

EHRENFEST_STATE_NAMES = (
    "delta_n",
    "rho_real",
    "rho_imag",
    "delta_b_real",
    "delta_b_imag",
)

FAN_MIGDAL_STATE_NAMES = (
    *EHRENFEST_STATE_NAMES,
    "phonon_population",
    "phonon_coherence",
    "delta_corr_real",
    "delta_corr_imag",
    "delta_corr_imag_plus",
    "delta_corr_imag_minus",
    "delta_corr_real_plus",
    "delta_corr_real_minus",
)

EXTENDED_FAN_MIGDAL_STATE_NAMES = (
    *FAN_MIGDAL_STATE_NAMES,
    "anomalous_relative_real",
    "anomalous_relative_imag",
)


@dataclass(frozen=True)
class DimerParameters:
    """Dimensionless scalar-model parameters with ``t_hop = hopping``."""

    hopping: float = 1.0
    gamma: float = 0.5
    lambda_ep: float = 1.5
    drive_amplitude: float = 1.0
    pulse_width: float = 1.0
    eq95_source_scale: float = 1.0
    eq97_source_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.hopping <= 0.0:
            raise ValueError("hopping must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.lambda_ep < 0.0:
            raise ValueError("lambda_ep must be nonnegative")
        if self.pulse_width <= 0.0:
            raise ValueError("pulse_width must be positive")

    @property
    def omega_ph(self) -> float:
        """Phonon frequency from ``gamma = omega_ph / t_hop``."""

        return self.gamma * self.hopping

    @property
    def coupling(self) -> float:
        """Coupling from ``lambda = 2 g^2 / (t_hop omega_ph)``."""

        return sqrt(0.5 * self.lambda_ep * self.hopping * self.omega_ph)

    def drive_difference(self, time: float) -> float:
        """Return the dimer field difference used below Eq. (83)."""

        scaled_time = time / self.pulse_width
        envelope = exp(-0.5 * scaled_time * scaled_time)
        return 2.0 * self.drive_amplitude * sin(pi * time / 4.0) * envelope


@dataclass(frozen=True)
class GaussianSineDrive:
    """One or more delayed copies of the declared dimer pulse.

    Each pulse is causal in its local time ``s = time - delay`` and has field
    difference ``2 A sin(pi s / 4) exp[-s^2/(2 width^2)]`` for ``s >= 0``.
    """

    amplitude: float = 1.0
    pulse_width: float = 1.0
    delays: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        if self.pulse_width <= 0.0:
            raise ValueError("pulse_width must be positive")
        if not self.delays:
            raise ValueError("at least one pulse delay is required")
        if any(delay < 0.0 for delay in self.delays):
            raise ValueError("pulse delays must be nonnegative")
        if tuple(sorted(self.delays)) != self.delays:
            raise ValueError("pulse delays must be sorted")

    @classmethod
    def from_parameters(
        cls,
        parameters: DimerParameters,
        *,
        delays: tuple[float, ...] = (0.0,),
    ) -> GaussianSineDrive:
        """Build the pulse protocol associated with ``parameters``."""

        return cls(
            amplitude=parameters.drive_amplitude,
            pulse_width=parameters.pulse_width,
            delays=delays,
        )

    def difference(self, time: float) -> float:
        """Return the sum of all causal pulse field differences."""

        value = 0.0
        for delay in self.delays:
            local_time = time - delay
            if local_time < 0.0:
                continue
            scaled_time = local_time / self.pulse_width
            envelope = exp(-0.5 * scaled_time * scaled_time)
            value += (
                2.0
                * self.amplitude
                * sin(pi * local_time / 4.0)
                * envelope
            )
        return value

    def derivative(self, time: float) -> float:
        """Return the right-continuous time derivative of the pulse sum."""

        value = 0.0
        for delay in self.delays:
            local_time = time - delay
            if local_time < 0.0:
                continue
            scaled_time = local_time / self.pulse_width
            envelope = exp(-0.5 * scaled_time * scaled_time)
            carrier = pi * local_time / 4.0
            value += 2.0 * self.amplitude * envelope * (
                (pi / 4.0) * np.cos(carrier)
                - (local_time / self.pulse_width**2) * sin(carrier)
            )
        return float(value)


@dataclass(frozen=True)
class IntegrationSummary:
    """Compact result from a threshold-aware fixed-step integration."""

    final_time: float
    final_state: FloatArray
    max_abs_state: float
    steps: int
    failure_time: float | None
    failure_component: str | None

    @property
    def diverged(self) -> bool:
        return self.failure_time is not None


def _require_state(state: FloatArray, names: tuple[str, ...]) -> FloatArray:
    array = np.asarray(state, dtype=float)
    if array.shape != (len(names),):
        raise ValueError(f"expected state shape {(len(names),)}, got {array.shape}")
    return array


def ehrenfest_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Evaluate the five real Ehrenfest equations (78)--(82)."""

    delta_n, rho_real, rho_imag, delta_b_real, delta_b_imag = _require_state(
        state, EHRENFEST_STATE_NAMES
    )
    hopping = parameters.hopping
    omega_ph = parameters.omega_ph
    coupling = parameters.coupling
    drive = parameters.drive_difference(time)

    return np.array(
        [
            4.0 * hopping * rho_imag,
            rho_imag * drive + 2.0 * coupling * rho_imag * delta_b_real,
            -hopping * delta_n
            - 2.0 * coupling * rho_real * delta_b_real
            - drive * rho_real,
            omega_ph * delta_b_imag,
            -omega_ph * delta_b_real - 2.0 * coupling * delta_n,
        ],
        dtype=float,
    )


def ehrenfest_invariant(state: FloatArray) -> float:
    """Return Eq. (86), the electronic Bloch-length invariant."""

    delta_n, rho_real, rho_imag, _, _ = _require_state(
        state, EHRENFEST_STATE_NAMES
    )
    return float(delta_n**2 + 4.0 * (rho_real**2 + rho_imag**2))


def ehrenfest_fixed_point(
    parameters: DimerParameters,
    *,
    branch: int = 1,
) -> FloatArray:
    """Return the zero-drive minimum-energy candidate in Eqs. (84)--(86)."""

    if branch not in (-1, 1):
        raise ValueError("branch must be -1 or 1")

    lambda_ep = parameters.lambda_ep
    if lambda_ep <= 1.0:
        delta_n = 0.0
        rho_real = 0.5
        delta_b_real = 0.0
    else:
        delta_n = branch * sqrt(1.0 - lambda_ep**-2)
        rho_real = 0.5 / lambda_ep
        delta_b_real = (
            -2.0 * parameters.coupling * delta_n / parameters.omega_ph
        )

    return np.array([delta_n, rho_real, 0.0, delta_b_real, 0.0], dtype=float)


def hartree_fock_zero_correlation_state() -> FloatArray:
    """Return the uncorrelated initial state used in the divergence comparison."""

    state = np.zeros(len(FAN_MIGDAL_STATE_NAMES), dtype=float)
    state[FAN_MIGDAL_STATE_NAMES.index("rho_real")] = 0.5
    return state


def fan_migdal_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Evaluate the thirteen real Fan--Migdal equations (87)--(99).

    ``eq95_source_scale`` and ``eq97_source_scale`` expose the two correlation
    source products for diagnostic ablations. Their physical values are one;
    other values do not define candidate physical equations.
    """

    (
        delta_n,
        rho_real,
        rho_imag,
        delta_b_real,
        delta_b_imag,
        phonon_population,
        phonon_coherence,
        delta_corr_real,
        delta_corr_imag,
        delta_corr_imag_plus,
        delta_corr_imag_minus,
        delta_corr_real_plus,
        delta_corr_real_minus,
    ) = _require_state(state, FAN_MIGDAL_STATE_NAMES)

    hopping = parameters.hopping
    omega_ph = parameters.omega_ph
    coupling = parameters.coupling
    drive = parameters.drive_difference(time)
    phonon_factor = 1.0 + 2.0 * phonon_population - 2.0 * phonon_coherence

    return np.array(
        [
            # Eq. (87)
            4.0 * hopping * rho_imag,
            # Eq. (88)
            rho_imag * drive
            + 2.0 * coupling * rho_imag * delta_b_real
            + 2.0 * coupling * delta_corr_imag_minus,
            # Eq. (89)
            -hopping * delta_n
            - 2.0 * coupling * rho_real * delta_b_real
            - drive * rho_real
            - 2.0 * coupling * delta_corr_real_plus,
            # Eq. (90)
            omega_ph * delta_b_imag,
            # Eq. (91)
            -omega_ph * delta_b_real - 2.0 * coupling * delta_n,
            # Eq. (92)
            -4.0 * coupling * delta_corr_imag,
            # Eq. (93)
            4.0 * coupling * delta_corr_imag,
            # Eq. (94)
            hopping * delta_corr_imag_minus
            + omega_ph * delta_corr_imag,
            # Eq. (95)
            -hopping * delta_corr_real_minus
            - omega_ph * delta_corr_real
            - parameters.eq95_source_scale
            * 0.25
            * coupling
            * (1.0 - delta_n**2),
            # Eq. (96)
            -delta_corr_real_minus * drive
            - omega_ph * delta_corr_real_plus
            - 2.0
            * coupling
            * delta_b_real
            * delta_corr_real_minus
            + coupling * rho_real * delta_n,
            # Eq. (97)
            -4.0 * hopping * delta_corr_real
            - delta_corr_real_plus * drive
            - omega_ph * delta_corr_real_minus
            - 2.0
            * coupling
            * delta_b_real
            * delta_corr_real_plus
            - parameters.eq97_source_scale
            * coupling
            * rho_real
            * phonon_factor,
            # Eq. (98)
            delta_corr_imag_minus * drive
            + omega_ph * delta_corr_imag_plus
            + 2.0
            * coupling
            * delta_b_real
            * delta_corr_imag_minus
            + coupling * rho_imag * phonon_factor,
            # Eq. (99)
            4.0 * hopping * delta_corr_imag
            + delta_corr_imag_plus * drive
            + omega_ph * delta_corr_imag_minus
            + 2.0
            * coupling
            * delta_b_real
            * delta_corr_imag_plus
            - coupling * rho_imag * delta_n,
        ],
        dtype=float,
    )


def fan_migdal_with_anomalous_rhs(
    time: float,
    state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Evaluate the minimal fifteen-real-coordinate Eq. (14) reduction.

    The two added coordinates are the real and imaginary parts of the
    connected anomalous relative-mode density

    ``phi = <b_- b_->_c = A[0, 0] - A[0, 1]``.

    They supply the Eq. (14c) dynamics omitted by Eqs. (87)--(99).  The
    analytically decoupled coherent center mode is quotiented out.
    """

    array = _require_state(state, EXTENDED_FAN_MIGDAL_STATE_NAMES)
    scalar = array[: len(FAN_MIGDAL_STATE_NAMES)]
    anomalous_real, anomalous_imag = array[-2:]
    derivative = fan_migdal_rhs(time, scalar, parameters)

    coupling = parameters.coupling
    omega_ph = parameters.omega_ph
    rho_imag = scalar[FAN_MIGDAL_STATE_NAMES.index("rho_imag")]
    delta_corr_real = scalar[
        FAN_MIGDAL_STATE_NAMES.index("delta_corr_real")
    ]
    delta_corr_imag = scalar[
        FAN_MIGDAL_STATE_NAMES.index("delta_corr_imag")
    ]

    derivative[
        FAN_MIGDAL_STATE_NAMES.index("delta_corr_imag_plus")
    ] += 2.0 * coupling * rho_imag * anomalous_imag
    derivative[
        FAN_MIGDAL_STATE_NAMES.index("delta_corr_real_plus")
    ] += 2.0 * coupling * rho_imag * anomalous_real

    anomalous_derivative = np.array(
        [
            2.0 * omega_ph * anomalous_imag
            + 8.0 * coupling * delta_corr_imag,
            -2.0 * omega_ph * anomalous_real
            - 8.0 * coupling * delta_corr_real,
        ],
        dtype=float,
    )
    return np.concatenate([derivative, anomalous_derivative])


def finite_difference_jacobian(
    rhs: RhsFunction,
    time: float,
    state: FloatArray,
    *,
    relative_step: float = 1e-7,
) -> FloatArray:
    """Return a central finite-difference Jacobian of ``rhs``."""

    point = np.asarray(state, dtype=float)
    if point.ndim != 1:
        raise ValueError("state must be one-dimensional")
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")

    jacobian = np.empty((point.size, point.size), dtype=float)
    for column in range(point.size):
        step = relative_step * max(1.0, abs(float(point[column])))
        offset = np.zeros_like(point)
        offset[column] = step
        jacobian[:, column] = (
            rhs(time, point + offset) - rhs(time, point - offset)
        ) / (2.0 * step)
    return jacobian


def integrate_rk4(
    rhs: RhsFunction,
    initial_state: FloatArray,
    *,
    final_time: float,
    time_step: float,
    failure_threshold: float = 1e4,
    state_names: tuple[str, ...] | None = None,
) -> IntegrationSummary:
    """Integrate with RK4 and stop at the first declared failure threshold."""

    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if failure_threshold <= 0.0:
        raise ValueError("failure_threshold must be positive")

    state = np.asarray(initial_state, dtype=float).copy()
    if state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")
    if state_names is not None and len(state_names) != state.size:
        raise ValueError("state_names length must match initial_state")

    time = 0.0
    steps = 0
    max_abs_state = float(np.max(np.abs(state)))
    failure_time: float | None = None
    failure_component: str | None = None

    while time < final_time:
        step = min(time_step, final_time - time)
        k1 = rhs(time, state)
        k2 = rhs(time + 0.5 * step, state + 0.5 * step * k1)
        k3 = rhs(time + 0.5 * step, state + 0.5 * step * k2)
        k4 = rhs(time + step, state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time += step
        steps += 1

        component = int(np.argmax(np.abs(state)))
        current_max = float(abs(state[component]))
        max_abs_state = max(max_abs_state, current_max)
        if not np.all(np.isfinite(state)) or current_max > failure_threshold:
            failure_time = time
            failure_component = (
                state_names[component] if state_names is not None else str(component)
            )
            break

    return IntegrationSummary(
        final_time=time,
        final_state=state,
        max_abs_state=max_abs_state,
        steps=steps,
        failure_time=failure_time,
        failure_component=failure_component,
    )
