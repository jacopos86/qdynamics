r"""Order-parameterized Pauli--Weyl cumulant hierarchy for the Holstein dimer.

The fixed one-up/one-down sector is represented by two site qubits and the
interacting relative phonon quadratures ``[x, p] = i``.  A hierarchy of order
``r`` retains every spin-exchange-symmetric Hermitian Weyl moment through
total degree ``r`` and reconstructs generated degree-``r+1`` moments by
setting their connected cumulant to zero.  The decoupled center phonon adds a
complex coherent amplitude to the real moment coordinates.

``MomentHierarchy`` is the public interface.  It owns basis generation,
coordinate packing, Pauli/Moyal algebra, terminal cumulant reconstruction,
the autonomous velocity, and contraction into the earlier matrix variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb, factorial
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

import numpy as np

from .hubbard_dimer import DimerParameters, FloatArray, _require_state
from .matrix_reference import MatrixDimerState

IDENTITY = "I"
PAULI_X = "X"
PAULI_Y = "Y"
PAULI_Z = "Z"
PAULI_LABELS = (IDENTITY, PAULI_X, PAULI_Y, PAULI_Z)
_PAULI_ORDER = {label: index for index, label in enumerate(PAULI_LABELS)}


@dataclass(frozen=True, order=True)
class MomentKey:
    """One Hermitian spin--boson Weyl moment."""

    spin_up: str
    spin_down: str
    x_power: int
    p_power: int

    def __post_init__(self) -> None:
        if self.spin_up not in _PAULI_ORDER:
            raise ValueError(f"unknown spin_up Pauli label {self.spin_up!r}")
        if self.spin_down not in _PAULI_ORDER:
            raise ValueError(
                f"unknown spin_down Pauli label {self.spin_down!r}"
            )
        if self.x_power < 0 or self.p_power < 0:
            raise ValueError("Weyl exponents must be nonnegative")
        if _PAULI_ORDER[self.spin_up] > _PAULI_ORDER[self.spin_down]:
            raise ValueError("MomentKey spin labels must be canonical")

    @property
    def degree(self) -> int:
        return (
            int(self.spin_up != IDENTITY)
            + int(self.spin_down != IDENTITY)
            + self.x_power
            + self.p_power
        )


@dataclass(frozen=True)
class _OperatorKey:
    spin_up: str
    spin_down: str
    x_power: int
    p_power: int


_KeyLike = MomentKey | _OperatorKey
_PrimitiveToken = tuple[str, str]


class PreparedTerminalMomentClosure(Protocol):
    """State-local resolver for retained and generated moments."""

    def moment(self, key: MomentKey) -> float:
        """Return one retained or terminal reconstructed moment."""


class TerminalMomentClosure(Protocol):
    """Prepare one autonomous terminal-moment reconstruction."""

    name: str

    def prepare(
        self,
        moments: Mapping[MomentKey, float],
        maximum_degree: int,
    ) -> PreparedTerminalMomentClosure:
        """Bind the closure to one retained moment state."""


def _canonical_key(
    spin_up: str,
    spin_down: str,
    x_power: int,
    p_power: int,
) -> MomentKey:
    if _PAULI_ORDER[spin_up] > _PAULI_ORDER[spin_down]:
        spin_up, spin_down = spin_down, spin_up
    return MomentKey(spin_up, spin_down, x_power, p_power)


@lru_cache(maxsize=None)
def build_moment_keys(maximum_degree: int) -> tuple[MomentKey, ...]:
    """Return the complete symmetry-adapted basis through ``maximum_degree``."""

    if maximum_degree < 2:
        raise ValueError("maximum_degree must be at least two")
    keys: list[MomentKey] = []
    for total_degree in range(1, maximum_degree + 1):
        for spin_up_index, spin_up in enumerate(PAULI_LABELS):
            for spin_down in PAULI_LABELS[spin_up_index:]:
                spin_degree = int(spin_up != IDENTITY) + int(
                    spin_down != IDENTITY
                )
                boson_degree = total_degree - spin_degree
                if boson_degree < 0:
                    continue
                for x_power in range(boson_degree, -1, -1):
                    keys.append(
                        MomentKey(
                            spin_up,
                            spin_down,
                            x_power,
                            boson_degree - x_power,
                        )
                    )
    return tuple(keys)


def _moment_name(key: MomentKey) -> str:
    return (
        f"moment_{key.spin_up.lower()}{key.spin_down.lower()}"
        f"_x{key.x_power}_p{key.p_power}"
    )


def _falling_factorial(value: int, order: int) -> int:
    if order > value:
        return 0
    return factorial(value) // factorial(value - order)


@lru_cache(maxsize=None)
def _weyl_product(
    left_x: int,
    left_p: int,
    right_x: int,
    right_p: int,
) -> tuple[tuple[int, int, complex], ...]:
    """Multiply Weyl monomials with the exact Moyal star product."""

    terms: dict[tuple[int, int], complex] = {}
    maximum_order = left_x + left_p + right_x + right_p
    for order in range(maximum_order + 1):
        for p_derivatives_left in range(order + 1):
            x_derivatives_left = order - p_derivatives_left
            x_derivatives_right = p_derivatives_left
            p_derivatives_right = order - p_derivatives_left
            if (
                x_derivatives_left > left_x
                or p_derivatives_left > left_p
                or x_derivatives_right > right_x
                or p_derivatives_right > right_p
            ):
                continue
            coefficient = (
                (0.5j) ** order
                / factorial(order)
                * (-1) ** p_derivatives_left
                * comb(order, p_derivatives_left)
                * _falling_factorial(left_x, x_derivatives_left)
                * _falling_factorial(left_p, p_derivatives_left)
                * _falling_factorial(right_x, x_derivatives_right)
                * _falling_factorial(right_p, p_derivatives_right)
            )
            powers = (
                left_x
                + right_x
                - x_derivatives_left
                - x_derivatives_right,
                left_p
                + right_p
                - p_derivatives_left
                - p_derivatives_right,
            )
            terms[powers] = terms.get(powers, 0.0j) + coefficient
    return tuple(
        (x_power, p_power, coefficient)
        for (x_power, p_power), coefficient in sorted(terms.items())
        if abs(coefficient) > 1e-15
    )


def _pauli_product(left: str, right: str) -> tuple[str, complex]:
    if left == IDENTITY:
        return right, 1.0 + 0.0j
    if right == IDENTITY:
        return left, 1.0 + 0.0j
    if left == right:
        return IDENTITY, 1.0 + 0.0j
    cyclic = {
        (PAULI_X, PAULI_Y): PAULI_Z,
        (PAULI_Y, PAULI_Z): PAULI_X,
        (PAULI_Z, PAULI_X): PAULI_Y,
    }
    if (left, right) in cyclic:
        return cyclic[(left, right)], 1.0j
    return cyclic[(right, left)], -1.0j


def _operator_product(
    left: _KeyLike,
    right: _KeyLike,
) -> dict[MomentKey, complex]:
    spin_up, coefficient_up = _pauli_product(left.spin_up, right.spin_up)
    spin_down, coefficient_down = _pauli_product(
        left.spin_down, right.spin_down
    )
    terms: dict[MomentKey, complex] = {}
    for x_power, p_power, boson_coefficient in _weyl_product(
        left.x_power,
        left.p_power,
        right.x_power,
        right.p_power,
    ):
        key = _canonical_key(spin_up, spin_down, x_power, p_power)
        coefficient = coefficient_up * coefficient_down * boson_coefficient
        terms[key] = terms.get(key, 0.0j) + coefficient
    return terms


def _commutator(
    left: _KeyLike,
    right: _KeyLike,
) -> dict[MomentKey, complex]:
    terms = _operator_product(left, right)
    for key, coefficient in _operator_product(right, left).items():
        terms[key] = terms.get(key, 0.0j) - coefficient
    return {
        key: coefficient
        for key, coefficient in terms.items()
        if abs(coefficient) > 1e-14
    }


def _tokens_for_key(key: MomentKey) -> tuple[_PrimitiveToken, ...]:
    tokens: list[_PrimitiveToken] = []
    if key.spin_up != IDENTITY:
        tokens.append(("up", key.spin_up))
    if key.spin_down != IDENTITY:
        tokens.append(("down", key.spin_down))
    tokens.extend(("x", "") for _ in range(key.x_power))
    tokens.extend(("p", "") for _ in range(key.p_power))
    return tuple(tokens)


def _key_for_tokens(tokens: Iterable[_PrimitiveToken]) -> MomentKey:
    spin_up = IDENTITY
    spin_down = IDENTITY
    x_power = 0
    p_power = 0
    for subsystem, label in tokens:
        if subsystem == "up":
            if spin_up != IDENTITY:
                raise ValueError("a cumulant block contains two up-spin factors")
            spin_up = label
        elif subsystem == "down":
            if spin_down != IDENTITY:
                raise ValueError(
                    "a cumulant block contains two down-spin factors"
                )
            spin_down = label
        elif subsystem == "x":
            x_power += 1
        elif subsystem == "p":
            p_power += 1
        else:  # pragma: no cover
            raise ValueError(f"unknown primitive subsystem {subsystem!r}")
    return _canonical_key(spin_up, spin_down, x_power, p_power)


@lru_cache(maxsize=None)
def _set_partitions(size: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if size < 1:
        return ((),)
    partitions: list[tuple[tuple[int, ...], ...]] = [((0,),)]
    for item in range(1, size):
        expanded: list[tuple[tuple[int, ...], ...]] = []
        for partition in partitions:
            expanded.append((*partition, (item,)))
            for block_index in range(len(partition)):
                blocks = list(partition)
                blocks[block_index] = (*blocks[block_index], item)
                expanded.append(tuple(blocks))
        partitions = expanded
    return tuple(partitions)


def _raw_retained_moment(
    tokens: tuple[_PrimitiveToken, ...],
    moments: Mapping[MomentKey, float],
) -> float:
    if not tokens:
        return 1.0
    key = _key_for_tokens(tokens)
    try:
        return float(moments[key])
    except KeyError as error:  # pragma: no cover
        raise ValueError(f"required retained moment is missing: {key}") from error


def _cumulant(
    tokens: tuple[_PrimitiveToken, ...],
    moments: Mapping[MomentKey, float],
    cache: dict[tuple[_PrimitiveToken, ...], float],
) -> float:
    if tokens in cache:
        return cache[tokens]
    value = _raw_retained_moment(tokens, moments)
    for partition in _set_partitions(len(tokens)):
        if len(partition) == 1:
            continue
        product = 1.0
        for block in partition:
            block_tokens = tuple(tokens[index] for index in block)
            product *= _cumulant(block_tokens, moments, cache)
        value -= product
    cache[tokens] = value
    return value


def _zero_terminal_cumulant_moment(
    key: MomentKey,
    moments: Mapping[MomentKey, float],
    maximum_degree: int,
    cumulant_cache: dict[tuple[_PrimitiveToken, ...], float] | None = None,
) -> float:
    if key.degree == 0:
        return 1.0
    if key.degree <= maximum_degree:
        return float(moments[key])
    if key.degree != maximum_degree + 1:
        raise ValueError(
            f"order-{maximum_degree} hierarchy generated unsupported "
            f"degree-{key.degree} moment: {key}"
        )
    tokens = _tokens_for_key(key)
    cache = {} if cumulant_cache is None else cumulant_cache
    reconstructed = 0.0
    for partition in _set_partitions(len(tokens)):
        if len(partition) == 1:
            continue
        product = 1.0
        for block in partition:
            block_tokens = tuple(tokens[index] for index in block)
            product *= _cumulant(block_tokens, moments, cache)
        reconstructed += product
    return reconstructed


@dataclass
class _ZeroCumulantResolver:
    moments: Mapping[MomentKey, float]
    maximum_degree: int
    cumulant_cache: dict[tuple[_PrimitiveToken, ...], float] = field(
        default_factory=dict
    )

    def moment(self, key: MomentKey) -> float:
        return _zero_terminal_cumulant_moment(
            key,
            self.moments,
            self.maximum_degree,
            self.cumulant_cache,
        )


@dataclass(frozen=True)
class ZeroCumulantClosure:
    """Set the first omitted connected cumulant to zero."""

    name: str = "zero_cumulant"

    def prepare(
        self,
        moments: Mapping[MomentKey, float],
        maximum_degree: int,
    ) -> PreparedTerminalMomentClosure:
        return _ZeroCumulantResolver(moments, maximum_degree)


ZERO_CUMULANT_CLOSURE = ZeroCumulantClosure()


def _hamiltonian_terms(
    time: float,
    parameters: DimerParameters,
) -> tuple[tuple[float, _OperatorKey], ...]:
    hopping = parameters.hopping
    drive_half = 0.5 * parameters.drive_difference(time)
    omega_half = 0.5 * parameters.omega_ph
    coupling = parameters.coupling
    return (
        (-hopping, _OperatorKey(PAULI_X, IDENTITY, 0, 0)),
        (-hopping, _OperatorKey(IDENTITY, PAULI_X, 0, 0)),
        (drive_half, _OperatorKey(PAULI_Z, IDENTITY, 0, 0)),
        (drive_half, _OperatorKey(IDENTITY, PAULI_Z, 0, 0)),
        (omega_half, _OperatorKey(IDENTITY, IDENTITY, 2, 0)),
        (omega_half, _OperatorKey(IDENTITY, IDENTITY, 0, 2)),
        (coupling, _OperatorKey(PAULI_Z, IDENTITY, 1, 0)),
        (coupling, _OperatorKey(IDENTITY, PAULI_Z, 1, 0)),
    )


@dataclass(frozen=True)
class MomentHierarchy:
    """Complete symmetry-adapted moment hierarchy through one total degree."""

    maximum_degree: int
    moment_keys: tuple[MomentKey, ...] = field(init=False)
    state_names: tuple[str, ...] = field(init=False)
    _moment_index: Mapping[MomentKey, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        keys = build_moment_keys(self.maximum_degree)
        names = (
            "center_phonon_real",
            "center_phonon_imag",
            *tuple(_moment_name(key) for key in keys),
        )
        index = MappingProxyType(
            {key: position + 2 for position, key in enumerate(keys)}
        )
        object.__setattr__(self, "moment_keys", keys)
        object.__setattr__(self, "state_names", names)
        object.__setattr__(self, "_moment_index", index)

    @property
    def coordinate_count(self) -> int:
        return len(self.state_names)

    def pack(
        self,
        center_amplitude: complex,
        moments: Mapping[MomentKey, float],
    ) -> FloatArray:
        missing = set(self.moment_keys).difference(moments)
        extra = set(moments).difference(self.moment_keys)
        if missing or extra:
            raise ValueError(
                f"moment mapping mismatch: {len(missing)} missing, "
                f"{len(extra)} extra"
            )
        center = complex(center_amplitude)
        values = [center.real, center.imag]
        values.extend(float(moments[key]) for key in self.moment_keys)
        return np.asarray(values, dtype=float)

    def unpack(
        self,
        state: FloatArray,
    ) -> tuple[complex, dict[MomentKey, float]]:
        array = _require_state(state, self.state_names)
        center = complex(array[0], array[1])
        moments = {
            key: float(array[index])
            for key, index in self._moment_index.items()
        }
        return center, moments

    def moment_value(self, state: FloatArray, key: MomentKey) -> float:
        array = _require_state(state, self.state_names)
        try:
            return float(array[self._moment_index[key]])
        except KeyError as error:
            raise ValueError(f"moment {key} is not retained") from error

    def closed_moment(
        self,
        key: MomentKey,
        moments: Mapping[MomentKey, float],
        *,
        closure: TerminalMomentClosure | None = None,
    ) -> float:
        selected_closure = (
            ZERO_CUMULANT_CLOSURE if closure is None else closure
        )
        return selected_closure.prepare(
            moments,
            self.maximum_degree,
        ).moment(key)

    def rhs(
        self,
        time: float,
        state: FloatArray,
        parameters: DimerParameters,
        *,
        closure: TerminalMomentClosure | None = None,
    ) -> FloatArray:
        center, moments = self.unpack(state)
        selected_closure = (
            ZERO_CUMULANT_CLOSURE if closure is None else closure
        )
        closure_resolver = selected_closure.prepare(
            moments,
            self.maximum_degree,
        )
        center_velocity = -1j * (
            parameters.omega_ph * center
            + np.sqrt(2.0) * parameters.coupling
        )
        derivatives: dict[MomentKey, float] = {}
        hamiltonian = _hamiltonian_terms(time, parameters)
        generated_cache: dict[MomentKey, float] = {}
        for observable in self.moment_keys:
            derivative = 0.0j
            for hamiltonian_coefficient, hamiltonian_key in hamiltonian:
                for generated_key, commutator_coefficient in _commutator(
                    hamiltonian_key,
                    observable,
                ).items():
                    if generated_key not in generated_cache:
                        generated_cache[generated_key] = closure_resolver.moment(
                            generated_key
                        )
                    derivative += (
                        1j
                        * hamiltonian_coefficient
                        * commutator_coefficient
                        * generated_cache[generated_key]
                    )
            if abs(derivative.imag) > 5e-11:
                raise FloatingPointError(
                    "Hermitian moment acquired a complex velocity: "
                    f"{observable} -> {derivative}"
                )
            derivatives[observable] = float(derivative.real)
        return self.pack(center_velocity, derivatives)

    def to_matrix_state(self, state: FloatArray) -> MatrixDimerState:
        """Contract hierarchy coordinates into ``(rho, B, N, A, C)``."""

        center, moments = self.unpack(state)
        return _matrix_state_from_moments(center, moments)

    def matrix_derivative(
        self,
        state: FloatArray,
        derivative: FloatArray,
    ) -> MatrixDimerState:
        """Apply the differential of :meth:`to_matrix_state`."""

        _, moments = self.unpack(state)
        center_derivative, derivative_moments = self.unpack(derivative)
        return _matrix_derivative_from_moments(
            moments,
            center_derivative,
            derivative_moments,
        )

    def energy(
        self,
        time: float,
        state: FloatArray,
        parameters: DimerParameters,
    ) -> float:
        """Return the instantaneous transformed-Hamiltonian expectation."""

        center, moments = self.unpack(state)
        spin_x = _one_spin_value(moments, PAULI_X)
        spin_z = _one_spin_value(moments, PAULI_Z)
        spin_z_x = _one_spin_value(moments, PAULI_Z, 1, 0)
        relative_energy = 0.5 * parameters.omega_ph * (
            _boson_value(moments, 2, 0)
            + _boson_value(moments, 0, 2)
            - 1.0
        )
        center_energy = (
            parameters.omega_ph * abs(center) ** 2
            + 2.0
            * np.sqrt(2.0)
            * parameters.coupling
            * center.real
        )
        return float(
            -2.0 * parameters.hopping * spin_x
            + parameters.drive_difference(time) * spin_z
            + relative_energy
            + 2.0 * parameters.coupling * spin_z_x
            + center_energy
        )


def _one_spin_value(
    moments: Mapping[MomentKey, float],
    pauli: str,
    x_power: int = 0,
    p_power: int = 0,
) -> float:
    return float(
        moments[_canonical_key(IDENTITY, pauli, x_power, p_power)]
    )


def _boson_value(
    moments: Mapping[MomentKey, float],
    x_power: int,
    p_power: int,
) -> float:
    return float(
        moments[MomentKey(IDENTITY, IDENTITY, x_power, p_power)]
    )


def _electron_density(
    moments: Mapping[MomentKey, float],
) -> np.ndarray:
    spin_x = _one_spin_value(moments, PAULI_X)
    spin_y = _one_spin_value(moments, PAULI_Y)
    spin_z = _one_spin_value(moments, PAULI_Z)
    return 0.5 * np.array(
        [
            [1.0 + spin_z, spin_x - 1j * spin_y],
            [spin_x + 1j * spin_y, 1.0 - spin_z],
        ],
        dtype=complex,
    )


def _electron_density_derivative(
    derivative_moments: Mapping[MomentKey, float],
) -> np.ndarray:
    spin_x = _one_spin_value(derivative_moments, PAULI_X)
    spin_y = _one_spin_value(derivative_moments, PAULI_Y)
    spin_z = _one_spin_value(derivative_moments, PAULI_Z)
    return 0.5 * np.array(
        [
            [spin_z, spin_x - 1j * spin_y],
            [spin_x + 1j * spin_y, -spin_z],
        ],
        dtype=complex,
    )


def _spin_boson_covariances(
    moments: Mapping[MomentKey, float],
) -> dict[str, tuple[float, float]]:
    mean_x = _boson_value(moments, 1, 0)
    mean_p = _boson_value(moments, 0, 1)
    return {
        pauli: (
            _one_spin_value(moments, pauli, 1, 0)
            - _one_spin_value(moments, pauli) * mean_x,
            _one_spin_value(moments, pauli, 0, 1)
            - _one_spin_value(moments, pauli) * mean_p,
        )
        for pauli in (PAULI_X, PAULI_Y, PAULI_Z)
    }


def _spin_boson_covariance_derivatives(
    moments: Mapping[MomentKey, float],
    derivative_moments: Mapping[MomentKey, float],
) -> dict[str, tuple[float, float]]:
    mean_x = _boson_value(moments, 1, 0)
    mean_p = _boson_value(moments, 0, 1)
    mean_x_derivative = _boson_value(derivative_moments, 1, 0)
    mean_p_derivative = _boson_value(derivative_moments, 0, 1)
    derivatives: dict[str, tuple[float, float]] = {}
    for pauli in (PAULI_X, PAULI_Y, PAULI_Z):
        spin = _one_spin_value(moments, pauli)
        spin_derivative = _one_spin_value(derivative_moments, pauli)
        derivatives[pauli] = (
            _one_spin_value(derivative_moments, pauli, 1, 0)
            - spin_derivative * mean_x
            - spin * mean_x_derivative,
            _one_spin_value(derivative_moments, pauli, 0, 1)
            - spin_derivative * mean_p
            - spin * mean_p_derivative,
        )
    return derivatives


def _local_correlations(
    covariances: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    cov_x_x, cov_x_p = covariances[PAULI_X]
    cov_y_x, cov_y_p = covariances[PAULI_Y]
    cov_z_x, cov_z_p = covariances[PAULI_Z]
    relative_correlation = np.array(
        [
            [
                0.25 * (cov_z_x + 1j * cov_z_p),
                0.25
                * (
                    cov_x_x
                    + cov_y_p
                    + 1j * (cov_x_p - cov_y_x)
                ),
            ],
            [
                0.25
                * (
                    cov_x_x
                    - cov_y_p
                    + 1j * (cov_x_p + cov_y_x)
                ),
                -0.25 * (cov_z_x + 1j * cov_z_p),
            ],
        ],
        dtype=complex,
    )
    return np.stack([relative_correlation, -relative_correlation])


def _matrix_state_from_moments(
    center: complex,
    moments: Mapping[MomentKey, float],
) -> MatrixDimerState:
    mean_x = _boson_value(moments, 1, 0)
    mean_p = _boson_value(moments, 0, 1)
    relative_amplitude = complex(mean_x, mean_p) / np.sqrt(2.0)
    coherent_phonon = np.array(
        [
            (center + relative_amplitude) / np.sqrt(2.0),
            (center - relative_amplitude) / np.sqrt(2.0),
        ],
        dtype=complex,
    )
    covariance_xx = _boson_value(moments, 2, 0) - mean_x**2
    covariance_pp = _boson_value(moments, 0, 2) - mean_p**2
    covariance_xp = _boson_value(moments, 1, 1) - mean_x * mean_p
    relative_population = 0.5 * (
        covariance_xx + covariance_pp - 1.0
    )
    relative_anomalous = 0.5 * (
        covariance_xx - covariance_pp + 2j * covariance_xp
    )
    relative_projector = 0.5 * np.array(
        [[1.0, -1.0], [-1.0, 1.0]], dtype=complex
    )
    return MatrixDimerState(
        electron_density=_electron_density(moments),
        coherent_phonon=coherent_phonon,
        phonon_density=relative_population * relative_projector,
        anomalous_phonon_density=(
            relative_anomalous * relative_projector
        ),
        electron_phonon_correlation=_local_correlations(
            _spin_boson_covariances(moments)
        ),
    )


def _matrix_derivative_from_moments(
    moments: Mapping[MomentKey, float],
    center_derivative: complex,
    derivative_moments: Mapping[MomentKey, float],
) -> MatrixDimerState:
    mean_x = _boson_value(moments, 1, 0)
    mean_p = _boson_value(moments, 0, 1)
    mean_x_derivative = _boson_value(derivative_moments, 1, 0)
    mean_p_derivative = _boson_value(derivative_moments, 0, 1)
    relative_amplitude_derivative = complex(
        mean_x_derivative, mean_p_derivative
    ) / np.sqrt(2.0)
    coherent_derivative = np.array(
        [
            (center_derivative + relative_amplitude_derivative)
            / np.sqrt(2.0),
            (center_derivative - relative_amplitude_derivative)
            / np.sqrt(2.0),
        ],
        dtype=complex,
    )
    covariance_xx_derivative = (
        _boson_value(derivative_moments, 2, 0)
        - 2.0 * mean_x * mean_x_derivative
    )
    covariance_pp_derivative = (
        _boson_value(derivative_moments, 0, 2)
        - 2.0 * mean_p * mean_p_derivative
    )
    covariance_xp_derivative = (
        _boson_value(derivative_moments, 1, 1)
        - mean_x_derivative * mean_p
        - mean_x * mean_p_derivative
    )
    relative_population_derivative = 0.5 * (
        covariance_xx_derivative + covariance_pp_derivative
    )
    relative_anomalous_derivative = 0.5 * (
        covariance_xx_derivative
        - covariance_pp_derivative
        + 2j * covariance_xp_derivative
    )
    relative_projector = 0.5 * np.array(
        [[1.0, -1.0], [-1.0, 1.0]], dtype=complex
    )
    return MatrixDimerState(
        electron_density=_electron_density_derivative(derivative_moments),
        coherent_phonon=coherent_derivative,
        phonon_density=relative_population_derivative * relative_projector,
        anomalous_phonon_density=(
            relative_anomalous_derivative * relative_projector
        ),
        electron_phonon_correlation=_local_correlations(
            _spin_boson_covariance_derivatives(
                moments,
                derivative_moments,
            )
        ),
    )


@lru_cache(maxsize=None)
def moment_hierarchy(maximum_degree: int) -> MomentHierarchy:
    """Return a cached hierarchy instance for one retained degree."""

    return MomentHierarchy(maximum_degree)


THIRD_ORDER_HIERARCHY = moment_hierarchy(3)
FOURTH_ORDER_HIERARCHY = moment_hierarchy(4)
