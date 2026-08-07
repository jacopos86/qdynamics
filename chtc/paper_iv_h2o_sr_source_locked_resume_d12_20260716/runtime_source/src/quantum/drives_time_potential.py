"""Time-dependent onsite density drive for Suzuki-Trotter evolution.

Physics
-------
The drive adds a time-dependent onsite potential to the Hubbard Hamiltonian:

    H_drive(t) = Σ_{i,σ} v_i(t) n_{iσ}

where  n_{iσ} = (I − Z_{q(i,σ)}) / 2  under Jordan-Wigner.

This module provides:

* ``gaussian_sinusoid_waveform``  –  v(t) = A sin(ωt+φ) exp(−t²/(2t̄²))
* ``GaussianSinusoidSitePotential``  –  v_i(t) = s_i · v(t)
* ``DensityDriveTemplate``  –  precomputed JW Z-labels for every (site, spin)
* ``TimeDependentOnsiteDensityDrive``  –  runtime callable returning
  ``{label_exyz: Δcoeff}`` at each Trotter slice
* ``build_gaussian_sinusoid_density_drive``  –  convenience builder

The ``site_potential`` field is a generic ``Callable[[float], np.ndarray]``,
so alternative waveforms can be plugged in without changing the rest of the
module.

Reference-propagator approximation order
-----------------------------------------
The pipelines use a piecewise-constant matrix-exponential propagator as the
"exact" reference for time-dependent fidelity computations.  The method is
**not** a true time-ordered exponential; its order depends on the sampling
rule chosen for the sub-interval representative time ``t_k``:

``time_sampling="midpoint"``  (default)
    Samples at the midpoint of each slice:  t_k = t₀ + (k + ½)Δt.
    This is the **exponential midpoint / Magnus-2** integrator.  For a
    non-autonomous linear ODE dψ/dt = −i H(t) ψ, the leading local error is
    O(Δt³) and the global error is **O(Δt²)** (second-order convergence).
    The JSON metadata field ``reference_method`` records
    ``"exponential_midpoint_magnus2_order2"``.

``time_sampling="left"``
    Samples at the left endpoint:  t_k = t₀ + k Δt.
    This is a first-order (Euler) exponential integrator.  Global error is
    **O(Δt)**.  The JSON metadata field records
    ``"exponential_left_endpoint_order1"``.

``time_sampling="right"``
    Samples at the right endpoint:  t_k = t₀ + (k + 1) Δt.
    Also first-order.  The JSON metadata field records
    ``"exponential_right_endpoint_order1"``.

The ``--exact-steps-multiplier M`` pipeline flag refines the reference by
running it at N_ref = M × trotter_steps while keeping the Trotter circuit at
trotter_steps.  With midpoint sampling this strictly improves reference
quality because O(Δt²) → O((Δt/M)²).  The multiplier is recorded in the
JSON metadata as ``reference_steps_multiplier``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index


# ---------------------------------------------------------------------------
# Approximation-order metadata
# ---------------------------------------------------------------------------

#: Maps the ``time_sampling`` choice to a descriptive method name for JSON
#: output.  Helps downstream readers distinguish the approximation order from
#: a true time-ordered exponential.
REFERENCE_METHOD_NAMES: Dict[str, str] = {
    "midpoint": "exponential_midpoint_magnus2_order2",
    "left":     "exponential_left_endpoint_order1",
    "right":    "exponential_right_endpoint_order1",
}


def reference_method_name(time_sampling: str) -> str:
    """Return the canonical method name string for *time_sampling*.

    Parameters
    ----------
    time_sampling : str
        One of ``"midpoint"``, ``"left"``, ``"right"``.

    Returns
    -------
    str
        A descriptive string for the JSON ``reference_method`` field:

        * ``"midpoint"`` → ``"exponential_midpoint_magnus2_order2"``
          (second-order; equivalent to one step of the Magnus-2 / exponential
          midpoint integrator)
        * ``"left"``     → ``"exponential_left_endpoint_order1"``
        * ``"right"``    → ``"exponential_right_endpoint_order1"``

    Raises
    ------
    ValueError
        If *time_sampling* is not one of the supported values.
    """
    key = str(time_sampling).strip().lower()
    try:
        return REFERENCE_METHOD_NAMES[key]
    except KeyError:
        raise ValueError(
            f"time_sampling must be one of {set(REFERENCE_METHOD_NAMES)}, got {key!r}"
        )


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def gaussian_sinusoid_waveform(
    t: float,
    *,
    A: float,
    omega: float,
    tbar: float,
    phi: float = 0.0,
) -> float:
    """v(t) = A · sin(ω t + φ) · exp(−t² / (2 t̄²))."""
    t_f = float(t)
    tbar_f = float(tbar)
    if tbar_f <= 0.0:
        raise ValueError("tbar must be > 0")
    arg = float(omega) * t_f + float(phi)
    env = math.exp(-(t_f * t_f) / (2.0 * tbar_f * tbar_f))
    return float(A) * math.sin(arg) * env


def evaluate_drive_waveform(
    times: Sequence[float] | np.ndarray,
    drive_config: Dict[str, float | int | str | None],
    amplitude: float,
) -> np.ndarray:
    """Evaluate scalar drive coefficients on a time grid used by the report.

    The returned waveform is the same scalar ``f(t)`` that multiplies the
    drive operator in the Hamiltonian, with ``t0`` applied exactly as in the
    evolution kernels (sampled at ``t + t0``).
    """
    arr = np.asarray(times, dtype=float)
    shape = arr.shape
    flat_t = arr.reshape(-1)

    omega = float(drive_config.get("drive_omega", drive_config.get("omega", 1.0)))
    tbar = float(drive_config.get("drive_tbar", drive_config.get("tbar", 1.0)))
    phi = float(drive_config.get("drive_phi", drive_config.get("phi", 0.0)))
    t0 = float(drive_config.get("drive_t0", drive_config.get("t0", 0.0)))

    vals = np.array(
        [
            gaussian_sinusoid_waveform(
                float(t) + t0,
                A=float(amplitude),
                omega=omega,
                tbar=tbar,
                phi=phi,
            )
            for t in flat_t
        ],
        dtype=float,
    )
    return vals.reshape(shape)


# ---------------------------------------------------------------------------
# Spatial weights
# ---------------------------------------------------------------------------

def default_spatial_weights(
    n_sites: int,
    *,
    mode: str,
    custom: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return spatial weights s_i for each site.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    mode : str
        ``'dimer_bias'``  – requires ``n_sites == 2``, returns ``[+1, −1]``
        ``'staggered'``   – returns ``[(-1)^i]``
        ``'custom'``      – uses the provided *custom* sequence of length *n_sites*
    custom : sequence of float, optional
        Required when *mode* is ``'custom'``.
    """
    n = int(n_sites)
    if n <= 0:
        raise ValueError("n_sites must be positive")
    mode_n = str(mode).strip().lower()
    if mode_n == "dimer_bias":
        if n != 2:
            raise ValueError("pattern_mode='dimer_bias' requires n_sites == 2")
        return np.array([1.0, -1.0], dtype=float)
    if mode_n == "staggered":
        return np.array(
            [1.0 if (i % 2 == 0) else -1.0 for i in range(n)],
            dtype=float,
        )
    if mode_n == "custom":
        if custom is None:
            raise ValueError("custom spatial weights required when mode='custom'")
        if len(custom) != n:
            raise ValueError("custom spatial weights must have length n_sites")
        return np.array([float(x) for x in custom], dtype=float)
    raise ValueError(
        f"pattern_mode must be one of {{'dimer_bias','staggered','custom'}}, "
        f"got {mode_n!r}"
    )


# ---------------------------------------------------------------------------
# Site-resolved potential dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianSinusoidSitePotential:
    """Site-resolved potential v_i(t) = s_i · v(t)."""

    weights: np.ndarray  # length n_sites
    A: float
    omega: float
    tbar: float
    phi: float = 0.0

    def v_scalar(self, t: float) -> float:
        """Evaluate the scalar waveform v(t)."""
        return gaussian_sinusoid_waveform(
            t,
            A=float(self.A),
            omega=float(self.omega),
            tbar=float(self.tbar),
            phi=float(self.phi),
        )

    def v_sites(self, t: float) -> np.ndarray:
        """Return array of v_i(t) for each site."""
        v = self.v_scalar(t)
        return np.asarray(self.weights, dtype=float) * float(v)


# ---------------------------------------------------------------------------
# JW density-drive Z-label template
# ---------------------------------------------------------------------------

def _z_label_exyz(*, nq_total: int, qubit_index: int) -> str:
    """Build an exyz-format Pauli label with Z on *qubit_index* (0 = rightmost)."""
    nq = int(nq_total)
    q = int(qubit_index)
    if nq <= 0:
        raise ValueError("nq_total must be positive")
    if q < 0 or q >= nq:
        raise ValueError(f"qubit_index {q} out of range [0, {nq})")
    z_pos = nq - 1 - q
    return ("e" * z_pos) + "z" + ("e" * (nq - 1 - z_pos))


@dataclass(frozen=True)
class DensityDriveTemplate:
    """Precomputed JW density-drive Z labels (Pauli words are static)."""

    n_sites: int
    nq_total: int
    indexing: str
    electron_qubit_offset: int
    z_labels_site_spin: Dict[tuple, str]  # (site, spin) -> label
    ordered_z_labels: List[str]
    identity_label: str

    @classmethod
    def build(
        cls,
        *,
        n_sites: int,
        nq_total: int,
        indexing: str,
        electron_qubit_offset: int = 0,
    ) -> "DensityDriveTemplate":
        n = int(n_sites)
        nq = int(nq_total)
        if nq <= 0:
            raise ValueError("nq_total must be positive")
        if n <= 0:
            raise ValueError("n_sites must be positive")
        if 2 * n + int(electron_qubit_offset) > nq:
            raise ValueError("electron register does not fit inside nq_total")

        z_map: Dict[tuple, str] = {}
        ordered: List[str] = []
        for i in range(n):
            for spin in (SPIN_UP, SPIN_DN):
                p_mode = mode_index(i, spin, indexing=str(indexing), n_sites=n)
                q = int(electron_qubit_offset) + int(p_mode)
                lbl = _z_label_exyz(nq_total=nq, qubit_index=q)
                z_map[(i, int(spin))] = lbl
                ordered.append(lbl)

        return cls(
            n_sites=n,
            nq_total=nq,
            indexing=str(indexing),
            electron_qubit_offset=int(electron_qubit_offset),
            z_labels_site_spin=z_map,
            ordered_z_labels=ordered,
            identity_label="e" * nq,
        )

    def labels_exyz(self, *, include_identity: bool = False) -> List[str]:
        """Return all drive Pauli labels (Z terms, optionally identity)."""
        out = list(self.ordered_z_labels)
        if bool(include_identity):
            out.append(self.identity_label)
        return out

    def z_label(self, site: int, spin: int) -> str:
        """Return Z-label for (site, spin)."""
        return self.z_labels_site_spin[(int(site), int(spin))]


# ---------------------------------------------------------------------------
# Runtime drive object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDependentOnsiteDensityDrive:
    """Drive providing additive Pauli coefficients Δc(label, t).

    Mapping:  n_{iσ} = (I − Z_{q(i,σ)}) / 2

    So the drive adds:
        Δc[Z_{q(i,σ)}](t) = −½ v_i(t)

    and optionally an identity term:
        Δc[I](t) = +½ Σ_{i,σ} v_i(t) = Σ_i v_i(t)
    """

    template: DensityDriveTemplate
    site_potential: Callable[[float], np.ndarray]
    include_identity: bool = False
    coeff_tol: float = 0.0

    def coeff_map_exyz(self, t: float) -> Dict[str, float]:
        """Return ``{label_exyz: additive_coeff}`` at time *t*."""
        V = np.asarray(self.site_potential(float(t)), dtype=float).reshape(-1)
        if V.size != int(self.template.n_sites):
            raise ValueError(
                f"site_potential returned length {V.size}, expected {self.template.n_sites}"
            )

        out: Dict[str, float] = {}
        tol = float(self.coeff_tol)
        for i in range(int(self.template.n_sites)):
            vi = float(V[i])
            if tol > 0.0 and abs(vi) <= tol:
                continue
            coeff_z = -0.5 * vi
            for spin in (SPIN_UP, SPIN_DN):
                lbl = self.template.z_label(i, int(spin))
                out[lbl] = out.get(lbl, 0.0) + float(coeff_z)

        if bool(self.include_identity):
            # Σ_{i,σ} (+0.5) v_i = Σ_i v_i   (two spins per site)
            coeff_id = float(np.sum(V))
            if tol <= 0.0 or abs(coeff_id) > tol:
                out[self.template.identity_label] = (
                    out.get(self.template.identity_label, 0.0) + coeff_id
                )

        return out


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_gaussian_sinusoid_density_drive(
    *,
    n_sites: int,
    nq_total: int,
    indexing: str,
    A: float,
    omega: float,
    tbar: float,
    phi: float = 0.0,
    pattern_mode: str = "staggered",
    custom_weights: Optional[Sequence[float]] = None,
    include_identity: bool = False,
    electron_qubit_offset: int = 0,
    coeff_tol: float = 0.0,
) -> TimeDependentOnsiteDensityDrive:
    """Convenience builder used by pipeline scripts."""
    weights = default_spatial_weights(
        int(n_sites),
        mode=str(pattern_mode),
        custom=custom_weights,
    )
    potential = GaussianSinusoidSitePotential(
        weights=weights,
        A=float(A),
        omega=float(omega),
        tbar=float(tbar),
        phi=float(phi),
    )
    template = DensityDriveTemplate.build(
        n_sites=int(n_sites),
        nq_total=int(nq_total),
        indexing=str(indexing),
        electron_qubit_offset=int(electron_qubit_offset),
    )
    return TimeDependentOnsiteDensityDrive(
        template=template,
        site_potential=potential.v_sites,
        include_identity=bool(include_identity),
        coeff_tol=float(coeff_tol),
    )
