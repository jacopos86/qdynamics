"""Effective-angle geometry for deletion blocks on one ordered ansatz.

These mathematical operations use current coordinate angles and generator
coefficient bounds. Permission thresholds, caching, and post-refit continuity
tests belong to their callers.
"""

from __future__ import annotations

import math
from typing import Sequence


def effective_deletion_angle(
    theta: Sequence[float], coefficients: Sequence[float], deleted: Sequence[int]
) -> float:
    r"""Return :math:`a_D = \sum_{\mu\in D}|\theta_\mu c_\mu|`.

    ``theta`` and ``coefficients`` have the same coordinate order and length.
    ``deleted`` contains distinct valid indices. For a shared multi-Pauli
    coordinate, ``coefficients[mu]`` is its generator coefficient L1 bound.
    Inputs are validated by the checkpoint adapter. The empty block returns
    zero. Scalar summation follows the supplied index order to preserve the
    existing permission calculation at threshold boundaries.
    """

    return float(sum(abs(float(theta[i] * coefficients[i])) for i in deleted))


def deletion_ray_bound(effective_angle: float) -> float:
    r"""Return :math:`\sin(\min\{\pi/2,a_D\})` for nonnegative ``a_D``.

    Unitary invariance and the Fubini--Study triangle inequality bound the
    finite ray distance after removing the entire block by this value. The
    bound saturates at one and is zero for an empty deletion. It is a
    preliminary bound, not the measured post-refit ray distance.
    """

    return float(math.sin(min(math.pi / 2.0, effective_angle)))


__all__ = ["deletion_ray_bound", "effective_deletion_angle"]
