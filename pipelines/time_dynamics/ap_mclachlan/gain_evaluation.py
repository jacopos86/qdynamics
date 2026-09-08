"""Gain approximations on a fixed retained support."""

from collections.abc import Iterable


def factorized_capture(
    base_retained_capture: float, singleton_captures: Iterable[float]
) -> float:
    r"""Return the signed singleton approximation to block capture.

    .. math::

       \widetilde Q(R\cup A)=Q(R)+\sum_{a\in A}[Q(R\cup\{a\})-Q(R)].

    All singletons use the same retained support and solve policy. This is
    a ranking approximation, not a joint projection or an upper bound.
    Signed increments are not clipped; the singleton case is returned
    directly so it agrees bitwise with the original solve.
    """

    base = float(base_retained_capture)
    captures = tuple(float(value) for value in singleton_captures)
    if len(captures) == 1:
        return captures[0]
    return base + sum(value - base for value in captures)


__all__ = ["factorized_capture"]
