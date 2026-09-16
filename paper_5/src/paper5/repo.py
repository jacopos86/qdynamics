"""Locate the parent Holstein checkout, lazily.

``paper5`` is shipped to execute nodes as ``paper_5/src`` alone, without the
parent repository that owns ``src/quantum``.  Only two modules need that
parent -- ``paper5.quantum_imports`` and ``paper5.stability.report`` -- so
the root is resolved on first use, never at import: ``import paper5`` and
every module under ``paper5.stability`` that does not touch the parent
import without it, and the two that need it raise the same ``RuntimeError``
they always raised, at the point they need it.

``REPO_ROOT`` is a module attribute resolved on first access (PEP 562) and
then cached as an ordinary module global; ``find_repo_root`` raises when the
parent is absent, exactly as before; ``locate_repo_root`` is the non-raising
form and returns ``None``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT_MARKER = ("src", "quantum", "pauli_polynomial_class.py")
_MISSING_PARENT_MESSAGE = "Could not locate parent repo root with src/quantum."


def locate_repo_root(start: str | Path | None = None) -> Path | None:
    """Return the parent Holstein checkout that owns ``src/quantum``, or ``None``.

    The walk starts at ``start`` (this file when omitted) and climbs its
    resolved ancestors until one holds
    ``src/quantum/pauli_polynomial_class.py``.
    """

    origin = Path(start) if start is not None else Path(__file__)
    current = origin.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if candidate.joinpath(*_PARENT_MARKER).exists():
            return candidate
    return None


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the parent Holstein checkout that owns the shared quantum package.

    Raises ``RuntimeError`` when no ancestor holds it, as it always did.
    """

    root = locate_repo_root(start)
    if root is None:
        raise RuntimeError(_MISSING_PARENT_MESSAGE)
    return root


def _resolved_repo_root() -> Path:
    """The cached ``REPO_ROOT``, resolving it on the first call."""

    cached = globals().get("REPO_ROOT")
    if cached is None:
        cached = find_repo_root()
        globals()["REPO_ROOT"] = cached
    return cached


def __getattr__(name: str) -> Path:
    # PEP 562: ``paper5.repo.REPO_ROOT`` and ``from paper5.repo import
    # REPO_ROOT`` resolve the parent on first access and cache it as a real
    # module global, so later lookups never come back here.  Without the
    # parent the access raises the RuntimeError find_repo_root raises,
    # nothing is cached, and importing this module never triggers the walk.
    if name == "REPO_ROOT":
        return _resolved_repo_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "REPO_ROOT"})


def ensure_repo_root_on_path() -> Path:
    """Make the parent repo importable for Paper 5 scripts and notebooks.

    Raises ``RuntimeError`` when the parent is absent; this is the call the
    quantum-importing module makes, so it fails exactly when it needs it.
    """

    repo = _resolved_repo_root()
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


__all__ = [
    "REPO_ROOT",
    "ensure_repo_root_on_path",
    "find_repo_root",
    "locate_repo_root",
]
