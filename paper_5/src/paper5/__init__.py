"""Paper 5 workspace package.

Importing ``paper5`` does not need the parent Holstein checkout.  The parent
root is resolved on first use of ``paper5.REPO_ROOT`` (equivalently
``paper5.repo.REPO_ROOT``), and only ``paper5.quantum_imports`` and
``paper5.stability.report`` need it at import; see ``paper5.repo``.
"""

from __future__ import annotations

from .repo import ensure_repo_root_on_path, find_repo_root, locate_repo_root

__all__ = [
    "REPO_ROOT",
    "ensure_repo_root_on_path",
    "find_repo_root",
    "locate_repo_root",
]


def __getattr__(name: str):
    # ``paper5.REPO_ROOT`` stays available, resolved on first access through
    # paper5.repo (PEP 562) and cached here as the same object.  Without the
    # parent that access raises paper5.repo's RuntimeError; ``import paper5``
    # never does.
    if name == "REPO_ROOT":
        from . import repo

        root = repo.REPO_ROOT
        globals()["REPO_ROOT"] = root
        return root
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
