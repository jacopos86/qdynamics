from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the parent Holstein checkout that owns the shared quantum package."""

    origin = Path(start) if start is not None else Path(__file__)
    current = origin.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        quantum_core = candidate / "src" / "quantum" / "pauli_polynomial_class.py"
        if quantum_core.exists():
            return candidate

    raise RuntimeError("Could not locate parent repo root with src/quantum.")


REPO_ROOT = find_repo_root()


def ensure_repo_root_on_path() -> Path:
    """Make the parent repo importable for Paper 5 scripts and notebooks."""

    repo = REPO_ROOT
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo
