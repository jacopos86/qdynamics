"""Paper 5 workspace package."""

from .repo import REPO_ROOT, ensure_repo_root_on_path, find_repo_root

__all__ = [
    "REPO_ROOT",
    "ensure_repo_root_on_path",
    "find_repo_root",
]
