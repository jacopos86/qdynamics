"""``paper5`` imports without the parent Holstein checkout.

Cluster 10554184 of the CWRMF acceleration lane died at import on the
execute node: ``paper5/__init__.py`` imported ``paper5.repo``, whose
module-level ``REPO_ROOT = find_repo_root()`` raised because the bundle
ships ``paper_5/src`` without the parent's ``src/quantum``.  The
carried-witness chain imports nothing from the parent; only
``paper5.quantum_imports`` and ``paper5.stability.report`` do.

Pinned here: the package and the carried-witness driver import from a tree
holding only ``paper_5/src``; the parent-needing entry points raise the same
``RuntimeError`` at the point they need the parent and not before; and with
the parent present every existing form of access resolves to the path it
always did.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import paper5
import paper5.repo
from paper5.repo import find_repo_root, locate_repo_root

PAPER5_SRC = Path(paper5.__file__).resolve().parents[1]
WORKSPACE = PAPER5_SRC.parent
PARENT_MARKER = Path("src", "quantum", "pauli_polynomial_class.py")
MISSING_PARENT_MESSAGE = "Could not locate parent repo root with src/quantum."
DRIVER = "paper5.stability.apcm_carried_witness_analysis"


def _ancestor_holding_the_parent_marker(path: Path) -> Path | None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / PARENT_MARKER).exists():
            return candidate
    return None


def _python(
    argv: list[str], cwd: Path, *, pythonpath: str
) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter whose only import root beyond site-packages is ``pythonpath``."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONSTARTUP")
    }
    env["PYTHONPATH"] = pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *argv],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.fixture(scope="module")
def bundle_like_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``paper_5/src`` copied -- never linked -- into a tree with no parent above it."""

    root = tmp_path_factory.mktemp("paper5_without_parent")
    holder = _ancestor_holding_the_parent_marker(root)
    assert holder is None, (
        f"the temporary root {root} sits under {holder}, which holds "
        f"{PARENT_MARKER}; the tree cannot stand for one without the parent"
    )
    shutil.copytree(
        PAPER5_SRC,
        root / "paper_5" / "src",
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    assert not [path for path in (root / "paper_5").rglob("*") if path.is_symlink()]
    return root


def test_the_package_and_the_carried_witness_driver_import_without_the_parent(
    bundle_like_tree: Path,
) -> None:
    located = _python(
        [
            "-c",
            "import paper5, paper5.repo, paper5.stability\n"
            "print(paper5.__file__)\n"
            "print(paper5.repo.locate_repo_root())",
        ],
        bundle_like_tree,
        pythonpath="paper_5/src",
    )
    assert located.returncode == 0, located.stderr
    where, root = located.stdout.splitlines()
    assert Path(where).resolve().is_relative_to(bundle_like_tree.resolve())
    assert root == "None"

    helped = _python(["-m", DRIVER, "--help"], bundle_like_tree, pythonpath="paper_5/src")
    assert helped.returncode == 0, helped.stderr
    assert helped.stdout.startswith("usage:")
    assert MISSING_PARENT_MESSAGE not in helped.stderr


@pytest.mark.parametrize(
    "statement",
    [
        "import paper5.repo as repo; repo.REPO_ROOT",
        "import paper5; paper5.REPO_ROOT",
        "from paper5.repo import REPO_ROOT",
        "from paper5 import REPO_ROOT",
        "import paper5.repo as repo; repo.ensure_repo_root_on_path()",
        "import paper5.repo as repo; repo.find_repo_root()",
        "import paper5.quantum_imports",
        "import paper5.stability.report",
    ],
)
def test_the_parent_needing_entry_points_raise_the_same_error_only_when_used(
    bundle_like_tree: Path, statement: str
) -> None:
    result = _python(["-c", statement], bundle_like_tree, pythonpath="paper_5/src")
    assert result.returncode != 0, (statement, result.stdout)
    assert f"RuntimeError: {MISSING_PARENT_MESSAGE}" in result.stderr, (
        statement,
        result.stderr,
    )


def test_behaviour_with_the_parent_present_is_unchanged(tmp_path: Path) -> None:
    root = find_repo_root()
    assert isinstance(root, Path)
    assert (root / PARENT_MARKER).is_file()
    assert root == Path(paper5.__file__).resolve().parents[3]
    assert locate_repo_root() == root

    assert paper5.repo.REPO_ROOT == root
    assert vars(paper5.repo)["REPO_ROOT"] is paper5.repo.REPO_ROOT
    assert paper5.REPO_ROOT is paper5.repo.REPO_ROOT
    assert paper5.repo.ensure_repo_root_on_path() == root
    assert str(root) in sys.path
    assert "REPO_ROOT" in dir(paper5.repo)

    with pytest.raises(
        RuntimeError, match=r"^Could not locate parent repo root with src/quantum\.$"
    ):
        find_repo_root(tmp_path)
    assert locate_repo_root(tmp_path) is None

    probe = _python(
        [
            "-c",
            "from paper5 import REPO_ROOT as a\n"
            "from paper5.repo import REPO_ROOT as b\n"
            "import paper5.quantum_imports as q\n"
            "from paper5.stability import report\n"
            "print(a)\nprint(b)\nprint(report.REPO_ROOT)\n"
            "print(report.TEX_SOURCE.relative_to(report.REPO_ROOT).as_posix())\n"
            "print(isinstance(q.jw_creation(2, 0), q.PauliPolynomial))",
        ],
        WORKSPACE.parent,
        pythonpath=str(PAPER5_SRC),
    )
    assert probe.returncode == 0, probe.stderr
    a, b, c, tex, ladder_is_pauli_polynomial = probe.stdout.splitlines()
    assert a == b == c == str(root)
    assert tex == (
        "MATH/paper_facing/paper_V_high_u_gkba/paper_v_scalar_stability_diagnostic.tex"
    )
    assert ladder_is_pauli_polynomial == "True"
