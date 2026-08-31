"""Committed-tree digests are reproduced once per repository and revision.

Every prospective package build reproduces the full ``git archive`` digest of
its frozen base revision several times.  That stream is ~1.5 GiB for the
Paper-II tree, so the repeated reproductions dominated packaging without adding
tamper evidence: a full commit ID is content-addressed and its archive cannot
change underneath a running process.  These tests pin the memoized behavior and
prove it never merges two distinct revisions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import pytest

import pipelines.time_dynamics.prospective_apm_exchange_chtc_package as apm_package_module
import pipelines.time_dynamics.prospective_avqds_runtime_source as runtime_source_module
import pipelines.time_dynamics.prospective_avqds_scout_chtc_package as scout_package_module
import pipelines.time_dynamics.prospective_avqds_trajectory_chtc_package as trajectory_package_module


_HELPERS = (
    (runtime_source_module, "_full_archive_sha"),
    (scout_package_module, "_reproduce_full_git_archive_sha"),
    (apm_package_module, "_reproduce_full_git_archive_sha"),
    (trajectory_package_module, "_reproduce_full_archive_sha"),
)
_HELPER_IDS = (
    "runtime_source",
    "scout_package",
    "apm_package",
    "trajectory_package",
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture()
def two_revision_repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Paper II Test")
    (root / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    _git(root, "add", "alpha.txt")
    _git(root, "commit", "-q", "-m", "first")
    first = _git(root, "rev-parse", "HEAD")
    (root / "beta.txt").write_text("beta\n", encoding="utf-8")
    _git(root, "add", "beta.txt")
    _git(root, "commit", "-q", "-m", "second")
    second = _git(root, "rev-parse", "HEAD")
    assert first != second
    return root, first, second


@pytest.fixture(autouse=True)
def _isolated_caches() -> Iterator[None]:
    for module, _ in _HELPERS:
        module._FULL_ARCHIVE_SHA_CACHE.clear()
    yield
    for module, _ in _HELPERS:
        module._FULL_ARCHIVE_SHA_CACHE.clear()


@pytest.mark.parametrize("module,name", _HELPERS, ids=_HELPER_IDS)
def test_full_archive_digest_streams_once_per_revision(
    module, name, two_revision_repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, first, second = two_revision_repository
    helper = getattr(module, name)

    calls: list[str] = []
    real_popen = subprocess.Popen

    def _counting_popen(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and tuple(argv[:2]) == ("git", "archive"):
            calls.append(str(argv[-1]))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", _counting_popen)

    first_digest = helper(root, first)
    assert helper(root, first) == first_digest
    assert helper(root, first) == first_digest
    assert calls == [first]

    second_digest = helper(root, second)
    assert second_digest != first_digest
    assert helper(root, second) == second_digest
    assert calls == [first, second]


@pytest.mark.parametrize("module,name", _HELPERS, ids=_HELPER_IDS)
def test_full_archive_digest_matches_an_uncached_reproduction(
    module, name, two_revision_repository
) -> None:
    root, first, _second = two_revision_repository
    helper = getattr(module, name)

    cached = helper(root, first)
    module._FULL_ARCHIVE_SHA_CACHE.clear()
    assert helper(root, first) == cached


@pytest.mark.parametrize("module,name", _HELPERS, ids=_HELPER_IDS)
def test_separate_repositories_do_not_share_a_cache_entry(
    module, name, two_revision_repository, tmp_path: Path
) -> None:
    root, first, _second = two_revision_repository
    helper = getattr(module, name)
    helper(root, first)

    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "config", "user.email", "test@example.invalid")
    _git(other, "config", "user.name", "Paper II Test")
    (other / "gamma.txt").write_text("gamma\n", encoding="utf-8")
    _git(other, "add", "gamma.txt")
    _git(other, "commit", "-q", "-m", "only")
    unrelated = _git(other, "rev-parse", "HEAD")

    keys = set(module._FULL_ARCHIVE_SHA_CACHE)
    assert keys == {(str(root.resolve()), first)}
    helper(other, unrelated)
    assert set(module._FULL_ARCHIVE_SHA_CACHE) == keys | {
        (str(other.resolve()), unrelated)
    }
