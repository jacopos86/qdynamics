#!/usr/bin/env python3
"""Diagnostic legacy HH ADAPT entrypoint from the 2026-03-22 route.

This entrypoint executes the exact `pipelines/hardcoded/adapt_pipeline.py`
snapshot stored in git at commit `6ec111c35c064bbaff2ad4405568b992512a1de4`.

It is intentionally diagnostic-only:
- it lets us rerun the old top-level selection/scoring path inside this repo
- it avoids vendoring thousands of lines of historical source by hand
- it still uses the current repo's shared helper modules unless those are
  separately frozen elsewhere

The purpose is scaffold-regression analysis, not a permanent public surface.
"""

from __future__ import annotations

import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


LEGACY_COMMIT = "6ec111c35c064bbaff2ad4405568b992512a1de4"
LEGACY_REL_PATH = "pipelines/hardcoded/adapt_pipeline.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_LEGACY_HELPER_MODULES: tuple[tuple[str, str], ...] = (
    ("pipelines.hardcoded.hh_continuation_types", "pipelines/hardcoded/hh_continuation_types.py"),
    ("pipelines.hardcoded.hh_continuation_generators", "pipelines/hardcoded/hh_continuation_generators.py"),
    ("pipelines.hardcoded.hh_continuation_motifs", "pipelines/hardcoded/hh_continuation_motifs.py"),
    ("pipelines.hardcoded.hh_continuation_symmetry", "pipelines/hardcoded/hh_continuation_symmetry.py"),
    ("pipelines.hardcoded.hh_continuation_rescue", "pipelines/hardcoded/hh_continuation_rescue.py"),
    ("pipelines.hardcoded.hh_continuation_stage_control", "pipelines/hardcoded/hh_continuation_stage_control.py"),
    ("pipelines.hardcoded.hh_continuation_scoring", "pipelines/hardcoded/hh_continuation_scoring.py"),
    ("pipelines.hardcoded.hh_continuation_pruning", "pipelines/hardcoded/hh_continuation_pruning.py"),
    ("pipelines.hardcoded.hh_backend_compile_oracle", "pipelines/hardcoded/hh_backend_compile_oracle.py"),
)


def _git_show(rel_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:{rel_path}"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            "Failed to load legacy source from git history "
            f"({LEGACY_COMMIT}:{rel_path})."
            + (f" git stderr: {stderr}" if stderr else "")
        )
    return result.stdout


def _load_legacy_source() -> str:
    return _git_show(LEGACY_REL_PATH)


def _load_legacy_helper_module(module_name: str, rel_path: str) -> types.ModuleType:
    source = _git_show(rel_path)
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__file__": str((REPO_ROOT / rel_path).resolve()),
            "__name__": module_name,
            "__package__": str(module_name.rsplit(".", 1)[0]),
            "__builtins__": __builtins__,
            "LEGACY_SOURCE_COMMIT": LEGACY_COMMIT,
        }
    )
    sys.modules[module_name] = module
    compiled = compile(
        source,
        filename=f"{rel_path}@{LEGACY_COMMIT}",
        mode="exec",
    )
    exec(compiled, module.__dict__)
    return module


def _preload_legacy_helpers() -> None:
    for module_name, rel_path in _LEGACY_HELPER_MODULES:
        _load_legacy_helper_module(module_name=module_name, rel_path=rel_path)


@lru_cache(maxsize=1)
def _load_legacy_module() -> types.ModuleType:
    _preload_legacy_helpers()
    source = _load_legacy_source()
    module_name = "adapt_pipeline_legacy_20260322_impl"
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__file__": str(Path(__file__).resolve()),
            "__name__": module_name,
            "__package__": "pipelines.hardcoded",
            "__builtins__": __builtins__,
            "LEGACY_SOURCE_COMMIT": LEGACY_COMMIT,
        }
    )
    sys.modules[module_name] = module
    compiled = compile(
        source,
        filename=f"{LEGACY_REL_PATH}@{LEGACY_COMMIT}",
        mode="exec",
    )
    exec(compiled, module.__dict__)
    return module


def legacy_source_commit() -> str:
    return str(LEGACY_COMMIT)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    module = _load_legacy_module()
    return module.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    module = _load_legacy_module()
    module.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
