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

import argparse
import os
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


LEGACY_COMMIT = "6ec111c35c064bbaff2ad4405568b992512a1de4"
LEGACY_REL_PATH = "pipelines/hardcoded/adapt_pipeline.py"
REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCORER_PRIVATE_MODULE = "pipelines.static_adapt._legacy_hh_continuation_scoring_20260322"
LEGACY_GEOMETRY_MODE_ENV = "HH_LEGACY_PHASE3_GEOMETRY_MODE"
LEGACY_GEOMETRY_MODES = ("proxy_reduced", "exact_reduced")
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

_LEGACY_PHASE3_CALL_SNIPPET = """                        feat_full = build_full_candidate_features(
                            base_feature=feat_candidate_base,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            candidate_term=candidate_term,
                            window_terms=list(window_terms),
                            window_labels=[str(x) for x in window_labels],
                            cfg=phase2_score_cfg,
                            novelty_oracle=phase2_novelty_oracle,
                            curvature_oracle=phase2_curvature_oracle,
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                            optimizer_memory=active_memory,
                            motif_library=(phase3_input_motif_library if phase3_enabled else None),
                            target_num_sites=int(num_sites),
                        )"""
_LEGACY_PHASE3_CALL_REPLACEMENT = """                        feat_full = build_full_candidate_features(
                            base_feature=feat_candidate_base,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            candidate_term=candidate_term,
                            window_terms=list(window_terms),
                            window_labels=[str(x) for x in window_labels],
                            cfg=phase2_score_cfg,
                            novelty_oracle=phase2_novelty_oracle,
                            curvature_oracle=phase2_curvature_oracle,
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                            optimizer_memory=active_memory,
                            motif_library=(phase3_input_motif_library if phase3_enabled else None),
                            target_num_sites=int(num_sites),
                            legacy_selected_ops=list(selected_ops),
                            legacy_theta=np.asarray(theta, dtype=float),
                            legacy_psi_ref=np.asarray(psi_ref, dtype=complex),
                            legacy_h_compiled=h_compiled,
                        )"""
_LEGACY_STATIC_DV_SOURCE_PATCHES: tuple[tuple[str, str], ...] = (
    (
        """        v_t=None,
        v0=float(dv),""",
        """        v_t=float(dv),
        v0=None,""",
    ),
    (
        """            v_t=None,
            v0=float(args.dv),""",
        """            v_t=float(args.dv),
            v0=None,""",
    ),
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


def _normalize_legacy_geometry_mode(mode: str | None) -> str:
    mode_key = str(mode or "proxy_reduced").strip().lower()
    if mode_key not in LEGACY_GEOMETRY_MODES:
        raise ValueError(
            "phase3_legacy_geometry_mode must be one of "
            f"{set(LEGACY_GEOMETRY_MODES)}."
        )
    return str(mode_key)


def _extract_legacy_geometry_mode(argv: Sequence[str] | None) -> tuple[str, list[str] | None]:
    if argv is None:
        raw_args = list(sys.argv[1:])
    else:
        raw_args = [str(arg) for arg in argv]
    stripped: list[str] = []
    mode = "proxy_reduced"
    idx = 0
    while idx < len(raw_args):
        token = str(raw_args[idx])
        if token == "--phase3-legacy-geometry-mode":
            if idx + 1 >= len(raw_args):
                raise SystemExit("argument --phase3-legacy-geometry-mode: expected one argument")
            mode = _normalize_legacy_geometry_mode(raw_args[idx + 1])
            idx += 2
            continue
        if token.startswith("--phase3-legacy-geometry-mode="):
            mode = _normalize_legacy_geometry_mode(token.split("=", 1)[1])
            idx += 1
            continue
        stripped.append(token)
        idx += 1
    return str(mode), stripped


def _load_legacy_source() -> str:
    source = _git_show(LEGACY_REL_PATH)
    if _LEGACY_PHASE3_CALL_SNIPPET in source:
        source = source.replace(_LEGACY_PHASE3_CALL_SNIPPET, _LEGACY_PHASE3_CALL_REPLACEMENT, 1)
    for old_snippet, new_snippet in _LEGACY_STATIC_DV_SOURCE_PATCHES:
        if old_snippet in source:
            source = source.replace(old_snippet, new_snippet)
    return source


def _load_legacy_helper_module(module_name: str, rel_path: str) -> types.ModuleType:
    source = _git_show(rel_path)
    return _load_helper_module_from_source(module_name=module_name, source=source, filename=f"{rel_path}@{LEGACY_COMMIT}")


def _load_helper_module_from_source(
    *,
    module_name: str,
    source: str,
    filename: str,
    extra_globals: Mapping[str, Any] | None = None,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__file__": str(filename),
            "__name__": module_name,
            "__package__": str(module_name.rsplit(".", 1)[0]),
            "__builtins__": __builtins__,
            "LEGACY_SOURCE_COMMIT": LEGACY_COMMIT,
        }
    )
    if isinstance(extra_globals, Mapping):
        module.__dict__.update(dict(extra_globals))
    sys.modules[module_name] = module
    compiled = compile(source, filename=filename, mode="exec")
    exec(compiled, module.__dict__)
    return module


def _load_local_helper_module(
    *,
    module_name: str,
    rel_path: str,
    extra_globals: Mapping[str, Any] | None = None,
) -> types.ModuleType:
    abs_path = (REPO_ROOT / rel_path).resolve()
    source = abs_path.read_text(encoding="utf-8")
    return _load_helper_module_from_source(
        module_name=module_name,
        source=source,
        filename=str(abs_path),
        extra_globals=extra_globals,
    )


def _preload_legacy_helpers(geometry_mode: str) -> None:
    geometry_mode_key = _normalize_legacy_geometry_mode(geometry_mode)
    os.environ[LEGACY_GEOMETRY_MODE_ENV] = str(geometry_mode_key)
    for module_name, rel_path in _LEGACY_HELPER_MODULES:
        if module_name == "pipelines.hardcoded.hh_continuation_scoring":
            _load_legacy_helper_module(
                module_name=LEGACY_SCORER_PRIVATE_MODULE,
                rel_path=rel_path,
            )
            _load_local_helper_module(
                module_name=module_name,
                rel_path="pipelines/static_adapt/hh_continuation_scoring_legacy_bridge.py",
                extra_globals={
                    "LEGACY_SCORER_MODULE_NAME": LEGACY_SCORER_PRIVATE_MODULE,
                    "PHASE3_LEGACY_GEOMETRY_MODE": str(geometry_mode_key),
                },
            )
            continue
        _load_legacy_helper_module(module_name=module_name, rel_path=rel_path)


@lru_cache(maxsize=None)
def _load_legacy_module(geometry_mode: str = "proxy_reduced") -> types.ModuleType:
    geometry_mode_key = _normalize_legacy_geometry_mode(geometry_mode)
    _preload_legacy_helpers(geometry_mode_key)
    source = _load_legacy_source()
    module_name = f"adapt_pipeline_legacy_20260322_impl_{geometry_mode_key}"
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__file__": str(Path(__file__).resolve()),
            "__name__": module_name,
            "__package__": "pipelines.hardcoded",
            "__builtins__": __builtins__,
            "LEGACY_SOURCE_COMMIT": LEGACY_COMMIT,
            "PHASE3_LEGACY_GEOMETRY_MODE": str(geometry_mode_key),
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
    geometry_mode, stripped_argv = _extract_legacy_geometry_mode(argv)
    module = _load_legacy_module(geometry_mode)
    args = module.parse_args(stripped_argv)
    setattr(args, "phase3_legacy_geometry_mode", str(geometry_mode))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    geometry_mode, stripped_argv = _extract_legacy_geometry_mode(argv)
    module = _load_legacy_module(geometry_mode)
    module.main(stripped_argv)


if __name__ == "__main__":
    main(sys.argv[1:])
