#!/usr/bin/env python3
"""Generate implementation-guide assets for the Hubbard pipeline repo.

Outputs:
- docs/repo_guide_assets/repo_guide_summary.json
- docs/repo_guide_assets/repo_guide_artifact_metrics.json
- docs/repo_guide_assets/*.png (deterministic architecture + artifact-derived figures)

Design goals:
- Read-only analysis of repository source files.
- Deterministic output ordering for reproducibility.
- No dependency on graphviz; uses matplotlib + networkx only.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


STYLE: dict[str, str] = {
    "text_primary": "#111111",
    "text_secondary": "#222222",
    "edge_dark": "#202020",
    "edge_mid": "#333333",
    "background_light": "#F8F8F8",
    "panel_light": "#F2F2F2",
    "blue_fill": "#E8F1FA",
    "green_fill": "#EAF6EA",
    "orange_fill": "#FFF1E0",
    "purple_fill": "#F1EAF6",
}

CANONICAL_CASES: list[str] = [
    "artifacts/json/H_L2_static_t1.0_U4.0_S64_heavy.json",
    "artifacts/json/H_L3_static_t1.0_U4.0_S128_heavy.json",
    "artifacts/json/H_L4_vt_t1.0_U4.0_S256_dyn.json",
]

TARGET_FUNCTIONS: list[str] = [
    "_simulate_trajectory",
    "_evolve_trotter_suzuki2_absolute",
    "_evolve_piecewise_exact",
    "_site_resolved_number_observables",
    "_spin_orbital_bit_index",
    "_run_hardcoded_vqe",
    "vqe_minimize",
    "hartree_fock_bitstring",
    "hartree_fock_statevector",
    "evaluate_drive_waveform",
    "build_density_drive_from_args",
]


@dataclass(frozen=True)
class RepoPaths:
    workspace_root: Path
    subrepo_root: Path
    docs_dir: Path
    assets_dir: Path


def _resolve_repo_paths(script_path: Path) -> RepoPaths:
    subrepo_root = script_path.resolve().parents[1]
    workspace_root = subrepo_root.parent
    docs_dir = subrepo_root / "docs"
    assets_dir = docs_dir / "repo_guide_assets"
    return RepoPaths(
        workspace_root=workspace_root,
        subrepo_root=subrepo_root,
        docs_dir=docs_dir,
        assets_dir=assets_dir,
    )


def _rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _source_files(paths: RepoPaths) -> dict[str, list[Path]]:
    quantum_py = sorted((paths.workspace_root / "src" / "quantum").glob("*.py"))
    pipeline_py = sorted((paths.subrepo_root / "pipelines").glob("*.py"))
    pipeline_sh = sorted((paths.subrepo_root / "pipelines").glob("*.sh"))
    top_tests = sorted(paths.workspace_root.glob("test_*.py"))

    return {
        "quantum_py": quantum_py,
        "pipeline_py": pipeline_py,
        "pipeline_sh": pipeline_sh,
        "top_tests": top_tests,
        "analysis_py": sorted(quantum_py + pipeline_py + top_tests),
    }


def _path_to_module(path: Path, paths: RepoPaths) -> str:
    rel = path.resolve().relative_to(paths.workspace_root.resolve())
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<unparsed>"


def _extract_add_argument_calls(tree: ast.AST) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        if not node.args:
            continue

        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue

        if not first.value.startswith("--"):
            continue

        rec: dict[str, Any] = {
            "flag": first.value,
            "line": int(getattr(node, "lineno", -1)),
        }

        for kw in node.keywords:
            if kw.arg is None:
                continue
            if kw.arg in {
                "default",
                "type",
                "choices",
                "action",
                "required",
                "help",
                "metavar",
                "dest",
            }:
                rec[kw.arg] = _safe_literal(kw.value)

        flags.append(rec)

    flags.sort(key=lambda x: (x.get("line", 0), x["flag"]))
    return flags


def _extract_constants(tree: ast.AST) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    const_re = re.compile(r"^_?[A-Z][A-Z0-9_]*$")

    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and const_re.match(target.id):
                    out.append(
                        {
                            "name": target.id,
                            "line": int(node.lineno),
                            "value": _safe_literal(node.value),
                        }
                    )
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and const_re.match(node.target.id):
                out.append(
                    {
                        "name": node.target.id,
                        "line": int(node.lineno),
                        "value": _safe_literal(node.value) if node.value is not None else None,
                    }
                )

    out.sort(key=lambda x: (x["line"], x["name"]))
    return out


def _extract_top_level_defs(tree: ast.AST) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    body = tree.body if isinstance(tree, ast.Module) else []
    for node in body:
        if isinstance(node, ast.ClassDef):
            classes.append({"name": node.name, "line": int(node.lineno)})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": node.name, "line": int(node.lineno)})

    return classes, functions


def _extract_imports(
    tree: ast.AST,
    *,
    local_stems: set[str],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    internal: set[str] = set()
    external: set[str] = set()
    edges: list[dict[str, str]] = []

    def classify(mod: str) -> str:
        if mod.startswith("src.") or mod.startswith("pipelines"):
            return "internal"
        head = mod.split(".")[0]
        if mod in local_stems or head in local_stems:
            return "internal"
        return "external"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                kind = classify(mod)
                if kind == "internal":
                    internal.add(mod)
                else:
                    external.add(mod.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not mod:
                continue
            kind = classify(mod)
            if kind == "internal":
                internal.add(mod)
            else:
                external.add(mod.split(".")[0])

    for dst in sorted(internal):
        edges.append({"dst": dst})

    return sorted(internal), sorted(external), edges


def _build_local_stems(py_files: list[Path], paths: RepoPaths) -> set[str]:
    stems: set[str] = set()
    for path in py_files:
        stems.add(path.stem)
        stems.add(_path_to_module(path, paths))
    return stems


def _file_with_lines(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return list(enumerate(lines, start=1))


def _find_pattern(path: Path, pattern: str, *, flags: int = 0) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    regex = re.compile(pattern, flags)
    for line_no, line in _file_with_lines(path):
        if regex.search(line):
            out.append({"line": line_no, "snippet": line.strip()})
    return out


def _extract_invariants(paths: RepoPaths) -> dict[str, Any]:
    agents_path = paths.workspace_root / "AGENTS.md"
    compare_path = paths.subrepo_root / "pipelines" / "compare_hardcoded_vs_qiskit_pipeline.py"
    qiskit_path = paths.subrepo_root / "pipelines" / "qiskit_hubbard_baseline_pipeline.py"
    hardcoded_path = paths.subrepo_root / "pipelines" / "hardcoded_hubbard_pipeline.py"
    design_note_path = paths.subrepo_root / "pipelines" / "DESIGN_NOTE_QISKIT_BASELINE_TIMEDEP.md"

    invariants: dict[str, Any] = {}

    pauli_hits = _find_pattern(agents_path, r"Use `e/x/y/z` internally")
    invariants["pauli_symbol_convention_exyz"] = {
        "found": bool(pauli_hits),
        "expected": "Use e/x/y/z internally; convert to I/X/Y/Z at boundaries.",
        "evidence": [{"file": _rel(agents_path, paths.workspace_root), **h} for h in pauli_hits[:3]],
    }

    order_hits = _find_pattern(agents_path, r"left-to-right = q_\(n-1\) \.\.\. q_0")
    rightmost_hits = _find_pattern(agents_path, r"qubit 0 is rightmost")
    invariants["pauli_string_qubit_ordering"] = {
        "found": bool(order_hits and rightmost_hits),
        "expected": "Pauli words are q_(n-1)...q_0 left-to-right; qubit 0 is rightmost.",
        "evidence": [
            {"file": _rel(agents_path, paths.workspace_root), **h}
            for h in (order_hits + rightmost_hits)[:4]
        ],
    }

    jw_plus_hits = _find_pattern(agents_path, r"fermion_plus_operator")
    jw_minus_hits = _find_pattern(agents_path, r"fermion_minus_operator")
    invariants["jw_mapping_source_of_truth"] = {
        "found": bool(jw_plus_hits and jw_minus_hits),
        "expected": "JW mapping should use fermion_plus_operator/fermion_minus_operator from pauli_polynomial_class.py.",
        "evidence": [
            {"file": _rel(agents_path, paths.workspace_root), **h}
            for h in (jw_plus_hits + jw_minus_hits)[:4]
        ],
    }

    safe_hits = _find_pattern(compare_path, r"_SAFE_TEST_THRESHOLD\s*:\s*float\s*=\s*1e-10")
    invariants["drive_safe_test_threshold"] = {
        "found": bool(safe_hits),
        "expected": "A=0 drive safe-test threshold is 1e-10.",
        "evidence": [{"file": _rel(compare_path, paths.workspace_root), **h} for h in safe_hits[:2]],
    }

    passthrough_hits = _find_pattern(compare_path, r"def _build_drive_args")
    passthrough_comment_hits = _find_pattern(compare_path, r"forwarded verbatim")
    invariants["drive_flag_passthrough"] = {
        "found": bool(passthrough_hits),
        "expected": "Compare pipeline forwards drive flags verbatim to both sub-pipelines.",
        "evidence": [
            {"file": _rel(compare_path, paths.workspace_root), **h}
            for h in (passthrough_hits + passthrough_comment_hits)[:4]
        ],
    }

    split_hits_q = _find_pattern(qiskit_path, r"if not has_drive")
    split_hits_drive = _find_pattern(qiskit_path, r"PauliEvolutionGate is NOT used here")
    split_hits_design = _find_pattern(design_note_path, r"Option A")
    invariants["static_vs_driven_reference_split"] = {
        "found": bool(split_hits_q and split_hits_drive),
        "expected": (
            "No-drive path uses static eigendecomposition/PauliEvolutionGate branch, while drive-enabled "
            "path uses shared scipy/numpy piecewise kernels."
        ),
        "evidence": [
            {"file": _rel(qiskit_path, paths.workspace_root), **h}
            for h in (split_hits_q[:2] + split_hits_drive[:2])
        ]
        + [
            {"file": _rel(design_note_path, paths.workspace_root), **h}
            for h in split_hits_design[:1]
        ],
    }

    ref_hits_h = _find_pattern(hardcoded_path, r"reference_method")
    invariants["reference_method_metadata"] = {
        "found": bool(ref_hits_h),
        "expected": "Drive metadata includes reference_method and reference_steps_multiplier.",
        "evidence": [{"file": _rel(hardcoded_path, paths.workspace_root), **h} for h in ref_hits_h[:3]],
    }

    return invariants


def _signature_from_source(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    start = int(node.lineno) - 1
    end = int(getattr(node, "end_lineno", node.lineno))
    src = "\n".join(lines[start:end])
    m = re.search(r"def\s+" + re.escape(node.name) + r"\s*\((?:.|\n)*?\):", src)
    if m:
        return m.group(0).replace("\n", " ").strip()
    return f"def {node.name}(...)"


def _called_names_in_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                names.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                names.add(sub.func.attr)
    return sorted(names)


def _extract_implementation_evidence(paths: RepoPaths) -> dict[str, Any]:
    target_files = [
        paths.subrepo_root / "pipelines" / "hardcoded_hubbard_pipeline.py",
        paths.subrepo_root / "pipelines" / "qiskit_hubbard_baseline_pipeline.py",
        paths.workspace_root / "src" / "quantum" / "vqe_latex_python_pairs.py",
        paths.workspace_root / "src" / "quantum" / "hartree_fock_reference_state.py",
        paths.workspace_root / "src" / "quantum" / "drives_time_potential.py",
    ]

    function_defs: dict[str, list[dict[str, Any]]] = {name: [] for name in TARGET_FUNCTIONS}

    for fpath in target_files:
        if not fpath.exists():
            continue
        src_text = fpath.read_text(encoding="utf-8")
        lines = src_text.splitlines()
        tree = ast.parse(src_text)

        for node in tree.body if isinstance(tree, ast.Module) else []:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_defs:
                entry = {
                    "function": node.name,
                    "file": _rel(fpath, paths.workspace_root),
                    "line_start": int(node.lineno),
                    "line_end": int(getattr(node, "end_lineno", node.lineno)),
                    "signature": _signature_from_source(lines, node),
                    "defaults": {
                        "num_args": len(getattr(node.args, "args", [])),
                        "num_defaults": len(getattr(node.args, "defaults", [])),
                        "defaults": [ast.unparse(d) for d in getattr(node.args, "defaults", [])],
                    },
                    "calls": _called_names_in_function(node),
                }
                function_defs[node.name].append(entry)

    chosen: list[dict[str, Any]] = []
    for fname in TARGET_FUNCTIONS:
        candidates = sorted(function_defs.get(fname, []), key=lambda x: (x["file"], x["line_start"]))
        if candidates:
            chosen.append(candidates[0])

    target_name_set = {x["function"] for x in chosen}
    call_edges: list[dict[str, str]] = []
    for rec in chosen:
        src_name = rec["function"]
        for called in rec.get("calls", []):
            if called in target_name_set:
                call_edges.append({"src": src_name, "dst": called})

    called_by: dict[str, list[str]] = {name: [] for name in target_name_set}
    for e in call_edges:
        called_by[e["dst"]].append(e["src"])
    for k in list(called_by.keys()):
        called_by[k] = sorted(set(called_by[k]))

    for rec in chosen:
        rec["called_by"] = called_by.get(rec["function"], [])

    missing = [name for name in TARGET_FUNCTIONS if name not in target_name_set]

    chosen.sort(key=lambda x: (x["file"], x["line_start"], x["function"]))
    call_edges.sort(key=lambda x: (x["src"], x["dst"]))

    return {
        "target_functions": TARGET_FUNCTIONS,
        "found_functions": [x["function"] for x in chosen],
        "missing_functions": missing,
        "function_records": chosen,
        "call_graph_edges": call_edges,
    }


def _collect_module_summary(paths: RepoPaths, files: dict[str, list[Path]]) -> dict[str, Any]:
    py_files = files["analysis_py"]
    local_stems = _build_local_stems(py_files, paths)

    modules: list[dict[str, Any]] = []
    internal_edges: list[dict[str, str]] = []
    external_deps: set[str] = set()

    for py_path in py_files:
        rel_path = _rel(py_path, paths.workspace_root)
        module_name = _path_to_module(py_path, paths)

        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        classes, functions = _extract_top_level_defs(tree)
        cli_flags = _extract_add_argument_calls(tree)
        constants = _extract_constants(tree)
        internal, external, edges = _extract_imports(tree, local_stems=local_stems)

        external_deps.update(external)
        for edge in edges:
            internal_edges.append({"src": module_name, "dst": edge["dst"]})

        modules.append(
            {
                "module": module_name,
                "path": rel_path,
                "classes": classes,
                "functions": functions,
                "imports": {
                    "internal": internal,
                    "external": external,
                },
                "cli_flags": cli_flags,
                "constants": constants,
            }
        )

    modules.sort(key=lambda m: m["path"])
    internal_edges.sort(key=lambda e: (e["src"], e["dst"]))

    return {
        "modules": modules,
        "internal_import_edges": internal_edges,
        "external_dependencies": sorted(external_deps),
    }


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _trajectory_array(traj: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([_safe_float(row.get(key, float("nan"))) for row in traj], dtype=float)


def _compute_case_metrics(case_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    settings = payload.get("settings", {})
    traj = payload.get("trajectory", [])

    required_top = ["settings", "trajectory", "vqe", "ground_state"]
    missing_top = [k for k in required_top if k not in payload]

    if not isinstance(traj, list) or not traj:
        return {
            "case": case_name,
            "valid": False,
            "missing_top": missing_top,
            "errors": ["trajectory missing or empty"],
        }

    times = _trajectory_array(traj, "time")
    fidelity = _trajectory_array(traj, "fidelity")
    e_stat_exact = _trajectory_array(traj, "energy_static_exact")
    e_stat_ans = _trajectory_array(traj, "energy_static_exact_ansatz")
    e_stat_trot = _trajectory_array(traj, "energy_static_trotter")
    e_tot_exact = _trajectory_array(traj, "energy_total_exact")
    e_tot_trot = _trajectory_array(traj, "energy_total_trotter")

    n_up0 = _trajectory_array(traj, "n_up_site0_trotter")
    n_dn0 = _trajectory_array(traj, "n_dn_site0_trotter")
    doublon = _trajectory_array(traj, "doublon_trotter")
    staggered = _trajectory_array(traj, "staggered_trotter")

    err_stat_trot_vs_ans = np.abs(e_stat_trot - e_stat_ans)
    err_stat_trot_vs_exact = np.abs(e_stat_trot - e_stat_exact)
    err_tot_trot_vs_exact = np.abs(e_tot_trot - e_tot_exact)

    drive_cfg = settings.get("drive") if isinstance(settings.get("drive"), dict) else {}
    has_drive = bool(drive_cfg)

    static_total_delta = np.abs(e_tot_trot - e_stat_trot)

    vqe = payload.get("vqe", {}) if isinstance(payload.get("vqe"), dict) else {}
    gs = payload.get("ground_state", {}) if isinstance(payload.get("ground_state"), dict) else {}

    vqe_energy = _safe_float(vqe.get("energy"), float("nan"))
    gs_filtered = _safe_float(gs.get("exact_energy_filtered"), float("nan"))
    vqe_abs_err = abs(vqe_energy - gs_filtered) if np.isfinite(vqe_energy) and np.isfinite(gs_filtered) else float("nan")

    occupancy_bounds_ok = bool(
        np.nanmin(n_up0) >= -1e-10
        and np.nanmax(n_up0) <= 1.0 + 1e-10
        and np.nanmin(n_dn0) >= -1e-10
        and np.nanmax(n_dn0) <= 1.0 + 1e-10
    )

    doublon_bounds_ok = bool(np.nanmin(doublon) >= -1e-10 and np.nanmax(doublon) <= float(settings.get("L", 0)) + 1e-10)

    required_traj_keys = [
        "time",
        "fidelity",
        "energy_static_exact",
        "energy_static_trotter",
        "energy_total_exact",
        "energy_total_trotter",
        "n_up_site0_trotter",
        "n_dn_site0_trotter",
        "doublon_trotter",
        "staggered_trotter",
    ]
    sample_keys = set(traj[0].keys()) if isinstance(traj[0], dict) else set()
    missing_traj_keys = sorted([k for k in required_traj_keys if k not in sample_keys])

    out = {
        "case": case_name,
        "valid": len(missing_top) == 0 and len(missing_traj_keys) == 0,
        "missing_top": missing_top,
        "missing_trajectory_keys": missing_traj_keys,
        "settings": {
            "L": int(settings.get("L", -1)),
            "ordering": str(settings.get("ordering", "?")),
            "trotter_steps": int(settings.get("trotter_steps", -1)),
            "num_times": int(settings.get("num_times", -1)),
            "term_order": str(settings.get("term_order", "?")),
            "has_drive": has_drive,
            "reference_method": str(drive_cfg.get("reference_method", "eigendecomposition")) if has_drive else "eigendecomposition",
            "reference_steps_multiplier": _safe_float(drive_cfg.get("reference_steps_multiplier"), float("nan")) if has_drive else float("nan"),
        },
        "vqe": {
            "energy": vqe_energy,
            "exact_energy_filtered": gs_filtered,
            "abs_error": vqe_abs_err,
            "optimizer_method": str(vqe.get("optimizer_method", "")),
            "num_parameters": int(vqe.get("num_parameters", -1)) if vqe.get("num_parameters") is not None else -1,
            "best_restart": int(vqe.get("best_restart", -1)) if vqe.get("best_restart") is not None else -1,
        },
        "time_grid": {
            "t_start": _safe_float(times[0]),
            "t_end": _safe_float(times[-1]),
            "num_points": int(times.size),
            "dt_mean": _safe_float(np.nanmean(np.diff(times))) if times.size > 1 else float("nan"),
        },
        "fidelity": {
            "min": _safe_float(np.nanmin(fidelity)),
            "max": _safe_float(np.nanmax(fidelity)),
            "final": _safe_float(fidelity[-1]),
        },
        "energy_errors": {
            "static_trot_vs_exact_ans_max": _safe_float(np.nanmax(err_stat_trot_vs_ans)),
            "static_trot_vs_exact_ans_mean": _safe_float(np.nanmean(err_stat_trot_vs_ans)),
            "static_trot_vs_exact_gs_max": _safe_float(np.nanmax(err_stat_trot_vs_exact)),
            "total_trot_vs_exact_max": _safe_float(np.nanmax(err_tot_trot_vs_exact)),
            "total_minus_static_trot_max": _safe_float(np.nanmax(static_total_delta)),
        },
        "observables": {
            "n_up_site0_range": [_safe_float(np.nanmin(n_up0)), _safe_float(np.nanmax(n_up0))],
            "n_dn_site0_range": [_safe_float(np.nanmin(n_dn0)), _safe_float(np.nanmax(n_dn0))],
            "doublon_range": [_safe_float(np.nanmin(doublon)), _safe_float(np.nanmax(doublon))],
            "staggered_range": [_safe_float(np.nanmin(staggered)), _safe_float(np.nanmax(staggered))],
        },
        "consistency_checks": {
            "occupancy_bounds_ok": occupancy_bounds_ok,
            "doublon_bounds_ok": doublon_bounds_ok,
            "static_equals_total_when_no_drive_max_delta": _safe_float(np.nanmax(static_total_delta)) if not has_drive else float("nan"),
            "drive_changes_total_energy_max_delta": _safe_float(np.nanmax(static_total_delta)) if has_drive else float("nan"),
        },
    }

    return out


def _extract_artifact_metrics(paths: RepoPaths) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for rel_case in CANONICAL_CASES:
        p = paths.subrepo_root / rel_case
        case_name = p.name
        if not p.exists():
            cases.append(
                {
                    "case": case_name,
                    "path": rel_case,
                    "exists": False,
                    "valid": False,
                    "error": "missing file",
                }
            )
            continue

        payload = json.loads(p.read_text(encoding="utf-8"))
        rec = _compute_case_metrics(case_name=case_name, payload=payload)
        rec["path"] = rel_case
        rec["exists"] = True
        cases.append(rec)

    all_valid = all(bool(c.get("valid", False)) for c in cases)

    return {
        "canonical_cases": CANONICAL_CASES,
        "all_valid": all_valid,
        "cases": cases,
    }


def _prep_axes(title: str, subtitle: str = "", *, figsize: tuple[float, float] = (13.5, 8.0)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98, color=STYLE["text_primary"])
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11, color=STYLE["text_secondary"])
    return fig, ax


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _draw_graph(
    g: nx.DiGraph,
    *,
    path: Path,
    title: str,
    subtitle: str,
    pos: dict[str, tuple[float, float]] | None = None,
    node_groups: dict[str, str] | None = None,
) -> None:
    fig, ax = _prep_axes(title, subtitle)

    if pos is None:
        pos = nx.spring_layout(g, seed=7, k=1.0)

    palette = {
        "quantum": "#6A9BCF",
        "pipeline": "#F2AE63",
        "test": "#74B58A",
        "core": "#B7A2CF",
        "artifact": "#C8A687",
        "metrics": "#9FC6CF",
        "other": "#D7D7D7",
    }

    node_colors = []
    for node in g.nodes:
        grp = "other"
        if node_groups is not None:
            grp = node_groups.get(node, "other")
        node_colors.append(palette.get(grp, palette["other"]))

    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=1900,
        edgecolors=STYLE["edge_dark"],
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(g, pos, font_size=8.5, font_color=STYLE["text_primary"], ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.2,
        edge_color=STYLE["edge_mid"],
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )

    _save(fig, path)


def _text_box_figure(
    out: Path,
    *,
    title: str,
    subtitle: str,
    body: str,
    fontsize: float = 10.5,
) -> None:
    fig, ax = _prep_axes(title, subtitle)
    ax.text(
        0.04,
        0.92,
        body,
        family="monospace",
        fontsize=fontsize,
        va="top",
        ha="left",
        color=STYLE["text_primary"],
        bbox={"facecolor": STYLE["background_light"], "edgecolor": STYLE["edge_mid"], "boxstyle": "round,pad=0.6"},
    )
    _save(fig, out)


def _diagram_repo_structure(paths: RepoPaths, files: dict[str, list[Path]], out: Path) -> None:
    lines = [
        "workspace/",
        "|- AGENTS.md",
        "|- src/quantum/",
        f"|  |- {len(files['quantum_py'])} python modules (operator/core/vqe/drive)",
        "|- test_*.py",
        f"|  |- {len(files['top_tests'])} implementation-contract tests",
        "|- Fermi-Hamil-JW-VQE-TROTTER-PIPELINE/",
        "|  |- pipelines/",
        f"|  |  |- {len(files['pipeline_py'])} python runners",
        f"|  |  |- {len(files['pipeline_sh'])} shell orchestrators",
        "|  |- artifacts/json/",
        "|  |- artifacts/pdf/",
        "|  |- docs/repo_guide_assets/*.png",
        "|  |- docs/repo_implementation_guide.md",
        "|  |- scripts/generate_repo_guide_assets.py",
        "|  |- scripts/build_repo_implementation_guide.sh",
    ]
    _text_box_figure(
        out,
        title="Repository Structure Map",
        subtitle="Implementation workspace and artifact zones",
        body="\n".join(lines),
        fontsize=11.0,
    )


def _diagram_internal_import_dag(summary: dict[str, Any], out: Path) -> None:
    g = nx.DiGraph()
    groups: dict[str, str] = {}

    for mod in summary["modules"]:
        src = mod["module"]
        if src.startswith("src.quantum"):
            groups[src] = "quantum"
        elif src.startswith("Fermi-Hamil-JW-VQE-TROTTER-PIPELINE.pipelines"):
            groups[src] = "pipeline"
        elif src.startswith("test_"):
            groups[src] = "test"
        else:
            groups[src] = "other"

        g.add_node(src)
        for dst in mod["imports"]["internal"]:
            if dst.startswith("src.") or dst.startswith("pipelines"):
                g.add_edge(src, dst)

    keep = [n for n in g.nodes if n.startswith("src.quantum") or "pipelines" in n]
    sub = g.subgraph(keep).copy()
    if not sub.nodes:
        sub.add_node("(no internal edges found)")

    _draw_graph(
        sub,
        path=out,
        title="Internal Import DAG",
        subtitle="src/quantum + pipeline module relationships",
        node_groups=groups,
    )


def _diagram_operator_model(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "PauliLetter\n(pauli_letters_module)",
        "PauliTerm\n(qubitization_module)",
        "PauliPolynomial\n(pauli_polynomial_class)",
        "fermion_plus/minus\n(JW ladder helpers)",
        "pauli_words.PauliTerm\n(compat alias)",
    ]
    g.add_nodes_from(nodes)
    g.add_edge(nodes[0], nodes[1])
    g.add_edge(nodes[1], nodes[2])
    g.add_edge(nodes[1], nodes[4])
    g.add_edge(nodes[2], nodes[3])

    pos = {
        nodes[0]: (-1.5, 0.0),
        nodes[1]: (-0.2, 0.0),
        nodes[2]: (1.2, 0.35),
        nodes[3]: (2.6, 0.35),
        nodes[4]: (1.2, -0.8),
    }

    groups = {nodes[0]: "core", nodes[1]: "core", nodes[2]: "core", nodes[3]: "quantum", nodes[4]: "core"}

    _draw_graph(
        g,
        path=out,
        title="Operator Layer Object Model",
        subtitle="Canonical PauliTerm source and JW helper chain",
        pos=pos,
        node_groups=groups,
    )


def _diagram_qubit_ordering(out: Path) -> None:
    body = textwrap.dedent(
        """
        Example (nq = 6)

        Pauli label string:   e z x y e z
                              | | | | | |
        Character index:      0 1 2 3 4 5
        Physical qubit:       5 4 3 2 1 0

        State bitstring formatting in outputs: q_(n-1)...q_0
        Basis index in statevector: index = sum_q bit_q * 2^q

        Consequence:
        - String position p maps to qubit q = nq - 1 - p.
        - q0 is least significant bit in basis index arithmetic.
        - All occupancy extraction uses (idx >> q) & 1.
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Qubit Ordering and Bit Indexing Convention",
        subtitle="Pauli strings and statevector indexing must agree exactly",
        body=body,
        fontsize=11.0,
    )


def _diagram_jw_flow(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "fermion_plus_operator(JW, nq, j)",
        "fermion_minus_operator(JW, nq, j)",
        "jw_number_operator\n n_p = (I - Z_p)/2",
        "build_hubbard_kinetic/onsite/potential",
        "build_hubbard_hamiltonian\n H = H_t + H_U + H_v",
    ]
    g.add_nodes_from(nodes)
    g.add_edge(nodes[0], nodes[3])
    g.add_edge(nodes[1], nodes[3])
    g.add_edge(nodes[2], nodes[3])
    g.add_edge(nodes[3], nodes[4])

    pos = {
        nodes[0]: (-1.5, 1.0),
        nodes[1]: (-1.5, -0.2),
        nodes[2]: (-1.5, -1.35),
        nodes[3]: (0.8, -0.2),
        nodes[4]: (2.8, -0.2),
    }

    _draw_graph(
        g,
        path=out,
        title="JW Mapping Flow",
        subtitle="Source-of-truth helpers into Hubbard Hamiltonian terms",
        pos=pos,
        node_groups={n: "quantum" for n in nodes},
    )


def _diagram_hardcoded_pipeline(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "parse_args",
        "build_hubbard_hamiltonian",
        "collect coeff_map/labels/matrix",
        "run hardcoded VQE",
        "choose initial state",
        "simulate trajectory\nexact + exact_ansatz + trotter",
        "emit JSON payload",
        "render pipeline PDF",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from([(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)])
    _draw_graph(
        g,
        path=out,
        title="Hardcoded Pipeline Flow",
        subtitle="Execution sequence in pipelines/hardcoded_hubbard_pipeline.py",
    )


def _diagram_qiskit_pipeline(out: Path) -> None:
    fig, ax = _prep_axes(
        "Qiskit Baseline Flow (Static vs Drive)",
        "No-drive uses PauliEvolutionGate branch; drive uses shared numerical kernels",
    )

    boxes = [
        (0.05, 0.79, 0.24, 0.11, "parse_args"),
        (0.35, 0.79, 0.29, 0.11, "build_qiskit_qubit_hamiltonian"),
        (0.68, 0.79, 0.27, 0.11, "run qiskit VQE/QPE"),
        (0.12, 0.50, 0.34, 0.14, "if not has_drive:\nPauliEvolutionGate branch"),
        (0.54, 0.50, 0.34, 0.14, "if has_drive:\nshared trotter kernel"),
        (0.35, 0.25, 0.30, 0.12, "shared exact reference\n(expm_multiply or eig)"),
        (0.35, 0.08, 0.30, 0.11, "aligned JSON/PDF output"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, color=STYLE["blue_fill"], ec=STYLE["edge_mid"], lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=10, color=STYLE["text_primary"])

    arrows = [
        ((0.29, 0.845), (0.35, 0.845)),
        ((0.64, 0.845), (0.68, 0.845)),
        ((0.72, 0.79), (0.30, 0.64)),
        ((0.80, 0.79), (0.71, 0.64)),
        ((0.30, 0.50), (0.47, 0.37)),
        ((0.71, 0.50), (0.53, 0.37)),
        ((0.50, 0.25), (0.50, 0.19)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color=STYLE["edge_mid"]))

    _save(fig, out)


def _diagram_shared_drive_backend(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "hardcoded pipeline",
        "qiskit baseline pipeline",
        "drive_coeff_provider_exyz(t)",
        "_evolve_trotter_suzuki2_absolute",
        "_evolve_piecewise_exact\n(expm_multiply midpoint)",
        "trajectory rows",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            (nodes[0], nodes[2]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[3], nodes[5]),
            (nodes[4], nodes[5]),
        ]
    )
    pos = {
        nodes[0]: (-1.9, 1.0),
        nodes[1]: (-1.9, -0.5),
        nodes[2]: (0.0, 0.2),
        nodes[3]: (1.8, 0.9),
        nodes[4]: (1.8, -0.5),
        nodes[5]: (3.8, 0.2),
    }
    groups = {
        nodes[0]: "pipeline",
        nodes[1]: "pipeline",
        nodes[2]: "quantum",
        nodes[3]: "quantum",
        nodes[4]: "quantum",
        nodes[5]: "artifact",
    }
    _draw_graph(
        g,
        path=out,
        title="Shared Drive Dynamics Backend",
        subtitle="Both implementations converge on common drive-enabled kernels",
        pos=pos,
        node_groups=groups,
    )


def _diagram_compare_fanout(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "compare pipeline",
        "build HC command",
        "build QK command",
        "run HC subprocess",
        "run QK subprocess",
        "load H/Q JSON",
        "compute metrics",
        "emit per-L PDF + summary",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[3], nodes[5]),
            (nodes[4], nodes[5]),
            (nodes[5], nodes[6]),
            (nodes[6], nodes[7]),
        ]
    )
    _draw_graph(
        g,
        path=out,
        title="Compare Pipeline Fan-Out / Fan-In",
        subtitle="Command orchestration and reconciliation path",
    )


def _diagram_amplitude_six_runs(out: Path) -> None:
    fig, ax = _prep_axes(
        "Amplitude Comparison Mode (6 Sub-runs per L)",
        "disabled + A0 + A1 for both hardcoded and qiskit",
    )

    col_x = [0.08, 0.38, 0.68]
    labels = ["disabled", "A0", "A1"]

    for i, x in enumerate(col_x):
        ax.text(x + 0.1, 0.9, labels[i], ha="center", va="center", fontsize=12, fontweight="bold", color=STYLE["text_primary"])

    for i, x in enumerate(col_x):
        rect_h = plt.Rectangle((x, 0.62), 0.22, 0.16, color=STYLE["blue_fill"], ec=STYLE["edge_mid"], lw=1.2)
        rect_q = plt.Rectangle((x, 0.36), 0.22, 0.16, color=STYLE["orange_fill"], ec=STYLE["edge_mid"], lw=1.2)
        ax.add_patch(rect_h)
        ax.add_patch(rect_q)
        ax.text(x + 0.11, 0.70, f"HC run\n{labels[i]}", ha="center", va="center", fontsize=10, color=STYLE["text_primary"])
        ax.text(x + 0.11, 0.44, f"QK run\n{labels[i]}", ha="center", va="center", fontsize=10, color=STYLE["text_primary"])

    out_box = plt.Rectangle((0.30, 0.08), 0.40, 0.16, color=STYLE["green_fill"], ec=STYLE["edge_mid"], lw=1.2)
    ax.add_patch(out_box)
    ax.text(
        0.50,
        0.16,
        "amp_{tag}.pdf + amp_{tag}_metrics.json\n(safe_test + delta_vqe_hc_minus_qk_at_A0/A1)",
        ha="center",
        va="center",
        fontsize=10,
        color=STYLE["text_primary"],
    )

    for x in [0.19, 0.49, 0.79]:
        ax.annotate("", xy=(0.50, 0.24), xytext=(x, 0.36), arrowprops=dict(arrowstyle="->", lw=1.2, color=STYLE["edge_mid"]))

    _save(fig, out)


def _diagram_safe_test_logic(out: Path) -> None:
    fig, ax = _prep_axes(
        "Safe-Test Invariant Logic",
        "A=0 drive trajectory must match no-drive trajectory within threshold",
    )

    steps = [
        (0.08, 0.76, 0.34, 0.14, "Load no-drive HC/QK trajectories"),
        (0.58, 0.76, 0.34, 0.14, "Load A0=0 HC/QK trajectories"),
        (0.33, 0.50, 0.34, 0.14, "Compute max abs deltas\n(fidelity, energy channels)"),
        (0.33, 0.24, 0.34, 0.14, "Pass iff max(delta) < 1e-10"),
    ]

    for x, y, w, h, t in steps:
        ax.add_patch(plt.Rectangle((x, y), w, h, color=STYLE["panel_light"], ec=STYLE["edge_mid"], lw=1.2))
        ax.text(x + w / 2.0, y + h / 2.0, t, ha="center", va="center", fontsize=10, color=STYLE["text_primary"])

    ax.annotate("", xy=(0.50, 0.64), xytext=(0.25, 0.76), arrowprops=dict(arrowstyle="->", lw=1.3, color=STYLE["edge_mid"]))
    ax.annotate("", xy=(0.50, 0.64), xytext=(0.75, 0.76), arrowprops=dict(arrowstyle="->", lw=1.3, color=STYLE["edge_mid"]))
    ax.annotate("", xy=(0.50, 0.38), xytext=(0.50, 0.50), arrowprops=dict(arrowstyle="->", lw=1.3, color=STYLE["edge_mid"]))

    _save(fig, out)


def _diagram_artifact_contract(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "H_{tag}.json",
        "Q_{tag}.json",
        "HvQ_{tag}_metrics.json",
        "HvQ_summary.json",
        "HvQ_{tag}.pdf",
        "HvQ_bundle.pdf",
        "amp_{tag}.pdf",
        "amp_{tag}_metrics.json",
        "commands.txt",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            (nodes[0], nodes[2]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[4], nodes[5]),
            (nodes[6], nodes[5]),
            (nodes[7], nodes[5]),
            (nodes[8], nodes[3]),
        ]
    )

    _draw_graph(
        g,
        path=out,
        title="Artifact and JSON Contract Map",
        subtitle="How generated outputs compose into comparison/report bundles",
    )


def _diagram_test_contracts(files: dict[str, list[Path]], out: Path) -> None:
    tests = [p.name for p in files["top_tests"]]
    bullets = [
        "drive pass-through and default routing",
        "exact-steps multiplier behavior and metadata",
        "energy_total observable semantics and A=0 checks",
        "time-dependent drive waveform behavior",
        "fidelity subspace projector semantics",
        "import-level integration and aliases",
    ]
    body = "Detected tests:\n- " + "\n- ".join(tests)
    body += "\n\nContract domains:\n- " + "\n- ".join(bullets)
    _text_box_figure(
        out,
        title="Test Suite as Implementation Contract",
        subtitle="Top-level tests encode physics-to-code assumptions",
        body=body,
        fontsize=10.5,
    )


def _diagram_extension_playbook(out: Path) -> None:
    fig, ax = _prep_axes(
        "Extension Playbook Decision Tree",
        "Where to change code without violating algebra or drive invariants",
    )

    nodes = [
        (0.38, 0.82, 0.24, 0.10, "Need a change"),
        (0.10, 0.60, 0.32, 0.10, "Operator algebra behavior?"),
        (0.58, 0.60, 0.32, 0.10, "Pipeline/report behavior?"),
        (0.10, 0.38, 0.32, 0.12, "Prefer wrappers/shims\navoid base-file rewrites"),
        (0.58, 0.38, 0.32, 0.12, "Update parse_args + pass-through\n+ docs + tests"),
        (0.34, 0.15, 0.32, 0.12, "Run regressions + safe-test\nvalidate JSON schema stability"),
    ]

    for x, y, w, h, t in nodes:
        ax.add_patch(plt.Rectangle((x, y), w, h, color=STYLE["panel_light"], ec=STYLE["edge_mid"], lw=1.2))
        ax.text(x + w / 2.0, y + h / 2.0, t, ha="center", va="center", fontsize=10, color=STYLE["text_primary"])

    arrows = [
        ((0.50, 0.82), (0.26, 0.70)),
        ((0.50, 0.82), (0.74, 0.70)),
        ((0.26, 0.60), (0.26, 0.50)),
        ((0.74, 0.60), (0.74, 0.50)),
        ((0.26, 0.38), (0.50, 0.27)),
        ((0.74, 0.38), (0.50, 0.27)),
    ]

    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color=STYLE["edge_mid"]))

    _save(fig, out)


def _hf_bitstring(n_sites: int, n_up: int, n_dn: int, ordering: str) -> str:
    nq = 2 * int(n_sites)
    bits = [0] * nq

    if ordering == "blocked":
        up_modes = list(range(int(n_sites)))
        dn_modes = list(range(int(n_sites), 2 * int(n_sites)))
    elif ordering == "interleaved":
        up_modes = [2 * i for i in range(int(n_sites))]
        dn_modes = [2 * i + 1 for i in range(int(n_sites))]
    else:
        raise ValueError(ordering)

    for q in up_modes[: int(n_up)]:
        bits[q] = 1
    for q in dn_modes[: int(n_dn)]:
        bits[q] = 1

    return "".join(str(bits[q]) for q in reversed(range(nq)))


def _diagram_hf_table(out: Path, *, n_sites: int, ordering: str) -> None:
    n_up = (n_sites + 1) // 2
    n_dn = n_sites // 2
    nq = 2 * n_sites
    bitstring = _hf_bitstring(n_sites, n_up, n_dn, ordering)

    fig, ax = _prep_axes(
        f"HF Occupancy Map (L={n_sites}, ordering={ordering})",
        "Half-filling placement of spin orbitals onto qubit bits",
        figsize=(12.5, 8.0),
    )

    rows: list[list[str]] = []
    for site in range(n_sites):
        if ordering == "blocked":
            q_up = site
            q_dn = n_sites + site
        else:
            q_up = 2 * site
            q_dn = 2 * site + 1
        rows.append([str(site), str(q_up), str(1 if site < n_up else 0), str(q_dn), str(1 if site < n_dn else 0)])

    table = ax.table(
        cellText=rows,
        colLabels=["site i", "qubit q(i,up)", "occ_up", "qubit q(i,dn)", "occ_dn"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.7)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(STYLE["edge_mid"])
        if key[0] == 0:
            cell.set_text_props(color=STYLE["text_primary"], fontweight="bold")
            cell.set_facecolor(STYLE["blue_fill"])
        else:
            cell.set_text_props(color=STYLE["text_primary"])
            cell.set_facecolor("white")

    ax.text(
        0.5,
        0.08,
        f"n_up={n_up}, n_dn={n_dn}, N_q={nq}, bitstring q_(Nq-1)...q_0 = {bitstring}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=STYLE["text_primary"],
    )

    _save(fig, out)


def _diagram_bit_index_examples(out: Path) -> None:
    body = textwrap.dedent(
        """
        Basis index extraction examples (N_q = 6)

        idx = 13  -> binary(q5..q0)=001101
        q0 = ((13 >> 0) & 1) = 1
        q1 = ((13 >> 1) & 1) = 0
        q2 = ((13 >> 2) & 1) = 1
        q3 = ((13 >> 3) & 1) = 1
        q4 = ((13 >> 4) & 1) = 0
        q5 = ((13 >> 5) & 1) = 0

        place value: idx = sum_q bit_q * 2^q = 1*2^0 + 0*2^1 + 1*2^2 + 1*2^3

        site occupation expectation in code:
            n_up[site] += up_bit * prob(idx)
            n_dn[site] += dn_bit * prob(idx)

        where prob(idx) = |psi[idx]|^2 and up_bit/dn_bit are extracted with shifts + masks.
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Bit Index and Place-Value Examples",
        subtitle="How basis index integer maps to qubit occupancies",
        body=body,
        fontsize=10.8,
    )


def _diagram_theta_layout(out: Path) -> None:
    fig, ax = _prep_axes(
        "Theta Vector Layout and Generator Assignment",
        "Hardcoded UCCSD ansatz: theta is a real vector indexed by term order across reps",
    )

    ax.add_patch(plt.Rectangle((0.05, 0.72), 0.90, 0.18, color=STYLE["blue_fill"], ec=STYLE["edge_mid"], lw=1.2))
    ax.text(
        0.50,
        0.81,
        "theta = [theta_0, theta_1, ..., theta_(num_parameters-1)]\nnum_parameters = reps * len(base_terms)",
        ha="center",
        va="center",
        fontsize=11,
        color=STYLE["text_primary"],
    )

    x0 = 0.08
    w = 0.12
    labels = ["rep0\nterm0", "rep0\nterm1", "rep0\nterm2", "...", "rep1\nterm0", "rep1\nterm1", "..."]
    for i, lbl in enumerate(labels):
        x = x0 + i * (w + 0.015)
        if x + w > 0.95:
            break
        ax.add_patch(plt.Rectangle((x, 0.42), w, 0.16, color=STYLE["orange_fill"], ec=STYLE["edge_mid"], lw=1.1))
        ax.text(x + w / 2, 0.50, lbl, ha="center", va="center", fontsize=9.5, color=STYLE["text_primary"])

    ax.text(
        0.50,
        0.24,
        "prepare_state: iterate reps, then base_terms, apply exp(-i * theta[k] * G_k), increment k",
        ha="center",
        va="center",
        fontsize=10.5,
        color=STYLE["text_primary"],
        bbox={"facecolor": STYLE["green_fill"], "edgecolor": STYLE["edge_mid"], "boxstyle": "round,pad=0.4"},
    )

    _save(fig, out)


def _diagram_optimizer_flow(out: Path) -> None:
    fig, ax = _prep_axes(
        "Inner Optimizer and Restart Selection",
        "energy_fn(theta)=<psi(theta)|H|psi(theta)>; keep restart with lowest final energy",
    )

    boxes = [
        (0.06, 0.80, 0.88, 0.10, "Initialize RNG(seed), npar, best_energy=+inf"),
        (0.06, 0.62, 0.38, 0.12, "Restart r:\n x0 = 0.3 * Normal(0,1)^npar"),
        (0.56, 0.62, 0.38, 0.12, "SciPy minimize(energy_fn, x0, method, bounds, maxiter)\n(or fallback coordinate search)"),
        (0.06, 0.40, 0.88, 0.12, "Extract theta_opt, energy, nfev, nit, success, message"),
        (0.06, 0.20, 0.88, 0.12, "If energy < best_energy: update best_theta and best_restart"),
    ]

    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, color=STYLE["panel_light"], ec=STYLE["edge_mid"], lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color=STYLE["text_primary"])

    arrows = [
        ((0.50, 0.80), (0.25, 0.74)),
        ((0.50, 0.80), (0.75, 0.74)),
        ((0.25, 0.62), (0.50, 0.52)),
        ((0.75, 0.62), (0.50, 0.52)),
        ((0.50, 0.40), (0.50, 0.32)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color=STYLE["edge_mid"]))

    _save(fig, out)


def _diagram_term_traversal(out: Path) -> None:
    fig, ax = _prep_axes(
        "Term Traversal, Collection, and Ordering",
        "Reordering affects trotter product sequence, not Hamiltonian coefficients",
    )

    boxes = [
        (0.06, 0.78, 0.88, 0.11, "Traverse PauliPolynomial terms -> collect label/coeff map"),
        (0.06, 0.58, 0.40, 0.12, "native order:\nlabels as collected"),
        (0.54, 0.58, 0.40, 0.12, "sorted order:\nlabels lexicographically sorted"),
        (0.06, 0.36, 0.88, 0.12, "Build compiled Pauli actions per label (perm + phase)"),
        (0.06, 0.16, 0.88, 0.12, "Suzuki-2 uses ordered label list in forward/reverse passes"),
    ]

    fills = [STYLE["blue_fill"], STYLE["orange_fill"], STYLE["purple_fill"], STYLE["green_fill"], STYLE["panel_light"]]
    for (x, y, w, h, label), fill in zip(boxes, fills):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=fill, ec=STYLE["edge_mid"], lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10.2, color=STYLE["text_primary"])

    arrows = [
        ((0.50, 0.78), (0.26, 0.70)),
        ((0.50, 0.78), (0.74, 0.70)),
        ((0.26, 0.58), (0.50, 0.48)),
        ((0.74, 0.58), (0.50, 0.48)),
        ((0.50, 0.36), (0.50, 0.28)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color=STYLE["edge_mid"]))

    _save(fig, out)


def _diagram_suzuki2(out: Path) -> None:
    body = textwrap.dedent(
        """
        Suzuki-2 step with N trotter slices (dt = t/N)

            U_S2(dt) = [prod_j exp(-i c_j P_j dt/2)] [prod_j^rev exp(-i c_j P_j dt/2)]

        Implementation shape:
        1) choose ordered_labels (native or sorted)
        2) for each slice k:
             a) evaluate drive coefficients at sampling time
             b) forward pass over ordered labels, half-angle
             c) reverse pass over ordered labels, half-angle
        3) normalize diagnostics / continue

        Ordering impact:
        - finite dt: non-commuting term sequence changes local error terms
        - dt -> 0: ordering sensitivity shrinks (Trotter limit)
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Suzuki-2 Step Anatomy",
        subtitle="Forward/reverse product structure used by trotter evolution",
        body=body,
        fontsize=10.6,
    )


def _diagram_drive_waveform(out: Path) -> None:
    fig, ax = _prep_axes(
        "Drive Waveform Decomposition",
        "v(t)=A sin(omega t + phi) exp(-(t-t0)^2 / (2 tbar^2))",
    )

    t = np.linspace(-4.0, 4.0, 500)
    A = 0.3
    omega = 2.0
    phi = 0.25
    t0 = 0.5
    tbar = 1.2

    carrier = A * np.sin(omega * t + phi)
    env = np.exp(-((t - t0) ** 2) / (2.0 * tbar * tbar))
    wave = carrier * env

    ax.plot(t, wave, color="#1F77B4", linewidth=1.8, label="waveform v(t)")
    ax.plot(t, A * env, color="#D62728", linestyle="--", linewidth=1.2, label="+A envelope")
    ax.plot(t, -A * env, color="#D62728", linestyle="--", linewidth=1.2, label="-A envelope")
    ax.plot(t, carrier, color="#2CA02C", linewidth=1.0, alpha=0.55, label="carrier A sin(omega t+phi)")
    ax.axhline(0.0, color=STYLE["edge_mid"], linewidth=0.8)
    ax.grid(alpha=0.25)
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8, loc="upper right")

    _save(fig, out)


def _diagram_drive_patterns(out: Path) -> None:
    fig, ax = _prep_axes(
        "Drive Spatial Pattern Weights",
        "Site weights s_j used in v_j(t) = s_j * v(t)",
    )

    sites = np.arange(8)
    staggered = np.array([1 if i % 2 == 0 else -1 for i in sites], dtype=float)
    dimer_bias = np.array([1 if i % 2 == 0 else -1 for i in sites], dtype=float)
    custom = np.array([1.0, 0.4, -0.2, -0.8, 0.8, 0.2, -0.4, -1.0], dtype=float)

    ax.plot(sites, staggered, marker="o", label="staggered", color="#1F77B4")
    ax.plot(sites, dimer_bias, marker="s", label="dimer_bias", color="#FF7F0E")
    ax.plot(sites, custom, marker="^", label="custom example", color="#2CA02C")
    ax.set_xlabel("site j")
    ax.set_ylabel("s_j")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    _save(fig, out)


def _diagram_reference_split(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "has_drive?",
        "no drive",
        "drive enabled",
        "eigendecomposition\nexact static propagator",
        "piecewise exact\nexpm_multiply midpoint",
        "reference_method_name",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[3], nodes[5]),
            (nodes[4], nodes[5]),
        ]
    )
    pos = {
        nodes[0]: (0.0, 0.5),
        nodes[1]: (-1.3, 0.0),
        nodes[2]: (1.3, 0.0),
        nodes[3]: (-1.3, -1.0),
        nodes[4]: (1.3, -1.0),
        nodes[5]: (0.0, -2.0),
    }
    groups = {
        nodes[0]: "pipeline",
        nodes[1]: "pipeline",
        nodes[2]: "pipeline",
        nodes[3]: "quantum",
        nodes[4]: "quantum",
        nodes[5]: "artifact",
    }
    _draw_graph(
        g,
        path=out,
        title="Reference Propagator Split",
        subtitle="Static exact branch vs drive-enabled piecewise exact branch",
        pos=pos,
        node_groups=groups,
    )


def _diagram_trajectory_schema(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "trajectory row",
        "time",
        "fidelity",
        "energy_static_*",
        "energy_total_*",
        "n_up_site0_* / n_dn_site0_*",
        "doublon_*",
        "staggered_*",
        "n_site arrays",
        "plot pages",
    ]
    g.add_nodes_from(nodes)
    for n in nodes[1:-1]:
        g.add_edge(nodes[0], n)
        g.add_edge(n, nodes[-1])
    _draw_graph(
        g,
        path=out,
        title="Trajectory Row Schema to Plot Binding",
        subtitle="Every plotted channel is sourced directly from trajectory keys",
    )


def _diagram_formula_legend_energy(out: Path) -> None:
    body = textwrap.dedent(
        """
        Energy channels used in trajectory and plots

        energy_static_exact(t)  = <psi_exact_gs_ref(t)|H_static|psi_exact_gs_ref(t)>
        energy_static_exact_ans = <psi_exact_ansatz_ref(t)|H_static|psi_exact_ansatz_ref(t)>
        energy_static_trotter   = <psi_ansatz_trot(t)|H_static|psi_ansatz_trot(t)>

        energy_total_exact(t)   = <psi_exact_gs_ref(t)|H_static + H_drive(t0+t)|psi_exact_gs_ref(t)>
        energy_total_exact_ans  = <psi_exact_ansatz_ref(t)|H_static + H_drive(t0+t)|psi_exact_ansatz_ref(t)>
        energy_total_trotter    = <psi_ansatz_trot(t)|H_static + H_drive(t0+t)|psi_ansatz_trot(t)>

        When drive is disabled: energy_total_* == energy_static_*
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Formula Legend: Energy Channels",
        subtitle="Exact semantics used by pipeline output and figures",
        body=body,
        fontsize=10.0,
    )


def _diagram_formula_legend_fidelity(out: Path) -> None:
    body = textwrap.dedent(
        """
        Fidelity channel

        fidelity(t) = <psi_ansatz_trot(t) | P_exact_gs_subspace(t) | psi_ansatz_trot(t)>

        where P_exact_gs_subspace(t) projects onto the propagated ground-manifold basis
        selected at t=0 from the filtered sector (N_up, N_dn) with energy tolerance:

            E <= E0 + fidelity_subspace_energy_tol

        This is not full-state overlap to a single vector; it is projector fidelity
        against a (possibly multi-dimensional) reference subspace.
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Formula Legend: Fidelity",
        subtitle="Projector fidelity against filtered-sector propagated ground manifold",
        body=body,
        fontsize=10.4,
    )


def _diagram_formula_legend_occ(out: Path) -> None:
    body = textwrap.dedent(
        """
        Occupation and doublon channels

        For basis index idx with probability p(idx)=|psi[idx]|^2:

            n_up[site] += up_bit(site, idx) * p(idx)
            n_dn[site] += dn_bit(site, idx) * p(idx)

        Site-0 channels in trajectory:
            n_up_site0_* = n_up[0]
            n_dn_site0_* = n_dn[0]

        Doublon:
            doublon_total = sum_site <n_{site,up} n_{site,dn}>
            doublon_avg   = doublon_total / L

        Staggered order from total site density n_site:
            staggered = (1/L) * sum_i (-1)^i * n_site[i]
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Formula Legend: Occupancy, Doublon, Staggered",
        subtitle="Observable definitions implemented in trajectory loop",
        body=body,
        fontsize=10.1,
    )


def _load_case_payload(paths: RepoPaths, rel_path: str) -> dict[str, Any] | None:
    p = paths.subrepo_root / rel_path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _artifact_lineplot(out: Path, *, title: str, subtitle: str, x: np.ndarray, ys: list[tuple[str, np.ndarray, str]], ylabel: str) -> None:
    fig, ax = _prep_axes(title, subtitle, figsize=(12.0, 7.0))
    for label, y, color in ys:
        ax.plot(x, y, label=label, linewidth=1.4, color=color)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out)


def _artifact_error_plot(out: Path, *, title: str, subtitle: str, x: np.ndarray, errs: list[tuple[str, np.ndarray, str]]) -> None:
    fig, ax = _prep_axes(title, subtitle, figsize=(12.0, 7.0))
    for label, y, color in errs:
        ax.plot(x, y, label=label, linewidth=1.3, color=color)
    ax.set_xlabel("time")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    _save(fig, out)


def _diagram_case_audits(paths: RepoPaths, rel_path: str, outputs: dict[str, Path]) -> None:
    payload = _load_case_payload(paths, rel_path)
    if payload is None:
        for _, path in outputs.items():
            _text_box_figure(path, title="Missing Artifact", subtitle=rel_path, body="Artifact file not found.")
        return

    traj = payload.get("trajectory", [])
    if not traj:
        for _, path in outputs.items():
            _text_box_figure(path, title="Empty Trajectory", subtitle=rel_path, body="Trajectory is empty.")
        return

    t = _trajectory_array(traj, "time")
    e_stat_exact = _trajectory_array(traj, "energy_static_exact")
    e_stat_ans = _trajectory_array(traj, "energy_static_exact_ansatz")
    e_stat_trot = _trajectory_array(traj, "energy_static_trotter")
    e_tot_exact = _trajectory_array(traj, "energy_total_exact")
    e_tot_trot = _trajectory_array(traj, "energy_total_trotter")
    n_up0 = _trajectory_array(traj, "n_up_site0_trotter")
    n_dn0 = _trajectory_array(traj, "n_dn_site0_trotter")
    fidelity = _trajectory_array(traj, "fidelity")

    if "energy" in outputs:
        _artifact_lineplot(
            outputs["energy"],
            title=f"Artifact Audit: {Path(rel_path).name} Energy Channels",
            subtitle="Recomputed directly from JSON trajectory arrays",
            x=t,
            ys=[
                ("static exact", e_stat_exact, "#1f77b4"),
                ("static exact ansatz", e_stat_ans, "#2ca02c"),
                ("static trotter", e_stat_trot, "#d62728"),
                ("total exact", e_tot_exact, "#17becf"),
                ("total trotter", e_tot_trot, "#ff7f0e"),
            ],
            ylabel="energy",
        )

    if "site0" in outputs:
        _artifact_lineplot(
            outputs["site0"],
            title=f"Artifact Audit: {Path(rel_path).name} Site-0 Channels",
            subtitle="n_up0, n_dn0, and fidelity from trotter branch",
            x=t,
            ys=[
                ("n_up0 trotter", n_up0, "#1f77b4"),
                ("n_dn0 trotter", n_dn0, "#9467bd"),
                ("fidelity", fidelity, "#2ca02c"),
            ],
            ylabel="value",
        )

    if "error" in outputs:
        err1 = np.abs(e_stat_trot - e_stat_ans)
        err2 = np.abs(e_tot_trot - e_tot_exact)
        _artifact_error_plot(
            outputs["error"],
            title=f"Artifact Audit: {Path(rel_path).name} Energy Errors",
            subtitle="Absolute error channels on log scale",
            x=t,
            errs=[
                ("|E_static_trot - E_static_exact_ans|", err1 + 1e-18, "#d62728"),
                ("|E_total_trot - E_total_exact|", err2 + 1e-18, "#ff7f0e"),
            ],
        )


def _diagram_case_summary_from_metrics(out: Path, metrics: dict[str, Any]) -> None:
    cases = metrics.get("cases", [])
    valid_cases = [c for c in cases if c.get("valid")]
    if not valid_cases:
        _text_box_figure(
            out,
            title="Canonical Case Comparison",
            subtitle="No valid cases available",
            body="Artifact metrics did not contain valid canonical cases.",
        )
        return

    labels = [c.get("case", "?") for c in valid_cases]
    vqe_err = [float(c.get("vqe", {}).get("abs_error", float("nan"))) for c in valid_cases]
    emax = [float(c.get("energy_errors", {}).get("static_trot_vs_exact_ans_max", float("nan"))) for c in valid_cases]
    fid_final = [float(c.get("fidelity", {}).get("final", float("nan"))) for c in valid_cases]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = _prep_axes(
        "Canonical Case Comparison Summary",
        "Derived metrics from L2/L3/L4 canonical artifacts",
        figsize=(12.5, 7.5),
    )

    ax.bar(x - w, np.array(vqe_err, dtype=float), width=w, label="|VQE - exact_filtered|", color="#2ca02c")
    ax.bar(x, np.array(emax, dtype=float), width=w, label="max |E_trot - E_exact_ans|", color="#d62728")
    ax.bar(x + w, np.array(fid_final, dtype=float), width=w, label="final fidelity", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("metric value")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="best")

    _save(fig, out)


def _diagram_function_line_spans(out: Path, evidence: dict[str, Any]) -> None:
    recs = evidence.get("function_records", [])
    if not recs:
        _text_box_figure(out, title="Function Evidence", subtitle="No records", body="No implementation evidence records found.")
        return

    rows = []
    for r in recs:
        rows.append(
            f"{r['function']:<34} {r['file']:<65} L{int(r['line_start'])}-{int(r['line_end'])}"
        )

    body = "\n".join(rows)
    _text_box_figure(
        out,
        title="Function Evidence Map",
        subtitle="Target implementation functions with file and line spans",
        body=body,
        fontsize=9.2,
    )


def _diagram_call_graph_focus(out: Path, evidence: dict[str, Any]) -> None:
    g = nx.DiGraph()
    for rec in evidence.get("function_records", []):
        g.add_node(rec.get("function", "?"))
    for edge in evidence.get("call_graph_edges", []):
        g.add_edge(edge["src"], edge["dst"])

    if not g.nodes:
        g.add_node("(no evidence)")

    groups = {n: "quantum" for n in g.nodes}
    _draw_graph(
        g,
        path=out,
        title="Function Call Graph (Targeted)",
        subtitle="Call edges among targeted implementation functions",
        node_groups=groups,
    )


def _diagram_invariant_evidence(out: Path, invariants: dict[str, Any]) -> None:
    lines = []
    for name, payload in sorted(invariants.items()):
        found = payload.get("found")
        lines.append(f"{name}: found={found}")
        ev = payload.get("evidence", [])
        for e in ev[:2]:
            lines.append(f"  - {e.get('file')}:{e.get('line')}  {e.get('snippet')}")
        lines.append("")

    _text_box_figure(
        out,
        title="Invariant Evidence Snippets",
        subtitle="Detected invariant contracts with line-level evidence",
        body="\n".join(lines).strip(),
        fontsize=8.9,
    )


def _diagram_metrics_table(out: Path, metrics: dict[str, Any]) -> None:
    rows: list[list[str]] = []
    for case in metrics.get("cases", []):
        rows.append(
            [
                case.get("case", "?"),
                str(case.get("valid", False)),
                str(case.get("settings", {}).get("L", "?")),
                str(case.get("settings", {}).get("has_drive", "?")),
                f"{_safe_float(case.get('vqe', {}).get('abs_error', float('nan'))):.3e}",
                f"{_safe_float(case.get('energy_errors', {}).get('static_trot_vs_exact_ans_max', float('nan'))):.3e}",
            ]
        )

    fig, ax = _prep_axes(
        "Canonical Artifact Metrics Table",
        "Validation and error summary for L2/L3/L4 canonical cases",
        figsize=(12.8, 6.8),
    )

    table = ax.table(
        cellText=rows,
        colLabels=["case", "valid", "L", "drive", "|VQE-exact|", "max |E_trot-E_exact_ans|"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(STYLE["edge_mid"])
        if key[0] == 0:
            cell.set_text_props(color=STYLE["text_primary"], fontweight="bold")
            cell.set_facecolor(STYLE["blue_fill"])
        else:
            cell.set_text_props(color=STYLE["text_primary"])
            cell.set_facecolor("white")

    _save(fig, out)


def _diagram_quality_gate_summary(out: Path) -> None:
    body = textwrap.dedent(
        """
        Build quality gates (guide-level)

        - operator-core immutability gate
        - deterministic diagram regeneration gate
        - required section heading gate
        - local link/path resolution gate
        - figure embedding count gate
        - canonical artifact presence gate
        - evidence anchor presence gate
        - run appendix size cap gate
        - pytest snapshot capture gate
        - PDF page-count range gate
        """
    ).strip("\n")
    _text_box_figure(
        out,
        title="Guide Build Quality Gates",
        subtitle="Build script enforces documentation correctness and reproducibility",
        body=body,
        fontsize=10.8,
    )


def _diagram_plot_meaning_map(out: Path) -> None:
    g = nx.DiGraph()
    nodes = [
        "trajectory arrays",
        "3D density surfaces",
        "3D scalar lanes",
        "error heatmaps",
        "drive waveform/spectrum",
        "2D summary page",
        "focused static/total energy pages",
        "appendix text pages",
    ]
    g.add_nodes_from(nodes)
    g.add_edges_from(
        [
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[0], nodes[3]),
            (nodes[0], nodes[4]),
            (nodes[0], nodes[5]),
            (nodes[0], nodes[6]),
            (nodes[5], nodes[7]),
        ]
    )
    _draw_graph(
        g,
        path=out,
        title="Plot Generation Meaning Map",
        subtitle="How trajectory channels feed every PDF page family",
    )


def _generate_diagrams(
    paths: RepoPaths,
    summary: dict[str, Any],
    files: dict[str, list[Path]],
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    diagrams: list[tuple[str, str, str, Any]] = [
        ("01_repo_structure_map.png", "Repository Structure Map", "Workspace and artifact topology", lambda out: _diagram_repo_structure(paths, files, out)),
        ("02_internal_import_dag.png", "Internal Import DAG", "src/quantum and pipeline imports", lambda out: _diagram_internal_import_dag(summary, out)),
        ("03_operator_layer_object_model.png", "Operator Layer Object Model", "Pauli abstraction chain", _diagram_operator_model),
        ("04_qubit_ordering_convention.png", "Qubit Ordering Convention", "String to qubit index mapping", _diagram_qubit_ordering),
        ("05_jw_mapping_flow.png", "JW Mapping Flow", "Ladder helper flow into Hamiltonian", _diagram_jw_flow),
        ("06_hardcoded_pipeline_flow.png", "Hardcoded Pipeline Flow", "Execution sequence", _diagram_hardcoded_pipeline),
        ("07_qiskit_pipeline_static_vs_drive.png", "Qiskit Baseline Flow", "Static vs drive branch", _diagram_qiskit_pipeline),
        ("08_shared_drive_backend.png", "Shared Drive Backend", "Common drive kernels", _diagram_shared_drive_backend),
        ("09_compare_pipeline_fanout_fanin.png", "Compare Fan-Out/Fan-In", "Orchestration path", _diagram_compare_fanout),
        ("10_amplitude_comparison_six_runs.png", "Amplitude 6-Run Workflow", "disabled/A0/A1 x HC/QK", _diagram_amplitude_six_runs),
        ("11_safe_test_invariant_logic.png", "Safe-Test Logic", "A=0 no-drive equivalence gate", _diagram_safe_test_logic),
        ("12_artifact_json_contract_map.png", "Artifact Contract Map", "JSON/PDF dependency graph", _diagram_artifact_contract),
        ("13_test_contract_coverage.png", "Test Contract Coverage", "Tests as executable spec", lambda out: _diagram_test_contracts(files, out)),
        ("14_extension_playbook_decision_tree.png", "Extension Playbook", "Safe change decision tree", _diagram_extension_playbook),
        ("15_hf_blocked_L2_table.png", "HF Occupancy Table L2 blocked", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=2, ordering="blocked")),
        ("16_hf_blocked_L3_table.png", "HF Occupancy Table L3 blocked", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=3, ordering="blocked")),
        ("17_hf_blocked_L4_table.png", "HF Occupancy Table L4 blocked", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=4, ordering="blocked")),
        ("18_hf_interleaved_L2_table.png", "HF Occupancy Table L2 interleaved", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=2, ordering="interleaved")),
        ("19_hf_interleaved_L3_table.png", "HF Occupancy Table L3 interleaved", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=3, ordering="interleaved")),
        ("20_hf_interleaved_L4_table.png", "HF Occupancy Table L4 interleaved", "Half-filling map", lambda out: _diagram_hf_table(out, n_sites=4, ordering="interleaved")),
        ("21_bit_index_place_value_examples.png", "Bit Index Examples", "Basis index extraction", _diagram_bit_index_examples),
        ("22_theta_vector_layout.png", "Theta Vector Layout", "Parameter indexing in ansatz", _diagram_theta_layout),
        ("23_optimizer_restart_flow.png", "Optimizer Restart Flow", "Inner VQE optimization logic", _diagram_optimizer_flow),
        ("24_term_traversal_vs_ordering.png", "Term Traversal vs Ordering", "collection/order/compiled actions", _diagram_term_traversal),
        ("25_suzuki2_step_anatomy.png", "Suzuki-2 Anatomy", "Forward/reverse pass implementation", _diagram_suzuki2),
        ("26_drive_waveform_decomposition.png", "Drive Waveform", "Carrier-envelope decomposition", _diagram_drive_waveform),
        ("27_drive_spatial_patterns.png", "Drive Spatial Patterns", "staggered/dimer/custom weights", _diagram_drive_patterns),
        ("28_reference_propagator_split.png", "Reference Propagator Split", "static eig vs drive piecewise exact", _diagram_reference_split),
        ("29_trajectory_row_schema_map.png", "Trajectory Schema Map", "row keys to plot channels", _diagram_trajectory_schema),
        ("30_formula_legend_energy.png", "Formula Legend Energy", "energy observable definitions", _diagram_formula_legend_energy),
        ("31_formula_legend_fidelity.png", "Formula Legend Fidelity", "projector fidelity definition", _diagram_formula_legend_fidelity),
        ("32_formula_legend_occupancy_doublon.png", "Formula Legend Occupancy", "n_up/n_dn/doublon/staggered", _diagram_formula_legend_occ),
        (
            "33_artifact_L2_energy_audit.png",
            "Artifact L2 Energy Audit",
            "L2 heavy static energy channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[0], {"energy": out}),
        ),
        (
            "34_artifact_L2_site0_audit.png",
            "Artifact L2 Site0 Audit",
            "L2 heavy static site-0/fidelity channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[0], {"site0": out}),
        ),
        (
            "35_artifact_L3_energy_audit.png",
            "Artifact L3 Energy Audit",
            "L3 heavy static energy channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[1], {"energy": out}),
        ),
        (
            "36_artifact_L3_site0_audit.png",
            "Artifact L3 Site0 Audit",
            "L3 heavy static site-0/fidelity channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[1], {"site0": out}),
        ),
        (
            "37_artifact_L4_total_energy_drive_audit.png",
            "Artifact L4 Drive Energy Audit",
            "L4 drive-enabled static/total energy channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[2], {"energy": out}),
        ),
        (
            "38_artifact_L4_drive_waveform_audit.png",
            "Artifact L4 Site0/Fidelity Audit",
            "L4 drive-enabled site-0/fidelity channels",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[2], {"site0": out}),
        ),
        (
            "39_artifact_L4_reference_method_audit.png",
            "Artifact L4 Error Audit",
            "L4 drive energy absolute errors",
            lambda out: _diagram_case_audits(paths, CANONICAL_CASES[2], {"error": out}),
        ),
        ("40_case_comparison_summary.png", "Case Comparison Summary", "L2/L3/L4 derived metric comparison", lambda out: _diagram_case_summary_from_metrics(out, metrics)),
        ("41_function_line_span_map.png", "Function Evidence Map", "Target function line spans", lambda out: _diagram_function_line_spans(out, summary["implementation_evidence"])),
        ("42_function_call_graph_focus.png", "Function Call Graph", "Call edges among target functions", lambda out: _diagram_call_graph_focus(out, summary["implementation_evidence"])),
        ("43_invariant_evidence_snippets.png", "Invariant Evidence", "Line-level invariant snippets", lambda out: _diagram_invariant_evidence(out, summary["invariants"])),
        ("44_canonical_artifact_metrics_table.png", "Canonical Metrics Table", "Validation/error summary table", lambda out: _diagram_metrics_table(out, metrics)),
        ("45_quality_gate_summary.png", "Quality Gate Summary", "Guide build gate inventory", _diagram_quality_gate_summary),
        ("46_plot_meaning_map.png", "Plot Meaning Map", "Trajectory arrays to output pages", _diagram_plot_meaning_map),
    ]

    manifest: list[dict[str, str]] = []
    for fname, title, desc, fn in diagrams:
        out = paths.assets_dir / fname
        fn(out)
        manifest.append({"file": str(out.relative_to(paths.subrepo_root)), "title": title, "description": desc})

    return manifest


def generate(paths: RepoPaths) -> dict[str, Any]:
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    paths.assets_dir.mkdir(parents=True, exist_ok=True)

    files = _source_files(paths)
    extracted = _collect_module_summary(paths, files)
    invariants = _extract_invariants(paths)
    evidence = _extract_implementation_evidence(paths)
    artifact_metrics = _extract_artifact_metrics(paths)

    metrics_json_path = paths.assets_dir / "repo_guide_artifact_metrics.json"
    metrics_json_path.write_text(json.dumps(artifact_metrics, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(paths.workspace_root),
        "subrepo_root": str(paths.subrepo_root),
        "counts": {
            "quantum_modules": len(files["quantum_py"]),
            "pipeline_python": len(files["pipeline_py"]),
            "pipeline_shell": len(files["pipeline_sh"]),
            "top_level_tests": len(files["top_tests"]),
            "analyzed_python_files": len(files["analysis_py"]),
        },
        "files": {
            "quantum_py": [_rel(p, paths.workspace_root) for p in files["quantum_py"]],
            "pipeline_py": [_rel(p, paths.workspace_root) for p in files["pipeline_py"]],
            "pipeline_sh": [_rel(p, paths.workspace_root) for p in files["pipeline_sh"]],
            "top_tests": [_rel(p, paths.workspace_root) for p in files["top_tests"]],
        },
        "style_palette": STYLE,
        "canonical_artifacts": CANONICAL_CASES,
        "artifact_metrics_json": str(metrics_json_path.relative_to(paths.subrepo_root)),
        **extracted,
        "invariants": invariants,
        "implementation_evidence": evidence,
    }

    diagram_manifest = _generate_diagrams(paths, summary, files, artifact_metrics)
    summary["diagram_manifest"] = diagram_manifest

    out_json = paths.assets_dir / "repo_guide_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate repo implementation-guide assets.")
    parser.add_argument(
        "--subrepo-root",
        type=Path,
        default=None,
        help="Optional override for subrepo root (defaults to script parent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subrepo_root is not None:
        subrepo_root = args.subrepo_root.resolve()
        script_like = (subrepo_root / "scripts" / "generate_repo_guide_assets.py").resolve()
        paths = _resolve_repo_paths(script_like)
    else:
        paths = _resolve_repo_paths(Path(__file__))

    summary = generate(paths)
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_json": str(paths.assets_dir / "repo_guide_summary.json"),
                "artifact_metrics_json": str(paths.assets_dir / "repo_guide_artifact_metrics.json"),
                "diagram_count": len(summary.get("diagram_manifest", [])),
                "analyzed_python_files": summary.get("counts", {}).get("analyzed_python_files"),
                "target_functions_found": len(summary.get("implementation_evidence", {}).get("found_functions", [])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
