#!/usr/bin/env python3
"""Generate a long, code/docs-only HH noise-model repository guide PDF.

This runner is documentation-oriented: it reads code/docs metadata and builds
an exhaustive noise-stack guide (no trajectory execution required).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.pdf_utils import (
    current_command_string,
    get_PdfPages,
    get_plt,
    render_compact_table,
    render_parameter_manifest,
    render_text_page,
    require_matplotlib,
)

ROOT_MODULES: list[str] = [
    "pipelines.exact_bench.noise_oracle_runtime",
    "pipelines.exact_bench.hh_noise_hardware_validation",
    "pipelines.exact_bench.hh_noise_robustness_seq_report",
    "pipelines.exact_bench.hh_seq_transition_utils",
]

DOC_REFERENCE_PATHS: list[Path] = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "pipelines" / "run_guide.md",
    REPO_ROOT / "docs" / "repo_implementation_guide.md",
    REPO_ROOT / "docs" / "LLM_RESEARCH_CONTEXT.md",
    REPO_ROOT / "docs" / "HH_IMPLEMENTATION_STATUS.md",
    REPO_ROOT / "pipelines" / "exact_bench" / "README.md",
]

REQUIRED_SECTION_MARKERS: list[str] = [
    "SECTION: SCOPE + SOURCES",
    "SECTION: ARCHITECTURE MAP",
    "SECTION: ORACLE CONTRACT",
    "SECTION: VALIDATION PIPELINE CONTRACT",
    "SECTION: ROBUSTNESS REPORT CONTRACT",
    "SECTION: TEST + DOC COVERAGE",
    "SECTION: REFACTOR PLAYBOOK",
    "SECTION: APPENDIX SYMBOL INDEX",
    "SECTION: APPENDIX CLI FLAG INDEX",
    "SECTION: APPENDIX IMPORT EDGES",
    "SECTION: COMMAND + PROVENANCE",
]


def _human_front_matter_pages() -> list[dict[str, Any]]:
    """Human-readable front section pages prepended before deep technical appendices."""
    return [
        {
            "title": "Purpose And Audience",
            "summary": "This guide explains how noise behavior is implemented and validated in this repository.",
            "bullets": [
                "Primary audience: engineers extending/refactoring noise paths.",
                "Secondary audience: reviewers verifying contracts and regressions.",
                "Tertiary audience: automation/LLM tools consuming stable metadata.",
                "This front section is task-first; later sections remain exhaustive references.",
            ],
        },
        {
            "title": "One-Minute Noise Model Map",
            "summary": "Four core modules define the noise stack end-to-end.",
            "bullets": [
                "noise_oracle_runtime: backend + estimator orchestration.",
                "hh_noise_hardware_validation: single-case noisy validation runner.",
                "hh_noise_robustness_seq_report: staged long-form robustness report.",
                "hh_seq_transition_utils: shared transition/pool/drive helper contracts.",
            ],
        },
        {
            "title": "How To Use This Document",
            "summary": "Read the front pages first, then jump into sectioned references as needed.",
            "bullets": [
                "If you are changing behavior, read Refactor Playbook and tests first.",
                "If you are diagnosing a run, read Oracle + Validation contracts first.",
                "If you need complete symbol coverage, use appendix indices near the end.",
                "Section markers are stable for scripted PDF text-gate checks.",
            ],
        },
        {
            "title": "Mode Selection (Practical)",
            "summary": "Pick noise mode by intent before tuning any optimization knobs.",
            "bullets": [
                "ideal: deterministic baseline and parity checks.",
                "shots: local stochastic estimator behavior without device model.",
                "aer_noise: backend-derived noise model emulation.",
                "runtime: real IBM Runtime backend execution path.",
            ],
        },
        {
            "title": "Inputs You Must Hold Constant",
            "summary": "Noise-mode comparisons are only meaningful if physics settings are fixed.",
            "bullets": [
                "Keep L, t, U, dv, omega0, g_ep, n_ph_max unchanged across modes.",
                "Keep ordering, boundary, and ansatz consistent across comparisons.",
                "Keep trajectory grid knobs fixed (t_final, num_times, trotter_steps).",
                "Change one axis at a time to isolate estimator/backend effects.",
            ],
        },
        {
            "title": "Reproducibility Checklist",
            "summary": "Seeds and repeats define stochastic reproducibility envelopes.",
            "bullets": [
                "Set seed/noise-seed and vqe/adapt seeds explicitly.",
                "Record shots, oracle_repeats, and oracle_aggregate every run.",
                "Capture exact command string and generated timestamp in artifacts.",
                "Persist JSON sidecar so comparisons are machine-auditable.",
            ],
        },
        {
            "title": "Backend Resolution Rules",
            "summary": "Backend resolution behavior differs by mode and fake-backend controls.",
            "bullets": [
                "ideal mode uses statevector estimator path.",
                "shots/aer_noise prefer qiskit_aer estimator paths locally.",
                "runtime requires IBM credentials and backend-name availability.",
                "use-fake-backend drives deterministic fake-provider noise simulation.",
            ],
        },
        {
            "title": "OMP/SHM Failure Behavior",
            "summary": "Local Aer can fail in restricted environments; fallback logic is explicit.",
            "bullets": [
                "OMP/SHM aborts are treated as environment-level failures, not physics bugs.",
                "allow-aer-fallback enables sampler_shots fallback for those failures.",
                "omp-shm-workaround applies environment controls for Aer startup resilience.",
                "Fallback details are emitted into backend/execution_fallback JSON fields.",
            ],
        },
        {
            "title": "Legacy Parity (Why It Matters)",
            "summary": "Legacy parity ensures new noiseless-estimator paths preserve old physics outputs.",
            "bullets": [
                "Requires exact time-grid match, not just close trajectory lengths.",
                "Uses strict max_abs_delta gate against selected observables.",
                "Per-observable delta metrics are stored in legacy_parity payload blocks.",
                "Comparison plot output is available for quick visual forensic checks.",
            ],
        },
        {
            "title": "Validation Runner Outputs",
            "summary": "Single-run validation emits a compact but comprehensive artifact set.",
            "bullets": [
                "JSON includes settings, noise_config, backend, vqe/adapt, trajectory.",
                "PDF is manifest-first with summary and trajectory/noisy-vs-ideal pages.",
                "Optional compare plot overlays new outputs with legacy references.",
                "Execution fallback metadata is explicit for postmortem review.",
            ],
        },
        {
            "title": "Reading Trajectory Delta Fields",
            "summary": "Noise analysis hinges on measured-minus-ideal deltas per observable.",
            "bullets": [
                "energy_static_* and energy_total_* deltas track estimator/noise bias.",
                "site-0 occupation deltas expose spin-channel sensitivity.",
                "doublon/staggered deltas often surface nontrivial response drift.",
                "Always interpret delta panels with mode, shots, and repeats context.",
            ],
        },
        {
            "title": "Staged Robustness Flow",
            "summary": "Long-form robustness report chains warm-start, ADAPT, and final VQE.",
            "bullets": [
                "Transition logic tracks |DeltaE| slope windows for stage handoff.",
                "Pool B is strict-union constrained and audited for provenance.",
                "Noiseless matrix benchmarks integrators before noisy overlays.",
                "Noisy matrix then adds per-mode estimator/backend effects.",
            ],
        },
        {
            "title": "Noisy Method Matrix",
            "summary": "Noisy dynamics can be benchmarked per method and per mode.",
            "bullets": [
                "Default methods: suzuki2 and cfqm4; cfqm6 optional.",
                "Profiles include static and drive where configured.",
                "Success/failure is tracked per profile/method/mode branch.",
                "Mode availability failures are reported as structured diagnostics.",
            ],
        },
        {
            "title": "Benchmark Cost Metrics",
            "summary": "Hardware-oriented proxy costs are tracked beside runtime totals.",
            "bullets": [
                "term_exp_count_total, pauli_rot_count_total, cx_proxy_total.",
                "sq_proxy_total, depth_proxy_total for additional complexity signals.",
                "wall_total_s and oracle_eval_s_total track runtime decomposition.",
                "Use proxy budgets as primary comparison axis; wall time is secondary.",
            ],
        },
        {
            "title": "Test Coverage Strategy",
            "summary": "Tests enforce parser, fallback, parity, and benchmark-contract invariants.",
            "bullets": [
                "CLI tests verify defaulting and strict-minimum gate behavior.",
                "Oracle tests verify deterministic/fallback semantics and aggregations.",
                "Parity tests verify strict tolerance and time-grid mismatch handling.",
                "Benchmark tests verify method parsing, schema, and proxy-cost determinism.",
            ],
        },
        {
            "title": "Safe Refactor Boundaries",
            "summary": "Certain contracts are non-negotiable when modifying noise code.",
            "bullets": [
                "Do not alter operator-core algebra modules for this workflow.",
                "Preserve exyz internal conventions and boundary conversions.",
                "Preserve manifest-first PDF contract for generated reports.",
                "Preserve strict legacy parity semantics and observable key meaning.",
            ],
        },
        {
            "title": "High-Risk Changes To Treat Carefully",
            "summary": "A few change categories frequently introduce subtle regressions.",
            "bullets": [
                "Changing default mode flags without updating run-guide contracts.",
                "Mutating JSON key meaning while keeping key names unchanged.",
                "Breaking fallback diagnostics or suppressing failure reason fields.",
                "Changing trajectory-grid assumptions without parity/test updates.",
            ],
        },
        {
            "title": "Practical Extension Workflow",
            "summary": "Use a repeatable workflow whenever adding a new noise feature.",
            "bullets": [
                "Add/adjust parser flags and defaults first.",
                "Thread feature through runner/oracle/report paths with clear payload keys.",
                "Add focused tests for parser + behavior + schema deltas.",
                "Update run-guide and regenerate this guide artifact.",
            ],
        },
        {
            "title": "Debugging Quickstart",
            "summary": "Triage sequence for confusing noise behavior or mode failures.",
            "bullets": [
                "First inspect backend info + execution_fallback fields in JSON.",
                "Then inspect mode-specific unavailable reasons in robustness report.",
                "Then compare noisy-ideal deltas against repeats/shots settings.",
                "Finally run legacy parity anchor for noiseless-path confidence.",
            ],
        },
        {
            "title": "Quick Commands",
            "summary": "Common command anchors for daily usage.",
            "bullets": [
                "Targeted tests: pytest -q test/test_hh_noise_*.py",
                "Validation runner: pipelines/exact_bench/hh_noise_hardware_validation.py",
                "Robustness runner: pipelines/exact_bench/hh_noise_robustness_seq_report.py",
                "This guide build: pipelines/shell/build_hh_noise_model_repo_guide.sh",
            ],
        },
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a long, manifest-first HH noise model repository guide PDF "
            "from code/docs metadata."
        )
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=REPO_ROOT / "docs" / "HH_noise_model_repo_guide.pdf",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "json" / "hh_noise_model_repo_guide_index.json",
    )
    p.add_argument("--skip-pdf", action="store_true")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "docs" / "repo_guide_assets" / "repo_guide_summary.json",
    )
    p.add_argument("--max-rows-per-table-page", type=int, default=26)
    p.set_defaults(include_tests=True)
    p.add_argument("--include-tests", dest="include_tests", action="store_true")
    p.add_argument("--no-include-tests", dest="include_tests", action="store_false")
    p.set_defaults(include_docs=True)
    p.add_argument("--include-docs", dest="include_docs", action="store_true")
    p.add_argument("--no-include-docs", dest="include_docs", action="store_false")
    return p.parse_args(argv)


def _load_summary(summary_json: Path) -> dict[str, Any]:
    if not summary_json.exists():
        return {}
    return json.loads(summary_json.read_text(encoding="utf-8"))


def _module_to_relpath(module: str) -> Path:
    return Path(*str(module).split(".")).with_suffix(".py")


def _module_to_abspath(module: str) -> Path:
    return REPO_ROOT / _module_to_relpath(module)


def _safe_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<unparsed>"


def _extract_add_argument_calls(tree: ast.AST) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
        if not str(first.value).startswith("--"):
            continue

        rec: dict[str, Any] = {
            "flag": str(first.value),
            "line": int(getattr(node, "lineno", -1)),
        }
        for kw in node.keywords:
            if kw.arg in {
                "default",
                "choices",
                "type",
                "help",
                "action",
                "required",
                "dest",
                "metavar",
            }:
                rec[str(kw.arg)] = _safe_literal(kw.value)
        out.append(rec)
    out.sort(key=lambda x: (str(x.get("flag", "")), int(x.get("line", 0))))
    return out


def _extract_constants(tree: ast.AST) -> list[dict[str, Any]]:
    const_re = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
    out: list[dict[str, Any]] = []

    body = tree.body if isinstance(tree, ast.Module) else []
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and const_re.match(target.id):
                    out.append(
                        {
                            "name": str(target.id),
                            "line": int(getattr(node, "lineno", -1)),
                            "value": _safe_literal(node.value),
                        }
                    )
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and const_re.match(node.target.id):
                out.append(
                    {
                        "name": str(node.target.id),
                        "line": int(getattr(node, "lineno", -1)),
                        "value": _safe_literal(node.value) if node.value is not None else None,
                    }
                )

    out.sort(key=lambda x: (int(x["line"]), str(x["name"])))
    return out


def _extract_top_level_defs(tree: ast.AST) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes: list[dict[str, Any]] = []
    functions: list[dict[str, Any]] = []

    body = tree.body if isinstance(tree, ast.Module) else []
    for node in body:
        if isinstance(node, ast.ClassDef):
            classes.append({"name": str(node.name), "line": int(getattr(node, "lineno", -1))})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"name": str(node.name), "line": int(getattr(node, "lineno", -1))})

    classes.sort(key=lambda x: (int(x["line"]), str(x["name"])))
    functions.sort(key=lambda x: (int(x["line"]), str(x["name"])))
    return classes, functions


def _extract_internal_imports(tree: ast.AST) -> list[str]:
    internal: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name)
                if name.startswith("src.") or name.startswith("pipelines") or name.startswith("test."):
                    internal.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = str(node.module or "")
            if mod.startswith("src.") or mod.startswith("pipelines") or mod.startswith("test."):
                internal.add(mod)

    return sorted(internal)


def _fallback_module_record(module: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "module": str(module),
            "path": str(path.relative_to(REPO_ROOT)),
            "classes": [],
            "functions": [],
            "imports": {"internal": [], "external": []},
            "cli_flags": [],
            "constants": [],
            "fallback_ast": False,
            "missing": True,
        }

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes, functions = _extract_top_level_defs(tree)
    constants = _extract_constants(tree)
    cli_flags = _extract_add_argument_calls(tree)
    imports_internal = _extract_internal_imports(tree)

    return {
        "module": str(module),
        "path": str(path.relative_to(REPO_ROOT)),
        "classes": classes,
        "functions": functions,
        "imports": {"internal": imports_internal, "external": []},
        "cli_flags": cli_flags,
        "constants": constants,
        "fallback_ast": True,
        "missing": False,
    }


def _module_record_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    mods = summary.get("modules", []) if isinstance(summary, dict) else []
    if not isinstance(mods, list):
        return out
    for rec in mods:
        if not isinstance(rec, dict):
            continue
        module = rec.get("module")
        if isinstance(module, str) and module:
            out[module] = rec
    return out


def _summary_internal_edges(summary: dict[str, Any]) -> list[dict[str, str]]:
    edges = summary.get("internal_import_edges", []) if isinstance(summary, dict) else []
    out: list[dict[str, str]] = []
    if not isinstance(edges, list):
        return out
    for rec in edges:
        if not isinstance(rec, dict):
            continue
        src = rec.get("src")
        dst = rec.get("dst")
        if isinstance(src, str) and isinstance(dst, str) and src and dst:
            out.append({"src": src, "dst": dst})
    out.sort(key=lambda x: (x["src"], x["dst"]))
    return out


_SCOPE_EXPANSION_MATH = "S = Roots U DirectDeps(Roots) U InboundTests(S) U DocsRefs(S)"


def _resolve_noise_scope(
    *,
    summary: dict[str, Any],
    summary_json_path: Path,
    include_tests: bool,
    include_docs: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    module_map = _module_record_map(summary)
    summary_edges = _summary_internal_edges(summary)
    summary_mtime = (
        float(summary_json_path.stat().st_mtime)
        if summary_json_path.exists()
        else None
    )

    records: dict[str, dict[str, Any]] = {}

    def _ensure_record(module: str) -> dict[str, Any]:
        if module in records:
            return records[module]
        if module in module_map:
            rec = dict(module_map[module])
            rec["fallback_ast"] = False
            rec["missing"] = False
            rec_path = rec.get("path")
            module_path = (
                REPO_ROOT / str(rec_path)
                if isinstance(rec_path, str) and rec_path
                else _module_to_abspath(module)
            )
            # If summary metadata predates the module source, refresh from AST.
            if (
                summary_mtime is not None
                and module_path.exists()
                and float(module_path.stat().st_mtime) > float(summary_mtime)
            ):
                ast_rec = _fallback_module_record(module, module_path)
                if not bool(ast_rec.get("missing", False)):
                    rec = ast_rec
            records[module] = rec
            return rec
        rec = _fallback_module_record(module, _module_to_abspath(module))
        records[module] = rec
        return rec

    for mod in ROOT_MODULES:
        _ensure_record(mod)

    deps: set[str] = set()
    for mod in ROOT_MODULES:
        rec = _ensure_record(mod)
        imports = rec.get("imports", {}) if isinstance(rec, dict) else {}
        internal = imports.get("internal", []) if isinstance(imports, dict) else []
        if isinstance(internal, list):
            for dst in internal:
                if isinstance(dst, str) and dst:
                    deps.add(dst)

    for edge in summary_edges:
        if edge["src"] in ROOT_MODULES:
            deps.add(edge["dst"])

    for dep in sorted(deps):
        _ensure_record(dep)

    inbound_tests: set[str] = set()
    if include_tests:
        candidate_scope = set(records.keys())
        for mod, rec in sorted(module_map.items()):
            path = str(rec.get("path", ""))
            if not path.startswith("test/"):
                continue
            imports = rec.get("imports", {}) if isinstance(rec, dict) else {}
            internal = imports.get("internal", []) if isinstance(imports, dict) else []
            if not isinstance(internal, list):
                continue
            if any(isinstance(dst, str) and dst in candidate_scope for dst in internal):
                inbound_tests.add(mod)
                _ensure_record(mod)

    doc_refs = _collect_doc_references(modules=sorted(records.keys())) if include_docs else []
    docs_in_scope = sorted({str(rec["path"]) for rec in doc_refs})

    resolved_modules = sorted(records.keys(), key=lambda m: str(records[m].get("path", "")))
    dependency_modules = sorted(set(resolved_modules) - set(ROOT_MODULES) - inbound_tests)

    scope = {
        "root_modules": sorted(ROOT_MODULES),
        "resolved_modules": resolved_modules,
        "dependency_modules": dependency_modules,
        "inbound_test_modules": sorted(inbound_tests),
        "doc_reference_files": docs_in_scope,
    }
    return scope, records


def _collect_doc_references(*, modules: list[str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    patterns: list[tuple[str, str]] = []
    for mod in modules:
        path_token = str(_module_to_relpath(mod))
        file_token = path_token.split("/")[-1]
        patterns.append((mod, mod))
        patterns.append((mod, path_token))
        patterns.append((mod, file_token))

    seen: set[tuple[str, int, str]] = set()
    for path in DOC_REFERENCE_PATHS:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        for line_no, line in enumerate(lines, start=1):
            lowered = str(line).lower()
            for mod, token in patterns:
                if token.lower() in lowered:
                    key = (rel, line_no, mod)
                    if key in seen:
                        continue
                    seen.add(key)
                    docs.append(
                        {
                            "path": rel,
                            "line": int(line_no),
                            "module": str(mod),
                            "snippet": str(line).strip(),
                        }
                    )
    docs.sort(key=lambda x: (x["path"], int(x["line"]), x["module"]))
    return docs


def _extract_edges(
    *,
    scope_modules: list[str],
    records: dict[str, dict[str, Any]],
    summary_edges: list[dict[str, str]],
) -> list[dict[str, str]]:
    in_scope = set(scope_modules)
    edges: set[tuple[str, str]] = set()

    for edge in summary_edges:
        src = edge["src"]
        dst = edge["dst"]
        if src in in_scope or dst in in_scope:
            edges.add((src, dst))

    for src in scope_modules:
        rec = records.get(src, {})
        imports = rec.get("imports", {}) if isinstance(rec, dict) else {}
        internal = imports.get("internal", []) if isinstance(imports, dict) else []
        if not isinstance(internal, list):
            continue
        for dst in internal:
            if isinstance(dst, str) and dst:
                edges.add((str(src), str(dst)))

    out = [{"src": s, "dst": d} for (s, d) in sorted(edges, key=lambda x: (x[0], x[1]))]
    return out


_SYMBOL_INDEX_ORDER_MATH = "sort(symbols) by (path, line, kind, name)"


def _extract_symbol_index(*, modules: list[str], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mod in modules:
        rec = records.get(mod, {})
        path = str(rec.get("path", str(_module_to_relpath(mod))))

        for cls in rec.get("classes", []) if isinstance(rec.get("classes", []), list) else []:
            if not isinstance(cls, dict):
                continue
            rows.append(
                {
                    "module": str(mod),
                    "path": path,
                    "kind": "CLASS",
                    "name": str(cls.get("name", "")),
                    "line": int(cls.get("line", -1)),
                }
            )

        for fn in rec.get("functions", []) if isinstance(rec.get("functions", []), list) else []:
            if not isinstance(fn, dict):
                continue
            rows.append(
                {
                    "module": str(mod),
                    "path": path,
                    "kind": "FUNC",
                    "name": str(fn.get("name", "")),
                    "line": int(fn.get("line", -1)),
                }
            )

        for const in rec.get("constants", []) if isinstance(rec.get("constants", []), list) else []:
            if not isinstance(const, dict):
                continue
            rows.append(
                {
                    "module": str(mod),
                    "path": path,
                    "kind": "CONST",
                    "name": str(const.get("name", "")),
                    "line": int(const.get("line", -1)),
                }
            )

    rows.sort(key=lambda x: (x["path"], int(x["line"]), x["kind"], x["name"]))
    return rows


def _is_noise_relevant_flag(flag: str) -> bool:
    needle = str(flag).lower()
    hints = (
        "noise",
        "shot",
        "oracle",
        "backend",
        "runtime",
        "aer",
        "fallback",
        "legacy",
        "parity",
        "compare-observables",
        "output-compare-plot",
        "omp",
        "noisy",
        "method",
    )
    return any(h in needle for h in hints)


def _extract_cli_index(*, modules: list[str], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for mod in modules:
        rec = records.get(mod, {})
        path = str(rec.get("path", str(_module_to_relpath(mod))))
        cli_flags = rec.get("cli_flags", []) if isinstance(rec.get("cli_flags", []), list) else []
        for flag_rec in cli_flags:
            if not isinstance(flag_rec, dict):
                continue
            flag = str(flag_rec.get("flag", ""))
            if not flag.startswith("--"):
                continue
            if not _is_noise_relevant_flag(flag):
                continue
            rows.append(
                {
                    "module": str(mod),
                    "path": path,
                    "flag": flag,
                    "line": int(flag_rec.get("line", -1)),
                    "default": flag_rec.get("default", None),
                    "choices": flag_rec.get("choices", None),
                    "type": str(flag_rec.get("type", "")),
                    "action": str(flag_rec.get("action", "")),
                    "help": str(flag_rec.get("help", "")),
                }
            )

    rows.sort(key=lambda x: (x["path"], x["flag"], int(x["line"])))
    return rows


def _extract_test_anchors(*, scope: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mod in scope.get("inbound_test_modules", []):
        rec = records.get(mod, {})
        path = str(rec.get("path", str(_module_to_relpath(str(mod)))))
        funcs = rec.get("functions", []) if isinstance(rec.get("functions", []), list) else []
        for fn in funcs:
            if not isinstance(fn, dict):
                continue
            rows.append(
                {
                    "module": str(mod),
                    "path": path,
                    "name": str(fn.get("name", "")),
                    "line": int(fn.get("line", -1)),
                }
            )
    rows.sort(key=lambda x: (x["path"], int(x["line"]), x["name"]))
    return rows


def _module_role(module: str) -> str:
    if module == "pipelines.exact_bench.noise_oracle_runtime":
        return "shared expectation oracle"
    if module == "pipelines.exact_bench.hh_noise_hardware_validation":
        return "single-run HH noisy validation"
    if module == "pipelines.exact_bench.hh_noise_robustness_seq_report":
        return "staged HH robustness + long report"
    if module == "pipelines.exact_bench.hh_seq_transition_utils":
        return "Pool-B/transition helper kernel"
    if module.startswith("test."):
        return "test coverage"
    if module.startswith("docs"):
        return "documentation"
    return "direct dependency"


def _manifest_preview_lines(
    *,
    model: str,
    ansatz: str,
    drive_enabled: bool,
    t: float,
    U: float,
    dv: float,
) -> list[str]:
    return [
        "PARAMETER MANIFEST",
        f"Model family/name: {model}",
        f"Ansatz type(s): {ansatz}",
        f"Drive enabled: {bool(drive_enabled)}",
        f"t: {float(t)}",
        f"U: {float(U)}",
        f"dv: {float(dv)}",
    ]


def _render_manifest(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    scope = index_payload.get("scope", {})
    render_parameter_manifest(
        pdf,
        model="Hubbard-Holstein (noise-model repository guide)",
        ansatz="N/A (documentation artifact)",
        drive_enabled=False,
        t=0.0,
        U=0.0,
        dv=0.0,
        extra={
            "scope_root_modules": ", ".join(scope.get("root_modules", [])),
            "resolved_module_count": len(scope.get("resolved_modules", [])),
            "include_tests": bool(index_payload.get("sources", {}).get("include_tests", False)),
            "include_docs": bool(index_payload.get("sources", {}).get("include_docs", False)),
            "summary_json": index_payload.get("sources", {}).get("summary_json"),
        },
        command=str(index_payload.get("run_command", "")),
    )


def _render_human_front_matter(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    """Render exactly 20 human-readable pages ahead of deep technical sections."""
    require_matplotlib()
    plt = get_plt()
    pages = _human_front_matter_pages()
    generated = str(index_payload.get("generated_at_utc", ""))

    for idx, rec in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        ax.axis("off")

        header = "SECTION: HUMAN-READABLE FRONT MATTER"
        if idx > 1:
            header = f"SECTION: HUMAN-READABLE FRONT MATTER (cont. {idx})"
        ax.text(0.03, 0.97, header, ha="left", va="top", fontsize=10, fontweight="bold")
        ax.text(
            0.03,
            0.93,
            f"Human Guide Page {idx}/{len(pages)}: {str(rec.get('title', 'Untitled'))}",
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        y = 0.87
        summary = str(rec.get("summary", "")).strip()
        if summary:
            for wrapped in textwrap.wrap(f"Summary: {summary}", width=104):
                ax.text(0.03, y, wrapped, ha="left", va="top", fontsize=11)
                y -= 0.038
            y -= 0.012

        bullets = rec.get("bullets", [])
        if isinstance(bullets, list):
            for raw in bullets:
                line = str(raw).strip()
                if not line:
                    continue
                wrapped_lines = textwrap.wrap(f"- {line}", width=102, subsequent_indent="  ")
                for wrapped in wrapped_lines:
                    if y < 0.10:
                        break
                    ax.text(0.05, y, wrapped, ha="left", va="top", fontsize=10)
                    y -= 0.034
                if y < 0.10:
                    break

        ax.text(
            0.03,
            0.04,
            f"Generated: {generated}",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        ax.text(
            0.97,
            0.04,
            "Noise model repository guide",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        pdf.savefig(fig)
        plt.close(fig)


def _render_table_pages(
    pdf: Any,
    *,
    title: str,
    section_marker: str,
    col_labels: list[str],
    rows: list[list[str]],
    max_rows_per_page: int,
) -> None:
    require_matplotlib()
    plt = get_plt()
    chunk = max(1, int(max_rows_per_page))

    if not rows:
        render_text_page(
            pdf,
            [section_marker, "", title, "", "No rows available."],
            fontsize=9,
        )
        return

    for start in range(0, len(rows), chunk):
        sub = rows[start : start + chunk]
        page_idx = (start // chunk) + 1
        fig, ax = plt.subplots(figsize=(11.0, 8.5))
        subtitle = title if page_idx == 1 else f"{title} (cont. {page_idx})"
        render_compact_table(
            ax,
            title=f"{section_marker} | {subtitle}",
            col_labels=col_labels,
            rows=sub,
            fontsize=7,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _render_section_scope_sources(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    scope = index_payload.get("scope", {})
    sources = index_payload.get("sources", {})

    lines = [
        "SECTION: SCOPE + SOURCES",
        "",
        "Noise-model full-repo guide scope is rooted at 4 fixed modules.",
        "Scope expansion policy: roots + direct internal deps + inbound tests + doc references.",
        "",
        f"summary_json: {sources.get('summary_json')}",
        f"summary_found: {sources.get('summary_found')}",
        f"fallback_ast_used: {sources.get('fallback_ast_used')}",
        f"include_tests: {sources.get('include_tests')}",
        f"include_docs: {sources.get('include_docs')}",
        "",
        f"root_modules ({len(scope.get('root_modules', []))}):",
    ]
    lines.extend([f"  - {x}" for x in scope.get("root_modules", [])])
    lines.append("")
    lines.append(f"resolved_modules ({len(scope.get('resolved_modules', []))}):")
    lines.extend([f"  - {x}" for x in scope.get("resolved_modules", [])])
    lines.append("")
    lines.append(f"inbound_test_modules ({len(scope.get('inbound_test_modules', []))}):")
    lines.extend([f"  - {x}" for x in scope.get("inbound_test_modules", [])])
    lines.append("")
    lines.append(f"doc_reference_files ({len(scope.get('doc_reference_files', []))}):")
    lines.extend([f"  - {x}" for x in scope.get("doc_reference_files", [])])
    render_text_page(pdf, lines, fontsize=9)


def _render_section_architecture_map(
    pdf: Any,
    *,
    index_payload: dict[str, Any],
    max_rows_per_page: int,
) -> None:
    modules = index_payload.get("modules", [])
    edges = index_payload.get("import_edges", [])

    module_rows = [
        [
            str(m.get("path", "")),
            str(m.get("module", "")),
            str(m.get("role", "")),
            str(m.get("symbol_count", "")),
            str(m.get("cli_flag_count", "")),
        ]
        for m in modules
    ]
    _render_table_pages(
        pdf,
        title="Architecture modules in resolved noise scope",
        section_marker="SECTION: ARCHITECTURE MAP",
        col_labels=["path", "module", "role", "symbols", "cli_flags"],
        rows=module_rows,
        max_rows_per_page=max_rows_per_page,
    )

    edge_rows = [[str(e.get("src", "")), str(e.get("dst", ""))] for e in edges]
    _render_table_pages(
        pdf,
        title="Internal import edges (scope-focused)",
        section_marker="SECTION: ARCHITECTURE MAP",
        col_labels=["src", "dst"],
        rows=edge_rows,
        max_rows_per_page=max_rows_per_page,
    )


def _render_section_oracle_contract(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    flags = [x for x in index_payload.get("cli_flags", []) if str(x.get("module", "")).endswith("hh_noise_hardware_validation")]
    flag_set = sorted({str(x.get("flag", "")) for x in flags})
    lines = [
        "SECTION: ORACLE CONTRACT",
        "",
        "Module: pipelines.exact_bench.noise_oracle_runtime",
        "Core execution modes: ideal, shots, aer_noise, runtime.",
        "Fallback contract: OMP/SHM failures in Aer paths may route to sampler_shots fallback when enabled.",
        "Runtime contract: runtime mode requires IBM Runtime backend resolution and credentials.",
        "",
        "Key behavior surfaces:",
        "  - OracleConfig / OracleEstimate / NoiseBackendInfo",
        "  - ExpectationOracle.evaluate aggregate policy (mean|median over repeats)",
        "  - _resolve_noise_backend, _build_estimator, _run_estimator_job",
        "  - _looks_like_openmp_shm_abort and fallback activation",
        "",
        "Relevant validation CLI flags (from hh_noise_hardware_validation):",
    ]
    lines.extend([f"  - {f}" for f in flag_set])
    render_text_page(pdf, lines, fontsize=9)


def _render_section_validation_contract(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    lines = [
        "SECTION: VALIDATION PIPELINE CONTRACT",
        "",
        "Module: pipelines.exact_bench.hh_noise_hardware_validation",
        "Contract summary:",
        "  1. Build HH/Hubbard Hamiltonian + ansatz/reference state.",
        "  2. Instantiate shared noisy and ideal oracles.",
        "  3. Optional ADAPT stage under noisy oracle.",
        "  4. Optional VQE stage under noisy oracle.",
        "  5. Optional Trotter trajectory with noisy-vs-ideal deltas.",
        "  6. Optional legacy parity audit against locked reference JSON.",
        "  7. Emit JSON + manifest-first PDF (+ optional compare plot).",
        "",
        "Primary JSON areas:",
        "  - settings, noise_config, backend, execution_fallback",
        "  - vqe, adapt, trajectory, summary",
        "  - legacy_parity (when enabled)",
        "",
        "Legacy parity strictness:",
        "  - exact time-grid match required",
        "  - max_abs_delta <= legacy_parity_tol (default 1e-10)",
    ]
    render_text_page(pdf, lines, fontsize=9)


def _render_section_robustness_contract(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    lines = [
        "SECTION: ROBUSTNESS REPORT CONTRACT",
        "",
        "Module: pipelines.exact_bench.hh_noise_robustness_seq_report",
        "Staged flow:",
        "  1. warm-start HVA",
        "  2. ADAPT Pool-B strict union (UCCSD_lifted + HVA + PAOP_full)",
        "  3. final seeded conventional VQE",
        "  4. noiseless matrix (suzuki2/magnus2/cfqm4/cfqm6)",
        "  5. noisy method matrix per profile and mode",
        "",
        "Noisy matrix defaults:",
        "  - noise_modes: ideal,shots,aer_noise",
        "  - noisy_methods: suzuki2,cfqm4",
        "",
        "Benchmark schema (per successful method+mode):",
        "  - benchmark_cost.{term_exp_count_total,pauli_rot_count_total,cx_proxy_total,sq_proxy_total,depth_proxy_total}",
        "  - benchmark_runtime.{wall_total_s,circuit_build_s_total,oracle_eval_s_total,oracle_calls_total}",
        "",
        "Output contract:",
        "  - long manifest-first PDF with section markers and caption strips",
        "  - JSON with equation_registry and plot_contracts for auditability",
    ]
    render_text_page(pdf, lines, fontsize=9)


def _render_section_test_doc_coverage(
    pdf: Any,
    *,
    index_payload: dict[str, Any],
    max_rows_per_page: int,
) -> None:
    tests = index_payload.get("tests", [])
    docs = index_payload.get("docs", [])

    test_lines = [
        "SECTION: TEST + DOC COVERAGE",
        "",
        "Inbound tests document and enforce noise-stack contracts.",
        "Doc references capture run-guide + implementation-guide contract anchors.",
        "",
        f"test_anchor_count: {len(tests)}",
        f"doc_reference_count: {len(docs)}",
    ]
    render_text_page(pdf, test_lines, fontsize=9)

    test_rows = [
        [
            str(t.get("path", "")),
            str(t.get("module", "")),
            str(t.get("name", "")),
            str(t.get("line", "")),
        ]
        for t in tests
    ]
    _render_table_pages(
        pdf,
        title="Inbound test anchors (function-level)",
        section_marker="SECTION: TEST + DOC COVERAGE",
        col_labels=["path", "module", "test_function", "line"],
        rows=test_rows,
        max_rows_per_page=max_rows_per_page,
    )

    doc_rows = [
        [
            str(d.get("path", "")),
            str(d.get("module", "")),
            str(d.get("line", "")),
            str(d.get("snippet", ""))[:88],
        ]
        for d in docs
    ]
    _render_table_pages(
        pdf,
        title="Documentation anchors (noise-scope references)",
        section_marker="SECTION: TEST + DOC COVERAGE",
        col_labels=["path", "module", "line", "snippet"],
        rows=doc_rows,
        max_rows_per_page=max_rows_per_page,
    )


def _render_section_refactor_playbook(pdf: Any) -> None:
    lines = [
        "SECTION: REFACTOR PLAYBOOK",
        "",
        "Safe-change boundaries:",
        "  - Keep operator-core files untouched (pauli_letters_module.py, pauli_words.py, qubitization_module.py).",
        "  - Keep Pauli ordering and exyz/ixyz boundary conversion conventions intact.",
        "  - Keep legacy parity gate semantics (exact time grid + strict tolerance).",
        "  - Keep AER fallback semantics opt-in/opt-out via flags.",
        "",
        "When extending noise behavior:",
        "  1. update CLI parse + help text",
        "  2. update JSON schema docs/run-guide sections",
        "  3. add/adjust unit tests for new contract branches",
        "  4. preserve manifest-first PDF generation",
        "",
        "Regression checklist:",
        "  - test_hh_noise_model_repo_guide.py",
        "  - test_hh_noise_validation_cli.py",
        "  - test_hh_noise_oracle_runtime.py",
        "  - test_hh_noise_robustness_benchmarks.py",
    ]
    render_text_page(pdf, lines, fontsize=9)


def _render_section_symbol_index(
    pdf: Any,
    *,
    index_payload: dict[str, Any],
    max_rows_per_page: int,
) -> None:
    symbols = index_payload.get("symbols", [])
    rows = [
        [
            str(s.get("path", "")),
            str(s.get("module", "")),
            str(s.get("kind", "")),
            str(s.get("name", "")),
            str(s.get("line", "")),
        ]
        for s in symbols
    ]
    _render_table_pages(
        pdf,
        title="Full symbol index for resolved noise scope",
        section_marker="SECTION: APPENDIX SYMBOL INDEX",
        col_labels=["path", "module", "kind", "name", "line"],
        rows=rows,
        max_rows_per_page=max_rows_per_page,
    )


def _render_section_cli_index(
    pdf: Any,
    *,
    index_payload: dict[str, Any],
    max_rows_per_page: int,
) -> None:
    cli = index_payload.get("cli_flags", [])
    rows = [
        [
            str(c.get("path", "")),
            str(c.get("flag", "")),
            str(c.get("default", ""))[:20],
            str(c.get("choices", ""))[:26],
            str(c.get("line", "")),
        ]
        for c in cli
    ]
    _render_table_pages(
        pdf,
        title="Noise-relevant CLI flag index",
        section_marker="SECTION: APPENDIX CLI FLAG INDEX",
        col_labels=["path", "flag", "default", "choices", "line"],
        rows=rows,
        max_rows_per_page=max_rows_per_page,
    )


def _render_section_import_edges(
    pdf: Any,
    *,
    index_payload: dict[str, Any],
    max_rows_per_page: int,
) -> None:
    edges = index_payload.get("import_edges", [])
    rows = [[str(e.get("src", "")), str(e.get("dst", ""))] for e in edges]
    _render_table_pages(
        pdf,
        title="Scope-focused import edge appendix",
        section_marker="SECTION: APPENDIX IMPORT EDGES",
        col_labels=["src", "dst"],
        rows=rows,
        max_rows_per_page=max_rows_per_page,
    )


def _render_section_provenance(pdf: Any, *, index_payload: dict[str, Any]) -> None:
    lines = [
        "SECTION: COMMAND + PROVENANCE",
        "",
        f"generated_at_utc: {index_payload.get('generated_at_utc')}",
        f"run_command: {index_payload.get('run_command')}",
        f"summary_json: {index_payload.get('sources', {}).get('summary_json')}",
        f"summary_found: {index_payload.get('sources', {}).get('summary_found')}",
        f"scope_module_count: {len(index_payload.get('scope', {}).get('resolved_modules', []))}",
        f"symbol_count: {len(index_payload.get('symbols', []))}",
        f"cli_flag_count: {len(index_payload.get('cli_flags', []))}",
        f"edge_count: {len(index_payload.get('import_edges', []))}",
        f"doc_ref_count: {len(index_payload.get('docs', []))}",
        f"test_anchor_count: {len(index_payload.get('tests', []))}",
        "",
        "Required section markers included:",
    ]
    lines.extend([f"  - {m}" for m in REQUIRED_SECTION_MARKERS])
    render_text_page(pdf, lines, fontsize=9)


def _write_pdf(
    *,
    output_pdf: Path,
    index_payload: dict[str, Any],
    max_rows_per_table_page: int,
) -> None:
    require_matplotlib()
    PdfPages = get_PdfPages()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(output_pdf)) as pdf:
        _render_manifest(pdf, index_payload=index_payload)
        _render_human_front_matter(pdf, index_payload=index_payload)
        _render_section_scope_sources(pdf, index_payload=index_payload)
        _render_section_architecture_map(
            pdf,
            index_payload=index_payload,
            max_rows_per_page=max_rows_per_table_page,
        )
        _render_section_oracle_contract(pdf, index_payload=index_payload)
        _render_section_validation_contract(pdf, index_payload=index_payload)
        _render_section_robustness_contract(pdf, index_payload=index_payload)
        _render_section_test_doc_coverage(
            pdf,
            index_payload=index_payload,
            max_rows_per_page=max_rows_per_table_page,
        )
        _render_section_refactor_playbook(pdf)
        _render_section_symbol_index(
            pdf,
            index_payload=index_payload,
            max_rows_per_page=max_rows_per_table_page,
        )
        _render_section_cli_index(
            pdf,
            index_payload=index_payload,
            max_rows_per_page=max_rows_per_table_page,
        )
        _render_section_import_edges(
            pdf,
            index_payload=index_payload,
            max_rows_per_page=max_rows_per_table_page,
        )
        _render_section_provenance(pdf, index_payload=index_payload)


def _write_json(*, output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_index(*, args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary_json)
    summary = _load_summary(summary_path)

    scope, records = _resolve_noise_scope(
        summary=summary,
        summary_json_path=summary_path,
        include_tests=bool(args.include_tests),
        include_docs=bool(args.include_docs),
    )
    summary_edges = _summary_internal_edges(summary)
    scope_modules = list(scope.get("resolved_modules", []))

    modules = []
    for module in scope_modules:
        rec = records[module]
        modules.append(
            {
                "module": module,
                "path": str(rec.get("path", str(_module_to_relpath(module)))),
                "role": _module_role(module),
                "fallback_ast": bool(rec.get("fallback_ast", False)),
                "symbol_count": int(
                    len(rec.get("functions", []))
                    + len(rec.get("classes", []))
                    + len(rec.get("constants", []))
                ),
                "cli_flag_count": int(len(rec.get("cli_flags", []))),
            }
        )
    modules.sort(key=lambda x: (x["path"], x["module"]))

    symbols = _extract_symbol_index(modules=scope_modules, records=records)
    cli_flags = _extract_cli_index(modules=scope_modules, records=records)
    import_edges = _extract_edges(
        scope_modules=scope_modules,
        records=records,
        summary_edges=summary_edges,
    )
    docs = _collect_doc_references(modules=scope_modules) if bool(args.include_docs) else []
    tests = _extract_test_anchors(scope=scope, records=records) if bool(args.include_tests) else []

    fallback_ast_used = any(bool(rec.get("fallback_ast", False)) for rec in records.values())

    payload: dict[str, Any] = {
        "scope": scope,
        "sources": {
            "summary_json": str(summary_path),
            "summary_found": bool(summary_path.exists()),
            "fallback_ast_used": bool(fallback_ast_used),
            "include_tests": bool(args.include_tests),
            "include_docs": bool(args.include_docs),
        },
        "modules": modules,
        "symbols": symbols,
        "cli_flags": cli_flags,
        "import_edges": import_edges,
        "docs": docs,
        "tests": tests,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_command": current_command_string(),
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.max_rows_per_table_page) < 1:
        raise ValueError("--max-rows-per-table-page must be >= 1")

    payload = _build_index(args=args)
    _write_json(output_json=Path(args.output_json), payload=payload)

    if not bool(args.skip_pdf):
        _write_pdf(
            output_pdf=Path(args.output_pdf),
            index_payload=payload,
            max_rows_per_table_page=int(args.max_rows_per_table_page),
        )

    print(f"Wrote JSON: {Path(args.output_json)}")
    if not bool(args.skip_pdf):
        print(f"Wrote PDF:  {Path(args.output_pdf)}")


if __name__ == "__main__":
    main()
