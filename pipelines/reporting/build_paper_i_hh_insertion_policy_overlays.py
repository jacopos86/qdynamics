#!/usr/bin/env python3
"""Build Paper-I HH insertion-policy overlays with matched-error cost markers.

Macro curves are the four page-1 trajectories in the evolving stationary-core
report.  Resource tuples are attached to stationary always-insertion RA and
Append-ADAPT at their selected matched-error prefixes.  No-insertion RA uses
its first crossing in the weak-Holstein panels and iteration 10 in the
strong-Holstein panels.
Singleton curves are the page-2 stationary-source trajectories, with
plateau-insertion RA and Append-ADAPT as the cost pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v7_9392883_20260729"
)
V8_ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v8_9392920_20260729"
)
V7_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v7_chtc"
)
STATIONARY_REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
STATIONARY_REPORT_PDF = STATIONARY_REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress.pdf"
)
STATIONARY_PROVENANCE = STATIONARY_REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_provenance.json"
)
LOCAL_ALWAYS_PROVENANCE = STATIONARY_REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_local_paused_prefix_page1_provenance.json"
)
RECOVERY_ADAPTER = REPO_ROOT / (
    "raw_outputs/paper_i_ra_adapt_stationary_core_recovery_20260730/"
    "recovery_adapter.json"
)
STATIONARY_CURVE_CACHE = OUTPUT_DIR / (
    f"{STEM}_stationary_page1_macro_curve_cache.json"
)
STATIONARY_SINGLETON_CURVE_CACHE = OUTPUT_DIR / (
    f"{STEM}_stationary_page2_singleton_curve_cache.json"
)
APPEND_REGISTRY = REPO_ROOT / (
    "agent_guidance/static-adapt/reporting/canonical-append-registry-v1.json"
)
WORKING_RECORD = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_preplateau_equal_energy_and_round50_error_reporting_20260731.md"
)

PLOT_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
TITLE_BY_REGIME = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
DISPLAY = {
    "macro": {
        "representation": "intact_macro",
        "title": "Stationary-source undecomposed-generator insertion policies",
        "designated_ra": "always",
        "stem": f"{STEM}_macro_insertion_policy_matched_error_candidate",
    },
    "singleton": {
        "representation": "projected_singleton",
        "title": "Stationary-source single-Pauli-word insertion policies",
        "designated_ra": "plateau",
        "stem": f"{STEM}_singleton_insertion_policy_matched_error_candidate",
    },
}
STYLE: Mapping[str, Mapping[str, Any]] = {
    "always": {
        "label": "Always-insertion RA",
        "color": "#8B1A1A",
        "linewidth": 2.05,
        "marker": "*",
    },
    "plateau": {
        "label": "Plateau-insertion RA",
        "color": "#E45756",
        "linewidth": 1.80,
        "marker": "D",
    },
    "no_insertion": {
        "label": "No-insertion RA",
        "color": "#F2A0A0",
        "linewidth": 1.45,
        "marker": "s",
    },
    "append": {
        "label": "Append-ADAPT",
        "color": "#4C78A8",
        "linewidth": 1.55,
        "marker": "o",
    },
}

MACRO_APPEND_PROC = {
    "weak_weak": 0,
    "intermediate_weak": 8,
    "strong_weak_u8": 16,
    "weak_strong": 24,
    "intermediate_strong": 32,
    "strong_strong_u8": 40,
}
MACRO_PLATEAU_PROC = {
    regime: 2 + 8 * index for index, regime in enumerate(PLOT_ORDER)
}
MACRO_NO_INSERTION_PROC = {
    regime: index for index, regime in enumerate(PLOT_ORDER)
}
SINGLETON_PLATEAU_PROC = {
    "weak_weak": 6,
    "intermediate_weak": 14,
    "strong_weak_u8": 22,
    "weak_strong": 30,
    "intermediate_strong": 38,
    "strong_strong_u8": 46,
}
SINGLETON_NO_INSERTION_PROC = {
    regime: 5 + 8 * index for index, regime in enumerate(PLOT_ORDER)
}

# (matched error, RA crossing round, Append crossing round, RA tuple,
# Append tuple).  Tuple order: (N2q, D2q, Dc, W1q, S_alg).
MATCHED: Mapping[str, Mapping[str, tuple[Any, ...]]] = {
    "macro": {
        # Filled dynamically from the page-1 stationary trajectories.
    },
    "singleton": {
        "weak_weak": (
            1.001780e-9,
            35,
            37,
            (146, 112, 487, 285, 90_910),
            (194, 176, 727, 357, 281_774),
        ),
        "intermediate_weak": (
            1.443102e-8,
            34,
            30,
            (120, 90, 525, 256, 93_269),
            (140, 124, 619, 254, 250_765),
        ),
        "strong_weak_u8": (
            1.407003e-6,
            12,
            13,
            (28, 19, 139, 78, 9_110),
            (50, 44, 251, 111, 19_896),
        ),
        "weak_strong": (
            6.387528e-4,
            35,
            48,
            (128, 95, 491, 277, 126_847),
            (262, 226, 867, 488, 1_030_044),
        ),
        "intermediate_strong": (
            1.412750e-4,
            33,
            49,
            (118, 89, 465, 257, 165_467),
            (266, 235, 931, 479, 1_175_668),
        ),
        "strong_strong_u8": (
            4.421326e-8,
            45,
            42,
            (188, 148, 846, 375, 200_972),
            (192, 165, 776, 360, 821_233),
        ),
    },
}

EXPECTED_TERMINAL = {
    "macro": {
        "weak_weak": (3.721554e-4, 4.893695e-4),
        "intermediate_weak": (1.730800e-4, 2.670662e-2),
        "strong_weak_u8": (1.389587e-6, 1.386808e-6),
        "weak_strong": (3.961177e-2, 3.961709e-2),
        "intermediate_strong": (2.483570e-2, 2.483639e-2),
    },
    "singleton": {
        "weak_weak": (1.287859e-14, 9.416689e-10),
        "intermediate_weak": (9.482493e-11, 3.197095e-9),
        "strong_weak_u8": (8.006202e-7, 8.010331e-7),
        "weak_strong": (4.240144e-5, 6.059548e-4),
        "intermediate_strong": (4.655843e-6, 1.085717e-4),
        "strong_strong_u8": (4.073966e-8, 1.161507e-8),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _archive_for_process(process: int) -> Path:
    paths = tuple(ARCHIVE_DIR.glob(f"*proc_{process}.tar.gz"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one v7 archive for process {process}, found {len(paths)}"
        )
    return paths[0]


def _archive_for_process_in(directory: Path, process: int) -> Path:
    paths = tuple(directory.glob(f"*proc_{process}.tar.gz"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one archive for process {process} in {directory}, "
            f"found {len(paths)}"
        )
    return paths[0]


def _archive_summary(process: int) -> tuple[dict[str, Any], dict[str, str]]:
    archive = _archive_for_process(process)
    raw: bytes | None = None
    with tarfile.open(archive, mode="r|gz") as handle:
        for member in handle:
            if member.name != "worker_outputs/summary.json":
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"{archive}: unreadable summary")
            raw = extracted.read()
            break
    if raw is None:
        raise ValueError(f"{archive}: missing worker_outputs/summary.json")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"{archive}: summary is not an object")
    source = {
        "path": str(archive.relative_to(REPO_ROOT)),
        "summary_member": "worker_outputs/summary.json",
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return payload, source


def _archive_summary_in(
    directory: Path,
    process: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    paths = tuple(directory.glob(f"*proc_{process}.tar.gz"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one archive for process {process} in {directory}, "
            f"found {len(paths)}"
        )
    archive = paths[0]
    raw: bytes | None = None
    with tarfile.open(archive, mode="r|gz") as handle:
        for member in handle:
            if member.name != "worker_outputs/summary.json":
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                raise ValueError(f"{archive}: unreadable summary")
            raw = extracted.read()
            break
    if raw is None:
        raise ValueError(f"{archive}: missing worker_outputs/summary.json")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"{archive}: summary is not an object")
    return payload, {
        "path": str(archive.relative_to(REPO_ROOT)),
        "summary_member": "worker_outputs/summary.json",
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stationary_authority() -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    provenance = _load_json(STATIONARY_PROVENANCE)
    if (
        provenance.get("schema")
        != "paper_i_ra_adapt_stationary_core_master_cross_revision_partial_progress_v1"
        or provenance.get("metric") != "same_cutoff_absolute_energy_error"
    ):
        raise ValueError("stationary page-1 provenance identity drifted")
    macro_sources: dict[str, Mapping[str, Any]] = {}
    exact_by_regime: dict[str, Mapping[str, Any]] = {}
    for raw in provenance.get("included_sources", []):
        if not isinstance(raw, Mapping):
            continue
        execution_id = str(raw.get("execution_id", ""))
        if "macro" not in execution_id:
            continue
        macro_sources[execution_id] = raw
        regime = str(raw.get("regime_id", ""))
        exact = raw.get("exact_same_cutoff_energy")
        if regime in PLOT_ORDER and exact is not None:
            exact_by_regime[regime] = {
                "exact_same_cutoff_energy": float(exact),
            }
    if set(exact_by_regime) != set(PLOT_ORDER):
        raise ValueError("stationary page-1 exact-energy manifest is incomplete")
    return macro_sources, exact_by_regime


def _recovery_cells() -> dict[str, Mapping[str, Any]]:
    payload = _load_json(RECOVERY_ADAPTER)
    if payload.get("schema") != "paper_i_ra_adapt_stationary_core_recovery_adapter_v1":
        raise ValueError("stationary recovery adapter identity drifted")
    return {
        str(row["target_execution_id"]): row["cell"]
        for row in payload.get("cells", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("cell"), Mapping)
    }


def _local_always_trace(
    checkpoint: Path,
    *,
    exact_energy: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import ijson
    from ijson.common import ObjectBuilder

    energies: list[float] = []
    depths: list[int] = []
    reference_state: Mapping[str, Any] | None = None
    active_key: str | None = None
    builder: ObjectBuilder | None = None
    depth = 0
    with checkpoint.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if active_key is not None:
                assert builder is not None
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                    if depth == 0:
                        reference_state = builder.value
                        active_key = None
                        builder = None
                continue
            if prefix in {"ansatz_input_state", "adapt_vqe.ansatz_input_state"} and event == "start_map":
                active_key = prefix
                builder = ObjectBuilder()
                builder.event(event, value)
                depth = 1
            elif prefix == "adapt_vqe.history_tail.item.depth" and event in {
                "number",
                "integer",
            }:
                depths.append(int(value))
            elif (
                prefix == "adapt_vqe.history_tail.item.energy_after_opt"
                and event in {"number", "integer", "double"}
            ):
                energies.append(float(value))
    if depths != list(range(1, len(depths) + 1)) or len(energies) != len(depths):
        raise ValueError(f"{checkpoint}: local stationary trace is not contiguous")
    if not isinstance(reference_state, Mapping):
        raise ValueError(f"{checkpoint}: preserved reference state is unavailable")
    return [
        {"k": k, "error": abs(energy - exact_energy)}
        for k, energy in zip(depths, energies, strict=True)
    ], dict(reference_state)


def _selected_local_checkpoint(checkpoint: Path, *, k: int) -> Mapping[str, Any]:
    import ijson
    from ijson.common import ObjectBuilder

    history_index = 0
    active = False
    builder: ObjectBuilder | None = None
    depth = 0
    with checkpoint.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if active:
                assert builder is not None
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                    if depth == 0:
                        result = builder.value
                        if not isinstance(result, Mapping):
                            raise TypeError("selected active-prefix checkpoint is not an object")
                        return result
                continue
            if prefix == "adapt_vqe.history_tail.item" and event == "start_map":
                history_index += 1
            elif (
                history_index == k
                and prefix == "adapt_vqe.history_tail.item.active_prefix_checkpoint"
                and event == "start_map"
            ):
                active = True
                builder = ObjectBuilder()
                builder.event(event, value)
                depth = 1
    raise ValueError(f"{checkpoint}: no active-prefix checkpoint at k={k}")


def _compile_local_always_cost(
    checkpoint_path: Path,
    *,
    reference_state_payload: Mapping[str, Any],
    k: int,
) -> tuple[int, int, int, int, int]:
    from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
        compile_historical_displayed_convention,
        derive_execution_order_repaired_checkpoint,
        reconstruct_reference_state,
        validate_active_prefix_checkpoint,
    )
    from pipelines.exact_bench.paper_i_s_alg_accounting import runtime_prefix_work

    checkpoint = _selected_local_checkpoint(checkpoint_path, k=k)
    try:
        validated = validate_active_prefix_checkpoint(
            checkpoint,
            expected_outer_iteration=k,
        )
    except ValueError:
        repaired, _repair = derive_execution_order_repaired_checkpoint(
            checkpoint,
            expected_outer_iteration=k,
        )
        validated = validate_active_prefix_checkpoint(
            repaired,
            expected_outer_iteration=k,
        )
    reference_state, _reference_receipt = reconstruct_reference_state(
        {"ansatz_input_state": dict(reference_state_payload)},
        num_qubits=validated.num_qubits,
    )
    compiled = compile_historical_displayed_convention(
        validated,
        reference_state=reference_state,
    )
    metrics = compiled["metrics"]
    raw = compiled["raw_compile_payload"]
    receipt = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("local stationary prefix lacks its estimator ledger")
    work = runtime_prefix_work(
        method="RA-ADAPT",
        representation="intact_macro",
        accepted_prefix_length=k,
        estimator_ledger_receipt=receipt,
    )
    return (
        int(metrics["N2q"]),
        int(metrics["D2q"]),
        int(metrics["Dc"]),
        int(raw["qiskit_pretranspile_pauli_1q_work_total"]),
        int(work["S_alg"]),
    )


def _ra_trace(summary: Mapping[str, Any]) -> list[dict[str, float | int]]:
    rows = summary.get("accepted_error_trace")
    if not isinstance(rows, list):
        raise TypeError("RA summary lacks accepted_error_trace")
    trace = [
        {
            "k": int(row["controller_round"]),
            "error": float(row["absolute_energy_error"]),
        }
        for row in rows
    ]
    if [row["k"] for row in trace] != list(range(1, 51)):
        raise ValueError("RA trace is not controller rounds 1..50")
    return trace


def _append_trace(
    summary: Mapping[str, Any], *, exact_energy: float
) -> list[dict[str, float | int]]:
    rows = summary.get("accepted_history")
    if not isinstance(rows, list) or len(rows) != 50:
        raise TypeError("Append summary lacks a 50-round accepted_history")
    trace = [
        {
            "k": int(row["controller_round"]),
            "error": abs(float(row["energy_after"]) - exact_energy),
        }
        for row in rows
    ]
    if [row["k"] for row in trace] != list(range(1, 51)):
        raise ValueError("Append trace is not controller rounds 1..50")
    return trace


def _stationary_macro_curves() -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load exactly the four macro trajectories shown on report page 1."""

    authority_sha = _sha256(STATIONARY_PROVENANCE)
    local_sha = _sha256(LOCAL_ALWAYS_PROVENANCE)
    recovery_sha = _sha256(RECOVERY_ADAPTER)
    if STATIONARY_CURVE_CACHE.is_file():
        cached = _load_json(STATIONARY_CURVE_CACHE)
        cached_inputs = cached.get("compile_inputs")
        inputs_include_no_insertion = isinstance(cached_inputs, Mapping) and all(
            isinstance(cached_inputs.get(regime), Mapping)
            and cached_inputs[regime].get("no_insertion_archive")
            for regime in PLOT_ORDER
        )
        if inputs_include_no_insertion and cached.get("source_sha256") == {
            "stationary_provenance": authority_sha,
            "local_always_provenance": local_sha,
            "recovery_adapter": recovery_sha,
        }:
            return (
                cached["curves"],
                cached["sources"],
                cached["compile_inputs"],
            )

    authority, exact_rows = _stationary_authority()
    recovery = _recovery_cells()
    local_payload = _load_json(LOCAL_ALWAYS_PROVENANCE)
    local_records = {
        str(row["execution_id"]): row
        for row in local_payload.get("records", [])
        if isinstance(row, Mapping)
    }
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    compile_inputs: dict[str, dict[str, Any]] = {}

    for regime_index, regime in enumerate(PLOT_ORDER):
        exact = float(exact_rows[regime]["exact_same_cutoff_energy"])
        append_summary, append_source = _archive_summary(
            MACRO_APPEND_PROC[regime]
        )
        plateau_process = MACRO_PLATEAU_PROC[regime]
        if regime == "strong_strong_u8":
            plateau_summary, plateau_source = _archive_summary_in(
                REPO_ROOT
                / "raw_outputs/paper_i_ra_adapt_stationary_core_v7_partial_report_20260729/"
                "proc42_validation_input",
                plateau_process,
            )
        else:
            plateau_summary, plateau_source = _archive_summary(plateau_process)
        none_summary, none_source = _archive_summary_in(
            V8_ARCHIVE_DIR,
            MACRO_NO_INSERTION_PROC[regime],
        )
        append_trace = _append_trace(append_summary, exact_energy=exact)
        plateau_trace = _ra_trace(plateau_summary)
        none_trace = _ra_trace(none_summary)

        always_execution_id = (
            f"core__{regime}__"
            f"nph{3 if regime_index < 3 else 7}__ra_macro_always"
        )
        reference_state: Mapping[str, Any] | None = None
        local_checkpoint: Path | None = None
        if always_execution_id in local_records:
            local_record = local_records[always_execution_id]
            source = local_record.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"{always_execution_id}: local source is missing")
            checkpoint_binding = source.get("checkpoint")
            if not isinstance(checkpoint_binding, Mapping):
                raise ValueError(f"{always_execution_id}: checkpoint binding is missing")
            local_checkpoint = Path(str(checkpoint_binding.get("path", "")))
            always_trace, reference_state = _local_always_trace(
                local_checkpoint,
                exact_energy=exact,
            )
            always_source: dict[str, Any] = {
                "kind": "authenticated_local_paused_prefix",
                "checkpoint": dict(checkpoint_binding),
                "marker": local_record.get("marker"),
                "status": local_record.get("status"),
            }
        else:
            recovered = recovery.get(always_execution_id)
            if recovered is None:
                raise ValueError(f"{always_execution_id}: page-1 always curve is missing")
            always_trace = [
                {"k": int(row["k"]), "error": float(row["error"])}
                for row in recovered.get("points", [])
                if int(row.get("k", 0)) >= 1
            ]
            always_source = {
                "kind": "explicit_recovery_adapter",
                "path": str(RECOVERY_ADAPTER.relative_to(REPO_ROOT)),
                "sha256": _sha256(RECOVERY_ADAPTER),
                "marker": recovered.get("marker"),
                "status": recovered.get("terminal", {}).get("status"),
            }

        curves[regime] = {
            "always": always_trace,
            "plateau": plateau_trace,
            "no_insertion": none_trace,
            "append": append_trace,
        }
        sources[regime] = {
            "page1_report": str(STATIONARY_REPORT_PDF.relative_to(REPO_ROOT)),
            "page1_provenance": str(STATIONARY_PROVENANCE.relative_to(REPO_ROOT)),
            "always": always_source,
            "plateau": plateau_source,
            "no_insertion": none_source,
            "append": {
                **append_source,
                "qualification": (
                    "science-identical v7 duplicate used to materialize the "
                    "page-1 v6 Append trace after the v6 local archive was retired"
                ),
            },
        }
        compile_inputs[regime] = {
            "append_archive": str(
                _archive_for_process(MACRO_APPEND_PROC[regime])
            ),
            "no_insertion_archive": str(
                _archive_for_process_in(
                    V8_ARCHIVE_DIR,
                    MACRO_NO_INSERTION_PROC[regime],
                )
            ),
            "always_local_checkpoint": (
                str(local_checkpoint) if local_checkpoint is not None else None
            ),
            "always_reference_state": (
                dict(reference_state) if reference_state is not None else None
            ),
            "always_prefix_compile_available": local_checkpoint is not None,
        }

        expected_ids = {
            "append": f"core__{regime}__nph{3 if regime_index < 3 else 7}__append_macro",
            "plateau": f"core__{regime}__nph{3 if regime_index < 3 else 7}__ra_macro_plateau",
            "no_insertion": f"core__{regime}__nph{3 if regime_index < 3 else 7}__ra_macro_append_only",
        }
        for method, execution_id in expected_ids.items():
            row = authority.get(execution_id)
            if row is None:
                if method == "plateau" and execution_id in recovery:
                    expected_terminal = float(recovery[execution_id]["terminal"]["error"])
                else:
                    raise ValueError(f"page-1 provenance lacks {execution_id}")
            else:
                expected_terminal = float(row["terminal"]["error"])
            actual_terminal = float(curves[regime][method][-1]["error"])
            if not math.isclose(
                actual_terminal,
                expected_terminal,
                rel_tol=2.0e-9,
                abs_tol=2.0e-12,
            ):
                raise ValueError(
                    f"{regime}/{method}: page-1 terminal closure drifted "
                    f"({actual_terminal} != {expected_terminal})"
                )
    STATIONARY_CURVE_CACHE.write_text(
        json.dumps(
            {
                "schema": "paper_i_stationary_page1_macro_curve_cache_v1",
                "source_sha256": {
                    "stationary_provenance": authority_sha,
                    "local_always_provenance": local_sha,
                    "recovery_adapter": recovery_sha,
                },
                "curves": curves,
                "sources": sources,
                "compile_inputs": compile_inputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return curves, sources, compile_inputs


def _effective_plateau_k(trace: Sequence[Mapping[str, Any]]) -> int:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    selection = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=int(row["k"]),
                absolute_energy_error=float(row["error"]),
            )
            for row in trace
        )
    )
    return int(selection.controller_round)


def _select_preplateau_match(
    ra_trace: Sequence[Mapping[str, Any]],
    append_trace: Sequence[Mapping[str, Any]],
) -> tuple[float, int, int]:
    cutoff = min(
        _effective_plateau_k(ra_trace),
        _effective_plateau_k(append_trace),
    )
    ra_window = [row for row in ra_trace if int(row["k"]) <= cutoff]
    append_window = [row for row in append_trace if int(row["k"]) <= cutoff]
    target = max(
        min(float(row["error"]) for row in ra_window),
        min(float(row["error"]) for row in append_window),
    )
    inclusive = math.nextafter(target, math.inf)
    ra_k = next(
        int(row["k"]) for row in ra_window if float(row["error"]) <= inclusive
    )
    append_k = next(
        int(row["k"])
        for row in append_window
        if float(row["error"]) <= inclusive
    )
    return target, ra_k, append_k


def _compile_append_cost(archive: Path, *, k: int) -> tuple[int, int, int, int, int]:
    from pipelines.reporting import build_paper_i_ra_vs_adapt_common_accuracy_cost_pdf as common
    from pipelines.static_adapt.ra_adapt import append as append_module
    from pipelines.static_adapt.ra_adapt.pools import (
        build_candidate_inventory_lineage_receipt,
    )

    execution_id = archive.name.split("__cluster_")[0]
    cell = {
        "execution_id": execution_id,
        "method_family": "append",
        "attempt_path": str(archive),
        "result_member": "worker_outputs/result.json",
        "package_dir": str(V7_PACKAGE_DIR),
        "job": _load_json(V7_PACKAGE_DIR / "jobs" / f"{execution_id}.json"),
    }
    common.master._configure_package_dir(V7_PACKAGE_DIR)
    original_validator = append_module._validate_resolved_append_protocol

    def _source_locked_historical_validator(problem: Any, protocol: Any) -> Any:
        """Materialize the signed historical pool without current-file locks."""

        request = protocol.request
        parent, executable = append_module._append_inventories(problem, request)
        if parent.receipt != protocol.parent_inventory:
            raise ValueError("historical Append parent inventory drifted")
        if executable.receipt != protocol.executable_pool:
            raise ValueError("historical Append executable inventory drifted")
        lineage = build_candidate_inventory_lineage_receipt(executable)
        if (
            protocol.lineage_authority.get("candidate_inventory_lineage")
            != lineage.authority_binding()
        ):
            raise ValueError("historical Append inventory lineage drifted")
        return request, parent, executable, lineage

    append_module._validate_resolved_append_protocol = (
        _source_locked_historical_validator
    )
    try:
        prefix = common._append_prefix_from_archive(cell, controller_round=k)
    finally:
        append_module._validate_resolved_append_protocol = original_validator
    compiled = common._compile_cost(prefix, cache={})
    costs = compiled["costs"]
    return tuple(int(costs[key]) for key in ("N2q", "D2q", "Dc", "W1q", "S_alg"))


def _compile_ra_cost(archive: Path, *, k: int) -> tuple[int, int, int, int, int]:
    from pipelines.reporting import build_paper_i_ra_vs_adapt_common_accuracy_cost_pdf as common

    cell = {
        "execution_id": archive.name.split("__cluster_")[0],
        "method_family": "ra",
        "attempt_path": str(archive),
        "result_member": "worker_outputs/result.json",
    }
    prefix = common._ra_prefix_from_archive(cell, controller_round=k)
    compiled = common._compile_cost(prefix, cache={})
    costs = compiled["costs"]
    return tuple(int(costs[key]) for key in ("N2q", "D2q", "Dc", "W1q", "S_alg"))


def _singleton_append_traces() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    registry = _load_json(APPEND_REGISTRY)
    if registry.get("schema") != "paper_i_canonical_append_registry_v1":
        raise ValueError("canonical Append registry schema drifted")
    traces: dict[str, list[dict[str, Any]]] = {}
    for record in registry.get("records", []):
        regime = str(record.get("regime"))
        prefixes = record.get("accepted_prefixes")
        if regime not in PLOT_ORDER or not isinstance(prefixes, list):
            continue
        trace = [
            {
                "k": int(row["controller_round"]),
                "error": float(row["absolute_energy_error"]),
            }
            for row in prefixes
        ]
        if [row["k"] for row in trace] != list(range(1, 51)):
            raise ValueError(f"{regime}: canonical Append trace is not 1..50")
        traces[regime] = trace
    if set(traces) != set(PLOT_ORDER):
        raise ValueError("canonical singleton Append registry is incomplete")
    return traces, {
        "path": str(APPEND_REGISTRY.relative_to(REPO_ROOT)),
        "sha256": _sha256(APPEND_REGISTRY),
    }


def _stationary_singleton_authority() -> dict[str, Mapping[str, Any]]:
    provenance = _load_json(STATIONARY_PROVENANCE)
    if (
        provenance.get("schema")
        != "paper_i_ra_adapt_stationary_core_master_cross_revision_partial_progress_v1"
        or provenance.get("metric") != "same_cutoff_absolute_energy_error"
    ):
        raise ValueError("stationary page-2 provenance identity drifted")
    rows = {
        str(row["execution_id"]): row
        for row in provenance.get("included_sources", [])
        if isinstance(row, Mapping)
        and "singleton" in str(row.get("execution_id", ""))
    }
    expected = {
        f"core__{regime}__nph{3 if index < 3 else 7}__{route}"
        for index, regime in enumerate(PLOT_ORDER)
        for route in (
            "append_singleton",
            "ra_singleton_append_only",
            "ra_singleton_plateau",
        )
    }
    expected.update(
        {
            "core__weak_weak__nph3__ra_singleton_always",
            "core__intermediate_weak__nph3__ra_singleton_always",
        }
    )
    if not expected.issubset(rows):
        missing = sorted(expected.difference(rows))
        raise ValueError(
            "stationary page-2 provenance is incomplete: " + ", ".join(missing)
        )
    return rows


def _stationary_singleton_curves() -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load exactly the available trajectories shown on report page 2."""

    source_sha256 = {
        "stationary_provenance": _sha256(STATIONARY_PROVENANCE),
        "recovery_adapter": _sha256(RECOVERY_ADAPTER),
        "append_registry": _sha256(APPEND_REGISTRY),
    }
    if STATIONARY_SINGLETON_CURVE_CACHE.is_file():
        cached = _load_json(STATIONARY_SINGLETON_CURVE_CACHE)
        if cached.get("source_sha256") == source_sha256:
            return cached["curves"], cached["sources"], {}

    from concurrent.futures import ThreadPoolExecutor

    authority = _stationary_singleton_authority()
    recovery = _recovery_cells()
    append_traces, append_source = _singleton_append_traces()

    archive_jobs: dict[tuple[str, str], int] = {}
    for regime in PLOT_ORDER:
        archive_jobs[(regime, "no_insertion")] = SINGLETON_NO_INSERTION_PROC[
            regime
        ]
        if regime != "weak_strong":
            archive_jobs[(regime, "plateau")] = SINGLETON_PLATEAU_PROC[regime]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            key: executor.submit(_archive_summary, process)
            for key, process in archive_jobs.items()
        }
        archive_rows = {key: future.result() for key, future in futures.items()}

    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for index, regime in enumerate(PLOT_ORDER):
        nph = 3 if index < 3 else 7
        execution_ids = {
            "append": f"core__{regime}__nph{nph}__append_singleton",
            "no_insertion": (
                f"core__{regime}__nph{nph}__ra_singleton_append_only"
            ),
            "plateau": f"core__{regime}__nph{nph}__ra_singleton_plateau",
            "always": f"core__{regime}__nph{nph}__ra_singleton_always",
        }
        none_summary, none_source = archive_rows[(regime, "no_insertion")]
        if regime == "weak_strong":
            recovered_plateau = recovery.get(execution_ids["plateau"])
            if recovered_plateau is None:
                raise ValueError("page-2 weak--strong plateau recovery is missing")
            plateau_trace = [
                {"k": int(row["k"]), "error": float(row["error"])}
                for row in recovered_plateau.get("points", [])
                if int(row.get("k", 0)) >= 1
            ]
            plateau_source: dict[str, Any] = {
                "kind": "explicit_recovery_adapter",
                "path": str(RECOVERY_ADAPTER.relative_to(REPO_ROOT)),
                "sha256": source_sha256["recovery_adapter"],
                "status": recovered_plateau.get("terminal", {}).get("status"),
            }
        else:
            plateau_summary, plateau_source = archive_rows[(regime, "plateau")]
            plateau_trace = _ra_trace(plateau_summary)

        regime_curves: dict[str, list[dict[str, Any]]] = {
            "plateau": plateau_trace,
            "no_insertion": _ra_trace(none_summary),
            "append": append_traces[regime],
        }
        regime_sources: dict[str, Any] = {
            "page2_report": str(STATIONARY_REPORT_PDF.relative_to(REPO_ROOT)),
            "page2_provenance": str(STATIONARY_PROVENANCE.relative_to(REPO_ROOT)),
            "plateau": plateau_source,
            "no_insertion": none_source,
            "append": append_source,
        }
        recovered_always = recovery.get(execution_ids["always"])
        if recovered_always is not None:
            regime_curves["always"] = [
                {"k": int(row["k"]), "error": float(row["error"])}
                for row in recovered_always.get("points", [])
                if int(row.get("k", 0)) >= 1
            ]
            regime_sources["always"] = {
                "kind": "explicit_recovery_adapter",
                "path": str(RECOVERY_ADAPTER.relative_to(REPO_ROOT)),
                "sha256": source_sha256["recovery_adapter"],
                "status": recovered_always.get("terminal", {}).get("status"),
            }
        else:
            regime_sources["always"] = {
                "status": "pending_in_stationary_page2_report"
            }

        for policy, trace in regime_curves.items():
            if [int(row["k"]) for row in trace] != list(range(1, 51)):
                raise ValueError(
                    f"singleton/{regime}/{policy}: page-2 trace is not 1..50"
                )
            authority_row = authority.get(execution_ids[policy])
            if authority_row is None:
                raise ValueError(
                    f"singleton/{regime}/{policy}: page-2 authority is missing"
                )
            expected_terminal = float(authority_row["terminal"]["error"])
            actual_terminal = float(trace[-1]["error"])
            if not math.isclose(
                actual_terminal,
                expected_terminal,
                rel_tol=2.0e-9,
                abs_tol=2.0e-12,
            ):
                raise ValueError(
                    f"singleton/{regime}/{policy}: page-2 terminal closure drifted "
                    f"({actual_terminal} != {expected_terminal})"
                )
            source = regime_sources[policy]
            if "summary_sha256" in source:
                expected_summary_sha = authority_row.get("summary_file_sha256")
                if source["summary_sha256"] != expected_summary_sha:
                    raise ValueError(
                        f"singleton/{regime}/{policy}: summary binding drifted"
                    )

        curves[regime] = regime_curves
        sources[regime] = regime_sources

    STATIONARY_SINGLETON_CURVE_CACHE.write_text(
        json.dumps(
            {
                "schema": "paper_i_stationary_page2_singleton_curve_cache_v1",
                "source_sha256": source_sha256,
                "curves": curves,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return curves, sources, {}


def _context_curves(kind: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Recover non-designated context curves from the previous provenance."""

    path = OUTPUT_DIR / f"{DISPLAY[kind]['stem']}_provenance.json"
    payload = _load_json(path)
    context: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in payload.get("rows", []):
        regime = str(row.get("regime"))
        curves = row.get("curves")
        if regime not in PLOT_ORDER or not isinstance(curves, Mapping):
            continue
        context[regime] = {}
        for key, curve in curves.items():
            if not isinstance(curve, Mapping):
                continue
            points = curve.get("displayed_points")
            if isinstance(points, list):
                context[regime][str(key)] = [
                    {
                        "k": int(point["k"]),
                        "error": float(point["error"]),
                    }
                    for point in points
                ]
    if set(context) != set(PLOT_ORDER):
        raise ValueError(f"{kind}: previous overlay provenance is incomplete")
    return context


def _designated_curves(
    kind: str,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if kind == "macro":
        return _stationary_macro_curves()
    return _stationary_singleton_curves()


def _macro_matches(
    curves: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    compile_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[Any, ...]]:
    matches: dict[str, tuple[Any, ...]] = {}
    for regime in PLOT_ORDER:
        match_error, ra_k, append_k = _select_preplateau_match(
            curves[regime]["always"],
            curves[regime]["append"],
        )
        if regime == "strong_strong_u8":
            # The discrete crossings straddle the common level.  Compare the
            # two requested adjacent-round alternatives and retain the pair
            # with the smaller energy mismatch.
            ra_by_k = {
                int(row["k"]): float(row["error"])
                for row in curves[regime]["always"]
            }
            append_by_k = {
                int(row["k"]): float(row["error"])
                for row in curves[regime]["append"]
            }
            alternatives = (
                (ra_k - 1, append_k),
                (ra_k, append_k + 1),
            )
            ra_k, append_k = min(
                alternatives,
                key=lambda pair: abs(
                    ra_by_k[pair[0]] - append_by_k[pair[1]]
                ),
            )
            match_error = max(ra_by_k[ra_k], append_by_k[append_k])
        source = compile_inputs[regime]
        local_checkpoint = source.get("always_local_checkpoint")
        reference_state = source.get("always_reference_state")
        ra_costs: tuple[int, int, int, int, int] | None
        if local_checkpoint and isinstance(reference_state, Mapping):
            ra_costs = _compile_local_always_cost(
                Path(str(local_checkpoint)),
                reference_state_payload=reference_state,
                k=ra_k,
            )
        else:
            # The complete-Xrev weak--weak archive was retired locally after
            # the authenticated page-1 recovery projection was generated.
            ra_costs = None
        append_costs = _compile_append_cost(
            Path(str(source["append_archive"])),
            k=append_k,
        )
        if regime in ("weak_strong", "intermediate_strong", "strong_strong_u8"):
            no_k = 10
        else:
            inclusive = math.nextafter(match_error, math.inf)
            no_k = next(
                int(row["k"])
                for row in curves[regime]["no_insertion"]
                if float(row["error"]) <= inclusive
            )
        no_costs = _compile_ra_cost(
            Path(str(source["no_insertion_archive"])),
            k=no_k,
        )
        matches[regime] = (
            match_error,
            ra_k,
            append_k,
            ra_costs,
            append_costs,
            no_k,
            no_costs,
        )
    return matches


def _compact_integer(value: int) -> str:
    value = int(value)
    if value < 10_000:
        return str(value)
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"{mantissa}\mathrm{{e}}{int(exponent)}"


def _scientific_integer(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return rf"{mantissa}\mathrm{{e}}{int(exponent)}"


def _tuple_text(marker: str, k: int, costs: Sequence[int]) -> str:
    if len(costs) != 5:
        raise ValueError("cost tuple must contain five coordinates")
    values = ",".join(
        [
            *(_compact_integer(value) for value in costs[:-1]),
            _scientific_integer(costs[-1]),
        ]
    )
    return rf"${marker}\ k={k}:\ ({values})$"


def _assert_terminal(
    *, kind: str, regime: str, ra_trace: Sequence[Mapping[str, Any]], append_trace: Sequence[Mapping[str, Any]]
) -> None:
    expected_ra, expected_append = EXPECTED_TERMINAL[kind][regime]
    actual_ra = float(ra_trace[-1]["error"])
    actual_append = float(append_trace[-1]["error"])
    for label, actual, expected in (
        ("RA", actual_ra, expected_ra),
        ("Append", actual_append, expected_append),
    ):
        if not math.isclose(actual, expected, rel_tol=1.0e-5, abs_tol=2.0e-12):
            raise ValueError(
                f"{kind}/{regime}: {label} terminal error {actual} != {expected}"
            )


def build_overlay(kind: str) -> tuple[Path, Path]:
    if kind not in DISPLAY:
        raise ValueError(f"unknown representation: {kind}")
    designated, designated_sources, compile_inputs = _designated_curves(kind)
    config = DISPLAY[kind]
    designated_ra = str(config["designated_ra"])
    matched_rows: Mapping[str, tuple[Any, ...]] = (
        _macro_matches(designated, compile_inputs)
        if kind == "macro"
        else MATCHED[kind]
    )

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.65, 5.75), dpi=300)
    provenance_rows: list[dict[str, Any]] = []
    display_keys = (
        ("always", "plateau", "no_insertion", "append")
        if kind == "macro"
        else ("plateau", "no_insertion", "append")
    )

    for index, regime in enumerate(PLOT_ORDER):
        ax = axes.flat[index]
        trajectories = dict(designated[regime])

        plotted: dict[str, Any] = {}
        errors: list[float] = []
        for key in display_keys:
            trace = trajectories.get(key)
            if not trace:
                continue
            x = [int(row["k"]) for row in trace]
            y = [float(row["error"]) for row in trace]
            if x != list(range(1, len(x) + 1)):
                raise ValueError(f"{kind}/{regime}/{key}: trace is not contiguous from k=1")
            style = STYLE[key]
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle="-",
                zorder=2 if key != designated_ra else 3,
            )
            errors.extend(value for value in y if value > 0)
            plotted[key] = {
                "displayed_points": [
                    {"k": k, "error": error}
                    for k, error in zip(x, y, strict=True)
                ],
                "terminal_round": x[-1],
                "terminal_error": y[-1],
                "role": (
                    "designated_matched_error_comparator"
                    if key in (designated_ra, "append") and regime in designated
                    else "trajectory_context"
                ),
            }

        matched_record: dict[str, Any] | None = None
        if regime in matched_rows:
            if kind == "macro":
                (
                    match_error,
                    ra_k,
                    append_k,
                    ra_costs,
                    append_costs,
                    no_k,
                    no_costs,
                ) = matched_rows[regime]
            else:
                match_error, ra_k, append_k, ra_costs, append_costs = matched_rows[regime]
                no_k = None
                no_costs = None
            ra_trace = trajectories[designated_ra]
            append_trace = trajectories["append"]
            if kind != "macro":
                _assert_terminal(
                    kind=kind,
                    regime=regime,
                    ra_trace=ra_trace,
                    append_trace=append_trace,
                )
            ra_error = float(ra_trace[ra_k - 1]["error"])
            append_error = float(append_trace[append_k - 1]["error"])
            ax.axhline(
                match_error,
                color="#555555",
                linewidth=0.9,
                linestyle=(0, (4, 2)),
                zorder=1,
            )
            ax.scatter(
                [ra_k],
                [ra_error],
                marker=STYLE[designated_ra]["marker"],
                s=58 if designated_ra == "always" else 30,
                color=STYLE[designated_ra]["color"],
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
            )
            ax.scatter(
                [append_k],
                [append_error],
                marker=STYLE["append"]["marker"],
                s=31,
                color=STYLE["append"]["color"],
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
            )
            no_error: float | None = None
            if no_k is not None and no_costs is not None:
                no_trace = trajectories["no_insertion"]
                no_error = float(no_trace[no_k - 1]["error"])
                ax.scatter(
                    [no_k],
                    [no_error],
                    marker=STYLE["no_insertion"]["marker"],
                    s=30,
                    color=STYLE["no_insertion"]["color"],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=6,
                )
            if ra_costs is not None:
                ra_tuple_text = _tuple_text(
                    r"\star" if designated_ra == "always" else r"\diamond",
                    ra_k,
                    ra_costs,
                )
            else:
                ra_tuple_text = rf"$\star\ k={ra_k}:\ \mathrm{{source\ archive\ retired}}$"
            ax.text(
                0.98,
                0.965,
                ra_tuple_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=STYLE[designated_ra]["color"],
                fontsize=5.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
            )
            ax.text(
                0.98,
                0.895,
                _tuple_text(r"\bullet", append_k, append_costs),
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=STYLE["append"]["color"],
                fontsize=5.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
            )
            if no_k is not None and no_costs is not None:
                ax.text(
                    0.98,
                    0.825,
                    _tuple_text(r"\blacksquare", no_k, no_costs),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color=STYLE["no_insertion"]["color"],
                    fontsize=5.6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.4},
                )
            matched_record = {
                "matched_error": match_error,
                "selector": (
                    (
                        "closer of the two adjacent-round strong--strong "
                        "crossing brackets requested for the discrete match"
                        if kind == "macro" and regime == "strong_strong_u8"
                        else "common attainable error within the shared window "
                        "ending at the earlier effective plateau"
                    )
                ),
                "ra": {
                    "policy": designated_ra,
                    "crossing_round": ra_k,
                    "crossing_error": ra_error,
                    "cost_tuple": (
                        dict(zip(("N2q", "D2q", "Dc", "W1q", "S_alg"), ra_costs))
                        if ra_costs is not None
                        else None
                    ),
                    "cost_status": (
                        "compiled"
                        if ra_costs is not None
                        else "unavailable_source_archive_retired"
                    ),
                },
                "append": {
                    "crossing_round": append_k,
                    "crossing_error": append_error,
                    "cost_tuple": dict(
                        zip(("N2q", "D2q", "Dc", "W1q", "S_alg"), append_costs)
                    ),
                },
                "no_insertion": (
                    {
                        "reported_round": no_k,
                        "reported_error": no_error,
                        "selection_policy": (
                            "fixed k=10 for strong-Holstein macro panels"
                            if regime in (
                                "weak_strong",
                                "intermediate_strong",
                                "strong_strong_u8",
                            )
                            else "first crossing of the panel matched-error level"
                        ),
                        "cost_tuple": dict(
                            zip(
                                ("N2q", "D2q", "Dc", "W1q", "S_alg"),
                                no_costs,
                            )
                        ),
                    }
                    if no_k is not None and no_costs is not None
                    else None
                ),
            }
        ax.set_yscale("log")
        ax.set_xlim(1, 53)
        low = 10 ** math.floor(math.log10(min(errors)))
        display_upper = 1.0
        if low >= display_upper:
            low = 0.1
        ax.set_ylim(low, display_upper)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        low_exponent = int(round(math.log10(low)))
        major_step = max(1, math.ceil(abs(low_exponent) / 6))
        major_exponents = list(range(0, low_exponent - 1, -major_step))
        if major_exponents[-1] != low_exponent:
            major_exponents.append(low_exponent)
        ax.yaxis.set_major_locator(
            FixedLocator([10.0**exponent for exponent in major_exponents])
        )
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
        ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.set_title(TITLE_BY_REGIME[regime], fontsize=9.0, pad=3)
        if index >= 3:
            ax.set_xlabel("ADAPT iteration", fontsize=8.2)
        if index % 3 == 0:
            ax.set_ylabel(r"Energy error, $\Delta E$", fontsize=8.2)

        provenance_rows.append(
            {
                "regime": regime,
                "display_y_limits": [low, display_upper],
                "curves": plotted,
                "matched_error_comparison": matched_record,
                "designated_sources": {
                    key: value
                    for key, value in designated_sources.get(regime, {}).items()
                    if key in display_keys
                    or key
                    in {
                        "page1_report",
                        "page1_provenance",
                        "page2_report",
                        "page2_provenance",
                    }
                },
                "qualification": (
                    (
                        "Weak--strong singleton plateau output failed only the "
                        "G5 plateau-domain exercise guard and is retained as a "
                        "qualified diagnostic observation."
                        if kind == "singleton" and regime == "weak_strong"
                        else (
                            "The authenticated weak--weak stationary trajectory is "
                            "shown, but its retired source archive prevents exact "
                            "selected-prefix Qiskit recompilation."
                            if kind == "macro" and regime == "weak_weak"
                            else None
                        )
                    )
                ),
            }
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=STYLE[key]["color"],
            linewidth=STYLE[key]["linewidth"],
            marker=STYLE[key]["marker"],
            markersize=5.3,
            markeredgecolor="white",
            label=STYLE[key]["label"],
        )
        for key in display_keys
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            label="Matched-error level",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.008),
        ncol=5,
        frameon=False,
        fontsize=7.2,
    )
    fig.suptitle(str(config["title"]), fontsize=11.2, y=0.955)
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.875,
        bottom=0.09,
        wspace=0.18,
        hspace=0.29,
    )

    output_png = OUTPUT_DIR / f"{config['stem']}_plot.png"
    output_provenance = OUTPUT_DIR / f"{config['stem']}_provenance.json"
    fig.savefig(output_png, dpi=300, facecolor="white")
    plt.close(fig)

    provenance = {
        "schema": "paper_i_hh_insertion_policy_matched_error_overlay_v1",
        "representation": config["representation"],
        "metric": "same_cutoff_absolute_energy_error",
        "display_horizon": (
            "available authenticated page-1 prefix (k=21--50)"
            if kind == "macro"
            else (
                "k=50 for each displayed page-2 trajectory"
            )
        ),
        "round50_reporting_policy": "energy error only; no round-50 costs",
        "display_y_policy": (
            "logarithmic; every panel is cropped above at 1e0 without "
            "truncating the stored trajectory"
        ),
        "matched_error_reporting_policy": {
            "resource_tuple_fields": ["N2q", "D2q", "Dc", "W1q", "S_alg"],
            "macro_designated_pair": "stationary always-insertion RA versus Append-ADAPT",
            "macro_additional_comparator": "no-insertion RA",
            "macro_no_insertion_prefix_policy": (
                "first crossing in weak-Holstein panels; fixed k=10 in "
                "strong-Holstein panels"
            ),
            "singleton_designated_pair": "plateau-insertion RA versus Append-ADAPT",
            "line_policy": "one panel-specific horizontal matched-error line",
            "annotation_policy": (
                "macro: three cost tuples per panel; singleton: two cost tuples "
                "per available panel; matched-error value is not printed inside "
                "the panel"
            ),
        },
        "curve_policy": {
            key: {
                "label": value["label"],
                "color": value["color"],
                "linestyle": "solid",
            }
            for key, value in STYLE.items()
            if key in display_keys
        },
        "context_curve_source": (
            (
                "All four macro trajectories are the page-1 stationary-core "
                "trajectories bound by the named evolving-report provenance."
            )
            if kind == "macro"
            else (
                "The three displayed singleton trajectories are the page-2 "
                "stationary-core trajectories bound by the named evolving-report "
                "provenance."
            )
        ),
        "stationary_page1_authority": (
            {
                "pdf": {
                    "path": str(STATIONARY_REPORT_PDF.relative_to(REPO_ROOT)),
                    "sha256": _sha256(STATIONARY_REPORT_PDF),
                },
                "provenance": {
                    "path": str(STATIONARY_PROVENANCE.relative_to(REPO_ROOT)),
                    "sha256": _sha256(STATIONARY_PROVENANCE),
                },
            }
            if kind == "macro"
            else None
        ),
        "stationary_page2_authority": (
            {
                "pdf": {
                    "path": str(STATIONARY_REPORT_PDF.relative_to(REPO_ROOT)),
                    "sha256": _sha256(STATIONARY_REPORT_PDF),
                    "page": 2,
                },
                "provenance": {
                    "path": str(STATIONARY_PROVENANCE.relative_to(REPO_ROOT)),
                    "sha256": _sha256(STATIONARY_PROVENANCE),
                },
            }
            if kind == "singleton"
            else None
        ),
        "working_numerical_record": {
            "path": str(WORKING_RECORD.relative_to(REPO_ROOT)),
            "sha256": _sha256(WORKING_RECORD),
        },
        "rows": provenance_rows,
        "output": {
            "path": str(output_png.relative_to(REPO_ROOT)),
            "sha256": _sha256(output_png),
        },
    }
    output_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_png, output_provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        choices=("all", "macro", "singleton"),
        default="all",
    )
    args = parser.parse_args()
    kinds = (
        ("macro", "singleton")
        if args.representation == "all"
        else (args.representation,)
    )
    for kind in kinds:
        png, provenance = build_overlay(kind)
        print(png)
        print(provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
