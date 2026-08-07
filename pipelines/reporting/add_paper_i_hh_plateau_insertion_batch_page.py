#!/usr/bin/env python3
"""Append one insertion-policy evidence page to the Paper-I review PDF."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from pypdf import PdfReader, PdfWriter

try:
    import ijson
except ImportError as exc:  # pragma: no cover - operational dependency message
    raise SystemExit(
        "ijson is required for bounded-memory extraction of the CHTC archives"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    _compile_comparator_at_k,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
    qiskit_cost_fields,
)

OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{STEM}.pdf"
MAIN_PROVENANCE = OUTPUT_DIR / f"{STEM}_provenance.json"
ARCHIVE_DIR = REPO_ROOT / "tmp/chtc_retrieval/paper_i_insertion_20260726"
TRACKER = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
TERMINAL_COST_CACHE = OUTPUT_DIR / (
    f"{STEM}_insertion_terminal_cost_cache.json"
)
TERMINAL_COST_CACHE_SCHEMA = "paper_i_hh_insertion_terminal_cost_cache_v1"

REGIMES = (
    (0, "intermediate_strong", "Intermediate--strong"),
    (1, "intermediate_weak", "Intermediate--weak"),
    (2, "strong_strong_u8", "Strong--strong"),
    (3, "strong_weak_u8", "Strong--weak"),
    (4, "weak_strong", "Weak--strong"),
    (5, "weak_weak", "Weak--weak"),
)
PLOT_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
TITLE_BY_REGIME = {regime: title for _proc, regime, title in REGIMES}
HISTORY_SCALARS = frozenset(
    {
        "accepted",
        "accepted_admission",
        "delta_abs_current",
        "depth",
        "energy_after_opt",
        "logical_num_parameters_after_opt",
        "logical_parameters_added_this_step",
        "nfev_opt",
        "parameters_added_this_step",
        "selected_logical_op",
        "selected_op",
        "selected_position",
    }
)
RECEIPT_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_terminal_cost_cache() -> dict[str, Any]:
    if not TERMINAL_COST_CACHE.is_file():
        return {
            "schema": TERMINAL_COST_CACHE_SCHEMA,
            "entries": {},
        }
    payload = json.loads(TERMINAL_COST_CACHE.read_text(encoding="utf-8"))
    if payload.get("schema") != TERMINAL_COST_CACHE_SCHEMA:
        raise ValueError("terminal-cost cache schema drift")
    if not isinstance(payload.get("entries"), dict):
        raise ValueError("terminal-cost cache entries are not a mapping")
    return payload


def _write_terminal_cost_cache(payload: Mapping[str, Any]) -> None:
    temporary = TERMINAL_COST_CACHE.with_suffix(
        TERMINAL_COST_CACHE.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(TERMINAL_COST_CACHE)


def _terminal_cost_cache_key(
    *,
    source_evidence_sha256: str,
    representation: str,
    insertion_policy: str,
) -> str:
    return _canonical_sha256(
        {
            "schema": TERMINAL_COST_CACHE_SCHEMA,
            "source_evidence_sha256": source_evidence_sha256,
            "representation": representation,
            "insertion_policy": insertion_policy,
            "qiskit_compile_identity": "table_i_basis_gate_transpile_v1",
            "accepted_prefix_length": 50,
        }
    )


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    payload = b"" if contents is None else contents.get_data()
    return hashlib.sha256(payload).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _member_name(archive: Path, regime: str) -> str:
    suffix = f"/{regime}/json/current.json"
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [name for name in bundle.getnames() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"{archive.name}: expected one {suffix}, found {matches!r}")
    return matches[0]


def _extract_compact_current(
    *,
    archive: Path,
    member_name: str,
) -> dict[str, Any]:
    state_builder: Any | None = None
    state_depth = 0
    settings_builder: Any | None = None
    settings_depth = 0
    nested_builder: Any | None = None
    nested_depth = 0
    nested_target: str | None = None
    current_row: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    adapt_scalars: dict[str, Any] = {}
    state: dict[str, Any] | None = None
    settings: dict[str, Any] | None = None

    with tarfile.open(archive, "r:gz") as bundle:
        stream = bundle.extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(member_name)
        for prefix, event, raw_value in ijson.parse(stream):
            value = _plain(raw_value)

            if state_builder is not None:
                state_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    state_depth += 1
                elif event in {"end_map", "end_array"}:
                    state_depth -= 1
                    if state_depth == 0:
                        state = _plain(state_builder.value)
                        state_builder = None
                continue
            if prefix == "ansatz_input_state" and event == "start_map":
                state_builder = ijson.common.ObjectBuilder()
                state_builder.event(event, raw_value)
                state_depth = 1
                continue

            if settings_builder is not None:
                settings_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    settings_depth += 1
                elif event in {"end_map", "end_array"}:
                    settings_depth -= 1
                    if settings_depth == 0:
                        settings = _plain(settings_builder.value)
                        settings_builder = None
                continue
            if prefix == "settings" and event == "start_map":
                settings_builder = ijson.common.ObjectBuilder()
                settings_builder.event(event, raw_value)
                settings_depth = 1
                continue

            if nested_builder is not None:
                nested_builder.event(event, raw_value)
                if event in {"start_map", "start_array"}:
                    nested_depth += 1
                elif event in {"end_map", "end_array"}:
                    nested_depth -= 1
                    if nested_depth == 0:
                        if current_row is None or nested_target is None:
                            raise RuntimeError("history nested-object state drift")
                        current_row[nested_target] = _plain(nested_builder.value)
                        nested_builder = None
                        nested_target = None
                continue

            if prefix == "adapt_vqe.history.item" and event == "start_map":
                if history:
                    history[-1].pop("_ordered_active_operators", None)
                current_row = {}
                continue
            if prefix == "adapt_vqe.history.item" and event == "end_map":
                if current_row is None:
                    raise RuntimeError("history row closed without opening")
                history.append(current_row)
                current_row = None
                continue
            if current_row is not None:
                base = "adapt_vqe.history.item."
                if prefix.startswith(base):
                    relative = prefix[len(base) :]
                    nested_key = (
                        "_ordered_active_operators"
                        if relative
                        == "active_prefix_checkpoint.ordered_active_operators"
                        else relative
                    )
                    if (
                        relative
                        in {
                            "selected_records",
                            "post_admission_prune",
                            "active_prefix_checkpoint.ordered_active_operators",
                        }
                        and event in {"start_array", "start_map"}
                    ):
                        nested_builder = ijson.common.ObjectBuilder()
                        nested_builder.event(event, raw_value)
                        nested_depth = 1
                        nested_target = nested_key
                        continue
                    if "." not in relative and relative in HISTORY_SCALARS and event in {
                        "boolean",
                        "number",
                        "string",
                        "null",
                    }:
                        current_row[relative] = value
                        continue
                    receipt_base = (
                        "active_prefix_checkpoint.estimator_ledger_receipt."
                    )
                    if relative.startswith(receipt_base) and event in {
                        "number",
                        "string",
                    }:
                        receipt_relative = relative[len(receipt_base) :]
                        receipt = current_row.setdefault(
                            "_compact_estimator_receipt",
                            {
                                "cumulative_raw_occurrences": {
                                    "components": {}
                                }
                            },
                        )
                        if receipt_relative in {"status", "outer_iteration"}:
                            receipt[receipt_relative] = value
                        component_base = "cumulative_raw_occurrences.components."
                        if receipt_relative.startswith(component_base):
                            component = receipt_relative[len(component_base) :]
                            if component in RECEIPT_COMPONENTS:
                                receipt["cumulative_raw_occurrences"]["components"][
                                    component
                                ] = value
                        elif receipt_relative == "cumulative_raw_occurrences.total":
                            receipt["cumulative_raw_occurrences"]["total"] = value
                continue

            if prefix.startswith("adapt_vqe.") and "." not in prefix[len("adapt_vqe.") :]:
                key = prefix[len("adapt_vqe.") :]
                if key in {
                    "abs_delta_e",
                    "adapt_beam_enabled",
                    "ansatz_depth",
                    "exact_gs_energy",
                    "method",
                    "success",
                } and event in {"boolean", "number", "string", "null"}:
                    adapt_scalars[key] = value

    if not history or state is None or settings is None:
        raise ValueError(f"{archive.name}: compact current extraction is incomplete")
    if int(adapt_scalars.get("ansatz_depth", -1)) != len(history):
        raise ValueError(f"{archive.name}: history/depth mismatch")
    return {
        "adapt_vqe": {
            **adapt_scalars,
            "history": history,
            "history_count": len(history),
            "history_tail_count": 0,
        },
        "ansatz_input_state": state,
        "settings": settings,
    }


def _compile_terminal(
    payload: Mapping[str, Any],
    *,
    representation: str,
    insertion_policy: str,
) -> dict[str, Any]:
    from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
        reconstruct_reference_state,
    )
    from pipelines.exact_bench.paper_i_s_alg_accounting import runtime_prefix_work
    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        compile_table_i_pauli_label_groups,
    )
    history = payload["adapt_vqe"]["history"]
    k = len(history)
    receipt = history[-1].get("_compact_estimator_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("terminal history row lacks a compact estimator receipt")
    work = runtime_prefix_work(
        method="SNAKE",
        representation=representation,
        accepted_prefix_length=k,
        estimator_ledger_receipt=receipt,
    )

    ordered = history[-1].get("_ordered_active_operators")
    if not isinstance(ordered, list) or not ordered:
        raise ValueError("terminal history row lacks ordered active operators")
    labels = [str(operator["label"]) for operator in ordered]
    groups = [
        [
            str(term["pauli_exyz"])
            for term in operator["serialized_terms_exyz_in_execution_order"]
        ]
        for operator in ordered
    ]
    if not groups:
        raise ValueError("terminal reconstructed ansatz is empty")
    reference_state, reference_receipt = reconstruct_reference_state(
        payload,
        num_qubits=len(groups[0][0]),
    )
    compiled = compile_table_i_pauli_label_groups(
        pauli_label_groups=groups,
        num_qubits=len(groups[0][0]),
        reference_state=reference_state,
        source_kind=f"paper_i_hh_{insertion_policy}_insertion_terminal_prefix",
    )
    qiskit = qiskit_cost_fields(compiled)
    return {
        "active_depth": len(labels),
        **qiskit,
        "S_alg": int(work["S_alg"]),
        "S_alg_components": dict(work["components"]),
        "S_alg_receipt": work,
        "qiskit_compile_identity": "table_i_basis_gate_transpile_v1",
        "reconstruction": {
            "mode": "source_checkpoint_ordered_active_operators",
            "operator_count": len(labels),
            "labels": labels,
        },
        "reference_state": reference_receipt,
    }


def _collect(
    *,
    cluster_id: int,
    representation: str,
    insertion_policy: str,
    selected_regimes: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    terminal_cost_cache = _load_terminal_cost_cache()
    cache_entries = terminal_cost_cache["entries"]
    for proc, regime, title in REGIMES:
        if regime not in selected_regimes:
            continue
        archive = ARCHIVE_DIR / f"{cluster_id}.{proc}__{regime}_transfer.tar.gz"
        compact = ARCHIVE_DIR / f"{cluster_id}.{proc}__{regime}_compact.json.gz"
        if compact.is_file():
            with gzip.open(compact, "rt", encoding="utf-8") as handle:
                compact_payload = json.load(handle)
            if compact_payload.get("schema") != (
                "paper_i_hh_insertion_compact_transfer_evidence_v1"
            ):
                raise ValueError(f"{compact.name}: compact evidence schema drift")
            identity = compact_payload.get("identity")
            if not isinstance(identity, Mapping) or (
                int(identity.get("cluster_id") or -1) != int(cluster_id)
                or int(identity.get("proc_id", -1)) != int(proc)
                or str(identity.get("regime")) != regime
            ):
                raise ValueError(f"{compact.name}: compact evidence identity drift")
            payload = compact_payload["payload"]
            source = {
                **dict(compact_payload["source"]),
                "compact_evidence": str(compact.relative_to(REPO_ROOT)),
                "compact_evidence_sha256": _sha256(compact),
                "source_mode": "validated_compact_transfer_evidence",
            }
        elif archive.is_file():
            member = _member_name(archive, regime)
            payload = _extract_compact_current(
                archive=archive,
                member_name=member,
            )
            source = {
                "archive": str(archive.relative_to(REPO_ROOT)),
                "archive_sha256": _sha256(archive),
                "member": member,
                "source_mode": "full_transfer_archive",
            }
        else:
            raise FileNotFoundError(
                f"missing full archive and compact evidence for {cluster_id}.{proc}"
            )
        history = payload["adapt_vqe"]["history"]
        if len(history) != 50:
            raise ValueError(f"{regime}: expected 50 completed rounds")
        source_evidence_sha256 = str(
            source.get("compact_evidence_sha256")
            or source.get("archive_sha256")
        )
        cache_key = _terminal_cost_cache_key(
            source_evidence_sha256=source_evidence_sha256,
            representation=representation,
            insertion_policy=insertion_policy,
        )
        cached = cache_entries.get(cache_key)
        if isinstance(cached, Mapping):
            if (
                cached.get("source_evidence_sha256") != source_evidence_sha256
                or cached.get("representation") != representation
                or cached.get("insertion_policy") != insertion_policy
                or cached.get("regime") != regime
                or not isinstance(cached.get("terminal_costs"), Mapping)
            ):
                raise ValueError(f"{regime}: terminal-cost cache identity drift")
            terminal = dict(cached["terminal_costs"])
        else:
            terminal = _compile_terminal(
                payload,
                representation=representation,
                insertion_policy=insertion_policy,
            )
            cache_entries[cache_key] = {
                "source_evidence_sha256": source_evidence_sha256,
                "representation": representation,
                "insertion_policy": insertion_policy,
                "regime": regime,
                "terminal_costs": terminal,
            }
            _write_terminal_cost_cache(terminal_cost_cache)
        trajectory = [
            {
                "k": int(row["depth"]),
                "delta_E": float(row["delta_abs_current"]),
            }
            for row in history
        ]
        rows.append(
            {
                "regime": regime,
                "title": title,
                "trajectory": trajectory,
                "terminal_k": 50,
                "terminal_error": float(trajectory[-1]["delta_E"]),
                "terminal_costs": terminal,
                "source": source,
            }
        )
        print(
            f"{regime}: k=50, deltaE={trajectory[-1]['delta_E']:.8e}, "
            f"S_alg={terminal['S_alg']}",
            flush=True,
        )
    return {
        "schema": "paper_i_hh_insertion_batch_evidence_v2",
        "cluster_id": cluster_id,
        "representation": representation,
        "insertion_policy": insertion_policy,
        "selected_regimes": list(selected_regimes),
        "partial_batch": len(selected_regimes) != len(REGIMES),
        "rows": rows,
    }


def _compact_sci(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}\\mathrm{{e}}{int(exponent)}"


def _plot(
    *,
    evidence: Mapping[str, Any],
    representation: str,
    insertion_policy: str,
    plot_order: tuple[str, ...],
    path: Path,
) -> dict[str, Any]:
    tracker = json.loads(TRACKER.read_text(encoding="utf-8"))
    routes = {str(row["id"]): row for row in tracker["routes"]}
    insertion_key = f"{insertion_policy}_insertion"
    insertion_label = (
        "Always-insertion SNAKE"
        if insertion_policy == "always"
        else "Plateau-insertion SNAKE"
    )
    if representation == "intact_macro":
        snake_route = routes["sr_macro_physical_lanes_nph3_7"]
        append_route = routes["append_adapt_macro_nph3_7"]
        title = (
            "Macro commutation-reduced always-insertion SNAKE"
            if insertion_policy == "always"
            else "Macro plateau-triggered commutation-insertion SNAKE"
        )
    else:
        snake_route = routes["no_overlap_trust_projected_phase3_nph3_7"]
        append_route = routes["append_adapt_projected_singleton_nph3_7"]
        title = (
            "Projected-singleton commutation-reduced always-insertion SNAKE"
            if insertion_policy == "always"
            else "Projected-singleton plateau-triggered commutation-insertion SNAKE"
        )
    evidence_rows = {str(row["regime"]): row for row in evidence["rows"]}

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    partial_batch = len(plot_order) != len(PLOT_ORDER)
    if partial_batch:
        fig, axes = plt.subplots(1, len(plot_order), figsize=(7.65, 3.35), dpi=300)
        axes_flat = np.atleast_1d(axes).flat
        title = f"{title} (three completed regimes)"
    else:
        fig, axes = plt.subplots(2, 3, figsize=(7.65, 5.75), dpi=300)
        axes_flat = axes.flat
    provenance_rows: list[dict[str, Any]] = []
    for index, regime in enumerate(plot_order):
        ax = axes_flat[index]
        insertion = evidence_rows[regime]
        snake = snake_route["results"][regime]
        append = append_route["results"][regime]
        curves = (
            (
                insertion_key,
                insertion["trajectory"],
                "#8b1a1a",
                "-",
                2.0,
            ),
            ("append_only_snake", snake["trajectory"][:50], "#c44e52", "--", 1.35),
            ("append_adapt", append["trajectory"][:50], "#4c72b0", ":", 1.55),
        )
        all_errors: list[float] = []
        for _key, trajectory, color, linestyle, linewidth in curves:
            x = [int(row.get("k", row.get("round"))) for row in trajectory]
            y = [float(row.get("delta_E", row.get("error"))) for row in trajectory]
            all_errors.extend(y)
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        ax.scatter(
            [50],
            [float(insertion["terminal_error"])],
            color="#8b1a1a",
            marker="*",
            s=64,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        append_terminal = append["trajectory"][49]
        ax.scatter(
            [50],
            [float(append_terminal["error"])],
            color="#4c72b0",
            marker="o",
            s=30,
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
        )
        insertion_cost = insertion["terminal_costs"]
        append_prefix, append_compile_source = _compile_comparator_at_k(
            source=append["source"],
            trajectory=append["trajectory"],
            k=50,
            representation=representation,
        )
        append_cost = {
            **append_route["costs"][regime],
            **qiskit_cost_fields(append_prefix["qiskit"]),
        }
        append_s_alg = int(append["s_alg"])
        ax.text(
            0.98,
            0.96,
            paper_i_cost_tuple_latex(
                insertion_cost,
                marker=r"\star",
                format_s_alg=_compact_sci,
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#8b1a1a",
            fontsize=6.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.5},
        )
        ax.text(
            0.98,
            0.875,
            paper_i_cost_tuple_latex(
                {**append_cost, "S_alg": append_s_alg},
                marker=r"\bullet",
                format_s_alg=_compact_sci,
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="#4c72b0",
            fontsize=6.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.5},
        )
        ax.set_yscale("log")
        ax.set_xlim(1, 53)
        low = 10 ** math.floor(math.log10(min(value for value in all_errors if value > 0)))
        high = 10 ** math.ceil(math.log10(max(all_errors)))
        if math.isclose(low, high):
            low /= 10
            high *= 10
        ax.set_ylim(low, high)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
        ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
        ax.tick_params(axis="both", labelsize=7.5)
        ax.set_title(TITLE_BY_REGIME[regime], fontsize=9.0, pad=3)
        if partial_batch or index >= 3:
            ax.set_xlabel("ADAPT iteration", fontsize=8.2)
        if index % 3 == 0:
            ax.set_ylabel(r"Energy error, $\Delta E$", fontsize=8.2)
        provenance_rows.append(
            {
                "regime": regime,
                "point_counts": {
                    insertion_key: 50,
                    "append_only_snake": 50,
                    "append_adapt": 50,
                },
                "marker_k": 50,
                f"{insertion_key}_terminal_error": insertion["terminal_error"],
                "append_adapt_terminal_error": float(append_terminal["error"]),
                f"{insertion_key}_costs": insertion_cost,
                "append_adapt_costs": {
                    **append_cost,
                    "S_alg": append_s_alg,
                },
                "append_adapt_compile_source": append_compile_source,
            }
        )
    handles = (
        Line2D([0], [0], color="#8b1a1a", linewidth=2.0, label=insertion_label),
        Line2D([0], [0], color="#c44e52", linestyle="--", linewidth=1.35, label="Append-only SNAKE"),
        Line2D([0], [0], color="#4c72b0", linestyle=":", linewidth=1.55, label="Append-ADAPT"),
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(title, fontsize=11.2, y=0.955)
    if partial_batch:
        fig.subplots_adjust(
            left=0.085,
            right=0.99,
            top=0.78,
            bottom=0.17,
            wspace=0.18,
        )
    else:
        fig.subplots_adjust(
            left=0.085,
            right=0.99,
            top=0.88,
            bottom=0.09,
            wspace=0.18,
            hspace=0.29,
        )
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "tracker": {
            "path": str(TRACKER.relative_to(REPO_ROOT)),
            "sha256": _sha256(TRACKER),
        },
        "rows": provenance_rows,
    }


def _compile_tex(tex_path: Path) -> Path:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(tex_path.parent),
        str(tex_path),
    ]
    for _ in range(2):
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        choices=("macro", "singleton"),
        required=True,
    )
    parser.add_argument(
        "--insertion-policy",
        choices=("plateau", "always"),
        default="plateau",
    )
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument(
        "--regime",
        action="append",
        choices=PLOT_ORDER,
        help="Regime to include; repeat for a labeled partial-batch page.",
    )
    parser.add_argument(
        "--page-number",
        type=int,
        help="Explicit additive destination page number.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help=(
            "Compile and cache terminal costs for the selected regimes without "
            "changing the aggregate PDF."
        ),
    )
    args = parser.parse_args()

    representation = (
        "intact_macro" if args.representation == "macro" else "projected_singleton"
    )
    if args.page_number is not None:
        page_number = int(args.page_number)
    elif args.insertion_policy == "plateau":
        page_number = 11 if args.representation == "macro" else 12
    else:
        page_number = 13 if args.representation == "macro" else 14
    if page_number < 1:
        raise ValueError("--page-number must be positive")
    selected_regimes = tuple(args.regime or PLOT_ORDER)
    if len(selected_regimes) != len(set(selected_regimes)):
        raise ValueError("--regime values must be unique")
    plot_order = tuple(regime for regime in PLOT_ORDER if regime in selected_regimes)
    if len(plot_order) != len(selected_regimes):
        raise ValueError("selected-regime ordering drift")
    partial_suffix = (
        f"_partial{len(plot_order)}" if len(plot_order) != len(PLOT_ORDER) else ""
    )
    expected_base_pages = page_number - 1
    page_stem = (
        f"{STEM}_{args.representation}_{args.insertion_policy}"
        f"_insertion{partial_suffix}_batch_page{page_number}"
    )
    page_png = OUTPUT_DIR / f"{page_stem}_plot.png"
    page_tex = OUTPUT_DIR / f"{page_stem}.tex"
    page_pdf = OUTPUT_DIR / f"{page_stem}.pdf"
    evidence_path = OUTPUT_DIR / f"{page_stem}_evidence.json"
    provenance_path = OUTPUT_DIR / f"{page_stem}_provenance.json"
    backup_pdf = OUTPUT_DIR / (
        f"{STEM}_pre_{args.representation}_{args.insertion_policy}"
        f"_batch_page{page_number}.pdf"
    )

    if args.compile_only:
        evidence = _collect(
            cluster_id=int(args.cluster_id),
            representation=representation,
            insertion_policy=str(args.insertion_policy),
            selected_regimes=selected_regimes,
        )
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(evidence_path)
        return 0

    reader = PdfReader(str(FINAL_PDF))
    if len(reader.pages) != expected_base_pages:
        raise ValueError(
            f"refusing non-additive update: expected {expected_base_pages} existing pages, "
            f"found {len(reader.pages)}"
        )
    if not backup_pdf.exists():
        shutil.copy2(FINAL_PDF, backup_pdf)

    evidence = _collect(
        cluster_id=int(args.cluster_id),
        representation=representation,
        insertion_policy=str(args.insertion_policy),
        selected_regimes=selected_regimes,
    )
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_provenance = _plot(
        evidence=evidence,
        representation=representation,
        insertion_policy=str(args.insertion_policy),
        plot_order=plot_order,
        path=page_png,
    )
    scope_sentence = (
        "This interim page shows the three completed regimes only. "
        if len(plot_order) != len(PLOT_ORDER)
        else ""
    )
    page_tex.write_text(
        "\n".join(
            (
                r"\documentclass[10pt]{article}",
                r"\usepackage[letterpaper,margin=0.32in]{geometry}",
                r"\usepackage{graphicx}",
                r"\usepackage{amsmath}",
                r"\pagestyle{empty}",
                r"\begin{document}",
                r"\centering",
                rf"\includegraphics[width=0.985\textwidth]{{{page_png}}}",
                r"\vspace{-0.5em}",
                (
                    r"\parbox{0.96\textwidth}{\small "
                    rf"{scope_sentence}"
                    r"All curves show same-cutoff energy error through iteration 50. "
                    rf"Endpoint tuples report ${PAPER_I_QISKIT_COST_TUPLE_LATEX}$ for "
                    rf"{args.insertion_policy}-insertion SNAKE "
                    r"(red star) and Append-ADAPT (blue circle); "
                    r"$W_{1q}$ is genuine Qiskit-emitted Pauli-rotation one-qubit "
                    r"work before transpilation; append-only SNAKE is retained as "
                    r"trajectory context.}"
                ),
                r"\end{document}",
                "",
            )
        ),
        encoding="utf-8",
    )
    _compile_tex(page_tex)
    if len(PdfReader(str(page_pdf)).pages) != 1:
        raise ValueError("generated insertion-policy page is not one page")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_page(PdfReader(str(page_pdf)).pages[0])
    with FINAL_PDF.open("wb") as handle:
        writer.write(handle)
    final_reader = PdfReader(str(FINAL_PDF))
    if len(final_reader.pages) != page_number:
        raise ValueError("final PDF page count did not increase by one")
    backup_reader = PdfReader(str(backup_pdf))
    preserved_page_hashes = [
        _page_content_sha256(page) for page in backup_reader.pages
    ]
    final_prefix_hashes = [
        _page_content_sha256(page)
        for page in final_reader.pages[:expected_base_pages]
    ]
    if final_prefix_hashes != preserved_page_hashes:
        raise ValueError("existing PDF page content changed during additive update")

    provenance = {
        "schema": "paper_i_hh_insertion_batch_page_v2",
        "representation": representation,
        "insertion_policy": str(args.insertion_policy),
        "cluster_id": int(args.cluster_id),
        "selected_regimes": list(plot_order),
        "partial_batch": len(plot_order) != len(PLOT_ORDER),
        "page_number": page_number,
        "additive_update": True,
        "base_page_count": expected_base_pages,
        "final_page_count": page_number,
        "validation": {
            "preserved_page_content_sha256": preserved_page_hashes,
            "preserved_page_count": expected_base_pages,
        },
        "sources": {
            "evidence": {
                "path": str(evidence_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(evidence_path),
            },
            "backup_pdf": {
                "path": str(backup_pdf.relative_to(REPO_ROOT)),
                "sha256": _sha256(backup_pdf),
            },
        },
        "plot": plot_provenance,
        "outputs": {
            "final_pdf": {
                "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(FINAL_PDF),
            },
            "page_pdf": {
                "path": str(page_pdf.relative_to(REPO_ROOT)),
                "sha256": _sha256(page_pdf),
            },
            "page_png": {
                "path": str(page_png.relative_to(REPO_ROOT)),
                "sha256": _sha256(page_png),
            },
            "page_tex": {
                "path": str(page_tex.relative_to(REPO_ROOT)),
                "sha256": _sha256(page_tex),
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    main = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
    provenance_key = (
        f"{args.representation}_{args.insertion_policy}_insertion"
        f"{partial_suffix}_batch_page"
    )
    main[provenance_key] = provenance
    main.setdefault("generated", {})["pdf"] = {
        "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
        "sha256": _sha256(FINAL_PDF),
        "pages": page_number,
    }
    main.setdefault("validation", {})["page_count"] = page_number
    MAIN_PROVENANCE.write_text(
        json.dumps(main, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(FINAL_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
