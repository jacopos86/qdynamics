#!/usr/bin/env python3
"""Render the three reader-facing RA appendix L=3 convergence figures.

The figures are observations of the immutable L=3 v6 campaign.  This script
rehashes each complete cell before reading it, recompiles every signed
post-prune accepted prefix under the evidence contract to obtain ``N1q``,
closes the stored ``N2q/D2q/Dc`` triplet against that recompilation, derives
``W1q`` from the sealed termwise Pauli blocks, and reads prefix ``S_alg`` from
the signed estimator-ledger checkpoint.

Each figure carries minimal RA plus one extension.  Markers are placed at the
first round reaching ``1e-6`` when a curve gets there and at the common
terminal horizon otherwise, so an extension that does not converge is still
reported at a directly comparable round rather than dropped.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys

_SCRIPT_PATH = Path(__file__).resolve()
if str(_SCRIPT_PATH.parents[2]) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_PATH.parents[2]))

from pipelines.hardcoded.adapt_circuit_execution import (
    build_structural_ansatz_circuit,
)
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend,
    compiled_gate_stats,
    resolve_backend_targets,
    safe_circuit_depth,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
CAMPAIGN_ID = "ra_adapt_appendix_demos_l3_20260820_v8"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "chtc/ra_adapt_appendix_demos_20260820/results_l3_v8/figures"
)
ACCURACY_TARGET = 1.0e-6
TUPLE_FIELDS = ("N2q", "D2q", "Dc", "N1q", "W1q", "S_alg")
TUPLE_LATEX = r"(N_{2q},D_{2q},D_c,N_{1q},W_{1q},S_{\rm alg})"
PLOT_FLOOR = 1.0e-16
_COMPILED_RESOURCE_CACHE: dict[str, Mapping[str, int]] = {}
_COMPILE_BACKEND: Any | None = None

BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"


@dataclass(frozen=True)
class Curve:
    cell_id: str
    label: str
    color: str
    marker: str
    rounds: tuple[int, ...]
    errors: tuple[float, ...]
    resources: tuple[Mapping[str, int], ...]
    marker_round: int
    marker_reason: str
    accepted_deletion_rounds: tuple[int, ...]
    source_files: Mapping[str, str]

    @property
    def marker_index(self) -> int:
        return self.rounds.index(self.marker_round)

    @property
    def marker_resources(self) -> Mapping[str, int]:
        return self.resources[self.marker_index]

    @property
    def terminal_round(self) -> int:
        return self.rounds[-1]

    @property
    def terminal_resources(self) -> Mapping[str, int]:
        return self.resources[-1]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Mapping[str, Any], *, field: str = "sha256") -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON: {path}") from exc


def _require_signed(value: Any, *, owner: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a mapping.")
    claimed = value.get("sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{owner} lacks a canonical SHA-256.")
    if claimed != _canonical_sha(value):
        raise ValueError(f"{owner} canonical SHA-256 is invalid.")
    return value


def _artifact_inventory(cell_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cell_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("Published cell artifacts cannot contain symlinks.")
        if not path.is_file() or path.name == "artifact_receipt.json":
            continue
        rows.append(
            {
                "path": path.relative_to(cell_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha(path),
            }
        )
    return rows


def validate_published_cell(cell_dir: Path) -> Mapping[str, Any]:
    if not cell_dir.is_dir() or cell_dir.is_symlink():
        raise ValueError(f"Published cell is unavailable: {cell_dir}")
    receipt = _require_signed(
        _read_json(cell_dir / "artifact_receipt.json"),
        owner=f"{cell_dir.name} artifact receipt",
    )
    if (
        receipt.get("schema") != "ra_adapt_appendix_cell_artifact_receipt_v1"
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("cell_id") != cell_dir.name
    ):
        raise ValueError(f"{cell_dir.name} artifact identity drifted.")
    inventory = _artifact_inventory(cell_dir)
    if inventory != receipt.get("files"):
        raise ValueError(f"{cell_dir.name} artifact inventory drifted.")
    if hashlib.sha256(_canonical_bytes(inventory)).hexdigest() != receipt.get(
        "files_sha256"
    ):
        raise ValueError(f"{cell_dir.name} aggregate artifact SHA drifted.")
    return receipt


def _compile_backend() -> Any:
    global _COMPILE_BACKEND
    if _COMPILE_BACKEND is not None:
        return _COMPILE_BACKEND
    targets, audit = resolve_backend_targets(
        requested_names=("FakeMarrakesh",),
        preferred_fake_backends=("FakeMarrakesh",),
        allow_preferred_fallback=False,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )
    if (
        len(targets) != 1
        or targets[0].requested_name != "FakeMarrakesh"
        or targets[0].resolved_name != "FakeMarrakesh"
        or targets[0].resolution_kind != "fake_exact"
        or targets[0].using_fake_backend is not True
        or len(audit) != 1
        or audit[0].get("success") is not True
        or audit[0].get("resolution_kind") != "fake_exact"
    ):
        raise ValueError("Appendix N1q projection requires exact FakeMarrakesh.")
    _COMPILE_BACKEND = targets[0].backend_obj
    return _COMPILE_BACKEND


def _reference_state(payload: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    raw = payload.get("ansatz_input_state")
    if not isinstance(raw, Mapping):
        raise ValueError("Appendix N1q projection lacks its reference state.")
    num_qubits = int(raw.get("nq_total", 0))
    amplitudes = raw.get("amplitudes_qn_to_q0")
    if num_qubits < 1 or not isinstance(amplitudes, Mapping) or not amplitudes:
        raise ValueError("Appendix N1q reference state is malformed.")
    state = np.zeros(1 << num_qubits, dtype=complex)
    for raw_bits, raw_value in amplitudes.items():
        bits = str(raw_bits)
        if len(bits) != num_qubits or set(bits).difference({"0", "1"}):
            raise ValueError("Appendix N1q reference bitstring is invalid.")
        if not isinstance(raw_value, Mapping):
            raise ValueError("Appendix N1q reference amplitude is invalid.")
        value = complex(
            float(raw_value.get("re", 0.0)),
            float(raw_value.get("im", 0.0)),
        )
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("Appendix N1q reference amplitude is nonfinite.")
        state[int(bits, 2)] = value
    norm = float(np.linalg.norm(state))
    if not math.isfinite(norm) or not math.isclose(
        norm, 1.0, rel_tol=0.0, abs_tol=1.0e-10
    ):
        raise ValueError("Appendix N1q reference state is not normalized.")
    return state, num_qubits


def _active_operators(
    prefix: Mapping[str, Any], *, num_qubits: int
) -> tuple[AnsatzTerm, ...]:
    rows = prefix.get("ordered_active_operators")
    if not isinstance(rows, list) or len(rows) != int(
        prefix.get("active_ansatz_depth", -1)
    ):
        raise ValueError("Appendix N1q active-prefix depth is invalid.")
    operators: list[AnsatzTerm] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError("Appendix N1q active operator must be a mapping.")
        terms = raw_row.get("serialized_terms_exyz_in_execution_order")
        if (
            int(raw_row.get("active_position", -1)) != index
            or raw_row.get("execution_mode") != "termwise_product"
            or not isinstance(terms, list)
            or not terms
        ):
            raise ValueError("Appendix N1q active operator is malformed.")
        polynomial = PauliPolynomial("JW")
        for raw_term in terms:
            if not isinstance(raw_term, Mapping):
                raise ValueError("Appendix N1q Pauli term must be a mapping.")
            pauli = str(raw_term.get("pauli_exyz", ""))
            coefficient = complex(
                float(raw_term.get("coeff_re", 0.0)),
                float(raw_term.get("coeff_im", 0.0)),
            )
            if (
                int(raw_term.get("nq", -1)) != num_qubits
                or len(pauli) != num_qubits
                or set(pauli).difference({"e", "x", "y", "z"})
                or set(pauli) == {"e"}
                or not math.isfinite(coefficient.real)
                or not math.isfinite(coefficient.imag)
                or abs(coefficient) <= 1.0e-12
            ):
                raise ValueError("Appendix N1q Pauli term is invalid.")
            polynomial.add_term(
                PauliTerm(num_qubits, ps=pauli, pc=coefficient)
            )
        operators.append(
            AnsatzTerm(
                label=str(raw_row.get("label", "")),
                polynomial=polynomial,
                execution_mode="termwise_product",
            )
        )
    return tuple(operators)


def _termwise_one_qubit_work(operators: Sequence[Any]) -> int:
    """Return canonical pretranspile Pauli-synthesis one-qubit work."""

    total = 0
    for operator in operators:
        if not isinstance(operator, Mapping):
            raise ValueError("Active operator row must be a mapping.")
        if operator.get("execution_mode") != "termwise_product":
            raise ValueError("Appendix W1q projection requires termwise products.")
        terms = operator.get("serialized_terms_exyz_in_execution_order")
        if not isinstance(terms, list) or not terms:
            raise ValueError("Active operator lost its serialized Pauli terms.")
        for term in terms:
            if not isinstance(term, Mapping):
                raise ValueError("Serialized Pauli term must be a mapping.")
            pauli = str(term.get("pauli_exyz", ""))
            if not pauli or set(pauli) - set("exyz") or set(pauli) == {"e"}:
                raise ValueError("Serialized Pauli term is not a physical e/x/y/z word.")
            coeff_re = float(term.get("coeff_re"))
            coeff_im = float(term.get("coeff_im"))
            if not math.isfinite(coeff_re) or not math.isfinite(coeff_im):
                raise ValueError("Serialized Pauli coefficient must be finite.")
            total += 2 * pauli.count("x") + 4 * pauli.count("y") + 1
    return total


def _post_transpile_resources(
    *,
    payload: Mapping[str, Any],
    prefix: Mapping[str, Any],
    evidence_compile: Mapping[str, Any],
) -> Mapping[str, int]:
    compile_input_sha256 = str(
        evidence_compile.get("compile_input_sha256", "")
    )
    if len(compile_input_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in compile_input_sha256
    ):
        raise ValueError("Appendix accepted-prefix compile identity is invalid.")
    cached = _COMPILED_RESOURCE_CACHE.get(compile_input_sha256)
    if cached is not None:
        return cached
    reference_state, num_qubits = _reference_state(payload)
    operators = _active_operators(prefix, num_qubits=num_qubits)
    _, circuit = build_structural_ansatz_circuit(
        operators,
        nq=num_qubits,
        ref_state=reference_state,
        structure_theta_value=1.0,
    )
    compiled = compile_circuit_for_backend(
        circuit,
        _compile_backend(),
        seed_transpiler=7,
        optimization_level=1,
        initial_layout=None,
    )["compiled"]
    stats = compiled_gate_stats(compiled)
    resources = {
        "N2q": int(stats["compiled_count_2q"]),
        "D2q": int(stats["compiled_depth_2q"]),
        "Dc": int(safe_circuit_depth(compiled)),
        "N1q": int(stats["compiled_count_1q"]),
    }
    stored = {
        field: int(evidence_compile[field]) for field in ("N2q", "D2q", "Dc")
    }
    if any(resources[field] != value for field, value in stored.items()):
        raise ValueError("Appendix N1q recompilation lost stored compile closure.")
    _COMPILED_RESOURCE_CACHE[compile_input_sha256] = resources
    return resources


def load_curve(
    cell_id: str,
    *,
    label: str,
    color: str,
    marker: str,
    marker_mode: str,
    runtime_root: Path = RUNTIME_ROOT,
) -> Curve:
    cell_dir = runtime_root / "cells" / cell_id
    receipt = validate_published_cell(cell_dir)
    summary_path = cell_dir / "cell_summary.json"
    evidence_path = cell_dir / "accepted_run_evidence.json"
    checkpoint_path = cell_dir / "observations/current.json"
    summary = _require_signed(_read_json(summary_path), owner=f"{cell_id} summary")
    evidence = _require_signed(
        _read_json(evidence_path), owner=f"{cell_id} accepted evidence"
    )
    checkpoint = _read_json(checkpoint_path)

    if summary.get("cell_id") != cell_id:
        raise ValueError(f"{cell_id} summary identity drifted.")
    accepted = summary.get("accepted_rounds")
    evidence_rounds = evidence.get("accepted_rounds")
    prefix_rows = (
        checkpoint.get("adapt_vqe", {}).get("active_prefix_checkpoints")
        if isinstance(checkpoint, Mapping)
        else None
    )
    if (
        not isinstance(accepted, list)
        or not accepted
        or not isinstance(evidence_rounds, list)
        or not isinstance(prefix_rows, list)
        or len(accepted) != len(evidence_rounds)
        or len(accepted) != len(prefix_rows)
    ):
        raise ValueError(f"{cell_id} accepted-prefix sources do not close.")

    initial_error = abs(
        float(summary["initial_energy"]) - float(summary["exact_same_cutoff_energy"])
    )
    if not math.isfinite(initial_error) or initial_error <= 0.0:
        raise ValueError(f"{cell_id} initial error is invalid.")
    rounds = [0]
    errors = [initial_error]
    resources: list[Mapping[str, int]] = [
        {field: 0 for field in TUPLE_FIELDS}
    ]
    accepted_deletion_rounds: list[int] = []
    branch_s_alg = 0

    for expected_round, (row, evidence_row, prefix) in enumerate(
        zip(accepted, evidence_rounds, prefix_rows, strict=True), start=1
    ):
        if not all(isinstance(value, Mapping) for value in (row, evidence_row, prefix)):
            raise ValueError(f"{cell_id} accepted prefix row is malformed.")
        if (
            row.get("controller_round") != expected_round
            or evidence_row.get("controller_round") != expected_round
            or prefix.get("outer_iteration") != expected_round
        ):
            raise ValueError(f"{cell_id} accepted rounds are not contiguous.")
        error = float(row.get("same_cutoff_absolute_error"))
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(f"{cell_id} accepted error is invalid.")
        summary_compile = row.get("compile")
        evidence_compile = evidence_row.get("compile")
        if not isinstance(summary_compile, Mapping) or not isinstance(
            evidence_compile, Mapping
        ):
            raise ValueError(f"{cell_id} compile receipt is missing.")
        stored_triplet = {
            field: int(summary_compile[field]) for field in ("N2q", "D2q", "Dc")
        }
        if any(
            int(evidence_compile[field]) != value
            for field, value in stored_triplet.items()
        ):
            raise ValueError(f"{cell_id} compile triplet lost evidence closure.")
        if evidence_row.get("source_checkpoint_sha256") != prefix.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{cell_id} checkpoint identity drifted.")
        ledger = prefix.get("estimator_ledger_receipt")
        cumulative = (
            ledger.get("cumulative_executed_queries")
            if isinstance(ledger, Mapping)
            else None
        )
        delta = (
            ledger.get("executed_query_delta")
            if isinstance(ledger, Mapping)
            else None
        )
        if not isinstance(cumulative, Mapping) or not isinstance(
            delta, Mapping
        ):
            raise ValueError(f"{cell_id} estimator ledger is missing.")
        # S_alg is branch local.  The ledger's cumulative counter is global:
        # on a beam run it charges every executed branch, including the ones
        # the search discarded, which overstates the delivered ansatz's cost
        # roughly fourfold.  Accumulate this branch's own per-round deltas
        # instead, and drop the work spent on prune siblings that were
        # measured and rejected -- those are discarded branches too.
        branch_s_alg += int(delta.get("S_alg"))
        prune_row_for_s_alg = evidence_row.get("pruning")
        if isinstance(prune_row_for_s_alg, Mapping) and str(
            prune_row_for_s_alg.get("status")
        ) == "rejected":
            typed_prune = prune_row_for_s_alg.get("typed_receipt")
            if isinstance(typed_prune, Mapping):
                branch_s_alg -= int(typed_prune.get("trial_s_alg", 0) or 0)
        s_alg = int(branch_s_alg)
        if s_alg < 0:
            raise ValueError(f"{cell_id} S_alg is negative.")
        compile_resources = _post_transpile_resources(
            payload=checkpoint,
            prefix=prefix,
            evidence_compile=evidence_compile,
        )
        one_qubit_work = _termwise_one_qubit_work(
            prefix.get("ordered_active_operators", [])
        )
        pruning = evidence_row.get("pruning")
        if isinstance(pruning, Mapping):
            count = int(pruning.get("accepted_deletion_count", 0))
            if count not in (0, 1):
                raise ValueError(f"{cell_id} prune count is invalid.")
            if count == 1:
                accepted_deletion_rounds.append(expected_round)
        rounds.append(expected_round)
        errors.append(max(error, PLOT_FLOOR))
        resources.append(
            {**compile_resources, "W1q": one_qubit_work, "S_alg": s_alg}
        )

    hits = [
        round_id
        for round_id, error in zip(rounds[1:], errors[1:], strict=True)
        if error <= ACCURACY_TARGET
    ]
    if marker_mode == "crossing":
        if not hits:
            raise ValueError(
                f"{cell_id} never reaches the requested accuracy target."
            )
        marker_round = hits[0]
        marker_reason = f"first same-cutoff error <= {ACCURACY_TARGET:.0e}"
    elif marker_mode == "terminal":
        marker_round = rounds[-1]
        marker_reason = "common terminal horizon"
    elif marker_mode == "crossing_or_terminal":
        # An arm that never reaches the target is reported at the shared
        # terminal round rather than omitted, and the reason records which.
        if hits:
            marker_round = hits[0]
            marker_reason = (
                f"first same-cutoff error <= {ACCURACY_TARGET:.0e}"
            )
        else:
            marker_round = rounds[-1]
            marker_reason = (
                "common terminal horizon; target "
                f"{ACCURACY_TARGET:.0e} not reached"
            )
    else:
        raise ValueError(f"Unknown marker mode: {marker_mode}")

    source_files = {
        "artifact_receipt.json": str(receipt["sha256"]),
        "cell_summary.json": _file_sha(summary_path),
        "accepted_run_evidence.json": _file_sha(evidence_path),
        "observations/current.json": _file_sha(checkpoint_path),
    }
    return Curve(
        cell_id=cell_id,
        label=label,
        color=color,
        marker=marker,
        rounds=tuple(rounds),
        errors=tuple(errors),
        resources=tuple(resources),
        marker_round=marker_round,
        marker_reason=marker_reason,
        accepted_deletion_rounds=tuple(accepted_deletion_rounds),
        source_files=source_files,
    )


def build_figure_specs(
    *, runtime_root: Path = RUNTIME_ROOT
) -> tuple[dict[str, Any], ...]:
    def baseline() -> Curve:
        return load_curve(
            "l3_baseline_h15",
            label="Minimal RA",
            color=BLUE,
            marker="o",
            marker_mode="crossing_or_terminal",
            runtime_root=runtime_root,
        )

    return tuple(
        {
            "slug": f"ra_appendix_l3_{slug}_convergence_v8",
            "kind": kind,
            "curves": (
                baseline(),
                load_curve(
                    cell_id,
                    label=label,
                    color=RED,
                    marker="*",
                    marker_mode="crossing_or_terminal",
                    runtime_root=runtime_root,
                ),
            ),
        }
        for slug, kind, cell_id, label in (
            ("prune", "prune", "l3_metric_ablation_h15", "RA + prune"),
            (
                "prune_gated",
                "prune",
                "l3_metric_ablation_gated_h15",
                "RA + prune (gated)",
            ),
            ("batch", "method", "l3_batching_h15", "RA + batch"),
            ("beam", "method", "l3_beam_h15", "RA + beam"),
            ("beam_cost", "method", "l3_beam_cost_h15", "RA + beam (cost)"),
        )
    )


def _tuple_text(row: Mapping[str, int], *, marker_latex: str) -> str:
    values = ",".join(str(int(row[field])) for field in TUPLE_FIELDS)
    return rf"${marker_latex}\;({values})$"


def _draw_figure(spec: Mapping[str, Any], *, output_root: Path) -> dict[str, str]:
    curves = tuple(spec["curves"])
    fig, ax = plt.subplots(figsize=(3.35, 2.62))
    for curve in curves:
        ax.plot(
            curve.rounds,
            curve.errors,
            color=curve.color,
            linewidth=1.45,
            solid_capstyle="round",
            zorder=2,
        )
        marker_index = curve.marker_index
        marker_size = 67 if curve.marker == "*" else 31
        ax.scatter(
            [curve.marker_round],
            [curve.errors[marker_index]],
            marker=curve.marker,
            s=marker_size,
            color=curve.color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

    maximum_round = max(curve.rounds[-1] for curve in curves)
    all_errors = [error for curve in curves for error in curve.errors]
    ax.set_yscale("log")
    ax.set_xlim(-0.12, maximum_round + 0.20)
    tick_step = 2 if maximum_round > 6 else 1
    ax.set_xticks(range(0, maximum_round + 1, tick_step))
    ax.set_ylim(max(PLOT_FLOOR * 0.55, min(all_errors) * 0.35), max(all_errors) * 2.3)
    ax.set_xlabel(r"Accepted RA round $k$", fontsize=8.2)
    ax.set_ylabel(r"Same-cutoff $|E_k-E_{\rm ED}|/t$", fontsize=8.2)
    ax.tick_params(axis="both", which="major", labelsize=7.1, length=2.8)
    ax.tick_params(axis="y", which="minor", length=1.8)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#ECECEC", linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=curve.color,
            linewidth=1.45,
            marker=curve.marker,
            markersize=5.5 if curve.marker == "*" else 4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=curve.label,
        )
        for curve in curves
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=6.8,
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        handlelength=1.7,
        handletextpad=0.45,
    )

    ax.text(
        0.98,
        0.965,
        rf"${TUPLE_LATEX}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.5},
    )
    tuple_markers = (r"\bullet", r"\diamond", r"\star")
    # Two rows per curve: the matched-accuracy round (where every arm first
    # reaches the target) and the common terminal round.  An extension that
    # is identical at matched accuracy but cheaper at the horizon -- or the
    # reverse -- is invisible with only one of the two.
    row = 0
    for index, curve in enumerate(curves):
        marker_latex = (
            tuple_markers[index]
            if len(curves) == 3
            else (r"\bullet" if index == 0 else r"\star")
        )
        entries = [(curve.marker_round, curve.marker_resources)]
        if curve.terminal_round != curve.marker_round:
            entries.append((curve.terminal_round, curve.terminal_resources))
        for round_id, resources in entries:
            ax.text(
                0.98,
                0.885 - 0.062 * row,
                rf"$k\!=\!{int(round_id)}$  "
                + _tuple_text(resources, marker_latex=marker_latex),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=5.85,
                color=curve.color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.3,
                },
            )
            row += 1

    if spec["kind"] == "prune":
        # Annotate whatever the prune arm actually did.  At L=2 a deletion is
        # accepted; at L=3 the marginal-gain test nominates but never accepts,
        # and the figure has to say so rather than assert the L=2 outcome.
        prune_curve = curves[1]
        deletions = tuple(prune_curve.accepted_deletion_rounds)
        if deletions:
            # Mark every round that accepted a deletion, and label the set
            # once.  One annotation per deletion overlaps illegibly as soon
            # as prune fires in consecutive rounds.
            xs = [r for r in deletions]
            ys = [prune_curve.errors[prune_curve.rounds.index(r)] for r in deletions]
            ax.scatter(
                xs,
                ys,
                s=26,
                marker="v",
                facecolors="none",
                edgecolors=prune_curve.color,
                linewidths=0.9,
                zorder=6,
            )
            span = (
                f"round {deletions[0]}"
                if len(deletions) == 1
                else f"rounds {deletions[0]}\u2013{deletions[-1]}"
            )
            ax.text(
                0.035,
                0.30,
                f"$\\triangledown$ accepted deletion ({span})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=6.1,
                color=prune_curve.color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.35,
                },
            )
        if not deletions:
            ax.text(
                0.5,
                0.035,
                "no deletion accepted",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.1,
                color=prune_curve.color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 0.35,
                },
            )

    fig.subplots_adjust(left=0.185, right=0.985, bottom=0.19, top=0.98)
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = output_root / f"{spec['slug']}.pdf"
    png_path = output_root / f"{spec['slug']}.png"
    metadata = {
        "Title": str(spec["slug"]),
        "Author": "RA-ADAPT appendix evidence workflow",
        "Creator": SCRIPT_PATH.name,
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, format="pdf", metadata=metadata)
    fig.savefig(
        png_path,
        format="png",
        dpi=450,
        metadata={"Software": SCRIPT_PATH.name},
    )
    plt.close(fig)
    return {
        "pdf": _file_sha(pdf_path),
        "png": _file_sha(png_path),
    }


def _curve_provenance(curve: Curve) -> dict[str, Any]:
    return {
        "cell_id": curve.cell_id,
        "label": curve.label,
        "color": curve.color,
        "marker": curve.marker,
        "rounds": list(curve.rounds),
        "same_cutoff_absolute_errors": list(curve.errors),
        "marker_round": curve.marker_round,
        "marker_reason": curve.marker_reason,
        "marker_cost_tuple": {
            field: int(curve.marker_resources[field]) for field in TUPLE_FIELDS
        },
        "terminal_round": int(curve.terminal_round),
        "terminal_cost_tuple": {
            field: int(curve.terminal_resources[field])
            for field in TUPLE_FIELDS
        },
        "accepted_deletion_rounds": list(curve.accepted_deletion_rounds),
        "source_files": dict(curve.source_files),
    }


def _campaign_source_commit(runtime_root: Path) -> str:
    """Read the executed source commit from the campaign's own source facts."""

    facts = _read_json(Path(runtime_root) / "source_facts.json")
    if not isinstance(facts, Mapping):
        raise ValueError("Campaign source facts are malformed.")
    head = str(facts.get("head", ""))
    if len(head) != 40:
        raise ValueError("Campaign source facts lost their executed commit.")
    return head


def generate(
    *,
    runtime_root: Path = RUNTIME_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Mapping[str, Any]:
    specs = build_figure_specs(runtime_root=runtime_root)
    outputs: dict[str, Any] = {}
    for spec in specs:
        outputs[str(spec["slug"])] = {
            "files": _draw_figure(spec, output_root=output_root),
            "curves": [_curve_provenance(curve) for curve in spec["curves"]],
        }
    payload = {
        "schema": "ra_adapt_appendix_l3_convergence_figures_v8_provenance_v1",
        "campaign_id": CAMPAIGN_ID,
        "science_source_git_commit": _campaign_source_commit(runtime_root),
        "generator_representation": "macro_generator_v1",
        "lattice_sites": 3,
        "generator_relative_path": SCRIPT_PATH.relative_to(REPO_ROOT).as_posix(),
        "generator_file_sha256": _file_sha(SCRIPT_PATH),
        "accuracy_target": ACCURACY_TARGET,
        "error_definition": "same_cutoff_absolute_energy_error_over_t",
        "accepted_prefix_semantics": (
            "prefix_closes_immediately_after_accepted_state_and_excludes_later_"
            "selector_scans_and_terminal_only_diagnostics_v1"
        ),
        "cost_tuple_fields": list(TUPLE_FIELDS),
        "s_alg_definition": (
            "branch_local_cumulative_executed_queries_excluding_discarded_"
            "beam_branches_and_rejected_prune_siblings_v1"
        ),
        "n1q_definition": (
            "post_transpile_one_qubit_quantum_ops_excluding_"
            "barrier_delay_id_measure_reset"
        ),
        "w1q_definition": (
            "termwise_pauli_pretranspile_work_sum_2_nx_plus_4_ny_plus_1_v1"
        ),
        "plot_policy": {
            "one_column_one_figure_per_file": True,
            "solid_lines": True,
            "logarithmic_error_axis": True,
            "one_marker_per_trajectory": True,
            "target_guide_line": False,
        },
        "figures": outputs,
    }
    provenance = {**payload, "sha256": _canonical_sha(payload)}
    provenance_path = (
        output_root / "ra_appendix_l3_convergence_figures_v8_provenance.json"
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return provenance


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    provenance = generate(runtime_root=args.runtime_root, output_root=args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "sha256": provenance["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
