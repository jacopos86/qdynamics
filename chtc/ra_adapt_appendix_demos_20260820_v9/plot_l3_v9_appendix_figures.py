"""L=3 appendix figures for the v9 campaign.

Three controlled comparisons against the RA-append baseline (batch, generator
ablation, cost-weighted beam), plus the noise and AAVQE-comparator panels.

The beam panel draws the search itself: at each round the live parent splits
into children, the roads not taken are drawn as short stubs, and the winning
lineage continues as the reported trajectory.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_ID = "ra_adapt_appendix_demos_l3_20260820_v9"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
REPORT_ROOT = REPO_ROOT / "chtc/ra_adapt_appendix_demos_20260820/results_l3_v9"
EXACT_ENERGY = -2.1414935642714075
ACCURACY_TARGET = 1.0e-6
PLOT_FLOOR = 1.0e-16
BASELINE_LABEL = "RA-append"

_V8_PLOT = (
    REPO_ROOT
    / "chtc/ra_adapt_appendix_demos_20260820_v8"
    / "plot_l3_v8_appendix_convergence.py"
)


def _v8():
    spec = importlib.util.spec_from_file_location("plotv8", _V8_PLOT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plotv8"] = module
    spec.loader.exec_module(module)
    return module


HELP = _v8()


def load_cell(cell_id: str, runtime_root: Path = RUNTIME_ROOT) -> dict:
    """Errors, resource tuples and (for beam cells) the branch tree."""

    cell_dir = runtime_root / "cells" / cell_id
    evidence = json.loads((cell_dir / "accepted_run_evidence.json").read_text())
    checkpoint = json.loads((cell_dir / "observations/current.json").read_text())
    rows = evidence["accepted_rounds"]
    prefixes = checkpoint["adapt_vqe"]["active_prefix_checkpoints"]

    rounds = [0]
    errors = [abs(float(checkpoint["adapt_vqe"].get("initial_energy", 1.25))
                  - EXACT_ENERGY)]
    resources = [{k: 0 for k in ("N2q", "D2q", "Dc", "N1q", "W1q", "S_alg")}]
    deletions: list[int] = []
    branch_s_alg = 0
    for index, (row, prefix) in enumerate(zip(rows, prefixes, strict=True), 1):
        ledger = prefix["estimator_ledger_receipt"]["executed_query_delta"]
        branch_s_alg += int(ledger["S_alg"])
        pruning = row.get("pruning")
        if isinstance(pruning, Mapping):
            if str(pruning.get("status")) == "rejected":
                typed = pruning.get("typed_receipt")
                if isinstance(typed, Mapping):
                    branch_s_alg -= int(typed.get("trial_s_alg", 0) or 0)
            if int(pruning.get("accepted_deletion_count", 0)) == 1:
                deletions.append(index)
        compiled = HELP._post_transpile_resources(
            payload=checkpoint, prefix=prefix, evidence_compile=row["compile"]
        )
        rounds.append(index)
        errors.append(
            max(abs(float(row["accepted_energy"]) - EXACT_ENERGY), PLOT_FLOOR)
        )
        resources.append(
            {
                **compiled,
                "W1q": HELP._termwise_one_qubit_work(
                    prefix.get("ordered_active_operators", [])
                ),
                "S_alg": int(branch_s_alg),
            }
        )
    marker = next(
        (r for r, e in zip(rounds[1:], errors[1:], strict=True)
         if e <= ACCURACY_TARGET),
        None,
    )
    return {
        "cell_id": cell_id,
        "rounds": rounds,
        "errors": errors,
        "resources": resources,
        "accepted_deletion_rounds": deletions,
        "marker_round": marker,
        "beam_tree": _beam_tree(evidence, errors[0]),
    }


def _beam_tree(evidence: Mapping, initial_error: float):
    """Every executed child as (round, parent_error, child_error, fate)."""

    diagnostics = evidence.get("beam_search_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    error_by_branch: dict[str, float] = {}
    segments = []
    for index, round_row in enumerate(diagnostics["rounds"], start=1):
        survivors = set(round_row.get("survivor_branch_ids") or ())
        for child in round_row["children"]:
            child_id = str(child["branch_id"])
            parent_id = child.get("parent_branch_id")
            parent_error = (
                initial_error
                if parent_id is None
                else error_by_branch.get(str(parent_id), initial_error)
            )
            child_error = max(
                abs(float(child["accepted_energy"]) - EXACT_ENERGY), PLOT_FLOOR
            )
            error_by_branch[child_id] = child_error
            segments.append(
                {
                    "round": index,
                    "parent_error": parent_error,
                    "child_error": child_error,
                    "survived": child_id in survivors,
                }
            )
    return segments


def _tuple_text(row: Mapping) -> str:
    return (
        f"({row['N2q']}, {row['D2q']}, {row['Dc']}, "
        f"{row['N1q']}, {row['W1q']}, {row['S_alg']})"
    )


def draw(spec: Mapping, output_root: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.6, 4.3))
    for curve in spec["curves"]:
        tree = curve.get("beam_tree")
        if tree:
            for segment in tree:
                axis.plot(
                    [segment["round"] - 1, segment["round"]],
                    [segment["parent_error"], segment["child_error"]],
                    color=curve["color"],
                    alpha=0.55 if segment["survived"] else 0.22,
                    linewidth=0.9 if segment["survived"] else 0.7,
                    linestyle="-" if segment["survived"] else (0, (2, 1.5)),
                    zorder=1,
                )
        axis.plot(
            curve["rounds"], curve["errors"],
            color=curve["color"], marker=curve["marker"], markersize=4.0,
            linewidth=1.7, label=curve["label"], zorder=3,
        )
        for round_id in curve.get("accepted_deletion_rounds", ()):
            index = curve["rounds"].index(round_id)
            axis.annotate(
                "deletion", xy=(round_id, curve["errors"][index]),
                xytext=(round_id - 1.6, curve["errors"][index] * 45.0),
                fontsize=7, color=curve["color"],
                arrowprops={"arrowstyle": "->", "color": curve["color"],
                            "linewidth": 0.9},
            )
    axis.axhline(ACCURACY_TARGET, color="0.45", linewidth=0.9,
                 linestyle=":", zorder=0)
    axis.text(0.4, ACCURACY_TARGET * 1.5, r"$10^{-6}t$", fontsize=7,
              color="0.35")
    axis.set_yscale("log")
    axis.set_xlabel("accepted iteration $k$")
    axis.set_ylabel(r"$|E_k - E_{\rm ED}|\;/\;t$")
    axis.set_title(spec["title"], fontsize=10)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(fontsize=8, loc="upper right", framealpha=0.95)

    lines = []
    for curve in spec["curves"]:
        if curve["marker_round"] is None:
            lines.append(f"{curve['label']}: no crossing")
            continue
        index = curve["rounds"].index(curve["marker_round"])
        lines.append(
            f"{curve['label']} @ k={curve['marker_round']}: "
            f"{_tuple_text(curve['resources'][index])}"
        )
    axis.text(
        0.015, 0.030, "\n".join(lines), transform=axis.transAxes, fontsize=6.6,
        va="bottom", ha="left", family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "0.75", "linewidth": 0.6, "alpha": 0.95},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / spec["name"]
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem.with_suffix(".png")




def _termwise_work_from_terms(operators) -> int:
    """W1q for operators held as pool terms rather than serialized rows.

    Same count the RA path applies to its serialized prefix rows -- two units
    per X, four per Y, one per term -- reached through the term objects
    because Append's checkpoint stores labels, not serialized Pauli words.
    """

    total = 0
    for operator in operators:
        for term in operator.polynomial.return_polynomial():
            word = str(term.pw2strng()).lower()
            if not word or set(word) - set("exyz"):
                raise ValueError(
                    "Comparator W1q requires an e/x/y/z Pauli word."
                )
            total += 2 * word.count("x") + 4 * word.count("y") + 1
    return total


AAVQE_RUN = Path(
    "/private/tmp/claude-501/"
    "-Users-jakestrobel-Documents-Holstein-implementation-"
    "Holstein-test-fullclone-3/73d98d63-6b65-46a4-be2e-4456d522f086/"
    "scratchpad/run_aavqe_none"
)
AAVQE_LABEL = "AAVQE"


def load_aavqe(run_dir: Path = AAVQE_RUN, horizon: int = 15) -> dict:
    """Comparator curve, priced on the same backend as the RA arms.

    Append records compiled resources only at rounds it was asked for, so the
    tuples are recomputed here from each round's accepted operator labels
    through the same transpile path the RA cells use.  The estimator column is
    Append's own cumulative prefix count, which has no discarded-branch
    component to exclude.
    """

    from pipelines.hardcoded.adapt_circuit_execution import (
        build_structural_ansatz_circuit,
    )
    from pipelines.qiskit_backend_tools import (
        compile_circuit_for_backend,
        compiled_gate_stats,
        resolve_backend_targets,
        safe_circuit_depth,
    )
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        RAAdaptAppendixPureHubbardCandidateAdapter,
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_generator_representation,
    )

    payload = json.loads((run_dir / "result.json").read_text())
    history = payload["result_payload"]["history"][:horizon]
    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    adapter = RAAdaptAppendixPureHubbardCandidateAdapter(
        candidate_representation_id=(
            ra_adapt_appendix_generator_representation(3)
        )
    )
    pool = {
        str(record.label): record
        for record in adapter.executable_pool(problem).candidates
    }
    reference = np.asarray(
        problem.reference_state.build_state(),
        dtype=complex,
    ).reshape(-1)
    num_qubits = int(round(math.log2(reference.size)))
    targets, _audit = resolve_backend_targets(
        requested_names=("FakeMarrakesh",),
        preferred_fake_backends=("FakeMarrakesh",),
        allow_preferred_fallback=False,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )

    rounds = [0]
    errors = [abs(1.25 - EXACT_ENERGY)]
    resources = [{k: 0 for k in ("N2q", "D2q", "Dc", "N1q", "W1q", "S_alg")}]
    for row in history:
        checkpoint = row["active_prefix_checkpoint"]
        operators = [
            pool[str(label)].term
            for label in checkpoint["accepted_operator_labels"]
        ]
        _layout, circuit = build_structural_ansatz_circuit(
            operators,
            nq=num_qubits,
            ref_state=reference,
            structure_theta_value=1.0,
        )
        compiled = compile_circuit_for_backend(
            circuit,
            targets[0].backend_obj,
            seed_transpiler=7,
            optimization_level=1,
            initial_layout=None,
        )["compiled"]
        stats = compiled_gate_stats(compiled)
        rounds.append(int(row["controller_round"]))
        errors.append(
            max(
                abs(float(row["energy_after"]) - EXACT_ENERGY),
                PLOT_FLOOR,
            )
        )
        resources.append(
            {
                "N2q": int(stats["compiled_count_2q"]),
                "D2q": int(stats["compiled_depth_2q"]),
                "Dc": int(safe_circuit_depth(compiled)),
                "N1q": int(stats["compiled_count_1q"]),
                "W1q": _termwise_work_from_terms(operators),
                "S_alg": int(
                    checkpoint["estimator_prefix"]
                    ["cumulative_executed_queries"]["S_alg"]
                ),
            }
        )
    marker = next(
        (
            r
            for r, e in zip(rounds[1:], errors[1:], strict=True)
            if e <= ACCURACY_TARGET
        ),
        None,
    )
    return {
        "cell_id": "aavqe_h20",
        "rounds": rounds,
        "errors": errors,
        "resources": resources,
        "accepted_deletion_rounds": [],
        "marker_round": marker,
        "beam_tree": None,
    }


BASELINE_CELL = "l3_baseline_h15"
FIGURE_SPECS = (
    {
        "name": "ra_appendix_l3_batch_convergence_v9",
        "title": "RA-append vs combinatorial batch admission "
                 "($L=3$ Hubbard, $U/t=1.25$)",
        "cells": (
            (BASELINE_CELL, BASELINE_LABEL, "#3B6EA5", "o"),
            ("l3_batching_h15", "RA + batch", "#C44E52", "*"),
            ("aavqe", AAVQE_LABEL, "#7F7F7F", "x"),
        ),
    },
    {
        "name": "ra_appendix_l3_prune_convergence_v9",
        "title": "RA-append vs measured generator ablation "
                 "($L=3$ Hubbard, $U/t=1.25$)",
        "cells": (
            (BASELINE_CELL, BASELINE_LABEL, "#3B6EA5", "o"),
            ("l3_metric_ablation_gated_h15", "RA + ablation", "#C44E52", "D"),
            ("aavqe", AAVQE_LABEL, "#7F7F7F", "x"),
        ),
    },
    {
        "name": "ra_appendix_l3_beam_convergence_v9",
        "title": "RA-append vs cost-weighted beam continuation "
                 "($L=3$ Hubbard, $U/t=1.25$)",
        "cells": (
            (BASELINE_CELL, BASELINE_LABEL, "#3B6EA5", "o"),
            ("l3_beam_escalated_h15", "RA + beam", "#C44E52", "^"),
            ("aavqe", AAVQE_LABEL, "#7F7F7F", "x"),
        ),
    },
)


def build(output_root: Path = REPORT_ROOT / "figures") -> dict:
    loaded: dict[str, dict] = {}
    written = {}
    for spec in FIGURE_SPECS:
        curves = []
        for cell_id, label, color, marker in spec["cells"]:
            if cell_id not in loaded:
                loaded[cell_id] = (
                    load_aavqe()
                    if cell_id == "aavqe"
                    else load_cell(cell_id)
                )
            curve = dict(loaded[cell_id])
            curve.update(label=label, color=color, marker=marker,
                         beam_tree=None)
            curves.append(curve)
        path = draw({**spec, "curves": curves}, output_root)
        written[spec["name"]] = str(path)
        print(f"wrote {path}")

    provenance = {
        "schema": "ra_adapt_appendix_l3_v9_figure_provenance_v1",
        "campaign_id": CAMPAIGN_ID,
        "baseline_label": BASELINE_LABEL,
        "accuracy_target": ACCURACY_TARGET,
        "cost_tuple_fields": ["N2q", "D2q", "Dc", "N1q", "W1q", "S_alg"],
        "s_alg_scope": "branch_local_winning_lineage_v1",
        "cells": {
            cell_id: {
                "marker_round": record["marker_round"],
                "terminal_cost_tuple": record["resources"][-1],
                "marker_cost_tuple": (
                    None
                    if record["marker_round"] is None
                    else record["resources"][
                        record["rounds"].index(record["marker_round"])
                    ]
                ),
                "same_cutoff_absolute_errors": record["errors"],
                "accepted_deletion_rounds": record["accepted_deletion_rounds"],
            }
            for cell_id, record in loaded.items()
        },
        "figures": written,
    }
    (output_root / "ra_appendix_l3_v9_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True)
    )
    return provenance


if __name__ == "__main__":
    build()
