#!/usr/bin/env python3
"""Build the Paper-I RA singleton qubit-support and Pauli-axis diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727"
    / "retrieved_phase0_completed_20260809"
)
DEFAULT_OUTPUT_PDF = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_singleton_qubit_support_pauli_diagnostic.pdf"
)
DEFAULT_OUTPUT_TEX = DEFAULT_OUTPUT_PDF.with_suffix(".tex")
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_PDF.with_name(
    f"{DEFAULT_OUTPUT_PDF.stem}_provenance.json"
)
DEFAULT_ASSET_DIR = DEFAULT_OUTPUT_PDF.with_name(
    f"{DEFAULT_OUTPUT_PDF.stem}_assets"
)
DEFAULT_BUILD_DIR = (
    REPO_ROOT / "tmp/pdfs/paper_i_ra_singleton_qubit_support_pauli_diagnostic"
)

CLUSTER_ID = 9605157
ROUTE_ID = "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
AXES = ("x", "y", "z")
AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}
AXIS_COLORS = {
    "x": "#d95f59",
    "y": "#3f78ad",
    "z": "#d09a32",
}
REGIME_ORDER = {
    "weak_weak": 0,
    "intermediate_weak": 1,
    "strong_weak_u8": 2,
    "weak_strong": 3,
    "intermediate_strong": 4,
    "strong_strong_u8": 5,
}

BG = "#f7f6f2"
INK = "#172033"
MUTED = "#5e6878"
GRID = "#d9dee7"


@dataclass(frozen=True)
class Transition:
    controller_round: int
    energy_before: float
    energy_after: float
    pauli_word: str
    support: tuple[int, ...]
    insertion_position: int

    @property
    def raw_drop(self) -> float:
        return max(0.0, float(self.energy_before) - float(self.energy_after))


@dataclass(frozen=True)
class RegimeRun:
    proc_id: int
    regime_id: str
    archive_path: Path
    archive_sha256: str
    archive_size_bytes: int
    receipt_path: Path
    receipt_sha256: str
    adapter_path: Path
    adapter_sha256: str
    n_ph_max: int
    total_qubits: int
    u_over_t: float
    g_ep: float
    omega0: float
    lambda_value: float
    exact_same_cutoff_energy: float
    transitions: tuple[Transition, ...]

    @property
    def sector(self) -> str:
        return "weak" if self.n_ph_max == 3 else "strong"

    @property
    def total_raw_drop(self) -> float:
        return sum(row.raw_drop for row in self.transitions)

    @property
    def initial_energy(self) -> float:
        return self.transitions[0].energy_before

    @property
    def terminal_energy(self) -> float:
        return self.transitions[-1].energy_after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--provenance-json", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--top-supports", type=int, default=12)
    return parser.parse_args()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_self_digest(payload: dict[str, Any], *, path: Path) -> str:
    claimed = str(payload.get("sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual = sha256_bytes(canonical_json_bytes(unsigned))
    if actual != claimed:
        raise ValueError(f"self-digest mismatch: {path}")
    return actual


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    output = dict(unsigned)
    output["sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    encoded = json.dumps(
        output,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "pdf.fonttype": 42,
        }
    )


def extract_pauli_word(selected_operator: Any, *, total_qubits: int) -> str:
    raw = str(selected_operator)
    prefix = "guarded_singleton::"
    if not raw.startswith(prefix):
        raise ValueError(f"unexpected singleton operator label: {raw}")
    word = raw[len(prefix) :].strip().lower().replace("i", "e")
    if len(word) != int(total_qubits):
        raise ValueError(
            f"Pauli word width mismatch: {len(word)} != {total_qubits}: {word}"
        )
    invalid = sorted(set(word).difference({"e", "x", "y", "z"}))
    if invalid:
        raise ValueError(f"invalid Pauli letters in {word}: {invalid}")
    return word


def support_from_word(word: str) -> tuple[int, ...]:
    return tuple(
        int(qubit)
        for qubit, letter in enumerate(reversed(word))
        if letter != "e"
    )


def result_member_name(archive: tarfile.TarFile) -> str:
    matches = [
        name for name in archive.getnames() if name.endswith("/result/result.json")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one result/result.json member, found {len(matches)}")
    return matches[0]


def load_result_fragments(
    archive_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import ijson

    with tarfile.open(archive_path, "r:gz") as archive:
        member = result_member_name(archive)
        problem_stream = archive.extractfile(member)
        if problem_stream is None:
            raise ValueError(f"could not stream {member}")
        problem = next(ijson.items(problem_stream, "run.problem"))
        transition_stream = archive.extractfile(member)
        if transition_stream is None:
            raise ValueError(f"could not restream {member}")
        transitions = list(
            ijson.items(transition_stream, "run.accepted_transitions.item")
        )
    return dict(problem), [dict(row) for row in transitions]


def load_regime_run(source_dir: Path, proc_id: int) -> RegimeRun:
    archive_path = source_dir / f"{CLUSTER_ID}.{proc_id}_full.tar.gz"
    receipt_path = source_dir / f"{CLUSTER_ID}.{proc_id}_retrieval_receipt.json"
    adapter_path = source_dir / f"{CLUSTER_ID}.{proc_id}_completed_report_adapter.json"
    for path in (archive_path, receipt_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    receipt_sha = verify_self_digest(receipt, path=receipt_path)
    adapter_sha = verify_self_digest(adapter, path=adapter_path)
    if receipt.get("status") != "passed" or not bool(
        receipt.get("byte_identity_passed")
    ):
        raise ValueError(f"retrieval receipt did not pass: {receipt_path}")
    if int(receipt.get("cluster_id", -1)) != CLUSTER_ID or int(
        receipt.get("proc_id", -1)
    ) != proc_id:
        raise ValueError(f"retrieval identity mismatch: {receipt_path}")
    if int(adapter.get("cluster_id", -1)) != CLUSTER_ID or int(
        adapter.get("proc_id", -1)
    ) != proc_id:
        raise ValueError(f"adapter identity mismatch: {adapter_path}")
    if int(adapter.get("controller_rounds_completed", -1)) != 50:
        raise ValueError(f"adapter is not a completed k=50 result: {adapter_path}")

    local_archive = receipt.get("local_archive")
    if not isinstance(local_archive, dict):
        raise ValueError(f"local archive receipt missing: {receipt_path}")
    expected_size = int(local_archive.get("size_bytes", -1))
    expected_sha = str(local_archive.get("sha256", ""))
    actual_size = archive_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"archive size mismatch for proc {proc_id}: {actual_size} != {expected_size}"
        )
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_sha:
        raise ValueError(f"archive SHA-256 mismatch for proc {proc_id}")

    problem, raw_transitions = load_result_fragments(archive_path)
    total_qubits = int(problem["total_qubits"])
    transitions: list[Transition] = []
    for expected_round, raw in enumerate(raw_transitions, start=1):
        controller_round = int(raw["controller_round"])
        if controller_round != expected_round:
            raise ValueError(
                f"nonconsecutive controller round for proc {proc_id}: "
                f"{controller_round} != {expected_round}"
            )
        word = extract_pauli_word(
            raw["selected_operator"], total_qubits=total_qubits
        )
        support = support_from_word(word)
        if not support:
            raise ValueError(f"identity singleton was accepted at round {controller_round}")
        transition = Transition(
            controller_round=controller_round,
            energy_before=float(raw["energy_before"]),
            energy_after=float(raw["energy_after"]),
            pauli_word=word,
            support=support,
            insertion_position=int(raw["insertion_position"]),
        )
        if transition.energy_after > transition.energy_before + 1.0e-10:
            raise ValueError(
                f"accepted energy increased at proc {proc_id}, round {controller_round}"
            )
        if transitions and not math.isclose(
            transitions[-1].energy_after,
            transition.energy_before,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"accepted energy chain broke at proc {proc_id}, round {controller_round}"
            )
        transitions.append(transition)
    if len(transitions) != 50:
        raise ValueError(f"expected 50 accepted transitions for proc {proc_id}")

    terminal = adapter.get("terminal")
    if not isinstance(terminal, dict):
        raise ValueError(f"terminal adapter record missing: {adapter_path}")
    if int(terminal.get("k", -1)) != 50 or not math.isclose(
        float(terminal["energy"]),
        transitions[-1].energy_after,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"terminal adapter/result mismatch: {adapter_path}")

    g_ep = float(problem["g_ep"])
    omega0 = float(problem["omega0"])
    hopping = float(problem["t"])
    lambda_value = 2.0 * g_ep * g_ep / (hopping * omega0)
    exact_energy = float(adapter["exact_same_cutoff_energy"])
    return RegimeRun(
        proc_id=proc_id,
        regime_id=str(adapter["regime_id"]),
        archive_path=archive_path,
        archive_sha256=actual_sha,
        archive_size_bytes=actual_size,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        adapter_path=adapter_path,
        adapter_sha256=adapter_sha,
        n_ph_max=int(problem["n_ph_max"]),
        total_qubits=total_qubits,
        u_over_t=float(problem["u"]) / hopping,
        g_ep=g_ep,
        omega0=omega0,
        lambda_value=lambda_value,
        exact_same_cutoff_energy=exact_energy,
        transitions=tuple(transitions),
    )


def exact_support_metrics(run: RegimeRun) -> dict[tuple[int, ...], dict[str, Any]]:
    metrics: dict[tuple[int, ...], dict[str, Any]] = {}
    for transition in run.transitions:
        row = metrics.setdefault(
            transition.support,
            {
                "raw_drop": 0.0,
                "count": 0,
                "axis_raw_drop": {axis: 0.0 for axis in AXES},
                "words": defaultdict(int),
            },
        )
        row["raw_drop"] += transition.raw_drop
        row["count"] += 1
        row["words"][transition.pauli_word] += 1
        width = len(transition.support)
        for axis in AXES:
            row["axis_raw_drop"][axis] += (
                transition.raw_drop * transition.pauli_word.count(axis) / width
            )
    return metrics


def co_support_metrics(run: RegimeRun) -> dict[str, Any]:
    import numpy as np

    nq = run.total_qubits
    pair_drop = np.zeros((nq, nq), dtype=float)
    pair_count = np.zeros((nq, nq), dtype=int)
    axis_drop = np.zeros((3, nq), dtype=float)
    axis_count = np.zeros((3, nq), dtype=int)
    for transition in run.transitions:
        support = transition.support
        width = len(support)
        for qubit in support:
            pair_drop[qubit, qubit] += transition.raw_drop / width
            pair_count[qubit, qubit] += 1
            letter = transition.pauli_word[-1 - qubit]
            axis_row = AXIS_INDEX[letter]
            axis_drop[axis_row, qubit] += transition.raw_drop / width
            axis_count[axis_row, qubit] += 1
        if width >= 2:
            divisor = math.comb(width, 2)
            for left_index, left in enumerate(support):
                for right in support[left_index + 1 :]:
                    pair_drop[left, right] += transition.raw_drop / divisor
                    pair_count[left, right] += 1
    return {
        "pair_drop": pair_drop,
        "pair_count": pair_count,
        "axis_drop": axis_drop,
        "axis_count": axis_count,
    }


def support_label(support: tuple[int, ...]) -> str:
    return "{" + ",".join(f"q{qubit}" for qubit in support) + "}"


def format_share_percent(value: float) -> str:
    if value == 0.0:
        return "0.0%"
    if value < 0.1:
        return "<0.1%"
    return f"{value:.1f}%"


def sector_top_supports(
    runs: Iterable[RegimeRun], *, top_supports: int
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], dict[str, float | int]]]:
    totals: dict[tuple[int, ...], dict[str, float | int]] = defaultdict(
        lambda: {
            "raw_drop": 0.0,
            "count": 0,
            "normalized_share_percent_sum": 0.0,
            "maximum_normalized_share_percent": 0.0,
        }
    )
    for run in runs:
        if run.total_raw_drop <= 0.0:
            raise ValueError(
                f"regime {run.regime_id} must have positive realized path drop"
            )
        for support, row in exact_support_metrics(run).items():
            normalized_share = 100.0 * float(row["raw_drop"]) / run.total_raw_drop
            totals[support]["raw_drop"] = float(totals[support]["raw_drop"]) + float(
                row["raw_drop"]
            )
            totals[support]["count"] = int(totals[support]["count"]) + int(
                row["count"]
            )
            totals[support]["normalized_share_percent_sum"] = float(
                totals[support]["normalized_share_percent_sum"]
            ) + normalized_share
            totals[support]["maximum_normalized_share_percent"] = max(
                float(totals[support]["maximum_normalized_share_percent"]),
                normalized_share,
            )
    ordered = sorted(
        totals,
        key=lambda support: (
            -float(totals[support]["normalized_share_percent_sum"]),
            -float(totals[support]["maximum_normalized_share_percent"]),
            -int(totals[support]["count"]),
            support,
        ),
    )
    return ordered[:top_supports], dict(totals)


def sector_register_map(run: RegimeRun) -> str:
    if run.n_ph_max == 3:
        return (
            "q0=f0(up), q1=f1(up), q2=f0(down), q3=f1(down); "
            "q4-q5=phonon site 0, q6-q7=phonon site 1"
        )
    return (
        "q0=f0(up), q1=f1(up), q2=f0(down), q3=f1(down); "
        "q4-q6=phonon site 0, q7-q9=phonon site 1"
    )


def save_figure_atomic(figure: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.pdf")
    figure.savefig(temporary, format="pdf")
    temporary.replace(destination)


def build_support_bar_page(
    runs: list[RegimeRun],
    *,
    sector_label: str,
    top_supports: int,
    destination: Path,
) -> dict[str, Any]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    selected, sector_totals = sector_top_supports(
        runs, top_supports=top_supports
    )
    selected_set = set(selected)
    category_keys: list[tuple[int, ...] | str] = list(selected) + ["other"]
    category_codes = [f"S{index + 1}" for index in range(len(selected))] + ["S*"]
    run_metrics = {run.regime_id: exact_support_metrics(run) for run in runs}

    raw_stack = np.zeros((len(category_keys), len(runs), 3), dtype=float)
    counts = np.zeros((len(category_keys), len(runs)), dtype=int)
    for run_index, run in enumerate(runs):
        for support, row in run_metrics[run.regime_id].items():
            category_index = (
                selected.index(support)
                if support in selected_set
                else len(category_keys) - 1
            )
            counts[category_index, run_index] += int(row["count"])
            for axis_index, axis in enumerate(AXES):
                raw_stack[category_index, run_index, axis_index] += float(
                    row["axis_raw_drop"][axis]
                )
    regime_raw_totals = np.asarray(
        [run.total_raw_drop for run in runs], dtype=float
    )
    if np.any(regime_raw_totals <= 0.0):
        raise ValueError("each regime must have positive realized path drop")
    stack = 100.0 * raw_stack / regime_raw_totals[np.newaxis, :, np.newaxis]
    heights = stack.sum(axis=2)
    normalized_regime_sums = heights.sum(axis=0)
    if not np.allclose(normalized_regime_sums, 100.0, atol=1.0e-9, rtol=0.0):
        raise ValueError(
            "singleton support shares do not sum to 100 percent within each regime"
        )

    configure_matplotlib()
    fig = plt.figure(figsize=(11.0, 8.5))
    sector_title = "Weak" if sector_label == "weak" else "Strong"
    lambda_value = runs[0].lambda_value
    fig.text(
        0.048,
        0.955,
        f"{sector_title} Holstein sector: normalized singleton descent by exact qubit support",
        fontsize=15.2,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.048,
        0.920,
        rf"Global-singleton RA route, accepted $k=50$; fixed $\lambda={lambda_value:.2f}$; each Hubbard regime is normalized by its own total realized path drop.",
        fontsize=8.8,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.952,
        0.985,
        f"PAPER I DIAGNOSTIC | {sector_title.upper()} HOLSTEIN",
        fontsize=6.8,
        color=MUTED,
        ha="right",
        va="top",
        weight="bold",
    )

    ax = fig.add_axes([0.015, 0.155, 0.735, 0.72], projection="3d")
    ax.set_proj_type("ortho")
    x_positions = np.arange(len(category_keys), dtype=float) * 1.16
    y_positions = np.arange(len(runs), dtype=float) * 1.55
    dx = 0.48
    dy = 0.52
    for category_index in range(len(category_keys)):
        for run_index in range(len(runs)):
            base = 0.0
            for axis_index, axis in enumerate(AXES):
                value = float(stack[category_index, run_index, axis_index])
                if value <= 0.0:
                    continue
                ax.bar3d(
                    x_positions[category_index] - dx / 2,
                    y_positions[run_index] - dy / 2,
                    base,
                    dx,
                    dy,
                    value,
                    color=AXIS_COLORS[axis],
                    edgecolor="#263244",
                    linewidth=0.28,
                    alpha=0.93,
                    shade=True,
                    zsort="average",
                )
                base += value

    largest = max(float(heights.max()), 1.0e-12)
    label_offset = largest * 0.016
    label_floor = largest * 0.010
    for category_index in range(len(category_keys)):
        for run_index in range(len(runs)):
            count = int(counts[category_index, run_index])
            if count <= 0:
                continue
            height = float(heights[category_index, run_index])
            ax.text(
                x_positions[category_index],
                y_positions[run_index],
                max(height, label_floor) + label_offset,
                str(count),
                fontsize=5.8,
                color=INK,
                ha="center",
                va="bottom",
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.07",
                    "facecolor": BG,
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
            )

    ax.set_xlim(-0.7, x_positions[-1] + 0.75)
    ax.set_ylim(-0.72, y_positions[-1] + 0.75)
    ax.set_zlim(0.0, largest * 1.14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_codes, fontsize=6.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"U/t={run.u_over_t:g}" for run in runs], fontsize=7.2)
    ax.set_xlabel("exact singleton support", labelpad=9)
    ax.set_ylabel("Hubbard strength", labelpad=9)
    ax.set_zlabel(r"within-regime realized-drop share  (\%)", labelpad=7)
    ax.view_init(elev=31, azim=-50)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.74))
        axis.pane.set_edgecolor(GRID)

    key_x = 0.775
    fig.text(key_x, 0.865, "PAULI-AXIS CREDIT", fontsize=7.2, color=MUTED, weight="bold")
    for axis_index, axis in enumerate(AXES):
        y = 0.835 - axis_index * 0.032
        fig.patches.append(
            mpl.patches.Rectangle(
                (key_x, y - 0.007),
                0.012,
                0.012,
                transform=fig.transFigure,
                facecolor=AXIS_COLORS[axis],
                edgecolor="none",
            )
        )
        fig.text(key_x + 0.019, y, axis.upper(), fontsize=7.0, color=INK, va="center")

    fig.text(
        key_x,
        0.720,
        "SUPPORT KEY (MEAN REGIME SHARE)",
        fontsize=7.2,
        color=MUTED,
        weight="bold",
    )
    support_receipts: list[dict[str, Any]] = []
    for index, support in enumerate(selected):
        code = category_codes[index]
        total = float(sector_totals[support]["raw_drop"])
        count = int(sector_totals[support]["count"])
        mean_share = float(
            sector_totals[support]["normalized_share_percent_sum"]
        ) / len(runs)
        y = 0.692 - index * 0.036
        fig.text(
            key_x,
            y,
            f"{code} {support_label(support)}  n={count}  avg={format_share_percent(mean_share)}",
            fontsize=5.6,
            family="DejaVu Sans Mono",
            color=INK,
            va="center",
        )
        support_receipts.append(
            {
                "code": code,
                "support": list(support),
                "sector_total_raw_drop": total,
                "mean_within_regime_share_percent": mean_share,
                "sector_count": count,
            }
        )
    other_drop = 0.0
    other_count = 0
    other_share_sum = 0.0
    for support, row in sector_totals.items():
        if support in selected_set:
            continue
        other_drop += float(row["raw_drop"])
        other_count += int(row["count"])
        other_share_sum += float(row["normalized_share_percent_sum"])
    other_mean_share = other_share_sum / len(runs)
    fig.text(
        key_x,
        0.692 - len(selected) * 0.036,
        f"S* other supports  n={other_count}  avg={format_share_percent(other_mean_share)}",
        fontsize=5.6,
        family="DejaVu Sans Mono",
        color=INK,
        va="center",
    )

    fig.text(
        0.048,
        0.100,
        r"Each accepted word contributes $d_k=\max(0,E_{k-1}-E_k)$.  Bar height is "
        r"$p_{S,r}=100\sum_{k:S_k=S}d_k/\sum_kd_k$; within every regime, $\sum_Sp_{S,r}=100\%$.",
        fontsize=7.25,
        color=INK,
        va="top",
    )
    fig.text(
        0.048,
        0.063,
        "Integer labels are accepted occurrence counts. This is path-dependent realized descent, not a connected-correlation observable or a causal ablation.",
        fontsize=6.75,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.048,
        0.036,
        f"Register: {sector_register_map(runs[0])}. Pauli words are q_(n-1)...q_0. Source: cluster {CLUSTER_ID}; Page-12 global-singleton Phase-0 route.",
        fontsize=6.3,
        color=MUTED,
        va="bottom",
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)
    return {
        "page_kind": "stacked_3d_exact_support",
        "sector": sector_label,
        "lambda": lambda_value,
        "top_support_count": len(selected),
        "support_order": support_receipts,
        "other": {
            "code": "S*",
            "sector_total_raw_drop": other_drop,
            "mean_within_regime_share_percent": other_mean_share,
            "sector_count": other_count,
        },
        "height_scale": "linear_percent_of_each_regime_total_realized_drop",
        "normalization": "100*support_regime_raw_drop/sum_support(support_regime_raw_drop)",
        "stack_semantics": "within_regime_normalized_fractional_X_Y_Z_letter_credit",
        "count_label_semantics": "accepted_occurrence_count",
        "regime_raw_drop_denominators": {
            run.regime_id: float(total)
            for run, total in zip(runs, regime_raw_totals, strict=True)
        },
        "regime_normalized_share_sums": {
            run.regime_id: float(total)
            for run, total in zip(runs, normalized_regime_sums, strict=True)
        },
    }


def draw_register_boundaries(ax: Any, *, n_ph_max: int) -> None:
    boundaries = [3.5, 5.5 if n_ph_max == 3 else 6.5]
    for boundary in boundaries:
        ax.axvline(boundary, color="#ffffff", linewidth=1.5, alpha=0.95)
        ax.axhline(boundary, color="#ffffff", linewidth=1.5, alpha=0.95)


def build_co_support_page(
    runs: list[RegimeRun],
    *,
    sector_label: str,
    destination: Path,
) -> dict[str, Any]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib()
    fig = plt.figure(figsize=(11.0, 8.5))
    sector_title = "Weak" if sector_label == "weak" else "Strong"
    lambda_value = runs[0].lambda_value
    fig.text(
        0.048,
        0.955,
        f"{sector_title} Holstein sector: singleton qubit co-support and Pauli-axis fingerprints",
        fontsize=14.8,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.048,
        0.920,
        rf"Fixed $\lambda={lambda_value:.2f}$; color is percent of each regime's total realized $k=50$ energy descent; integers are accepted occurrence counts.",
        fontsize=8.6,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.952,
        0.985,
        f"PAPER I DIAGNOSTIC | {sector_title.upper()} HOLSTEIN",
        fontsize=6.8,
        color=MUTED,
        ha="right",
        va="top",
        weight="bold",
    )

    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.915,
        bottom=0.165,
        top=0.865,
        height_ratios=(4.35, 1.40),
        hspace=0.22,
        wspace=0.22,
    )
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad(color=BG, alpha=1.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=100.0)
    page_metrics: list[dict[str, Any]] = []
    image = None
    for column, run in enumerate(runs):
        metrics = co_support_metrics(run)
        pair_drop = metrics["pair_drop"]
        pair_count = metrics["pair_count"]
        axis_drop = metrics["axis_drop"]
        axis_count = metrics["axis_count"]
        total = run.total_raw_drop
        pair_percent = 100.0 * pair_drop / total
        axis_percent = 100.0 * axis_drop / total
        nq = run.total_qubits

        pair_ax = fig.add_subplot(grid[0, column])
        lower_mask = np.tril(np.ones((nq, nq), dtype=bool), k=-1)
        masked_pair = np.ma.array(pair_percent, mask=lower_mask)
        image = pair_ax.imshow(
            masked_pair,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        pair_ax.set_title(
            rf"$U/t={run.u_over_t:g}$" + "\n" + rf"total $\sum d_k={total:.4g}$",
            fontsize=8.2,
            weight="bold",
            pad=5,
        )
        pair_ax.set_xticks(range(nq))
        pair_ax.set_yticks(range(nq))
        pair_ax.set_xticklabels([f"q{q}" for q in range(nq)], fontsize=6.3)
        pair_ax.set_yticklabels([f"q{q}" for q in range(nq)], fontsize=6.3)
        pair_ax.set_xlabel("second supported qubit", fontsize=7.1)
        if column == 0:
            pair_ax.set_ylabel("first supported qubit", fontsize=7.1)
        pair_ax.set_xticks(np.arange(-0.5, nq, 1), minor=True)
        pair_ax.set_yticks(np.arange(-0.5, nq, 1), minor=True)
        pair_ax.grid(which="minor", color="#ffffff", linewidth=0.42, alpha=0.50)
        pair_ax.tick_params(which="minor", bottom=False, left=False)
        draw_register_boundaries(pair_ax, n_ph_max=run.n_ph_max)
        for row in range(nq):
            for col in range(row, nq):
                count = int(pair_count[row, col])
                if count <= 0:
                    continue
                value = float(pair_percent[row, col])
                text_color = "white" if value >= 42.0 else INK
                pair_ax.text(
                    col,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=5.45,
                    color=text_color,
                    weight="bold",
                )

        axis_ax = fig.add_subplot(grid[1, column])
        axis_ax.imshow(
            axis_percent,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )
        axis_ax.set_aspect("auto", adjustable="box")
        axis_ax.set_xticks(range(nq))
        axis_ax.set_xticklabels([f"q{q}" for q in range(nq)], fontsize=6.3)
        axis_ax.set_yticks(range(3))
        axis_ax.set_yticklabels(["X", "Y", "Z"], fontsize=7.0)
        axis_ax.set_xlabel("qubit", fontsize=7.1)
        if column == 0:
            axis_ax.set_ylabel("Pauli letter", fontsize=7.1)
        axis_ax.set_xticks(np.arange(-0.5, nq, 1), minor=True)
        axis_ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis_ax.grid(which="minor", color="#ffffff", linewidth=0.42, alpha=0.50)
        axis_ax.tick_params(which="minor", bottom=False, left=False)
        for boundary in (3.5, 5.5 if run.n_ph_max == 3 else 6.5):
            axis_ax.axvline(boundary, color="#ffffff", linewidth=1.5, alpha=0.95)
        for axis_index in range(3):
            for qubit in range(nq):
                count = int(axis_count[axis_index, qubit])
                if count <= 0:
                    continue
                value = float(axis_percent[axis_index, qubit])
                text_color = "white" if value >= 42.0 else INK
                axis_ax.text(
                    qubit,
                    axis_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=5.45,
                    color=text_color,
                    weight="bold",
                )

        page_metrics.append(
            {
                "regime_id": run.regime_id,
                "u_over_t": run.u_over_t,
                "total_raw_drop": total,
                "pair_percent": pair_percent.tolist(),
                "pair_count": pair_count.tolist(),
                "axis_percent": axis_percent.tolist(),
                "axis_count": axis_count.tolist(),
            }
        )

    if image is None:
        raise ValueError("co-support page received no runs")
    color_ax = fig.add_axes([0.935, 0.245, 0.016, 0.545])
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_label("realized-drop credit (%)", fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.3)

    fig.text(
        0.048,
        0.112,
        r"Upper triangle: $C_{ij}=\sum_{k:\,i,j\in S_k} d_k/\binom{|S_k|}{2}$; diagonal: $C_{ii}=\sum_{k:\,i\in S_k}d_k/|S_k|$. "
        r"The divisor prevents a wide Pauli word from receiving duplicated pair credit.",
        fontsize=6.75,
        color=INK,
        va="top",
    )
    fig.text(
        0.048,
        0.076,
        r"Axis panels: $A_{a q}=\sum_{k:\,P_{k,q}=a}d_k/|S_k|$, $a\in\{X,Y,Z\}$. "
        "All color scales are linear from 0 to 100%; blank pair cells were never jointly supported.",
        fontsize=6.65,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.048,
        0.039,
        f"Register: {sector_register_map(runs[0])}. White separators mark fermion/phonon and phonon-site boundaries. Source: cluster {CLUSTER_ID}, k=50.",
        fontsize=6.3,
        color=MUTED,
        va="bottom",
    )
    save_figure_atomic(fig, destination)
    plt.close(fig)
    return {
        "page_kind": "co_support_and_pauli_axis_heatmaps",
        "sector": sector_label,
        "lambda": lambda_value,
        "color_scale": "linear_percent_of_regime_total_raw_realized_drop_0_to_100",
        "integer_label_semantics": "accepted_occurrence_count",
        "regimes": page_metrics,
    }


def tex_escape_path(path: str) -> str:
    return path.replace("\\", "/")


def build_tex(output_tex: Path, page_assets: list[Path]) -> None:
    relative_assets = [
        tex_escape_path(str(path.resolve().relative_to(output_tex.parent.resolve())))
        for path in page_assets
    ]
    lines = [
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[margin=0in]{geometry}",
        r"\usepackage{pdfpages}",
        r"\pagestyle{empty}",
        r"\begin{document}",
    ]
    for asset in relative_assets:
        lines.append(rf"\includepdf[pages=1,fitpaper=true]{{{asset}}}")
    lines.append(r"\end{document}")
    encoded = ("\n".join(lines) + "\n").encode("ascii")
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_tex.with_name(f".{output_tex.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output_tex)


def compile_tex(output_tex: Path, output_pdf: Path, build_dir: Path) -> None:
    from pypdf import PdfReader

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required for this evidence PDF")
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        latexmk,
        "-g",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={build_dir}",
        output_tex.name,
    ]
    completed = subprocess.run(
        command,
        cwd=output_tex.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX build failed:\n"
            + completed.stdout[-8000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    built_pdf = build_dir / output_pdf.name
    if not built_pdf.is_file():
        raise FileNotFoundError(built_pdf)
    reader = PdfReader(str(built_pdf))
    if len(reader.pages) != 4:
        raise ValueError(f"expected four pages, found {len(reader.pages)}")
    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if not math.isclose(width, 792.0, abs_tol=1.0) or not math.isclose(
            height, 612.0, abs_tol=1.0
        ):
            raise ValueError(
                f"page {index} is not US-letter landscape: {width} x {height}"
            )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_pdf.with_name(f".{output_pdf.name}.tmp")
    shutil.copy2(built_pdf, temporary)
    temporary.replace(output_pdf)


def serializable_support_metrics(run: RegimeRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for support, metrics in sorted(
        exact_support_metrics(run).items(),
        key=lambda item: (-float(item[1]["raw_drop"]), item[0]),
    ):
        rows.append(
            {
                "support": list(support),
                "support_label": support_label(support),
                "raw_drop": float(metrics["raw_drop"]),
                "count": int(metrics["count"]),
                "axis_raw_drop": {
                    axis: float(metrics["axis_raw_drop"][axis]) for axis in AXES
                },
                "words": dict(sorted(metrics["words"].items())),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.top_supports <= 0:
        raise ValueError("--top-supports must be positive")
    runs = [load_regime_run(args.source_dir, proc_id) for proc_id in range(6)]
    runs.sort(key=lambda run: REGIME_ORDER[run.regime_id])
    if [run.regime_id for run in runs] != list(REGIME_ORDER):
        raise ValueError("six-regime ordering/identity mismatch")
    weak_runs = [run for run in runs if run.sector == "weak"]
    strong_runs = [run for run in runs if run.sector == "strong"]
    if len(weak_runs) != 3 or len(strong_runs) != 3:
        raise ValueError("expected three regimes in each Holstein sector")
    if {run.total_qubits for run in weak_runs} != {8}:
        raise ValueError("weak-Holstein singleton register must contain eight qubits")
    if {run.total_qubits for run in strong_runs} != {10}:
        raise ValueError("strong-Holstein singleton register must contain ten qubits")

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    page_assets = [
        args.asset_dir / "page01_weak_holstein_exact_support_bar3d.pdf",
        args.asset_dir / "page02_strong_holstein_exact_support_bar3d.pdf",
        args.asset_dir / "page03_weak_holstein_co_support_heatmaps.pdf",
        args.asset_dir / "page04_strong_holstein_co_support_heatmaps.pdf",
    ]
    page_receipts = [
        build_support_bar_page(
            weak_runs,
            sector_label="weak",
            top_supports=args.top_supports,
            destination=page_assets[0],
        ),
        build_support_bar_page(
            strong_runs,
            sector_label="strong",
            top_supports=args.top_supports,
            destination=page_assets[1],
        ),
        build_co_support_page(
            weak_runs,
            sector_label="weak",
            destination=page_assets[2],
        ),
        build_co_support_page(
            strong_runs,
            sector_label="strong",
            destination=page_assets[3],
        ),
    ]
    build_tex(args.output_tex, page_assets)
    compile_tex(args.output_tex, args.output_pdf, args.build_dir)

    provenance = {
        "schema": "paper_i_ra_singleton_qubit_support_pauli_diagnostic_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": {
            "paper_lane": "Paper I static Hubbard-Holstein diagnostic",
            "route_id": ROUTE_ID,
            "cluster_id": CLUSTER_ID,
            "candidate_representation": "single_pauli_word_v1",
            "accepted_horizon": 50,
            "selection_semantics": "accepted_occurrences_in_controller_order",
            "energy_semantics": "same_cutoff_path_dependent_realized_energy_descent",
            "causal_claim": False,
            "connected_correlation_claim": False,
        },
        "metric_contract": {
            "transition_drop": "d_k=max(0,E_before_k-E_after_k)",
            "exact_support_drop": "D_S=sum_{k:S_k=S} d_k",
            "exact_support_count": "N_S=sum_k 1[S_k=S]",
            "axis_support_credit": "D_{S,a}=sum_{k:S_k=S} d_k*n_a(P_k)/|S_k|",
            "within_regime_support_share": "p_{S,r}=100*D_{S,r}/sum_S D_{S,r}",
            "within_regime_axis_support_share": "p_{S,a,r}=100*D_{S,a,r}/sum_S D_{S,r}",
            "pair_credit": "C_ij=sum_{k:i,j in S_k} d_k/binom(|S_k|,2)",
            "diagonal_credit": "C_ii=sum_{k:i in S_k} d_k/|S_k|",
            "qubit_axis_credit": "A_aq=sum_{k:P_kq=a} d_k/|S_k|",
            "pauli_order": "q_(n-1)...q_0; qubit 0 is rightmost",
        },
        "figure_contract": {
            "page_count": 4,
            "top_supports_per_sector": args.top_supports,
            "other_support_bucket": True,
            "bar_height_scale": "linear_percent_of_each_regime_total_realized_drop",
            "bar_stack": "X_Y_Z_fractional_letter_credit",
            "heatmap_scale": "linear_0_to_100_percent_of_regime_total_drop",
            "cell_labels": "accepted_occurrence_counts",
        },
        "pages": [
            {
                **receipt,
                "path": repo_relative(path),
                "sha256": sha256_file(path),
            }
            for receipt, path in zip(page_receipts, page_assets, strict=True)
        ],
        "regimes": [
            {
                "proc_id": run.proc_id,
                "regime_id": run.regime_id,
                "n_ph_max": run.n_ph_max,
                "total_qubits": run.total_qubits,
                "u_over_t": run.u_over_t,
                "g_ep": run.g_ep,
                "omega0": run.omega0,
                "lambda": run.lambda_value,
                "exact_same_cutoff_energy": run.exact_same_cutoff_energy,
                "accepted_rounds": len(run.transitions),
                "initial_energy": run.initial_energy,
                "terminal_energy": run.terminal_energy,
                "total_raw_drop": run.total_raw_drop,
                "archive": {
                    "path": repo_relative(run.archive_path),
                    "size_bytes": run.archive_size_bytes,
                    "sha256": run.archive_sha256,
                },
                "retrieval_receipt": {
                    "path": repo_relative(run.receipt_path),
                    "sha256": run.receipt_sha256,
                },
                "completed_report_adapter": {
                    "path": repo_relative(run.adapter_path),
                    "sha256": run.adapter_sha256,
                },
                "exact_supports": serializable_support_metrics(run),
            }
            for run in runs
        ],
        "outputs": {
            "pdf": {
                "path": repo_relative(args.output_pdf),
                "sha256": sha256_file(args.output_pdf),
                "page_count": 4,
            },
            "tex": {
                "path": repo_relative(args.output_tex),
                "sha256": sha256_file(args.output_tex),
            },
        },
    }
    write_json_atomic(args.provenance_json, provenance)
    final_provenance = json.loads(args.provenance_json.read_text(encoding="utf-8"))
    verify_self_digest(final_provenance, path=args.provenance_json)
    print(
        json.dumps(
            {
                "pdf": str(args.output_pdf),
                "pdf_sha256": sha256_file(args.output_pdf),
                "provenance": str(args.provenance_json),
                "provenance_sha256": sha256_file(args.provenance_json),
                "pages": 4,
                "regimes": [run.regime_id for run in runs],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
