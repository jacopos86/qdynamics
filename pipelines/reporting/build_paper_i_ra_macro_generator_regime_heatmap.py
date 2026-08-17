#!/usr/bin/env python3
"""Build a six-regime macro-generator accepted-drop heat map for Paper I.

The figure uses the source-locked stationary-core RA-ADAPT macro append-only
trajectories.  For generator class ``t`` and interaction regime ``r`` it shows

    R[t, r] = sum_k max(0, E_before[k] - E_after[k]),

where the sum includes each accepted occurrence of class ``t`` once.  This is
an admission-path diagnostic, not leave-one-out attribution in the terminal
ansatz.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile
import textwrap
from typing import Any, BinaryIO

import ijson
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from pipelines.contracts.static_provenance import classify_hh_full_meta_label


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    REPO_ROOT
    / "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v8_9392920_20260729"
)
RESULT_MEMBER = "worker_outputs/result.json"
ROUTE_LABEL = "stationary-core RA-ADAPT macro append-only"
CLUSTER_ID = 9392920
EXPECTED_ROUNDS = 50
DISPLAY_FLOOR = 1e-12

REGIMES = (
    {
        "id": "weak_weak",
        "proc": 0,
        "archive": "core__weak_weak__nph3__ra_macro_append_only__cluster_9392920__proc_0.tar.gz",
        "u_label": "weak U",
        "lambda_label": "weak e-ph",
    },
    {
        "id": "intermediate_weak",
        "proc": 1,
        "archive": "core__intermediate_weak__nph3__ra_macro_append_only__cluster_9392920__proc_1.tar.gz",
        "u_label": "intermediate U",
        "lambda_label": "weak e-ph",
    },
    {
        "id": "strong_weak_u8",
        "proc": 2,
        "archive": "core__strong_weak_u8__nph3__ra_macro_append_only__cluster_9392920__proc_2.tar.gz",
        "u_label": "strong U",
        "lambda_label": "weak e-ph",
    },
    {
        "id": "weak_strong",
        "proc": 3,
        "archive": "core__weak_strong__nph7__ra_macro_append_only__cluster_9392920__proc_3.tar.gz",
        "u_label": "weak U",
        "lambda_label": "strong e-ph",
    },
    {
        "id": "intermediate_strong",
        "proc": 4,
        "archive": "core__intermediate_strong__nph7__ra_macro_append_only__cluster_9392920__proc_4.tar.gz",
        "u_label": "intermediate U",
        "lambda_label": "strong e-ph",
    },
    {
        "id": "strong_strong_u8",
        "proc": 5,
        "archive": "core__strong_strong_u8__nph7__ra_macro_append_only__cluster_9392920__proc_5.tar.gz",
        "u_label": "strong U",
        "lambda_label": "strong e-ph",
    },
)

CLASS_NAMES = {
    "hva_layer": "HVA layer",
    "hh_termwise_unit": "HVA unit term",
    "hh_termwise_quadrature": "HVA quadrature",
    "hh_hamiltonian_block": "Hamiltonian block",
    "hh_fermionic_reusable": "correlated fermionic",
    "hh_phonon_linear": "phonon linear",
    "hh_phonon_quadratic": "phonon quadratic",
    "hh_vlf_sq": "VLF quadratic",
    "uccsd_sing": "UCCSD single",
    "uccsd_dbl": "UCCSD double",
    "uccsd_paop_product_seq_ferm": "UCCSD x PAOP (fermion step)",
    "uccsd_paop_product_seq_motif": "UCCSD x PAOP (motif step)",
    "uccsd_paop_product": "UCCSD x PAOP product",
    "paop_cloud_p": "PAOP cloud-p",
    "paop_cloud_x": "PAOP cloud-x",
    "paop_disp": "PAOP displacement",
    "paop_dbl": "PAOP doublon",
    "paop_hopdrag": "PAOP hopping drag",
    "paop_dbl_p": "PAOP doublon-p",
    "paop_dbl_x": "PAOP doublon-x",
    "paop_curdrag": "PAOP current drag",
    "paop_hop2": "PAOP second-order hop",
    "paop_other": "other PAOP",
}

BG = "#f7f6f2"
INK = "#172033"
MUTED = "#5e6878"
GRID = "#d9dee7"
ACCENT = "#d97706"


class HashingReader:
    """Update a SHA-256 digest as a binary stream is consumed."""

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        payload = self.source.read(size)
        if payload:
            self.digest.update(payload)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=REPO_ROOT
        / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_heatmap.pdf",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=REPO_ROOT
        / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_heatmap_provenance.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digested(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["sha256"] = hashlib.sha256(canonical_json_bytes(out)).hexdigest()
    return out


def _number(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def read_regime(spec: dict[str, Any]) -> dict[str, Any]:
    archive_path = SOURCE_ROOT / str(spec["archive"])
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    transitions: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    problem: dict[str, Any] = {}
    route: dict[str, Any] = {}
    selected_fields = {
        "controller_round",
        "energy_after",
        "energy_before",
        "generator_id",
        "selected_operator",
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(RESULT_MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError(f"missing {RESULT_MEMBER} in {archive_path}")
        hashing_stream = HashingReader(stream)
        for prefix, event, value in ijson.parse(hashing_stream):
            if event in {"string", "number", "boolean", "null"}:
                transition_prefix = "run.accepted_transitions.item."
                if prefix.startswith(transition_prefix):
                    field = prefix[len(transition_prefix) :]
                    if field in selected_fields:
                        current[field] = value
                elif prefix.startswith("run.problem."):
                    field = prefix.removeprefix("run.problem.")
                    if "." not in field:
                        problem[field] = value
                elif prefix.startswith("run.route."):
                    field = prefix.removeprefix("run.route.")
                    if "." not in field:
                        route[field] = value
            elif prefix == "run.accepted_transitions.item" and event == "end_map":
                missing = selected_fields.difference(current)
                if missing:
                    raise ValueError(
                        f"{spec['id']} transition is missing {sorted(missing)}"
                    )
                transitions.append(dict(current))
                current.clear()
        result_sha256 = hashing_stream.digest.hexdigest()

    if len(transitions) != EXPECTED_ROUNDS:
        raise ValueError(
            f"{spec['id']} has {len(transitions)} accepted rounds, "
            f"expected {EXPECTED_ROUNDS}"
        )
    rounds = [int(row["controller_round"]) for row in transitions]
    if rounds != list(range(1, EXPECTED_ROUNDS + 1)):
        raise ValueError(f"{spec['id']} accepted-round order drifted")

    class_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "raw_drop": 0.0, "labels": set()}
    )
    raw_total = 0.0
    for row in transitions:
        before = _number(row["energy_before"], label="energy_before")
        after = _number(row["energy_after"], label="energy_after")
        signed_drop = before - after
        if signed_drop < -1e-9:
            raise ValueError(
                f"{spec['id']} round {row['controller_round']} worsened by "
                f"{-signed_drop:.3e}"
            )
        accepted_drop = max(0.0, signed_drop)
        label = str(row["selected_operator"])
        class_key = classify_hh_full_meta_label(label)
        if class_key is None:
            raise ValueError(
                f"{spec['id']} selected unclassified macro generator {label!r}"
            )
        class_row = class_rows[class_key]
        class_row["count"] += 1
        class_row["raw_drop"] += accepted_drop
        class_row["labels"].add(label)
        raw_total += accepted_drop

    t = _number(problem.get("t"), label="t")
    omega0 = _number(problem.get("omega0"), label="omega0")
    g_ep = _number(problem.get("g_ep"), label="g_ep")
    u = _number(problem.get("u"), label="u")
    lambda_ep = 2.0 * g_ep**2 / (t * omega0)
    return {
        **spec,
        "archive_path": archive_path,
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "result_member": RESULT_MEMBER,
        "result_sha256": result_sha256,
        "result_size_bytes": member.size,
        "problem": {
            "u_over_t": u / t,
            "lambda": lambda_ep,
            "g_over_omega0": g_ep / omega0,
            "n_ph_max": int(problem["n_ph_max"]),
        },
        "route": route,
        "accepted_rounds": len(transitions),
        "raw_total": raw_total,
        "classes": {
            key: {
                "count": int(value["count"]),
                "raw_drop": float(value["raw_drop"]),
                "labels": sorted(value["labels"]),
            }
            for key, value in sorted(class_rows.items())
        },
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
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


def class_display_name(class_key: str) -> str:
    return CLASS_NAMES.get(class_key, class_key.replace("_", " "))


def build_figure(rows: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    classes = sorted(
        {key for row in rows for key in row["classes"]},
        key=lambda key: -sum(
            float(row["classes"].get(key, {}).get("raw_drop", 0.0))
            for row in rows
        ),
    )
    matrix = np.zeros((len(classes), len(rows)), dtype=float)
    counts = np.zeros_like(matrix, dtype=int)
    for col, row in enumerate(rows):
        for row_index, class_key in enumerate(classes):
            entry = row["classes"].get(class_key)
            if entry is None:
                continue
            matrix[row_index, col] = float(entry["raw_drop"])
            counts[row_index, col] = int(entry["count"])

    positive = matrix[matrix > DISPLAY_FLOOR]
    if positive.size == 0:
        raise ValueError("no accepted energy drops exceed the display floor")
    vmin = float(positive.min())
    vmax = float(positive.max())

    configure_matplotlib()
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.07,
        0.955,
        "Macro-generator accepted energy drop across interaction regimes",
        fontsize=15.5,
        weight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.07,
        0.92,
        f"{ROUTE_LABEL}; each accepted macro occurrence counted once through k = {EXPECTED_ROUNDS}.",
        fontsize=9.2,
        color=MUTED,
        va="top",
    )
    fig.text(
        0.93,
        0.89,
        "PAPER I DIAGNOSTIC",
        fontsize=7.5,
        weight="bold",
        color=MUTED,
        ha="right",
        va="bottom",
    )

    ax = fig.add_axes([0.22, 0.22, 0.67, 0.61])
    masked = np.ma.masked_where(matrix <= DISPLAY_FLOOR, matrix)
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#eceef2")
    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([class_display_name(key) for key in classes], fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [
            f"{row['u_label']}\nU/t={row['problem']['u_over_t']:g}"
            for row in rows
        ],
        fontsize=7.7,
    )
    ax.tick_params(length=0)
    ax.set_xlabel("interaction regime (electronic strength within each electron-phonon block)")
    ax.set_ylabel("macro-generator class")
    ax.axvline(2.5, color="white", linewidth=4)
    ax.text(
        1,
        1.055,
        f"weak electron-phonon coupling  (lambda={rows[0]['problem']['lambda']:.2f})",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.4,
        weight="bold",
        color=INK,
    )
    ax.text(
        4,
        1.055,
        f"strong electron-phonon coupling  (lambda={rows[3]['problem']['lambda']:.2f})",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8.4,
        weight="bold",
        color=INK,
    )

    log_mid = (math.log10(vmin) + math.log10(vmax)) / 2.0
    for row_index in range(len(classes)):
        for col in range(len(rows)):
            value = matrix[row_index, col]
            count = counts[row_index, col]
            if value <= 0:
                label = "-"
                color = "#9aa2ae"
            elif value <= DISPLAY_FLOOR:
                label = f"<1e-12\n({count}x)"
                color = "#7b8491"
            else:
                label = f"{value:.1e}\n({count}x)"
                color = "white" if math.log10(value) < log_mid else "#0b1324"
            ax.text(
                col,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
                weight="bold" if value > 0 else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)

    cax = fig.add_axes([0.905, 0.22, 0.018, 0.61])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("raw cumulative accepted drop (log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    definition = (
        r"Cell value:  $R_{t,r}=\sum_{k\in(t,r)} "
        r"\max(0,E_{\mathrm{before},k}-E_{\mathrm{after},k})$."
    )
    fig.text(0.07, 0.145, definition, fontsize=8.2, color=INK, va="top")
    fig.text(
        0.07,
        0.118,
        "Parentheses give accepted-occurrence counts. Gray cells mean no admission, or an admitted "
        "class whose total drop is at or below the 1e-12 display floor.",
        fontsize=7.4,
        color=MUTED,
        va="top",
    )
    caveat = (
        "Interpretation: this is path-dependent admission credit, not final-ansatz causal attribution. "
        "Large early electronic drops can dominate the raw scale; the logarithmic color normalization "
        "retains smaller phononic and dressed-response contributions without changing their values."
    )
    fig.text(
        0.07,
        0.085,
        textwrap.fill(caveat, width=145),
        fontsize=7.0,
        color=MUTED,
        va="top",
        linespacing=1.18,
    )
    fig.text(
        0.07,
        0.027,
        f"Source: CHTC cluster {CLUSTER_ID}, six stationary-core macro append-only trajectories; "
        "full archive and result-member hashes are recorded in the companion provenance JSON.",
        fontsize=6.8,
        color=MUTED,
        va="bottom",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches=None)
    plt.close(fig)
    return {
        "classes": classes,
        "matrix": matrix.tolist(),
        "counts": counts.tolist(),
        "color_scale": {
            "kind": "log",
            "vmin": vmin,
            "vmax": vmax,
            "display_floor": DISPLAY_FLOOR,
        },
    }


def main() -> None:
    args = parse_args()
    rows = [read_regime(dict(spec)) for spec in REGIMES]
    figure = build_figure(rows, args.output_pdf)
    provenance = digested(
        {
            "schema": "paper_i_ra_macro_generator_regime_heatmap_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_kind": "diagnostic",
            "paper_promotion_status": "not_manuscript_evidence",
            "route": ROUTE_LABEL,
            "cluster_id": CLUSTER_ID,
            "accepted_rounds_per_regime": EXPECTED_ROUNDS,
            "definition": {
                "raw_drop": "max(0, energy_before - energy_after) for each accepted macro occurrence",
                "aggregation": "sum raw_drop by HH full-meta generator class and interaction regime",
                "classification": "classify_hh_full_meta_label from pipelines.contracts.static_provenance",
                "interpretation": "path-dependent admission credit; not terminal leave-one-out attribution",
                "display_floor": "raw values at or below 1e-12 are retained in provenance but shown as below-floor cells",
            },
            "regimes": [
                {
                    "id": row["id"],
                    "proc": row["proc"],
                    "problem": row["problem"],
                    "accepted_rounds": row["accepted_rounds"],
                    "raw_total": row["raw_total"],
                    "classes": row["classes"],
                    "source": {
                        "archive_path": str(row["archive_path"].relative_to(REPO_ROOT)),
                        "archive_sha256": row["archive_sha256"],
                        "archive_size_bytes": row["archive_size_bytes"],
                        "result_member": row["result_member"],
                        "result_sha256": row["result_sha256"],
                        "result_size_bytes": row["result_size_bytes"],
                    },
                }
                for row in rows
            ],
            "figure": figure,
            "output_pdf": str(args.output_pdf.relative_to(REPO_ROOT)),
        }
    )
    args.provenance_json.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_json.write_bytes(canonical_json_bytes(provenance) + b"\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
