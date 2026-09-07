"""Generate Paper-II manuscript assets for the realized-Q compiled route.

The input is the provenance-bound selector comparison JSON together with its
fresh terminal-cost sidecars.  This producer emits reader-facing LaTeX tables,
one representative trajectory figure, and a digest manifest.  It does not
modify manuscript sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


Q_ROUTE = "paper_ii_exchange_q_compiled_v1"
DRIVE_ORDER = {
    "weakslow": 0,
    "fastweak": 1,
    "midres": 2,
    "slowstrong": 3,
    "strongfast": 4,
    "fastfast": 5,
}
CONTROLLER_ORDER = {
    "delta_theta_5e-5": 0,
    "state_motion_1e-3_plus_parameter_5e-5": 1,
    "state_motion_5e-5": 2,
    "state_motion_5e-5_sub15": 3,
}
# Drives are named by their amplitude and frequency, not by internal run
# labels: the historical labels order the two words inconsistently and two of
# them do not decompose into amplitude and frequency at all.
DRIVE_TEX = {
    "weakslow": r"$(0.3,\,1)$",
    "fastweak": r"$(0.6,\,3)$",
    "midres": r"$(1.2,\,2)$",
    "slowstrong": r"$(1.2,\,1)$",
    "strongfast": r"$(2.4,\,3)$",
    "fastfast": r"$(0.6,\,6)$",
}
CONTROLLER_TEX = {
    "delta_theta_5e-5": r"$\delta\theta$",
    "state_motion_1e-3_plus_parameter_5e-5": "combined",
    "state_motion_5e-5": "state motion",
    "state_motion_5e-5_sub15": "state motion (15 subdiv.)",
}
# Controllers withheld from the main results (user direction 2026-09-04).  The
# combined cap mixes a state-motion and a raw-parameter bound in one arm, so it
# is neither of the two constructions the paper contrasts.
DEFAULT_EXCLUDED_CONTROLLERS = ("state_motion_1e-3_plus_parameter_5e-5",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sci(value: float, digits: int = 2) -> str:
    if value == 0.0:
        return "$0$"
    mantissa, exponent = f"{float(value):.{digits}e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def _integer(value: int) -> str:
    return f"${int(value):,}$".replace(",", "{,}")


def _tuple_tex(values: Sequence[int]) -> str:
    return "$({})$".format(",".join(str(int(value)) for value in values))


_PAULI_ROW = re.compile(r"::r\d+::[a-z]+$")
_INSERTION_OCCURRENCE = re.compile(r"::ins[^:]*")


def _generator_id(label: str) -> str:
    """Strip Pauli-row and insertion-occurrence suffixes to name the generator.

    The selector caps insertions and deletions per time point by generator, but
    one generator expands into several Pauli coordinates, so counting support
    labels overstates the number of structural actions.  Everything reported in
    the support-motion table is counted here at generator granularity.
    """

    text = _INSERTION_OCCURRENCE.sub("", str(label))
    while _PAULI_ROW.search(text):
        text = _PAULI_ROW.sub("", text)
    return text


def _generator_motion(route: Mapping[str, Any]) -> dict[str, int]:
    """Insertion and deletion history of every generator over the trajectory.

    ``inserted``/``deleted`` count structural events.  ``reinserted`` counts a
    deletion followed at any later time point by a re-insertion of the same
    generator; ``retired`` counts an insertion followed at any later time point
    by its deletion.  Both look over the whole trajectory rather than a fixed
    window.
    """

    history: dict[str, list[str]] = {}
    exchanges = 0
    current: Counter[str] | None = None
    for source in route.get("run_sources", []):
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            raise ValueError(f"trajectory artifact missing: {path}")
        payload = json.loads(path.read_text())
        # Committed exchange patches (D and A both nonempty), summed over every
        # selector round of every time point; a support diff between reported
        # points cannot see a delete-only round followed by an insert-only one.
        exchanges += sum(
            int(row.get("patch_exchange_commit_count") or 0) for row in payload.get("plot_rows", [])
        )
        for point in payload["trajectory"]["points"]:
            labels = ((point.get("fixed_step") or {}).get("support_labels")) or ()
            if not labels:
                continue
            support = Counter(_generator_id(value) for value in labels)
            if current is None:
                current = support
                continue
            added, removed = support - current, current - support
            for name in added:
                history.setdefault(name, []).append("+")
            for name in removed:
                history.setdefault(name, []).append("-")
            current = support
    inserted = sum(events.count("+") for events in history.values())
    deleted = sum(events.count("-") for events in history.values())
    reinserted = sum(
        1
        for events in history.values()
        for i in range(len(events) - 1)
        if events[i] == "-" and events[i + 1] == "+"
    )
    retired = sum(
        1
        for events in history.values()
        for i in range(len(events) - 1)
        if events[i] == "+" and events[i + 1] == "-"
    )
    return {
        "generators": len(history),
        "inserted": inserted,
        "deleted": deleted,
        "reinserted": reinserted,
        "retired": retired,
        "exchanges": exchanges,
    }


def _geometry_entries(support: int) -> int:
    """Entries in one full McLachlan geometry at support size ``support``."""

    return support * (support + 1) // 2 + support + 2


def _recovered_estimator_total(route: Mapping[str, Any]) -> int | None:
    """Total estimator count for an arm whose reported total is an underestimate.

    AVQDS materializes every append it tries and takes a full geometry one
    coordinate wider, so its candidate cost is the sum over time points of the
    number of appends tried there times the geometry cost at the current
    support plus one.  An analysis that knows only the total number of tried
    appends can bound that sum below but not evaluate it.  Both factors are
    recorded at every time point of the trajectory, so the sum is recovered
    here and the recovered count is checked against the recorded total before
    it is used.
    """

    work = route["estimator_work"]
    recorded = work.get("candidate_full_geometry_evaluation_count")
    if recorded is None:
        return None
    # Appendix identity: every greedy round scores the whole pool at the
    # support it starts from and takes a geometry one coordinate wider; each
    # round appends exactly one coordinate, so the first round of a time
    # point starts at the recorded post-commit support minus the number of
    # rounds.  This reproduces every persisted AVQDS receipt bitwise.
    tried = 0
    candidate = 0
    for source in route.get("run_sources", []):
        path = Path(str(source.get("path", "")))
        if not path.is_file():
            return None
        for row in json.loads(path.read_text())["plot_rows"]:
            scored = int(row.get("patch_scored_count") or 0)
            pool = int(row.get("patch_candidate_count") or 0)
            if not scored or not pool:
                continue
            rounds = scored // pool
            base = int(row["logical_parameter_count"]) - rounds
            for r in range(rounds):
                tried += pool
                candidate += pool * _geometry_entries(base + r + 1)
    if tried != int(recorded):
        return None
    return (
        int(work["logical_integration_geometry_work"])
        + int(work.get("logical_checkpoint_base_geometry_work") or 0)
        + candidate
        + int(work.get("logical_certification_geometry_work") or 0)
        + int(work.get("logical_refit_estimator_work") or 0)
    )


WORK_UNIT = 10 ** 8
STEP_UNIT = 10 ** 3
ERROR_UNIT = 10 ** -4


def _scaled(value: float, unit: float) -> str:
    """Three significant figures in the table's declared unit."""

    scaled = float(value) / unit
    if scaled >= 100.0:
        return f"{scaled:.0f}"
    if scaled >= 10.0:
        return f"{scaled:.1f}"
    return f"{scaled:.2f}"


def _cost_record(root: Path, cell_id: str, route: str) -> tuple[dict[str, int], Path]:
    path = root / cell_id / f"{route}.json"
    payload = _load_json(path)
    result = payload.get("compile_result")
    if not isinstance(result, Mapping):
        raise ValueError(f"missing compile_result in {path}")
    record = {name: int(result[name]) for name in ("N2q", "D2q", "Dc")}
    if any(value < 0 for value in record.values()):
        raise ValueError(f"negative terminal compiled cost in {path}")
    return record, path


def _q_cells(
    comparison: Mapping[str, Any],
    terminal_cost_root: Path,
    excluded_controllers: Sequence[str] = (),
    supplement: Mapping[str, Any] | None = None,
    supplement_cost_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Assemble the reported cells: one APM arm and one AVQDS comparator each.

    This comparison is the only source.  A cell is reported when it carries
    both arms with an exact estimator receipt; a cell whose rerun has not
    returned is omitted and listed in the manifest rather than back-filled from
    a legacy record.
    """

    excluded = set(excluded_controllers)
    spare = {str(c["cell_id"]): c for c in (supplement or {}).get("cells", [])}
    cells: list[dict[str, Any]] = []
    omitted: list[dict[str, str]] = []
    for raw_cell in comparison.get("cells", []):
        cell_id = str(raw_cell["cell_id"])
        if str(raw_cell.get("controller")) in excluded:
            continue
        routes = raw_cell.get("routes", {})
        arms: dict[str, Any] = {}
        totals: dict[str, int] = {}
        roots: dict[str, Path] = {}
        supplemented: list[str] = []
        missing: list[str] = []
        for route in ("q_compiled", "avqds"):
            payload = routes.get(route)
            root = terminal_cost_root
            if payload is None or payload.get("accuracy") is None:
                payload = spare.get(cell_id, {}).get("routes", {}).get(route)
                root = supplement_cost_root
                if payload is None or payload.get("accuracy") is None or root is None:
                    missing.append(f"{route} trajectory")
                    continue
                supplemented.append(route)
            work = payload["estimator_work"]
            if work.get("logical_total_estimator_work_status") == "exact":
                total = int(work["logical_total_estimator_work"])
            else:
                recovered = _recovered_estimator_total(payload)
                if recovered is None:
                    missing.append(f"{route} estimator total")
                    continue
                total = recovered
            if not (root / cell_id / f"{route}.json").is_file():
                missing.append(f"{route} terminal compiled cost")
                continue
            arms[route], totals[route], roots[route] = payload, total, root
        if missing:
            omitted.append({"cell_id": cell_id, "missing": ", ".join(missing)})
            continue
        q = arms["q_compiled"]
        if q.get("route_bundle") != Q_ROUTE:
            raise ValueError(f"unexpected Q route in {cell_id}")
        if q.get("trajectory_kind") != "direct_t0_t20":
            raise ValueError(f"non-final trajectory in {cell_id}")
        q_cost, q_cost_path = _cost_record(roots["q_compiled"], cell_id, "q_compiled")
        av_cost, av_cost_path = _cost_record(roots["avqds"], cell_id, "avqds")
        cells.append(
            {
                "cell_id": cell_id,
                "drive": str(raw_cell["drive"]),
                "controller": str(raw_cell["controller"]),
                "apm_cut": float(raw_cell["apm_cut"]),
                "q": q,
                "avqds": arms["avqds"],
                "q_cost": q_cost,
                "avqds_cost": av_cost,
                "q_cost_path": q_cost_path,
                "avqds_cost_path": av_cost_path,
                "motion": _generator_motion(arms["q_compiled"]),
                "q_work": totals["q_compiled"],
                "avqds_work": totals["avqds"],
                "supplemented_arms": supplemented,
            }
        )
    cells.sort(
        key=lambda cell: (DRIVE_ORDER[cell["drive"]], CONTROLLER_ORDER[cell["controller"]])
    )
    if not cells:
        raise ValueError("comparison contains no reportable APM cells")
    return cells, omitted


def _table_tex(cells, comparison_sha: str, pending: Sequence[tuple[str, str]] = ()) -> str:
    r"""Emit the main frontier table and the support-motion table.

    The frontier table carries one row per (cell, method) so that the
    Paper-I resource quantities -- $N_{2q}$, $D_{2q}$, $D_c$, $\mathcal E$ --
    are printed per arm rather than folded into a tuple, with the internal
    time-step count alongside them.  Trajectory accuracy is reported as the
    three physical observables (total energy, doublon occupation, charge
    imbalance); infidelity is not printed.
    """

    lines = [
        "% Generated by q_compiled_manuscript_assets.py.",
        f"% comparison_sha256={comparison_sha}",
        r"\newcommand{\PaperIIFrontierTable}{%",
        r"\begin{table*}[t]",
        r"\caption{Completed $t=20$ AP-McLachlan (APM) trajectories and their "
        r"locked AVQDS comparators.  Drives are identified by their amplitude "
        r"and frequency $(A,\omega)$.  $\overline{|\Delta E|}$, $\overline{|\Delta D|}$ and "
        r"$\overline{|\Delta n|}$ are mean absolute errors in total energy, "
        r"doublon occupation and charge imbalance against the independent exact "
        r"reference, averaged over all 501 reporting times.  $N_\theta$ is the "
        r"terminal active-coordinate count, $(N_{2q},D_{2q},D_c)$ the terminal "
        r"FakeMarrakesh compiled two-qubit count, two-qubit depth and total "
        r"depth, $\mathcal E$ the total logical scalar-estimator count "
        r"(Appendix~\ref{app:estimator_query_accounting}), and $N_{\rm step}$ "
        r"the internal integration step count.  Where an AVQDS comparator "
        r"predates the persisted estimator ledger, its $\mathcal E$ is the "
        r"Appendix identity evaluated round by round from the recorded pool "
        r"size and support at every time point; that reconstruction reproduces "
        r"every persisted AVQDS ledger exactly.\label{tab:t20_q_compiled_frontier}}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"$(A,\omega)$ & controller & method & $\overline{|\Delta E|}$ & "
        r"$\overline{|\Delta D|}$ & $\overline{|\Delta n|}$ & $N_\theta$ & "
        r"$N_{2q}$ & $D_{2q}$ & $D_c$ & $\mathcal E$ & $N_{\rm step}$ \\",
        r"\midrule",
    ]
    for index, cell in enumerate(cells):
        if index:
            lines.append(r"\addlinespace[1.5pt]")
        for position, (label, route, cost, work) in enumerate(
            (
                ("APM", cell["q"], cell["q_cost"], cell["q_work"]),
                ("AVQDS", cell["avqds"], cell["avqds_cost"], cell["avqds_work"]),
            )
        ):
            accuracy = route["accuracy"]
            head = (
                [DRIVE_TEX[cell["drive"]], CONTROLLER_TEX[cell["controller"]]]
                if position == 0
                else ["", ""]
            )
            lines.append(
                " & ".join(
                    head
                    + [
                        label,
                        _scaled(accuracy["mean_abs_energy_error"], ERROR_UNIT),
                        _scaled(accuracy["mean_abs_doublon_error"], ERROR_UNIT),
                        _scaled(accuracy["mean_abs_charge_imbalance_error"], ERROR_UNIT),
                        str(int(route["n_theta"])),
                        str(int(cost["N2q"])),
                        str(int(cost["D2q"])),
                        str(int(cost["Dc"])),
                        _scaled(work, WORK_UNIT),
                        _scaled(int(route["time_step_count"]), STEP_UNIT),
                    ]
                )
                + r" \\"
            )
    for drive, controller in pending:
        lines.append(r"\addlinespace[1.5pt]")
        lines.append(
            " & ".join(
                [DRIVE_TEX[drive], CONTROLLER_TEX[controller], "pending"]
                + ["--"] * 9
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            r"}",
            "",
            r"\newcommand{\PaperIISupportMotionTable}{%",
            r"\begin{table*}[t]",
            r"\caption{Structural actions over each reported APM trajectory, counted at "
            r"generator granularity: the selector caps insertions and deletions "
            r"per time point by generator, and one generator expands into "
            r"several Pauli coordinates.  $\mathcal E$ repeats the total "
            r"scalar-estimator count of Table~\ref{tab:t20_q_compiled_frontier} "
            r"in units of $10^{8}$.  \emph{Inserted} and \emph{deleted} count "
            r"structural events; \emph{exch.} counts committed exchange patches, "
            r"$D\ne\varnothing$ and $A\ne\varnothing$, over every selector "
            r"round of every time point.  \emph{del.\ then re-ins.} "
            r"counts a generator deleted and then re-inserted at any later time "
            r"point, and \emph{ins.\ then del.} a generator inserted and then "
            r"deleted at any later time point; both range over the whole "
            r"trajectory.  \emph{finalists tested/failed} gives deletion-bearing "
            r"finalists materialized and sent to the hard gates, over those "
            r"failing a gate after materialization; a $\geq$ prefix marks a "
            r"trajectory whose attempt record keeps only the final selector round "
            r"at time points with several commits, so both counts are lower "
            r"bounds there.}",
            r"\label{tab:t20_q_compiled_support_motion}",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{llrrrrrrc}",
            r"\toprule",
            r"$(A,\omega)$ & ctrl. & $\mathcal E$ & ins. & del. & exch. & "
            r"\shortstack{del.\ then\\re-ins.} & \shortstack{ins.\ then\\del.} & "
            r"\shortstack{finalists\\tested/failed} \\",
            r"\midrule",
        ]
    )
    for cell in cells:
        motion = cell["motion"]
        gates = cell["q"]["deletion_gate_outcomes"]
        lines.append(
            " & ".join(
                [
                    DRIVE_TEX[cell["drive"]],
                    CONTROLLER_TEX[cell["controller"]],
                    _scaled(cell["q_work"], WORK_UNIT),
                    str(motion["inserted"]),
                    str(motion["deleted"]),
                    str(motion["exchanges"]),
                    str(motion["reinserted"]),
                    str(motion["retired"]),
                    ("" if gates.get("attempt_record_complete", True) else r"$\geq$")
                    + f"{int(gates['deletion_bearing_finalist_attempt_count'])}/{int(gates['post_delete_gate_failure_count'])}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", r"}", ""])
    return "\n".join(lines)


def _trajectory_rows(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    stitched: list[Mapping[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        rows = payload.get("plot_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing plot rows in {path}")
        for row in rows:
            if stitched and float(row["time"]) == float(stitched[-1]["time"]):
                continue
            if stitched and float(row["time"]) < float(stitched[-1]["time"]):
                raise ValueError(f"nonmonotone stitched trajectory at {path}")
            stitched.append(row)
    if len(stitched) != 501:
        raise ValueError(f"expected 501 stitched plot rows, found {len(stitched)}")
    return stitched


def _representative_figure(
    comparison: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    output: Path,
    cell_id: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cell = next((item for item in cells if item["cell_id"] == cell_id), None)
    if cell is None:
        raise ValueError(f"representative cell is unavailable: {cell_id}")
    q_paths = [Path(source["path"]) for source in cell["q"]["run_sources"]]
    av_paths = [Path(source["path"]) for source in cell["avqds"]["run_sources"]]
    q_rows = _trajectory_rows(q_paths)
    av_rows = _trajectory_rows(av_paths)

    raw_cell = next(item for item in comparison["cells"] if item["cell_id"] == cell_id)
    reference_manifest_path = Path(raw_cell["reference"]["manifest_path"])
    reference_manifest = _load_json(reference_manifest_path)
    reference_path = reference_manifest_path.parent / str(reference_manifest["data_file"])
    with np.load(reference_path) as ref:
        t_ref = np.asarray(ref["time"], dtype=float)
        ref_series = (
            np.asarray(ref["energy"], dtype=float),
            np.asarray(ref["doublon"], dtype=float),
            np.asarray(ref["site_occupations"], dtype=float)[:, 0]
            - np.asarray(ref["site_occupations"], dtype=float)[:, 1],
        )

    def series(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, tuple[Any, Any, Any]]:
        time = np.asarray([row["time"] for row in rows], dtype=float)
        energy = np.asarray([row["energy_expectation"] for row in rows], dtype=float)
        doublon = np.asarray([row["doublon"] for row in rows], dtype=float)
        delta_n = np.asarray(
            [row["site_occupations"][0] - row["site_occupations"][1] for row in rows],
            dtype=float,
        )
        return time, (energy, doublon, delta_n)

    t_q, q_series = series(q_rows)
    t_av, av_series = series(av_rows)
    if not (
        np.allclose(t_q, t_av, rtol=0.0, atol=1.0e-12)
        and np.allclose(t_q, t_ref, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("representative trajectories do not share the reporting grid")

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True)
    ylabels = (r"$E(t)$", r"$D(t)$", r"$\Delta n(t)$")
    panel_labels = ("(a)", "(b)", "(c)")
    for axis, q_values, av_values, exact_values, ylabel, panel_label in zip(
        axes, q_series, av_series, ref_series, ylabels, panel_labels
    ):
        axis.plot(
            t_q,
            q_values,
            color="#8b1a1a",
            linewidth=1.35,
            label=r"APM (realized $\Gamma$)",
        )
        axis.plot(t_av, av_values, color="#1f5aa6", linewidth=1.15, linestyle="--", label="AVQDS")
        axis.plot(t_ref, exact_values, color="black", linewidth=1.0, linestyle=":", label="exact")
        axis.set_ylabel(ylabel)
        axis.set_xlabel(r"$t$")
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.text(
            0.03,
            0.95,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=3,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88), pad=0.55, w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "cell_id": cell_id,
        "q_trajectories": [
            {"path": str(path), "sha256": _sha256(path)} for path in q_paths
        ],
        "avqds_trajectories": [
            {"path": str(path), "sha256": _sha256(path)} for path in av_paths
        ],
        "reference_manifest": {
            "path": str(reference_manifest_path),
            "sha256": _sha256(reference_manifest_path),
        },
        "reference_data": {"path": str(reference_path), "sha256": _sha256(reference_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--terminal-cost-root", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--representative-cell",
        default="strongfast__delta_theta_5e-5",
    )
    parser.add_argument(
        "--pending-cell",
        action="append",
        default=None,
        metavar="DRIVE:CONTROLLER",
        help="drive:controller shown as a pending row; the cell has no trajectory",
    )
    parser.add_argument(
        "--supplement-comparison-json",
        type=Path,
        default=None,
        help="comparison supplying arms still pending in the primary campaign",
    )
    parser.add_argument("--supplement-terminal-cost-root", type=Path, default=None)
    parser.add_argument(
        "--exclude-controller",
        action="append",
        default=None,
        help=(
            "controller id withheld from the main results; repeatable. "
            "Defaults to the combined state-motion-plus-parameter cap."
        ),
    )
    parser.add_argument("--adoption-note", required=True)
    args = parser.parse_args()

    comparison = _load_json(args.comparison_json)
    if comparison.get("schema") not in {
        "paper_ii_selector_comparison_v2",
        "paper_ii_selector_comparison_v3",
    }:
        raise ValueError("unexpected comparison schema")
    excluded = (
        tuple(args.exclude_controller)
        if args.exclude_controller is not None
        else DEFAULT_EXCLUDED_CONTROLLERS
    )
    unknown = sorted(set(excluded) - set(CONTROLLER_ORDER))
    if unknown:
        raise ValueError(f"unknown controller id(s) excluded: {unknown}")
    pending = tuple(
        tuple(entry.split(":", 1)) for entry in (args.pending_cell or ())
    )
    for drive, controller in pending:
        if drive not in DRIVE_TEX or controller not in CONTROLLER_ORDER:
            raise ValueError(f"unknown pending cell {drive}:{controller}")
    supplement = (
        _load_json(args.supplement_comparison_json)
        if args.supplement_comparison_json is not None
        else None
    )
    cells, omitted = _q_cells(
        comparison,
        args.terminal_cost_root,
        excluded,
        supplement,
        args.supplement_terminal_cost_root,
    )
    comparison_sha = _sha256(args.comparison_json)

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(_table_tex(cells, comparison_sha, pending), encoding="utf-8")
    figure_sources = _representative_figure(
        comparison,
        cells,
        args.figure,
        args.representative_cell,
    )

    manifest = {
        "schema": "paper_ii_q_compiled_manuscript_assets_v1",
        "adoption_note": str(args.adoption_note),
        "route_bundle": Q_ROUTE,
        "comparison": {
            "path": str(args.comparison_json),
            "sha256": comparison_sha,
            "generated_utc": comparison.get("generated_utc"),
            "upstream_status": comparison.get("status"),
        },
        "coverage": {
            "locked_cells": int(comparison["summary"]["locked_cell_count"]),
            "q_compiled_cells": len(cells),
            "cell_ids": [cell["cell_id"] for cell in cells],
            "omitted_cells": omitted,
            "pending_cells": [f"{d}:{c}" for d, c in pending],
            "supplemented_arms": {
                cell["cell_id"]: cell["supplemented_arms"]
                for cell in cells
                if cell["supplemented_arms"]
            },
            "excluded_controllers": list(excluded),
        },
        "terminal_cost_inputs": [
            {
                "cell_id": cell["cell_id"],
                "q_compiled": {
                    "path": str(cell["q_cost_path"]),
                    "sha256": _sha256(cell["q_cost_path"]),
                },
                "avqds": {
                    "path": str(cell["avqds_cost_path"]),
                    "sha256": _sha256(cell["avqds_cost_path"]),
                },
            }
            for cell in cells
        ],
        "figure_sources": figure_sources,
        "outputs": {
            "table": {"path": str(args.table), "sha256": _sha256(args.table)},
            "figure": {"path": str(args.figure), "sha256": _sha256(args.figure)},
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
