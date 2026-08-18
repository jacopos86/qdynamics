#!/usr/bin/env python3
"""Paper-I-method comparison PDF: historical vs all-phase-adaptive RA routes.

Six regime convergence panels plus the cost-tuple table
(N2q, D2q, Dc, W1q, S_alg), comparing:

- plateau insertion (historical cluster 9605157, k50) vs plateau-RA
  subpolicies (plateau_append_phase0, plateau_position_phase0);
- append insertion (historical cluster 9398375, r70 clipped to k<=50) vs
  append-RA subpolicies (append_ra, append_position_phase0);
- always_open_position_phase0 overlays once cells complete.

New-route cells are read from worker archives extracted under --ra-cells-dir
(subdirs x_<execution_id_prefix>/run/summary/summary.json). Regenerate with
fresh extractions as campaign cells land; panels without RA data yet are
annotated as pending. This script lives outside the pipelines/src source
inventory on purpose: rebuilding it never perturbs sealed campaign packages.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
HIST_PLATEAU_GLOB = str(HERE / "retrieved_phase0_completed_20260809/9605157.*_completed_report_adapter.json")
# The published paper's append baseline: the six POWELL tolerance-matched
# r50 append trajectories adopted by author direction on 2026-08-16 —
# resolved through the published package's figure provenance so the series
# is exactly what Paper_I.pdf plots.
PAPER_PACKAGE_PROVENANCE = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/"
    "output/paper_packages/Paper_I_full_20260816/figures/"
    "paper_i_ra_vs_append_matched_singleton_plateau_20260812/"
    "paper_i_ra_vs_append_matched_singleton_plateau_provenance.json"
)
ERR_FLOOR = 1e-16
# Fallback when the paper-package provenance is regenerated with a schema that
# drops adopted_replacements: the tolmatch overlay provenance maps the same six
# published runs (identical summary/result payloads, receipt-pinned SHAs).
TOLMATCH_OVERLAY_PROVENANCE = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/"
    "output/pdf/paper_i_append_powell_tolmatch_overlay_20260816/"
    "paper_i_append_powell_tolmatch_overlay_20260816_provenance.json"
)
K_MAX = 50

# Style system: dashed lines are reserved for the two historical baselines.
# Every live RA series is SOLID with its own hue + marker shape, encoding
# policy x P0-population consistently on every page:
#   natural endpoint-P0  = sky blue  / circle      natural position-P0 = orange / diamond
#   forced  endpoint-P0  = vermillion/ triangle    forced  position-P0 = purple / inv-triangle
#   floors  endpoint-P0  = green     / square      floors  position-P0 = pink   / pentagon
FAMILIES = {
    # series key: (panel family, label, color, linestyle, marker)
    "hist_plateau": ("plateau", "plateau historical (fixed24)", "#3d3d3d", "--", None),
    "hist_append": ("append", "AAVQE published r50", "#0b3d91", "--", None),
    "plateau_append_phase0": ("plateau", "plateau-RA", "#56b4e9", "-", "o"),
    "plateau_position_phase0": ("plateau", "plateau-RA [pos-P0]", "#e69f00", "-", "D"),
    "append_ra": ("append", "append-RA", "#56b4e9", "-", "o"),
    "append_position_phase0": ("append", "append-RA [pos-P0] (retired)", "#e69f00", "-", "D"),
    "always_open_position_phase0": ("open", "always-open-RA", "#e69f00", "-", "D"),
    "forced_append_ra": ("append", "forced-k50 append-RA", "#d55e00", "-", "^"),
    "forced_plateau_append_phase0": ("plateau", "forced-k50 plateau-RA", "#d55e00", "-", "^"),
    "forced_plateau_position_phase0": ("plateau", "forced-k50 plateau-RA [pos-P0]", "#9467bd", "-", "v"),
    "forced_always_open_position_phase0": ("open", "forced-k50 always-open-RA", "#9467bd", "-", "v"),
    "floors_append_ra": ("append", "floors-RA append (10/7/4)", "#009e73", "-", "s"),
    "floors_plateau_append_phase0": ("plateau", "floors-RA plateau (10/7/4)", "#009e73", "-", "s"),
    "floors_plateau_position_phase0": ("plateau", "floors-RA plateau [pos-P0] (10/7/4)", "#cc79a7", "-", "p"),
    "floors_always_open_position_phase0": ("open", "floors-RA always-open (10/7/4)", "#cc79a7", "-", "p"),
}
TERSE_STATUS = {
    "append_ra": "running 9662333",
    "plateau_append_phase0": "running 9662333",
    "plateau_position_phase0": "running 9662333",
    "always_open_position_phase0": "running 9662334",
    "forced_append_ra": "running 9662370",
    "forced_plateau_append_phase0": "running 9662370",
    "forced_plateau_position_phase0": "running 9662370",
    "forced_always_open_position_phase0": "running 9662370",
}


def load_hist_plateau() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(HIST_PLATEAU_GLOB)):
        adapter = json.loads(Path(path).read_text())
        costs = adapter["terminal"]["costs"]
        rows[str(adapter["regime_id"])] = {
            "points": [(int(p["k"]), float(p["error"])) for p in adapter["points"]],
            "exact_energy": float(adapter["exact_same_cutoff_energy"]),
            "k_final": int(adapter["terminal"]["k"]),
            "err_final": float(adapter["terminal"]["error"]),
            "costs": {k: int(costs[k]) for k in ("N2q", "D2q", "Dc", "W1q", "S_alg")},
        }
    return rows


def _read_json_maybe_gz(path: Path) -> dict:
    """Read a JSON file that may have been gzipped in place (Aug 16 batch)."""

    if path.is_file():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    with gzip.open(gz, "rt") as handle:
        return json.load(handle)


def _append_salg_prefix(result_path: Path, cache_path: Path) -> dict[int, int]:
    """Cumulative charged S_alg per append round from the run ledger."""

    if cache_path.is_file():
        return {int(k): int(v) for k, v in json.loads(cache_path.read_text()).items()}
    payload = _read_json_maybe_gz(result_path)
    occurrences = payload["result_payload"]["estimator_call_ledger"]["occurrences"]
    per_round: dict[int, int] = {}
    for row in occurrences:
        if not row.get("charged", False):
            continue
        scope = str(row.get("consumer_scope", ""))
        if not scope.startswith("append_round_"):
            continue
        round_index = int(scope.split("append_round_")[1].split(":")[0])
        per_round[round_index] = per_round.get(round_index, 0) + 1
    prefix: dict[int, int] = {}
    total = 0
    for round_index in sorted(per_round):
        total += per_round[round_index]
        prefix[round_index] = total
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prefix))
    return prefix


def _hist_append_entries() -> list[tuple[str, Path, dict | None]]:
    """(regime, summary_path, result_ref) for the six published AAVQE runs."""

    provenance = json.loads(PAPER_PACKAGE_PROVENANCE.read_text())
    if "adopted_replacements" in provenance:
        return [
            (
                str(replacement["regime"]),
                Path(replacement["source"]["summary"]["path"]),
                replacement["source"].get("result"),
            )
            for replacement in provenance["adopted_replacements"]
        ]
    overlay = json.loads(TOLMATCH_OVERLAY_PROVENANCE.read_text())
    entries: list[tuple[str, Path, dict | None]] = []
    for record in overlay["regimes"]:
        regime = str(record["execution_id"]).split("__")[1]
        matched = record["local_matched_tolerance"]
        entries.append(
            (regime, Path(matched["summary"]["path"]), matched.get("result"))
        )
    return entries


def load_hist_append(exact_by_regime: dict[str, float], workdir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for regime, summary_path, result_ref in _hist_append_entries():
        summary = json.loads(summary_path.read_text())
        exact = exact_by_regime[regime]
        points = [
            (
                int(row["controller_round"]),
                max(abs(float(row["energy_after"]) - exact), ERR_FLOOR),
            )
            for row in summary["accepted_history"]
            if 1 <= int(row["controller_round"]) <= K_MAX
        ]
        terminal = summary["resources"]["terminal_compiled_resources"]
        accounting = summary["estimator_accounting"]
        salg_prefix: dict[int, int] = {}
        if isinstance(result_ref, dict) and workdir is not None:
            try:
                salg_prefix = _append_salg_prefix(
                    Path(result_ref["path"]),
                    Path(workdir) / f"append_salg_prefix_{regime}.json",
                )
            except (OSError, KeyError, ValueError, TypeError):
                salg_prefix = {}
        result_payload_words: list[str] = []
        words_cache = (
            Path(workdir) / f"hist_words_{regime}.json"
            if workdir is not None
            else None
        )
        if words_cache is not None and words_cache.is_file():
            result_payload_words = [
                str(w) for w in json.loads(words_cache.read_text())
            ]
        else:
            try:
                payload = _read_json_maybe_gz(Path(result_ref["path"]))
                result_payload_words = [
                    str(label).split("::", 1)[1]
                    for label in payload["result_payload"][
                        "accepted_operator_labels"
                    ]
                ]
                del payload
                if words_cache is not None and result_payload_words:
                    words_cache.write_text(json.dumps(result_payload_words))
            except (OSError, KeyError, ValueError, TypeError, IndexError):
                result_payload_words = []
        rows[regime] = {
            "cell_name": f"hist_append_{regime}",
            "words": result_payload_words,
            "regime": regime,
            "points": points,
            "k_final": int(summary["controller_rounds_completed"]),
            "err_final": points[-1][1] if points else float("nan"),
            "clipped": False,
            "salg_by_round": salg_prefix,
            "costs": {
                "N2q": int(terminal["compiled_count_2q_total"]),
                "D2q": int(terminal["compiled_depth_2q_total"]),
                "Dc": int(terminal["compiled_depth_total"]),
                "W1q": int(terminal["qiskit_pretranspile_pauli_1q_work_total"]),
                "S_alg": int(accounting["S_alg"]),
            },
        }
    return rows


_REGIME_QUBITS_REF: dict[str, "object"] = {}


def _ra_prefix_tuple(
    cell_name: str,
    words: list[str],
    regime: str,
    k: int,
    workdir: Path,
) -> dict[str, int] | None:
    """Table-I compile of the accepted k-prefix ansatz (cached)."""

    if not words or k < 1 or k > len(words):
        return None
    cache = workdir / f"ra_tuple_{cell_name}_k{k}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    try:
        import numpy as np
        from pipelines.exact_bench.table_i_qiskit_resource_compile import (
            compile_table_i_pauli_label_groups,
        )
        from pipelines.reporting.paper_i_qiskit_cost_tuple import (
            qiskit_cost_fields,
        )
        from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
            build_paper_i_ra_hh_regime_problem,
        )

        if regime not in _REGIME_QUBITS_REF:
            problem = build_paper_i_ra_hh_regime_problem(regime)
            _REGIME_QUBITS_REF[regime] = np.asarray(
                problem.reference_state.build_state(), dtype=complex
            ).reshape(-1)
        reference = _REGIME_QUBITS_REF[regime]
        groups = [[word] for word in words[:k]]
        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=groups,
            num_qubits=len(groups[0][0]),
            reference_state=reference,
            source_kind="paper_i_ra_allphase_adaptive_prefix_v1",
        )
        fields = qiskit_cost_fields(compiled)
        tuple_row = {
            key: int(fields[key]) for key in ("N2q", "D2q", "Dc", "W1q")
        }
    except Exception:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tuple_row))
    return tuple_row


def _ensure_checkpoint_words(cells_dir: Path, cell_dir: Path) -> list[str]:
    """Accepted operator Pauli words from the cell checkpoint (extracting
    it from the fetched archive on first use)."""

    checkpoint = cell_dir / "run/checkpoints/current.json"
    if not checkpoint.is_file():
        import tarfile

        prefix = cell_dir.name[2:]
        matches = sorted(
            (cells_dir / "archives").glob(prefix + "__*.tar.gz")
        )
        if not matches:
            return []
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(matches[-1]) as archive:
                member = archive.extractfile("./run/checkpoints/current.json")
                if member is None:
                    return []
                checkpoint.write_bytes(member.read())
        except (OSError, KeyError, tarfile.TarError):
            return []
    try:
        payload = json.loads(checkpoint.read_text())
        return [
            str(op).split("::", 1)[1]
            for op in payload["adapt_vqe"]["operators"]
        ]
    except (OSError, KeyError, ValueError, IndexError):
        return []


def load_live_cells(
    tails_dir: Path,
    exact_by_regime: dict[str, float],
) -> dict[tuple[str, str], dict]:
    """Partial trajectories of RUNNING cells, parsed line-by-line from
    condor_tail AI_LOG dumps (<exec_id>.tail). Memory-safe: never loads
    checkpoints; each tail is a few MB of text."""

    rows: dict[tuple[str, str], dict] = {}
    if tails_dir is None or not tails_dir.is_dir():
        return rows
    import re as _re

    iter_re = _re.compile(
        r'"best_op": "[^:"]+::([a-z]+)".*?"depth": (\d+).*?"energy": '
        r"(-?[0-9.eE+-]+)"
    )
    for tail in sorted(tails_dir.glob("allphase_maxk50__*.tail")):
        name = tail.name[len("allphase_maxk50__"):-len(".tail")]
        name = name.rsplit("__nph", 1)[0]
        arm, regime = None, None
        for reg in REGIMES:
            if name.endswith("__" + reg):
                regime = reg
                arm = name[: -(len(reg) + 2)]
        if arm is None or arm.startswith("hist_") or regime not in exact_by_regime:
            continue
        exact = exact_by_regime[regime]
        words: list[str] = []
        points: list[tuple[int, float]] = []
        with tail.open() as handle:
            for line in handle:
                if '"hardcoded_adapt_iter"' not in line:
                    continue
                match = iter_re.search(line)
                if not match:
                    continue
                word, depth, energy = match.groups()
                k = int(depth)
                if 1 <= k <= K_MAX:
                    words.append(word)
                    points.append(
                        (k, max(abs(float(energy) - exact), ERR_FLOOR))
                    )
        if not points:
            continue
        if points[0][0] != 1:
            # Truncated stream or a mid-run continuation: the word list does
            # not start at round 1, so prefix tuples cannot be compiled from
            # it. The trajectory itself is still valid (absolute k).
            words = []
        rows[(arm, regime)] = {
            "cell_name": f"live_{arm}_{regime}",
            "words": words,
            "regime": regime,
            "points": points,
            "k_final": points[-1][0],
            "err_final": points[-1][1],
            "clipped": False,
            "salg_by_round": {},
            "live": True,
        }
    return rows


def load_ra_cells(cells_dir: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(cells_dir.glob("x_*/run/summary/summary.json")):
        summary = json.loads(path.read_text())
        cell_name = path.parents[2].name  # x_allphase_maxk50__<arm>__<regime>
        arm = next((a for a in FAMILIES if f"__{a}__" in cell_name), None)
        regime = next((r for r in REGIMES if cell_name.endswith(r) or f"__{r}" in cell_name), None)
        trace = summary.get("accepted_error_trace") or []
        if arm is None or regime is None or not trace:
            continue
        work = summary.get("canonical_all_work") or {}
        salg_by_round: dict[int, int] = {}
        result_path = path.parents[1] / "result/result.json"
        if result_path.is_file():
            try:
                payload = json.loads(result_path.read_text())
                prefix_rows = payload["run"]["canonical_reporting"][
                    "accepted_prefix_work"
                ]
                salg_by_round = {
                    index + 1: int(row["s_alg"])
                    for index, row in enumerate(prefix_rows)
                }
            except (OSError, KeyError, ValueError, TypeError):
                salg_by_round = {}
        rows[(arm, regime)] = {
            "cell_name": cell_name,
            "words": _ensure_checkpoint_words(cells_dir, path.parents[2]),
            "regime": regime,
            "salg_by_round": salg_by_round,
            "points": [
                (int(row["controller_round"]), max(float(row["absolute_energy_error"]), ERR_FLOOR))
                for row in trace
            ],
            "k_final": int(trace[-1]["controller_round"]),
            "err_final": max(float(trace[-1]["absolute_energy_error"]), ERR_FLOOR),
            "costs": {"S_alg": int(work.get("s_alg", 0))},
            "horizon_scope": str(summary.get("horizon_scope", "")),
        }
    return rows


SHORT = {
    "hist_plateau": "plateau hist (9605157, k50)",
    "hist_append": "AAVQE (published tolmatch r50)",
    "plateau_append_phase0": "plateau-RA (append-P0)",
    "plateau_position_phase0": "plateau-RA (position-P0)",
    "append_ra": "append-RA",
    "append_position_phase0": "append-RA (position-P0, cross-check)",
    "always_open_position_phase0": "always-open-RA",
}
RUNNING_STATUS = {
    "append_ra": "running (CHTC 9662333)",
    "plateau_append_phase0": "running (CHTC 9662333)",
    "plateau_position_phase0": "running (CHTC 9662333)",
    "always_open_position_phase0": "running (CHTC 9662334)",
    "forced_append_ra": "running (CHTC 9662370)",
    "forced_plateau_append_phase0": "running (CHTC 9662370)",
    "forced_plateau_position_phase0": "running (CHTC 9662370)",
    "forced_always_open_position_phase0": "running (CHTC 9662370)",
}
SHORT.update({
    "floors_append_ra": "floors-RA append (10/7/4)",
    "floors_plateau_append_phase0": "floors-RA plateau (append-P0)",
    "floors_plateau_position_phase0": "floors-RA plateau (position-P0)",
    "floors_always_open_position_phase0": "floors-RA always-open",
    "forced_append_ra": "forced-k50 append-RA",
    "forced_plateau_append_phase0": "forced-k50 plateau-RA (append-P0)",
    "forced_plateau_position_phase0": "forced-k50 plateau-RA (position-P0)",
    "forced_always_open_position_phase0": "forced-k50 always-open-RA",
})
ARM_PAGES = (
    (
        "Arm 1 - Append: published baseline vs natural-RA vs forced-k50",
        "Append-only insertion. Baseline = published Paper-I POWELL "
        "tolerance-matched append (r50). Natural-RA stops at the "
        "authenticated Phase-III no-positive terminal (max k=50); "
        "forced-k50 force-admits through no-positive rounds to exactly "
        "k=50.",
        ("hist_append", "append_ra", "forced_append_ra", "floors_append_ra"),
        (),
    ),
    (
        "Arm 2 - Plateau (append-endpoint Phase 0): historical vs natural vs forced",
        "Plateau-commutation insertion with endpoint-generator Phase 0. "
        "Baseline = historical plateau campaign 9605157 (k50). Forced-k50 "
        "tests whether the historical deep staircase returns under the "
        "new signed Qiskit scoring.",
        ("hist_plateau", "plateau_append_phase0", "forced_plateau_append_phase0", "floors_plateau_append_phase0"),
        (),
    ),
    (
        "Arm 3 - Plateau (position-record Phase 0): historical vs natural vs forced",
        "Plateau-commutation insertion with position-record Phase 0 "
        "(representation ablation of Arm 2). Baseline = historical "
        "plateau campaign 9605157 (k50).",
        ("hist_plateau", "plateau_position_phase0", "forced_plateau_position_phase0", "floors_plateau_position_phase0"),
        (),
    ),
    (
        "Arm 4 - Always-open: natural vs forced (historical plateau context)",
        "Always-commutation-reduced insertion: every reduced position "
        "available each round. New arm with no historical counterpart; "
        "historical plateau shown for context.",
        ("hist_plateau", "always_open_position_phase0",
         "forced_always_open_position_phase0",
         "floors_always_open_position_phase0"),
        (),
    ),
)
FAMILY_PAGES = (
    (
        "Append family: published Paper-I append vs append-RA",
        "Append-only insertion: each round may only add one operator at the "
        "circuit end. Baseline = the published Paper-I POWELL tolerance-"
        "matched append trajectories (r50). RA = all-phase-adaptive "
        "shortlists, Qiskit-costed Phases I-III, authenticated natural-"
        "terminal stopping under max k=50.",
        ("hist_append", "append_ra", "forced_append_ra", "floors_append_ra"),
        ("forced-k50 append-RA (planned): Phase-III no-positive rounds "
         "force-admit the argmax signed-score candidate; every cell runs "
         "to exactly k=50.",),
    ),
    (
        "Plateau family: historical plateau vs plateau-RA subpolicies",
        "Plateau-commutation insertion: interior positions open per the "
        "commutation-reduction plateau law. Baseline = historical Paper-I "
        "plateau campaign 9605157 (k50, fixed shortlists, Phase II-III "
        "Qiskit; P0 = endpoint generators). TWO distinct RA subpolicies "
        "run here: [P0: endpoint-generators] ranks whole generators at "
        "the append endpoint; [P0: position-records] ranks "
        "(generator, insertion-position) pairs. Same insertion law, "
        "different Phase-0 population identity.",
        ("hist_plateau", "plateau_append_phase0", "plateau_position_phase0",
         "forced_plateau_append_phase0", "forced_plateau_position_phase0"),
        ("forced-k50 plateau arms (planned): forced admission through "
         "no-positive plateaus, exact k=50 - directly tests whether the "
         "historical deep staircase returns under the new scoring.",),
    ),
    (
        "Always-open insertion (new arm; historical plateau for context)",
        "Always-commutation-reduced insertion: every reduced position is "
        "available every round (maximal insertion freedom). Phase 0 is "
        "position-records by construction (the open insertion domain IS a "
        "position-record concept; no endpoint-generator variant exists). "
        "No historical counterpart; the historical plateau baseline is "
        "context only. One natural cluster (9662334) runs now; the "
        "forced-k50 twin is planned.",
        ("hist_plateau", "always_open_position_phase0",
         "forced_always_open_position_phase0",
         "floors_always_open_position_phase0"),
        ("forced-k50 always-open (planned): same freedom, forced to "
         "exact k=50.",),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra-cells-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--status-note", default="")
    parser.add_argument("--live-tails-dir", type=Path, default=None)
    args = parser.parse_args()

    hist_plateau = load_hist_plateau()
    exact = {r: v["exact_energy"] for r, v in hist_plateau.items()}
    hist_append = load_hist_append(exact, args.workdir)
    ra = load_ra_cells(args.ra_cells_dir)
    # Live partial trajectories of running cells fill any (arm, regime) hole
    # a completed archive has not filled yet; completed evidence wins.
    for key, entry in load_live_cells(args.live_tails_dir, exact).items():
        if key not in ra:
            ra[key] = entry

    def entry_for(key: str, regime: str):
        if key == "hist_plateau":
            return hist_plateau.get(regime)
        if key == "hist_append":
            return hist_append.get(regime)
        return ra.get((key, regime))

    live_depths: dict[str, int] = {}
    depth_file = args.workdir / "live_depths.txt"
    if depth_file.is_file():
        for line in depth_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                live_depths[parts[0]] = int(parts[1])

    def live_depth_for(key: str, regime: str) -> int | None:
        for cell, depth in live_depths.items():
            if f"__{key}__{regime}__" in cell + "__":
                return depth
        return None

    forced_present = any(arm.startswith("forced_") for arm, _ in ra)
    pages = ARM_PAGES if forced_present else FAMILY_PAGES

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        # ------------------------------------------------------------------
        # PAGE 1 - Paper-I facing results: AAVQE vs the two pinned RA
        # methods (forced-k50 append, forced-k50 always-open). The SAME
        # method is shown in every regime - no per-regime selection.
        paper_keys = [
            "hist_append",
            "forced_append_ra",
            "forced_always_open_position_phase0",
        ]
        paper_labels = {
            "hist_append": "AAVQE (published tolmatch r50)",
            "forced_append_ra": "forced-k50 append-RA",
            "forced_always_open_position_phase0": "forced-k50 always-open-RA",
        }

        def _p1_entry(key: str, regime: str):
            if key == "hist_append":
                return hist_append.get(regime)
            return ra.get((key, regime))

        def _p1_err_at(entry, k):
            for kk, err in entry["points"]:
                if kk == k:
                    return err
            return None

        fig = plt.figure(figsize=(15, 11.5))
        grid = fig.add_gridspec(
            3, 3, height_ratios=(1.0, 1.0, 1.45),
            hspace=0.45, top=0.9, bottom=0.03,
        )
        p1_rows = []
        for index, regime in enumerate(REGIMES):
            ax = fig.add_subplot(grid[index // 3, index % 3])
            aavqe = hist_append.get(regime)
            kstar = None
            if aavqe:
                target = aavqe["err_final"] * (10 ** 0.1)
                kstar = next(
                    (kk for kk, err in aavqe["points"] if err <= target),
                    aavqe["k_final"],
                )
            pending_here = []
            for key in paper_keys:
                entry = _p1_entry(key, regime)
                _, _, color, style, marker = FAMILIES[key]
                label = paper_labels[key]
                if entry is None:
                    pending_here.append(label + " [pending]")
                    continue
                if entry.get("live"):
                    label += f" (live @ k={entry['k_final']})"
                ax.plot(
                    [q[0] for q in entry["points"]],
                    [q[1] for q in entry["points"]],
                    style, color=color,
                    linewidth=1.9 if key == "hist_append" else 1.5,
                    marker=marker, markersize=3.6,
                    markevery=(list(FAMILIES).index(key) % 3, 3),
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.4, label=label,
                )
                # Table rows (AAVQE first per regime, highlighted below).
                if key == "hist_append":
                    star_costs = _ra_prefix_tuple(
                        entry.get("cell_name", ""), entry.get("words", []),
                        regime, kstar, args.workdir,
                    ) or {}
                    term_costs = entry.get("costs", {})
                    salg = entry.get("salg_by_round", {})
                    salg_star = salg.get(kstar, "?")
                else:
                    star_costs = _ra_prefix_tuple(
                        entry.get("cell_name", ""), entry.get("words", []),
                        regime, kstar, args.workdir,
                    ) or {}
                    term_costs = _ra_prefix_tuple(
                        entry.get("cell_name", ""), entry.get("words", []),
                        regime, entry["k_final"], args.workdir,
                    ) or {}
                    salg = entry.get("salg_by_round", {})
                    salg_star = salg.get(kstar, "-")
                e_star = _p1_err_at(entry, kstar) if kstar else None
                term_tag = (
                    f"{entry['k_final']}L" if entry.get("live")
                    else (
                        str(entry["k_final"])
                        if key == "hist_append"
                        else f"{entry['k_final']}t"
                    )
                )
                p1_rows.append([
                    regime,
                    paper_labels[key],
                    f"{kstar}* / {term_tag}",
                    (f"{e_star:.1e}" if e_star is not None else "-")
                    + f" / {entry['err_final']:.1e}",
                    f"{star_costs.get('N2q', '?')} / {term_costs.get('N2q', '?')}",
                    f"{star_costs.get('D2q', '?')} / {term_costs.get('D2q', '?')}",
                    f"{star_costs.get('Dc', '?')} / {term_costs.get('Dc', '?')}",
                    f"{star_costs.get('W1q', '?')} / {term_costs.get('W1q', '?')}",
                    f"{salg_star} / "
                    + str(
                        term_costs.get("S_alg", entry.get("costs", {}).get("S_alg", "-"))
                        if key == "hist_append"
                        else entry.get("costs", {}).get("S_alg", "-")
                    ),
                ])
            ax.set_yscale("log")
            ax.set_title(regime.replace("_", " "), fontsize=11)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_xlim(0, K_MAX + 2)
            if pending_here:
                ax.text(0.97, 0.95, "\n".join(pending_here),
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="#666666",
                        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#cccccc"))
            if index == 0:
                ax.legend(fontsize=7.5, loc="lower left")
            if index // 3 == 1:
                ax.set_xlabel("accepted controller round $k$")
            if index % 3 == 0:
                ax.set_ylabel(r"$|E_k - E_{\mathrm{exact}}|$")
        tbl_ax = fig.add_subplot(grid[2, :])
        tbl_ax.axis("off")
        if p1_rows:
            header = ["regime", "method", "k* / k_term", "err k* / term",
                      "N2q", "D2q", "Dc", "W1q", "S_alg"]
            table = tbl_ax.table(
                cellText=p1_rows, colLabels=header, loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.6)
            table.scale(1.0, 1.22)
            for row_index, row in enumerate(p1_rows, start=1):
                if row[1].startswith("AAVQE"):
                    for col in range(len(header)):
                        table[row_index, col].set_facecolor("#dce9f7")
        fig.suptitle(
            "Paper-I results - AAVQE vs forced-k50 RA (append / always-open)",
            fontsize=13, y=0.975,
        )
        fig.text(
            0.06, 0.917,
            "One method per series across ALL regimes (no per-regime "
            "selection). k* = first AAVQE round within 0.1 decades of its "
            "terminal error; all tuples compiled with the same Table-I "
            "Qiskit path. 't' = authenticated natural terminal, 'L' = live "
            "partial run."
            + ("   [" + args.status_note + "]" if args.status_note else ""),
            fontsize=8.5, color="#333333", ha="left", va="bottom", wrap=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Cover page: full campaign map.
        fig, ax = plt.subplots(figsize=(15, 11.5))
        ax.axis("off")
        done = {}
        for (arm, regime) in ra:
            done.setdefault(arm, []).append(regime)
        cover_rows = [
            ["historical plateau", "cluster 9605157 (fixed shortlists, phase II-III Qiskit)", "complete (6/6)", "plateau pages"],
            ["published append", "POWELL tolmatch r50 (adopted in Paper_I.pdf 2026-08-16)", "complete (6/6)", "append page"],
            ["append-RA (natural)", "CHTC 9662333 procs 0-5", f"{len(done.get('append_ra', []))}/6 complete", "append page"],
            ["plateau-RA append-P0 (natural)", "CHTC 9662333 procs 6-11", f"{len(done.get('plateau_append_phase0', []))}/6 complete", "plateau page(s)"],
            ["plateau-RA position-P0 (natural)", "CHTC 9662333 procs 12-17", f"{len(done.get('plateau_position_phase0', []))}/6 complete", "plateau page(s)"],
            ["always-open-RA (natural)", "CHTC 9662334 procs 0-5", f"{len(done.get('always_open_position_phase0', []))}/6 complete", "always-open page"],
            ["forced-k50 (4 arms x 6 regimes)", "24 cells; forced admission at no-positive rounds, exact k=50", ("data present" if forced_present else "running (CHTC 9662370)"), "per-arm pages once data lands"],
            ["append-position-RA", "withdrawn by user (single append-RA arm policy); one banked strong_weak cross-check", "retired", "-"],
        ]
        cover = ax.table(
            cellText=cover_rows,
            colLabels=["route / campaign", "provenance", "status", "PDF location"],
            loc="center", cellLoc="left",
            colWidths=[0.22, 0.40, 0.16, 0.18],
        )
        cover.auto_set_font_size(False)
        cover.set_fontsize(9)
        cover.scale(1.0, 1.8)
        ax.set_title(
            "Paper-I RA campaign map - all routes, one page each family/arm\n"
            "All RA science: gradient-only Phase 0, adaptive shortlists 0-III, "
            "Qiskit-costed Phases I-III, POWELL(200), seeds 7/7, singleton "
            "admission, no pruning/beam.\n" + (args.status_note or ""),
            fontsize=12, pad=28,
        )
        pdf.savefig(fig)
        plt.close(fig)


        for page_title, description, keys, planned_notes in pages:
            fig = plt.figure(figsize=(15, 11.5))
            grid = fig.add_gridspec(
                3, 3, height_ratios=(1.0, 1.0, 1.45),
                hspace=0.42, top=0.865, bottom=0.015,
            )
            axes = [
                [fig.add_subplot(grid[r, c]) for c in range(3)]
                for r in range(2)
            ]
            for index, regime in enumerate(REGIMES):
                ax = axes[index // 3][index % 3]
                pending_here = []
                for key in keys:
                    entry = entry_for(key, regime)
                    if entry is None:
                        if not key.startswith("hist_"):
                            depth = live_depth_for(key, regime)
                            suffix = TERSE_STATUS.get(key, "pending")
                            if depth:
                                suffix += f", at k={depth}"
                            pending_here.append(
                                SHORT.get(key, key) + " [" + suffix + "]"
                            )
                        continue
                    _, label, color, style, marker = FAMILIES[key]
                    points = entry["points"]
                    is_baseline = key.startswith("hist_")
                    if entry.get("live"):
                        label += f" (live @ k={entry['k_final']})"
                    # Policies retrace each other until they diverge, so
                    # stagger marker phase and stack natural on top: every
                    # series stays visible on a shared trajectory.
                    series_rank = list(FAMILIES).index(key)
                    if key.startswith(("forced_", "floors_")):
                        z_order = 2.4
                    elif is_baseline:
                        z_order = 2.2
                    else:
                        z_order = 2.8
                    ax.plot([q[0] for q in points], [q[1] for q in points],
                            style, color=color,
                            linewidth=1.9 if is_baseline else 1.5,
                            marker=marker,
                            markersize=3.6,
                            markevery=(series_rank % 3, 3),
                            markerfacecolor=color, markeredgecolor="white",
                            markeredgewidth=0.4, zorder=z_order, label=label)
                ax.set_yscale("log")
                ax.set_title(regime.replace("_", " "), fontsize=11)
                ax.grid(True, which="both", alpha=0.25)
                ax.set_xlim(0, K_MAX + 2)
                if pending_here:
                    ax.text(0.97, 0.95,
                            "pending:\n" + "\n".join(pending_here),
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7, color="#666666",
                            bbox=dict(boxstyle="round", fc="#f5f5f5",
                                      ec="#cccccc"))
                if index // 3 == 1:
                    ax.set_xlabel("accepted controller round $k$")
                if index % 3 == 0:
                    ax.set_ylabel(r"$|E_k - E_{\mathrm{exact}}|$")
            handles, labels = [], []
            for ax_row in axes:
                for ax in ax_row:
                    for handle, label in zip(*ax.get_legend_handles_labels()):
                        if label not in labels:
                            handles.append(handle)
                            labels.append(label)
            fig.legend(handles, labels, loc="upper right", ncol=1,
                       fontsize=8, frameon=False,
                       bbox_to_anchor=(0.99, 0.985))

            table_ax = fig.add_subplot(grid[2, :])
            table_ax.axis("off")
            dual_point = "hist_append" in keys
            if dual_point:
                columns = ["regime", "series", "k* / k_term",
                           "err k* / term", "N2q", "D2q", "Dc", "W1q",
                           "S_alg"]
            else:
                columns = ["regime", "series", "point", "err",
                           "N2q", "D2q", "Dc", "W1q", "S_alg"]
            rows = []

            def err_at(entry, k):
                for kk, err in entry["points"]:
                    if kk == k:
                        return err
                return None

            def fmt_pair(a, b):
                return f"{a} / {b}"

            def tuple_pair(entry, kstar, kterm, workdir):
                star = _ra_prefix_tuple(
                    entry.get("cell_name", ""), entry.get("words", []),
                    entry.get("regime", ""), kstar, workdir,
                ) or {}
                term = _ra_prefix_tuple(
                    entry.get("cell_name", ""), entry.get("words", []),
                    entry.get("regime", ""), kterm, workdir,
                ) or {}
                return star, term

            for regime in REGIMES:
                hist = entry_for(keys[0], regime)
                ra_entries = [
                    (key, entry_for(key, regime))
                    for key in keys
                    if not key.startswith("hist_")
                ]
                ra_present = [(k, e) for k, e in ra_entries if e is not None]
                if dual_point and hist is not None and ra_present:
                    target = hist["err_final"] * (10 ** 0.1)
                    kstar = next(
                        (kk for kk, err in hist["points"] if err <= target),
                        hist["k_final"],
                    )
                    costs = hist.get("costs", {})
                    salg_hist = hist.get("salg_by_round", {})
                    star, _ = tuple_pair(
                        hist, kstar, hist["k_final"], args.workdir
                    )
                    e_star = err_at(hist, kstar)
                    rows.append([
                        regime, SHORT.get(keys[0], keys[0]),
                        fmt_pair(f"{kstar}*", hist["k_final"]),
                        fmt_pair(
                            f"{e_star:.1e}" if e_star is not None else "-",
                            f"{hist['err_final']:.1e}",
                        ),
                        fmt_pair(star.get("N2q", "?"), costs.get("N2q", "?")),
                        fmt_pair(star.get("D2q", "?"), costs.get("D2q", "?")),
                        fmt_pair(star.get("Dc", "?"), costs.get("Dc", "?")),
                        fmt_pair(star.get("W1q", "?"), costs.get("W1q", "?")),
                        fmt_pair(
                            salg_hist.get(kstar, "?"),
                            costs.get("S_alg", "?"),
                        ),
                    ])
                    for key, entry in ra_present:
                        salg_ra = entry.get("salg_by_round", {})
                        star, term = tuple_pair(
                            entry, kstar, entry["k_final"], args.workdir
                        )
                        e_star = err_at(entry, kstar)
                        term_tag = (
                            f"{entry['k_final']}L"
                            if entry.get("live")
                            else f"{entry['k_final']}t"
                        )
                        rows.append([
                            regime, SHORT.get(key, key),
                            fmt_pair(f"{kstar}*", term_tag),
                            fmt_pair(
                                f"{e_star:.1e}" if e_star is not None
                                else "-",
                                f"{entry['err_final']:.1e}",
                            ),
                            fmt_pair(star.get("N2q", "-"),
                                     term.get("N2q", "pend.")),
                            fmt_pair(star.get("D2q", "-"),
                                     term.get("D2q", "pend.")),
                            fmt_pair(star.get("Dc", "-"),
                                     term.get("Dc", "pend.")),
                            fmt_pair(star.get("W1q", "-"),
                                     term.get("W1q", "pend.")),
                            fmt_pair(
                                salg_ra.get(kstar, "-"),
                                entry["costs"].get("S_alg", "pend."),
                            ),
                        ])
                    continue
                for key in keys:
                    entry = entry_for(key, regime)
                    if entry is None:
                        continue
                    costs = dict(entry.get("costs", {}))
                    if "N2q" not in costs and entry.get("words"):
                        costs.update(
                            _ra_prefix_tuple(
                                entry.get("cell_name", ""),
                                entry.get("words", []),
                                entry.get("regime", regime),
                                entry["k_final"],
                                args.workdir,
                            )
                            or {}
                        )
                    rows.append([
                        regime, SHORT.get(key, key),
                        f"k={entry['k_final']}",
                        f"{entry['err_final']:.2e}",
                        str(costs.get("N2q", "pend.")),
                        str(costs.get("D2q", "pend.")),
                        str(costs.get("Dc", "pend.")),
                        str(costs.get("W1q", "pend.")),
                        str(costs.get("S_alg", "pend.")),
                    ])
            for key in keys:
                if key.startswith("hist_"):
                    continue
                missing = [r for r in REGIMES if (key, r) not in ra]
                if missing:
                    rows.append([
                        ", ".join(m.replace("_", " ") for m in missing),
                        SHORT.get(key, key),
                        "-", RUNNING_STATUS.get(key, "pending"),
                        "-", "-", "-", "-", "-",
                    ])
            widths = [0.13, 0.20, 0.08, 0.14, 0.08, 0.08, 0.09, 0.08, 0.12]
            max_rows_on_plot_page = 20
            first_rows = rows[:max_rows_on_plot_page]
            spill_rows = rows[max_rows_on_plot_page:]
            table = table_ax.table(cellText=first_rows, colLabels=columns,
                                   loc="upper center", cellLoc="center",
                                   colWidths=widths)
            table.auto_set_font_size(False)
            table.set_fontsize(6.5)
            table.scale(1.0, 1.02)
            for row_index, row in enumerate(first_rows, start=1):
                if "AAVQE" in str(row[1]):
                    for col_index in range(len(columns)):
                        table[(row_index, col_index)].set_facecolor("#dce9f7")
            if spill_rows:
                table_ax.text(
                    0.5, 0.005,
                    f"(table continues on next page: {len(spill_rows)} more rows)",
                    transform=table_ax.transAxes, ha="center",
                    fontsize=7, color="#555555",
                )

            planned_text = "  |  ".join(planned_notes)
            fig.text(0.06, 0.004, "PLANNED: " + planned_text,
                     fontsize=8, color="#555555", ha="left")
            fig.suptitle(page_title, fontsize=13, y=0.985)
            note = description
            if args.status_note:
                note += "   [" + args.status_note + "]"
            fig.text(0.06, 0.878, note, fontsize=8.5, color="#333333",
                     ha="left", wrap=True, va="bottom")
            pdf.savefig(fig)
            plt.close(fig)

            while spill_rows:
                chunk, spill_rows = spill_rows[:34], spill_rows[34:]
                fig, ax = plt.subplots(figsize=(15, 11.5))
                ax.axis("off")
                cont = ax.table(cellText=chunk, colLabels=columns,
                                loc="upper center", cellLoc="center",
                                colWidths=widths)
                cont.auto_set_font_size(False)
                cont.set_fontsize(6.5)
                cont.scale(1.0, 1.05)
                for row_index, row in enumerate(chunk, start=1):
                    if "AAVQE" in str(row[1]):
                        for col_index in range(len(columns)):
                            cont[(row_index, col_index)].set_facecolor("#dce9f7")
                ax.set_title(page_title + " - cost table (continued)",
                             fontsize=11)
                pdf.savefig(fig)
                plt.close(fig)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
