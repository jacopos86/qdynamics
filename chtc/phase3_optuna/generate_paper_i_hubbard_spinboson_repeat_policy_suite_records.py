#!/usr/bin/env python3
"""Generate Paper-I Hubbard/spin-boson comparator repeat-policy suites.

These records are intentionally narrower than the generic Table-I generator:
we need fixed-horizon/no-target continuation records and explicit repeat-policy
separation for the four Hubbard/spin-boson L=2 cases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (  # noqa: E402
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_TARGET,
)

DEFAULT_INPUT_ROOT = REPO_ROOT / "chtc" / "phase3_optuna" / "input"
DEFAULT_RUN_ROOT = Path("raw_outputs")
RECORD_FILENAME = "generic_static_table_records.tsv"

METHODS: tuple[tuple[str, str], ...] = (
    ("append", "static_full_meta_append_adapt_vqe"),
    ("tetris", "static_tetris_qubit_adapt_vqe"),
    ("geo", "static_geo_adapt_vqe"),
    ("qubit_qeb", "static_qubit_qeb_adapt_vqe"),
)


@dataclass(frozen=True)
class CaseSpec:
    family: str
    case_id: str
    cap: int
    same_cutoff_exact_gs_energy: str
    exact_reference_energy: str = ""
    exact_reference_n_ph_max: str = ""


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        cap=80,
        same_cutoff_exact_gs_energy="-1.7655644370746375",
    ),
    CaseSpec(
        family="hubbard",
        case_id="hubbard_L2_three_model_strong",
        cap=20,
        same_cutoff_exact_gs_energy="-1.3860009363293826",
    ),
    CaseSpec(
        family="spin_boson",
        case_id="spin_boson_L2_nph1_three_model_weak",
        cap=20,
        same_cutoff_exact_gs_energy="-0.0016671283442304832",
        exact_reference_energy="-0.0016685214031349203",
        exact_reference_n_ph_max="5",
    ),
    CaseSpec(
        family="spin_boson",
        case_id="spin_boson_L2_nph2_three_model_strong",
        cap=20,
        same_cutoff_exact_gs_energy="-0.006696413817604584",
        exact_reference_energy="-0.00669648167383492",
        exact_reference_n_ph_max="6",
    ),
)

SUITES: Mapping[str, Mapping[str, str]] = {
    "no_repeat": {
        "dir_name": "paper_i_tables_i_ii_no_repeat_comparator_capmatch_20260610_v1",
        "suite_id": "suite_A_no_repeat",
        "repeat_policy": "selected_labels_excluded",
        "phase3_adapt_allow_repeats": "false",
        "description": "native finite-pool/no-repeat comparator suite; Geo-ADAPT is forced no-repeat here",
    },
    "repeat_enabled": {
        "dir_name": "paper_i_tables_i_ii_repeat_enabled_comparator_capmatch_20260610_v1",
        "suite_id": "suite_B_repeat_enabled",
        "repeat_policy": "with_replacement_except_immediate_repeat",
        "phase3_adapt_allow_repeats": "true",
        "description": "repeat-enabled diagnostic comparator suite; not merged with native baseline labels",
    },
}

FIELDS: tuple[str, ...] = (
    "record_id",
    "suite_id",
    "suite_description",
    "repeat_policy",
    "family",
    "case_id",
    "method_key",
    "algorithm_id",
    "suite_profile",
    "optimizer_profile",
    "energy_stop_target",
    "first_hit_thresholds",
    "generic_adapt_stop_policy",
    "phase3_adapt_max_depth",
    "phase3_adapt_allow_repeats",
    "same_cutoff_exact_gs_energy",
    "exact_reference_energy",
    "exact_reference_n_ph_max",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "planned_output_root",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_id(*, suite_id: str, case_id: str, method_key: str) -> str:
    return f"{suite_id}__{case_id}__{method_key}"


def _rows_for_suite(*, suite_key: str, output_root: Path) -> list[dict[str, str]]:
    suite = SUITES[suite_key]
    rows: list[dict[str, str]] = []
    for case in CASES:
        for method_key, algorithm_id in METHODS:
            rows.append(
                {
                    "record_id": _record_id(
                        suite_id=str(suite["suite_id"]),
                        case_id=case.case_id,
                        method_key=method_key,
                    ),
                    "suite_id": str(suite["suite_id"]),
                    "suite_description": str(suite["description"]),
                    "repeat_policy": str(suite["repeat_policy"]),
                    "family": case.family,
                    "case_id": case.case_id,
                    "method_key": method_key,
                    "algorithm_id": algorithm_id,
                    "suite_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
                    "optimizer_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
                    # Must remain blank. fixed_horizon_no_target_v1 rejects target-hit stopping.
                    "energy_stop_target": "",
                    "first_hit_thresholds": f"{float(PAPER_I_MAIN_TABLES_SPSA_TARGET):.4g}",
                    "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
                    "phase3_adapt_max_depth": str(case.cap),
                    "phase3_adapt_allow_repeats": str(suite["phase3_adapt_allow_repeats"]),
                    "same_cutoff_exact_gs_energy": case.same_cutoff_exact_gs_energy,
                    "exact_reference_energy": case.exact_reference_energy,
                    "exact_reference_n_ph_max": case.exact_reference_n_ph_max,
                    "primary_energy_metric": "same_cutoff_abs_delta_e",
                    "same_cutoff_error_role": "primary",
                    "planned_output_root": str(output_root / str(suite["dir_name"])),
                }
            )
    return rows


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=list(FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in FIELDS})


def _write_local_runner(path: Path, *, records_path: Path, output_root: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"RECORDS={shlex.quote(str(records_path))}",
        f"OUTROOT={shlex.quote(str(output_root))}",
        "mkdir -p \"$OUTROOT\"",
    ]
    for row in rows:
        record_id = str(row["record_id"])
        row_out = output_root / record_id
        lines.append(
            "python -m chtc.phase3_optuna.generic_static_table_runner "
            f"{shlex.quote(record_id)} \"$RECORDS\" {shlex.quote(str(row_out))}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def generate(*, input_root: Path, output_root: Path, suite_keys: Iterable[str]) -> dict[str, Any]:
    suite_summaries: list[dict[str, Any]] = []
    for suite_key in suite_keys:
        if suite_key not in SUITES:
            raise ValueError(f"Unknown suite {suite_key!r}; expected one of {sorted(SUITES)}")
        suite = SUITES[suite_key]
        suite_dir = input_root / str(suite["dir_name"])
        suite_output_root = output_root / str(suite["dir_name"])
        records_path = suite_dir / RECORD_FILENAME
        rows = _rows_for_suite(suite_key=suite_key, output_root=output_root)
        _write_tsv(records_path, rows)
        runner_path = suite_dir / "run_local_sequential.sh"
        _write_local_runner(runner_path, records_path=records_path, output_root=suite_output_root, rows=rows)
        summary = {
            "suite_key": suite_key,
            "suite_id": suite["suite_id"],
            "description": suite["description"],
            "repeat_policy": suite["repeat_policy"],
            "phase3_adapt_allow_repeats": suite["phase3_adapt_allow_repeats"],
            "input_dir": str(suite_dir),
            "records_path": str(records_path),
            "records_sha256": _sha256(records_path),
            "local_runner": str(runner_path),
            "local_runner_sha256": _sha256(runner_path),
            "planned_output_root": str(suite_output_root),
            "record_count": len(rows),
            "case_ids": sorted({row["case_id"] for row in rows}),
            "algorithm_ids": sorted({row["algorithm_id"] for row in rows}),
            "stop_policy": "fixed_horizon_no_target_v1",
            "energy_stop_target": "absent/blank",
            "first_hit_thresholds": [float(PAPER_I_MAIN_TABLES_SPSA_TARGET)],
            "primary_energy_metric": "same_cutoff_abs_delta_e",
        }
        (suite_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["manifest_path"] = str(suite_dir / "manifest.json")
        summary["manifest_sha256"] = _sha256(suite_dir / "manifest.json")
        suite_summaries.append(summary)
    overall = {
        "schema": "paper_i_hubbard_spinboson_repeat_policy_suite_records_v1",
        "generated_by": "chtc.phase3_optuna.generate_paper_i_hubbard_spinboson_repeat_policy_suite_records",
        "suite_count": len(suite_summaries),
        "record_count": sum(int(s["record_count"]) for s in suite_summaries),
        "suites": suite_summaries,
    }
    return overall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITES),
        default=None,
        help="Suite(s) to generate. Defaults to both no_repeat and repeat_enabled.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_keys = tuple(args.suite or ("no_repeat", "repeat_enabled"))
    summary = generate(input_root=args.input_root, output_root=args.output_root, suite_keys=suite_keys)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
