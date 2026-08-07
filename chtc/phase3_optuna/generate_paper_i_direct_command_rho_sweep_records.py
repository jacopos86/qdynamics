#!/usr/bin/env python3
"""Generate a direct-command Paper-I rho sweep.

This is deliberately not an Optuna/record-wrapper batch.  Each row starts from
an already materialized `adapt_pipeline` command and duplicates it across the
rho grid, changing only the trust-region rho and output/current paths.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "chtc" / "phase3_optuna"
BATCH_ID = "paper_i_direct_command_rho_sweep_20260610_v1"
INPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
SOURCES_DIR = INPUT_DIR / "sources"
RHO_GRID = (0.05, 0.1, 0.25, 0.5, 1.0)
OLD_FETCH_ROOT = Path(
    "/Users/jakestrobel/LocalProjects/Holstein_test_fullclone_3_local_outputs/"
    "raw_outputs/chtc_fetches/rho_sweep_7566650_fixed_settings_20260608/raw_outputs"
)


@dataclass(frozen=True)
class CaseTemplate:
    case_id: str
    family: str
    regime: str
    benchmark_id: str
    template_command: str
    note: str


def _template(case: str, benchmark: str) -> str:
    return str(
        OLD_FETCH_ROOT
        / f"routeA_paper_i_fixed_settings_rho_sweep_20260607_v1_{case}_rho0p25"
        / "run"
        / benchmark
        / "trial_0000"
        / benchmark
        / "logs"
        / "command.sh"
    )


CASES = (
    CaseTemplate(
        "hubbard_weak",
        "hubbard",
        "weak",
        "hubbard_L2_three_model_weak",
        _template("hubbard_weak", "hubbard_L2_three_model_weak"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "hubbard_strong",
        "hubbard",
        "strong",
        "hubbard_L2_three_model_strong",
        _template("hubbard_strong", "hubbard_L2_three_model_strong"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "spin_boson_weak",
        "spin_boson",
        "weak",
        "spin_boson_L2_nph1_three_model_weak",
        _template("spin_boson_weak", "spin_boson_L2_nph1_three_model_weak"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "spin_boson_strong",
        "spin_boson",
        "strong",
        "spin_boson_L2_nph2_three_model_strong",
        _template("spin_boson_strong", "spin_boson_L2_nph2_three_model_strong"),
        "Direct command template from old rho=0.25 materialized run; operator-replay strictness is not asserted.",
    ),
    CaseTemplate(
        "hh_weak_weak",
        "hubbard_holstein",
        "weak_weak",
        "hh_L2_nph2_three_model_sym_weak_weak",
        _template("hh_weak_weak", "hh_L2_nph2_three_model_sym_weak_weak"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "hh_strong_weak",
        "hubbard_holstein",
        "strong_weak",
        "hh_L2_nph2_three_model_sym_strong_weak",
        _template("hh_strong_weak", "hh_L2_nph2_three_model_sym_strong_weak"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "hh_weak_strong",
        "hubbard_holstein",
        "weak_strong",
        "hh_L2_nph4_three_model_sym_weak_strong",
        _template("hh_weak_strong", "hh_L2_nph4_three_model_sym_weak_strong"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
    CaseTemplate(
        "hh_strong_strong",
        "hubbard_holstein",
        "strong_strong",
        "hh_L2_nph4_three_model_sym_strong_strong",
        _template("hh_strong_strong", "hh_L2_nph4_three_model_sym_strong_strong"),
        "Direct command template from old rho=0.25 materialized run.",
    ),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_command(path: Path) -> list[str]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("set "):
            continue
        if "pipelines.static_adapt.adapt_pipeline" in stripped:
            return shlex.split(stripped)
    raise ValueError(f"no adapt_pipeline command found in {path}")


def _get_flag(tokens: list[str], flag: str) -> str | None:
    try:
        idx = tokens.index(flag)
    except ValueError:
        return None
    return tokens[idx + 1] if idx + 1 < len(tokens) else None


def _rho_slug(rho: float) -> str:
    return str(rho).replace(".", "p")


def _stage_selected_logical(case_dir: Path, selected_path: str | None) -> dict[str, Any] | None:
    if not selected_path:
        return None
    selected = Path(selected_path)
    if selected.is_absolute() and str(selected).startswith("/work/raw_outputs/"):
        rel = Path(str(selected).removeprefix("/work/raw_outputs/"))
        local = OLD_FETCH_ROOT / rel
    else:
        local = selected if selected.is_absolute() else REPO_ROOT / selected
    if not local.exists():
        raise FileNotFoundError(f"selected logical source not found: {local}")
    dest = case_dir / "source_selected_logical.json"
    shutil.copy2(local, dest)
    return {
        "original_path": str(local),
        "staged_path": str(dest.relative_to(REPO_ROOT)),
        "sha256": _sha256(dest),
    }


def main() -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    row_specs: list[dict[str, Any]] = []

    for case in CASES:
        command_path = Path(case.template_command)
        if not command_path.exists():
            raise FileNotFoundError(command_path)
        case_dir = SOURCES_DIR / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        staged_command = case_dir / "template_command_rho0p25.sh"
        shutil.copy2(command_path, staged_command)
        tokens = _read_command(staged_command)
        selected_info = _stage_selected_logical(case_dir, _get_flag(tokens, "--adapt-selected-logical-source-json"))
        template_info = {
            "original_path": str(command_path),
            "staged_path": str(staged_command.relative_to(REPO_ROOT)),
            "sha256": _sha256(staged_command),
            "template_phase2_rho": _get_flag(tokens, "--phase2-rho"),
            "template_phase1_score_mode": _get_flag(tokens, "--phase1-score-mode"),
            "selected_logical": selected_info,
            "has_optuna_wrapper": any("phase3_policy_optuna" in token or "oracle-grid" in token for token in tokens),
        }
        if template_info["has_optuna_wrapper"]:
            raise RuntimeError(f"forbidden wrapper in {command_path}")
        for rho in RHO_GRID:
            row_id = f"{case.case_id}_rho{_rho_slug(rho)}"
            row_specs.append(
                {
                    "row_id": row_id,
                    "batch_id": BATCH_ID,
                    "case": asdict(case),
                    "rho": rho,
                    "rho_flag": "--phase2-rho",
                    "runner_mode": "direct_adapt_pipeline_command_template",
                    "template": template_info,
                    "allowed_mutations": [
                        "python executable path normalization",
                        "--phase2-rho",
                        "--output-json path",
                        "--adapt-current-json path if present in template",
                        "portable rewrite of staged selected-logical source path if present in template",
                    ],
                }
            )

    specs = {
        "schema": "paper_i_direct_command_rho_sweep_specs_v1",
        "batch_id": BATCH_ID,
        "rho_grid": list(RHO_GRID),
        "row_count": len(row_specs),
        "run_class": "candidate_fixed_settings_sensitivity",
        "strict_operator_replay_claim": False,
        "note": "Generated after user instructed full paper-settings rho sweep; rows use materialized direct adapt_pipeline commands, not Optuna wrapper.",
        "rows": row_specs,
    }
    specs_path = INPUT_DIR / "direct_command_rho_sweep_specs.json"
    specs_path.write_text(json.dumps(specs, indent=2) + "\n")
    (INPUT_DIR / "direct_command_rho_sweep_row_ids.txt").write_text("\n".join(row["row_id"] for row in row_specs) + "\n")

    submit_path = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
    submit_path.write_text(
        "\n".join(
            [
                "universe = vanilla",
                "executable = chtc/phase3_optuna/run_direct_command_rho_sweep_task_apptainer.sh",
                "arguments = $(row_id)",
                "should_transfer_files = YES",
                "when_to_transfer_output = ON_EXIT_OR_EVICT",
                "transfer_executable = True",
                "preserve_relative_paths = True",
                "transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna",
                "transfer_output_files = raw_outputs, logs",
                f"log = logs/{BATCH_ID}.$(Cluster).$(Process).log",
                f"output = logs/{BATCH_ID}.$(Cluster).$(Process).out",
                f"error = logs/{BATCH_ID}.$(Cluster).$(Process).err",
                "stream_output = False",
                "stream_error = False",
                "requirements = TARGET.HasSIF",
                "request_cpus = 10",
                "request_memory = 48GB",
                "request_disk = 122880MB",
                "+MaxRuntime = 172800",
                f'+JobBatchName = "holstein-{BATCH_ID}"',
                "environment = \"PHASE3_TERMINATE_ON_STALE_PROGRESS=1 PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 PHASE3_PROGRESS_STALE_AFTER_SEC=3600 PHASE3_HEARTBEAT_INTERVAL_SEC=60 PHASE3_SHELL_HEARTBEAT_SEC=60\"",
                f"queue row_id from chtc/phase3_optuna/input/{BATCH_ID}/direct_command_rho_sweep_row_ids.txt",
                "",
            ]
        )
    )
    print(json.dumps({"batch_id": BATCH_ID, "rows": len(row_specs), "submit": str(submit_path), "specs": str(specs_path)}, indent=2))


if __name__ == "__main__":
    main()
