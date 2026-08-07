#!/usr/bin/env python3
"""Bounded Route-C HH weak-regime diagnostic autopilot.

Diagnostic-only driver for the route-c-plateau-acquisition-v1 branch.  It runs
fixed, allowlisted Route C variants over the HH weak regimes, updates a Markdown
and PDF tracker, and stops at a wall-clock budget.  It does not stop unrelated
jobs, stage files, push, merge, or update paper-facing artifacts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.request
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO / "raw_outputs/diagnostics/route_c_autopilot_20260610_v1"
TRACKER_DIR = REPO / "docs/reports/route_c_autopilot"
TRACKER_MD = TRACKER_DIR / "route_c_autopilot_tracker.md"
TRACKER_PDF = TRACKER_DIR / "route_c_autopilot_tracker.pdf"
SUMMARY_JSON = RUN_ROOT / "autopilot_summary.json"
JOB_ID = "holstein-route-c-hh-weak-regime-autopilot-v1"
REMOTE_API = "http://127.0.0.1:8765/api/jobs"
TARGET = 2.0e-4

BASE_COMMANDS = {
    "weak_weak": {
        "case_id": "hh_L2_nph2_three_model_sym_weak_weak",
        "label": "HH weak-weak, n_ph_work=2",
        "command": REPO / "raw_outputs/diagnostics/route_c_plateau_v1_20260608/hh_weak_weak_active_dormant_v3/trial_0000/hh_L2_nph2_three_model_sym_weak_weak/logs/command.sh",
        "prior_route_c_error": 1.127398826068604e-03,
        "same_cutoff_ed": -0.9183814647368329,
    },
    "weak_strong": {
        "case_id": "hh_L2_nph4_three_model_sym_weak_strong",
        "label": "HH weak-strong, n_ph_work=4",
        "command": REPO / "raw_outputs/diagnostics/route_c_plateau_v1_20260608/hh_weak_strong_active_dormant_v3/trial_0000/hh_L2_nph4_three_model_sym_weak_strong/logs/command.sh",
        "prior_route_c_energy_current": -1.064783902343971,
        "same_cutoff_ed": -1.138579200359,  # diagnostic same-cutoff ED from prior audit
    },
}

VARIANTS = [
    {
        "name": "spqngd_seed_r005_c64",
        "case": "weak_weak",
        "depth": 60,
        "maxiter": 1600,
        "seed_count": 64,
        "seed_radius": 0.05,
        "seed": 7701,
        "qngd_maxiter": 48,
        "unlock_margin": 3.0e-05,
    },
    {
        "name": "spqngd_seed_r010_c96",
        "case": "weak_weak",
        "depth": 70,
        "maxiter": 1800,
        "seed_count": 96,
        "seed_radius": 0.10,
        "seed": 7711,
        "qngd_maxiter": 48,
        "unlock_margin": 3.0e-05,
    },
    {
        "name": "spqngd_seed_r005_c64",
        "case": "weak_strong",
        "depth": 55,
        "maxiter": 2000,
        "seed_count": 64,
        "seed_radius": 0.05,
        "seed": 8701,
        "qngd_maxiter": 40,
        "unlock_margin": 3.0e-05,
    },
    {
        "name": "spqngd_seed_r010_c96",
        "case": "weak_strong",
        "depth": 60,
        "maxiter": 2200,
        "seed_count": 96,
        "seed_radius": 0.10,
        "seed": 8711,
        "qngd_maxiter": 40,
        "unlock_margin": 3.0e-05,
    },
]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_base_command(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    command_line = ""
    for line in reversed(lines):
        if line.startswith("/") or line.startswith("python"):
            command_line = line
            break
    if not command_line:
        raise RuntimeError(f"No python command found in {path}")
    return shlex.split(command_line)


def set_flag(argv: list[str], flag: str, value: str | int | float | None = None) -> list[str]:
    argv = list(argv)
    while flag in argv:
        i = argv.index(flag)
        del argv[i]
        if i < len(argv) and not str(argv[i]).startswith("--"):
            del argv[i]
    argv.append(flag)
    if value is not None:
        argv.append(str(value))
    return argv


def remove_flag(argv: list[str], flag: str) -> list[str]:
    argv = list(argv)
    while flag in argv:
        i = argv.index(flag)
        del argv[i]
        if i < len(argv) and not str(argv[i]).startswith("--"):
            del argv[i]
    return argv


def result_paths(case_key: str, variant_name: str) -> dict[str, Path]:
    case_id = str(BASE_COMMANDS[case_key]["case_id"])
    case_dir = RUN_ROOT / case_key / variant_name / "trial_0000" / case_id
    return {
        "case_dir": case_dir,
        "json_dir": case_dir / "json",
        "logs_dir": case_dir / "logs",
        "result": case_dir / "json/result.json",
        "current": case_dir / "json/current.json",
        "command": case_dir / "logs/command.sh",
        "stdout": case_dir / "logs/stdout.log",
        "stderr": case_dir / "logs/stderr.log",
        "run_note": case_dir / "json/route_c_autopilot_run_note.json",
    }


def build_variant_command(variant: dict[str, Any]) -> tuple[list[str], dict[str, Path]]:
    case_key = str(variant["case"])
    paths = result_paths(case_key, str(variant["name"]))
    argv = read_base_command(Path(BASE_COMMANDS[case_key]["command"]))
    replacements: dict[str, str | int | float | None] = {
        "--phase3-plateau-acquisition-mode": "novelty_cost_v1",
        "--phase3-plateau-acquisition-score": "log_volume_v1",
        "--phase3-plateau-duplicate-policy": "block_exact_position_v1",
        "--phase3-plateau-unlock-margin": variant["unlock_margin"],
        "--phase3-plateau-seed-probe-mode": "dormant_new_random_v1",
        "--phase3-plateau-seed-probe-count": variant["seed_count"],
        "--phase3-plateau-seed-probe-radius": variant["seed_radius"],
        "--phase3-plateau-seed-probe-seed": variant["seed"],
        "--phase3-plateau-trial-optimizer": "sp_qngd",
        "--phase3-plateau-trial-qngd-maxiter": variant["qngd_maxiter"],
        "--adapt-max-depth": variant["depth"],
        "--adapt-maxiter": variant["maxiter"],
        "--adapt-current-json": str(paths["current"].relative_to(REPO)),
        "--output-json": str(paths["result"].relative_to(REPO)),
        "--static-route-id": "route_c",
        "--allow-legacy-static-route": None,
    }
    for flag, value in replacements.items():
        argv = set_flag(argv, flag, value)
    argv = set_flag(argv, "--skip-pdf", None)
    argv = set_flag(argv, "--skip-trajectory", None)
    # Keep this diagnostic small enough to be a route-method investigation, not a paper artifact.
    argv = remove_flag(argv, "--adapt-final-pdf")
    return argv, paths


def extract_adapt_payload(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    if not isinstance(data, dict):
        return None
    av = data.get("adapt_vqe")
    return av if isinstance(av, dict) else None


def extract_metrics(result_or_current: Path, case_key: str) -> dict[str, Any]:
    av = extract_adapt_payload(result_or_current)
    if not av:
        return {"available": False, "path": str(result_or_current)}
    energy = av.get("energy")
    ed = av.get("exact_gs_energy", BASE_COMMANDS[case_key].get("same_cutoff_ed"))
    abs_delta = av.get("abs_delta_e")
    if abs_delta is None and energy is not None and ed is not None:
        try:
            abs_delta = abs(float(energy) - float(ed))
        except Exception:
            abs_delta = None
    route_c = av.get("route_c_plateau_acquisition") if isinstance(av.get("route_c_plateau_acquisition"), dict) else {}
    state = route_c.get("state") if isinstance(route_c, dict) else {}
    config = route_c.get("config") if isinstance(route_c, dict) else {}
    hist = av.get("history") if isinstance(av.get("history"), list) else []
    route_events = []
    seed_best_drops = []
    for row in hist:
        if not isinstance(row, dict):
            continue
        ev = row.get("route_c_plateau_acquisition")
        if isinstance(ev, dict):
            route_events.append(ev)
            val = ev.get("seed_probe_best_drop")
            try:
                if val is not None and math.isfinite(float(val)):
                    seed_best_drops.append(float(val))
            except Exception:
                pass
    unlocks = [ev for ev in route_events if ev.get("event") == "successful_unlock"]
    failed = [ev for ev in route_events if ev.get("event") == "failed_unlock_dormant_admission"]
    return {
        "available": True,
        "path": str(result_or_current),
        "energy": energy,
        "same_cutoff_ed": ed,
        "abs_delta_e": abs_delta,
        "target_hit_2e-4": (None if abs_delta is None else bool(float(abs_delta) <= TARGET)),
        "ansatz_depth": av.get("ansatz_depth"),
        "num_parameters": av.get("num_parameters"),
        "logical_num_parameters": av.get("logical_num_parameters"),
        "nfev_total": av.get("nfev_total"),
        "stop_reason": av.get("stop_reason"),
        "success": av.get("success"),
        "dormant_count": state.get("dormant_count") if isinstance(state, dict) else None,
        "failed_unlock_count": state.get("failed_unlock_count") if isinstance(state, dict) else None,
        "unlock_count": state.get("unlock_count") if isinstance(state, dict) else None,
        "route_event_count": len(route_events),
        "successful_unlock_events": len(unlocks),
        "failed_unlock_events": len(failed),
        "best_seed_probe_drop": (max(seed_best_drops) if seed_best_drops else None),
        "plateau_score_formula": config.get("score_formula") if isinstance(config, dict) else None,
    }


def current_runner_conflicts() -> list[dict[str, Any]]:
    try:
        with urllib.request.urlopen(REMOTE_API, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    conflicts = []
    for run in data.get("runs", []):
        if run.get("status") == "running" and run.get("job_id") != JOB_ID:
            conflicts.append({
                "id": run.get("id"),
                "job_id": run.get("job_id"),
                "job_name": run.get("job_name"),
                "created_at": run.get("created_at"),
            })
    return conflicts


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Route C weak-regime autopilot tracker")
    lines.append("")
    lines.append(f"Generated UTC: `{utc_now()}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Diagnostic-only; not paper-facing.")
    lines.append("- Branch target: `route-c-plateau-acquisition-v1`.")
    lines.append("- Cases: HH weak-weak and HH weak-strong only.")
    lines.append("- Method family: Route C variants with log-volume plateau acquisition, finite-amplitude dormant/new seed probes, and SP-QNGD plateau trial refit.")
    lines.append("- Hard stops: wall-clock budget, target hit, route plumbing failure, or no remaining variants.")
    lines.append("")
    lines.append("## Current status")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Started UTC: `{summary.get('started_utc')}`")
    lines.append(f"- Updated UTC: `{summary.get('updated_utc')}`")
    lines.append(f"- Budget hours: `{summary.get('budget_hours')}`")
    if summary.get("active_run"):
        lines.append(f"- Active run: `{summary.get('active_run')}`")
    if summary.get("blocker"):
        lines.append(f"- Blocker: `{summary.get('blocker')}`")
    lines.append("")
    lines.append("## Insight log")
    insights = summary.get("insights") or []
    if not insights:
        lines.append("- No completed diagnostic insight yet.")
    else:
        for item in insights[-20:]:
            lines.append(f"- `{item.get('time')}`: {item.get('text')}")
    lines.append("")
    lines.append("## Runs")
    lines.append("| Variant | Case | Status | abs_delta_e | target | Energy | Depth | Dormant | Unlocks | Seed best drop |")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for run in summary.get("runs", []):
        metrics = run.get("metrics") or {}
        def fmt(v: Any) -> str:
            if v is None:
                return "--"
            try:
                if isinstance(v, float) or isinstance(v, int):
                    return f"{float(v):.6e}"
            except Exception:
                pass
            return str(v)
        target = metrics.get("target_hit_2e-4")
        target_s = "hit" if target is True else ("miss" if target is False else "--")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("variant")),
                    str(run.get("case")),
                    str(run.get("status")),
                    fmt(metrics.get("abs_delta_e")),
                    target_s,
                    fmt(metrics.get("energy")),
                    fmt(metrics.get("ansatz_depth")),
                    fmt(metrics.get("dormant_count")),
                    fmt(metrics.get("successful_unlock_events") or metrics.get("unlock_count")),
                    fmt(metrics.get("best_seed_probe_drop")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Interpretation rules")
    lines.append("- If seed probes improve probe energy but unlocks remain zero, optimizer/refit strength is still the bottleneck.")
    lines.append("- If many dormant directions accrue with no seed-probe energy decrease and no unlock, Route C acquisition may be buying geometric span that is not energetically useful at this cutoff.")
    lines.append("- If a run reaches `abs_delta_e <= 2e-4`, compare compiled cost against current Route A before any further scientific conclusion.")
    lines.append("")
    lines.append("## Run artifact pointers")
    for run in summary.get("runs", []):
        result_path = run.get("result_json") or run.get("current_json") or ""
        if result_path:
            lines.append(f"- `{run.get('case')}/{run.get('variant')}`: `{result_path}`")
    lines.append("")
    lines.append("## Raw artifacts")
    lines.append(f"- Summary JSON: `{SUMMARY_JSON.relative_to(REPO)}`")
    lines.append(f"- Tracker PDF: `{TRACKER_PDF.relative_to(REPO)}`")
    lines.append(f"- Run root: `{RUN_ROOT.relative_to(REPO)}`")
    lines.append("")
    return "\n".join(lines)


def render_pdf(md_text: str, pdf_path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.units import inch
    except Exception as exc:
        (pdf_path.with_suffix(".pdf.missing_dependency.txt")).write_text(
            f"reportlab unavailable: {exc}\nMarkdown tracker: {TRACKER_MD}\n"
        )
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=0.55*inch, rightMargin=0.55*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 8
    styles["BodyText"].leading = 10
    styles["Code"].fontSize = 6
    styles["Code"].leading = 7
    story = []
    for block in md_text.split("\n\n"):
        if block.startswith("# "):
            story.append(Paragraph(block[2:], styles["Title"]))
        elif block.startswith("## "):
            story.append(Paragraph(block[3:], styles["Heading2"]))
        elif block.startswith("| "):
            story.append(Preformatted(block, styles["Code"]))
        else:
            safe = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1, 0.08*inch))
    doc.build(story)


def update_tracker(summary: dict[str, Any]) -> None:
    summary["updated_utc"] = utc_now()
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(SUMMARY_JSON, summary)
    md = render_markdown(summary)
    TRACKER_MD.write_text(md)
    render_pdf(md, TRACKER_PDF)


def initial_summary(budget_hours: float) -> dict[str, Any]:
    return {
        "schema": "route_c_weak_regime_autopilot_v1",
        "status": "initialized",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "budget_hours": float(budget_hours),
        "job_id": JOB_ID,
        "scope": "HH weak regimes only; Route C variants only; diagnostic; not paper-facing",
        "runs": [],
        "insights": [],
    }


def add_insight(summary: dict[str, Any], text: str) -> None:
    summary.setdefault("insights", []).append({"time": utc_now(), "text": str(text)})


def classify_insight(run_record: dict[str, Any]) -> str:
    m = run_record.get("metrics") or {}
    case = run_record.get("case")
    variant = run_record.get("variant")
    abs_delta = m.get("abs_delta_e")
    unlocks = m.get("successful_unlock_events") or m.get("unlock_count") or 0
    dormant = m.get("dormant_count")
    seed_drop = m.get("best_seed_probe_drop")
    if m.get("target_hit_2e-4") is True:
        return f"{case}/{variant} hit target with abs_delta_e={float(abs_delta):.6e}; cost comparison against Route A is now required."
    if unlocks:
        return f"{case}/{variant} produced {unlocks} unlock(s) but did not hit target; inspect activated generators and cost."
    if seed_drop is not None and float(seed_drop) > 0.0:
        return f"{case}/{variant} found positive finite-probe energy drop {float(seed_drop):.6e} but no target hit; refit/optimizer remains suspect."
    return f"{case}/{variant} did not unlock; dormant_count={dormant}, abs_delta_e={abs_delta}."


def run_variant(variant: dict[str, Any], remaining_sec: float, summary: dict[str, Any]) -> dict[str, Any]:
    case_key = str(variant["case"])
    name = str(variant["name"])
    argv, paths = build_variant_command(variant)
    for p in (paths["json_dir"], paths["logs_dir"]):
        p.mkdir(parents=True, exist_ok=True)
    command_script = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(REPO)) + "\nexport PYTHONPATH=.\n" + " ".join(shlex.quote(x) for x in argv) + "\n"
    paths["command"].write_text(command_script)
    paths["command"].chmod(0o755)
    run_note = {
        "schema": "route_c_autopilot_variant_run_note_v1",
        "classification": "diagnostic",
        "not_paper_facing": True,
        "case": case_key,
        "case_id": BASE_COMMANDS[case_key]["case_id"],
        "variant": name,
        "variant_settings": variant,
        "source_command": str(BASE_COMMANDS[case_key]["command"].relative_to(REPO)),
        "generated_utc": utc_now(),
    }
    write_json(paths["run_note"], run_note)
    record: dict[str, Any] = {
        "case": case_key,
        "variant": name,
        "status": "running",
        "started_utc": utc_now(),
        "command_sh": str(paths["command"].relative_to(REPO)),
        "stdout_log": str(paths["stdout"].relative_to(REPO)),
        "stderr_log": str(paths["stderr"].relative_to(REPO)),
        "result_json": str(paths["result"].relative_to(REPO)),
        "current_json": str(paths["current"].relative_to(REPO)),
        "metrics": {},
    }
    summary["runs"].append(record)
    summary["active_run"] = f"{case_key}/{name}"
    update_tracker(summary)
    t0 = time.time()
    # Nice the heavy diagnostic so unrelated jobs remain responsive.
    exec_argv = ["/usr/bin/nice", "-n", "10", *argv]
    with paths["stdout"].open("w") as out, paths["stderr"].open("w") as err:
        try:
            proc = subprocess.run(exec_argv, cwd=REPO, stdout=out, stderr=err, timeout=max(1, int(remaining_sec)))
            record["returncode"] = int(proc.returncode)
            record["status"] = "completed" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
            record["returncode"] = None
            add_insight(summary, f"Budget timeout during {case_key}/{name}; scoped process was terminated by subprocess timeout.")
    record["elapsed_sec"] = round(time.time() - t0, 3)
    result_path = paths["result"] if paths["result"].exists() else paths["current"]
    record["metrics"] = extract_metrics(result_path, case_key)
    record["completed_utc"] = utc_now()
    add_insight(summary, classify_insight(record))
    summary.pop("active_run", None)
    update_tracker(summary)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Render tracker and commands without running diagnostics.")
    parser.add_argument("--wait-for-runner-idle", action="store_true", help="Wait while unrelated remote-runner jobs are active.")
    parser.add_argument("--max-idle-wait-min", type=float, default=120.0)
    args = parser.parse_args(argv)

    start = time.time()
    budget_sec = max(60.0, float(args.budget_hours) * 3600.0)
    summary = initial_summary(float(args.budget_hours))
    summary["status"] = "dry_run" if args.dry_run else "running"
    update_tracker(summary)

    if args.dry_run:
        for variant in VARIANTS:
            cmd, paths = build_variant_command(variant)
            summary["runs"].append({
                "case": variant["case"],
                "variant": variant["name"],
                "status": "planned",
                "result_json": str(paths["result"].relative_to(REPO)),
                "current_json": str(paths["current"].relative_to(REPO)),
                "command_preview": " ".join(shlex.quote(x) for x in cmd[:12]) + " ...",
            })
        add_insight(summary, "Dry-run rendered planned Route C weak-regime autopilot variants; no scientific command executed.")
        summary["status"] = "dry_run_completed"
        update_tracker(summary)
        return 0

    if args.wait_for_runner_idle:
        waited = 0.0
        while True:
            conflicts = current_runner_conflicts()
            if not conflicts:
                break
            summary["status"] = "waiting_for_unrelated_runner_jobs"
            summary["blocker"] = "unrelated remote-runner job active; not interfering"
            summary["unrelated_running_jobs"] = conflicts
            update_tracker(summary)
            if waited >= float(args.max_idle_wait_min) * 60.0:
                add_insight(summary, "Stopped before launch because unrelated remote-runner jobs remained active beyond idle-wait budget.")
                summary["status"] = "blocked_unrelated_runner_active"
                update_tracker(summary)
                return 2
            time.sleep(60.0)
            waited += 60.0

    summary.pop("blocker", None)
    summary["status"] = "running"
    update_tracker(summary)

    for variant in VARIANTS:
        elapsed = time.time() - start
        remaining = budget_sec - elapsed
        if remaining < 300.0:
            add_insight(summary, "Stopping because less than 5 minutes remain in budget.")
            break
        rec = run_variant(variant, remaining, summary)
        m = rec.get("metrics") or {}
        if m.get("target_hit_2e-4") is True:
            add_insight(summary, "Stopping on first useful Route C target hit; cost comparison should be next.")
            break
        if rec.get("status") == "timeout":
            break

    summary["status"] = "completed"
    summary["completed_utc"] = utc_now()
    update_tracker(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
