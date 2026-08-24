"""Generate the single Paper-II run manifest: commands and every parameter.

Written by a program rather than by hand so it cannot drift from the registry
or from the runner's actual defaults. Regenerate with:

    PYTHONPATH=. python3 -m pipelines.time_dynamics.write_paper_ii_manifest
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pipelines.time_dynamics import paper_ii_runs as R
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser as _runner_parser,
)

OUT = Path("agent_guidance/time-dynamics/paper-ii-run-parameters.md")

CANONICAL = [
    ("exchange", "this work", R.STATE_MOTION_CONTROL.control_id),
    ("append_only", "growth-only ablation of this work", R.STATE_MOTION_CONTROL.control_id),
    ("avqds", "comparator rule on shared numerics", R.STATE_MOTION_CONTROL.control_id),
    ("avqds_published", "comparator as published", R.DELTA_THETA_CONTROL.control_id),
]


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    parser = _runner_parser()
    defaults = vars(parser.parse_args(["--artifact-json", "x", "--output-json", "y"]))

    lines = [
        "# Paper II — run commands and parameter settings",
        "",
        "**Generated file — do not edit.** Regenerate with:",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m pipelines.time_dynamics.write_paper_ii_manifest",
        "```",
        "",
        f"Source commit at generation: `{sha}`",
        "",
        "Every Paper-II run is composed from the registry in",
        "`pipelines/time_dynamics/paper_ii_runs.py`. A run is one choice from each",
        "axis below; nothing else varies, and `test/test_paper_ii_runs.py` asserts",
        "that every registered command parses against the runner's live CLI.",
        "",
        "## Axes",
        "",
    ]
    lines.append("### Structural arms\n")
    lines.append(_table([(a, ARM.note.replace("\n", " ")) for a, ARM in R.ARMS.items()],
                        ["arm", "meaning"]))
    lines.append("\n### Insertion gates\n")
    lines.append(_table([(g, G.note.replace("\n", " ")) for g, G in R.GATES.items()],
                        ["gate", "meaning"]))
    lines.append("\n### Inner numerics (shared by every arm in a comparison)\n")
    lines.append(_table([(n, N.integrator, N.ridge_lambda, N.solve_damping, N.pinv_rcond)
                         for n, N in R.NUMERICS.items()],
                        ["id", "integrator", "ridge", "damping", "pinv rcond"]))
    lines.append("\n### Step control\n")
    lines.append(_table([(c, C.note.replace("\n", " ")) for c, C in R.STEP_CONTROLS.items()],
                        ["id", "meaning"]))
    lines.append("\n### Drives\n")
    lines.append(_table([(d, D.amplitude, D.omega) for d, D in R.DRIVES.items()],
                        ["drive", "A", "omega"]))
    lines.append("\n### Horizons\n")
    lines.append(_table([(h, H.t_final, H.num_times) for h, H in R.HORIZONS.items()],
                        ["horizon", "t_final", "checkpoints"]))
    lines.append("\n### Regimes\n")
    lines.append(_table([(r, RG.seed_path, RG.n_ph_max,
                          "built" if RG.available else "**MISSING**")
                         for r, RG in R.REGIMES.items()],
                        ["regime", "seed", "n_ph", "status"]))

    lines.append("\n## Commands\n")
    for arm, meaning, control in CANONICAL:
        run = R.build_run(regime="hh_snake_nph1", arm=arm, drive="strongfast",
                          horizon="t10", numerics=R.RK4_NUMERICS.numerics_id,
                          step_control=control,
                          output_json=f"output/<batch>/strongfast_{arm}/run.json")
        lines += [f"### `{arm}` — {meaning}\n", "```bash", run.shell(), "```", ""]

    lines.append("## Every effective parameter\n")
    lines.append("Registry-set values, then the runner defaults they sit on. A value "
                 "marked *default* is not stated anywhere in the registry; it is what "
                 "the runner uses when nothing overrides it.\n")
    for arm, meaning, control in CANONICAL:
        run = R.build_run(regime="hh_snake_nph1", arm=arm, drive="strongfast",
                          horizon="t10", numerics=R.RK4_NUMERICS.numerics_id,
                          step_control=control, output_json="<output>")
        eff = vars(parser.parse_args(list(run.argv())))
        rows = [(k, eff[k], "registry" if eff[k] != defaults.get(k) else "default")
                for k in sorted(eff) if k not in ("artifact_json", "output_json")]
        lines += [f"\n<details><summary><b>{arm}</b> — {len(rows)} parameters"
                  f" ({sum(1 for r in rows if r[2] == 'registry')} set by registry)"
                  "</summary>\n",
                  _table(rows, ["parameter", "value", "source"]), "\n</details>"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
