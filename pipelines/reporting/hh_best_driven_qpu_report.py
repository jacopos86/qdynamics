#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_CLAUDE_T1 = Path('artifacts/json/nisq_dynamics_pareto_L2_driven_t1p0_final_20260325.json')
DEFAULT_CLAUDE_T2 = Path('artifacts/json/nisq_dynamics_pareto_L2_driven_t2p0_final_20260325.json')
DEFAULT_OURS = Path('artifacts/agent_runs/20260326_hh_l2_driven_realtime_pareto_sweep/driven_locked7_short.json')
DEFAULT_COMPILE = Path('artifacts/json/compile_scout_locked7_marrakesh.json')
DEFAULT_SOURCE_PDF = Path('artifacts/reports/hh_driven_dynamics_comparison_20260326.pdf')
DEFAULT_OUTPUT = Path('artifacts/reports/hh_best_driven_qpu_report_20260326.pdf')


@dataclass(frozen=True)
class Candidate:
    label: str
    method: str
    horizon: str
    hw_2q: int
    depth: int
    fidelity_min: float
    delta_max: float
    note: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise TypeError(f'{path} must contain a top-level JSON object.')
    return payload


def _find_claude_result(payload: dict[str, Any], *, ordering: str, trotter_steps: int, prune_threshold: float) -> dict[str, Any]:
    for row in payload.get('results', []):
        if not isinstance(row, dict):
            continue
        if bool(row.get('skipped', False)):
            continue
        if str(row.get('ordering')) != ordering:
            continue
        if int(row.get('trotter_steps', -1)) != int(trotter_steps):
            continue
        if abs(float(row.get('prune_threshold', 0.0)) - float(prune_threshold)) > 1e-12:
            continue
        return row
    raise KeyError(f'No Claude result found for ordering={ordering}, trotter_steps={trotter_steps}, prune_threshold={prune_threshold}')


def _time_array(settings: dict[str, Any]) -> np.ndarray:
    return np.linspace(0.0, float(settings.get('t_final', 0.0)), int(settings.get('num_times', 0)))


def _render_manifest_page(pdf: PdfPages, candidates: list[Candidate]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.text(0.08, 0.93, 'Best driven QPU candidates', fontsize=18, fontweight='bold')
    lines = [
        'PARAMETER MANIFEST',
        '',
        'Model: Hubbard-Holstein',
        'Ansatzes: Claude Suzuki-2 Trotter shortlist + our fixed-manifold McLachlan locked7',
        'Drive enabled: True',
        't (hopping): 1.0',
        'U (interaction): 4.0',
        'dv (disorder): 0.0',
        'L: 2',
        'omega0: 1.0',
        'g_ep: 0.5',
        '',
        'Selection rule for this short PDF:',
        '- keep only candidates still relevant for a real QPU story',
        '- require fidelity ~0.9+ and |ΔE| around 1e-1 to 1e-2 when possible',
        '- prefer low two-qubit count; keep one higher-cost upgrade if the accuracy gain is material',
        '',
        f'Best numerical result: {candidates[0].label} | 2Q={candidates[0].hw_2q} | depth={candidates[0].depth} | min fidelity={candidates[0].fidelity_min:.4f} | max |ΔE|={candidates[0].delta_max:.4f}',
        f'Best simple hardware demo: {candidates[1].label} | 2Q={candidates[1].hw_2q} | depth={candidates[1].depth} | min fidelity={candidates[1].fidelity_min:.4f} | max |ΔE|={candidates[1].delta_max:.4f}',
        '',
        'Ruled out from the short list:',
        '- Claude t=2, 2-step Trotter: fidelity collapses (~0.66) and |ΔE| is too large (~1.17).',
        '- Our driven pareto McLachlan: nearly same trajectory quality as locked7, but much heavier cost/workload.',
        '- Pruned / non-weight-sorted Trotter variants: clearly inferior for hardware-facing demos.',
    ]
    y = 0.88
    for line in lines:
        fig.text(0.08, y, line, fontsize=11 if line == 'PARAMETER MANIFEST' else 10.5, family='DejaVu Sans Mono' if ':' in line and line.split(':',1)[0] in {'Model','Ansatzes','Drive enabled','t (hopping)','U (interaction)','dv (disorder)','L','omega0','g_ep'} else None)
        y -= 0.036 if line else 0.02
    pdf.savefig(fig)
    plt.close(fig)


def _render_table_page(pdf: PdfPages, candidates: list[Candidate]) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.axis('off')
    col_labels = ['Candidate', 'Method', 'Horizon', 'HW 2Q', 'Depth', 'Min fidelity', 'Max |ΔE|', 'Note']
    rows = [[c.label, c.method, c.horizon, str(c.hw_2q), str(c.depth), f'{c.fidelity_min:.4f}', f'{c.delta_max:.4f}', c.note] for c in candidates]
    col_widths = [0.22, 0.16, 0.13, 0.07, 0.07, 0.10, 0.09, 0.16]
    tbl = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d9edf7')
        elif r % 2 == 1:
            cell.set_facecolor('#f7f7f7')
    ax.set_title('Shortlist table — only the best remaining driven candidates', fontsize=14, pad=14)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _render_cost_page(pdf: PdfPages, candidates: list[Candidate]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    labels = [c.label.replace('Claude ', '').replace('Our ', '') for c in candidates]
    x = np.arange(len(candidates))

    ax = axes[0]
    ax.axvline(100, color='red', linestyle='--', alpha=0.6)
    ax.axhline(0.9, color='orange', linestyle=':', alpha=0.6)
    colors = ['tab:blue' if 'McLachlan' in c.label else 'tab:green' for c in candidates]
    ax.scatter([c.hw_2q for c in candidates], [c.fidelity_min for c in candidates], s=100, c=colors, edgecolors='black', linewidths=0.5)
    for i, c in enumerate(candidates):
        ax.annotate(labels[i], (c.hw_2q, c.fidelity_min), textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xlabel('HW 2Q gates')
    ax.set_ylabel('Min fidelity')
    ax.set_title('Cost vs fidelity')
    ax.set_xlim(0, 150)
    ax.set_ylim(0.88, 1.005)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, [c.delta_max for c in candidates], color=colors)
    ax.axhline(1e-1, color='orange', linestyle=':', alpha=0.6)
    ax.axhline(1e-2, color='green', linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right', fontsize=8)
    ax.set_ylabel('Max |ΔE|')
    ax.set_title('Worst energy error across trajectory')
    ax.grid(alpha=0.3, axis='y')
    for bar, c in zip(bars, candidates):
        ax.text(bar.get_x() + bar.get_width()/2, c.delta_max * 1.12, f'{c.delta_max:.3g}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Shortlist quality / cost summary', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _render_claude_deltae_page(pdf: PdfPages, t1_settings: dict[str, Any], t2_settings: dict[str, Any], t1_2: dict[str, Any], t1_3: dict[str, Any], t2_3: dict[str, Any]) -> None:
    t1_times = _time_array(t1_settings)
    t2_times = _time_array(t2_settings)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    ax = axes[0]
    ax.plot(t1_times, np.asarray(t1_2['energy_error_trajectory'], dtype=float), color='tab:green', linewidth=1.8, label='2-step')
    ax.plot(t1_times, np.asarray(t1_3['energy_error_trajectory'], dtype=float), color='tab:blue', linewidth=1.8, label='3-step')
    ax.axhline(1e-1, color='orange', linestyle=':', alpha=0.6)
    ax.axhline(1e-2, color='green', linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_title('Claude driven t=1: |ΔE| vs time')
    ax.set_xlabel('Time')
    ax.set_ylabel('|ΔE|')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t2_times, np.asarray(t2_3['energy_error_trajectory'], dtype=float), color='tab:blue', linewidth=1.8, label='3-step')
    ax.axhline(1e-1, color='orange', linestyle=':', alpha=0.6)
    ax.axhline(1e-2, color='green', linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_title('Claude driven t=2: |ΔE| vs time')
    ax.set_xlabel('Time')
    ax.set_ylabel('|ΔE|')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle('Claude shortlisted Trotter candidates — energy error traces', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def _render_ours_page(pdf: PdfPages, ours: dict[str, Any], compile_locked: dict[str, Any]) -> None:
    traj = list(ours.get('trajectory', []))
    times = np.asarray([row.get('physical_time', row.get('time', 0.0)) for row in traj], dtype=float)
    fidelity = np.asarray([row.get('audit', {}).get('fidelity_exact_audit', float('nan')) for row in traj], dtype=float)
    energy_ansatz = np.asarray([row.get('audit', {}).get('energy_ansatz_exact_audit', float('nan')) for row in traj], dtype=float)
    energy_ref = np.asarray([row.get('audit', {}).get('energy_reference_exact_audit', float('nan')) for row in traj], dtype=float)
    delta_e = np.asarray([row.get('audit', {}).get('abs_energy_total_error_exact_audit', float('nan')) for row in traj], dtype=float)
    summary = dict(ours.get('summary', {}))
    backend = dict(compile_locked.get('selected_backend', {}))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.3))

    ax = axes[0, 0]
    ax.plot(times, fidelity, color='tab:blue', linewidth=1.8)
    ax.axhline(0.9, color='orange', linestyle=':', alpha=0.6)
    ax.axhline(0.99, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Our locked7: fidelity vs time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fidelity vs exact')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, energy_ref, color='black', linewidth=2.0, label='exact')
    ax.plot(times, energy_ansatz, color='tab:blue', linewidth=1.8, label='locked7')
    ax.set_title('Our locked7: total energy vs time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Total energy')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(times, delta_e, color='tab:red', linewidth=1.8)
    ax.axhline(1e-1, color='orange', linestyle=':', alpha=0.6)
    ax.axhline(1e-2, color='green', linestyle=':', alpha=0.6)
    ax.set_yscale('log')
    ax.set_title('Our locked7: |ΔE| vs time')
    ax.set_xlabel('Time')
    ax.set_ylabel('|ΔE|')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    labels = ['HW 2Q', 'Depth', 'Oracle evals']
    vals = [int(backend.get('compiled_count_2q', 0)), int(backend.get('compiled_depth', 0)), int(summary.get('oracle_evaluations_total', 0))]
    ax.bar(np.arange(3), vals, color=['tab:blue', 'tab:green', 'tab:red'])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.set_title('Our locked7: cost / workload')
    ax.grid(alpha=0.3, axis='y')
    for i, v in enumerate(vals):
        ax.text(i, v * 1.08, str(v), ha='center', va='bottom', fontsize=8)

    fig.suptitle('Our best completed McLachlan candidate: locked7', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _render_hesitancy_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    fig.text(0.08, 0.93, 'Why I am still hesitant about our McLachlan dynamics', fontsize=17, fontweight='bold')
    lines = [
        '1. The per-shot circuit is cheap, but the full protocol is not.',
        '   - locked7 compiles to 25 two-qubit gates and depth 75 on FakeMarrakesh.',
        '   - but the completed driven run still needed 3232 oracle evaluations.',
        '',
        '2. It is not the same hardware object as the Trotter point.',
        '   - Claude\'s Trotter candidate is a direct propagation circuit with a straightforward 2Q / depth story.',
        '   - our McLachlan candidate is a variational control loop with repeated measurement / geometry updates.',
        '',
        '3. The current driven result is still an idealized protocol result.',
        '   - the completed sweep used ideal-noise trajectory auditing.',
        '   - that is strong evidence of algorithmic promise, but not yet a full end-to-end hardware-readiness proof.',
        '',
        '4. In practice, the controller can become the bottleneck before the state-prep circuit does.',
        '',
        'My honest read:',
        '- locked7 is the best completed numerical result we have.',
        '- Claude t=1, 2-step Trotter is still the easier first real-QPU demo to defend.',
    ]
    y = 0.86
    for line in lines:
        fig.text(0.08, y, line, fontsize=11)
        y -= 0.045 if line else 0.022
    pdf.savefig(fig)
    plt.close(fig)


def build_report(*, claude_t1_json: Path, claude_t2_json: Path, ours_json: Path, compile_json: Path, source_pdf: Path, output_pdf: Path) -> Path:
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyPDF2 is required to splice source PDF pages for hh_best_driven_qpu_report."
        ) from exc

    claude_t1 = _read_json(claude_t1_json)
    claude_t2 = _read_json(claude_t2_json)
    ours = _read_json(ours_json)
    compile_locked = _read_json(compile_json)

    t1_2 = _find_claude_result(claude_t1, ordering='weight_sorted', trotter_steps=2, prune_threshold=0.0)
    t1_3 = _find_claude_result(claude_t1, ordering='weight_sorted', trotter_steps=3, prune_threshold=0.0)
    t2_3 = _find_claude_result(claude_t2, ordering='weight_sorted', trotter_steps=3, prune_threshold=0.0)
    backend = dict(compile_locked.get('selected_backend', {}))
    summary = dict(ours.get('summary', {}))

    candidates = [
        Candidate('Our locked7 McLachlan', 'fixed-manifold McLachlan', 'driven, short run', int(backend.get('compiled_count_2q', -1)), int(backend.get('compiled_depth', -1)), float(summary.get('min_fidelity_exact_audit', float('nan'))), float(summary.get('max_abs_energy_total_error_exact_audit', float('nan'))), 'best completed numerical result'),
        Candidate('Claude Trotter 2-step (t=1)', 'Suzuki-2 Trotter', 'driven, t=1.0', int(t1_2['transpile']['fake_marrakesh']['compiled_count_2q']), int(t1_2['transpile']['fake_marrakesh']['depth']), float(t1_2['fidelity_min']), float(t1_2['energy_error_max']), 'best simple hardware demo'),
        Candidate('Claude Trotter 3-step (t=1)', 'Suzuki-2 Trotter', 'driven, t=1.0', int(t1_3['transpile']['fake_marrakesh']['compiled_count_2q']), int(t1_3['transpile']['fake_marrakesh']['depth']), float(t1_3['fidelity_min']), float(t1_3['energy_error_max']), 'higher-accuracy Trotter upgrade'),
        Candidate('Claude Trotter 3-step (t=2)', 'Suzuki-2 Trotter', 'driven, t=2.0', int(t2_3['transpile']['fake_marrakesh']['compiled_count_2q']), int(t2_3['transpile']['fake_marrakesh']['depth']), float(t2_3['fidelity_min']), float(t2_3['energy_error_max']), 'longer-horizon survivor'),
    ]

    tmp_generated = output_pdf.with_suffix('.generated.tmp.pdf')
    tmp_claude = output_pdf.with_suffix('.claude.tmp.pdf')
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(tmp_generated) as pdf:
        _render_manifest_page(pdf, candidates)
        _render_table_page(pdf, candidates)
        _render_cost_page(pdf, candidates)
        _render_claude_deltae_page(pdf, dict(claude_t1.get('settings', {})), dict(claude_t2.get('settings', {})), t1_2, t1_3, t2_3)
        _render_ours_page(pdf, ours, compile_locked)
        _render_hesitancy_page(pdf)

    # extract the existing Claude fidelity + total-energy page from the broader report
    src_reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    # pages: 1 manifest, 2 summary, 3 table, 4 cost, 5 Claude fidelity+energy in the combined report
    writer.add_page(src_reader.pages[4])
    with tmp_claude.open('wb') as fh:
        writer.write(fh)

    final = PdfWriter()
    gen_reader = PdfReader(str(tmp_generated))
    claude_reader = PdfReader(str(tmp_claude))
    final.add_page(gen_reader.pages[0])
    final.add_page(gen_reader.pages[1])
    final.add_page(gen_reader.pages[2])
    final.add_page(claude_reader.pages[0])
    final.add_page(gen_reader.pages[3])
    final.add_page(gen_reader.pages[4])
    final.add_page(gen_reader.pages[5])
    with output_pdf.open('wb') as fh:
        final.write(fh)

    tmp_generated.unlink(missing_ok=True)
    tmp_claude.unlink(missing_ok=True)
    return output_pdf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build concise best-driven-QPU PDF with energy, fidelity, and |ΔE| plots.')
    p.add_argument('--claude-t1-json', type=Path, default=DEFAULT_CLAUDE_T1)
    p.add_argument('--claude-t2-json', type=Path, default=DEFAULT_CLAUDE_T2)
    p.add_argument('--ours-json', type=Path, default=DEFAULT_OURS)
    p.add_argument('--compile-json', type=Path, default=DEFAULT_COMPILE)
    p.add_argument('--source-pdf', type=Path, default=DEFAULT_SOURCE_PDF)
    p.add_argument('--output-pdf', type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = build_report(
        claude_t1_json=args.claude_t1_json,
        claude_t2_json=args.claude_t2_json,
        ours_json=args.ours_json,
        compile_json=args.compile_json,
        source_pdf=args.source_pdf,
        output_pdf=args.output_pdf,
    )
    print(f'Wrote PDF: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
