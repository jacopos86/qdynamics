"""Build the working Paper II results PDF from validated, source-bound data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def tex(value):
    result = str(value)
    for before, after in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:
        result = result.replace(before, after)
    return result


def number(value, unit=1):
    return f'{value / unit:.3g}'


def make_figure(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8, 'axes.linewidth':0.65, 'pdf.fonttype':42,
                         'font.family':'DejaVu Sans', 'axes.labelsize':9})
    legend_rows = (len(data['records']) + 4) // 4
    fig, axes = plt.subplots(1,3,figsize=(7.8,2.55+0.23*(legend_rows-1)))
    series_keys = ('energy','doublon','imbalance')
    # Draw the reference last and use markers so coincident curves remain identifiable.
    for record in data['records']:
        for i,key in enumerate(series_keys):
            axes[i].plot(record['times'],record['series'][key],label=record['label'],
                color=record['color'],linestyle=record['linestyle'],linewidth=1.1,
                zorder=3 if record['id'].startswith('new_') else 2)
    for i,key in enumerate(series_keys):
        axes[i].plot(data['exact']['times'],data['exact']['series'][key],color='#111111',
            linestyle=':',linewidth=0.9,marker='o',markersize=2.1,markevery=40,
            markerfacecolor='white',markeredgewidth=0.55,label='Exact',zorder=4)
        axes[i].set_xlim(0,20)
        axes[i].set_xlabel(r'Time $t$')
        axes[i].set_title((r'(a) Total energy $E(t)$',r'(b) Mean doublon $D(t)$',
                           r'(c) Charge imbalance $\Delta n(t)$')[i],pad=7)
        axes[i].grid(alpha=.2,linewidth=.5)
        axes[i].tick_params(labelsize=7.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.005),
               frameon=False,fontsize=7.5,columnspacing=1.15,handlelength=2.9)
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.76-.05*(legend_rows-1),wspace=.42)
    fig.savefig(output)
    plt.close(fig)


def estimator_count(record):
    """Authoritative Paper II reuse-adjusted cost, shared by both tables."""
    return record['reuse_adjusted_work']


def result_pages(data):
    """One new run per page, with the same matched baseline pair."""
    baselines=[r for r in data['records'] if r['id'] in ('paper_ii_apm','avqds')]
    if len(baselines) != 2:
        raise ValueError('Both original baseline records are required')
    seen=set(); pages=[]
    for record in data['records']:
        if not record['id'].startswith('new_'):
            continue
        if record['raw_sha256'] in seen:
            raise ValueError('Duplicate completed trajectory in page inputs')
        if record['validation']['horizon'] != 20 or record['times'][-1] != 20:
            raise ValueError('Only complete t=20 results receive pages')
        if record['run_lock']['comparison_fingerprint'] != baselines[0]['run_lock']['comparison_fingerprint']:
            raise ValueError('New run requires matching baselines before automatic comparison')
        seen.add(record['raw_sha256'])
        pages.append([record,*baselines])
    if not pages:
        raise ValueError('No complete new-route result')
    return pages


PREAMBLE=[r'\documentclass[10pt,letterpaper]{article}',
        r'\usepackage[margin=0.55in]{geometry}',r'\usepackage{graphicx,booktabs,array,amsmath,xcolor}',
        r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern}',r'\usepackage[hidelinks]{hyperref}',
        r'\usepackage{fancyhdr}',r'\pagestyle{fancy}\fancyhf{}',
        r'\fancyfoot[L]{\footnotesize Paper II --- working results}',
        r'\fancyfoot[R]{\footnotesize\thepage}',r'\renewcommand{\headrulewidth}{0pt}',
        r'\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}',r'\begin{document}']



def page_content(data, records, figure_name):
    cfg=data['config']
    labels={r['id']:r.get('table_label',r['label']) for r in records}
    ordered=sorted(records,key=lambda r:(not r['id'].startswith('new_'),r['id']=='avqds'))
    budgets='; '.join(f"{labels[r['id']]}: {r['work_budget']:,}" for r in records if r['id'].startswith('new_'))
    lines=[
        r'{\Large\bfseries New-route APM: working results}\hfill{\small 9 September 2026}\par',
        r'\textbf{Weak--fast drive; state-motion controller; $t\in[0,20]$.}\par',
        r'{\small Hubbard--Holstein dimer, open boundary, $L=2$, hopping $1$, $U=4$, '
        r'$g_{\rm ep}=0.5$, $\omega_0=1$, $n_{\rm ph}\le1$ (6 qubits). Staggered density drive: '
        r'$A=0.6$, $\omega=3$, $\bar t=1$, $\phi=0$. Common seed and 501 reporting times, $\Delta t=0.04$.}\par',
        r'\vspace{-2pt}\includegraphics[width=\linewidth]{'+figure_name+r'}\par',
        r'{\footnotesize\textbf{Figure 1.} The three Paper II observables. Every curve uses the same seed, drive and reporting grid. '
        r'$D(t)$ is the mean site double occupancy; $\Delta n(t)=\langle n_1-n_2\rangle_t$. '
        r'Coincident trajectories can obscure one another; numerical errors are reported below.}\par',
        r'\vspace{3pt}\textbf{Table 1. Accuracy and terminal resources (Paper II Table I columns).}',
        r'\begin{center}\scriptsize\setlength{\tabcolsep}{3.2pt}',
        r'\begin{tabular}{lrrrrrrrrrr}\toprule',
        r'Method & $\tau_m$ & $\overline{|\Delta E|}$ & $\overline{|\Delta D|}$ & $\overline{|\Delta n|}$ '
        r'& $N_\theta$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $\mathcal E$ & $N_{\rm step}$ \\',
        r' & & $(10^{-4})$ & $(10^{-4})$ & $(10^{-4})$ & & & & & $(10^8)$ & $(10^3)$ \\ \midrule']
    for record in ordered:
        m=record['metrics']; c=record['terminal_cost']
        cells=[tex(labels[record['id']]),r'$3\times10^{-6}$',
            *[number(m[k],1e-4) for k in ('mean_abs_energy_error','mean_abs_doublon_error','mean_abs_charge_imbalance_error')],
            str(record['n_theta']),*[str(c[k]) for k in ('N2q','D2q','Dc')],
            number(estimator_count(record),1e8),number(record['steps'],1e3)]
        lines.append(' & '.join(cells)+r' \\')
    lines.extend([r'Exact & --- & 0 & 0 & 0 & --- & --- & --- & --- & --- & --- \\',
        r'\bottomrule\end{tabular}\end{center}',
        r'{\footnotesize Errors are mean absolute deviations from the independent exact reference over all 501 times. '
        r'Terminal resources use FakeMarrakesh, optimization level 1 and transpiler seed 7. '
        r'$\mathcal E$ is the '+'reuse-adjusted'+r' logical scalar-estimator count used in both tables.}\par',
        r'\vspace{4pt}\textbf{Table 2. Structural actions (Paper II Table II columns).}',
        r'\begin{center}\scriptsize\setlength{\tabcolsep}{4.3pt}',
        r'\begin{tabular}{lrrrrrrr}\toprule',
        r'Method & $\mathcal E$ & ins. & del. & exch. & '
        r'\shortstack{del. then\\re-ins.} & \shortstack{ins. then\\del.} & \shortstack{finalists\\tested/failed} \\',
        r' & $(10^8)$ & & & & & & \\ \midrule'])
    for record in ordered:
        motion=record['motion']; gates=record['deletion_gates']
        trials='---'
        if gates['applicable']:
            prefix=r'$\geq$' if not gates['attempt_record_complete'] else ''
            trials=prefix+str(gates['deletion_bearing_finalist_attempt_count'])+'/'+str(gates['post_delete_gate_failure_count'])
        lines.append(' & '.join([tex(labels[record['id']]),number(estimator_count(record),1e8),
            *[str(motion[k]) for k in ('inserted','deleted','exchanges','reinserted','retired')],trials])+r' \\')
    lines.extend([r'Exact & --- & --- & --- & --- & --- & --- & --- \\',r'\bottomrule\end{tabular}\end{center}',
        r'{\footnotesize Generator-level support changes follow Paper II\textquotesingle s reported-support convention; '
        r'exchanges count committed delete--add patches across all selector rounds. '
        r'$\mathcal E$ uses the same accounting as Table 1. '
        r"$\geq$ marks incomplete finalist-attempt history; ``---'' means not applicable.}\par",
        r'\vspace{5pt}\textbf{New-route configuration.} {\small Labels identify search and gain evaluation. Ascending cardinality tiers; '
        r'at most one insertion and $\lfloor0.15n\rfloor$ deletions per patch, '
        r'Patch-evaluation budgets per decision: '+tex(budgets)+r'. All APM rows use compiled-$Q$ ranking. '
        r'Euler integration, ridge $10^{-6}$, pseudoinverse cutoff $10^{-10}$, state-motion bound $5\times10^{-5}$.}\par',
        r'{\small This is a working comparison of completed runs at the declared settings. '
        r'It does not establish an equal-accuracy resource advantage. }',
        ])
    return lines



def write_notes(data, root):
    cfg=data['config']; records=data['records']
    labels={r['id']:r.get('table_label',r['label']) for r in records}
    notes=['# Working-results provenance and campaign notes', '',
        'Agent reference only. Keep these notes outside the reader-facing PDF.', '',
        '## Completed results', '',
        '| Method | Mean infidelity | Final coordinates | Wall time (min) |',
        '| --- | ---: | ---: | ---: |']
    for record in records:
        wall='-'
        if record.get('measurement'):
            measurement=json.loads(Path(record['measurement']).read_text())
            wall=f"{measurement['wall_seconds']/60:.1f}"
        notes.append(f"| {labels[record['id']]} | {number(record['metrics']['mean_infidelity'])} | {record['n_theta']} | {wall} |")
    notes.extend(['| Exact | 0 | - | - |', '',
        'Local wall time includes concurrent diagnostic work; it is not a cross-host speedup comparison. '
        'Mean infidelity uses the saved statevectors and the shared exact reference.', '',
        '## CHTC variants awaiting validated results', '',
        'This list comes from the report input manifest, not a live scheduler query. '
        'Pending jobs supply no scientific values to the PDF.', '',
        '| Search | Gain evaluator | Budget | Job |', '| --- | --- | ---: | --- |'])
    for pending in cfg['pending']:
        notes.append(f"| {pending['search']} | {pending['gain']} | {pending['budget']} | {pending['cluster_id']} |")
    notes.extend(['', '## Validation and provenance', '',
        f"Calibration: {data['calibration_status']}. Comparison: {data['comparability']['status']}.",
        'Raw digests, completion records, exact estimator ledgers and terminal reconstruction were checked. '
        'The content-addressed exact reference uses exponential midpoint with 40 substeps per reporting interval; '
        'it remains outside controller decisions.', '',
        '| Method | Code revision | Raw SHA-256 |', '| --- | --- | --- |'])
    for record in records:
        notes.append(f"| {labels[record['id']]} | `{record['code_revision']}` | `{record['raw_sha256']}` |")
    notes.extend(['', f"Seed SHA-256: `{cfg['seed_sha256']}`.",
        f"Exact-reference data SHA-256: `{data['exact']['data_sha256']}`.", '',
        'Full input paths, hashes, metrics and validation records are in `results.json`, '
        '`calibration_receipt.json`, `comparability.json`, `verification_receipt.json` and `data/`. '
        'Refresh instructions: `agent_guidance/time-dynamics/paper-ii-new-route-working-results.md`.', ''])
    (root/'provenance_notes.md').write_text('\n'.join(notes))

def build(result_path):
    data=json.loads(result_path.read_text())
    root=result_path.parent
    if data['calibration_status'] != 'OK' or data['comparability']['status'] != 'PASS':
        raise ValueError('Validation must pass before automatic rendering')
    pages=result_pages(data)
    lines=list(PREAMBLE); page_records=[]
    for i,records in enumerate(pages,1):
        if i > 1:
            lines.append(r'\clearpage')
        figure_name=f'observables_{i:03d}.pdf'
        make_figure({**data,'records':records},root/figure_name)
        lines.extend(page_content(data,records,figure_name))
        page_records.append(dict(page=i,run_id=records[0]['id'],
            raw_sha256=records[0]['raw_sha256'],comparators=[r['id'] for r in records[1:]]))
    lines.append(r'\end{document}')
    output=root/'paper_ii_new_route_results.tex'
    output.write_text('\n'.join(lines)+'\n')
    subprocess.run(['tectonic','--keep-logs','--reruns','2',output.name],cwd=root,check=True)
    pdf=output.with_suffix('.pdf')
    write_notes(data,root)
    receipt=dict(schema='paper_ii_working_results_pdf_v2',generated_utc=datetime.now(timezone.utc).isoformat(),
        pdf=str(pdf.resolve()),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        tex_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimator_basis='reuse_adjusted',pages=page_records,
        completed_arms=[r['id'] for r in data['records']],comparison_reference=data['exact']['key'],
        visual_inspection='Not performed: unrequested PDF viewing is prohibited.')
    (root/'pdf_build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    build(parser.parse_args().results)
