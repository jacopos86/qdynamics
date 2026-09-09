"""Verify every results page, common estimator accounting, and source-bound metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def verify(root):
    import pdfplumber
    from pipelines.reporting.build_paper_ii_new_route_results import result_pages, estimator_count, number, tex
    data=json.loads((root/'results.json').read_text())
    build=json.loads((root/'pdf_build_receipt.json').read_text());checks=[]
    def check(name,value):
        if not value:
            raise ValueError(name)
        checks.append(name)
    for name,key in [('results.json','results_sha256'),('paper_ii_new_route_results.pdf','pdf_sha256'),
                     ('paper_ii_new_route_results.tex','tex_sha256')]:
        check('digest_'+name,hashlib.sha256((root/name).read_bytes()).hexdigest()==build[key])
    check('calibration',data['calibration_status']=='OK')
    check('comparison',data['comparability']['status']=='PASS')
    pages=result_pages(data);basis='reuse_adjusted'
    check('authoritative_reuse_accounting',build['estimator_basis']==basis)
    source=(root/'paper_ii_new_route_results.tex').read_text().split(r'\clearpage')
    check('source_page_count',len(source)==len(pages)==len(build['pages']))
    for arm in data['records']:
        tag=arm['id']
        check(tag+'_complete_t20',len(arm['times'])==501 and arm['times'][0]==0 and arm['times'][-1]==20)
        check(tag+'_work_ledger',arm['work']['logical_total_estimator_work_status']=='exact')
        check(tag+'_state_norm',arm['validation']['max_state_norm_error']<1e-12)
        check(tag+'_terminal_reconstruction',arm['terminal_cost']['terminal_reconstruction_parity']['passed'])
        for key,metric in [('energy','mean_abs_energy_error'),('doublon','mean_abs_doublon_error'),('imbalance','mean_abs_charge_imbalance_error')]:
            values=arm['series'][key];exact=data['exact']['series'][key]
            check(tag+'_'+key+'_finite',len(values)==501 and all(map(math.isfinite,values)))
            mean=sum(abs(a-b) for a,b in zip(values,exact))/501
            check(tag+'_'+key+'_mean',math.isclose(mean,arm['metrics'][metric],rel_tol=1e-11,abs_tol=1e-15))
    with pdfplumber.open(root/'paper_ii_new_route_results.pdf') as pdf:
        check('one_page_per_completed_run',len(pdf.pages)==len(pages))
        for index,(page,records) in enumerate(zip(pdf.pages,pages)):
            text=page.extract_text(x_tolerance=2)
            check(f'page_{index+1}_plots_and_tables',all(s in text for s in ['Figure 1.','Table 1.','Table 2.']))
            check(f'page_{index+1}_no_admin_appendix','Evidence and campaign status' not in text)
            check(f'page_{index+1}_text_bounds',all(c['x0']>=-1 and c['x1']<=page.width+1 and c['top']>=-1 and c['bottom']<=page.height+1 for c in page.chars))
            for arm in records:
                label=arm.get('table_label',arm['label'])
                check(f'page_{index+1}_tables_'+arm['id'],text.count(label)>=2)
                check(f'page_{index+1}_plot_'+arm['id'],arm['label'] in text)
                rows=[line.split(' & ') for line in source[index].splitlines() if line.startswith(tex(label)+' & ')]
                expected=number(estimator_count(arm),1e8)
                check(f'page_{index+1}_equal_estimator_columns_'+arm['id'],len(rows)==2 and rows[0][9]==rows[1][1]==expected)
            check(f'page_{index+1}_exact',text.count('Exact')>=3)
    check('no_tex_overflow',not re.search(r'Overfull|undefined|LaTeX Warning',(root/'paper_ii_new_route_results.log').read_text()))
    check('notes_outside_pdf',(root/'provenance_notes.md').is_file())
    receipt=dict(status='PASS',check_count=len(checks),checks=checks,pdf_sha256=build['pdf_sha256'],
        results_sha256=build['results_sha256'],estimator_basis=basis,page_count=len(pages),
        layout_validation='PDF text positions and TeX log; no viewer or image rendering used.')
    (root/'verification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root)
