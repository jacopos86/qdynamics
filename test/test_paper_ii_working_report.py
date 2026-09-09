from copy import deepcopy
import json
from pathlib import Path
import pytest
from pipelines.reporting.build_paper_ii_new_route_results import result_pages, page_content, estimator_count
from pipelines.reporting.update_paper_ii_working_results import add_once, validate_arrival, update

ROOT=Path(__file__).resolve().parents[1]


def record(identity,digest):
    return dict(id=identity,raw_sha256=digest,validation={'horizon':20},times=[0,20],
        run_lock={'comparison_fingerprint':'same'})


def test_page_append_preserves_baselines_and_order():
    data={'records':[record('paper_ii_apm','old'),record('avqds','avqds'),record('new_first','first')]}
    before=result_pages(data)
    data['records'].append(record('new_second','second'))
    pages=result_pages(data)
    assert len(pages)==2 and pages[0]==before[0]
    assert [r['id'] for r in pages[1]]==['new_second','paper_ii_apm','avqds']


def test_duplicate_receipt_cannot_append_page():
    arms=[record('new_first','same')]
    assert not add_once(arms,record('new_first','same'))
    assert not add_once(arms,record('new_alias','same'))
    assert len(arms)==1
    with pytest.raises(ValueError,match='changed evidence'):
        add_once(arms,record('new_first','different'))


@pytest.mark.parametrize('change', ['partial','physics','duplicate'])
def test_invalid_result_cannot_be_rendered(change):
    data={'records':[record('paper_ii_apm','old'),record('avqds','avqds'),record('new_first','new')]}
    if change=='partial': data['records'][-1]['times'][-1]=1
    elif change=='physics': data['records'][-1]['run_lock']['comparison_fingerprint']='other'
    else: data['records'].append(record('new_duplicate','new'))
    with pytest.raises(ValueError): result_pages(data)


def test_both_tables_use_identical_old_apm_count():
    old=dict(record('paper_ii_apm','old'),label='APM - Paper II v1',work_budget=50000,
        metrics=dict(mean_abs_energy_error=0.000373,mean_abs_doublon_error=0.000200,
                     mean_abs_charge_imbalance_error=0.000365),
        terminal_cost=dict(N2q=164,D2q=128,Dc=442),n_theta=43,steps=497600,
        reuse_adjusted_work=526108803,work={'logical_total_estimator_work':682868732},
        motion=dict(inserted=43,deleted=30,exchanges=1,reinserted=10,retired=18),
        deletion_gates=dict(applicable=False))
    avqds={**old,**record('avqds','avqds'),'label':'AVQDS'}
    new={**old,**record('new_first','new'),'label':'APM - new'}
    data={'records':[old,avqds,new],'config':{}}
    rows=page_content(data,result_pages(data)[0],'test.pdf')
    old=[line.split(' & ') for line in rows if line.startswith('APM - Paper II v1 & ')]
    assert len(old)==2 and old[0][9]==old[1][1]=='5.26'


def test_unchanged_tick_preserves_pdf_bytes(tmp_path):
    inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps({'estimator_basis':'executed','arms':[]}))
    (tmp_path/'pdf_build_receipt.json').write_text('{}')
    pdf=tmp_path/'paper_ii_new_route_results.pdf';pdf.write_bytes(b'keep existing PDF')
    assert update(inputs,tmp_path,'unused')['status']=='UNCHANGED'
    assert pdf.read_bytes()==b'keep existing PDF'


def test_smoke_receipt_cannot_become_t20_page():
    package=Path('/Users/lordskyler/Holstein_test.worktrees/codex-paper-ii-t20-package-20260909/chtc/paper_ii_selector_recovery_smoke_w10000_339a1d05_20260909')
    if not (package/'input/manifest.json').exists():
        pytest.skip('Optional collected worker-smoke fixture is not present')
    manifest=json.loads((package/'input/manifest.json').read_text())
    arrivals=ROOT/'output/paper_ii_t20_successful_settings_local_20260909_2c00f5a4/chtc_smoke_arrival_20260909'
    receipt=next(arrivals.glob('*/*.archive.json'))
    with pytest.raises(ValueError,match='Not a complete t=20 action'):
        validate_arrival(receipt,package,manifest)


def test_bad_arrival_does_not_block_good_arrival(tmp_path, monkeypatch):
    from pipelines.reporting import update_paper_ii_working_results as updater
    inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps({'arms':[]}))
    (tmp_path/'pdf_build_receipt.json').write_text(json.dumps({'pages':[],'pdf_sha256':'old'}))
    def discover(cfg):
        cfg['arms'].extend([record('new_bad','bad'),record('new_good','good')])
        return ['new_bad','new_good'],[]
    seen=[]
    def publish(cfg,*args):
        seen.append([x['id'] for x in cfg['arms']])
        if cfg['arms'][-1]['id']=='new_bad': raise ValueError('bad science')
        receipt={'pages':[1],'pdf_sha256':'good'}
        (tmp_path/'pdf_build_receipt.json').write_text(json.dumps(receipt))
        return receipt,tmp_path
    monkeypatch.setattr(updater,'discover',discover)
    monkeypatch.setattr(updater,'publish',publish)
    result=updater.update(inputs,tmp_path,'unused')
    assert seen==[['new_bad'],['new_good']]
    assert result['added']==['new_good'] and len(result['failures'])==1
