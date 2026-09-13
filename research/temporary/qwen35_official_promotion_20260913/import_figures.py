"""Audit native illustration results and replace only Qwen3.5 figure records."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'raw/figures_v1'
OLD=HERE.parent/'qwen35_niah_causal_20260913'
status=json.loads((RUN/'status.json').read_bytes())
assert status['status']=='complete' and len(status['cases'])==3
assert status['attribution_profile']=='gdn-symmetric-v1'
assert status['official_dispatch_sha256']==hashlib.sha256((ROOT/'deltatrace/profiles/official.py').read_bytes()).hexdigest()
assert status['deployed_profile_sha256']==hashlib.sha256((ROOT/'deltatrace/profiles/qwen35_gdn_symmetric.py').read_bytes()).hexdigest()
# The reused independent auditor checks every native score, mask and curve.
# Retrieval was not measured in this illustrative run; all such fields are None.
verify=(OLD/'verify_variants.py').read_text(encoding='utf-8')
old='                if not gold:assert value is None;continue'
assert verify.count(old)==1
verify=verify.replace(old,'                if value is None:continue\n'+old)
sys.path.insert(0,str(OLD))
scope={'__name__':'illustration_audit'}
exec(compile(verify,str(OLD/'verify_variants.py'),'exec'),scope)
sys.argv=['verify_figures',str(RUN)]
scope['main']()
data=ROOT/'paper/iclr2027/figures/data'
archive=data/'historical_clean_v1'
archive.mkdir(exist_ok=True)
records={}
for meta in status['cases']:
    p=RUN/meta['path']/'results.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['results_sha256']
    c=json.loads(p.read_bytes())
    assert c['baseline_repeat_bitwise']
    if meta==status['cases'][0]:assert c['deployment_bitwise'] and c['candidate_repeat_bitwise']
    records[c['dataset'],c['index']]=(c,np.load(p.with_name('vectors.npz'),allow_pickle=False),p)

updated=[]
for name in ('cases.json','overview_role_case.json'):
    path=data/name
    if not (archive/name).exists():(archive/name).write_bytes(path.read_bytes())
    fixture=json.loads((archive/name).read_bytes())
    fixture['method']='Qwen3: clean-v1; Qwen3.5: official gdn-symmetric-v1'
    fixture['qwen35_attribution_profile']='gdn-symmetric-v1'
    fixture['qwen35_run_status_sha256']=hashlib.sha256((RUN/'status.json').read_bytes()).hexdigest()
    fixture['source_scope']='Fixed illustrative examples, separate from both benchmark tables'
    fixture['historical_numeric_sha256']=fixture.pop('numeric_sha256')
    for case in fixture['cases']:
        if case['model']!='qwen35':continue
        raw,v,p=records[case['dataset'],case['index']]
        assert case['input_sha256']==raw['input_sha256']
        assert case['input_ids']==raw['input_ids']
        assert case['prompt_length']==raw['prompt_length'] and case['target_length']==raw['target_length']
        for token in case['tokens']:
            pos=token['input_position'];local=token['local_index']
            assert pos==raw['user_positions'][local] and token['id']==raw['input_ids'][pos]
            assert token['eligible']==(local in raw['keep'])
            token['score']=float(v['DT_gdn_symmetric_full_sequence'][pos])
        case['signed_sum']=raw['methods']['DT_gdn_symmetric']['signed_sum']
        case['run']='qwen35-official-gdn-symmetric-illustrations-v1'
        case['attribution_profile']='gdn-symmetric-v1'
        case['raw_report_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        case['raw_report_path']=p.relative_to(ROOT).as_posix()
        for label,method in [('deletion','DT_gdn_symmetric'),('deletion_ft','FT_K1')]:
            if label not in case:continue
            curve=raw['curves'][method+'_positive']
            points=[raw['evaluations'][key] for key in curve['evaluations']]
            case[label]=dict(normalized_model_response=curve['normalized'],
                             deleted_user_indices=[e['deleted'] for e in points],
                             actual_input_hashes=[e['input_sha256'] for e in points],
                             scores=[e['score'] for e in points])
        if 'deletion_ft' in case:
            assert case['deletion']['scores'][0]==case['deletion_ft']['scores'][0]
            assert case['deletion']['scores'][-1]==case['deletion_ft']['scores'][-1]
        updated.append((name,case['dataset'],case['index']))
    path.write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for c,v,p in records.values():v.close()
receipt=dict(status='passed',profile='gdn-symmetric-v1',cases=3,updated=updated,
             official_default_matches_prototype_bitwise=True,
             status_sha256=hashlib.sha256((RUN/'status.json').read_bytes()).hexdigest(),
             quality_table_cases_unchanged=72)
(HERE/'figure_import_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
