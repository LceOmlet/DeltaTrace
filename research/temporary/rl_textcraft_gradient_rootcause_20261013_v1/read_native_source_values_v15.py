"""Read the bounded native endpoint evidence and compare its saved DT values."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import psutil
import torch
from counterfactual import reward_event_token_credit

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-source-delete-values-v15'
job=json.loads((DEST/'job.json').read_bytes())
state=json.loads((DEST/'result.json').read_bytes())
report=dict(unix=time.time(),job=job,state=state,rank_reports=[],points=[])
try:
    p=psutil.Process(job['pid']);report['alive']=p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:report['alive']=False
for rank in (0,1):
    path=DEST/f'rank{rank}.json'
    if path.exists():report['rank_reports'].append(dict(path=str(path),value=json.loads(path.read_bytes())))
for filename in ('stdout.log','stderr.log'):
    report[filename+'_tail']=(DEST/filename).read_text(errors='replace').splitlines()[-6:]
if state['phase']=='complete_native_source_values_not_repair':
    manifest=json.loads((DEST/'query-manifest.json').read_bytes())
    queries=torch.load(manifest['query_file'],map_location='cpu',weights_only=False)
    keyed={(q['UID'],q['original_response_slot']):q for q in queries}
    raw={}
    for block in report['rank_reports']:
        r=block['value'];assert r['scheduler_before']==r['scheduler_after']==80
        assert r['trainable_SHA256_before']==r['trainable_SHA256_after']
        for batch in r['batches']:
            for slot,q in enumerate(batch['queries']):
                key=(q['UID'],q['original_response_slot']);original=keyed[key]
                path=q['original_raw_file']
                if path not in raw:raw[path]=torch.load(path,map_location='cpu',weights_only=False)['rows']
                row=raw[path][q['original_raw_row']]
                assert row['traj_uid']==q['UID']
                assert row['target_offsets']==original['target_offsets']
                native=batch['passes']['single_EOS'];d=native['native_d'][slot]
                credit=reward_event_token_credit(torch.tensor([[[d]]],dtype=torch.float64),
                    torch.tensor([[row['reward']]],dtype=torch.float64),
                    torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool))
                V=float(credit.v_estimates[0,0]);A=float(credit.advantages[0,0])
                report['points'].append(dict(rank=r['rank'],**q,native_single_EOS_d=d,
                    native_single_EOS_V=V,native_single_EOS_raw_A=A,
                    saved_factual_target_logp=row['factual_target_logp'],
                    native_factual_target_logp=native['factual_joint_logp'][slot],
                    native_deleted_target_logp=native['reference_joint_logp'][slot],
                    factual_target_logp_drift=native['factual_joint_logp'][slot]-row['factual_target_logp'],
                    identity_pair_d=batch['passes']['identity']['native_d'][slot],
                    native_V_over_saved_V=V/q['saved_V'] if q['saved_V']!=0 else None))
    report['summaries_by_rank']={}
    for rank in (0,1):
        rows=[p for p in report['points'] if p['rank']==rank]
        report['summaries_by_rank'][str(rank)]=dict(points=len(rows),
            native_V_median=statistics.median(p['native_single_EOS_V'] for p in rows),
            saved_V_median=statistics.median(p['saved_V'] for p in rows),
            native_V_greater_than_saved=sum(p['native_single_EOS_V']>p['saved_V'] for p in rows),
            native_d_median=statistics.median(p['native_single_EOS_d'] for p in rows),
            saved_d_median=statistics.median(p['saved_DT_d'] for p in rows),
            native_V_over_saved_V_range=[min(p['native_V_over_saved_V'] for p in rows if p['native_V_over_saved_V'] is not None),
                                          max(p['native_V_over_saved_V'] for p in rows if p['native_V_over_saved_V'] is not None)],
            factual_logp_drift_range=[min(p['factual_target_logp_drift'] for p in rows),max(p['factual_target_logp_drift'] for p in rows)],
            identity_d_max_abs=max(abs(p['identity_pair_d']) for p in rows))
    report['limits']=[
        'Only the21 disputed finite source points are measured. They are not a representative token sample or a population-baseline proof.',
        'Both ranks read the same points; numerical repeats are not independent task evidence.',
        'Native single-EOS endpoint values retain the factual future Y/observations. This does not by itself establish an action-independent world-return baseline.',
        'Original versus current factual score drift and identity differences are retained. No new numerical tolerance is imposed.',
        'No native value replaces a training coefficient; there is no new full whitening, optimizer update, rollout or repair acceptance.']
assert not torch.cuda.is_initialized()
dest=OUT/f'native-source-values-v15-observation-{int(report["unix"])}.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
