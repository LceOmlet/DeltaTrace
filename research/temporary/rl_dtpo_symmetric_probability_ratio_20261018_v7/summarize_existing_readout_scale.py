"""Summarize native JSON readout reports already printed by the accepted owner.
No training metrics parser, model call, hook or credit computation is added.
"""
import json,time,math,hashlib
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
run=R/'runs/textcraft-dtpo-symmetric-probability-ratio-v7-8gpu-200-20261018-v7'
out=R/'receipts/dtpo-symmetric-probability-ratio-20261018-v7'
bindings=json.loads((run/'runtime-overrides/actual-worker-imports-after-first-update-v1.json').read_bytes())
logs=Path('/tmp/ray/session_2026-10-18_16-31-41_727574_2854659/logs')
reports={}
sources=[]
for b in bindings['results']:
    files=list(logs.glob('worker-*-%s.out'%b['pid']))
    assert len(files)==1
    f=files[0];raw=f.read_bytes()
    sources.append(dict(rank=b['rank'],pid=b['pid'],path=str(f),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()))
    rr=[]
    for n,line in enumerate(raw.decode(errors='replace').splitlines(),1):
        p=line.find('[DeltaTrace readout] ')
        if p>=0:
            d=json.loads(line[p+len('[DeltaTrace readout] '):])
            assert d['credit_transform']=='per_event_symmetric_probability_ratio_then_credit_sum'
            assert d['source_credit_normalization'] is None
            assert d['credit_variance_normalization'] is False
            rr.append(dict(line=n,report=d))
    reports[str(b['rank'])]=rr
m=min(3, min(len(v) for v in reports.values()))
summary=[]
for j in range(m):
    ds=[v[j]['report'] for v in reports.values()]
    result=dict(native_readout_batch_ordinal=j+1,ranks=len(ds),requests=sum(len(d['traces']) for d in ds),
        unique_traj_uids=len({t['traj_uid'] for d in ds for t in d['traces']}),groups={})
    for k in ('self_target','prior_source','other_policy'):
        gs=[d['raw_advantage_groups'][k] for d in ds]
        n=sum(g['tokens'] for g in gs);ss=sum(g['sumsq'] for g in gs);su=sum(g['sum'] for g in gs)
        res=dict(tokens=n,sum=su,sumsq=ss,RMS=math.sqrt(ss/n) if n else None,mean=su/n if n else None)
        if k in ('self_target','prior_source'):
            field='target_self_tokens' if k=='self_target' else 'prior_source_tokens'
            tn=sum(t[field] for d in ds for t in d['traces'])
            g2=sum(t['reward']**2*t[field] for d in ds for t in d['traces'])
            res.update(trace_tokens=tn,g_broadcast_sumsq=g2,g_broadcast_RMS=math.sqrt(g2/tn) if tn else None)
            if tn==n and g2>0:res['native_A_RMS_over_g_broadcast_RMS']=math.sqrt(ss/g2)
        result['groups'][k]=res
    gg=result['groups'];n=gg['self_target']['tokens']+gg['prior_source']['tokens']
    ss=gg['self_target']['sumsq']+gg['prior_source']['sumsq']
    g2=gg['self_target']['g_broadcast_sumsq']+gg['prior_source']['g_broadcast_sumsq']
    result['combined_target_and_source']=dict(tokens=n,RMS=math.sqrt(ss/n),g_broadcast_RMS=math.sqrt(g2/n),A_RMS_over_g_broadcast_RMS=math.sqrt(ss/g2),
        target_fraction=gg['self_target']['tokens']/n,target_share_of_A_squared_norm=gg['self_target']['sumsq']/ss)
    summary.append(result)
receipt=dict(collected_at=time.time(),run=str(run),scope='Existing native readout reports; complete causal policy sources plus self targets for nonzero group coefficients, including native DP padding. Not final retained PPO minibatches; RMS is coefficient scale, not measured gradient ratio.',sources=sources,readout_reports=reports,summary=summary,model_calls=0,backward_calls=0,training_mutations=0)
path=out/'native-readout-advantage-scale-first-three-v1.json'
assert not path.exists(), 'Preserve the already collected native report receipt.'
path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),summary=summary,counts={k:len(v) for k,v in reports.items()})))
