"""CPU verification of saved row geometry, witnesses and the stated limits."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_softmax_operator_geometry_20260909_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
assert r['status']=='actual_saved_rows_CPU_geometry_and_conditional_diagnostic_complete'
assert receipt['proc_exists'] is False and p==r['protocol']==read(A/'dt_softmax_operator_geometry_protocol_20260909.json')
for n,v in receipt['files'].items():assert sha(D/n)==v['sha256'] and (D/n).stat().st_size==v['bytes']
for n,v in p['files_sha256'].items():assert sha(D/n)==v
assert sha(D/'rows.npz')==r['vectors_sha256']
for k in ('GPU_calls','model_calls','DT_calls','scorer_calls','FT_calls'):assert r[k]==0
assert [(x['query'],x['head']) for x in r['rows']]==[(q,h) for q in p['queries'] for h in range(16)]
z=np.load(D/'rows.npz',allow_pickle=False)
soft=lambda a:np.exp(a-np.max(a))/np.exp(a-np.max(a)).sum()
close=lambda a,b:np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
bad=[];max_secant=0.;max_zero=0.
for row in r['rows']:
    prefix=f"q{row['query']}_h{row['head']}";z0=z[prefix+'_z0'];z1=z[prefix+'_z1']
    p0=soft(z0);p1=soft(z1);s=z1-z0;d=p1-p0;Dvalue=d@s
    g=z[prefix+'_g'];v=p1*(s-p1@s);variance=s@v
    for name,gvec in [('witness',z[prefix+'_witness']),('actual_g',g)]:
        base=p1*(gvec-p1@gvec);gs=base@s;gd=gvec@d
        native=gvec@base
        # Scalar quadratic identities independent of a materialized matrix.
        supported=native+gd*(gd-gs)/Dvalue
        sym=native-gs*gs/variance+gd*gd/Dvalue
        key=name+'_quadratic';close(native,row[key]['native']);close(supported,row[key]['supported']);close(sym,row[key]['symmetric'])
        assert sym>=-1e-10
    max_secant=max(max_secant,*map(abs,row['secant_residual'].values()))
    max_zero=max(max_zero,*map(abs,row['zero_sum'].values()))
    if row['supported_congruence_min_eigenvalue']<-1e-10:
        assert row['witness_quadratic']['supported']<0
        bad.append({k:row[k] for k in ('query','head','supported_congruence_min_eigenvalue','witness_quadratic','actual_g_quadratic')})
    # No empirical/native-effect claim: these are ideal FP64 softmax effects.
    for step in ('3','10'):
        actual=g@(soft(z[prefix+'_zclean'])-soft(z[prefix+'_z'+step]))
        close(actual,row['conditions'][step]['ideal_softmax_effect_at_saved_V0'])
assert len(bad)==1 and bad[0]['query']==547 and bad[0]['head']==13
summary={'status':'CPU_actual_row_geometry_audited_not_native_or_whole_quality','results_sha256':sha(D/'results.json'),
    'protocol_sha256':sha(D/'protocol.json'),'receipt_sha256':sha(D/'terminal_receipt.json'),'vectors_sha256':sha(D/'rows.npz'),
    'analyzer_sha256':sha(__file__),'rows':32,'model_DT_FA_FLA_scorer_calls':0,'wall_seconds':r['seconds'],
    'indefinite_supported_rows':bad,'negative_actual_seed_quadratic_count':sum(x['actual_g_quadratic']['supported']<-1e-10 for x in r['rows']),
    'max_ideal_secant_residual':max_secant,'max_ideal_row_sum':max_zero,'queries':r['query_summaries'],
    'decision':'Supported operator can violate PSD even on an actual saved row, but the actual production seed direction does not demonstrate that violation. This is not established as the cause of its MAS regression. Symmetric correction has mixed per-head conditional results and no whole-model evidence; no kernel implementation or promotion from this diagnostic alone.',
    'limits':'Only32selected saved MH0rows, ideal FP64 softmax of native BF16operands; no native FA replay, all-head scan, new benchmark, new MAS or real-input attribution. The guaranteed PSD form is a structural property, not a quality guarantee.'}
target=A/'dt_softmax_operator_geometry_summary_20260909.json';target.write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps({'summary_sha256':sha(target),'indefinite':len(bad),'actual_seed_negative':summary['negative_actual_seed_quadratic_count'],'max_secant':max_secant,'max_zero':max_zero}))
