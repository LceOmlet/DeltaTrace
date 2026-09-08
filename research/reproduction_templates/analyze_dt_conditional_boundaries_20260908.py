"""Recompute original metrics and all saved conditional-boundary bookkeeping."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_conditional_boundaries_20260908_v1'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((D/'results.json').read_bytes());receipt=json.loads((D/'terminal_receipt.json').read_bytes())
assert r['status']=='conditional_boundaries_complete' and not receipt['proc_exists']
for n,e in receipt['files'].items():assert sha((D/n).read_bytes())==e['sha256'],n
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    for n in z.namelist():
        assert '/' not in n and '\\' not in n
        raw=z.read(n)
        if (D/n).exists():assert (D/n).read_bytes()==raw
        else:(D/n).write_bytes(raw)
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
p=r['protocol'];v=np.load(D/'vectors.npz');out={}
auc=lambda x:float((x.sum()-.5*(x[0]+x[-1]))/(len(x)-1))
for key,row in r['cases'].items():
    full=v[key+'_DT_full'];score=v[key+'_DT_evaluated'];assert np.array_equal(full[:len(score)].astype(np.float32),score)
    keep=row['input']['keep'];curve=row['curve'];G=np.asarray(curve['scores']);S=curve['attr_sum']
    assert abs(float(score[keep].sum(dtype=np.float32))-S)<1e-5
    response=np.minimum.accumulate(np.clip((G-G[-1])/abs(G[0]-G[-1]),0,1))
    rho=[1.]
    for j in range(1,21):
        prev=set(curve['input_receipts'][j-1]['deleted_positions']);cur=set(curve['input_receipts'][j]['deleted_positions'])
        group=cur-prev;assert len(group)>0 and group<=set(keep)
        rho.append(rho[-1]-float(score[sorted(group)].sum(dtype=np.float32))/S)
    rho=np.asarray(rho)
    # FP32 reductions may use different reduction trees; raw saved density is
    # the actual original implementation. Both must agree at practical precision.
    assert np.max(np.abs(rho-curve['density']))<1e-6
    actual_rho=np.asarray(curve['density']);ap=np.abs(response-actual_rho)
    corrected=np.clip(response+ap,0,1);corrected=(corrected-corrected.min())/(corrected.max()-corrected.min())
    metrics=[auc(response),auc(corrected),auc(response+ap)]
    assert np.max(np.abs(np.asarray(metrics)-curve['return_metrics']))<1e-12
    points={}
    for step in [1,10,20]:
        a=row['B1_boundary_contractions'][str(step)];d=row['decomposition'][str(step)]
        B=float(score[curve['input_receipts'][step]['deleted_positions']].astype(np.float64).sum())
        actual=float(G[0]-G[step]);assert abs(B-d['DT_predicted_effect'])<1e-10
        e=[a[str(i+1)]-a[str(i)] for i in range(32)]
        assert np.max(np.abs(np.asarray(e)-list(d['decoder_errors'].values())))<1e-9
        assert abs(sum(d['terms'].values())-(actual-B))<1e-9
        blocks={str(i):row['DT']['layers'][str(i)]['block_type'] for i in range(32)}
        types={t:sum(e[i] for i in range(32) if blocks[str(i)]==t) for t in set(blocks.values())}
        endpoint=row['B2_endpoint_boundary_effects']
        b2_errors={str(i):endpoint[str(i+1)]-endpoint[str(i)] for i in range(32)}
        denominator=float(G[0]-G[-1]);denom_term=B*(1/denominator-1/S)
        identity=(actual-B)/denominator+denom_term
        rho_from_B=1-B/S;raw_response=1-actual/denominator
        assert abs(identity-(rho_from_B-raw_response))<1e-12
        points[str(step)]={**d,'decoder_type_sums':types,'largest_absolute_decoder':max(range(32),key=lambda i:abs(e[i])),
            'decoder0_B2_endpoint_residual':b2_errors['0'],'decoder0_B1_allEOS_residual':row['decomposition']['20']['decoder_errors']['0'],
            'MAS_denominator_term':denom_term,'raw_density_minus_response':identity}
    out[key]={'metrics':{'RISE':metrics[0],'MAS':metrics[1]},'points':points,
        'B2_root_residual':row['DT']['relative_residual'],'B2_root_effect':row['DT']['root_effect'],
        'B1_clean_minus_EOS':float(G[0]-G[-1]),'coefficient_CPU_bytes':row['coefficient_CPU_bytes'],
        'observer_equal':row.get('passive_observer_vectors_equal'),
        'observer_relative_L2':row.get('passive_observer_vector_relative_L2')}
summary={'status':'conditional_boundary_bookkeeping_and_original_metrics_verified','receipt':receipt,'cases':out,
    'budget':p['budget'],'job_seconds':r['job_seconds'],'FT_modified':False,'finite_rules_modified':False,
    'limits':['Actual production coefficients and original B1 activations observed passively; source hashes fixed.',
        'Full coefficient/activation tensors were transient CPU state; saved contractions can be algebraically audited but cannot be independently regenerated offline from retained raw tensors.',
        'Decoder groups include MLP,norm,residual and mixer; decoder0 prominence does not alone identify FLA recurrence.',
        'Default-precision B1/B2 endpoint controls and scoring precision retained; intermediate errors not automatically corrected by subtracting endpoint errors.',
        'Two previously used cases, fixed diagnostic steps; no population or candidate improvement claim.'],
    'decision':'Focus next diagnostic on decoder0 internals in the same two cases and already frozen deletion inputs. Do not switch seed or globally alter FLA based on coarse results.'}
(A/'dt_conditional_boundaries_summary_20260908.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':summary['status'],'cases':{k:{'metrics':x['metrics'],'step10_total_error':x['points']['10']['actual_minus_predicted'],
 'step10_decoder0':x['points']['10']['decoder_errors']['0'],'decoder0_B1_EOS':x['points']['10']['decoder0_B1_allEOS_residual'],
 'decoder0_B2_EOS':x['points']['10']['decoder0_B2_endpoint_residual']} for k,x in out.items()},'seconds':r['job_seconds']}))
