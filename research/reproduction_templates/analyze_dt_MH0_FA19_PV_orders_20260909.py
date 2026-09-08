"""Independent NumPy audit of two unchanged finite-FA PV-order calls.

No torch/private tensor loader, model, operator or metric execution. Checks
saved token contractions, actual source branches, signs, groups and transfer
algebra; private coefficient norm drift is retained as runtime evidence only.
"""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=lambda s:A/'snapshot'/str(s).lstrip('/')
FIELDS=('q_prediction','k_prediction','qk_prediction','v_prediction','captured_actual_effect',
        'core_prediction_minus_actual','core_at_BF16_seed','BF16_minus_stored_seed_actual_effect')
METHODS=('saved_current','same_LSE_control','reversed','diagnostic_average')
def close(a,b,tol=1e-7):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.max(np.abs(a-b),initial=0)<=tol,(a,b,tol)
def stats(x):
    x=np.asarray(x,dtype=np.float64);assert np.isfinite(x).all()
    return {'net':float(x.sum()),'positive':float(x[x>0].sum()),'negative':float(x[x<0].sum()),'absolute':float(np.abs(x).sum())}
def check_stats(saved,x):
    value=stats(x);assert saved.keys()==value.keys()
    for k,v in value.items():close(saved[k],v)
    return value

def audit(D):
    tick=time.perf_counter();D=Path(D);r=read(D/'results.json');p=read(D/'protocol.json');receipt=read(D/'terminal_receipt.json')
    v2=p.get('version')=='v2_only_native_unused_return_None_or_empty_guard'
    frozen='dt_MH0_FA19_PV_orders_protocol_20260909_v2.json' if v2 else 'dt_MH0_FA19_PV_orders_protocol_20260909.json'
    assert p==r['protocol']==read(A/frozen)
    assert receipt.get('proc_exists',receipt.get('pid_alive')) is False
    for name,item in receipt['files'].items():
        assert sha(D/name)==(item if isinstance(item,str) else item['sha256'])
        if isinstance(item,dict):assert (D/name).stat().st_size==item['bytes']
    for name,want in p['files_sha256'].items():assert sha(D/name)==want,name
    with zipfile.ZipFile(D/'review_bundle.zip') as z:
        for name in z.namelist():assert z.read(name)==(D/name).read_bytes()
    for item in p['protected_sources']:assert sha(local(item['path']))==item['sha256'],item['path']
    source=read(local(p['source_results_path']));sourcep=read(local(p['source_protocol_path']));boundary=read(local(p['boundary_results_path']))
    assert source['protocol']==sourcep and source['input']==boundary['cases']['morehopqa_0']['input']
    success=r['status']=='MH0_FA19_PV_orders_1publicFA2finiteFA_local_complete'
    out={'status':'independent_PV_order_local_audit_passed' if success else 'independent_PV_order_partial_failure_audit',
        'study_status':r['status'],'study_error':r.get('error'),'analyzer_sha256':sha(__file__),
        'protocol_sha256':sha(D/'protocol.json'),'results_sha256':sha(D/'results.json'),'receipt_sha256':sha(D/'terminal_receipt.json'),
        'actual_seconds':r['seconds'],'frozen_budget':p['budget'],'points':{},
        'actual_counts':{k:r[k] for k in ('native_FA_entered','native_FA_returned','finite_FA_entered','finite_FA_returned','model_calls','DT_calls','scorer_calls','FT_calls','generation_calls','compile_calls')}}
    c=out['actual_counts'];assert 0<=c['native_FA_returned']<=c['native_FA_entered']<=1 and 0<=c['finite_FA_returned']<=c['finite_FA_entered']<=2
    out['version']='v2' if v2 else 'v1'
    if v2:
        fail=p['v1_failure_provenance'];old=local(fail['directory']);oldr=read(old/'results.json')
        for name in ('results','protocol'):assert sha(old/(name+'.json'))==fail[name+'_sha256']
        assert sha(old/'terminal_receipt.json')==fail['receipt_sha256']
        assert oldr['status']=='failed' and oldr['native_FA_entered']==oldr['native_FA_returned']==1
        assert oldr['finite_FA_entered']==oldr['finite_FA_returned']==0
        close(oldr['seconds'],fail['seconds'],1e-12);out['prior_interface_failure']=fail
    elif not success and 'assert isinstance(unused,torch.Tensor)' in r.get('error',''):
        out['failure_scope']='V1 rejected the legitimate native None unused-probability return; one public FA completed and no finite FA ran. This interface guard error is not candidate/method failure.'
    for k in ('model_calls','DT_calls','scorer_calls','FT_calls','generation_calls','compile_calls'):assert c[k]==0
    if success:assert c['native_FA_returned']==1 and c['finite_FA_returned']==2
    assert r['kernel_phase_accounting']['phases_from_successfully_enqueued_calls']==3*sum(x['activity'].get('calls_enqueued',0) for x in r['calls'])
    assert r['kernel_phase_accounting']['phases_inside_unreturned_calls']==('unknown' if c['finite_FA_entered']!=c['finite_FA_returned'] else 0)
    out['actual_counts']['kernel_phase_accounting']=r['kernel_phase_accounting']
    if 'public_FA_unused_return_numel' in r:assert r['public_FA_unused_return_numel']==0
    if v2 and 'public_FA_unused_return_kind' in r:
        assert r['public_FA_unused_return_kind'] in ('None','Tensor')
        out['native_unused_probability_return_kind']=r['public_FA_unused_return_kind']
    if success:assert [x['name'] for x in r['calls']]==['same_LSE_control','reversed']
    for x in r['calls']:
        for name,v in x['operand_layouts'].items():
            assert v['contiguous'] and v['device'].startswith('cuda')
            expected=[1,4,853,256] if name in ('k0','k1','v0') else [1,16,853,256]
            if name in ('lse0','lse1'):expected=[1,16,853]
            assert v['shape']==expected and v['dtype']==('torch.float32' if name in ('u','lse0','lse1') else 'torch.bfloat16')
        if x['status']=='returned':
            a=x['activity'];assert a['calls_attempted']==a['calls_enqueued']==1 and a['kernel_launches_per_call']==3
            assert a['query_heads']==16 and a['kv_heads']==4 and a['valid_lengths']==[853]
            assert not a['GQA_input_expansion'] and not a['global_attention_matrix']
    for name in ('public_B2_output_drift','endpoint_controls','same_LSE_control_vs_saved_coefficient_drift',
        'reversed_vs_control_coefficient_drift','private_coefficients','GPU_peak_allocated','GPU_peak_reserved'):
        if name in r:out[name]=r[name]
    out['private_coefficient_scope']='Coefficient values and norm drift are runtime-produced evidence with a private artifact hash. This NumPy audit independently verifies projected token effects, not private .pt contents.'
    if not (D/'signed_token_contrasts.npz').exists():
        assert not success;out['partial_points_runtime_only']=r['points'];return out
    assert sha(D/'signed_token_contrasts.npz')==r['signed_token_contrasts']['sha256']
    out['vectors_sha256']=sha(D/'signed_token_contrasts.npz');z=np.load(D/'signed_token_contrasts.npz',allow_pickle=False)
    expected_keys={s+'_'+m+'_'+f for s in ('3','10','20','B2') for m in METHODS for f in FIELDS}
    if success:assert set(z.files)==expected_keys
    f=boundary['input_freeze_before_model_load']['morehopqa_0'];ids=np.asarray(f['input_ids'],dtype=np.int64)
    assert hashlib.sha256(ids.tobytes()).hexdigest()==p['input_sha256'];P=source['input']['prompt_length'];T=len(ids);keep=set(source['input']['keep'])
    for step,row in r['points'].items():
        if step+'_saved_current_q_prediction' not in z:continue
        assert step in ('3','10','20','B2');si='20' if step=='B2' else step
        receipt=p['frozen_input_receipts'][si];assert receipt==row['input_receipt']==boundary['cases']['morehopqa_0']['points'][si]['input_receipt']
        deleted=set(receipt['deleted_positions']);assert deleted<=keep
        altered=ids.copy();altered[sorted(deleted)]=f['target_ids'][-1]
        assert hashlib.sha256(altered.tobytes()).hexdigest()==receipt['input_sha256']
        ix={'deleted':sorted(deleted),'kept':sorted(keep-deleted),'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        assert sorted(t for positions in ix.values() for t in positions)==list(range(T))
        fields={m:{name:z[step+'_'+m+'_'+name] for name in FIELDS} for m in METHODS}
        result={'actual_pair':row['actual_pair'],'input_receipt':receipt,'methods':{}}
        ledger=source['conditional_ledgers'][step]['replayed_9term_ledger'];branch=ledger['branch_contractions']
        for name,k in [('q_prediction','q'),('k_prediction','k'),('v_prediction','v'),('captured_actual_effect','attention_output')]:close(fields['saved_current'][name].sum(),branch[k])
        close(row['original_saved_core'],ledger['terms']['finite_FA_core_including_seed_cast'])
        close(fields['saved_current']['core_prediction_minus_actual'].sum(),row['original_saved_core'])
        for method,v in fields.items():
            for a in v.values():assert a.shape==(1,T) and a.dtype==np.float64 and np.isfinite(a).all()
            close(v['q_prediction']+v['k_prediction'],v['qk_prediction'])
            close(v['qk_prediction']+v['v_prediction']-v['captured_actual_effect'],v['core_prediction_minus_actual'])
            close(v['core_prediction_minus_actual']-v['core_at_BF16_seed'],v['BF16_minus_stored_seed_actual_effect'])
            close(v['captured_actual_effect'],fields['saved_current']['captured_actual_effect'])
            close(v['BF16_minus_stored_seed_actual_effect'],fields['saved_current']['BF16_minus_stored_seed_actual_effect'])
            result['methods'][method]={'fields':{n:check_stats(row['methods'][method]['fields'][n],a) for n,a in v.items()},'groups':{}}
            for group,pos in ix.items():
                g=row['methods'][method]['groups'][group];assert g['count']==len(pos)
                result['methods'][method]['groups'][group]={'count':len(pos),'fields':{n:check_stats(g['fields'][n],a[0,pos]) for n,a in v.items()}}
            for n,a in v.items():close(sum(g['fields'][n]['net'] for g in result['methods'][method]['groups'].values()),a.sum())
        for name in FIELDS:close(fields['diagnostic_average'][name],(fields['same_LSE_control'][name]+fields['reversed'][name])*.5)
        selections=('qk_prediction','v_prediction','core_prediction_minus_actual')
        result['same_LSE_control_minus_saved_numerical_transfer']={n:check_stats(row['same_LSE_control_minus_saved_numerical_transfer'][n],fields['same_LSE_control'][n]-fields['saved_current'][n]) for n in selections}
        for method in ('reversed','diagnostic_average'):
            result[method+'_minus_same_LSE_control']={n:check_stats(row[method+'_minus_same_LSE_control'][n],fields[method][n]-fields['same_LSE_control'][n]) for n in selections}
        out['points'][step]=result
    out['compact_errors']={s:{m:v['methods'][m]['fields']['core_prediction_minus_actual']['net'] for m in METHODS} for s,v in out['points'].items()}
    out['sign_convention']='prediction minus actual; signed cancellation is preserved. Actual A coordinates are evaluation data, not candidate inputs.'
    out['decision_scope']='Only a local FA19 operator comparison. Adverse early/mid complete-core results can reject these concrete endpoint-order candidates for the current followup; they cannot disprove all endpoint allocations. Small local improvement alone does not establish changed whole-input attribution or original MAS/RISE, because downstream coefficients and compensations also change. No further GPU or candidate is authorized by this audit.'
    out['audit_seconds']=time.perf_counter()-tick
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',nargs='?',default=str(A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_FA19_PV_orders_20260909_v2'));ap.add_argument('--output',type=Path);args=ap.parse_args()
    result=audit(args.directory);target=args.output or A/('dt_MH0_FA19_PV_orders_summary_20260909_'+result['version']+'.json');target.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'output':str(target),'sha256':sha(target),'status':result['status'],'errors':result.get('compact_errors')}))
