"""Freeze real original-NI0 layer35 kernel validation and full operator costs."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=(A/'qwen_signed_secant_pv_rules.py').read_text()
needle="            mqn, mkn, mv = attention_layout("
assert base.count(needle)==1
probe="""            if ATTENTION_OBSERVER is not None:
                ATTENTION_OBSERVER(li, q0, kr0, vr0, q1, kr1, vr1, mout, raw0, raw1,
                                   q_product, k_product, mvr, scaling)
"""
instrumented='ATTENTION_OBSERVER=None\n'+base.replace(needle,probe+needle)
assert instrumented.removeprefix('ATTENTION_OBSERVER=None\n').replace(probe,'')==base
(A/'qwen_signed_secant_pv_operand_capture.py').write_text(instrumented,encoding='utf-8')
paired=(A/'qwen_signed_secant_paired_public_fa.py').read_text().replace('from qwen_signed_secant_pv_rules import','from qwen_signed_secant_pv_operand_capture import')
(A/'qwen_signed_secant_public_operand_capture.py').write_text(paired,encoding='utf-8')
old=(A/'public_fa_capture_probe_20260907.py').read_text()
driver=old[:old.index('\nsave()\ntry:\n')]
driver=driver.replace('from qwen_signed_secant_paired_public_fa import propagate_paired_secant as public_capture',
                      'from qwen_signed_secant_public_operand_capture import propagate_paired_secant as public_capture')
driver+='''
import gc,numpy as np
import qwen_signed_secant_pv_operand_capture as capture_module
from vendor_fa_finite_runtime import VendorFAFiniteP1
build_path=Path(p['build_result']);assert hashlib.sha256(build_path.read_bytes()).hexdigest()==p['build_result_sha256']
build=json.loads(build_path.read_text());assert build['status']=='finite_extension_compiled_not_executed'
extension=VendorFAFiniteP1(p['library'],p['library_sha256'])
report.update(operator_attempts=[],captured_layers=[],operator_model_backwards=0)
report['native_FA_implementation_sha256']=hashlib.sha256(Path(fa_native.__file__).read_bytes()).hexdigest()

def array(name,value):
    path=HERE/(name+'.npy');np.save(path,value.detach().cpu().numpy(),allow_pickle=False)
    return {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(value.shape),'dtype':str(value.dtype)}

def observe(li,q0,k0,v0,q1,k1,v1,u,raw0,raw1,qprod,kprod,vprod,scale):
    if li!=35:return
    assert not report['captured_layers']
    values={'q0':q0.half(),'k0':k0.half(),'v0':v0.half(),'q1':q1.half(),'k1':k1.half(),'v1':v1.half(),
        'u':u,'lse0':raw0['fa_lse'],'lse1':raw1['fa_lse'],
        'out0':raw0['fa_out'].transpose(1,2),'out1':raw1['fa_out'].transpose(1,2),
        'expected_dq':qprod*scale,'expected_dk':kprod*scale,'expected_dv':vprod}
    # Source core has already checked these against real Qwen/FA endpoint tensors.
    record={'layer':li,'scale':scale,'groups':model.model.layers[li].self_attn.num_key_value_groups,
        'operands':{name:array('layer35_'+name,value) for name,value in values.items()},'source':'Actual unchanged P1 pullback on original NI0, immediately before original finite attention layout; no fabricated inputs.'}
    report['captured_layers'].append(record);save()

def dense(ops,scale):
    from compiled_finite_rules import logarithmic_mean_with_checks
    q0,k0,q1,k1,v0,u=[ops[k].float() for k in ['q0','k0','q1','k1','v0','u']]
    n=q0.shape[-2];mask=torch.arange(n,device=q0.device)[None,:]>torch.arange(n,device=q0.device)[:,None]
    z0=(q0@k0.transpose(-1,-2))*scale;z1=(q1@k1.transpose(-1,-2))*scale
    z0=z0.masked_fill(mask,float('-inf'));z1=z1.masked_fill(mask,float('-inf'))
    mean,checks=logarithmic_mean_with_checks(z0,z1)
    t=u@v0.transpose(-1,-2);tau=mean.sum(-1);center=(mean*t).sum(-1)/tau
    ds=mean*(t-center[...,None]);p1=(z1-ops['lse1'][...,None]).exp()
    return {'dq':ds@((k0+k1)*.5)*scale,'dk':ds.transpose(-1,-2)@((q0+q1)*.5)*scale,
        'dv':p1.transpose(-1,-2)@u,'tau':tau,'center':center}

def standard_FA(ops,scale,groups):
    # Actual native GQA configuration, same original endpoint/U. Cost reference only.
    with torch.enable_grad():
        q=ops['q1'].transpose(1,2).detach().contiguous().requires_grad_(True)
        k=ops['k1'][:,::groups].transpose(1,2).detach().contiguous().requires_grad_(True)
        v=ops['v1'][:,::groups].transpose(1,2).detach().contiguous().requires_grad_(True)
        out=fa_native.flash_attn_func(q,k,v,dropout_p=0.,softmax_scale=scale,causal=True)
        grads=torch.autograd.grad(out,(q,k,v),ops['u'].transpose(1,2).half().contiguous())
    return {name:value.transpose(1,2) for name,value in zip(['dq','dk','dv'],grads)}

def operator_call(kind,ops,record,repeat,profile=False):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    attempt={'kind':kind,'repeat':repeat,'warmup':repeat==0,'profile':profile,'complete':False,'activity':{}}
    report['operator_attempts'].append(attempt)
    try:
        with torch.no_grad():
            if kind=='finite_FA':values=extension(ops,record['scale'],attempt['activity'])
            elif kind=='dense_P1':values=dense(ops,record['scale'])
            else:values=standard_FA(ops,record['scale'],record['groups'])
        torch.cuda.synchronize();attempt.update(complete=True,seconds=time.perf_counter()-tick,
            resident_bytes=resident,peak_bytes=torch.cuda.max_memory_allocated(),
            incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident)
        assert all(torch.isfinite(v).all() for v in values.values())
        if kind!='standard_FA' and repeat==0:
            attempt['output_arrays']={name:array(kind+'_'+name,value) for name,value in values.items()}
        attempt['output_stats']={name:{'shape':list(v.shape),'dtype':str(v.dtype),
            'sha256_values':hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()} for name,v in values.items()}
        return attempt
    except Exception:
        attempt['error']=traceback.format_exc();attempt['seconds']=time.perf_counter()-tick;raise
    finally:save()

save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    path=ROOT/'exp/exp2/data/niah_mq_q2.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256']['niah_mq_q2']
    ex=runner.ds_utils.load_cached(path)[0]
    expected=next(r for r in parent['records'] if (r['dataset'],r['idx'])==('niah_mq_q2',0))
    capture_module.ATTENTION_OBSERVER=observe
    try:original=strong_run(ex,expected,capture_mode='public')
    finally:capture_module.ATTENTION_OBSERVER=None
    report['records']=[{'dataset':'niah_mq_q2','idx':0,'original_result':original,
        'signed_vector_exact_to_parent':original['signed_full_sequence']==expected['native']['strong_secant_pv_content_P1']['signed_full_sequence']}]
    assert report['records'][0]['signed_vector_exact_to_parent'] and len(report['captured_layers'])==1
    record=report['captured_layers'][0]
    ops={name:torch.from_numpy(np.load(HERE/item['file'],allow_pickle=False)).to(model.device) for name,item in record['operands'].items() if not name.startswith('expected_') and name not in ['out0','out1']}
    for repeat in range(4):
        order=['dense_P1','finite_FA','standard_FA'];offset=repeat%3;order=order[offset:]+order[:offset]
        for kind in order:
            attempt=operator_call(kind,ops,record,repeat)
            print('OPERATOR_DONE',kind,repeat,attempt['seconds'],attempt['incremental_peak_bytes'],flush=True)
    report['profiles']={}
    for kind in ['finite_FA','standard_FA']:
        with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
            operator_call(kind,ops,record,4,profile=True)
        trace=kind+'_trace.json';prof.export_chrome_trace(str(HERE/trace))
        report['profiles'][kind]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),
            'GPU_kernels':[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']}
        del prof;save()
    native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    for key,value in p['capture_budget'].items():assert report[key]==value,(key,report[key],value)
    assert len(report['operator_attempts'])==14
    report['status']='complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    names=['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_trace.json')]
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(HERE/name,name)
    with zipfile.ZipFile(HERE/'actual_tensor_arrays.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in HERE.glob('*.npy'):z.write(x,x.name)
'''
ast.parse(driver);study=A/'vendor_fa_finite_actual_probe_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'public_fa_capture_probe_protocol_20260907.json').read_text())
for key in ['budget','repeats','previous_attempt','predeclared_decision','metadata_scope']:p.pop(key,None)
sources=list(p['sources'])+['qwen_signed_secant_pv_operand_capture.py','qwen_signed_secant_public_operand_capture.py','vendor_fa_finite_runtime.py']
b=json.loads((A/'vendor_fa_finite_build_summary_20260907.json').read_text())
p.update(purpose='Actual original NI0 layer35 complete attention operator diagnostic. P1 original vector still computed/returned unchanged. New finite FA extension uses genuine vendor FA tile primitives and default-style mixed precision; no quadrature or FlexAttention. Compare complete arrays and real local memory/time against dense P1 and actual installed GQA FA F+B. Not end-to-end acceleration, quality confirmation, or length scaling evidence.',
 study_sha256=sha(study),sources={name:sha(A/name) for name in sources},
 wait_for_pid=117414,wait_for_script='codex_vendor_fa_finite_build_20260907_v1',
 build_result='${ARTIFACT_ROOT}/codex_vendor_fa_finite_build_20260907_v1/results.json',build_result_sha256=b['raw_sha256'],
 library='${ARTIFACT_ROOT}/codex_vendor_fa_finite_build_20260907_v1/libdeltatrace_fa_finite.so',library_sha256=b['library']['sha256'],
 capture_budget={'native_root_forwards':1,'native_vjps':0,'evaluation_forwards':0,'ft_attribution_forwards':0,'ordinary_reference_forwards':0,
  'native_attribution_forwards':1,'manual_passes':1,'extra_layer_replay_calls':36,'extra_native_fa_attention_calls':36,
  'native_attribution_endpoint_trajectories':2,'extra_layer_replay_endpoint_trajectories':72},
 operator_budget={'dense_P1_calls':4,'finite_FA_calls':5,'finite_FA_kernel_launches':15,'standard_GQA_FA_forwards':5,'standard_GQA_FA_backwards':5,'model_backwards':0,'additional_model_forwards':0},
 predeclared_numerical_review={'max_relative_l2_to_frozen_dense_dq_dk_dv':0.01,
  'gate_scope':'Local implementation screening only. Does not certify original RISE/MAS, signs, end-to-end speed or long-sequence scaling. Failure preserved, no silent fallback.'},
 cost_scope='Time includes wrapper validation, dtype conversions, contiguous packing, means, output allocation and real kernel launches; excludes CPU arrays/results diagnostics. Resident excludes dense tensors from original model pullback: original pass finished first. Dense control is minimal required P1 arithmetic, not inflated by old audit-only tensors; standard FA retains native GQA.',
 selection=[['niah_mq_q2',0]],actual_layers=[35])
protocol=A/'vendor_fa_finite_actual_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_finite_actual_20260907_v1',
 f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in sources],
 '--request',str(A/'vendor_fa_finite_actual_launch_20260907.json')],check=True)
print('Frozen one real original attention layer:14operator calls; full P1 remains unchanged.')
