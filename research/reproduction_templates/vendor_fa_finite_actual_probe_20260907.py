"""True per-call preparation/capture/pullback timing; separate full diagnostic and production paths."""
import contextlib
import hashlib
import inspect
import json
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
ROOT = Path('${FLASHTRACE_ROOT}')
HERE = Path(__file__).resolve().parent
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')
os.environ['TRITON_CACHE_DIR']=str(HERE/'triton_cache')
sys.path.insert(0, str(ROOT))
p = json.loads((HERE/'protocol.json').read_text())
assert hashlib.sha256((HERE/'study.py').read_bytes()).hexdigest() == p['study_sha256']
for rel, expected in p['official_normalized_sources'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected

def source_module(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


wait_started=time.time()
predecessor=Path('/proc')/str(p['wait_for_pid'])
while predecessor.exists():
    try:
        command=(predecessor/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:
        break
    if p['wait_for_script'] not in command:
        break
    if time.time()-wait_started>p['maximum_queue_seconds']:
        raise TimeoutError('Predecessor still live; do not overlap GPU timing.')
    (HERE/'queue_status.json').write_text(json.dumps({'status':'waiting_for_predecessor','pid':p['wait_for_pid'],'seconds':time.time()-wait_started}))
    time.sleep(5)
queue_seconds=time.time()-wait_started
print('PREDECESSOR_TERMINAL',p['wait_for_pid'],'queue_seconds',queue_seconds,flush=True)

attr = source_module('llm_attr', ROOT/'llm_attr.py')
runner = source_module('official_both_runner', ROOT/'exp/exp2/run_exp.py')
both = source_module('ft_ifr_improve', ROOT/'ft_ifr_improve.py')
import torch
entry_started = time.time()

def checkpoint_receipt():
    path = Path(p['checkpoint_receipt'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == p['checkpoint_receipt_sha256']
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'complete' and receipt['checkpoint'] == p['checkpoint']
    results = []
    for expected in receipt['files']:
        file = Path(p['checkpoint'])/expected['name']
        before = file.stat()
        h = hashlib.sha256()
        with file.open('rb') as f:
            for block in iter(lambda: f.read(8*1024**2), b''):
                h.update(block)
        after = file.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        assert after.st_size == expected['bytes'] and h.hexdigest() == expected['sha256']
        results.append({'file': expected['name'], 'sha256': h.hexdigest()})
    return results

report = {'status': 'running', 'protocol': p, 'records': [], 'checkpoint_before': checkpoint_receipt(),
    'scope': 'Frozen independent checkpointed-secant evaluation. All candidate and original FT both1/legacy0-3 scores and curves computed freshly. Fixed confirmation or full official cross-task coverage specified by immutable protocol. No candidate tuning in this job.'}
print('CHECKPOINT_BEFORE_OK', flush=True)
torch.manual_seed(73)
model, tokenizer = runner.load_model(p['checkpoint'], 'cuda:0')
model.requires_grad_(False)
model_source = Path(inspect.getfile(type(model)))
assert hashlib.sha256(model_source.read_bytes()).hexdigest() == p['native_model_source_sha256']
assert not model.training and tokenizer.pad_token_id == tokenizer.eos_token_id
report.update(model_class=str(type(model)), dtype=str(next(model.parameters()).dtype),
              attention=model.config._attn_implementation, device=torch.cuda.get_device_name(), torch_version=torch.__version__)

def hooks():
    return sum(len(m._forward_hooks)+len(m._forward_pre_hooks)+len(m._backward_hooks) for m in model.modules())

@contextlib.contextmanager
def measured():
    assert hooks() == 0
    cost = {'native_forwards': 0, 'vjps': 0, 'native_decoder_layer_calls':0,'native_forward_trajectories':0,'native_decoder_layer_trajectories':0,'extra_replay_calls':0,'extra_replay_trajectories':0}
    inside_root=[False]
    def count(_module, _inputs, kwargs):
        value=kwargs.get('input_ids')
        if value is None:value=kwargs.get('inputs_embeds')
        if value is None:value=_inputs[0]
        inside_root[0]=True
        cost['native_forwards'] += 1
        cost['native_forward_trajectories'] += value.shape[0]
    h = model.register_forward_pre_hook(count,with_kwargs=True)
    def root_return(_module,_inputs,_output):inside_root[0]=False
    end_handle=model.register_forward_hook(root_return,always_call=True)
    def count_layer(_module,_inputs):
        cost["native_decoder_layer_calls"]+=1
        cost["native_decoder_layer_trajectories"]+=_inputs[0].shape[0]
        if not inside_root[0]:
            cost["extra_replay_calls"]+=1;cost["extra_replay_trajectories"]+=_inputs[0].shape[0]
    layer_handles=[layer.register_forward_pre_hook(count_layer) for layer in model.model.layers]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield cost
    finally:
        torch.cuda.synchronize()
        cost['seconds'] = time.perf_counter()-start
        cost['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        h.remove();end_handle.remove()
        for handle in layer_handles:
            handle.remove()
        assert hooks() == 0



import zipfile
import flash_attn.flash_attn_interface as fa_native
import transformers.integrations.flash_attention as fa_adapter
import transformers.utils.generic as generic
from qwen_signed_secant_native_paired_pv_rules import capture_checkpoint_pair
from qwen_signed_secant_native_paired_pv_rules import propagate_paired_secant as private_capture
from qwen_signed_secant_public_operand_capture import propagate_paired_secant as public_capture
for name,digest in p['sources'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
parent_path=Path(p['required_parent']);assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['required_parent_sha256']
parent=json.loads(parent_path.read_bytes());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
torch.backends.cuda.matmul.allow_tf32=False
def native_sources():
    return {str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [fa_native,fa_adapter,fa_native.flash_attn_cuda,generic]}
original_methods={name:type(m).forward for name,m in model.named_modules()}
def native_method_audit(require_no_instance=False):
    rows=[]
    for name,m in model.named_modules():
        assert type(m).forward is original_methods[name],name
        assert getattr(m.forward,'__func__',None) is original_methods[name] and getattr(m.forward,'__self__',None) is m,name
        if 'forward' in m.__dict__:rows.append(name)
        if require_no_instance:assert 'forward' not in m.__dict__,name
    assert hooks()==0
    return rows

assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_root_forwards=0,native_vjps=0,
 evaluation_forwards=0,ft_attribution_forwards=0,ordinary_reference_forwards=0,native_attribution_forwards=0,
 manual_passes=0,extra_layer_replay_calls=0,extra_native_fa_attention_calls=0,
 native_attribution_endpoint_trajectories=0,extra_layer_replay_endpoint_trajectories=0)
def save():
    temp=HERE/'results.partial';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(HERE/'results.json')
def strong_run(ex,expected,pv_rule='content_P1',capture_mode='private'):
    """Actual input preparation belongs inside this invocation's single timer.

    No report serialization occurs inside the timed API. Both branches are
    explicitly production paths without per-operator diagnostic ledgers.
    """
    propagate_paired_secant=public_capture if capture_mode=='public' else private_capture
    activity={'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]}
    torch.cuda.empty_cache();preprop_peak=0;finished=False
    report['manual_attempts']=report.get('manual_attempts',0)+1;save()
    try:
        with measured() as full:
            prep_tick=time.perf_counter()
            engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
            ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
            positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens)
            eligible=[positions[j] for j in keep]
            prep_host_seconds=time.perf_counter()-prep_tick
            baseline=ids.clone();baseline[0,torch.tensor(eligible,device=ids.device)]=tokenizer.eos_token_id
            reference,clean=capture_checkpoint_pair(model,baseline,ids,mask,plen)
            # Pullback resets its own peak counter. Preserve the earlier actual
            # preparation/capture peak rather than silently losing it.
            preprop_peak=torch.cuda.max_memory_allocated()
            result=propagate_paired_secant(model,reference,clean,pv_rule=pv_rule,**({'activity':activity} if capture_mode=='public' else {}))
            result['endpoint_scores32']={'before':reference['score32_sum64'],'after':clean['score32_sum64']}
            identity={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible}
            del reference,clean,baseline,ids,mask,engine
        finished=True
    finally:
        full['public_FA_activity']=activity
        report['extra_native_fa_attention_calls']+=activity['auxiliary_completed']
        full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])
        report['native_root_forwards']+=full['native_forwards'];report['native_attribution_forwards']+=full['native_forwards']
        report['native_attribution_endpoint_trajectories']+=full['native_forward_trajectories']
        report['extra_layer_replay_calls']+=full['extra_replay_calls']
        report['extra_layer_replay_endpoint_trajectories']+=full['extra_replay_trajectories']
        report.setdefault('native_attempt_costs',[]).append({'pv_rule':pv_rule,'capture_mode':capture_mode,'completed':finished,'cost':dict(full)})
        save()
    assert full['native_forwards']==1 and full['native_forward_trajectories']==2
    assert full['native_decoder_layer_calls']==72 and full['native_decoder_layer_trajectories']==144
    assert full['extra_replay_calls']==36 and full['extra_replay_trajectories']==72
    assert result['native_layer_replay_calls']==36 and result['extra_native_fa_attention_calls']==(36 if capture_mode=='public' else 0)
    for key,value in identity.items():assert value==expected[key],key
    full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])
    report['manual_passes']+=1
    assert result['extra_native_fa_attention_calls']==activity['auxiliary_completed']
    result['capture_mode']=capture_mode
    result.update(propagation_internal_seconds=result['seconds'],end_to_end_cost=full,seconds=full['seconds'],
        peak_allocated_bytes=full['peak_allocated_bytes'],preparation_host_seconds_this_call=prep_host_seconds,
        preparation_scope='Actually executed during this timed invocation; host subphase timing only, not a separately synchronized GPU duration.',
        kernel_probe_operands=[],per_operator_ledger_collected=False)
    scale=max(1.,abs(result['target_delta_score32_sum64']))
    result['numerical_review']={'relative_unassigned':abs(result['unassigned_total'])/scale,
        'relative_absolute_ledger':None,
        'relative_unbooked':None}
    assert all(result[k] is None for k in ['ledger','ledger_residual_sum','absolute_ledger_residual_sum','unbooked_rounding_residual'])
    return result

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
