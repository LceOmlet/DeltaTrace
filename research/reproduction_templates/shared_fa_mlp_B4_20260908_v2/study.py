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
# Reuse already uploaded pinned dependencies to keep the launch argument bounded.
import shutil
for name,digest in p['sources'].items():
    if not (HERE/name).exists():
        origin=Path(p['runtime_source_parent'])/name
        assert hashlib.sha256(origin.read_bytes()).hexdigest()==digest
        shutil.copyfile(origin,HERE/name)

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
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
from qwen_signed_secant_native_paired_pv_rules import propagate_paired_secant as private_capture
from qwen_signed_secant_paired_public_fa import propagate_paired_secant as public_capture
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
    temp=HERE/'results.partial';temp.write_text(json.dumps(report,ensure_ascii=False,separators=(",",":")));temp.replace(HERE/'results.json')
from functools import partial
from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant as finite_entry
from vendor_fa_finite_runtime import VendorFAFiniteP1
from native_backward_memory_reference_target_head import ordinary_input_backward
tick=time.perf_counter()
build_path=Path(p['build_result']);assert hashlib.sha256(build_path.read_bytes()).hexdigest()==p['build_result_sha256']
extension=VendorFAFiniteP1(p['library'],p['library_sha256'])
finite_capture=partial(finite_entry,finite_attention=extension)
report.update(finite_library_initialization_seconds=time.perf_counter()-tick,
              finite_FA_calls_attempted=0,finite_FA_calls_enqueued=0)

def strong_run(ex,expected,pv_rule='content_P1',capture_mode='finite'):
    """Actual input preparation belongs inside this invocation's single timer.

    No report serialization occurs inside the timed API. Both branches are
    explicitly production paths without per-operator diagnostic ledgers.
    """
    assert capture_mode in ['dense','finite']
    propagate_paired_secant=finite_capture if capture_mode=='finite' else public_capture
    finite_activity=[]
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
            result=propagate_paired_secant(model,reference,clean,pv_rule=pv_rule,activity=activity,**({'finite_activity':finite_activity} if capture_mode=='finite' else {}))
            result['endpoint_scores32']={'before':reference['score32_sum64'],'after':clean['score32_sum64']}
            identity={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible}
            del reference,clean,baseline,ids,mask,engine
        finished=True
    finally:
        full['public_FA_activity']=activity
        full['finite_FA_activity']=finite_activity
        report['finite_FA_calls_attempted']+=sum(x.get('calls_attempted',0) for x in finite_activity)
        report['finite_FA_calls_enqueued']+=sum(x.get('calls_enqueued',0) for x in finite_activity)
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
    assert result['native_layer_replay_calls']==36 and result['extra_native_fa_attention_calls']==36
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

from qwen_signed_secant_batched_vendor_fa_public import capture_batch,propagate_batch
from original_ft_coroutine_evaluation import evaluate_curves_batched
report.update(batch_runs=[],batch_profiles={},original_batched_evaluation_activity={})

def batch_run(examples,expected):
    activity={'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]};finite_activity=[]
    batch=len(examples);torch.cuda.empty_cache();preprop_peak=0;finished=False
    report['manual_attempts']=report.get('manual_attempts',0)+1;save()
    try:
        with measured() as cost:
            requests=[];identities=[]
            for ex,old in zip(examples,expected):
                engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
                ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
                positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens)
                eligible=[positions[j] for j in keep]
                identity={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible}
                assert all(value==old[key] for key,value in identity.items())
                baseline=ids.clone();baseline[0,torch.tensor(eligible,device=ids.device)]=tokenizer.eos_token_id
                requests.append((baseline,ids,plen));identities.append(identity)
                del engine,ids,mask,baseline
            before,after=capture_batch(model,requests,tokenizer.eos_token_id)
            preprop_peak=torch.cuda.max_memory_allocated()
            result=propagate_batch(model,before,after,activity=activity,finite_attention=extension,finite_activity=finite_activity)
            result['propagation_peak_before_full_max']=result['peak_allocated_bytes']
            result['capture_peak_before_propagation']=preprop_peak
            del before,after,requests
        finished=True
    finally:
        cost['public_FA_activity']=activity;cost['finite_FA_activity']=finite_activity
        cost['peak_allocated_bytes']=max(preprop_peak,cost['peak_allocated_bytes'])
        report['extra_native_fa_attention_calls']+=activity['auxiliary_completed']
        report['native_root_forwards']+=cost['native_forwards'];report['native_attribution_forwards']+=cost['native_forwards']
        report['native_attribution_endpoint_trajectories']+=cost['native_forward_trajectories']
        report['extra_layer_replay_calls']+=cost['extra_replay_calls']
        report['extra_layer_replay_endpoint_trajectories']+=cost['extra_replay_trajectories']
        report['finite_FA_calls_attempted']+=sum(x.get('calls_attempted',0) for x in finite_activity)
        report['finite_FA_calls_enqueued']+=sum(x.get('calls_enqueued',0) for x in finite_activity)
        report.setdefault('native_attempt_costs',[]).append({'capture_mode':'batch'+str(batch),'completed':finished,'cost':dict(cost)})
        save()
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==2*batch
    assert cost['native_decoder_layer_calls']==72 and cost['native_decoder_layer_trajectories']==144*batch
    assert cost['extra_replay_calls']==36 and cost['extra_replay_trajectories']==72*batch
    report['manual_passes']+=1
    result.update(end_to_end_cost=cost,propagation_internal_seconds=result['seconds'],seconds=cost['seconds'],peak_allocated_bytes=cost['peak_allocated_bytes'])
    return result

from qwen_signed_secant_paired_vendor_fa_gqa import propagate_paired_secant as compact_pair
from qwen_signed_secant_batched_vendor_fa_gqa_public import propagate_batch as compact_batch
from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA
from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse
compact_extension=VendorFAFiniteP1SharedMeanReuse(p['shared_mean_library'],p['shared_mean_library_sha256'])
old_extension=VendorFAFiniteP1CompactGQA(p['compact_library'],p['compact_library_sha256'])
old_pair=compact_pair;old_batch=compact_batch

from native_mlp_selective_checkpoint_reuse import NativeMLPSelectiveReuse
import torch.utils.checkpoint as torch_checkpoint
assert hashlib.sha256(Path(torch_checkpoint.__file__).read_bytes()).hexdigest()==p['torch_checkpoint_source_sha256']
report.update(integration_groups=[],fresh_attributions=0,quality_queries=0)
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    selection=p['integration_groups'][0]
    examples=[];rows=[]
    for dataset,index in selection:
        data=ROOT/f'exp/exp2/data/{dataset}.jsonl'
        assert hashlib.sha256(data.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        examples.append(runner.ds_utils.load_cached(data)[index])
        rows.append(next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index)))
    assert len(rows)==4
    group={'selection':selection,'example_batch_size':4,'runs':[],'comparisons':[]}
    report['integration_groups'].append(group)
    extension=compact_extension
    assert isinstance(extension,VendorFAFiniteP1SharedMeanReuse)
    finite_capture=partial(compact_pair,finite_attention=extension)
    propagate_batch=compact_batch
    for repeat in range(3):
        pair={}
        for mode in (['shared_mean','shared_mean_sac32'] if repeat<2 else ['shared_mean']):
            assert report['fresh_attributions']<5
            report['active']=[repeat,mode];save();native_method_audit(True)
            reuse={}
            # Count cache context construction, cleanup and synchronization for both methods.
            def invoke():
                torch.cuda.synchronize();tick=time.perf_counter()
                context=NativeMLPSelectiveReuse(model,reuse,first_cached_layer=4) if mode=='shared_mean_sac32' else contextlib.nullcontext()
                with context:
                    value=batch_run(examples,rows)
                torch.cuda.synchronize()
                value['all_in_seconds']=time.perf_counter()-tick
                value['native_MLP_reuse']=reuse
                return value
            if repeat==0 and mode=='shared_mean_sac32':
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as profiler:
                    value=invoke()
                trace=HERE/'sac32_B4_warm_profile.json';profiler.export_chrome_trace(str(trace))
                report['candidate_profile']={'file':trace.name,'sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                    'scope':'Budgeted candidate warm only; not steady wall timing.'}
                del profiler
            else:
                value=invoke()
            report['fresh_attributions']+=1;pair[mode]=value
            if mode=='shared_mean_sac32':
                assert reuse['installed_source_sha256']==p['torch_checkpoint_source_sha256']
                assert reuse['saved_mm_outputs']==reuse['reused_mm_outputs']==96
                assert reuse['retained_tensor_bytes']==0
                assert len(reuse['regions'])==32 and all(r['completed'] and r['input_exact'] for r in reuse['regions'])
            group['runs'].append({'mode':mode,'repeat':repeat,'warm':repeat==0,'result':value})
            save();print('SHARED_FA_MLP_B4',repeat,mode,value['all_in_seconds'],value['peak_allocated_bytes'],flush=True)
        if repeat==2:
            previous=next(x['result'] for x in group['runs'] if x['repeat']==1 and x['mode']=='shared_mean')
            assert pair['shared_mean']['signed_full_sequence']==previous['signed_full_sequence']
            assert pair['shared_mean']['endpoint_scores32']==previous['endpoint_scores32']
            continue
        old=pair['shared_mean'];new=pair['shared_mean_sac32']
        a=torch.tensor(old['signed_full_sequence'],dtype=torch.float64)
        b=torch.tensor(new['signed_full_sequence'],dtype=torch.float64)
        check={'repeat':repeat,'full_signed_exact':bool(torch.equal(a,b)),
            'max_abs_difference':float((a-b).abs().max()),'sign_flips':int(((a*b)<0).sum()),
            'endpoints_exact':old['endpoint_scores32']==new['endpoint_scores32']}
        group['comparisons'].append(check);save()
        assert check['full_signed_exact'] and check['endpoints_exact']
    assert report['fresh_attributions']==report['native_root_forwards']==5
    assert report['native_vjps']==report['evaluation_forwards']==report['ft_attribution_forwards']==0
    assert report['extra_layer_replay_calls']==report['extra_native_fa_attention_calls']==report['finite_FA_calls_enqueued']==180
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['active']=None;report['status']='shared_FA_native_MLP_B4_five_call_screen_complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['job_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]+[x.name for x in HERE.glob('*_profile.json')]:z.write(HERE/name,name)
