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
            result['capture_peak_before_propagation']=preprop_peak
            result['propagation_peak_before_full_max']=result['peak_allocated_bytes']
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

capacity_file=Path(p['capacity_file']);assert hashlib.sha256(capacity_file.read_bytes()).hexdigest()==p['capacity_sha256']
capacity=json.loads(capacity_file.read_text());assert capacity['status']=='complete' and capacity['native_forwards']==capacity['native_vjps']==0
report.update(gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,quality_evaluated=False)
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    for case_number,(dataset,index) in enumerate(p['selection']):
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(path)[index]
        engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
        ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
        positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens)
        expected={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':[positions[j] for j in keep]}
        digest=hashlib.sha256(json.dumps(expected['input_ids']).encode()).hexdigest()
        cap=next(x for x in capacity['datasets'][dataset]['records'] if x['idx']==index)
        assert digest==cap['input_ids_sha256'] and len(expected['input_ids'])==cap['full_tokens'] and plen==cap['formatted_prompt_tokens']
        row=dict(expected,dataset=dataset,idx=index,input_ids_sha256=digest,N=ids.shape[1],runs=[],profiles={})
        report['records'].append(row);del engine,ids,mask
        for repeat in range(4):
            order=['dense','finite'] if (case_number+repeat)%2==0 else ['finite','dense']
            for mode in order:
                native_method_audit(True);report['active']=[dataset,index,repeat,mode];save()
                value=strong_run(ex,expected,capture_mode=mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':value})
                save();print('ORIGINAL_LONG_DONE',dataset,index,repeat,mode,value['seconds'],value['peak_allocated_bytes'],flush=True)
        ids=torch.tensor([expected['input_ids']],device='cuda:0',dtype=torch.long);mask=torch.ones_like(ids)
        native_method_audit(True);report['active']=[dataset,index,'ordinary_input_backward'];save()
        # Reference helper has no parameter gradients and uses the same native
        # FA, actual target trajectory and native selected output head.
        reference=ordinary_input_backward(model,ids,mask,plen)
        row['ordinary_backward']=reference
        report['native_root_forwards']+=1;report['native_vjps']+=1;report['ordinary_reference_forwards']+=1
        del ids,mask;save()
        if [dataset,index]==p['profile_case']:
            native_method_audit(True);report['active']=[dataset,index,'finite_profile'];save()
            with torch.profiler.profile(activities=list(torch.profiler.supported_activities()),record_shapes=True,profile_memory=True) as prof:
                value=strong_run(ex,expected,capture_mode='finite')
            trace='long_finite_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
            names=[x.name for x in prof.events() if str(x.device_type)=='DeviceType.CUDA']
            record={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':value,
                 'actual_default_FA_forward_kernels':sum('flash_fwd_kernel' in n for n in names),
                 'actual_finite_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in names)}
            assert record['actual_default_FA_forward_kernels']==record['actual_finite_kernels']==108
            row['profiles']['finite']=record;del prof,value,names;save()
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    native_method_audit(True)
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_FA_trace.json')]:z.write(HERE/name,name)
