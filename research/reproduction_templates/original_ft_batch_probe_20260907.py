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



import numpy as np, zipfile
from original_ft_batched_evaluation import evaluate_requests
for name,digest in p['sources'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
parent_path=Path(p['required_parent']);assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['required_parent_sha256']
parent=json.loads(parent_path.read_bytes());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
import flash_attn.flash_attn_interface as fa_native
import transformers.integrations.flash_attention as fa_adapter
import transformers.utils.generic as generic
def native_sources():
    return {str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [fa_native,fa_adapter,fa_native.flash_attn_cuda,generic]}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
original_methods={name:type(m).forward for name,m in model.named_modules()}
def audit_methods():
    for name,m in model.named_modules():
        assert type(m).forward is original_methods[name]
        assert getattr(m.forward,'__func__',None) is original_methods[name] and 'forward' not in m.__dict__
    assert hooks()==0
evaluator=runner.llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
original_score_method=runner.llm_attr_eval.LLMAttributionEvaluator.compute_logprob_response_given_prompt
def save():
    t=HERE/'results.partial';t.write_text(json.dumps(report,indent=2));t.replace(HERE/'results.json')
report.update(scope=p['purpose'],native_sources_before=native_sources(),physical_evaluation_forwards=0,
 evaluation_trajectories=0,native_decoder_layer_calls=0,native_decoder_layer_trajectories=0,
 model_backwards=0,attribution_calls=0,attempts=[])
def make_requests(row,methods):
    ids=torch.tensor(row['input_ids'],device=model.device,dtype=torch.long)[None]
    prompt=ids[:,:row['prompt_len']];response=ids[:,row['prompt_len']:]
    requests={}
    for method in methods:
        for step,mask in enumerate(row['evaluation_masks'][method]):
            value=prompt.clone()
            positions=[row['user_positions'][j] for j in mask]
            if positions:value[0,positions]=tokenizer.eos_token_id
            requests[(method,step)]=(value,response)
    assert len(requests)==42
    return requests
def call(row,methods,batch_size,warm=False):
    assert evaluator.compute_logprob_response_given_prompt.__func__ is original_score_method
    audit_methods();torch.cuda.empty_cache();cost={};done=False
    attempt={'dataset':row['dataset'],'idx':row['idx'],'batch_size':batch_size,'warm':warm,'complete':False}
    report['attempts'].append(attempt)
    try:
        with measured() as cost:
            requests=make_requests(row,methods)
            if warm:requests=dict(list(requests.items())[:batch_size])
            values,counts=evaluate_requests(evaluator,requests,batch_size)
        done=True;attempt.update(complete=True,cost=cost,scheduler_counts=counts)
        assert counts['physical_evaluation_forwards']==cost['native_forwards']
        assert counts['evaluation_trajectories']==cost['native_forward_trajectories']
        assert cost['extra_replay_calls']==cost['extra_replay_trajectories']==cost['vjps']==0
        assert cost['native_decoder_layer_calls']==36*cost['native_forwards']
        assert cost['native_decoder_layer_trajectories']==36*cost['native_forward_trajectories']
        return {m:[values[(m,i)] for i in range(21)] for m in methods} if not warm else None,dict(cost)
    finally:
        attempt['cost']=dict(cost)
        for src,dst in [('native_forwards','physical_evaluation_forwards'),('native_forward_trajectories','evaluation_trajectories'),
                        ('native_decoder_layer_calls','native_decoder_layer_calls'),('native_decoder_layer_trajectories','native_decoder_layer_trajectories')]:
            report[dst]+=cost.get(src,0)
        save();audit_methods()
save()
try:
    # Original benchmark evaluator/backend unchanged for every batch size.
    model.set_attn_implementation(p['official_evaluation_backend']);audit_methods()
    for case_number,(dataset,index) in enumerate(p['selection']):
        row=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        methods=['strong_secant_pv_content_P1',p['FT_curve_control'][dataset]]
        record={'dataset':dataset,'idx':index,'methods':methods,'parent_input_ids_sha256':hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest(),
                'prompt_len':row['prompt_len'],'N':len(row['input_ids']),'runs':[],
                'original_curves':{m:row['metrics'][m]['raw_curve'] for m in methods}}
        report['records'].append(record)
        for batch_size in p['batch_sizes']:call(row,methods,batch_size,warm=True)
        for repeat in range(p['measured_repeats']):
            sizes=p['batch_sizes'];offset=(case_number+repeat)%len(sizes);order=sizes[offset:]+sizes[:offset]
            for batch_size in order:
                curves,cost=call(row,methods,batch_size)
                record['runs'].append({'repeat':repeat,'batch_size':batch_size,'cost':cost,'curves':curves})
                save()
        print('BATCH_CASE_DONE',dataset,index,flush=True)
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_after']==report['checkpoint_before']
    report['native_sources_after']=native_sources();assert report['native_sources_after']==report['native_sources_before']
    report['status']='complete';audit_methods()
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','original_ft_batched_evaluation.py','results.json']:
            z.write(HERE/name,name)
