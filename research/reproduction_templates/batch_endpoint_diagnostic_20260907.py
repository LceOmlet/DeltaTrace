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
parent_path=Path(p['batch_parent'])
assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['batch_parent_sha256']
parent=json.loads(parent_path.read_text())
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
def native_sources():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in parent['native_sources_before']}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_forwards=0,trajectories=0,native_decoder_layer_calls=0)
def save():
    temp=HERE/'results.partial';temp.write_text(json.dumps(report,indent=2));temp.replace(HERE/'results.json')
save()
try:
    torch.backends.cuda.matmul.allow_tf32=False
    model.set_attn_implementation(p['official_evaluation_backend'])
    evaluator=runner.llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
    assert evaluator.compute_logprob_response_given_prompt.__func__ is type(evaluator).compute_logprob_response_given_prompt
    for dataset,index in p['selection']:
        row=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0')
        prompt=ids[:,:row['prompt_len']];response=ids[:,row['prompt_len']:]
        for endpoint in ['clean','deleted']:
            query=prompt.clone()
            if endpoint=='deleted':query[0,torch.tensor(row['eligible_positions'],device=query.device)]=tokenizer.eos_token_id
            for repeat in range(2):
                for batch in [1,4]:
                    queries=torch.cat([query]*batch);responses=torch.cat([response]*batch)
                    assert all(torch.equal(queries[0],x) for x in queries)
                    assert all(torch.equal(responses[0],x) for x in responses)
                    with measured() as cost,torch.no_grad():
                        value=evaluator.compute_logprob_response_given_prompt(queries,responses)
                    assert value.shape==responses.shape and torch.isfinite(value).all()
                    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==batch and cost['native_decoder_layer_calls']==36 and cost['extra_replay_calls']==0
                    report['native_forwards']+=cost['native_forwards'];report['trajectories']+=cost['native_forward_trajectories'];report['native_decoder_layer_calls']+=cost['native_decoder_layer_calls']
                    report['records'].append({'dataset':dataset,'idx':index,'endpoint':endpoint,'repeat':repeat,'batch':batch,'input_ids_sha256':row['input_ids_sha256'],
                        'identical_inputs_within_batch':True,'token_logprobs':value.cpu().tolist(),'sums':[float(x.sum()) for x in value],'cost':cost})
                    save();print('DIRECT_ENDPOINT',dataset,index,endpoint,repeat,batch,report['records'][-1]['sums'],flush=True)
    for key,value in p['budget'].items():assert report[key]==value
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_after']==report['checkpoint_before']
    report['native_sources_after']=native_sources();assert report['native_sources_after']==report['native_sources_before']
    report['status']='complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']:z.write(HERE/name,name)
