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
parent_file=Path(p['parent']);assert hashlib.sha256(parent_file.read_bytes()).hexdigest()==p['parent_sha256']
parent=json.loads(parent_file.read_text());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
def native_sources():return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in parent['native_sources_before']}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_forwards=0,trajectories=0,native_vjps=0,quality_queries=0,records=[])
def save():
    tmp=HERE/'results.partial';tmp.write_text(json.dumps(report,separators=(',',':')));tmp.replace(HERE/'results.json')
def score(ids,plen):
    assert ids.shape[0]==4
    selected=torch.arange(plen-1,ids.shape[1]-1,device=ids.device)
    with measured() as cost,torch.no_grad():
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_attentions=False,logits_to_keep=selected)
        values=output.logits.float().log_softmax(-1).gather(2,ids[:,plen:,None]).squeeze(-1)
        scores=values.double().sum(-1)
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==4 and cost['native_decoder_layer_calls']==36 and cost['extra_replay_calls']==0
    assert torch.isfinite(values).all()
    report['native_forwards']+=1;report['trajectories']+=4
    return {'scores32_sum64':scores.cpu().tolist(),'target_logprobs32':values.cpu().tolist(),'cost':cost}
save()
try:
    model.set_attn_implementation('flash_attention_2');torch.backends.cuda.matmul.allow_tf32=False
    for case_no,selection in enumerate(p['selection']):
        source=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(selection['dataset'],selection['idx']))
        assert source['input_ids_sha256']==selection['input_ids_sha256']
        selected=selection['selected'];positions=[s['position'] for s in selected]
        assert len(positions)==len(set(positions))==24 and set(positions)<=set(source['eligible_positions'])
        finite=next(r['result'] for r in source['runs'] if r['mode']=='finite' and r['repeat']==1)
        assert all(s['finite_P1_score']==finite['signed_full_sequence'][s['position']] for s in selected)
        ids=torch.tensor([source['input_ids']],dtype=torch.long,device='cuda:0');plen=source['prompt_len']
        clean=ids.repeat(4,1);eos=clean.clone();eos[:,torch.tensor(source['eligible_positions'],device=ids.device)]=tokenizer.eos_token_id
        row={'dataset':source['dataset'],'idx':source['idx'],'input_ids_sha256':source['input_ids_sha256'],'N':ids.shape[1],'prompt_len':plen,
            'selected':selected,'baseline_repeats':{},'interventions':[]};report['records'].append(row)
        for name,baseline in [('clean',clean),('eos',eos)]:
            report['active']=[case_no,name,'baseline_first'];save();row['baseline_repeats'][name]=[score(baseline,plen)];save()
            for offset in range(0,24,4):
                group=positions[offset:offset+4];changed=baseline.clone()
                for lane,j in enumerate(group):changed[lane,j]=tokenizer.eos_token_id if name=='clean' else clean[lane,j]
                assert torch.equal(changed[:,plen:],clean[:,plen:])
                for lane,j in enumerate(group):
                    actual=(changed[lane]!=baseline[lane]).nonzero().flatten().tolist();assert actual==[j]
                value=score(changed,plen)
                base_scores=row['baseline_repeats'][name][0]['scores32_sum64']
                effects=[(a-b if name=='clean' else b-a) for a,b in zip(base_scores,value['scores32_sum64'])]
                row['interventions'].append({'background':name,'positions':group,'lanes':[0,1,2,3],'result':value,'conditional_effects':effects,
                    'actual_changed_positions_checked':True,'original_target_preserved':True})
                report['active']=[case_no,name,offset];save()
            row['baseline_repeats'][name].append(score(baseline,plen));save()
        if case_no==0:
            with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:extra=score(clean,plen)
            trace='native_conditional_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
            row['profile']={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':extra};del prof,extra
        print('CONDITIONAL_SIGN_DONE',row['dataset'],row['idx'],flush=True)
        del ids,clean,eos,changed
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    for k,v in p['budget'].items():assert report[k]==v
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+[f.name for f in HERE.glob('*_FA_trace.json')]:z.write(HERE/name,name)
