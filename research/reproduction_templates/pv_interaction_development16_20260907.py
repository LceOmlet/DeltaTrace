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



class Recorder(runner.llm_attr_eval.LLMAttributionEvaluator):
    def compute_logprob_response_given_prompt(self, prompt_ids, response_ids):
        assert torch.equal(response_ids, self.original_response)
        value = super().compute_logprob_response_given_prompt(prompt_ids, response_ids)
        changed = (prompt_ids[0] != self.original_prompt[0]).nonzero().flatten().tolist()
        inverse = {a:j for j,a in enumerate(self.positions)}
        assert all(a in inverse and int(prompt_ids[0,a]) == tokenizer.eos_token_id for a in changed)
        local = [inverse[a] for a in changed]
        assert set(local).issubset(self.keep)
        self.curve.append(float(value.sum()))
        self.deleted.append(local)
        return value

evaluator = Recorder(model, tokenizer)

def verify_masks(score, curve, masks, ids, positions, keep):
    w = score.detach().cpu().sum(0)
    keep = sorted(set(keep))
    steps = min(20, len(keep))
    assert len(curve) == len(masks) == steps+1 and masks[0] == []
    order = [keep[j] for j in torch.argsort(w[keep], descending=True).tolist()]
    expected = set()
    cursor = 0
    for step in range(steps):
        size = len(keep)//steps + int(step < len(keep)%steps)
        expected.update(order[cursor:cursor+size])
        cursor += size
        effective = {j for j in expected if int(ids[0,positions[j]]) != tokenizer.eos_token_id}
        assert set(masks[step+1]) == effective
    assert all(math.isfinite(v) for v in curve)

from qwen_signed_secant_native_paired_pv_rules import capture_checkpoint_pair, propagate_paired_secant
import statistics,zipfile,numpy as np
from native_backward_memory_reference_target_head import ordinary_input_backward
import flash_attn.flash_attn_interface as fa_native
import transformers.integrations.flash_attention as fa_adapter
import transformers.utils.generic as generic
for filename,digest in p['sources'].items():
    assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==digest
pilot_path=Path(p['required_pilot']);assert hashlib.sha256(pilot_path.read_bytes()).hexdigest()==p['required_pilot_sha256']
pilot=json.loads(pilot_path.read_bytes());assert pilot['status']=='complete' and len(pilot['records'])==3
assert pilot['records'][0]['profile']['actual_flash_forward_kernels']==72
quality_parent=pilot
fallback_file=Path(p['fallback_quality_parent'])
assert hashlib.sha256(fallback_file.read_bytes()).hexdigest()==p['fallback_quality_parent_sha256']
fallback_quality_parent=json.loads(fallback_file.read_bytes())
assert fallback_quality_parent['status']=='complete' and len(fallback_quality_parent['records'])==16
assert fallback_quality_parent['checkpoint_before']==fallback_quality_parent['checkpoint_after']==report['checkpoint_before']
assert fallback_quality_parent['protocol']['official_normalized_sources']==p['official_normalized_sources']
del pilot
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
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_root_forwards=0,native_vjps=0,evaluation_forwards=0,ft_attribution_forwards=0,ordinary_reference_forwards=0,native_attribution_forwards=0,manual_passes=0,extra_layer_replay_calls=0,extra_native_fa_attention_calls=0)
report.update(reused_quality_curves=0,fresh_quality_curves=0,native_attribution_endpoint_trajectories=0,extra_layer_replay_endpoint_trajectories=0)
def save():
    temp=HERE/'results.partial';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(HERE/'results.json')

def strong_run(ex,expected,pv_rule='symmetric'):
    """Actual input preparation belongs inside this invocation's single timer.

    No report serialization occurs inside the timed API. Both branches are
    explicitly production paths without per-operator diagnostic ledgers.
    """
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
            result=propagate_paired_secant(model,reference,clean,pv_rule=pv_rule)
            result['endpoint_scores32']={'before':reference['score32_sum64'],'after':clean['score32_sum64']}
            identity={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible}
            del reference,clean,baseline,ids,mask,engine
        finished=True
    finally:
        full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])
        report['native_root_forwards']+=full['native_forwards'];report['native_attribution_forwards']+=full['native_forwards']
        report['native_attribution_endpoint_trajectories']+=full['native_forward_trajectories']
        report['extra_layer_replay_calls']+=full['extra_replay_calls']
        report['extra_layer_replay_endpoint_trajectories']+=full['extra_replay_trajectories']
        report.setdefault('native_attempt_costs',[]).append({'pv_rule':pv_rule,'completed':finished,'cost':dict(full)})
        save()
    assert full['native_forwards']==1 and full['native_forward_trajectories']==2
    assert full['native_decoder_layer_calls']==72 and full['native_decoder_layer_trajectories']==144
    assert full['extra_replay_calls']==36 and full['extra_replay_trajectories']==72
    assert result['native_layer_replay_calls']==36 and result['extra_native_fa_attention_calls']==0
    for key,value in identity.items():assert value==expected[key],key
    full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])
    report['manual_passes']+=1
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

def nearby_ft_reference(ex,positions,keep):
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True);torch.cuda.empty_cache()
    with measured() as cost:
        attrs,_,fp,fk=runner.run_attribution({'model':model,'tokenizer':tokenizer,'attr_func':'ifr_multi_hop_both','chunk_tokens':128,'sink_chunk_tokens':32,'n_hops':1},ex,ex.target)
        score=attrs[0][:,:len(positions)].sum(0).cpu().float().tolist();del attrs
    assert fp==positions and fk==keep and cost['native_forwards']==1
    cost['scope']='ActualFT1 immediately after a native3-rule timing round;1warmup+3measured,actualinput andCPUscore inside timer.'
    report['native_root_forwards']+=1;report['ft_attribution_forwards']+=1
    restored=native_method_audit()
    for name,m in model.named_modules():
        if 'forward' in m.__dict__:
            assert name in restored and getattr(m.forward,'__func__',None) is original_methods[name]
            delattr(m,'forward') # Remove only author's restored original instance binding, exposing identical class method.
    model.set_attn_implementation(p['candidate_backend']);native_method_audit(True)
    return score,cost

save()
try:
    model.set_attn_implementation(p['candidate_backend']);native_method_audit(True)
    for dataset,indices in p['selection'].items():
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        examples=runner.ds_utils.load_cached(path)
        for idx in indices:
            report['active_case']={'dataset':dataset,'idx':idx,'phase':'native_input'};save()
            ex=examples[idx];torch.cuda.synchronize();prep_started=time.perf_counter()
            engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
            ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
            positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens);eligible=[positions[j] for j in keep]
            torch.cuda.synchronize();prep_seconds=time.perf_counter()-prep_started
            row={'dataset':dataset,'idx':idx,'input_ids':ids[0].tolist(),'input_ids_sha256':hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),'prompt_len':plen,
                'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible,'input_preparation_seconds':prep_seconds,'native':{},'scores':{},'metrics':{},'evaluation_masks':{},'evaluation_costs':{},'ft_costs':{}}
            report['records'].append(row);save()
            native_method_audit(True);report['active_case']['phase']='ordinary_backward_reference';save()
            row['memory_reference']=ordinary_input_backward(model,ids,mask,plen)
            report['native_root_forwards']+=1;report['native_vjps']+=1;report['ordinary_reference_forwards']+=1;save()

            row['repeats']={method:[] for method in p['native_methods']}
            row['ft_both1_repeats']=[]
            for repeat in range(4):
                offset=0 if repeat==0 else (repeat-1+idx)%3
                order=p['native_methods'][offset:]+p['native_methods'][:offset]
                for method in order:
                    rule=p['pv_rules'][method]
                    report['active_case'].update(phase='PV_rule_attribution',repeat=repeat,method=method,pv_rule=rule);save()
                    native_method_audit(True)
                    result=strong_run(ex,row,pv_rule=rule)
                    result.update(warmup=repeat==0,repeat=repeat,method=method,comparable_attribution_seconds=result['seconds'])
                    result['peak_bytes_above_reference']=result['peak_allocated_bytes']-row['memory_reference']['peak_allocated_bytes']
                    result['memory_gate_pass']=result['peak_bytes_above_reference']<=p['memory_allowance_bytes']
                    row['repeats'][method].append(result);save()
                    assert result['pv_rule']==rule and result['numerical_review']['relative_unbooked'] is None
                    print('ATTRIBUTION_DONE',dataset,idx,rule,repeat,result['seconds'],flush=True)
                # CurrentFT1 measured close to every native3-way round. Backend
                # selection/restoration is outside each attribution timer.
                ft_score,ft_cost=nearby_ft_reference(ex,positions,keep)
                if repeat==0:
                    row['scores']['flashtrace_both_hop1']=ft_score
                    row['ft_both1_warmup']=ft_cost
                    row['ft_costs']['flashtrace_both_hop1']=ft_cost
                else:
                    assert ft_score==row['scores']['flashtrace_both_hop1']
                    row['ft_both1_repeats'].append(ft_cost)
                save()
            for method in p['native_methods']:
                runs=row['repeats'][method];result=dict(runs[-1])
                result['comparable_attribution_seconds']=statistics.median(r['seconds'] for r in runs[1:])
                result['peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in runs[1:])
                result['peak_bytes_above_reference']=result['peak_allocated_bytes']-row['memory_reference']['peak_allocated_bytes']
                result['memory_gate_pass']=result['peak_bytes_above_reference']<=p['memory_allowance_bytes']
                row['native'][method]=result
                signed=torch.tensor(result['signed_full_sequence'],dtype=torch.float64)
                assert all(v==0 for j,v in enumerate(signed.tolist()) if j not in set(eligible))
                row['scores'][method]=signed[positions].clamp_min(0).float().tolist();del signed
            if dataset=='niah_mq_q2' and idx in [0,2]:
                row['profiles']={};row['profile_numerical_comparisons']={}
                for method in p['native_methods']:
                    rule=p['pv_rules'][method]
                    report['active_case'].update(phase='actual_native_PV_profile',method=method);save()
                    with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
                        diagnostic=strong_run(ex,row,pv_rule=rule)
                    row['profile_numerical_comparisons'][method]={'full_vector_equal':diagnostic['signed_full_sequence']==row['native'][method]['signed_full_sequence'],'per_operator_ledger_collected':False}
                    trace=f'{dataset}_{idx}_{rule}_native_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
                    names=[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
                    kernels=sum('flash_fwd_kernel' in name for name in names)
                    row['profiles'][method]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'actual_flash_forward_kernels':kernels,'pv_rule':rule,
                      'scope':'Extra real whole-attribution profile excluded from regular latency/peak comparison, included in all research calls. No mixed-endpoint model forward.'}
                    assert kernels==72 and diagnostic['native_layer_replay_calls']==36
                    del diagnostic,prof
            row['native_complete']=True;report.pop('active_case',None);save()
            print('NATIVE_DONE',dataset,idx,flush=True);del engine,ids,mask
    report['all_native_vectors_frozen_before_any_quality']=True;save()
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True)
    for row in report['records']:
        dataset,idx=row['dataset'],row['idx'];ex=runner.ds_utils.load_cached(ROOT/f'exp/exp2/data/{dataset}.jsonl')[idx]
        positions=row['user_positions'];keep=row['keep_local_indices'];plen=row['prompt_len']
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0')
        for hop in [0,2,3]:
            report['active_case']={'dataset':dataset,'idx':idx,'phase':'official_ft_both','hop':hop};save()
            native_method_audit();torch.cuda.empty_cache()
            with measured() as cost:
                attrs,_,fp,fk=runner.run_attribution({'model':model,'tokenizer':tokenizer,'attr_func':'ifr_multi_hop_both','chunk_tokens':128,'sink_chunk_tokens':32,'n_hops':hop},ex,ex.target)
                ft_score=attrs[0][:,:len(positions)].sum(0).cpu().float().tolist();del attrs
            assert fp==positions and fk==keep and cost['native_forwards']==1
            name=f'flashtrace_both_hop{hop}';row['scores'][name]=ft_score
            row['ft_costs'][name]=cost;report['native_root_forwards']+=1;report['ft_attribution_forwards']+=1
            row['restored_native_forward_attributes']=native_method_audit();save()
        native_method_audit();torch.cuda.empty_cache()
        legacy=attr.LLMIFRAttribution(model,tokenizer,show_progress=False)
        with measured() as cost:
            legacy_result=legacy.calculate_ifr_multi_hop(ex.prompt,target=ex.target,sink_span=tuple(ex.sink_span),thinking_span=tuple(ex.thinking_span),n_hops=3,renorm_threshold=0.0)
        assert cost['native_forwards']==1
        obs=legacy_result.metadata['ifr']['raw'].observation;cumulative=obs['base'].clone();position_tensor=torch.tensor(positions,dtype=torch.long)
        row['scores']['flashtrace_legacy_hop0']=cumulative.index_select(0,position_tensor).float().tolist()
        for hop,value in enumerate(obs['per_hop'],1):
            cumulative=cumulative+value;row['scores'][f'flashtrace_legacy_hop{hop}']=cumulative.index_select(0,position_tensor).float().tolist()
        row['legacy_joint_cost']=cost;row['legacy_cost_scope']='Fresh3-hop family capture gives cumulative0-3 vectors jointly; family time is not isolated per-hop latency.'
        report['native_root_forwards']+=1;report['ft_attribution_forwards']+=1
        del obs,cumulative,legacy_result,legacy,position_tensor
        row['restored_native_forward_attributes']=native_method_audit();torch.cuda.empty_cache();save()
        for result in row['native'].values():
            result['time_ratio_to_ft']=result['comparable_attribution_seconds']/statistics.median(r['seconds'] for r in row['ft_both1_repeats'])
            result['time_gate_pass']=result['time_ratio_to_ft']<=1
        evaluator.original_prompt=ids[:,:plen];evaluator.original_response=ids[:,plen:];evaluator.positions=positions;evaluator.keep=set(keep)
        gold=runner.ds_utils.ruler_gold_prompt_token_indices(ex,tokenizer)
        row['gold_full_local']=gold;row['gold_eligible_local']=sorted(set(gold)&set(keep))
        raw=tokenizer(' '+ex.prompt,add_special_tokens=False).input_ids
        assert all(raw[j]==int(ids[0,positions[j]]) for j in gold)
        assert set(row['scores'])==set(p['native_methods']+p['baselines'])
        parent_row=next((r for r in quality_parent['records'] if (r['dataset'],r['idx'])==(dataset,idx)),None)
        if parent_row is not None:
            selected_parent=quality_parent;parent_source=p['required_pilot'];parent_digest=p['required_pilot_sha256'];parent_map=p['parent_method_map']
        else:
            selected_parent=fallback_quality_parent;parent_source=p['fallback_quality_parent'];parent_digest=p['fallback_quality_parent_sha256'];parent_map=p['fallback_parent_method_map']
            parent_row=next(r for r in selected_parent['records'] if (r['dataset'],r['idx'])==(dataset,idx))
        for field in ['input_ids','prompt_len','user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local']:
            assert row[field]==parent_row[field]
        assert selected_parent['protocol']['checkpoint']==p['checkpoint']
        assert selected_parent['protocol']['official_evaluation_backend']==p['official_evaluation_backend']
        assert selected_parent['checkpoint_before']==report['checkpoint_before']
        assert selected_parent['native_sources_before']==report['native_sources_before']
        row['evaluation_provenance']={}
        for name in p['native_methods']+p['baselines']:
            parent_method=parent_map.get(name,name)
            if parent_method in parent_row['scores'] and row['scores'][name]==parent_row['scores'][parent_method]:
                import copy
                reuse_started=time.perf_counter()
                row['metrics'][name]=copy.deepcopy(parent_row['metrics'][parent_method])
                row['evaluation_masks'][name]=copy.deepcopy(parent_row['evaluation_masks'][parent_method])
                endpoints=parent_row['common_eager_evaluation_endpoints16']
                if 'common_eager_evaluation_endpoints16' in row:assert row['common_eager_evaluation_endpoints16']==endpoints
                else:row['common_eager_evaluation_endpoints16']=copy.deepcopy(endpoints)
                if parent_method in parent_row.get('recovery_topk_local',{}):
                    row.setdefault('recovery_topk_local',{})[name]=copy.deepcopy(parent_row['recovery_topk_local'][parent_method])
                row['evaluation_costs'][name]={'native_forwards':0,'native_decoder_layer_calls':0,'vjps':0,'seconds':time.perf_counter()-reuse_started,'scope':'CPU reuse of pinned original curve with identical current evaluator score vector and fixed inputs; no fresh evaluation model calls.'}
                row['evaluation_provenance'][name]={'mode':'identical_score_parent_curve','source':parent_source,'source_sha256':parent_digest,'source_method':parent_method,'historical_evaluation_cost':parent_row['evaluation_costs'][parent_method]}
                report['reused_quality_curves']+=1;save();continue
            row['evaluation_provenance'][name]={'mode':'fresh_original_curve','reason':'Current full evaluator score vector differs from pinned parent; numerical equality is only a reuse condition, not an acceptance gate.'}
            report['active_case']={'dataset':dataset,'idx':idx,'phase':'original_evaluation','method':name};save()
            score=torch.tensor(row['scores'][name],dtype=torch.float32);assert torch.isfinite(score).all() and (score>=0).all()
            evaluator.curve,evaluator.deleted=[],[]
            with measured() as cost,torch.no_grad():
                values=both.faithfulness_test_skip_tokens(evaluator,score[None],ex.prompt,ex.target,keep_prompt_token_indices=keep,user_prompt_indices=positions)
            report['native_root_forwards']+=cost['native_forwards'];report['evaluation_forwards']+=cost['native_forwards']
            verify_masks(score[None],evaluator.curve,evaluator.deleted,ids,positions,keep);assert cost['native_forwards']==21
            endpoints=[evaluator.curve[0],evaluator.curve[-1]]
            if 'common_eager_evaluation_endpoints16' in row:assert row['common_eager_evaluation_endpoints16']==endpoints
            else:row['common_eager_evaluation_endpoints16']=endpoints
            recovery=None if not gold else both.evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold)
            row['metrics'][name]={'rise':values[0],'mas':values[1],'mas_unnormalized':values[2],'recovery':recovery,'raw_curve':list(evaluator.curve)}
            row['evaluation_masks'][name]=list(evaluator.deleted);row['evaluation_costs'][name]=cost
            if recovery is not None:
                k=max(1,math.ceil(.1*len(keep)));selected=torch.topk(score[torch.tensor(keep,dtype=torch.long)].clamp_min(0),k,largest=True).indices.tolist();chosen=[keep[j] for j in selected]
                assert len(set(chosen)&set(row['gold_eligible_local']))/len(row['gold_eligible_local'])==recovery
                row.setdefault('recovery_topk_local',{})[name]=chosen
            report['fresh_quality_curves']+=1
            native_method_audit();save()
        row['complete']=True
        row['record_digest_sha256']=hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        report.pop('active_case',None);save();print('QUALITY_DONE',dataset,idx,flush=True)
        del ids
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    assert report['reused_quality_curves']+report['fresh_quality_curves']==176
    assert report['evaluation_forwards']==21*report['fresh_quality_curves']<=p['maximum_evaluation_forwards']
    executed=dict(p['budget']);executed['evaluation_forwards']=report['evaluation_forwards'];executed['native_root_forwards']+=report['evaluation_forwards']
    for key,value in executed.items():assert report[key]==value,key
    report['executed_budget']=executed
    report['status']='complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    files=[x for x in HERE.rglob('*') if x.is_file() and x.name not in ['results.partial','artifact_receipt.json'] and x.suffix!='.zip']
    (HERE/'artifact_receipt.json').write_text(json.dumps({str(x.relative_to(HERE)):{'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size} for x in files},indent=2))
    for name,suffixes in [('review_bundle.zip',{'.json','.py','.log'}),('actual_kernel_operands.zip',{'.npy'})]:
        with zipfile.ZipFile(HERE/name,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for x in files+[HERE/'artifact_receipt.json']:
                if (x.suffix in suffixes or (name=='review_bundle.zip' and x.suffix!='.npy')):z.write(x,str(x.relative_to(HERE)))
