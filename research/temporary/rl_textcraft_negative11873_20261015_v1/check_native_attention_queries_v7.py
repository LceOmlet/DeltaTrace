"""Identify which original joint-target queries supply the upstream negative FA signal.

Reuse the deployed finite FA owner for query partitions and the installed FA
forward for an independent value-route prediction. Query partitions are only
diagnostic linear-response inputs, never new training targets or span credit.
"""
import hashlib,inspect,json,os,sys,time
from pathlib import Path
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-neg11873-rootcause-20261015-v1/layer31-upstream-v6'
dt=Path(os.environ['DT_ROOT']);sys.path[:0]=[str(dt),str(dt/'clean/qwen35')]
from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256,RightPaddedLengths
from flash_attn import flash_attn_func

torch.set_num_threads(2)
data=torch.load(DEST/'native-transport-v6-rank0.pt',map_location='cpu',weights_only=True,mmap=True)
p=data['metadata']['layer31_payload'];focus_row=data['focus_row'];lv=p['FA_layout']['values']
env=json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
owner=VendorFAFiniteP1BF16D256(env['finite_library'],env['finite_library_sha256'])
ops={k:v.cuda().contiguous() for k,v in p['native_FA_operands'].items()}
start=p['native_row_start'];query_index=1871-start;groups=p['groups']
layout=RightPaddedLengths([lv['lengths'][focus_row]],lv['padded_length'],'cuda',
    coefficient_starts=[lv['coefficient_starts'][focus_row]],
    query_starts=[lv['query_starts'][focus_row]],query_padded_length=lv['query_padded_length'])
delta_k=(ops['k1'][0,:,1871].float()-ops['k0'][0,:,1871].float()).repeat_interleave(groups,dim=0)
delta_v=(p['v_pair'][1,:,1].float()-p['v_pair'][0,:,1].float()).cuda().repeat_interleave(groups,dim=0)
def effects(out):
    return dict(k=float((out['dk'][0,:,query_index].double()*delta_k.double()).sum()),
                v=float((out['dv'][0,:,query_index].double()*delta_v.double()).sum()))
result=dict(scope=__doc__,checkpoint=25,incident_exact_replay=False,transformer_calls=0,full_DT_calls=0,
    production_changes=0,tolerance=None,groups=[],source_bindings={},started_unix=time.time())
for name,fn in [('actual_finite_FA_owner',VendorFAFiniteP1BF16D256),('native_FA_forward',flash_attn_func)]:
    path=Path(inspect.getfile(inspect.unwrap(fn)));result['source_bindings'][name]=dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
result['source_bindings']['actual_finite_library']=dict(path=env['finite_library'],SHA256=env['finite_library_sha256'])
outpath=DEST/'native-attention-query-route-v7.json'
full=owner(ops,p['scale'],layout)
saved=p['native_FA_outputs']
result['same_operands_owner_baseline']=dict(effects=effects(full),
    dk_at_point_exact=bool(torch.equal(full['dk'][0,:,query_index].cpu(),saved['dk'][0,:,query_index])),
    dv_at_point_exact=bool(torch.equal(full['dv'][0,:,query_index].cpu(),saved['dv'][0,:,query_index])))

# Native FA owns the linear V response with factual Q/K fixed. All other value
# positions carry zero increment, so its per-query output is directly readable.
valid_length=lv['lengths'][focus_row];valid_queries=valid_length-start
native_delta=torch.zeros_like(ops['v0'][:,:,:valid_length].transpose(1,2)).contiguous()
native_delta[:,1871]=(p['v_pair'][1,:,1].float()-p['v_pair'][0,:,1].float()).cuda().to(native_delta.dtype)
with torch.no_grad():
    response=flash_attn_func(ops['q1'][:,:,:valid_queries].transpose(1,2).contiguous(),ops['k1'][:,:,:valid_length].transpose(1,2).contiguous(),
        native_delta,softmax_scale=p['scale'],causal=True,dropout_p=0.0)
weighted=response.transpose(1,2).double()*ops['u'][:,:,:valid_queries].to(torch.bfloat16).double()
per_query=torch.zeros(ops['u'].shape[2],device='cuda',dtype=torch.float64)
per_query[:valid_queries]=weighted.sum((0,1,3))
result['independent_original_FA_value_forward']=dict(total=float(per_query.sum()),
    logical_query_start=start,logical_query_length=valid_queries,logical_key_length=valid_length,
    largest_query_positions=[dict(position=int(i)+start,effect=float(per_query[i])) for i in per_query.abs().topk(10).indices.tolist()])

support=ops['u'][0].double().norm(dim=(0,2)).nonzero().flatten().tolist()
spans=[]
for i in support:
    if not spans or i!=spans[-1][-1]+1:spans.append([])
    spans[-1].append(i)
result['original_target_query_support']=dict(tokens=len(support),contiguous_spans=len(spans))
for span in spans:
    if span[-1]+start<1871:continue # Causality excludes earlier targets from this key/value position.
    masked=dict(ops);masked['u']=torch.zeros_like(ops['u'])
    masked['u'][:,:,span]=ops['u'][:,:,span]
    part=owner(masked,p['scale'],layout)
    item=dict(prediction_positions=[span[0]+start,span[-1]+start],tokens=len(span),
        native_owner_point_effects=effects(part),native_FA_value_forward=float(per_query[span].sum()))
    result['groups'].append(item);outpath.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(item),flush=True)
result.update(status='completed_original_FA_query_route',finished_unix=time.time())
result['sum_group_k']=sum(x['native_owner_point_effects']['k'] for x in result['groups'])
result['sum_group_v']=sum(x['native_owner_point_effects']['v'] for x in result['groups'])
outpath.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
