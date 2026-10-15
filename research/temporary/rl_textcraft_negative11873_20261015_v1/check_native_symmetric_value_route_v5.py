"""Native value-route prediction under the actually deployed symmetric profile.

Reuse average_memory_endpoint_orders itself, rather than substituting a
factual-only value adjoint. Original public convolution and FLA own all forward
and reverse operations. No transformer, DT rerun, PPO, deletion oracle or
production change is introduced.
"""
import hashlib,inspect,json,os,sys,time
from pathlib import Path
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-neg11873-rootcause-20261015-v1/layer30-mechanism-v3'
dt=Path(os.environ['DT_ROOT']);sys.path[:0]=[str(dt),str(dt/'clean/qwen35')]
from qwen35_gdn_finite import _scalar_secant,_silu_derivative
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders

torch.set_num_threads(2)
data=torch.load(DEST/'native-transport-v3-rank0.pt',map_location='cpu',weights_only=True,mmap=True)
p=data['metadata']['layer30_payload'];c=p['conv_silu']
assert p['norm_gate_rule']=='symmetric'
kd=p['num_k_heads']*p['head_k_dim'];vd=p['num_v_heads']*p['head_v_dim'];width=p['conv_width']
vs=slice(2*kd,2*kd+vd)
dx=p['projected_qkv_pair'][1].float()-p['projected_qkv_pair'][0].float()
pre=c['pre'].float();out=c['output'].float()
slope=_scalar_secant(pre[0],pre[1],out[0],out[1],_silu_derivative(pre[0]))
result=dict(scope=__doc__,checkpoint=25,incident_exact_replay=False,transformer_calls=0,
    full_DT_calls=0,optimizer_updates=0,formal_training_changes=0,tolerance=None,
    source_bindings={},groups=[],started_unix=time.time(),
    v4_limitation='Factual-only native value adjoint is not the actual symmetric DT value coefficient; v4 is retained as one endpoint control, not an owner-failure test.')
for name,fn in [('original_conv',causal_conv1d_fn),('original_FLA',chunk_gated_delta_rule),
                ('original_average',average_memory_endpoint_orders),('original_scalar_secant',_scalar_secant)]:
    path=Path(inspect.getfile(inspect.unwrap(fn)));result['source_bindings'][name]=dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
assert result['source_bindings']['original_average']['SHA256']=='dd6bbfff9aae4c679439af113a160e0bd8c4cf1b19bb17cab47a04ff7b0831d0'
output_path=DEST/'native-symmetric-value-route-v5.json'

impulse=torch.zeros((1,vd,width),device='cuda',dtype=c['pre'].dtype)
impulse[0,:,0]=dx[vs].cuda().to(impulse.dtype)
with torch.no_grad():pre_delta=causal_conv1d_fn(impulse,c['weight'][vs].cuda(),activation=None)
value_delta=(pre_delta.float()*slope[vs].cuda()[None]).transpose(1,2).reshape(1,width,p['num_v_heads'],p['head_v_dim'])

for group in p['fla_groups']:
    h=group['head_start'];heads=group['head_count'];t=group['focus_index']
    ep={k:v.cuda().contiguous() for k,v in group['native_endpoints'].items()}
    do=group['native_do'].cuda().contiguous();scale=group['scale']
    exponent=group['seed_exponent'];assert exponent is None or not bool(exponent.count_nonzero())
    native_vjps=[];native_forwards=[]
    def original_native_value_vjp(endpoints,seed,actual_scale):
        v=endpoints['v'][1:2].detach().requires_grad_(True)
        with torch.enable_grad():
            out,_=chunk_gated_delta_rule(q=endpoints['q'][1:2],k=endpoints['k'][1:2],v=v,
                g=endpoints['raw_g'][1:2],beta=endpoints['beta'][1:2],scale=actual_scale,
                initial_state=endpoints['h'][1:2,0].float().contiguous(),
                output_final_state=False,use_qk_l2norm_in_kernel=False)
            dv,=torch.autograd.grad(out,v,seed)
        native_vjps.append(dv.detach().float())
        return {'v':dv.float()}
    averaged=average_memory_endpoint_orders(original_native_value_vjp)(ep,do,scale)['v']
    saved=group['native_coefficients']['v'].cuda()
    difference=averaged-saved
    def original_native_value_forward(endpoints,seed,actual_scale):
        delta=torch.zeros_like(endpoints['v'][1:2]);delta[:,t:t+width]=value_delta[:,:,h:h+heads].to(delta.dtype)
        with torch.no_grad():
            response,_=chunk_gated_delta_rule(q=endpoints['q'][1:2],k=endpoints['k'][1:2],v=delta,
                g=endpoints['raw_g'][1:2],beta=endpoints['beta'][1:2],scale=actual_scale,
                initial_state=torch.zeros_like(endpoints['h'][1:2,0].float()),
                output_final_state=False,use_qk_l2norm_in_kernel=False)
        weighted=response.double()*seed.double()
        native_forwards.append(dict(endpoint='factual' if len(native_forwards)==0 else 'reference',
            total=float(weighted.sum()),next_position1871=float(weighted[:,t+1].sum()),
            per_head=weighted.sum((0,1,3)).cpu().tolist()))
        return {'per_time':weighted.sum((0,2,3)),'per_head':weighted.sum((0,1,3))}
    forward_average=average_memory_endpoint_orders(original_native_value_forward)(ep,do,scale)
    delta=torch.zeros_like(ep['v'][1:2]);delta[:,t:t+width]=value_delta[:,:,h:h+heads].to(delta.dtype)
    entry=dict(head_start=h,head_count=heads,
        observed_symmetric_VJP_dot=float((saved.double()*delta.double()).sum()),
        original_average_of_native_VJPs_dot=float((averaged.double()*delta.double()).sum()),
        coefficient_comparison=dict(max_abs=float(difference.abs().max()),
            relative_L2=float(difference.norm()/saved.norm()),
            no_asserted_arbitrary_tolerance=True),
        endpoint_native_forward_controls=native_forwards,
        original_average_native_forward=dict(total=float(forward_average['per_time'].sum()),
            next_position1871=float(forward_average['per_time'][t+1]),
            per_head=forward_average['per_head'].cpu().tolist()))
    result['groups'].append(entry);output_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(entry),flush=True)
    del ep,do,averaged,saved,difference,delta,native_vjps,forward_average
result.update(status='completed_original_symmetric_value_route_prediction',finished_unix=time.time())
result['total_original_average_native_forward']=sum(x['original_average_native_forward']['total'] for x in result['groups'])
result['total_original_average_from_next_position1871']=sum(x['original_average_native_forward']['next_position1871'] for x in result['groups'])
output_path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
