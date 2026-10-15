"""Check the localized value route with the installed original native operators.

No transformer, policy update, full DT or single-token deletion is run. This
uses the exact saved layer30 routing operands and original public causal-conv
and FLA forward/autograd. The changed value vector is the finite propagation
increment attributable to the focus source, not a claimed real deletion.
"""
import hashlib,inspect,json,os,sys,time
from pathlib import Path
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-neg11873-rootcause-20261015-v1/layer30-mechanism-v3'
sys.path.insert(0,str(Path(os.environ['DT_ROOT'])/'clean/qwen35'))
from qwen35_gdn_finite import _scalar_secant,_silu_derivative
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.set_num_threads(2)
data=torch.load(DEST/'native-transport-v3-rank0.pt',map_location='cpu',weights_only=True,mmap=True)
p=data['metadata']['layer30_payload'];c=p['conv_silu']
key_dim=p['num_k_heads']*p['head_k_dim'];value_dim=p['num_v_heads']*p['head_v_dim'];width=p['conv_width']
value_slice=slice(2*key_dim,2*key_dim+value_dim)
dx=p['projected_qkv_pair'][1].float()-p['projected_qkv_pair'][0].float()
pre=c['pre'].float();output=c['output'].float()
slope=_scalar_secant(pre[0],pre[1],output[0],output[1],_silu_derivative(pre[0]))
result=dict(scope=__doc__,checkpoint=25,incident_exact_replay=False,transformer_calls=0,
    full_DT_calls=0,optimizer_updates=0,formal_training_changes=0,tolerance=None,
    source_bindings={},groups=[],started_unix=time.time())
for name,fn in [('original_conv',causal_conv1d_fn),('original_FLA',chunk_gated_delta_rule),
                ('original_scalar_secant',_scalar_secant)]:
    path=Path(inspect.getfile(fn));result['source_bindings'][name]=dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
outpath=DEST/'native-value-route-v4.json'

# A native linear convolution impulse. Keep only the source value projection
# changed. The native operator, not a copied convolution, determines its lags.
impulse=torch.zeros((1,value_dim,width),device='cuda',dtype=c['pre'].dtype)
impulse[0,:,0]=dx[value_slice].cuda().to(impulse.dtype)
weight=c['weight'][value_slice].cuda()
with torch.no_grad():
    native_pre_delta=causal_conv1d_fn(impulse,weight,activation=None)
native_value_delta=native_pre_delta.float()*slope[value_slice].cuda()[None]
native_value_delta=native_value_delta.transpose(1,2).reshape(1,width,p['num_v_heads'],p['head_v_dim'])

for group in p['fla_groups']:
    ep=group['native_endpoints'];h=group['head_start'];heads=group['head_count'];t=group['focus_index']
    q,k=[ep[n][1:2].cuda().contiguous() for n in ['q','k']]
    v=ep['v'][1:2].cuda().contiguous().detach().requires_grad_(True)
    g=ep['raw_g'][1:2].cuda().contiguous();beta=ep['beta'][1:2].cuda().contiguous()
    initial=ep['h'][1:2,0].float().cuda().contiguous()
    do=group['native_do'].cuda().contiguous()
    exponent=group['seed_exponent']
    assert exponent is None or not bool(exponent.count_nonzero()),'Record a nonzero exponent explicitly before this check.'
    # Original factual FLA public backward checks the saved finite V branch.
    with torch.enable_grad():
        out,_=chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,scale=group['scale'],
            initial_state=initial,output_final_state=False,use_qk_l2norm_in_kernel=False)
        dv,=torch.autograd.grad(out,v,do)
    saved=group['native_coefficients']['v'].cuda()
    diff=dv.float()-saved.float();denom=saved.float().norm()
    # With factual Q/K/beta/g fixed, the state recurrence is linear in V.
    # Native forward on a value increment with zero incoming increment tests
    # that propagation independently of the backward contraction above.
    delta=torch.zeros_like(v);delta[:,t:t+width]=native_value_delta[:,:,h:h+heads].to(v.dtype)
    with torch.no_grad():
        changed,_=chunk_gated_delta_rule(q=q,k=k,v=delta,g=g,beta=beta,scale=group['scale'],
            initial_state=torch.zeros_like(initial),output_final_state=False,use_qk_l2norm_in_kernel=False)
    weighted=changed.double()*do.double()
    effects=weighted.sum((0,2,3))
    entry=dict(head_start=h,head_count=heads,
        saved_native_value_VJP_comparison=dict(max_abs=float(diff.abs().max()),
            relative_L2=float(diff.norm()/denom),
            cast_to_native_dtype_exact=bool(torch.equal(dv,saved.to(dv.dtype))),
            focus_window_saved_dot=float((saved[:,t:t+width].double()*delta[:,t:t+width].double()).sum()),
            focus_window_native_VJP_dot=float((dv[:,t:t+width].double()*delta[:,t:t+width].double()).sum())),
        independent_native_value_forward=dict(total=float(effects.sum()),
            output_at_next_position1871=float(effects[t+1]),
            other_output_positions=float(effects.sum()-effects[t+1]),
            per_head=weighted.sum((0,1,3)).cpu().tolist(),
            largest_output_positions=[dict(position=int(i)+group['original_time_start'],effect=float(effects[i]))
                for i in effects.abs().topk(8).indices.tolist()]))
    result['groups'].append(entry)
    outpath.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(entry),flush=True)
    del ep,q,k,v,g,beta,initial,do,out,dv,saved,diff,denom,delta,changed,weighted,effects
result.update(status='completed_original_native_value_route_checks',finished_unix=time.time())
result['total_native_forward_effect']=sum(x['independent_native_value_forward']['total'] for x in result['groups'])
result['total_from_next_position1871']=sum(x['independent_native_value_forward']['output_at_next_position1871'] for x in result['groups'])
outpath.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)
