"""Check the existing finite FLA at coincident endpoints using owner tolerances.

Native FLA supplies the actual saved intermediates. The unchanged pinned FLA
FP32 recurrence supplies the reference. No initial/final-state adjoints are
requested: the DT interface starts with zero state and only consumes output
adjoints. Head partitioning calls the same finite owner; it is not another
recurrence. This does not certify nonzero finite interventions or whole PPO.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import fla.utils
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from finite_fla_gpu import make_compiled_finite_pullback, finite_fla_pullback, slice_native_fla_endpoints
from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders
from verify_official_kernel_tolerances import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coefficient-start',type=int,default=0)
    args=p.parse_args()
    source=args.sources/'test_gated_delta_v041.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
    test=load('official_fla_reference',source)
    assert not fla.utils.FLA_CI_ENV
    owner=average_memory_endpoint_orders(make_compiled_finite_pullback(dynamic_shapes=True))
    eager_owner=average_memory_endpoint_orders(finite_fla_pullback)
    result=dict(scope=__doc__,cases=[])
    stage=importlib.import_module('fla.ops.gated_delta_rule.chunk').chunk_gated_delta_rule_fwd
    for length in (128,447):
        torch.manual_seed(42)
        shape=(4,length,32,128)
        q,k=[F.normalize(torch.rand(shape,device='cuda',dtype=torch.bfloat16),p=2,dim=-1).requires_grad_() for _ in range(2)]
        v=torch.rand(shape,device='cuda',dtype=torch.bfloat16).requires_grad_()
        beta=torch.rand(shape[:-1],device='cuda').sigmoid().requires_grad_()
        g=F.logsigmoid(torch.rand(shape[:-1],device='cuda')).requires_grad_()
        inputs=(q,k,v,beta,g)
        endpoints={}
        def capture(frame,event,value):
            if frame.f_code is stage.__code__ and event=='return' and value is not None:
                for name in ('q','k','v','g','beta','A','w','v_new','h'):
                    endpoints[name]=frame.f_locals[name].detach().repeat_interleave(2,0)
        assert sys.getprofile() is None
        sys.setprofile(capture)
        try:
            out,_=chunk_gated_delta_rule(q=q,k=k,v=v,beta=beta,g=g,scale=128**-.5,
                initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False)
        finally:sys.setprofile(None)
        endpoints['raw_g']=g.detach().repeat_interleave(2,0)
        upstream=torch.randn_like(out)
        native=torch.autograd.grad(out,inputs,upstream)
        ref,_=test.recurrent_gated_delta_rule_ref(q,k,v,beta,g,scale=128**-.5,
            initial_state=None,output_final_state=False)
        reference=torch.autograd.grad(ref,inputs,upstream.float())
        coefficients={}
        with torch.no_grad():
            coefficients['finite_full']=owner(endpoints,upstream,128**-.5)
            parts=[owner({name:value[:,:,start:start+8].contiguous() for name,value in endpoints.items()},
                         upstream[:,:,start:start+8].contiguous(),128**-.5) for start in range(0,32,8)]
            coefficients['finite_head8']={name:torch.cat([part[name] for part in parts],2) for name in parts[0]}
            if args.coefficient_start:
                cut=args.coefficient_start
                selected=slice_native_fla_endpoints(endpoints,cut)
                parts=[owner({name:value[:,:,start:start+8].contiguous() for name,value in selected.items()},
                             upstream[:,cut:,start:start+8].contiguous(),128**-.5) for start in range(0,32,8)]
                coefficients['finite_suffix_head8']={name:torch.cat([part[name] for part in parts],2) for name in parts[0]}
                # Separate cropping arithmetic from compiler scheduling on
                # the same real native endpoints, using the uncompiled owner.
                eager={}
                for name,values,do in [('full',endpoints,upstream),('suffix',selected,upstream[:,cut:])]:
                    pieces=[eager_owner({k:v[:,:,head:head+8].contiguous() for k,v in values.items()},
                                        do[:,:,head:head+8].contiguous(),128**-.5) for head in range(0,32,8)]
                    eager[name]={k:torch.cat([piece[k] for piece in pieces],2) for k in pieces[0]}
                result.setdefault('slice_arithmetic_diagnostics',[]).append(dict(length=length,
                    eager_full_vs_suffix={k:dict(exact=torch.equal(eager['full'][k][:,cut:],v),
                        max_abs=float(fla.utils.get_abs_err(eager['full'][k][:,cut:],v))) for k,v in eager['suffix'].items()},
                    compiled_vs_eager_suffix={k:dict(max_abs=float(fla.utils.get_abs_err(eager['suffix'][k],v)),
                        rms_ratio=float(fla.utils.get_err_ratio(eager['suffix'][k],v))) for k,v in coefficients['finite_suffix_head8'].items()}))
        variants={'native':dict(zip(('q','k','v','beta','g'),native)),**coefficients}
        for variant,actual in variants.items():
            row=dict(length=length,batch=4,heads=32,variant=variant,quantities={})
            if variant=='finite_suffix_head8':
                row['coefficient_start']=args.coefficient_start
                row['saved_incoming_state_nonzero']=bool(endpoints['h'][:,args.coefficient_start//64].count_nonzero())
                row['exact_full_owner_suffix']={name:torch.equal(coefficients['finite_head8'][name][:,args.coefficient_start:],value)
                    for name,value in actual.items()}
                row['full_owner_suffix_differences']={name:dict(
                    max_abs=float(fla.utils.get_abs_err(coefficients['finite_head8'][name][:,args.coefficient_start:],value)),
                    rms_ratio=float(fla.utils.get_err_ratio(coefficients['finite_head8'][name][:,args.coefficient_start:],value)))
                    for name,value in actual.items()}
            for name,ref_grad in zip(('q','k','v','beta','g'),reference):
                if variant=='finite_suffix_head8':ref_grad=ref_grad[:,args.coefficient_start:]
                threshold=.02 if name in ('beta','g') else .008
                item=dict(rms_ratio=float(fla.utils.get_err_ratio(ref_grad,actual[name])),
                          max_abs=float(fla.utils.get_abs_err(ref_grad,actual[name])),threshold=threshold)
                try:
                    fla.utils.assert_close('d'+name,ref_grad,actual[name],threshold)
                    item['status']='passed'
                except Exception as exc:item.update(status='failed',error=str(exc))
                row['quantities'][name]=item
            row['status']='passed' if all(x['status']=='passed' for x in row['quantities'].values()) else 'failed'
            result['cases'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    result['status']='passed' if all(row['status']=='passed' for row in result['cases']) else 'failed'
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
