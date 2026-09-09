"""Move controller diagnostics/synchronization, keeping native and finite math."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
source=root/'deltatrace/accelerated/qwen35/controller.py';raw=source.read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(raw)=='e3ac00cc1fc3c238f4cbf1f05f5af020641a1e232508235db8bcddc9f8efefba'
original=raw.decode().replace('\r\n','\n');s=original
replacements=[
 ('def _effect(m,x):return float((m.double()*(x[1::2].double()-x[0::2].double())).sum())',
  'def _effect(m,x):return (m.double()*(x[1::2].double()-x[0::2].double())).sum()'),
 ("root={};kwargs={};handles=[];calls=[];ledger={};head_shapes=[]", "root={};kwargs={};handles=[];calls=[];ledger={};head_shapes=[]\n        events=[];validity=[]"),
 ("            torch.cuda.synchronize();tick=time.perf_counter();value=fn();torch.cuda.synchronize()\n            calls.append({'kind':kind,'seconds':time.perf_counter()-tick,'allocated_after':torch.cuda.memory_allocated()})\n            return value", "            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)\n            begin.record();tick=time.perf_counter();value=fn();end.record()\n            calls.append({'kind':kind,'host_enqueue_seconds':time.perf_counter()-tick,'allocated_after':torch.cuda.memory_allocated()})\n            events.append((begin,end))\n            return value"),
 ("float((expected.float()-y.float()).norm()/expected.float().norm().clamp_min(1e-30))", "((expected.float()-y.float()).norm()/expected.float().norm().clamp_min(1e-30))"),
 ("float((aux.float()-actual_attention.float()).norm()/aux.float().norm().clamp_min(1e-30))", "((aux.float()-actual_attention.float()).norm()/aux.float().norm().clamp_min(1e-30))"),
 ("if not bool(torch.isfinite(new).all()):raise ValueError('Nonfinite DT coefficients.')", "validity.append(torch.isfinite(new).all())"),
 ('        return signed,info',"""        flags=torch.stack(validity).cpu().tolist()
        if not all(flags):raise ValueError('Nonfinite DT coefficients; deferred check failed before return.')
        for record,(begin,end) in zip(calls,events):
            record['stream_elapsed_seconds']=begin.elapsed_time(end)/1000
        scalar_tensors=[]
        def collect(value):
            if isinstance(value,torch.Tensor):
                if value.ndim!=0:raise ValueError('Unexpected nonscalar diagnostic')
                scalar_tensors.append(value)
            elif isinstance(value,dict):
                for item in value.values():collect(item)
            elif isinstance(value,(list,tuple)):
                for item in value:collect(item)
        collect(info)
        scalar_values=torch.stack([v.double() for v in scalar_tensors]).cpu().tolist()
        resolved={id(v):float(x) for v,x in zip(scalar_tensors,scalar_values)}
        def convert(value):
            if isinstance(value,torch.Tensor):return resolved[id(value)]
            if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
            if isinstance(value,list):return [convert(v) for v in value]
            if isinstance(value,tuple):return tuple(convert(v) for v in value)
            return value
        info=convert(info)
        info['controller_diagnostic_scheduling']={'all_32_finite_checks_passed':True,
            'deferred_scalar_count':len(scalar_values),'stage_timing':'CUDA stream elapsed plus host enqueue; no per-stage barrier'}
        info['complete_attribution_seconds_with_diagnostics']=time.perf_counter()-started
        return signed,info""")]
for old,new in replacements:
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
ast.parse(s);output=here/'qwen35_deferred_controller.py';output.write_text(s,encoding='utf-8',newline='\n')
(here/'qwen35_deferred_derivation.json').write_text(json.dumps({'source_sha256':sha(raw),'output_sha256':sha(output.read_bytes()),
    'changes':'only controller diagnostics/synchronization; no operator, model, FA or FLA changes',
    'diff':''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='controller.py',tofile=output.name))},indent=2)+'\n')
print('Generated controller with deferred diagnostics and stream event timing.')
