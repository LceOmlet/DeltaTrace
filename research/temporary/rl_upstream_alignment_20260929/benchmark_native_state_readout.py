"""Measure one installed FLA state interface at the observed prefix geometry.

This is an operator/API probe, not a model or DT accuracy/performance result.
Use the pinned owner's input fixture, native forward and native state readout.
No recurrence, backward, model, cache policy or tolerance is implemented here.
"""
from pathlib import Path
import hashlib
import subprocess

from stage_environment_entry import ENTRY, ROOT, REPO, remote


script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export CUDA_VISIBLE_DEVICES=6
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,inspect,json,pathlib,psutil,time
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from accelerated.native_capture_events import LocalCaptureEvents

root=pathlib.Path('@ROOT@')
out=root/'receipts/owner-b8-dispatch-20260930'/f'native-state-interface-{int(time.time())}'
out.mkdir()
record=dict(scope='Single native operator/API probe on idle GPU6, B4 T7488 H32 K/V128 FP16. Not full-model or DT parity or a production cache implementation.',
 pid=psutil.Process().pid,created_unix=psutil.Process().create_time(),
 diagnostic_commit='@COMMIT@',diagnostic_sha256='@SHA@',readouts=[])
path=out/'result.json'
def save():path.write_text(json.dumps(record,indent=2)+'\n')
save();print(json.dumps({'receipt':str(out),'pid':record['pid']}),flush=True)
free,total=torch.cuda.mem_get_info()
if free<16*1024**3:
 record['not_run']='Insufficient physical headroom for the optional operator probe'
 save();raise SystemExit(0)
source=root/'receipts/training-setup/official-kernel-tests/test_gated_delta_v041.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_chunk')
fixture=[]
for node in fn.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='tri' for x in t.elts) for t in node.targets):break
 fixture.append(node)
namespace=dict(torch=torch,F=F,IS_INTEL_ALCHEMIST=False,B=4,T=7488,H=32,D=128,
 dtype=torch.float16,device='cuda',gate_logit_normalizer=1.,mask_p=0.)
exec(compile(ast.fix_missing_locations(ast.Module(body=fixture,type_ignores=[])),str(source),'exec'),namespace)
captured={}
def event(frame,kind,value):
 if kind=='return' and value is not None:
  f=frame.f_locals
  for key in ('k','w','u','g','h','initial_state','final_state'):
   captured[key]=f[key].detach() if f[key] is not None else None
torch.cuda.reset_peak_memory_stats()
tick=time.perf_counter()
with torch.no_grad(),LocalCaptureEvents([chunk_gated_delta_rule_fwd.__code__],event,returns_only=True):
 native,native_state=chunk_gated_delta_rule(q=namespace['q'],k=namespace['k'],v=namespace['v'],
  g=namespace['g'],beta=namespace['beta'].to(torch.float16),
  scale=128**-.5,initial_state=namespace['h0'],output_final_state=True,
  use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize();record['native_forward_with_first_compile_seconds']=time.perf_counter()-tick
record['source_fixture']={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
owner_source=pathlib.Path(inspect.getsourcefile(chunk_gated_delta_rule_fwd_h))
record['state_owner']={'path':str(owner_source),'sha256':hashlib.sha256(owner_source.read_bytes()).hexdigest()}
record.update(intermediate_dtype=str(captured['h'].dtype),native_cache_dtype=str(native_state.dtype))
for length in (7488,3712):
 a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
 a.record();tick=time.perf_counter()
 with torch.no_grad():
  h,v_new,state=chunk_gated_delta_rule_fwd_h(k=captured['k'][:,:length].contiguous(),
   w=captured['w'][:,:length].contiguous(),u=captured['u'][:,:length].contiguous(),
   g=captured['g'][:,:length].contiguous(),initial_state=captured['initial_state'],output_final_state=True)
 b.record();b.synchronize()
 row=dict(tokens=length,host_and_wait_seconds=time.perf_counter()-tick,
  stream_seconds=a.elapsed_time(b)/1000,state_dtype=str(state.dtype),
  state_bytes=state.numel()*state.element_size(),intermediate_state_bytes=h.numel()*h.element_size())
 if length==7488:
  row.update(equal_to_original_cache_state=bool(torch.equal(state,native_state)),
   max_absolute_difference=float((state-native_state).abs().max()))
 else:
  # Original h holds the incoming state at each 64-token boundary in FP16.
  # This equality checks location identity only, not FP32 state accuracy.
  original=captured['h'][:,length//64]
  row.update(rounded_equal_to_original_intermediate=bool(torch.equal(state.to(original.dtype),original)),
   rounded_max_absolute_difference=float((state.to(original.dtype)-original).abs().max()))
 record['readouts'].append(row);del h,v_new,state
record.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
 physical_free_bytes=torch.cuda.mem_get_info()[0],rss_bytes=psutil.Process().memory_info().rss,
 pss_bytes=psutil.Process().memory_full_info().pss,finished_unix=time.time())
save();print(json.dumps(record),flush=True)
PY
'''

if __name__ == '__main__':
    remote(script.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY)
           .replace('@COMMIT@', subprocess.check_output(
               ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
