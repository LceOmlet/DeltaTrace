"""Read the native loss/backward/norm owners implicated by the real skip.

CPU/stdlib source inspection only: no worker RPC or model imports/changes.
"""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
code = r'''
import ast,difflib,hashlib,json,psutil,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
driver=psutil.Process(982372);assert driver.create_time()==1791553809.84
formal=root/'runs/textcraft-formal-stable-20261009-v1'
source=json.loads((formal/'source.json').read_bytes())
current=Path(source['verl_root']);pristine=root/'candidates/official-verl-20bd331'
requests={
 'verl/workers/actor/dp_actor.py':['__init__','_forward_micro_batch','_optimizer_step','update_policy'],
 'verl/trainer/ppo/core_algos.py':['compute_policy_loss','kl_penalty','agg_loss'],
 'verl/utils/fsdp_utils.py':['fsdp2_clip_grad_norm_'],
 'verl/utils/torch_functional.py':['entropy_from_logits','logprobs_from_logits','logprobs_from_logits_flash_attn','logprobs_from_logits_v2','masked_mean'],
 'verl/utils/experimental/torch_functional.py':[
  '_fused_linear_for_ppo_fwd','_fused_linear_for_ppo_bwd',
  'FusedLinearForPPOFunction','FusedLinearForPPO'],
}
files=[]
for relative,names in requests.items():
 path=current/relative;blob=path.read_bytes();text=blob.decode();tree=ast.parse(text)
 original=pristine/relative
 original_text=original.read_text() if original.exists() else None
 original_tree=ast.parse(original_text) if original_text is not None else None
 def functions(t):
  return {n.name:n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
 functions_here=functions(tree);functions_original=functions(original_tree) if original_tree else {}
 selected=[]
 for name in names:
  if name not in functions_here:continue
  node=functions_here[name];owner=functions_original.get(name)
  selected.append(dict(name=name,line=node.lineno,end_line=node.end_lineno,
   source=ast.get_source_segment(text,node),
   source_diff=None if owner is None else ''.join(difflib.unified_diff(
    original_text.splitlines(keepends=True)[max(0,owner.lineno-2):owner.end_lineno],
    text.splitlines(keepends=True)[max(0,node.lineno-2):node.end_lineno],
    fromfile='pinned-owner/'+name,tofile='active-source/'+name)),
   matches_pinned_owner_AST=owner is not None and ast.dump(node,include_attributes=False)==ast.dump(owner,include_attributes=False)))
 imports=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
 files.append(dict(path=str(path),resolved=str(path.resolve()),sha256=hashlib.sha256(blob).hexdigest(),
  official_path=str(original),
  official_sha256=hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,
  official_source=original_text if relative=='verl/utils/experimental/torch_functional.py' else None,
  imports=imports,functions=selected))
entry_hits=[]
for path in (formal/'entry').glob('*.py'):
 lines=path.read_text(errors='replace').splitlines()
 hits=[dict(line=i+1,context=lines[max(0,i-3):i+5]) for i,line in enumerate(lines)
  if any(key in line for key in ['inplace_backward','compute_entropy_from_logits','logprobs_from_logits','_forward_micro_batch','use_fused_kernels','apply_monkey_patch','fused_forward','qwen3_5'])]
 if hits:entry_hits.append(dict(path=str(path),resolved=str(path.resolve()),
  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hits=hits))
native_flag_lines=[]
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
for pid in [987808,989860]:
 path=next(ray.glob('*-'+str(pid)+'.out'))
 for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
  if any(key in line for key in ['Actor use_remove_padding=','Actor use_fused_kernels=','monkey patch','Monkey patch','fused kernel']):
   native_flag_lines.append(dict(path=str(path),line=number,text=line))
additional_sources=[]
for relative in ['verl/models/transformers/monkey_patch.py','verl/models/transformers/qwen3_5.py',
 'verl/models/transformers/torch_functional.py','verl/models/transformers/qwen3_vl.py',
 'verl/models/transformers/dense_common.py','verl/utils/experimental/torch_functional.py']:
 path=current/relative
 if path.is_file():additional_sources.append(dict(path=str(path),resolved=str(path.resolve()),
  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text=path.read_text()))
head_test=pristine/'tests/kernels/test_linear_cross_entropy.py'
owner_head_test=dict(path=str(head_test),sha256=hashlib.sha256(head_test.read_bytes()).hexdigest(),
 text=head_test.read_text()) if head_test.exists() else None
revision=subprocess.run(['git','-C',str(pristine),'rev-parse','HEAD'],capture_output=True,text=True) if shutil.which('git') else None
print(json.dumps(dict(unix=time.time(),driver=dict(pid=driver.pid,birth=driver.create_time()),
 upstream_commit=source['upstream_commit'],files=files,entry_hits=entry_hits,
 pristine_checkout_revision=revision.stdout.strip() if revision is not None and revision.returncode==0 else None,
 pristine_revision_unavailable_reason='git not installed on this host' if revision is None else revision.stderr.strip() if revision.returncode else None,
 owner_head_test=owner_head_test,
 native_flag_lines=native_flag_lines,additional_sources=additional_sources,
 model_calls=0,worker_RPC=0,production_changes=0)))
'''.replace('ROOT', repr(transport.ROOT), 1)
command = ('source ' + transport.ENTRY + '/metax-entry.env.sh\n'
           'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n')
r = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(), capture_output=True, timeout=45)
if r.returncode:
    raise RuntimeError(r.stderr.decode(errors='replace'))
r.check_returncode()
d = json.loads(r.stdout)
out = HERE/'direct-credit-records-20261009-v1'/('native-nonfinite-math-source-'+str(int(d['unix']))+'.json')
out.write_bytes(r.stdout)
print(json.dumps(dict(saved=str(out),files=[dict(path=f['path'],sha256=f['sha256'],
 functions=[dict(name=n['name'],line=n['line'],matches_pinned_owner_AST=n['matches_pinned_owner_AST'])
 for n in f['functions']]) for f in d['files']]),ensure_ascii=False))
