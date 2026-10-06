"""Prepare canonical owner imports and one default-inert callback composition.

No launch, model import, GPU work, checkpoint operation or formal file mutation.
This is an isolated prepared source candidate, not deployment or acceptance.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
OUT = HERE / 'production-wiring-v1'
ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
REMOTE_OUT = ROOT + '/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1'
PYTHON = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'

REMOTE_PREPARE = r'''
import ast,copy,difflib,hashlib,importlib.machinery,json,pathlib,time
root=pathlib.Path(@ROOT@);out=pathlib.Path(@OUT@)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source_path=root/'runs/appworld-fresh-native-conv-canonical-20261007-v3/appworld-dt/source.json'
assert sha(source_path)=='97e3cb754f505b79a52d3dc3464b9027ae7bdc38e7074b866f80dc7624be3481'
source=json.loads(source_path.read_bytes())
assert source['prior_source_sha256']=='3cd90b2db649cd477bc21398e7677dc8ea5534d5230296fc5f63c04037cd3427'
assert source['prepared_receipt_sha256']=='f5232ac1dc7cc13a6142902488fdb28243f86dd7de91774bb91c3279fe5c0ec3'
formal_dt=pathlib.Path(source['dt_root'])
formal_entry=root/'candidates/appworld-native-conv-initial-states-20261007-v1/entry'
assert not out.exists(), 'Do not overwrite any prepared candidate'
assert out.resolve().is_relative_to((root/'candidates/appworld-row-cuts-finite-20261007-v1').resolve())
bindings={str(formal_entry/k):v for k,v in source['entry_sha256'].items()}
bindings.update({str(formal_dt/k):v for k,v in source['dt_source_sha256'].items()})
for p,v in bindings.items():assert sha(p)==v,p
base=root/'candidates/appworld-row-cuts-finite-20261007-v1/combined-capacity-b8-v1'
owners={
 'qwen35_dense_finite_runner.py':('qwen35_dense_finite_runner_row_candidate.py','5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'),
 'qwen35_answer_finite.py':('qwen35_answer_finite.py','d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'),
 'qwen35_native_prefix_artifacts.py':('qwen35_native_prefix_artifacts.py','37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4'),
 'vendor_fa_finite_bf16_d256.py':('vendor_fa_finite_bf16_d256_row_candidate.py','3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089')}
lease=base/'native_prefix_leases.py'
assert sha(lease)=='b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'
library=root/'candidates/appworld-row-cuts-finite-20261007-v1/libfinite_row_query_starts.so'
assert sha(library)=='4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'
for canonical,(name,v) in owners.items():assert sha(base/name)==v,name
out.mkdir();entry=out/'entry';entry.mkdir();dt=out/'deltatrace';dt.mkdir()
links=[]
def children(old,new,exclude):
 for p in old.iterdir():
  if p.name not in exclude and p.name!='__pycache__':
   q=new/p.name;q.symlink_to(p,target_is_directory=p.is_dir())
   links.append(dict(path=str(q),target=str(p),directory=p.is_dir()))
children(formal_entry,entry,{'deltatrace_rollout.py','native_prefix_leases.py'})
children(formal_dt,dt,{'clean'})
(dt/'clean').mkdir();children(formal_dt/'clean',dt/'clean',{'qwen35'})
(dt/'clean/qwen35').mkdir();children(formal_dt/'clean/qwen35',dt/'clean/qwen35',set(owners))
for name,(target,v) in owners.items():
 (dt/'clean/qwen35'/name).symlink_to(base/target)
 assert sha(dt/'clean/qwen35'/name)==v
(entry/'native_prefix_leases.py').symlink_to(lease)
original=(formal_entry/'deltatrace_rollout.py').read_bytes()
assert hashlib.sha256(original).hexdigest()=='2c01c47e699b8b5a05206e881bda596cfbca03d1b89a7c3d3de2ce30ae5995a4'
(out/'frozen-deltatrace_rollout.py').write_bytes(original)
anchor="        self.readout_options['prefix_lease_factory'] = prepare_native_prefix_leases\n"
addition="""        if env.get('individual_prefixes', False) or env.get('boundary_row_storage', False):
            from functools import partial
            self.readout_options['prefix_lease_factory'] = partial(
                prepare_native_prefix_leases,
                individual_prefixes=env.get('individual_prefixes', False),
                boundary_row_storage=env.get('boundary_row_storage', False),
            )
"""
text=original.decode();assert text.count(anchor)==1
candidate=text.replace(anchor,anchor+addition)
(entry/'deltatrace_rollout.py').write_bytes(candidate.encode())
(out/'producer.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),candidate.splitlines(True),
 fromfile=str(formal_entry/'deltatrace_rollout.py'),tofile=str(entry/'deltatrace_rollout.py'))))
old_ast=ast.parse(original);new_ast=ast.parse(candidate)
expected=ast.dump(ast.parse(addition.strip()).body[0],include_attributes=False)
removed=[]
class ProjectDefault(ast.NodeTransformer):
 def visit_If(self,node):
  if ast.dump(node,include_attributes=False)==expected:
   removed.append(node.lineno);return None
  return self.generic_visit(node)
projected=ProjectDefault().visit(copy.deepcopy(new_ast))
assert len(removed)==1
assert ast.dump(projected,include_attributes=False)==ast.dump(old_ast,include_attributes=False)
numerical=pathlib.Path(source['candidate_environment']['path'])
assert sha(numerical)==source['candidate_environment']['sha256']
environment=json.loads(numerical.read_bytes());changed=copy.deepcopy(environment)
changed['qwen35'].update(finite_library=str(library),finite_library_sha256=sha(library),
 individual_prefixes=True,boundary_row_storage=True)
(out/'environment.json').write_text(json.dumps(changed,indent=2)+'\n')
changes={k:dict(before=environment['qwen35'].get(k),after=v) for k,v in changed['qwen35'].items() if environment['qwen35'].get(k)!=v}
assert set(changes)=={'finite_library','finite_library_sha256','individual_prefixes','boundary_row_storage'}
official=environment['qwen35']['official_root']
inherited=[p.replace(str(formal_dt),str(dt),1) if p==str(formal_dt) or p.startswith(str(formal_dt)+'/')
 else p.replace(str(formal_entry),str(entry),1) if p==str(formal_entry) or p.startswith(str(formal_entry)+'/') else p
 for p in source['pythonpath'].split(':')]
search=[str(dt),official,str(dt/'clean/qwen35'),str(entry),*inherited]
resolved={}
expected_imports={'deltatrace_rollout':entry/'deltatrace_rollout.py','native_prefix_leases':lease,
 **{name[:-3]:base/target for name,(target,_) in owners.items()},
 'qwen35_gdn_finite':formal_dt/'clean/qwen35/qwen35_gdn_finite.py',
 'finite_fla_gpu':formal_dt/'clean/qwen35/finite_fla_gpu.py',
 'native_dense_attention_capture':formal_dt/'clean/qwen35/native_dense_attention_capture.py',
 'native_target_logit_rows':formal_dt/'clean/qwen35/native_target_logit_rows.py',
 'reward_readout':formal_entry/'reward_readout.py'}
for name,p in expected_imports.items():
 spec=importlib.machinery.PathFinder.find_spec(name,search)
 assert spec and pathlib.Path(spec.origin).resolve()==p.resolve(),(name,None if spec is None else spec.origin,str(p))
 resolved[name]=dict(path=spec.origin,resolved_path=str(pathlib.Path(spec.origin).resolve()),sha256=sha(spec.origin))
for p,v in bindings.items():assert sha(p)==v,p
unchanged_factories={name:dict(path=str(dt/name),sha256=sha(dt/name),formal_sha256=sha(formal_dt/name))
 for name in ('profiles/official.py','profiles/qwen35_gdn_symmetric.py','accelerated/qwen35/qwen35_code_local_capture.py')}
for v in unchanged_factories.values():assert v['sha256']==v['formal_sha256']
receipt=dict(status='prepared_only_not_deployed_not_imported',observed_unix=time.time(),
 formal_source=dict(path=str(source_path),sha256=sha(source_path),inherited_frozen_sha256=source['prior_source_sha256'],prepared_sha256=source['prepared_receipt_sha256']),
 frozen_inventory_files=len(bindings),frozen_inventory_unchanged=True,
 producer=dict(path=str(entry/'deltatrace_rollout.py'),sha256=sha(entry/'deltatrace_rollout.py'),original_sha256=hashlib.sha256(original).hexdigest(),
 default_AST_equal=True,only_inserted_if_line=removed[0],callback_owner='native_prefix_leases.prepare_native_prefix_leases',
 configured_owner_kwargs=['individual_prefixes','boundary_row_storage']),
 dt_root=str(dt),entry=str(entry),environment=dict(path=str(out/'environment.json'),sha256=sha(out/'environment.json'),changed_qwen35_fields=changes),
 unchanged_factories=unchanged_factories,import_resolution=resolved,import_resolution_scope='stdlib PathFinder against exact producer prepend and candidate entry paths; modules were not executed',
 inherited_verl_root=source['verl_root'],inherited_loop_root=source['loop_root'],
 source_changes=['canonical runner/answer/artifact/finite-wrapper owner links','canonical entry prefix lease owner link','one default-inert partial callback configuration','four numerical-environment fields'],
 numerical_acceptance=False,launch=False,model=False,GPU=False,checkpoint=False,production_mutation=False,
 scope='No diagnostic class replacement or sys.modules injection. No PPO, vLLM, LOOP, LoRA, microbatch, trainer, Q/V/A or factory changes. Existing owners retain all computation.')
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(out=str(out),prepared_sha256=sha(out/'prepared.json'),producer=receipt['producer'],environment=receipt['environment'],import_resolution=resolved,frozen_inventory_unchanged=True,launch=False,GPU=False)))
'''


def main():
    spec = importlib.util.spec_from_file_location('_existing_entry_transport', AUDIT/'stage_environment_entry.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    OUT.mkdir(exist_ok=False)
    code = REMOTE_PREPARE.replace('@ROOT@', repr(ROOT)).replace('@OUT@', repr(REMOTE_OUT))
    script = f"CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES=-1 {PYTHON} - <<'PY'\n{code}\nPY\n"
    (OUT/'prepare-command.sh').write_text(script,encoding='utf8')
    result = subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
    (OUT/'prepare.stdout.json').write_bytes(result.stdout)
    (OUT/'prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    print('returncode',result.returncode)
    if result.returncode:
        raise SystemExit(result.returncode)
    for name in ('prepared.json','producer.diff','frozen-deltatrace_rollout.py'):
        subprocess.run(transport.SCP+[f'{transport.SSH[-1]}:{REMOTE_OUT}/{name}',str(OUT/name)],check=True,timeout=30)
    (OUT/'stager-source.json').write_text(json.dumps(dict(path=str(Path(__file__)),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),remote=REMOTE_OUT,prepared_only=True),indent=2)+'\n')


if __name__ == '__main__':
    main()
