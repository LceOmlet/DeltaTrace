"""Stage a separate entry around actual source24b9, using the frozen prepare owner.

Only the two reviewed prefix-interface files become local entry files. The
original DT/VERL/LOOP owners, model/assets/caches and launch budget are reused.
No submit, active-manifest mutation, model, CUDA, episode or checkpoint call.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
OWNER = AUDIT / 'direct-target-head-memory-20261007/v2/prepare_appworld_head_memory.py'
OWNER_SHA = 'aaaaddd13aef5b37c4b3d7d0cadf5807fc14f5fe6356d5352e554a8a06608a26'
SUBMIT = AUDIT / 'direct-target-semantics-20261007/submit_prepared_direct_targets.py'
SUBMIT_SHA = '8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
INTERFACE = AUDIT / 'direct-target-prefix-interface-20261007/v1/candidate'
FILES = {
    'deltatrace_rollout.py': ('494b53b0f40831f379a427b9e47df879487e55d8bbddb4541ad0be6f3a47cace',
                            '2aa5f55252a793458ff5bd569a5e203e18a99094e4fa3d3cd896b70c43b6b73e'),
    'reward_readout.py': ('7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27',
                         'b60251fdd3abd1687e5f8d6ca7415575acb0c88e60be30892a7d1513fe82c17c'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(code, before, after):
    assert code.count(before) == 1, before
    return code.replace(before, after, 1)


def render(owner, root, out, commit):
    code = owner.CODE
    code = code.replace('runs/direct-action-target-20261007-v3/appworld/appworld-dt/source.json',
                        'runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt/source.json')
    code = code.replace('70ffdcfd05fccb94c0cec70e5b1d83d8728e3335bb9f1a4e6bbec9299f53847b',
                        '24b9e671f3533d88047fead3c709bbfddbf5200053f3cd8f3d308cb30f179f11')
    code = code.replace('167065', '2001805').replace('1791344807.34', '1791362313.39')
    code = replace_once(code, "active=read(R/'active-training.json')",
        "active_before={name:sha(R/name) for name in ('active-training.json','active-source.json','formal-training.json')}\n"
        "active=read(R/'active-training.json')")
    start = code.index("base=O/'appworld';dt=O/'deltatrace';old_dt=P(prior['dt_root'])")
    end = code.index("output=R/'runs/direct-target-head-memory-20261007-v2/appworld/appworld-dt'", start)
    code = code[:start] + r'''base=O/'appworld';dt=P(prior['dt_root']);old_dt=dt
entry=base/'entry';old_entry=P(prior['entry'])
assert not base.exists(),'Preserve previous preparation attempts'
for directory,mapping in [(prior['entry'],prior['entry_sha256']),
                          (prior['verl_root'],prior['owner_head_sha256']),
                          (dt,prior['dt_source_sha256'])]:
 for name,digest in mapping.items():assert sha(P(directory)/name)==digest,(directory,name)
for path,digest in prior['source_bindings'].items():assert sha(path)==digest,path
assert prior['dt_source_sha256']['clean/qwen35/qwen35_dense_finite_runner.py']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
assert prior['dt_source_sha256']['clean/qwen35/qwen35_decoder_finite.py']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'
assert prior['dt_source_sha256']['clean/qwen35/qwen35_gdn_finite.py']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
assert prior['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
replacements=@REPLACEMENTS@
for name,(before,after) in replacements.items():
 assert prior['entry_sha256'][name]==before,name
 assert sha(O/'setup'/name)==after,name
# The existing owner's linked-file transport, applied only to the entry.
entry.mkdir(parents=True)
for original in old_entry.rglob('*'):
 relative=original.relative_to(old_entry)
 if '__pycache__' in relative.parts or '.git' in relative.parts:continue
 path=entry/relative
 if original.is_dir() and original.is_symlink():
  path.symlink_to(original.resolve(),target_is_directory=True)
 elif original.is_dir():path.mkdir(exist_ok=True,parents=True)
 elif original.is_file():
  path.parent.mkdir(exist_ok=True,parents=True);path.symlink_to(original.resolve())
for name,(before,after) in replacements.items():
 replacement=entry/name
 assert replacement.is_symlink()
 replacement.unlink();replacement.write_bytes((O/'setup'/name).read_bytes())
 assert not replacement.is_symlink() and sha(replacement)==after,name
 assert sha(old_entry/name)==before,name
entry_hashes={name:sha(entry/name) for name in prior['entry_sha256']}
changed=[name for name,digest in entry_hashes.items() if digest!=prior['entry_sha256'][name]]
assert set(changed)==set(replacements) and len(changed)==2,changed
dt_hashes={name:sha(dt/name) for name in prior['dt_source_sha256']}
assert dt_hashes==prior['dt_source_sha256']
''' + code[end:]
    code = code.replace('runs/direct-target-head-memory-20261007-v2',
                        'runs/direct-target-prefix-runtime-20261007-v1')
    code = replace_once(code, "value=str(path);prefix=str(old_dt)\n return str(dt)+value[len(prefix):] if value==prefix or value.startswith(prefix+'/') else value",
        "value=str(path);prefix=str(old_entry)\n return str(entry)+value[len(prefix):] if value==prefix or value.startswith(prefix+'/') else value")
    code = replace_once(code, "\nenv['DT_ROOT']=str(dt)\n", "\nenv['DT_ROOT']=str(dt)\nenv['DT_ENTRY_ROOT']=str(entry)\n")
    code = replace_once(code, "assert env['DT_ENTRY_ROOT']==prior['entry'] and env['VERL_ROOT']==prior['verl_root']",
        "assert env['DT_ENTRY_ROOT']==str(entry) and env['VERL_ROOT']==prior['verl_root']")
    code = replace_once(code, "options,sampling=launcher.options_for(output)",
        "options,sampling=launcher.options_for(output)\n"
        "baseline_options,baseline_sampling=launcher.options_for(P(sys.argv[3]))\n"
        "assert sampling==baseline_sampling")
    code = replace_once(code, "'owner_trajectory_batch','deltatrace_rollout','executed_target_spans','verl.trainer.ppo.ray_trainer',",
        "'owner_trajectory_batch','deltatrace_rollout','executed_target_spans','native_prefix_leases','verl.trainer.ppo.ray_trainer',")
    code = replace_once(code, "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner']:",
        "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite','qwen35_decoder_finite','qwen35_native_prefix_artifacts']:")
    code = code.replace("'5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'",
                        "'628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'")
    code = replace_once(code, "assert imports['verl.workers.actor.dp_actor']['sha256']==",
        "assert imports['reward_readout']['sha256']=='b60251fdd3abd1687e5f8d6ca7415575acb0c88e60be30892a7d1513fe82c17c'\n"
        "assert imports['deltatrace_rollout']['sha256']=='2aa5f55252a793458ff5bd569a5e203e18a99094e4fa3d3cd896b70c43b6b73e'\n"
        "assert imports['native_prefix_leases']['sha256']=='b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'\n"
        "assert imports['qwen35_native_prefix_artifacts']['sha256']=='37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4'\n"
        "assert imports['qwen35_decoder_finite']['sha256']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'\n"
        "assert imports['qwen35_gdn_finite']['sha256']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'\n"
        "assert imports['verl.workers.actor.dp_actor']['sha256']==")
    code = replace_once(code, "options=options,sampling=sampling,imports=imports,",
        "options=options,sampling=sampling,baseline_options=baseline_options,baseline_sampling=baseline_sampling,imports=imports,")
    code = replace_once(code, "[env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output)]",
        "[env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output),old['output']]")
    code = replace_once(code, "cwd=prior['entry'],env=cpu", "cwd=str(entry),env=cpu")
    code = replace_once(code, "old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']",
        "old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']\n"
        "assert inspection['baseline_options']=={key:(remap(value) if key=='data.custom_cls.path' else value) for key,value in old_options.items()}\n"
        "assert inspection['sampling']==inspection['baseline_sampling']\n"
        "for name,before in prior['actual_CPU_imports'].items():\n"
        " after=inspection['imports'][name]\n"
        " if name in ('reward_readout','deltatrace_rollout'):\n"
        "  assert after['path']==str(entry/(name+'.py')) and after['resolved_path']==after['path'],name\n"
        "  continue\n"
        " assert after['sha256']==before['sha256'],name\n"
        " assert after['path']==remap(before['path']),name\n"
        " assert after['resolved_path']==before['resolved_path'],name")
    code = replace_once(code, "assert set(differences)<=allowed,differences",
        "assert set(differences)<=allowed,differences\n"
        "for key,change in differences.items():\n"
        " if key=='data.custom_cls.path':\n"
        "  assert change['after']==remap(change['before'])\n"
        "  assert P(change['after']).is_symlink() and P(change['after']).resolve()==P(change['before']).resolve()\n"
        "  assert sha(change['after'])==sha(change['before'])\n"
        " else:assert change['after']==change['before'].replace(old['output'],str(output)),key\n"
        "environment_differences={key:dict(before=prior['environment'].get(key),after=env.get(key))\n"
        " for key in env.keys()|prior['environment'].keys() if env.get(key)!=prior['environment'].get(key)}\n"
        "assert set(environment_differences)<= {'DT_ENTRY_ROOT','PYTHONPATH'},environment_differences")
    code = replace_once(code, "allowed={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',",
        "allowed={'data.custom_cls.path','trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',")
    code = replace_once(code, "if not (path==str(old_dt) or path.startswith(str(old_dt)+'/'))}",
        "if not (path==str(old_entry) or path.startswith(str(old_entry)+'/'))}")
    code = replace_once(code, "bindings.update({str(dt/name):digest for name,digest in dt_hashes.items()})",
        "bindings.update({str(dt/name):digest for name,digest in dt_hashes.items()})\n"
        "bindings.update({str(entry/name):digest for name,digest in entry_hashes.items()})")
    code = replace_once(code, "('qwen35_answer_finite.py','prepare_appworld_head_memory.py','submit_prepared_direct_targets.py')",
        "('deltatrace_rollout.py','reward_readout.py','prepare_appworld_prefix_runtime.py','submit_prepared_direct_targets.py')")
    code = replace_once(code, "source=dict(prior,dt_root=str(dt),dt_source_sha256=dt_hashes,source_bindings=bindings,",
        "source=dict(prior,entry=str(entry),entry_sha256=entry_hashes,dt_root=str(dt),dt_source_sha256=dt_hashes,source_bindings=bindings,")
    code = replace_once(code, "official_configuration_differences=differences,",
        "official_configuration_differences=differences,environment_differences=environment_differences,")
    code = replace_once(code, "head_memory_preparation=dict(changed_dt_files=changed,previous_driver=previous,",
        "prefix_interface_preparation=dict(changed_entry_files=changed,changed_dt_files=[],previous_driver=previous,")
    code = replace_once(code, "baseline_launch=binding(old_launch),candidate_answer=binding(answer),",
        "baseline_launch=binding(old_launch),candidate_files={name:binding(entry/name) for name in replacements},")
    code = code.replace('prepare_appworld_head_memory.py', 'prepare_appworld_prefix_runtime.py')
    code = replace_once(code, "if 'DT_ROOT' in resource_env:resource_env['DT_ROOT']=str(dt)",
        "if 'DT_ROOT' in resource_env:resource_env['DT_ROOT']=str(dt)\n"
        " if 'DT_ENTRY_ROOT' in resource_env:resource_env['DT_ENTRY_ROOT']=str(entry)")
    code = replace_once(code, "argv=[str(output) if value==old['output'] else value for value in old['argv']]",
        "argv=[str(output) if value==old['output'] else remap(value) for value in old['argv']]")
    code = code.replace("environment=env,entry=prior['entry'],", "environment=env,entry=str(entry),")
    code = code.replace("configuration_differences=differences,entry=prior['entry'],", "configuration_differences=differences,entry=str(entry),")
    code = replace_once(code, "checked=submit.check_prepared(O,'AppWorld')",
        "checked=submit.check_prepared(O,'AppWorld')\n"
        "assert active_before=={name:sha(R/name) for name in active_before}\n"
        "for name,digest in prior['entry_sha256'].items():assert sha(old_entry/name)==digest,name\n"
        "assert dt_hashes=={name:sha(dt/name) for name in dt_hashes}\n"
        "(base/'active-preservation.json').write_text(json.dumps(dict(unchanged=True,before_and_after=active_before,observed_unix=time.time()),indent=2)+'\\n')")
    code = replace_once(code, "answer=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],",
        "head=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],\n"
        "    producer=inspection['imports']['deltatrace_rollout'],readout=inspection['imports']['reward_readout'],")
    code = code.replace('Real executed joint action targets; isolated full-vocabulary target-row temporary tiling;',
        'Real executed joint action targets; existing prefix factory/provider interface restored;')
    code = code.replace('plus explicit answer/runner imports', 'plus exact unchanged DT owner and candidate entry imports')
    code = code.replace('Preparation and CPU imports only;', 'Prepared-only isolated entry, exactly two interface files changed; CPU imports only;')
    values = {'@ROOT@': repr(root), '@OUT@': repr(out), '@COMMIT@': commit,
              '@ANSWER_SHA@': '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e',
              '@REPLACEMENTS@': repr(FILES), '@ORIGINAL_PREPARE_PATH@': str(OWNER.resolve()).replace('\\', '/'),
              '@ORIGINAL_PREPARE_SHA@': OWNER_SHA}
    for token,value in values.items():code=code.replace(token,value)
    assert not re.search(r'@[A-Z_]+@',code)
    ast.parse(code)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}',args.commit)
    assert sha(OWNER)==OWNER_SHA and sha(SUBMIT)==SUBMIT_SHA
    spec=importlib.util.spec_from_file_location('frozen_prefix_prepare_owner',OWNER)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    stage=owner.stage
    out=stage.ROOT+'/candidates/direct-target-prefix-runtime-20261007-v1'
    for name,(_,digest) in FILES.items():assert sha(INTERFACE/name)==digest,name
    code=render(owner,stage.ROOT,out,args.commit)
    archive=HERE/'setup-source.tar'
    with tarfile.open(archive,'w') as bundle:
        for name in FILES:bundle.add(INTERFACE/name,arcname=name)
        bundle.add(Path(__file__),arcname=Path(__file__).name)
        bundle.add(SUBMIT,arcname=SUBMIT.name)
    script='set -eu\nsource '+stage.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (HERE/'prepare-command.sh').write_text(script,encoding='utf-8')
    subprocess.run(stage.SSH+['mkdir','-p',out+'/setup'],check=True)
    subprocess.run(stage.SCP+[str(archive),stage.SSH[-1]+':'+out+'/setup-source.tar'],check=True)
    subprocess.run(stage.SSH+['tar','-xf',out+'/setup-source.tar','-C',out+'/setup'],check=True)
    result=subprocess.run(stage.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=240)
    (HERE/'prepare.stdout.txt').write_bytes(result.stdout)
    (HERE/'prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
    result.check_returncode()


if __name__=='__main__':main()
