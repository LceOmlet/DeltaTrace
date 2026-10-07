"""Prepared-only Text5013 adapter using the original Text CPU preparation branch.

No App DT/head/trim migration, submit, CUDA/model, service or checkpoint call.
The metadata/linked-entry scaffold reuses the reviewed preparation owner;
Text options/data/parser imports execute the exact original Text prepare seam.
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
TEXT_OWNER = AUDIT/'direct-action-target-20261007/prepare_direct_entries.py'
TEXT_OWNER_SHA = '6e051913763887e454b257b498bd25b7e93a7fd0bba0d0f6ad3c0fa6495c47b8'
APP_SCAFFOLD = HERE/'prepare_appworld_prefix_runtime.py'
APP_SCAFFOLD_SHA = '914ae13f3eebff881a5db1fa12bb01a56f99aefbfd97316de519e5ef3c192171'
INTERFACE = AUDIT/'direct-target-prefix-interface-20261007/v1/candidate-textcraft'
TEXT_READOUT = '814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b'
TEXT_BASELINE = '5013ebc878b972a5f52817f7c7de7ae55ddae7f1d61609e943e1ab55bf59b993'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    return owner


def original_text_inspection(owner):
    tree=ast.parse(owner.CODE.replace('@ROOT@',repr('/unused-root')).replace('@OUT@',repr('/unused-out')))
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='inspect_code' for t in n.targets))
    code=ast.literal_eval(assignment.value)
    once=module('text_prepare_scaffold',APP_SCAFFOLD).replace_once
    code=once(code,"options,sampling=(launcher.options_for(P(data),output) if task=='TextCraft' else launcher.options_for(output))",
        "assert task=='TextCraft'\n"
        "options,sampling=(launcher.options_for(P(data),output) if task=='TextCraft' else launcher.options_for(output))\n"
        "baseline_options,baseline_sampling=launcher.options_for(P(data),P(sys.argv[5]))\n"
        "assert sampling==baseline_sampling")
    code=once(code,"modules['launcher']=launcher",r'''modules['launcher']=launcher
# Existing producer path setup, imports only, no instance or factory call.
qwen=json.loads(P(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
dt=P(os.environ['DT_ROOT']);official=P(os.environ.get('DT_OFFICIAL_ROOT') or qwen['official_root'])
sys.path[:0]=[str(dt),str(official),str(dt/'clean/qwen35')]
sys.path.append(qwen['ft_extension_root'])
for name in ['native_prefix_leases','qwen35_dense_finite_runner','qwen35_answer_finite',
             'qwen35_decoder_finite','qwen35_gdn_finite','qwen35_native_prefix_artifacts']:
 modules[name]=importlib.import_module(name)
assert hashlib.sha256(P(modules['qwen35_dense_finite_runner'].__file__).read_bytes()).hexdigest()=='5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'
assert hashlib.sha256(P(modules['qwen35_answer_finite'].__file__).read_bytes()).hexdigest()=='d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'
assert hashlib.sha256(P(modules['qwen35_decoder_finite'].__file__).read_bytes()).hexdigest()=='047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197'
assert hashlib.sha256(P(modules['qwen35_gdn_finite'].__file__).read_bytes()).hexdigest()=='ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'
assert hashlib.sha256(P(modules['native_prefix_leases'].__file__).read_bytes()).hexdigest()=='b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'
assert hashlib.sha256(P(modules['qwen35_native_prefix_artifacts'].__file__).read_bytes()).hexdigest()=='37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4'
assert hashlib.sha256(P(modules['reward_readout'].__file__).read_bytes()).hexdigest()=='814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b'
assert hashlib.sha256(P(modules['deltatrace_rollout'].__file__).read_bytes()).hexdigest()=='2aa5f55252a793458ff5bd569a5e203e18a99094e4fa3d3cd896b70c43b6b73e'
assert not torch.distributed.is_initialized()''')
    code=once(code,"source=lambda m:dict(path=inspect.getfile(m),sha256=hashlib.sha256(P(inspect.getfile(m)).read_bytes()).hexdigest())",
        "source=lambda m:dict(path=inspect.getfile(m),resolved_path=str(P(inspect.getfile(m)).resolve()),sha256=hashlib.sha256(P(inspect.getfile(m)).read_bytes()).hexdigest())")
    code=once(code,"options=options,sampling=sampling,imports=", "options=options,sampling=sampling,baseline_options=baseline_options,baseline_sampling=baseline_sampling,imports=")
    code=once(code,"owner_command=runtime.owner_command(options),cuda_initialized=False,",
        "owner_command=runtime.owner_command(options),cuda_initialized=False,distributed_initialized=False,")
    ast.parse(code)
    return code


def render(scaffold,head_owner,text_owner,root,out,commit):
    code=scaffold.render(head_owner,root,out,commit)
    once=scaffold.replace_once
    code=code.replace("runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt/source.json",
                      "runs/direct-action-target-20261007-v3/textcraft/textcraft-dt/source.json")
    code=code.replace('24b9e671f3533d88047fead3c709bbfddbf5200053f3cd8f3d308cb30f179f11',TEXT_BASELINE)
    code=code.replace('2001805','110053').replace('1791362313.39','1791344324.6')
    code=code.replace('AppWorld','TextCraft').replace("O/'appworld'","O/'textcraft'")
    code=code.replace('runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt',
                      'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt')
    code=code.replace('7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27',
                      '31e2acfb760eb1dd118af78a5a887898357feacc1b0f1caec04f06006a3b7d07')
    code=code.replace('b60251fdd3abd1687e5f8d6ca7415575acb0c88e60be30892a7d1513fe82c17c',TEXT_READOUT)
    for before,after in {
        '628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f':'5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555',
        '1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234':'047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197',
        '448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72':'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9',
        '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e':'d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e',
    }.items():code=code.replace(before,after)
    code=code.replace("O/'setup/","O/'textcraft-setup/").replace("O/'setup'","O/'textcraft-setup'")
    code=code.replace('prepare_appworld_prefix_runtime.py','prepare_textcraft_prefix_runtime.py')
    code=once(code,"assert env['LOOP_ROOT']==prior['loop_root']", "assert env.get('LOOP_ROOT')==prior['environment'].get('LOOP_ROOT')")
    code=code.replace("env['CUDA_VISIBLE_DEVICES']=='4,5'", "env['CUDA_VISIBLE_DEVICES']=='2,3'")
    code=code.replace("devices=[4,5]", "devices=[2,3]")
    # Replace only the import/options snippet with the frozen Text prepare's
    # own Text branch. The outer linked-entry/metadata/submit validator is shared.
    start=code.index('inspect_code=r"""')
    end=code.index('with (base/\'cpu-imports.log\')',start)
    code=code[:start]+'inspect_code='+repr(original_text_inspection(text_owner))+'\n'+code[end:]
    code=once(code,"[env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output),old['output']]",
        "[env['VENV_PYTHON'],'-c',inspect_code,str(base),'TextCraft',str(output),old['argv'][old['argv'].index('--data')+1],old['output']]")
    code=once(code,"assert after['resolved_path']==before['resolved_path'],name",
        "assert after['resolved_path']==str(P(before['path']).resolve()),name")
    code=code.replace("task_owner=prior['loop_root']", "task_owner=str(P(prior['actual_CPU_imports']['official_parser']['path']).parents[2])")
    code=once(code,"assert active_before=={name:sha(R/name) for name in active_before}",
        "active_after=read(R/'active-training.json')\n"
        "assert old==next(j for j in active_after['jobs'] if j['task']=='TextCraft'),'TextCraft owner changed during CPU preparation'\n"
        "active_after_hashes={name:sha(R/name) for name in active_before}")
    code=once(code,"dict(unchanged=True,before_and_after=active_before,observed_unix=time.time())",
        "dict(textcraft_owner_unchanged=True,before=active_before,after=active_after_hashes,"
        "global_manifests_changed={name:dict(before=digest,after=active_after_hashes[name]) for name,digest in active_before.items() if digest!=active_after_hashes[name]},"
        "AppWorld_before=next((j for j in active['jobs'] if j['task']=='AppWorld'),None),AppWorld_after=next((j for j in active_after['jobs'] if j['task']=='AppWorld'),None),observed_unix=time.time(),"
        "scope='Only TextCraft owner identity is locked; concurrent original AppWorld submission is reported without writing active manifests')")
    code=once(code,"original_prepare_owner=dict(path=", "original_prepare_owner=dict(TextCraft_CPU_owner=dict(path="+repr(str(TEXT_OWNER.resolve()).replace('\\','/'))+",sha256="+repr(sha(TEXT_OWNER))+"),shared_metadata_owner=dict(path=")
    code=once(code,"sha256='aaaaddd13aef5b37c4b3d7d0cadf5807fc14f5fe6356d5352e554a8a06608a26'),\n                status=",
        "sha256='aaaaddd13aef5b37c4b3d7d0cadf5807fc14f5fe6356d5352e554a8a06608a26')),\n                status=")
    code=code.replace('plus exact unchanged DT owner and candidate entry imports','plus exact unchanged Text DT owner and candidate Text entry imports')
    ast.parse(code)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);args=parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}',args.commit)
    assert sha(APP_SCAFFOLD)==APP_SCAFFOLD_SHA and sha(TEXT_OWNER)==TEXT_OWNER_SHA
    scaffold=module('frozen_app_prefix_scaffold',APP_SCAFFOLD)
    assert sha(scaffold.OWNER)==scaffold.OWNER_SHA and sha(scaffold.SUBMIT)==scaffold.SUBMIT_SHA
    head_owner=module('frozen_metadata_owner',scaffold.OWNER)
    text_owner=module('frozen_textcraft_prepare_owner',TEXT_OWNER)
    stage=text_owner.stage;out=stage.ROOT+'/candidates/direct-target-prefix-runtime-20261007-v1'
    assert sha(INTERFACE/'reward_readout.py')==TEXT_READOUT
    assert sha(INTERFACE/'deltatrace_rollout.py')==scaffold.FILES['deltatrace_rollout.py'][1]
    code=render(scaffold,head_owner,text_owner,stage.ROOT,out,args.commit)
    archive=HERE/'textcraft-setup-source.tar'
    with tarfile.open(archive,'w') as bundle:
        for name in scaffold.FILES:bundle.add(INTERFACE/name,arcname=name)
        bundle.add(Path(__file__),arcname=Path(__file__).name)
        bundle.add(scaffold.SUBMIT,arcname=scaffold.SUBMIT.name)
    script='set -eu\nsource '+stage.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (HERE/'textcraft-prepare-command.sh').write_text(script,encoding='utf-8')
    subprocess.run(stage.SSH+['mkdir','-p',out+'/textcraft-setup'],check=True)
    subprocess.run(stage.SCP+[str(archive),stage.SSH[-1]+':'+out+'/textcraft-setup-source.tar'],check=True)
    subprocess.run(stage.SSH+['tar','-xf',out+'/textcraft-setup-source.tar','-C',out+'/textcraft-setup'],check=True)
    result=subprocess.run(stage.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=240)
    (HERE/'textcraft-prepare.stdout.txt').write_bytes(result.stdout);(HERE/'textcraft-prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'));result.check_returncode()


if __name__=='__main__':main()
