"""Launch a bounded original-native observation; never deploy or train."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP


def main(*, out_name='native-identity-roots-appworld-v1',
         remote_name='credit-native-identity-roots-appworld-20261009-v1',
         entry_script='inspect_native_identity_roots.py',additional_scripts=(),calls_per_rank=6,
         scope='Original 48 AppWorld trajectories / 16 states, twelve original B4 roots, no finite propagation'):
    out=HERE/out_name
    out.mkdir(exist_ok=True)
    assert not (out/'launch.json').exists(), 'Observe the original handle; do not repeat'
    runtime=json.loads((HERE.parents[4]/'experiments/rl/current_runtime.json').read_bytes())
    version=runtime['latest_fla_early_output_scale_20261009']
    binding=next(o for o in version['owners'] if o['task']=='appworld')
    assert version['version']=='fla-early-output-scale-20261009-v1'
    files=[HERE/'inspect_native_identity_roots.py',HERE/'layer-collection-inputs.json',
        HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/inspect_action_curve.py',
        HERE.parents[1]/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
    files=list(dict.fromkeys(files+[HERE/p for p in additional_scripts]+[HERE/entry_script]))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    target=ROOT+'/receipts/'+remote_name
    subprocess.run(SSH+['bash','-s'],input=('test ! -e '+target+' && mkdir '+target+'\n').encode(),check=True,timeout=30)
    subprocess.run(SCP+list(map(str,files))+[SSH[-1]+':'+target+'/'],check=True,timeout=50)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    body=r'''import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT);root=Path(ROOT)
for name,h in HASHES.items():
 p=out/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
 if p.suffix=='.py':ast.parse(p.read_text())
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
names=['qwen35_dense_finite_runner','qwen35_answer_finite','qwen35_gdn_finite','qwen35_decoder_finite','qwen35_native_prefix_artifacts']
owners={n:dict(source['actual_CPU_imports'][n]) for n in names}
owners['qwen35_gdn_finite']['sha256']=GDN['new_sha256']
for name,v in owners.items():assert hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'],name
assert Path(owners['qwen35_gdn_finite']['path']).resolve()==Path(GDN['versioned_path'])
spec=json.loads((out/'layer-collection-inputs.json').read_bytes())['tasks']['appworld']
assert len(spec['batches'])==12 and len(spec['entries'])==48
protocol=dict(scope=SCOPE,
 numerical_version=VERSION,numerical_owners=owners,plan_sha256=HASHES['layer-collection-inputs.json'],code_commit=COMMIT)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Diagnostic devices occupied'
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env.update(CUDA_VISIBLE_DEVICES='4,5',DT_TASK=source['startup_options']['env.env_name'],
 DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
argv=[env['VENV_PYTHON'],str(out/ENTRY_SCRIPT),'--source',str(source_path),'--output',str(out/'results'),'--case','appworld']
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),launched_unix=time.time(),code_commit=COMMIT,
 scripts=HASHES,argv=argv,devices=[4,5],protocol=protocol,native_root_B4_calls_per_rank=CALLS_PER_RANK,
 operations_excluded=['finite_seed','DT','optimizer','backward','rollout','checkpoint_restore'],
 formal_release=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''
    body='ROOT='+repr(ROOT)+'\nOUT='+repr(target)+'\nHASHES='+repr(hashes)+'\nGDN='+repr(binding)+'\nVERSION='+repr(version['version'])+'\nCOMMIT='+repr(commit)+'\nENTRY_SCRIPT='+repr(entry_script)+'\nCALLS_PER_RANK='+repr(calls_per_rank)+'\nSCOPE='+repr(scope)+'\n'+body
    command='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    (out/'command.sh').write_text(command,encoding='utf-8',newline='\n')
    result=subprocess.run(SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=55)
    (out/'launch.stdout').write_bytes(result.stdout);(out/'launch.stderr').write_bytes(result.stderr)
    result.check_returncode()
    receipt=json.loads(result.stdout)
    (out/'launch.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('physical_before','protocol')},indent=2))


if __name__=='__main__':main()
