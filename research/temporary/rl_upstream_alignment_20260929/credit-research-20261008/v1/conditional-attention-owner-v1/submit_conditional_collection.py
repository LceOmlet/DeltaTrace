"""Launch one paired, frozen development comparison using original owners."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COLLECTION = HERE.parent
AUDIT = COLLECTION.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('textcraft', 'appworld'), required=True)
    parser.add_argument('--candidate-kind', choices=('conditional_attention','endpoint_head'), default='conditional_attention')
    args = parser.parse_args()
    task = args.task
    endpoint = args.candidate_kind == 'endpoint_head'
    output_folder = COLLECTION/'endpoint-head-owner-v1' if endpoint else HERE
    output_folder.mkdir(exist_ok=True)
    remote = ROOT + ('/receipts/endpoint-head-collection-'+task+'-20261009-v1' if endpoint
                     else '/receipts/conditional-collection-'+task+'-20261008-v3')
    if not endpoint:
        compiled = json.loads((HERE/'compiled-owner.json').read_bytes())
        composition = json.loads((HERE/'composition-prepared.json').read_bytes())
    manifest = json.loads((COLLECTION/'manifest.json').read_bytes())
    frozen = json.loads((COLLECTION/'layer-collection-inputs.json').read_bytes())
    data = frozen['tasks'][task]
    entries = {e['traj_uid']: e for e in data['entries']}
    primary = [uid for group in manifest['tasks'][task]['groups']
               if group['split'] == 'development' for uid in group['first_stage_uids']]
    assert len(primary) == len(set(primary)) == 32
    extras = sorted(set(entries)-set(primary), key=lambda uid: (entries[uid]['selected_tokens'], uid))
    batches = []
    for uids, is_primary in [(primary, True), (extras, False)]:
        for offset in range(0, len(uids), 4):
            selected = uids[offset:offset+4]
            batches.append(dict(uids=selected, actual_rows=len(selected), primary=is_primary))
    assert len(batches) == 12
    for offset in range(0, len(batches), 2):
        count = max(len(entries[uid]['queries']) for b in batches[offset:offset+2] for uid in b['uids'])
        for b in batches[offset:offset+2]:
            b['query_rounds'] = (count+1)//2
    generated = None if endpoint else HERE/'composition-prepared'/task
    files = [HERE/'inspect_conditional_collection.py',
        COLLECTION/'inspect_author_collection.py',
        AUDIT/'direct-target-action-author-curve-20261007/v1/inspect_action_curve.py',
        AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
    if endpoint:
        assert task=='textcraft', 'First fixed comparison only; no automatic second-task launch'
        derivation=json.loads((COLLECTION/'endpoint-head-derivation.json').read_bytes())
        assert derivation['status']=='derived_only_unaccepted'
        files += [COLLECTION/'endpoint_head_seed.py', HERE/'textcraft-v3-rank0.json', HERE/'textcraft-v3-rank1.json']
    else:
        files += [HERE/'conditional_attention_endpoints.py', generated/'qwen35_decoder_finite.py', generated/'qwen35_dense_finite_runner.py']
    for p in files:
        if p.suffix=='.py': ast.parse(p.read_text(encoding='utf8'))
    if not endpoint:
        for item in composition['tasks'][task]:
            assert sha(generated/item['generated_name']) == item['generated_sha256']
    hashes = {p.name: sha(p) for p in files}
    assert hashes['inspect_action_curve.py'] == '7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6'
    assert hashes['inspect_extreme_endpoint.py'] == '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
    if endpoint:
        head=json.loads((COLLECTION/'layer-suboperations-textcraft-v2-observations/rank0.json').read_bytes())['owners']['qwen35_answer_finite']
        candidate=dict(kind='endpoint_head', seed=remote+'/endpoint_head_seed.py', head_owner=head,
            files=[dict(path=remote+'/endpoint_head_seed.py',sha256=hashes['endpoint_head_seed.py'])])
    else:
        verified = json.loads((HERE/'original-fa-check.json').read_bytes())
        library = compiled['remote_directory']+'/libfinite_conditional_research.so'
        wrapper = str(Path(library).parent/'vendor_fa_finite_bf16_d256.py').replace('\\', '/')
        candidate = dict(decoder=remote+'/candidate/qwen35_decoder_finite.py',
            runner=remote+'/candidate/qwen35_dense_finite_runner.py',
            library=library, library_sha256=compiled['library_sha256'], wrapper=wrapper,
            files=[dict(path=remote+'/candidate/'+item['generated_name'], sha256=item['generated_sha256'])
                   for item in composition['tasks'][task]])
        candidate['files'] += [dict(path=remote+'/conditional_attention_endpoints.py', sha256=hashes['conditional_attention_endpoints.py']),
            dict(path=library, sha256=compiled['library_sha256']),
            dict(path=wrapper, sha256=verified['launch']['candidate']['vendor_fa_finite_bf16_d256.py'])]
    task_spec=dict(entries=data['entries'],batches=batches,candidate=candidate,
                   wall_budget_seconds=1800 if task=='textcraft' else 4200)
    if endpoint:
        task_spec.update(candidate_label='endpoint_head',original_baseline=[
            dict(path=remote+'/textcraft-v3-rank'+str(i)+'.json',sha256=hashes['textcraft-v3-rank'+str(i)+'.json']) for i in range(2)])
    plan = dict(scope=__doc__, frozen_manifest_sha256=sha(COLLECTION/'manifest.json'),
        frozen_layer_inputs_sha256=sha(COLLECTION/'layer-collection-inputs.json'),
        metric_owner=manifest['metric_owner'], tasks={task:task_spec},
        controls='Original fresh actor and task-specific actual owners; B4, LoRA8/16, same dtype, same Y and identities. No optimizer, rollout, checkpoint restore, credit replacement or test-set tuning.',
        primary_trajectories=32, initial_states=16, extra_tail_trajectories=len(extras),
        native_calls_per_rank=[sum((21 if b['primary'] else 0) if endpoint else
            ((42 if b['primary'] else 1)+b['query_rounds']) for b in batches[rank::2]) for rank in range(2)],
        DT_calls_per_rank=6 if endpoint else 12,
        candidate_kind=args.candidate_kind,
        derivation_sha256=sha(COLLECTION/'endpoint-head-derivation.json') if endpoint else None)
    (output_folder/('comparison-inputs-'+task+'.json')).write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    subprocess.run(SSH+['mkdir','-p',remote+'/candidate'],check=True,timeout=30)
    for p in files:
        destination = remote+('/candidate/' if generated is not None and p.parent == generated else '/')+p.name
        subprocess.run(SCP+[str(p),SSH[-1]+':'+destination],check=True,timeout=45)
    subprocess.run(SCP+[str(output_folder/('comparison-inputs-'+task+'.json')),SSH[-1]+':'+remote+'/comparison-inputs.json'],check=True,timeout=45)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source_sha = ('2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
                  if task=='textcraft' else '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0')
    body = '''import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT)
assert not (out/'launch.json').exists(),'Do not duplicate this comparison launch'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in HASHES.items():
 p=out/('candidate' if name in ('qwen35_decoder_finite.py','qwen35_dense_finite_runner.py') else '')/name
 assert sha(p)==h
 if p.suffix=='.py':ast.parse(p.read_text())
plan=json.loads((out/'comparison-inputs.json').read_bytes())
for item in plan['tasks'][TASK]['candidate']['files']:assert sha(item['path'])==item['sha256']
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/TASK/(TASK+'-dt')/'source.json'
assert sha(source_path)==SOURCE_SHA
source=json.loads(source_path.read_bytes())
runner=source['actual_CPU_imports']['qwen35_dense_finite_runner']
assert sha(runner['path'])==runner['sha256']
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\\|\\s*[45]\\s+\\d+\\s+\\S',physical,re.M),'Research devices occupied'
(out/'before-physical.txt').write_text(physical)
env=dict(os.environ,**source['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
# The candidate directory is deliberately absent: original imports stay original.
dt=Path(source['dt_root']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
assert sha(Path(official)/'ft_ifr_improve.py')==plan['metric_owner']['sha256']
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
# Resolve the demonstrated missing metric import before loading the actor.
subprocess.run([env['VENV_PYTHON'],'-c','import ft_ifr_improve; import torch; assert not torch.cuda.is_initialized()'],env=dict(env,CUDA_VISIBLE_DEVICES='-1'),check=True,stdout=subprocess.DEVNULL,timeout=45)
argv=[env['VENV_PYTHON'],str(out/'inspect_conditional_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',TASK]
if TASK=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert sha(evidence)=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as stream:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(task=TASK,pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),
 code_commit=COMMIT,scripts=HASHES,argv=argv,devices=[4,5],source_path=str(source_path),source_sha256=SOURCE_SHA,
 actual_recorded_runner=runner,comparison_inputs_sha256=sha(out/'comparison-inputs.json'),
 operations=dict(optimizer=0,scheduler=0,rollout=0,checkpoint_restore=0),
 diagnostic_DT_B4_calls_per_rank=plan['DT_calls_per_rank'],wall_budget_per_worker_seconds=plan['tasks'][TASK]['wall_budget_seconds'],
 native_calls_per_rank=plan['native_calls_per_rank'],candidate_deployed=False,scope=plan['controls'])
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
    body = ('ROOT='+repr(ROOT)+'\nOUT='+repr(remote)+'\nTASK='+repr(task)+'\nHASHES='+repr(hashes)+
        '\nSOURCE_SHA='+repr(source_sha)+'\nCOMMIT='+repr(commit)+'\n'+body)
    shell = 'source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    (output_folder/(task+'-launch-command.sh')).write_text(shell,encoding='utf8',newline='\n')
    result = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=60)
    (output_folder/(task+'-launch.stderr.txt')).write_bytes(result.stderr)
    if result.returncode:
        print(result.stderr.decode(errors='replace'))
        result.check_returncode()
    receipt = json.loads(result.stdout)
    receipt['local_launcher_sha256'] = sha(__file__)
    (output_folder/(task+'-launch.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
