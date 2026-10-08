"""Stage frozen original-owner development diagnostics on free physical GPU4/5.

No candidate math, training, environment execution, restore or package setup.
The default launch composes the previously used native actor initializer.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import ENTRY,ROOT,SSH,SCP

REMOTE=ROOT+'/receipts/credit-author-development-collection-20261008-v1'
OWNER=AUDIT/'direct-target-action-author-curve-20261007/v1'
FILES=[HERE/'inspect_author_collection.py',HERE/'verify_author_batching.py',OWNER/'inspect_action_curve.py',
       AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']


if __name__=='__main__':
    corpus=json.loads((HERE/'corpus.json').read_bytes())
    manifest=json.loads((HERE/'manifest.json').read_bytes())
    strata=json.loads((HERE/'credit-strata.json').read_bytes())
    assert strata['provenance']['manifest.json']==hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()
    assert strata['provenance']['corpus.json']==hashlib.sha256((HERE/'corpus.json').read_bytes()).hexdigest()
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}
    # The later TextCraft-tested helper preserves the same metric/carrier and
    # additionally passes original TaskRunner-resolved scheduler steps.
    assert hashes['inspect_action_curve.py']=='7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6'
    assert hashes['inspect_extreme_endpoint.py']=='8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
    plan={'scope':__doc__,'seed':manifest['seed'],'metric_owner':manifest['metric_owner'],
          'diagnostic_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'frozen_manifest_sha256':hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),
          'corpus_sha256':hashlib.sha256((HERE/'corpus.json').read_bytes()).hexdigest(),
          'uniform_query_scope':'Four positions sampled uniformly without replacement within each frozen trajectory, independent of d/A/reward. Diagnostic model endpoints only; no output refinement or training change.',
          'credit_strata_sha256':hashlib.sha256((HERE/'credit-strata.json').read_bytes()).hexdigest(),
          'tail_scope':'Census of every observed ratio>2 source in the completed frozen development captures. Report (2,10], (10,100], >100 separately; no pooling with the bounded bulk or any training-credit replacement.',
          'tail_queries':{},'tasks':{}}
    for task,data in manifest['tasks'].items():
        records={r['traj_uid']:r for r in corpus['tasks'][task]['records']}
        files={f['sha256']:f for f in corpus['tasks'][task]['native_files']}
        entries=[]
        for group in data['groups']:
            if group['split']!='development':continue
            for uid in group['first_stage_uids']:
                record=records[uid]
                occurrence=record['native_occurrences'][0]
                entries.append({'task':task,'traj_uid':uid,'initial_state_sha256':group['initial_state_sha256'],
                    'previously_examined_group':group['previously_examined'],
                    'native':files[occurrence['file_sha256']],'batch_row':occurrence['batch_row'],
                    'source_tokens':occurrence['source_tokens'],'native_signed_sha256':occurrence['native_signed_sha256'],
                    'reward':record['reward']})
        assert len(entries)==32 and len({e['initial_state_sha256'] for e in entries})==16
        plan['tasks'][task]=entries
        plan['tail_queries'][task]=strata['tasks'][task]['complete_observed_ratio_gt_2_tail']
    subprocess.run(SSH+['bash','-s'],input=('set -eu\nmkdir -p '+REMOTE+'\n').encode(),check=True,timeout=30)
    for path in FILES:
        subprocess.run(SCP+[str(path),SSH[-1]+':'+REMOTE+'/'+path.name],check=True,timeout=45)
    body=r'''
import ast,hashlib,json,os,psutil,random,re,subprocess,sys,time
from pathlib import Path
import torch
torch.set_num_threads(1)
root=Path(ROOT);out=Path(OUT);plan=PLAN
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(t):
 t=t.detach().cpu().contiguous();h=hashlib.sha256(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes());return h.hexdigest()
assert not (out/'launch.json').exists(),'Do not duplicate this collection job'
for name,h in HASHES.items():
 assert sha(out/name)==h
 ast.parse((out/name).read_text())
for task,entries in plan['tasks'].items():
 for entry in entries:
  assert sha(entry['native']['path'])==entry['native']['sha256']
  native=torch.load(entry['native']['path'],map_location='cpu',weights_only=False)
  row=next(r for r in native['rows'] if r['batch_row']==entry['batch_row'])
  assert str(row['traj_uid'])==entry['traj_uid']
  positions=row['prompt_length']+row['prior'][row['suffix_positions']].nonzero().flatten()
  assert positions.numel()==entry['source_tokens'] and positions.numel()>=20
  assert row['selected'].numel()<=32768
  seed=int(hashlib.sha256((plan['seed']+'\0uniform-development-diagnostic\0'+task+'\0'+entry['traj_uid']).encode()).hexdigest(),16)
  indices=random.Random(seed).sample(range(positions.numel()),4)
  signed=native['native_signed'][row['batch_row'],positions]
  assert tensor_sha(signed)==entry['native_signed_sha256']
  entry['uniform_queries']=[{'source_index':i,'packed_slot':int(positions[i]),'token_id':int(row['selected'][positions[i]]),
     'saved_d':float(signed[i]),'inclusion_probability':4/positions.numel()} for i in indices]
  del native
plan['prepared_cpu_cuda_initialized']=torch.cuda.is_initialized()
assert not plan['prepared_cpu_cuda_initialized']
(out/'collection-inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
p=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(p.read_bytes());assert sha(p)=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
assert psutil.Process(2833207).create_time()==1791370325.16
for rank in (0,1):assert not (root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'/f'rank{rank}-release-update').exists()
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Selected devices are occupied'
(out/'before-physical.txt').write_text(physical)
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
assert sha(Path(official)/'ft_ifr_improve.py')==plan['metric_owner']['sha256']
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
sys.path[:0]=env['PYTHONPATH'].split(':')
from verify_author_batching import verify
bridge_receipt=verify(out/'batching-cpu-replay.json',official)
argv=[env['VENV_PYTHON'],str(out/'inspect_author_collection.py'),'--source',str(p),'--output',str(out/'results')]
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':process.pid,'birth':psutil.Process(process.pid).create_time(),'launched_unix':time.time(),
 'devices':[4,5],'argv':argv,'scripts':HASHES,'code_commit':plan['diagnostic_code_commit'],
 'batching_cpu_replay':{'path':str(out/'batching-cpu-replay.json'),'sha256':sha(out/'batching-cpu-replay.json')},
 'plan_path':str(out/'collection-inputs.json'),
 'plan_sha256':sha(out/'collection-inputs.json'),'source_path':str(p),'source_sha256':sha(p),
 'source_environment':{k:source['environment'][k] for k in ('DT_ROOT','DT_ENVIRONMENT_JSON','DT_OFFICIAL_ROOT','VENV_PYTHON','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','HF_HOME','HF_HUB_CACHE','PYTORCH_CUDA_ALLOC_CONF') if k in source['environment']},
 'scope':plan['scope'],'frozen_manifest_sha256':plan['frozen_manifest_sha256'],
 'expected_native_forward_calls_per_rank':184+max((len(q)+3)//4 for q in plan['tail_queries'].values()),'trajectories':64,'initial_states_per_task':16,
 'complete_development_tail_queries':{t:len(q) for t,q in plan['tail_queries'].items()},
 'uniform_single_deletions':256,'DT_calls':0,'optimizer_steps':0,'formal_restart':False,'checkpoint_restore':False,
 'training_credit_refinement':False,'new_GDN_candidate':False,'text_update_released':False}
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
    body='ROOT='+repr(ROOT)+'\nOUT='+repr(REMOTE)+'\nPLAN='+repr(plan)+'\nHASHES='+repr(hashes)+'\n'+body
    shell='set -eu\nsource '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    (HERE/'author-collection-launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
    run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=90)
    (HERE/'author-collection-launch.stderr.txt').write_bytes(run.stderr)
    if run.returncode:print(run.stderr.decode(errors='replace'))
    run.check_returncode();result=json.loads(run.stdout)
    result['local_submit_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'author-collection-launch.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_environment','argv')},ensure_ascii=False))
