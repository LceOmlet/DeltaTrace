"""Stage/launch one bounded existing-rule diagnosis; no production deployment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
OUT = transport.ROOT + '/receipts/direct-target-existing-pv-rule-20261008-v1'
files = [HERE / 'compare_existing_pv_rule.py',
         AUDIT / 'direct-target-native-mlp-memory-20261007/v2/profile_existing_offload.py',
         AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
parser = argparse.ArgumentParser()
parser.add_argument('--launch', action='store_true')
kind = parser.add_mutually_exclusive_group()
kind.add_argument('--curve', action='store_true', help='Separate original-author curve comparison; no new DT')
kind.add_argument('--layer', action='store_true', help='Original passive layer diagnosis on the current largest negative token')
kind.add_argument('--subops', action='store_true', help='Observe decoder31/30 branches and stop after decoder30; no full attribution is produced')
kind.add_argument('--gdn', action='store_true', help='Observe the original decoder30 norm/gate and finite FLA calls; save actual finite operands')
kind.add_argument('--memory', action='store_true', help='Compare the existing averaged/forward memory callbacks; other rules unchanged')
kind.add_argument('--memory-curve', action='store_true', help='Original author curves for the completed memory callback comparison')
kind.add_argument('--code-fence-only', action='store_true', help='One bounded original DT isolating the actual next code-fence score; unchanged original reference inputs')
kind.add_argument('--clean-gdn', action='store_true', help='Compare owner-preserved clean-v1 GDN rule defaults; current execution/kernels retained, no production switch')
kind.add_argument('--clean-gdn-curve', action='store_true', help='Original author curves for the completed preserved GDN rule comparison')
kind.add_argument('--factual-v', action='store_true', help='Diagnostic existing factual V coefficient with other symmetric fields; no conservation correction or production change')
kind.add_argument('--factual-v-curve', action='store_true', help='Original author curves for the completed factual-V diagnostic vector')
parser.add_argument('--revision',type=int,default=1,help='Distinct receipt directory for a corrected GDN diagnostic')
args = parser.parse_args()
if args.revision != 1 and not args.gdn:parser.error('revision is only used for GDN diagnostics')
if args.curve or args.memory_curve or args.clean_gdn_curve or args.factual_v_curve:
    OUT += '-factual-v-curves' if args.factual_v_curve else ('-clean-gdn-curves' if args.clean_gdn_curve else ('-memory-curves' if args.memory_curve else '-curves'))
    files = [HERE / 'compare_existing_pv_curves.py',
             AUDIT / 'direct-target-action-author-curve-20261007/v1/inspect_action_curve.py',
             AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
elif args.layer or args.subops or args.gdn:
    OUT += '-gdn' if args.gdn else ('-subops' if args.subops else '-layers')
    files = [AUDIT / 'direct-target-token-layer-effect-20261007/v1/inspect_layer_effect.py',
             AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
elif args.memory:
    OUT += '-memory'
elif args.code_fence_only:
    OUT += '-code-fence'
elif args.clean_gdn:
    OUT += '-clean-gdn'
elif args.factual_v:
    OUT += '-factual-v'
if args.gdn and args.revision != 1:OUT += '-v' + str(args.revision)
for p in files:
    compile(p.read_bytes(), str(p), 'exec')
subprocess.run(transport.SSH + ['bash', '-s'], input=('set -eu\nmkdir -p ' + OUT + '\n').encode(), check=True)
for p in files:
    subprocess.run(transport.SCP + [str(p), transport.SSH[-1] + ':' + OUT + '/' + p.name], check=True)
code = r"""
import hashlib,json,os,re,subprocess,time
from pathlib import Path
import psutil
root=Path(@ROOT@);out=Path(@OUT@);hashes=@HASHES@
for name,expected in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_sha=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
for key,expected in [('changed_file','changed_sha256'),('candidate_environment','candidate_environment_sha256')]:
 assert hashlib.sha256(Path(candidate[key]).read_bytes()).hexdigest()==candidate[expected]
native=root/'receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank1-readout-native-batch-16.pt'
assert hashlib.sha256(native.read_bytes()).hexdigest()=='3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a'
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK=source['startup_options']['env.env_name'],DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=candidate['candidate_dt_root'],DT_ENVIRONMENT_JSON=candidate['candidate_environment'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
if not @LAUNCH@:
 env['CUDA_VISIBLE_DEVICES']=''
 if @CURVE@:
  script='''import json,torch,hashlib
from pathlib import Path
root=Path(ROOT);out=Path(OUT);native=Path(NATIVE)
original=torch.load(native,map_location='cpu',weights_only=False)
assert len(original['rows'])==4
record=dict(scope='Original real B4, only the saved attribution vector is replaced by the two already-measured existing-rule vectors for author-metric evaluation',ranks={})
profiles=[(0,'original_symmetric_memory'),(1,'existing_forward_memory')] if @MEMORY_CURVE@ else [(0,'original_content1'),(1,'existing_content0')]
comparison='direct-target-existing-pv-rule-20261008-v1-memory' if @MEMORY_CURVE@ else 'direct-target-existing-pv-rule-20261008-v1'
if @CLEAN_CURVE@:
 profiles=[(0,'original_symmetric_memory'),(1,'existing_clean_gdn')]
 comparison='direct-target-existing-pv-rule-20261008-v1-clean-gdn'
if @FACTUAL_V_CURVE@:
 profiles=[(0,'original_symmetric_memory'),(1,'diagnostic_factual_V')]
 comparison='direct-target-existing-pv-rule-20261008-v1-factual-v'
for rank,mode in profiles:
 trace=root/'receipts'/comparison/'results'/('rank0-'+mode+'.pt')
 value=torch.load(trace,map_location='cpu',weights_only=False)
 assert value['native_sha256']==hashlib.sha256(native.read_bytes()).hexdigest()
 assert value['signed'].shape==original['native_signed'].shape
 other=torch.load(trace.with_name('rank1-'+mode+'.pt'),map_location='cpu',weights_only=False)
 assert torch.equal(value['signed'],other['signed'])
 hybrid=dict(original,native_signed=value['signed'],detail=value['detail'])
 hybrid['provenance']=dict(original_native_path=str(native),original_native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),attribution_path=str(trace),attribution_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),scope='Quality-evaluation carrier only, not a formal native capture')
 path=out/(mode+'-quality-input.pt');torch.save(hybrid,path)
 row=sorted(original['rows'],key=lambda x:x['batch_row'])[3]
 assert row['traj_uid']=='f0f85f5c-74b4-4670-a074-99a3a09dbb98' and row['trajectory_index']==64 and int(row['selected'][2883])==198
 case=dict(source_sha256='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0',native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),uid=row['traj_uid'],row=3,trajectory_index=64,response_slot=125,packed_slot=2883,token_id=198,rank=1,batch=16,native=str(path))
 record['ranks'][str(rank)]=dict(profile=mode,case=case,provenance=hybrid['provenance'],metric_source='Unchanged ft_ifr_improve.faithfulness_test_skip_tokens, k=20, signed RISE and positive-only MAS evaluation; original inspect_action_curve owner')
record['cuda_initialized']=torch.cuda.is_initialized();(out/'curve-inputs.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''.replace('ROOT',repr(str(root))).replace('OUT',repr(str(out))).replace('NATIVE',repr(str(native)))
 else:
  script='''import json,torch
from pathlib import Path
from reward_readout import DirectActionTargetReadout
d=torch.load(NATIVE,map_location='cpu',weights_only=False);result=[]
for item in sorted(d['rows'],key=lambda x:x['batch_row']):
 p=DirectActionTargetReadout._prepare_row(item['row'],0)
 assert torch.equal(p['selected'],item['selected']) and torch.equal(p['case']['target_ids'],item['case']['target_ids']) and p['target_offsets']==item['target_offsets']
 result.append(dict(uid=item['traj_uid'],length=p['selected'].numel(),targets=len(p['target_offsets'])))
 if @FENCE@ and item['batch_row']==3:
  first=p['target_offsets'][0]
  assert p['prompt_length']+first-1==2883 and int(p['case']['target_ids'][first])==71093
  result[-1].update(isolated_actual_target_id=71093,isolated_actual_predictor_position=2883)
print(json.dumps(dict(rows=result,cuda_initialized=torch.cuda.is_initialized(),scope='CPU input identity only, no model/DT/update')))
'''.replace('NATIVE',repr(str(native)))
 if @LAYER@:
  case=dict(source_sha256=source_sha,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),uid='f0f85f5c-74b4-4670-a074-99a3a09dbb98',row=3,trajectory_index=64,response_slot=125,packed_slot=2883,token_id=198,rank=1,batch=16,native=str(native))
  script+='\nfrom inspect_extreme_endpoint import load_request\nload_request('+repr(str(source_path))+','+repr(str(native))+','+repr(case)+')\nassert not torch.cuda.is_initialized()\n'
 r=subprocess.run([env['VENV_PYTHON'],'-c',script],env=env,cwd=out,capture_output=True)
 (out/'prepare.stdout.txt').write_bytes(r.stdout);(out/'prepare.stderr.txt').write_bytes(r.stderr);r.check_returncode()
 (out/'prepared.json').write_bytes(r.stdout);print(r.stdout.decode())
 if @LAYER@:
  binding=dict(case=case,runner_sha256=candidate['changed_sha256'],scope='Current actual extreme token on original unchanged B4; passive original single/joint boundary diagnosis only')
  if @SUBOPS@:binding.update(probe_decoder_layers=[31,30],stop_after_decoder=30)
  if @GDN@:binding.update(probe_gdn_layers=[30])
  (out/'case.json').write_text(json.dumps(binding,indent=2)+'\n')
else:
 assert (out/'prepared.json').exists() and not (out/'launch.json').exists()
 assert psutil.Process(2833207).create_time()==1791370325.16
 held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
 assert not any((held/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
 physical=subprocess.run(['mx-smi'],capture_output=True,check=True).stdout;(out/'before-physical.txt').write_bytes(physical)
 assert not any(re.match(r'^\|\s*[45]\s+\d+\s+\S',line) for line in physical.decode(errors='replace').splitlines())
 env['CUDA_VISIBLE_DEVICES']='4,5'
 program='inspect_layer_effect.py' if @LAYER@ else ('compare_existing_pv_curves.py' if @CURVE@ else 'compare_existing_pv_rule.py')
 argv=[env['VENV_PYTHON'],str(out/program),'--source',str(source_path)]
 if @MEMORY@:argv+=['--memory']
 if @FENCE@:argv+=['--code-fence-only']
 if @CLEAN@:argv+=['--clean-gdn']
 if @FACTUAL_V@:argv+=['--factual-v']
 if @LAYER@:argv+=['--case',str(out/'case.json')]
 elif not @CURVE@:argv+=['--native',str(native)]
 argv+=['--output',str(out/'results')]
 with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),code_commit=@COMMIT@,argv=argv,scripts=hashes,source_sha256=source_sha,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),memory_candidate=candidate,devices=[4,5],DT_calls_per_rank=0 if @CURVE@ else 2,native_forwards_per_rank=42 if @CURVE@ else 0,scope=('Original passive single/joint layer comparison, current largest negative token' if @LAYER@ else ('Existing averaged/forward memory diagnostic only' if @MEMORY@ or @MEMORY_CURVE@ else 'Existing content1/content0 diagnostic only'))+'; no profile deployment, no parameter/optimizer/scheduler updates, no rollout or checkpoint restore',formal_restart=False,credit_repaired=False,text_update_released=False)
 if @LAYER@:record['case_binding']=dict(path=str(out/'case.json'),sha256=hashlib.sha256((out/'case.json').read_bytes()).hexdigest(),value=json.loads((out/'case.json').read_bytes()))
 if @SUBOPS@:record.update(partial_DT_only=True,stop_after_decoder=30,full_signed_vector_produced=False)
 if @FENCE@:record.update(DT_calls_per_rank=1,scope='Same original real B4 and original original-profile DT; isolate the next actual code-fence score only, without changing reference IDs or retained target input tokens; diagnostic-only, no complete-event QVA export, profile switch, update, rollout or restore')
 if @CLEAN@:record.update(scope='Compare the owner-preserved clean-v1 GDN numerical rule defaults on the same actual B4; current offload/chunking/kernels/target/QVA/PPO retained, no frozen-runtime restore or formal profile switch',preserved_clean_owner_sha256='e5acd0b43d75677e0416e5b856dfad31f460562ef1968f02ac2357ba268b3cf0')
 if @CLEAN_CURVE@:record.update(scope='Unchanged original author cumulative deletion/RISE/MAS for the completed original-symmetric versus preserved-clean-GDN full vectors; same actual trajectory and native B4 scorer, no DT, optimizer, rollout, restore or formal profile change')
 if @FACTUAL_V@:record.update(scope='Diagnostic composition of existing factual V coefficients and remaining symmetric coefficients, justified by separately measured conditional V native effect; actual joint conservation residual retained without correction. Same input/target/reward/QVA/PPO, no production rule change.')
 if @FACTUAL_V_CURVE@:record.update(scope='Unchanged original author cumulative deletion/RISE/MAS for original-symmetric versus diagnostic factual-V full vectors; same actual trajectory and native B4 scorer, no DT, optimizer, rollout, restore or formal change')
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
"""
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
for k, v in dict(ROOT=transport.ROOT, OUT=OUT, HASHES={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, LAUNCH=args.launch, CURVE=args.curve or args.memory_curve or args.clean_gdn_curve or args.factual_v_curve, MEMORY=args.memory, MEMORY_CURVE=args.memory_curve, CLEAN_CURVE=args.clean_gdn_curve, FACTUAL_V_CURVE=args.factual_v_curve, FACTUAL_V=args.factual_v, FENCE=args.code_fence_only, CLEAN=args.clean_gdn, LAYER=args.layer or args.subops or args.gdn, SUBOPS=args.subops or args.gdn, GDN=args.gdn, COMMIT=commit).items():
    code = code.replace('@' + k + '@', repr(v))
command = 'set -eu\nsource ' + transport.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
prefix = 'gdn-' if args.gdn else ('subops-' if args.subops else ('layer-' if args.layer else ('curve-' if args.curve else '')))
if args.memory or args.memory_curve:prefix='memory-curve-' if args.memory_curve else 'memory-'
if args.code_fence_only:prefix='code-fence-'
if args.clean_gdn:prefix='clean-gdn-'
if args.clean_gdn_curve:prefix='clean-gdn-curve-'
if args.factual_v:prefix='factual-v-'
if args.factual_v_curve:prefix='factual-v-curve-'
if args.gdn and args.revision != 1:prefix = 'gdn-v' + str(args.revision) + '-'
phase = prefix + ('launch' if args.launch else 'prepare')
(HERE / (phase + '-command.sh')).write_text(command, encoding='utf8', newline='\n')
r = subprocess.run(transport.SSH + ['bash', '-s'], input=command.encode(), capture_output=True)
(HERE / (phase + '.stdout.txt')).write_bytes(r.stdout)
(HERE / (phase + '.stderr.txt')).write_bytes(r.stderr)
print(r.stdout.decode(errors='replace'))
if r.returncode:
    print(r.stderr.decode(errors='replace'))
r.check_returncode()
(HERE / (prefix + ('launch.json' if args.launch else 'prepared.json'))).write_bytes(r.stdout)
