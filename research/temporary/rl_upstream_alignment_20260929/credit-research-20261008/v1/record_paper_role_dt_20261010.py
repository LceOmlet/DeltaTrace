"""Collect completed owner outputs; preserve rejected diagnostics separately."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP, REPO
OUT=HERE/'paper-role-implementation-check-20261010-v3'
REMOTE=ROOT+'/receipts/paper-role-implementation-check-20261010-v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

for name in ('result.json','gdn-symmetric-v1.pt','clean-v1.pt','driver.log'):
    subprocess.run(SCP+[SSH[-1]+':'+REMOTE+'/'+name,str(OUT/name)],check=True,timeout=35)
v2=HERE/'paper-role-implementation-check-20261010-v2'
for name in ('result.json','driver.log'):
    subprocess.run(SCP+[SSH[-1]+':'+ROOT+'/receipts/paper-role-implementation-check-20261010-v2/'+name,
                        str(v2/name)],check=True,timeout=35)

body=r'''
import hashlib,json,psutil,subprocess,time,torch
from pathlib import Path
p=Path(REMOTE);l=json.loads((p/'launch.json').read_bytes());r=json.loads((p/'result.json').read_bytes())
assert r['phase']=='complete' and r['DT_calls']==2
files={x.name:dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest())
       for x in [p/'result.json',p/'gdn-symmetric-v1.pt',p/'clean-v1.pt',p/'driver.log']}
peaks={}
for profile in ('gdn-symmetric-v1','clean-v1'):
 data=torch.load(p/(profile+'.pt'),map_location='cpu',weights_only=False)
 assert data['signed'].shape==(4,869) and torch.isfinite(data['signed']).all()
 peaks[profile]={k:data['details'].get(k) for k in ('peak_allocated','peak_reserved','complete_attribution_seconds_with_diagnostics')}
print(json.dumps(dict(unix=time.time(),files=files,owner_details=peaks,
 process_same_birth=bool(psutil.pid_exists(l['pid']) and psutil.Process(l['pid']).create_time()==l['birth']),
 physical_after=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
 formal_same_birth=psutil.Process(982372).create_time()==1791553809.84,
 formal_source_sha256=hashlib.sha256((p.parents[1]/'runs/textcraft-formal-stable-20261009-v1/source.json').read_bytes()).hexdigest(),
 model_DT_optimizer_calls=0)))
'''
body='REMOTE='+repr(REMOTE)+'\n'+body
shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
run=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=40)
(OUT/'completion-command.sh').write_text(shell,encoding='utf-8',newline='\n')
(OUT/'completion.stderr').write_bytes(run.stderr)
run.check_returncode();done=json.loads(run.stdout)
(OUT/'completion-and-resource.json').write_text(json.dumps(done,indent=2)+'\n',encoding='utf-8')
for name,v in done['files'].items():assert sha(OUT/name)==v['sha256']
r=json.loads((OUT/'result.json').read_bytes());inputs=json.loads((OUT/'paper-inputs.json').read_bytes())
profiles=json.loads((REPO/'research/temporary/qwen35_fleet_20260917/source/deltatrace/profiles/sources.json').read_bytes())
actual=json.loads((HERE/'paper-role-implementation-check-20261010-v1/runtime-source.json').read_bytes())
for key in ('official.py','qwen35_gdn_symmetric.py'):
 item=next(x for x in actual['owners'] if x['path'].endswith('/profiles/'+key))
 assert item['sha256']==profiles['files']['deltatrace/profiles/'+key]
sym=r['profiles']['gdn-symmetric-v1'];clean=r['profiles']['clean-v1']
assert all(x['root_effect']==y['root_effect'] for x,y in zip(sym['rows'],clean['rows']))
assert all(x['all_finite'] and x['fixed_position_max_abs']==0 for p in (sym,clean) for x in p['rows'])
for profile in (sym,clean):
 names=profile['rows'][0]['names']
 walton=[x for x in names if x['name']=='William Walton']
 shakespeare=[x for x in names if x['name']=='William Shakespeare']
 assert len(walton)==2 and all(x['current_sum']<0 for x in walton)
 assert len(shakespeare)==1 and shakespeare[0]['current_sum']>0
assert all(c['input_ids'][-1]==r['eos_token_id'] for c in inputs['cases'])
assert r['owners']['GDN']['sha256']=='7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
assert done['formal_same_birth'] and done['formal_source_sha256']==r['source_sha256']
def receipt(p):return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
result=dict(status='completed_literal_paper_input_DTpipeline_diagnostic_not_an_overall_accuracy_certificate',
 observed_unix=done['unix'],accepted_diagnostic='paper-role-implementation-check-20261010-v3',
 numerical_source_commit='26bef6c8b2e49db118f46e3c05e86944dcf8e293',
 numerical_release='fla-early-output-scale-20261009-v1',
 scope='Same base weights, four original paper inputs, full stored response including EOS, original eligible source positions, B4 interleaved pairs. This does not evaluate the trained formal policy, long context capacity, single-deletion correctness, population tail recall, or RISE/MAS.',
 owners=r['owners'],formal_source_sha256=r['source_sha256'],formal_same_birth=True,
 eos_token_id=r['eos_token_id'],tokenizer_sha256=r['tokenizer_sha256'],
 original_profiles_source=receipt(REPO/'research/temporary/qwen35_fleet_20260917/source/deltatrace/profiles/sources.json'),
 raw={p.name:receipt(p) for p in [OUT/'launch.json',OUT/'result.json',OUT/'gdn-symmetric-v1.pt',OUT/'clean-v1.pt',OUT/'formal-eos-owner-check.json',OUT/'paper-inputs.json',OUT/'completion-and-resource.json']},
 original_case_sources=inputs['sources'],profiles=r['profiles'],
 checks=dict(profile_factory_and_symmetric_module_match_original_official_sources=True,
  stable_GDN_matches_accepted_numerical_release=True,literal_input_hashes_checked=True,
  original_target_ids_and_EOS_preserved=True,all_eight_vectors_finite=True,
  fixed_input_target_and_padding_positions_zero=True,paired_profile_endpoint_effects_exact_equal=True,
  both_Walton_occurrences_negative_in_both_profiles=True,
  Shakespeare_span_positive_in_both_profiles=True),
 official_numerical_verification=receipt(REPO/'experiments/rl/results_fla_early_output_scale_20261009.json'),
 tolerance_scope='Existing accepted actual-dtype kernel verification is bound by source identity, not rerun. No new tolerance and no attribution conservation threshold is represented as an official FA/FLA criterion. Historical paper scalar proximity is descriptive, not a numerical pass threshold.',
 resources=r['resources'],owner_resource_details=done['owner_details'],
 rejected_diagnostics=[receipt(HERE/('paper-role-implementation-check-20261010-'+v)/'rejected-diagnostic.json') for v in ('v1','v2')],
 diagnostic_corrections='v1 omitted the explicit tokenizer EOS and used producer fallback0; results are rejected. v2 passed native EOS248046 but invented assertion248044 failed before DT; removed that assertion. v3 uses the native tokenizer value directly. Formal readout already uses native tokenizer EOS and no production code was changed.',
 operations=dict(accepted_DT_calls=2,rejected_v1_DT_calls=2,failed_v2_DT_calls=0,
  optimizer=0,rollout=0,checkpoint_restore=0,production_changes=0),
 interpretation='The original role-contrast behavior is retained: both Walton occurrences are negative and Shakespeare is positive. This is joint EOS allocation at name spans. Mixed signs within a name and successful reproduction of this example do not establish exact individual deletion probability ratios or fix extreme-credit approximation error.')
dest=REPO/'experiments/rl/results_paper_role_implementation_20261010.json'
assert not dest.exists();dest.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(receipt=receipt(dest),phase=r['phase'],EOS=r['eos_token_id'],DT_calls=r['DT_calls'],
 profiles={k:dict(seconds=v['seconds'],role_case=v['rows'][0]) for k,v in r['profiles'].items()},
 resources=r['resources'],owner_details=done['owner_details']),ensure_ascii=False))
