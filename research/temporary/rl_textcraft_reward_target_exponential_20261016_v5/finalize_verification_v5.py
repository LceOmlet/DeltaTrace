"""Record completed contracts/capacity before the protected native launcher."""
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
PREPARED=AUDIT/'v5-prepared'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cpu=ET.parse(AUDIT/'CPU-behavior-v5.xml').getroot()
cases=cpu.findall('.//testcase')
assert len(cases)==22
assert not cpu.findall('.//failure') and not cpu.findall('.//error') and not cpu.findall('.//skipped')
native=read(AUDIT/'native-behavior-status.json')
assert native['status']=='completed'
assert native['optimizer_updates']==native['rollout_generation_calls']==0
reports=[]
for stage in ('native-small','native-multi-target','capacity-multi-32768'):
    for rank in (0,1):
        path=AUDIT/f'{stage}-rank{rank}.json'
        if stage=='native-small' and not path.exists():
            reference=read(AUDIT/'native-small-reused-receipts.json')[rank]
            path=Path(reference['path']);assert sha(path)==reference['SHA256']
        result=read(path)
        if stage=='native-small' and result['status']=='failed':
            assert result['error']=="KeyError('verl.utils.experimental.torch_functional')"
            assert len(result['finite_calls'])==1 and len(result['credit_calls'])==4
        else:assert result['status']=='completed'
        assert result['before']==result['after']
        if stage=='native-small' and 'failed-binding-metadata-r2' in str(path):
            (AUDIT/f'native-small-rank{rank}-reused.json').write_bytes(path.read_bytes())
        assert result['optimizer_updates']==result['generated_rollouts']==result['quality_evaluations']==0
        assert result['report']['credit_transform']=='per_event_exponential_then_credit_sum'
        assert result['report']['source_credit_normalization'] is None
        assert all(item['original_unscaled_seed'] for item in result['finite_calls'])
        assert all(item['exact_expression_equality'] for item in result['credit_calls'])
        if stage=='capacity-multi-32768':assert all(item['shape']==[8,32768] for item in result['finite_calls'])
        reports.append(result)
for stage in ('native-multi-target','capacity-multi-32768'):
    driver=read(AUDIT/(stage+'-driver.json'))
    assert driver['status']=='completed' and driver['native_per_event_QVA_sum_exact']
launch=read(PREPARED/'launch.json')
launch['options']['trainer.experiment_name']=launch['version']
launch['argv']=['trainer.experiment_name='+json.dumps(launch['version']) if arg.startswith('trainer.experiment_name=') else arg for arg in launch['argv']]
from omegaconf import OmegaConf
configuration=OmegaConf.load(PREPARED/'effective-config.yaml')
configuration.trainer.experiment_name=launch['version']
OmegaConf.save(configuration,PREPARED/'effective-config.yaml')
for result in reports:
    for record in result['bindings'].values():
        assert sha(record['path'])==record['SHA256']
        launch['source_bindings'][record['path']]=record['SHA256']
for name in ('passive_reward_readout_v5.py','install_passive_records_v5.py'):
    path=AUDIT/'source'/name;launch['source_bindings'][str(path)]=sha(path)
for path,digest in launch['source_bindings'].items():assert sha(path)==digest,path
checked=dict(version=launch['version'],unix=time.time(),checks={
    'native_owner_zero_reward_settlement_target_selection':True,
    'signed_multi_event_exp_before_sum_no_u_normalization':True,
    'native_original_transport_dedup_padding_scatter':True,
    'native_two_rank_original_unscaled_DT_to_exponential_QVA':True,
    'native_B4_per_rank_exact_32768_capacity_finite':True,
    'actor_optimizer_scheduler_unchanged_by_verification':True,
    'official_GRPO_PPO_owners_and_verified_efficiency_SHAs':True,
    'no_withdrawn_affine_seed_or_head_selection_candidate':True},
    unique_passing_behavior_tests=len(cases),CPU_result_SHA256=sha(AUDIT/'CPU-behavior-v5.xml'),
    capacity_seconds_by_rank=[r['seconds'] for r in reports if r['stage']=='capacity-multi-32768'],
    verification_limits='Native interface/capacity and exact composition checks; not training quality, long-term gradient repair, or a new DT prediction-accuracy experiment.',
    numerical_acceptance='Unchanged accepted DT and update owners by SHA. Exponential composition exact tensor equality; no invented tolerance.',
    coefficient_artifact=native['coefficient_artifact'],
    reused_native_small_computation=native.get('reused_completed_native_small_computation'),
    optimizer_updates=0,generated_rollouts=0,quality_evaluations=0)
(PREPARED/'interface-verification.json').write_text(json.dumps(checked,indent=2)+'\n')
(AUDIT/'interface-verification.json').write_bytes((PREPARED/'interface-verification.json').read_bytes())
(PREPARED/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
(AUDIT/'prepared-launch.json').write_bytes((PREPARED/'launch.json').read_bytes())
source=read(PREPARED/'source.json');source['actual_inherited_sources']=launch['source_bindings']
source['completed_verification']=str(PREPARED/'interface-verification.json')
(PREPARED/'source.json').write_text(json.dumps(source,indent=2)+'\n')
print(json.dumps(checked,indent=2))
