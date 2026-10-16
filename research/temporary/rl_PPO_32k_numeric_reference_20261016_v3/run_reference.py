"""Test-only native FA2 deterministic reference; production fast flags unchanged.

Reuse the completed v2 exact32768 data and native state, with the same original
critic/GAE/PPO and the same v2 offload candidate. No new training method.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERIFY='PPO-32k-numeric-reference-20261016-v3'
AUDIT=ROOT/'receipts'/VERIFY
AUDIT.mkdir(exist_ok=True)
owner=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
expected='f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
assert hashlib.sha256(owner.read_bytes()).hexdigest()==expected
code=owner.read_text()
old="AUDIT=ROOT/'receipts'/VERSION"
assert code.count(old)==1
code=code.replace(old,"AUDIT=ROOT/'receipts'/'"+VERIFY+"'")
old="    env['CUDA_VISIBLE_DEVICES']='2,3';env.pop('MACA_VISIBLE_DEVICES',None)"
assert code.count(old)==1
code=code.replace(old,"    env['FLASH_ATTENTION_DETERMINISTIC']='1'\n"+old)
start=code.index('original_data=torch.load(')
end=code.index('status=dict(',start)
replacement="""data=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/same-session-native-data-32768.pt',weights_only=False,map_location='cpu')
assert data.batch['input_ids'].shape==(8,32768)
"""
code=code[:start]+replacement+code[end:]
old="    status['status']='original_native_values_32768';write(AUDIT/'same-session-status.json',status)"
start=code.index(old)
end=code.index("    saved=group.observe_state('original_saved')",start)
replacement="""    status['status']='restore_existing_original_native_state';write(AUDIT/'same-session-status.json',status)
    checkpoint=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/numerical-state-original-VERL'
    group.load_checkpoint(str(checkpoint),del_local_after_load=False)
"""
code=code[:start]+replacement+code[end:]
code=code.replace("original_input_source=str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt'),",
    "original_input_source=str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/same-session-native-data-32768.pt'), test_only_FA_deterministic=True, production_flags_unchanged=True,")
code=code.replace('/tmp/ray-PPO32k-native-checkpoint-v2','/tmp/ray-PPO32k-numeric-reference-v3')
receipt=dict(source_owner=str(owner),source_owner_SHA256=expected,
    candidate_owner='PPO-32k-native-efficiency-20261016-v2',test_only_FA_deterministic=True,
    changes=['Independent diagnostic audit/Ray path','Original HF FA2 deterministic flag for numerical reference only',
        'Reuse completed exact32768 native data and original VERL state; no redundant forward or new checkpoint'],
    original_critic_GAE_PPO_optimizer_precision_preserved=True,formal_mutations=0)
(AUDIT/'source-composition.json').write_text(json.dumps(receipt,indent=2)+'\n')
exec(compile(code,str(owner),'exec'),globals())
