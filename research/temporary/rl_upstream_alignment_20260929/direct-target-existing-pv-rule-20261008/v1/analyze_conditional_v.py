"""Bind the completed original-native conditional V measurement, no acceptance fit."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]

def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

folder=HERE/'native-conditional-v-results'
transport=json.loads((folder/'transport.json').read_bytes())
for row in transport:
    assert binding(Path(row['local_path']))['sha256']==row['sha256']
measured=json.loads((folder/'results/result.json').read_bytes())
assert measured['phase']=='complete'
assert all(x['all_other_inputs_factual_pair_equal'] and x['factual_V_exact_equal']
           and x['initial_state_pair_equal'] for x in measured['groups'])
source=json.loads((HERE/'joint-finite-sources.json').read_bytes())
observation=json.loads(sorted(HERE.glob('native-conditional-v-observation-*.json'))[-1].read_bytes())
assert not observation['alive_same_birth'] and observation['textcraft_same_birth']
assert not any(observation['textcraft_releases'])
result=dict(
    status='Native conditional V measured; not a full DT repair or official finite-tolerance pass',
    launch=json.loads((folder/'launch.json').read_bytes()),
    source_audit=binding(HERE/'joint-finite-sources.json'),
    imported_files=[dict(module=r['module'],path=r['path'],sha256=r['sha256']) for r in source['files']],
    scope='Saved actual GDN30 single-deletion V pair, with Q/K/raw_g/beta fixed to the factual endpoint and the original shared factual incoming state. Calls the unchanged public FLA chunk_gated_delta_rule. Compare existing forward/symmetric V coefficients contracted with the same actual V difference.',
    original_native_owner=measured['owner'],
    verified_native_source_sha256=measured['verified_native_source_sha256'],
    groups=measured['groups'],summary=measured['summary'],
    limitations=[
        'V is conditionally linear with other GDN inputs fixed. This measurement isolates that branch; it is not a standalone token reward, complete token advantage, or joint causal effect of the deletion.',
        'The original forward V field closely matches this conditional native effect. That does not validate a full-vector composition with other symmetric fields or prove that original author deletion curves improve.',
        'Actual FP16 Q/K/V/beta and FP32 raw_g are recorded. The finite conditional residual is reported directly; no new official tolerance, pass threshold, scaling correction or credit clipping is introduced.',
        'Peak_live_allocated_bytes is the torch live allocator. Before/end physical GPU readings are retained but no continuous physical peak was sampled for this operator-only job.',
        'The original saved four head-group operand files remain on the remote host with hashes; no repeated model/full-DT/backward/rollout/optimizer/restore or formal restart occurred.',
    ],
    transport=binding(folder/'transport.json'),
    raw_result=binding(folder/'results/result.json'),
    terminal_observation=binding(sorted(HERE.glob('native-conditional-v-observation-*.json'))[-1]),
    formal_restart=False,production_rule_changed=False,TextCraft_update_released=False,
    credit_repaired=False,
)
target=REPO/'experiments/rl/results_native_conditional_V_20261008.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=binding(target),summary=result['summary']),ensure_ascii=False))
