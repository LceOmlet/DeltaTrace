"""Describe original-author TextCraft arrays and actual changed groups.

No RISE/MAS computation, sorting, training credit or numerical tolerance is
implemented here. All metric values/arrays are the unchanged author's returns.
"""
import csv
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
REPO=HERE.parents[4]
def binding(path):
    data=path.read_bytes()
    return dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

folder=HERE/'actual-results'
for item in json.loads((folder/'transport.json').read_bytes()):
    assert binding(Path(item['local_path']))['sha256']==item['sha256']
ranks=[json.loads((folder/f'results/rank{i}.json').read_bytes()) for i in (0,1)]
assert all(r['case_name']=='textcraft' and r['phase']=='complete' for r in ranks)
native_path=AUDIT/'direct-target-prefix-runtime-20261007/v1/textcraft-credit-cpu-complete-1791373053/rank1-readout-native-batch-21.pt'
assert binding(native_path)['sha256']==ranks[0]['native_sha256']
native=torch.load(native_path,map_location='cpu',weights_only=False)
candidate=ranks[0]['geometry']['candidate']
signed=native['native_signed'][candidate['row']].double()
views={}
csv_rows=[]
for name,view in ranks[0]['views'].items():
    other=ranks[1]['views'][name]
    assert view['author_arrays']==other['author_arrays'] and view['author_return']==other['author_return']
    assert [p['logp'] for p in view['score_points']]==[p['logp'] for p in other['score_points']]
    previous=set()
    groups=[]
    for index,point in enumerate(view['score_points']):
        changed=set(point['changed_input_positions'])
        assert previous<=changed
        added=sorted(changed-previous)
        values=signed[added]
        group=dict(step=index,changed_positions=len(changed),newly_changed_positions=len(added),
            DT_signed_sum_of_newly_changed=float(values.sum()),negative_DT_count=int((values<0).sum()),
            positive_DT_count=int((values>0).sum()),zero_DT_count=int((values==0).sum()),
            native_joint_logp=point['logp'],
            native_kept_minus_deleted_for_group=None if index==0 else view['score_points'][index-1]['logp']-point['logp'],
            Format_candidate_changed=candidate['packed_slot'] in changed,seconds=point['seconds'])
        groups.append(group)
        csv_rows.append(dict(view=name,**group))
        previous=changed
    views[name]=dict(author_return=dict(zip(view['author_fields'],view['author_return'])),
        author_arrays=view['author_arrays'],groups=groups,
        candidate_first_changed_step=next(g['step'] for g in groups if g['Format_candidate_changed']),
        native_forward_seconds=sum(p['seconds'] for p in view['score_points']),
        scope='Observed cumulative changed IDs. Group effects depend on the preceding deletions and are not individual-token deletion truth.')
controls=dict(ranks_same_author_arrays_and_scores=True,
    maximum_twin_score_difference=max(abs(x) for r in ranks for v in r['views'].values() for p in v['score_points'] for x in p['twin_row_score_differences']),
    other_three_row_scores_constant=all(len({tuple(p['other_three_joint_logp']) for v in r['views'].values() for p in v['score_points']})==1 for r in ranks),
    lora_B_zero=all(not s['nonzero'] for r in ranks for s in r['lora_B_local_shards']))
launch=json.loads((HERE/'launch.json').read_bytes())
terminal_path=sorted(HERE.glob('observation-*.json'))[-1]
terminal=json.loads(terminal_path.read_bytes())
assert terminal['completed'] and not terminal['driver']['same_birth']
assert terminal['textcraft_same_driver'] and not any(terminal['textcraft_release_present'])
sample_path=AUDIT/'direct-target-credit-sample-20261007/v1/sample-analysis.json'
sample=json.loads(sample_path.read_bytes())
format_point=next(p for p in sample['tasks']['textcraft']['points'] if p['token']==' Format')
result=dict(status='Original-author cumulative deletion on actual TextCraft Format trajectory complete; credit repair not claimed',
    scope='One actual selected TextCraft trajectory and its original B4 controls. Same actual joint action target Y, native scorer, exact saved IDs, author default k=20 and original author metric. Not overall attribution quality or a training acceptance threshold.',
    launch=launch,candidate=candidate,source_count=len(ranks[0]['source_positions']),
    author=ranks[0]['author'],views=views,controls=controls,
    original_imported_owners=[r['owners'] for r in ranks],
    source_sha256=ranks[0]['source_sha256'],native_sha256=ranks[0]['native_sha256'],
    actual_single_token_supplement=format_point,
    target_and_parser_context=binding(AUDIT/'direct-target-existing-pv-rule-20261008/v1/selected-target-context.json'),
    measurements=dict(native_forwards_per_rank=[r['native_forward_calls'] for r in ranks],DT_calls=0,
        rollout_calls=0,backward=0,optimizer=0,checkpoint_restore=0,
        driver_to_last_worker_complete_seconds=max(r['unix'] for r in ranks)-launch['launched_unix'],
        worker_peak_recorded_PSS_bytes=[max(p['pss_bytes'] for v in r['views'].values() for p in v['score_points']) for r in ranks],
        physical_peak_continuously_sampled=False),
    evaluation_only='Positive-only MAS input and original author normalization/running minimum/corrected_scores are evaluation operations. They do not alter raw training d/Q/V/A, whitening or PPO.',
    interpretation='The paired whole-trajectory author curves supplement the already measured individual Format deletion. Neither a favorable ranking nor a normalized curve proves that its exponential probability-ratio magnitude is accurate. The signed cumulative groups are retained without smoothing or forced monotonicity in the raw native scores.',
    official_FA_FLA_tolerance_claim=False,credit_repaired=False,production_profile_changed=False,
    formal_restart=False,TextCraft_update_released=False,AppWorld_state='terminal_not_restarted',
    unchanged=dict(lora_rank=8,lora_alpha=16,actor_microbatch_per_gpu=4,DT_minibatch_per_gpu=4,QVA=True,PPO=True,target=True,reward=True,whitening=True),
    sources=[binding(folder/'transport.json'),binding(native_path),binding(sample_path),binding(terminal_path),
        binding(HERE/'launch.json'),binding(REPO/'experiments/rl/upstream.lock')])
(HERE/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
receipt_path=REPO/'experiments/rl/results_textcraft_actual_author_curves_20261008.json'
receipt_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (HERE/'curve.csv').open('w',newline='',encoding='utf8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(receipt=binding(receipt_path),source_count=result['source_count'],controls=controls,
    measurements=result['measurements'],views={name:dict(author_return=view['author_return'],
        candidate_first_changed_step=view['candidate_first_changed_step'],last_groups=view['groups'][-3:]) for name,view in views.items()}),ensure_ascii=False))
