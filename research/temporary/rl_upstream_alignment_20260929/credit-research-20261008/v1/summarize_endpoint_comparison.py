"""Assess the frozen comparison; no selection, tuning, correction or deployment."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    analysis=HERE/'endpoint-head-owner-v1/textcraft-analysis.json'
    stable=ROOT/'experiments/rl/results_stable_negative_credit_20261008.json'
    data=json.loads(analysis.read_bytes())
    assert data['complete']
    original=json.loads(stable.read_bytes())['tasks']['textcraft']
    groups={}
    for cohort,values in data['cohorts'].items():
        baseline={(p['traj_uid'],p['packed_slot']):p for p in original['cohorts'][cohort]['points_with_identity']}
        for q in values['points_with_identity']:
            old=baseline[(q['traj_uid'],q['packed_slot'])]
            native_bin=next(c['native_ratio_bin'] for c in values['cells']
                if c['previously_examined']==q['previously_examined']
                and c['native_ratio_bin']==ratio_bin(q['fresh_native_single_d'])
                and c['original_ratio_bin']==ratio_bin(q['fresh_original_d']))
            key=(cohort,q['previously_examined'],ratio_bin(q['fresh_original_d']),native_bin)
            groups.setdefault(key,[]).append(dict(traj_uid=q['traj_uid'],packed_slot=q['packed_slot'],
                state=q['state'],original_spurious=old['spurious_DT_tail_all_references'],
                original_missed=old['missed_native_tail_all_references'],
                original_d=q['fresh_original_d'],candidate_d=q['fresh_endpoint_head_d'],
                native_d_interval=old['native_d_interval'],
                candidate_tail=q['fresh_endpoint_head_d'] < -math.log(2),
                candidate_sign_wrong_all_references=(
                    q['fresh_endpoint_head_d']<0 and min(old['native_d_values'])>=0) or (
                    q['fresh_endpoint_head_d']>=0 and max(old['native_d_values'])<0),
                candidate_impossible_all_references=(
                    min(old['factual_logp_observed_interval'])-q['fresh_endpoint_head_d']>0)))
    output=[]
    for key,points in groups.items():
        states={}
        for q in points:states.setdefault(q['state'],[]).append(q)
        counts=dict(points=len(points),states=len(states),
            original_spurious=sum(p['original_spurious'] for p in points),
            original_spurious_still_tail=sum(p['original_spurious'] and p['candidate_tail'] for p in points),
            original_spurious_sign_still_wrong=sum(p['original_spurious'] and p['candidate_sign_wrong_all_references'] for p in points),
            original_missed=sum(p['original_missed'] for p in points),
            original_missed_still_missed=sum(p['original_missed'] and not p['candidate_tail'] for p in points),
            candidate_impossible_all_references=sum(p['candidate_impossible_all_references'] for p in points))
        output.append(dict(cohort=key[0],previously_examined=key[1],original_ratio_bin=key[2],
            native_ratio_bin=key[3],counts=counts,points_with_identity=points,
            equal_state_tail_fraction=statistics.mean(statistics.mean(int(q['candidate_tail']) for q in rows)
                                                       for rows in states.values())))
    value=dict(task='textcraft',complete=True,evidence=[ref(analysis),ref(stable)],
        primary_metrics=data['primary_metrics'],robust_cross_cells=output,
        interpretation='Prior robust flags use all saved native precision/repeat references. Tail disappearance is distinct from sign correction. Cells, tasks and exposure remain separate; no pooled raw heavy-tail moments.',
        accepted_candidate=False,production_modified=False,formal_restart=False,recorder=ref(Path(__file__)))
    value['decision']='rejected_no_observed_quality_benefit'
    value['decision_evidence']='Both primary author means increased slightly (lower is better), all18 prior robust spurious tails remain, and both robust uniform missed tails remain. No statistical-significance or population-optimality claim.'
    value['next_task_GPU_calls']=0
    path=HERE/'endpoint-head-owner-v1/textcraft-summary.json'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(path),cells=[{k:v for k,v in cell.items() if k!='points_with_identity'} for cell in output]),ensure_ascii=False))


def ratio_bin(d):
    if d>=0:return '[0,1]'
    for upper,label in [(2,'(1,2]'),(10,'(2,10]'),(100,'(10,100]')]:
        if -d<=math.log(upper):return label
    return '(100,inf)'


if __name__=='__main__':main()
