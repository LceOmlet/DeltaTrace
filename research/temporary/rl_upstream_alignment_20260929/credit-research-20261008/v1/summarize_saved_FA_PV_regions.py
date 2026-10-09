"""Group the frozen original-FA key-partition residuals, without correction."""
import hashlib
import json
from pathlib import Path
import time

from analyze_attention_gate import state_frequency
from summarize_author_collection import quantiles

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def summary(points):
    result=dict(points=len(points),states=len({p['initial_state_sha256'] for p in points}),
                trajectories=len({p['traj_uid'] for p in points}))
    if not points:return dict(**result,missing='Empty, not zero')
    names=('before','at','after')
    result['conditional_values']={n:dict(signed=quantiles([p['PV_key_regions'][n] for p in points]),
        absolute=quantiles([abs(p['PV_key_regions'][n]) for p in points])) for n in (*names,'all')}
    result['largest_absolute_region_position_counts']={n:sum(n==max(names,key=lambda k:abs(p['PV_key_regions'][k])) for p in points) for n in names}
    result['state_equal_largest_absolute_region_frequency']={n:state_frequency(points,
        lambda p:n==max(names,key=lambda k:abs(p['PV_key_regions'][k]))) for n in names}
    result['arithmetic_controls']={n:quantiles([abs(p[n]) for p in points]) for n in
        ('partition_closure_residual','original_PV_replay_difference')}
    return result


def main():
    path=HERE/'FA-PV-key-regions-v1/support.json';data=json.loads(path.read_bytes())
    assert data['phase']=='complete' and len(data['points'])==165
    stats_path=REPO/'experiments/rl/results_grouped_negative_support_20261009.json'
    stats=json.loads(stats_path.read_bytes())['tasks']['textcraft']['cohorts']
    flags={(p['traj_uid'],p['packed_slot']):p for c in stats.values() for p in c['points_with_identity']}
    prior=HERE/'attention-pv-textcraft-v2/analysis.json'
    repeat={(p['traj_uid'],p['packed_slot']):all(p[k]==0 for k in
            ('DT_difference','native_difference','original_mixer_difference'))
            for p in json.loads(prior.read_bytes())['unchanged_output_comparisons']}
    points=data['points'];cohorts={}
    for cohort,c in stats.items():
        identities={(p['traj_uid'],p['packed_slot']) for p in c['points_with_identity']}
        rows=[p for p in points if (p['traj_uid'],p['packed_slot']) in identities]
        cells={}
        for p in rows:
            f=flags[p['traj_uid'],p['packed_slot']]
            assert f['token_id']==p['token_id']
            cells.setdefault(f['DT_ratio_bin']+':'+f['FP32_head_ratio_bin'],[]).append(p)
        bad=lambda p:flags[p['traj_uid'],p['packed_slot']]['robust_spurious_negative_tail']
        cohorts[cohort]=dict(all=summary(rows),crossed_ratio_cells={k:summary(v) for k,v in cells.items()},
            by_exposure={str(e):summary([p for p in rows if p['previously_examined']==e]) for e in (False,True)},
            robust_spurious_tail=summary([p for p in rows if bad(p)]),
            by_historical_repeatability={str(e):dict(all=summary([p for p in rows if repeat[p['traj_uid'],p['packed_slot']]==e]),
                robust_spurious_tail=summary([p for p in rows if repeat[p['traj_uid'],p['packed_slot']]==e and bad(p)])) for e in (False,True)})
    result=dict(scope=__doc__,observed_unix=time.time(),sources=[ref(path),ref(stats_path),ref(prior),ref(Path(__file__))],
        task='textcraft',all=summary(points),cohorts=cohorts,
        runtime={k:data[k] for k in ('pid','birth','elapsed_seconds','operations','sampled_PSS_peak_bytes','GPU_peak_allocated_bytes','GPU_peak_reserved_bytes','original_FA_owner')},
        algebra=dict(exact_arithmetic='u*(P_F-P_D)*(V_R-V_D)=sum over before, source, after key regions',
            computation='Original BF16 Q/K and original FA causal mask; only value operands are zero outside each disjoint key region',
            closure='Actual separate BF16 output rounding is measured and retained; no tolerance is invented and no credit is adjusted'),
        conclusions=[
            'On the 18 robust spurious tails, the source-key region has the largest absolute PV residual on 15 positions. This describes the measured ledger, not a share of the global credit error.',
            'The source-key conditional background and the previously measured multirow response both matter to a proposed internal rule. This does not accept a new repair or rescue a previously failed candidate.',
            'All source/native/census/exposure and historical repeatability strata are retained. The result is TextCraft evidence, not an AppWorld measurement.',
            'No new model, optimizer, DT call or training trajectory was used. Official FA/FLA tolerances and Q/V/A remain unchanged.'
        ],production_modified=False,credit_repaired=False,official_tolerance_test=False)
    output=REPO/'experiments/rl/results_saved_FA_PV_regions_20261009.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output),spurious_tail=cohorts['predicted_tail_census']['robust_spurious_tail'],runtime=result['runtime'])))


if __name__=='__main__':main()
