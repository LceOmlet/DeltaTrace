"""Group original saved operand support; energy is not credit or causal share."""
import hashlib
import json
from pathlib import Path
import time

from summarize_author_collection import quantiles
from analyze_attention_gate import state_frequency

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def summarize(points):
    base=dict(points=len(points),states=len({p['initial_state_sha256'] for p in points}),
              trajectories=len({p['traj_uid'] for p in points}))
    if not points:return dict(**base,missing='No observations; not zero')
    return dict(**base,operands={n:dict(
        literal_nonzero_before_source_positions=sum(p['operand_support'][n]['changed_rows']['before']>0 for p in points),
        literal_nonzero_after_source_positions=sum(p['operand_support'][n]['changed_rows']['after']>0 for p in points),
        state_equal_nonzero_after_source_frequency=state_frequency(points,lambda p:p['operand_support'][n]['changed_rows']['after']>0),
        conditional_energy_fraction_quantiles={k:quantiles([p['operand_support'][n]['energy_fraction'][k]
            for p in points if p['operand_support'][n]['energy_fraction'][k] is not None])
            for k in ('before','at','after')}) for n in ('query','key','value')})


def main():
    raw_path=HERE/'FA-source-support-v2/support.json'
    raw=json.loads(raw_path.read_bytes());assert raw['phase']=='complete'
    stats_path=REPO/'experiments/rl/results_grouped_negative_support_20261009.json'
    stats=json.loads(stats_path.read_bytes())['tasks']['textcraft']['cohorts']
    flags={(p['traj_uid'],p['packed_slot']):p for c in stats.values() for p in c['points_with_identity']}
    points=raw['points'];assert len(points)==len(flags)==165
    prior=HERE/'attention-pv-textcraft-v2/analysis.json'
    repeat={(p['traj_uid'],p['packed_slot']):all(p[k]==0 for k in
            ('DT_difference','native_difference','original_mixer_difference'))
            for p in json.loads(prior.read_bytes())['unchanged_output_comparisons']}
    for p in points:
        f=flags[p['traj_uid'],p['packed_slot']]
        assert f['token_id']==p['token_id']
        p['flags']=f
        p['historical_pair_unchanged']=repeat[p['traj_uid'],p['packed_slot']]
    cohorts={}
    for cohort in ('uniform','predicted_tail_census'):
        rows=[p for p in points if any(c['cohort']==cohort for c in p['previous_comparisons'])]
        cells={}
        for p in rows:
            cell=p['flags']['DT_ratio_bin']+':'+p['flags']['FP32_head_ratio_bin']
            cells.setdefault(cell,[]).append(p)
        cohorts[cohort]=dict(all=summarize(rows),
            by_exposure={str(e):summarize([p for p in rows if p['previously_examined']==e]) for e in (False,True)},
            crossed_ratio_cells={cell:summarize(v) for cell,v in cells.items()},
            robust_spurious_tail=summarize([p for p in rows if p['flags']['robust_spurious_negative_tail']]),
            by_historical_repeatability={str(e):dict(all=summarize([p for p in rows if p['historical_pair_unchanged']==e]),
                robust_spurious_tail=summarize([p for p in rows if p['historical_pair_unchanged']==e
                    and p['flags']['robust_spurious_negative_tail']])) for e in (False,True)})
    result=dict(scope=__doc__,observed_unix=time.time(),sources=[ref(raw_path),ref(stats_path),ref(prior),ref(Path(__file__))],
        raw_metadata=dict(pid=raw['pid'],birth=raw['birth'],elapsed_seconds=raw['elapsed_seconds'],
                          sampled_PSS_peak_bytes=raw['sampled_PSS_peak_bytes'],files=len(raw['files'])),
        all=summarize(points),cohorts=cohorts,
        findings=[
            'Every frozen single-deletion source changes later saved Q/K/V rows at the last FA layer; a source-local hidden-row replacement is not the same counterfactual.',
            'For the 18 robust spurious predicted tails the preceding Q/K/V rows are literally equal, while all have changes after the source. This is stronger locality evidence for this subset, not a pure operator causal attribution.',
            '46 of 165 other comparisons include nonzero preceding operand differences; no new zero tolerance or explanation is inferred from them. They remain explicit in the output.',
            'Operand energy is not credit magnitude. The support result excludes a one-row-only explanation, not every efficient structured propagation method.'
        ],operations=raw['operations'],production_modified=False,credit_repaired=False,official_tolerance_test=False)
    output=REPO/'experiments/rl/results_saved_FA_source_support_20261009.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output),all=result['all'],
        tail=cohorts['predicted_tail_census']['robust_spurious_tail'])))


if __name__=='__main__':main()
