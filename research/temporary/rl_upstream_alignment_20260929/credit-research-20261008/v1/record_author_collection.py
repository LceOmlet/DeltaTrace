"""Record completed frozen baseline measurements without declaring a credit repair."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def binding(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


if __name__=='__main__':
    summary_path=HERE/'author-collection-summary.json';summary=json.loads(summary_path.read_bytes())
    obs_path=sorted((HERE/'author-collection-observations').glob('[0-9]*.json'))[-1]
    obs=json.loads(obs_path.read_bytes());launch=json.loads((HERE/'author-collection-launch.json').read_bytes())
    assert 'results/completed.json' in obs['files'],'Do not record completion before the original actor group returns'
    ranks={str(r):obs['files'][f'results/rank{r}.json']['value'] for r in (0,1)}
    for value in ranks.values():
        assert value['phase']=='complete' and value['native_forward_calls']==194
        assert all(v==0 for v in value['operations'].values())
    manifest=json.loads((HERE/'manifest.json').read_bytes());strata=json.loads((HERE/'credit-strata.json').read_bytes())
    groups={u:g for t in manifest['tasks'].values() for g in t['groups'] for u in g['trajectory_uids']}
    findings={}
    for task in manifest['tasks']:
        assert summary['uniform_sample'][task]['measured_points']==128
        assert summary['observed_DT_tail_census'][task]['measured_points']==37
        expected={(p['traj_uid'],p['packed_slot']) for p in strata['tasks'][task]['complete_observed_ratio_gt_2_tail']}
        actual={}
        for rank in ranks.values():
            for point in rank['tail_results']:
                if point['traj_uid'] in strata['tasks'][task]['rows']:
                    key=point['traj_uid'],point['packed_slot'];assert key not in actual
                    actual[key]=point
        assert set(actual)==expected
        crossings=[p for p in actual.values() if p['native_single_d']>=0]
        uniform=summary['uniform_sample'][task]['results']['strata']
        bulk=('ratio_le_1','ratio_1_to_2')
        findings[task]={'observed_DT_tail_points':len(actual),
            'native_nonnegative_credit_among_DT_tail':len(crossings),
            'crossing_initial_state_groups':len({p['initial_state_sha256'] for p in crossings}),
            'crossing_previously_unexamined_state_groups':len({p['initial_state_sha256'] for p in crossings if not groups[p['traj_uid']]['previously_examined']}),
            'uniform_native_tail_missed_by_DT':sum(cell['points'] for a,row in uniform.items() if a in bulk for b,cell in row['native_strata'].items() if b not in bulk),
            'uniform_sample_points':128,
            'scope':'Counts in the declared frozen sample/census; not population rates, operator-tolerance assertions, GDN localization or gradient influence.'}
    history=[json.loads(p.read_bytes()) for p in (HERE/'author-collection-observations').glob('[0-9]*.json')]
    record={'status':'Completed original-owner baseline collection, not a method repair or training release',
        'observed_unix':obs['unix'],'launch':launch,'summary':summary,'observed_findings':findings,
        'original_collected_artifacts':json.loads((HERE/'author-original-artifacts.json').read_bytes()),
        'provenance':{p.name:binding(p) for p in (HERE/'manifest.json',HERE/'corpus.json',HERE/'credit-strata.json',
            HERE/'author-collection-launch.json',summary_path,obs_path,Path(__file__))},
        'cpu_boundary_replay':obs['files']['batching-cpu-replay.json'],
        'actual_owners':{r:{k:value[k] for k in ('pid','birth','owners','target_readers','metric_owner',
            'script_sha256','source_sha256','plan_sha256')} for r,value in ranks.items()},
        'work':{'native_forward_calls_per_rank':194,'native_forward_calls_total':388,
            'first_stage_trajectories_per_task':32,'initial_state_groups_per_task':16,
            'uniform_single_deletions_per_task':128,'observed_DT_tail_census_per_task':37,
            'DT_calls':0,'backward_calls':0,'optimizer_steps':0,'rollout_calls':0},
        'runtime':{'native_diagnostic_wall_seconds_by_rank':{r:v['seconds'] for r,v in ranks.items()},
            'host_min_observed_available_bytes':min(x['host_memory']['available'] for x in history),
            'rank_max_observed_pss_bytes':{str(r):max(x['files'].get(f'results/rank{r}.json',{}).get('value',{}).get('pss_bytes',0) for x in history) for r in (0,1)},
            'physical_last_snapshot':obs['physical'],'driver_alive':obs.get('driver_alive'),
            'scope':'Sampled phase observations, not a continuous physical VRAM peak measurement.'},
        'textcraft_hold':obs['textcraft_hold'],
        'limits':['Completed-capture development population only; two missing AppWorld development captures are retained as missing.',
            'Uniform first-stage samples and the observed DT-tail census have different inclusion rules and are never pooled.',
            'Measured native single-EOS model scores are not world-simulator results or kernel-tolerance tests.',
            'Policy-gradient influence and the historical degradation cause remain unmeasured by this forward-only study.',
            'No GDN candidate, numerical correction, coefficient replacement, held-out quality evaluation or training restart.']}
    path=ROOT/'experiments/rl/results_credit_author_collection_20261008.json'
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    runtime_path=ROOT/'experiments/rl/current_runtime.json';runtime=json.loads(runtime_path.read_bytes())
    item=runtime['latest_credit_stratified_collection']
    item['status']=record['status'];item['final_receipt']=binding(path)
    item['observation']=binding(obs_path)|{'unix':obs['unix']}
    runtime_path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':binding(path),'work':record['work'],'runtime_seconds':record['runtime']['native_diagnostic_wall_seconds_by_rank']},ensure_ascii=False))
