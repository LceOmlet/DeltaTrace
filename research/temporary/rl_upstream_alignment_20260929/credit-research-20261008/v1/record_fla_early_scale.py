"""Bind original numerical checks and both real B4 integration replays."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
LOCAL=HERE/'fla-early-scale-review-v1'


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    official=json.loads((LOCAL/'official-result.json').read_bytes())
    nonzero=json.loads((LOCAL/'nonzero-result.json').read_bytes())
    seeds=json.loads((LOCAL/'seed-census.json').read_bytes())
    prepared=json.loads((LOCAL/'prepared.json').read_bytes())
    assert official['status']==nonzero['status']=='passed'
    assert sum(len(c['checks']) for c in official['cases'])==22
    assert all(c['status']=='passed' for row in official['cases'] for c in row['checks'])
    assert len(nonzero['checks'])==5 and all(c['status']=='passed' for c in nonzero['checks'])
    assert all(r['new_FP16_nonfinite']==0 for r in seeds['ranks'])
    replays={}
    for task in ('textcraft','appworld'):
        directory=LOCAL/('replay-'+task)/'preserved'
        assert json.loads((directory/'completed.json').read_bytes())
        ranks=[json.loads((directory/f'rank{rank}.json').read_bytes()) for rank in (0,1)]
        for r in ranks:
            assert r['phase']=='complete' and r['operations']['DT']==1
            assert not any(r['operations'][k] for k in ('optimizer','backward','rollout','checkpoint_restore'))
            assert r['isolated_gdn_candidate']['candidate_sha256']==prepared['owners'][task]['candidate_sha256']
            assert all(math.isfinite(p['single_background_DT_d']) for b in r['batches'] for p in b['points'])
        replays[task]=dict(launch=ref(LOCAL/('replay-'+task)/'launch.json'),
            ranks=[dict(rank=r['rank'],phase=r['phase'],operations=r['operations'],elapsed_seconds=r['elapsed_seconds'],
                rounds=[b['rounds'] for b in r['batches']],actual_candidate=r['isolated_gdn_candidate'],
                ending_allocated=r['allocated'],ending_reserved=r['reserved'],PSS=r['process_pss_bytes'],
                source_sha256=r['source_sha256'],finite_runner=r['finite_runner']) for r in ranks],
            source_manifest=ref(directory/'manifest.json'))
    result=dict(status='verified_for_user_authorized_deployment',version=prepared['version'],
        parent_version='fla-seed-range-20261009-v1',prepared=ref(LOCAL/'prepared.json'),owners=prepared['owners'],
        algebra=prepared['identity'],range_scope=prepared['range_guard'],
        official_checks=dict(receipt=ref(LOCAL/'official-result.json'),checks_passed=22,
            thresholds=official['official_thresholds'],cases=official['cases'],
            scope=official['reference']),
        noncoincident_equivalence=dict(receipt=ref(LOCAL/'nonzero-result.json'),checks_passed=5,
            checks=nonzero['checks'],scope=nonzero['scope']),
        saved_range_census=dict(receipt=ref(LOCAL/'seed-census.json'),elements=sum(r['elements'] for r in seeds['ranks']),
            old_FP16_nonfinite=sum(r['old_FP16_nonfinite'] for r in seeds['ranks']),new_FP16_nonfinite=0,
            zeroed_L2_fractions=[r['zeroed_L2_fraction'] for r in seeds['ranks']]),
        replays=replays,cost=prepared['cost'],
        timing_scope='Cold integration calls and primitive first calls include compilation/preparation; no stable speedup claim. '
            'Same FLA callback count and tensor dimensions; accepted range guard retained.',
        QVA_whitening_PPO_changed=False,official_tolerance_changed=False,FA_changed=False,
        credit_clipping_or_correction=False,training_restart=False,checkpoint_restore=False,
        extreme_credit_accuracy_repaired=False,original_PPO_NaN_repaired=False,production_deployed=False)
    output=REPO/'experiments/rl/results_fla_early_output_scale_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=ref(output),status=result['status'],version=result['version'])))


if __name__=='__main__':
    main()
