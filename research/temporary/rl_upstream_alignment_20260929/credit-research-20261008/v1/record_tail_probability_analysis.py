"""Publish one common-frame confusion/recall record from completed captures."""
import hashlib
import json
import math
from pathlib import Path
import time

from tail_probability_statistics import positive_bounds

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
LOCAL=HERE/'tail-probability-sample-v1'


def ref(path):
    raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    sample=json.loads((LOCAL/'sample.json').read_bytes())
    native=LOCAL/'completed-native-metadata'
    status=json.loads((native/'native-status.json').read_bytes())
    assert status['phase']=='complete' and len(status['completed_jobs'])==4
    result=dict(scope=__doc__,unix=time.time(),sample=ref(LOCAL/'sample.json'),
        sample_source_commit=sample['source_commit'],native_source_commit='888f048d07af1ecac60fff9b8faf6b617c0ace7b',
        statistics_source_commit='c9e78c9b',metadata_manifest=ref(native/'completed-native-metadata-manifest.json'),
        metadata_archive=ref(LOCAL/'completed-native-metadata.tar.gz'),tasks={},
        sampling_scope='Completed frozen development capture frame; positive original reward sources only. Held-out data unchanged. '
            'Not a task-wide population or a measurement of training degradation.',
        reference_scope='The original single-EOS actual target probability ratio; union of all matching recorded native-dtype/head sensitivity views. '
            'These are observed numerical ranges, not confidence bounds on an exact oracle.',
        confidence_scope='95% finite-population sampling bounds, conditional on the recorded reference ranges; pointwise per task/threshold/state. '
            'No simultaneous claim across all groups. No model/numerical error coverage.',
        interpretation='Do not pool earlier diagnostic samples. TP/FP come from the same-frame predicted-tail census; '
            'FN/TN use the saved inclusion probabilities. Estimated FN totals are not numbers of observed positives. '
            'Magnitude unsupported does not mean its negative sign is unsupported.',
        production_changed=False,formal_training_restarted=False,extreme_attribution_repaired=False)
    for task,spec in sample['tasks'].items():
        path=LOCAL/'analysis-c9e78c9b'/f'{task}.json'
        data=json.loads(path.read_bytes())['tasks'][task]
        obs=data['observations'];table=data['confusion_by_cumulative_threshold']['2']
        assert len(obs)==spec['sampled_sources']==1061
        assert all(e['reward']>0 for e in spec['entries'])
        assert math.isclose(math.fsum(table['weighted_token_total_confusion'].values()),spec['completed_frame_sources'])
        tail=[p for p in obs if p['saved_d'] < -math.log(2)]
        missed=[p for p in obs if p['saved_d']>=-math.log(2) and positive_bounds(p['native_d_interval'],2)[0]]
        uncertain=[p for p in obs if p['saved_d']>=-math.log(2) and positive_bounds(p['native_d_interval'],2)==(0,1)]
        result['tasks'][task]=dict(analysis=ref(path),frame=data['frame'],
            unified_confusion_by_threshold=data['confusion_by_cumulative_threshold'],
            observed_FN_positions=len(missed),observed_FN_states=len({p['initial_state_sha256'] for p in missed}),
            observed_non_tail_reference_unresolved=len(uncertain),
            predicted_tail_sign=dict(supported_negative=sum(p['native_d_interval'][1]<0 for p in tail),
                excluded_negative=sum(p['native_d_interval'][0]>=0 for p in tail),
                unresolved=sum(p['native_d_interval'][0]<0<=p['native_d_interval'][1] for p in tail)),
            prior_reference_sensitivity=data['prior_reference_sensitivity'],cost=data['cost'],
            by_initial_state=data['by_initial_state'],crossed_magnitude_bins=data['crossed_magnitude_bins'])
    output=REPO/'experiments/rl/results_tail_probability_analysis_20261009.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=ref(output),tasks={t:{k:d[k] for k in ('observed_FN_positions','observed_FN_states','observed_non_tail_reference_unresolved','predicted_tail_sign')}
        for t,d in result['tasks'].items()})))


if __name__=='__main__':
    main()
