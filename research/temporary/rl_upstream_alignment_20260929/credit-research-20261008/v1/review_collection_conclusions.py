"""Review existing frozen-collection conclusions without another model query.

Reuse the original summaries. Keep uniform sampling and the predicted-tail
census separate; neither pooled error means nor new acceptance tests are made.
"""
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3].parent


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def compact_cohort(value):
    cells = []
    for predicted, group in value['results']['strata'].items():
        for native, cell in group['native_strata'].items():
            if not cell['points']:
                continue
            cells.append(dict(predicted_ratio_bin=predicted, native_ratio_bin=native,
                **{k: cell[k] for k in ('points', 'trajectories', 'initial_states',
                    'sign_crossings', 'abs_d_error', 'abs_A_over_r_error')},
                state_point_counts={s['initial_state_sha256']:s['points']
                                    for s in cell['state_groups']}))
    assert sum(c['points'] for c in cells) == value['measured_points']
    return dict(expected_points=value['expected_points'],
        measured_points=value['measured_points'], cells=cells)


def main():
    started = time.perf_counter()
    author = read(HERE/'author-collection-summary.json')
    gradients = read(HERE/'collection-error-gradient-analysis.json')
    feasibility = read(HERE/'conditional-attention-feasibility.json')
    cohorts = {label:{task:compact_cohort(value) for task,value in author[key].items()}
        for label,key in (('uniform','uniform_sample'),
                          ('predicted_tail_census','observed_DT_tail_census'))}
    missed = {}
    false_negative_credit = {}
    tail_bins = {'ratio_2_to_10','ratio_10_to_100','ratio_gt_100'}
    for task in ('textcraft','appworld'):
        selected = [c for c in cohorts['uniform'][task]['cells']
                    if c['predicted_ratio_bin'] not in tail_bins
                    and c['native_ratio_bin'] in tail_bins]
        state_ids = {s for c in selected for s in c['state_point_counts']}
        missed[task] = dict(points=sum(c['points'] for c in selected),
            denominator=cohorts['uniform'][task]['measured_points'],
            initial_states=len(state_ids), initial_state_ids=sorted(state_ids),
            scope='Only the frozen uniform source sample; not all-token population frequency.')
        selected = [c for c in cohorts['predicted_tail_census'][task]['cells']
                    if c['native_ratio_bin']=='ratio_le_1']
        false_negative_credit[task] = dict(points=sum(c['points'] for c in selected),
            denominator=cohorts['predicted_tail_census'][task]['measured_points'],
            initial_states=len({s for c in selected for s in c['state_point_counts']}),
            scope='Observed DT c>2 census crossing to native c<=1; not an official numerical-tolerance failure.')
    geometry = []
    for batch in gradients['minibatches']:
        components = {}
        for label,part in batch['components'].items():
            components[label] = {k:part[k] for k in ('points','states','geometry') if k in part}
        geometry.append(dict(original_optimizer_minibatch=batch['index'],
            original_full_pg_norm=batch['full_pg_norm'], components=components))
    result = dict(status='Completed evidence-scope review; no repair selected or deployed',
        observed_unix=time.time(), source=ref(Path(__file__)),
        method=ref(ROOT/'experiments/rl/PLAN.md'),
        evidence=[ref(HERE/name) for name in (
            'author-collection-summary.json','collection-error-gradient-analysis.json',
            'decoder-group-analysis.json','conditional-owner-lifetime.json',
            'conditional-owner-lifetime-command.sh',
            'complete_conditional_attention_feasibility.py',
            'conditional-attention-feasibility.json')],
        decision_correction=dict(
            rejected_inference='One real token located at a GDN boundary does not establish a population failure mechanism or the cause of historical training degradation.',
            current_inference='The frozen collections contain both spurious predicted negative tails and missed native negative tails. Their state distribution and original-PG influence differ. Whole-decoder residuals contain opposing terms and do not identify a particular attention kernel as the cause.',
            production_repair_selected=False,
            historical_window='This is the first real joint-action-target collection. It is not the old auxiliary-label degradation window and cannot retrospectively establish its cause.'),
        grouping=dict(
            variable='c=exp(-d), A/r=1-c; joint predicted/native ratio cells',
            bins=['[0,1]','(1,2]','(2,10]','(10,100]','(100,infinity)'],
            retained_axes=['task','sampling cohort','initial state','trajectory',
                           'previously examined identity','original optimizer minibatch'],
            no_moment_inference='A bounded conditional interval has bounded A/r. A finite sample does not establish moments of the unbounded tail or of the full population. No pooled raw advantage/error mean, fitted tail exponent or population variance is reported.',
            no_sampling_mix='Uniform four-source sampling and full predicted-tail census keep their own denominators. Error gradients from the uniform sample are not extrapolated to all unsampled positions.',
            exposure='Original by_exposure and state-cell distributions remain in the referenced author summary; no re-selection or re-weighting.'),
        cohorts=cohorts, missed_native_tail_in_uniform=missed,
        predicted_tail_crossing_to_native_positive_credit=false_negative_credit,
        primary_quality=dict(values=author['author_metrics_first_stage'],
            aggregation='Original cumulative deletion / RISE / MAS, trajectory score then equal initial-state mean. No combined task score; no missing or nonfinite value filled with zero.',
            status='Baseline only; no candidate-versus-baseline improvement claim.'),
        original_pg_geometry=geometry,
        gradient_limits='Fixed original whitening scale and original loss denominator; same original PG compared with measured coefficient-error gradients. Geometric addition is not a corrected/rewhitened PPO update. No AppWorld gradient was measured and no coefficient square mass substitutes for it.',
        conditional_attention_research=dict(
            status='Unaccepted internal finite-rule hypothesis with CPU algebra/cost analysis only',
            evidence_scope='Exact complete local-row identities motivate a bounded investigation of conditional-background propagation. Whole-decoder residuals do not prove that this attention rule is the cause or that replacing it improves whole-model attribution.',
            logical_contraction_ratios_current={t:v['workload_ratio']
                for t,v in feasibility['work']['frozen_collection'].items()},
            logical_contraction_ratios_native_LSE_reuse_baseline={t:v['conditional_over_native_LSE_reuse_ratio']
                for t,v in feasibility['work']['frozen_collection'].items()},
            fair_baseline='Pinned native FA already computes LSE. A passive-capture reuse seam is source-supported, not implemented or runtime-verified. The candidate cannot be marketed as free by comparing only against avoidable paired LSE replay.',
            cost_limits='Matrix contraction counts only, not measured runtime; row reductions, SFU, traffic and physical memory lifetimes remain unmeasured. The earlier 32k capacity receipt belongs to a different memory candidate.',
            scope='No outer scaling, clipping, top-k rescue forward, GDN/head correction or production edit. No heldout scoring or parameter sweep.'),
        formal_textcraft=feasibility['formal_textcraft'],
        remote_source_observed_unix=read(HERE/'conditional-owner-lifetime.json')['unix'],
        formal_appworld='Not restarted by this work',
        operations=dict(model=0,DT=0,GPU=0,gradient=0,update=0,rollout=0,
                        checkpoint_restore=0,production_modified=False),
        elapsed_seconds=time.perf_counter()-started)
    target = ROOT/'experiments/rl/results_credit_collection_scope_review_20261008.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(result=str(target),missed_native_tail=missed,
        predicted_tail_crossings=false_negative_credit,
        operations=result['operations'],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':
    main()
