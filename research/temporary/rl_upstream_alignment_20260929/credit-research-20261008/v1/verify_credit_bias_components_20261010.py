"""Review frozen coefficient-error arithmetic, without rerunning any model.

Decimal independently evaluates the exponential identity from the original
binary float inputs. This is an arithmetic audit, not a FA/FLA tolerance or
an estimate of the current policy-gradient bias. Original strata and all
recorded reference views remain separate.
"""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'experiments/rl/PLAN.md').exists())
INPUT = REPO / 'experiments/rl/results_credit_bias_components_20261010.json'
OUTPUT = REPO / 'experiments/rl/results_credit_bias_components_audit_20261010.json'


def ref(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    started = time.perf_counter()
    original = json.loads(INPUT.read_bytes())
    for source in original['sources']:
        actual = ref(source['path'])
        assert (actual['bytes'], actual['sha256']) == (source['bytes'], source['sha256'])
    source = json.loads(Path(original['sources'][0]['path']).read_bytes())
    result = dict(scope=__doc__, observed_unix=time.time(),
                  sources=[ref(INPUT), ref(__file__), *original['sources']], tasks={},
                  arithmetic_comparison='80-digit Decimal using exact original binary-float inputs; discrepancies reported without a new pass threshold',
                  metric_units='(A_DT-A_reference)/r before native masked whitening; positive-reward captures',
                  formula=dict(epsilon='d_DT-d_reference', c='exp(-d_reference)',
                               error='c*(1-exp(-epsilon))',
                               first_order='c*epsilon',
                               curvature='c*(1-exp(-epsilon)-epsilon)<=0'),
                  theoretical_scope=original['gradient_bias'],
                  sources_for_theoretical_scope=[
                      dict(url='https://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf',
                           item='Theorem 2 and equation 6: policy-marginal outcome ratio under its stated support condition; not a theorem identifying EOS with the policy marginal'),
                      dict(url='https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf',
                           item='Baseline/control-variate variance optimization; equal expected gradients do not establish minimum variance')],
                  limits=[
                      'Frozen base-model development captures, not the current trained formal policy.',
                      'Reference view ranges are numerical sensitivity records, not confidence limits or exact world-oracle bounds.',
                      'Crossed prediction/reference magnitude cells and initial-state coverage are retained; no pooled raw credit moment.',
                      'The curvature term is an exact algebraic component, not a causal share attributable to an operator.',
                      'In particular, all false-positive cells have d_DT below the reference by construction; that sign is not independent evidence of a directional error mechanism.',
                      'Exact deletion ratios remove the estimated-d error term; a source-only baseline argument does not by itself prove the whole source/self-target piecewise gradient unbiased.',
                      'No full policy-gradient bias or training degradation is measured here.',
                      'No scalar, credit clipping, calibration, reward modification, or new production rule is introduced.'],
                  operations=dict(model=0, DT=0, backward=0, optimizer=0,
                                  GPU=0, rollout=0, remote=0),
                  production_modified=False, official_tolerance_changed=False)
    with localcontext() as ctx:
        ctx.prec = 80
        for task, data in original['tasks'].items():
            points = data['observations']
            raw_points = source['tasks'][task]['observations']
            assert len(points) == len(raw_points) == 1061
            assert sum(x['sampled_positions'] for x in data['crossed_cells'].values()) == len(points)
            maximum = {name: 0.0 for name in ('A_over_r_error', 'first_order', 'curvature')}
            compared = 0
            positive_decimal_curvature = 0
            cells = []
            for point, raw in zip(points, raw_points):
                assert point['traj_uid'] == raw['traj_uid']
                assert point['packed_slot'] == raw['packed_slot']
                assert point['token_id'] == raw['token_id']
                assert point['d_DT'] == raw['saved_d']
                assert point['weight'] == raw['token_total_weight']
                expected_refs = {k: v['d'] for k, v in raw['scores'].items()}
                expected_refs.update(recorded_interval_low=raw['native_d_interval'][0],
                                     recorded_interval_high=raw['native_d_interval'][1])
                assert set(point['references']) == set(expected_refs)
                dhat = Decimal.from_float(point['d_DT'])
                for name, values in point['references'].items():
                    assert values['d_reference'] == expected_refs[name]
                    dref = Decimal.from_float(values['d_reference'])
                    epsilon = dhat - dref
                    c = (-dref).exp()
                    exact = dict(A_over_r_error=c - (-dhat).exp(),
                                 first_order=c * epsilon,
                                 curvature=c * (1 - (-epsilon).exp() - epsilon))
                    positive_decimal_curvature += int(exact['curvature'] > 0)
                    for field, value in exact.items():
                        difference = abs(Decimal.from_float(values[field]) - value)
                        maximum[field] = max(maximum[field], float(difference))
                    compared += 1
            for cell, statistics in data['crossed_cells'].items():
                views = {}
                for name, values in statistics['references'].items():
                    shares = values['negative_error_curvature_share_weighted_quantiles']
                    views[name] = dict(
                        median_log_ratio_error=values['weighted_quantiles']['d_error']['0.5'],
                        median_coefficient_error=values['weighted_quantiles']['A_over_r_error']['0.5'],
                        negative_error_positions=values['negative_error_positions'],
                        median_curvature_share_of_negative_error=None if shares is None else shares['0.5'])
                cells.append(dict(cell=cell, sampled_positions=statistics['sampled_positions'],
                                  sampled_states=statistics['sampled_states'],
                                  HT_source_total=statistics['HT_source_total'], reference_views=views))
            result['tasks'][task] = dict(
                frame=data['frame'], observations=len(points),
                initial_states=len(data['by_initial_state']), compared_reference_points=compared,
                maximum_absolute_float_vs_decimal_discrepancy=maximum,
                positive_decimal_curvature_points=positive_decimal_curvature,
                crossed_cells=cells)
    result['elapsed_seconds'] = time.perf_counter() - started
    assert not OUTPUT.exists(), 'Preserve the completed diagnostic rather than overwriting it'
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=ref(OUTPUT), elapsed_seconds=result['elapsed_seconds'],
                         tasks={k: {f: v[f] for f in ('observations', 'compared_reference_points',
                                   'maximum_absolute_float_vs_decimal_discrepancy', 'positive_decimal_curvature_points')}
                                for k, v in result['tasks'].items()})))


if __name__ == '__main__':
    main()
