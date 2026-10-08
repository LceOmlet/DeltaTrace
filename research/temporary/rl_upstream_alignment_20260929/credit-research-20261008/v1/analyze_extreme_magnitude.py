"""Quantify the existing frozen contrasts without mixing bulk and tails."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def summary(values):
    values = sorted(values)
    if not values:
        return None
    position = .95*(len(values)-1)
    low, high = math.floor(position), math.ceil(position)
    return dict(min=values[0], median=statistics.median(values),
        p95=values[low]+(position-low)*(values[high]-values[low]), max=values[-1])


def main():
    inputs = HERE/'layer-collection-inputs.json'
    frozen = json.loads(inputs.read_bytes())
    result = dict(observed_unix=time.time(), scope=__doc__,
        inputs=dict(path=str(inputs), sha256=hashlib.sha256(inputs.read_bytes()).hexdigest()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), tasks={},
        interpretation='Native original-model literal single-EOS contrasts, not exact world effects or official tolerance failures. A/r=1-exp(-d); reward is kept outside this dimensionless comparison. No pooled raw tail/bulk mean or population-moment claim.',
        mechanism_status='Exponential conversion explains amplification, not the source of wrong d. A selected GDN30 contrast establishes only a local background-composition discrepancy. Whole-decoder cancellation does not identify a collection main cause.',
        current_objective='Explain extreme attribution; historical entropy-dominance is fixed and separate.',
        operations=dict(model=0, DT=0, GPU=0, backward=0, optimizer=0))
    for task, data in frozen['tasks'].items():
        points = []
        for entry in data['entries']:
            for query in entry['queries']:
                points.append(dict(query, traj_uid=entry['traj_uid'],
                    initial_state_sha256=entry['initial_state_sha256'],
                    predicted_d=query.get('saved_d', query.get('d'))))
        sections = [
            ('uniform_missed_negative_tail', 'uniform', lambda q: math.exp(-q['native_single_d']) > 2),
            ('predicted_tail_crossing_to_nonnegative_native', 'predicted_tail_census', lambda q: q['native_single_d'] >= 0)]
        result['tasks'][task] = {}
        for name, cohort, condition in sections:
            population = [q for q in points if cohort in q['cohorts']]
            selected = [q for q in population if condition(q)]
            state_counts = {}
            for q in selected:
                key = q['initial_state_sha256']
                state_counts[key] = state_counts.get(key, 0)+1
            result['tasks'][task][name] = dict(points=len(selected), denominator=len(population),
                trajectories=len({q['traj_uid'] for q in selected}), initial_states=len(state_counts),
                state_counts=state_counts,
                predicted_d=summary([q['predicted_d'] for q in selected]),
                native_d=summary([q['native_single_d'] for q in selected]),
                predicted_A_over_r=summary([-math.expm1(-q['predicted_d']) for q in selected]),
                native_A_over_r=summary([-math.expm1(-q['native_single_d']) for q in selected]),
                points_with_identity=selected)
    output = REPO/'experiments/rl/results_extreme_attribution_magnitude_20261008.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(output), tasks={task:{name:{k:v for k,v in value.items()
        if k not in ('points_with_identity', 'state_counts')} for name,value in sections.items()}
        for task,sections in result['tasks'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
