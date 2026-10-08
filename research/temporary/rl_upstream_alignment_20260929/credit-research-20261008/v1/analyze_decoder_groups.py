"""Describe signed decoder-group residuals on the frozen measured collection.

This is an additive path decomposition, not an intervention on a component.
Changing a component would change the other coefficients and endpoints too.
No residual is subtracted from a credit and no layer is ranked as a cause.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_author_collection import quantiles, stratum

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def describe(rows, groups):
    if not rows:
        return dict(points=0, states=0, status='Empty; no statistics imputed')
    def group_values(items):
        return {name:[math.fsum(p['residuals'][i] for i in indices) for p in items]
                for name,indices in groups.items()}
    def summaries(items):
        vals = group_values(items)
        out = {name:dict(signed=quantiles(v), absolute=quantiles([abs(x) for x in v]),
                        negative=sum(x<0 for x in v), positive=sum(x>0 for x in v),
                        zero=sum(x==0 for x in v)) for name,v in vals.items()}
        body = [math.fsum((vals['linear_attention_decoders'][i],
                          vals['full_attention_decoders'][i])) for i in range(len(items))]
        head = vals['final_norm_and_output_head']
        out['opposite_body_head_signs'] = sum(x*y<0 for x,y in zip(body,head))
        return out
    by_state = {}
    for p in rows:
        by_state.setdefault(p['initial_state_sha256'], []).append(p)
    states = [dict(initial_state_sha256=s, points=len(items),
        trajectories=len({p['traj_uid'] for p in items}), summaries=summaries(items))
        for s,items in sorted(by_state.items())]
    # Bounded sign frequencies retain the pre-existing trajectory -> state
    # aggregation. They do not average unbounded d or advantage values.
    frequencies = {}
    for name,indices in groups.items():
        state_frequencies = []
        for items in by_state.values():
            trajectories = {}
            for p in items:
                trajectories.setdefault(p['traj_uid'], []).append(
                    math.fsum(p['residuals'][i] for i in indices))
            state_frequencies.append(sum(sum(x<0 for x in values)/len(values)
                for values in trajectories.values())/len(trajectories))
        frequencies[name] = sum(state_frequencies)/len(state_frequencies)
    return dict(points=len(rows), trajectories=len({p['traj_uid'] for p in rows}),
        states=len(states), point_distributions=summaries(rows), by_state=states,
        negative_frequency_equal_trajectory_then_state=frequencies)


def main():
    started = time.perf_counter()
    owner_path = HERE/'gdn-owner-readonly.json'
    cfg = json.loads(owner_path.read_bytes())['model_config']['text_config']
    groups = {name:[i for i,t in enumerate(cfg['layer_types']) if t==kind]
        for name,kind in [('linear_attention_decoders','linear_attention'),
                          ('full_attention_decoders','full_attention')]}
    groups['final_norm_and_output_head'] = [cfg['num_hidden_layers']]
    assert sorted(i for indices in groups.values() for i in indices)==list(range(33))
    tasks = {}
    for task in ('textcraft','appworld'):
        inputs = [HERE/('layer-'+task+'-observations')/f'rank{rank}.json' for rank in (0,1)]
        points = []
        for path in inputs:
            value = json.loads(path.read_bytes())
            assert value['phase']=='complete'
            for batch in value['batches']:
                points.extend(batch['points'])
        assert len({(p['traj_uid'],p['packed_slot']) for p in points})==165
        cohorts = {}
        for cohort in ('uniform','predicted_tail_census'):
            selected = []
            for p in points:
                for previous in p['previous_comparisons']:
                    if previous['cohort']!=cohort:
                        continue
                    d = previous.get('saved_d',previous.get('d'))
                    selected.append(dict(p, crossed_cell=stratum(d)+':'+stratum(previous['native_single_d'])))
            cells = {}
            for cell in sorted({p['crossed_cell'] for p in selected}):
                rows = [p for p in selected if p['crossed_cell']==cell]
                cells[cell] = dict(all=describe(rows,groups), by_exposure={str(exposed):
                    describe([p for p in rows if p['previously_examined']==exposed],groups)
                    for exposed in (False,True)})
            cohorts[cohort] = dict(points=len(selected), cells=cells)
        closure = [math.fsum(p['residuals'])-(p['fresh_DT_d']-p['native_single_d']) for p in points]
        tasks[task] = dict(inputs=[ref(p) for p in inputs], points=len(points),
            decomposition_roundoff=quantiles(closure), cohorts=cohorts)
    result = dict(scope=__doc__, groups=groups, model_configuration=ref(owner_path),
        source=ref(Path(__file__)), tasks=tasks,
        interpretation='Each decoder group includes that decoder family\'s MLP, normalizations, residual, gate and mixer. A full_attention decoder sum is not a FlashAttention kernel error; a linear_attention decoder sum is not a GDN kernel error. Opposing sums can cancel. These are conditional descriptors, not causal shares, a candidate, a new metric, an official tolerance or a production credit correction.',
        primary_quality_receipt=ref(HERE/'author-collection-summary.json'),
        actual_gradient_receipt=ref(HERE/'collection-error-gradient-analysis.json'),
        model_calls=0, DT_calls=0, updates=0, elapsed_seconds=time.perf_counter()-started)
    (HERE/'decoder-group-analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(tasks={t:v['points'] for t,v in tasks.items()},
                         elapsed_seconds=result['elapsed_seconds'],model_calls=0,DT_calls=0)))


if __name__=='__main__':
    main()
