"""Account for existing historical target derivatives; no model execution."""
import json
import math
from collections import defaultdict
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST = OUT/'historical-target-derivatives-v1'
state = json.loads((DEST/'result.json').read_bytes())
report = {'unix':time.time(),'scope':__doc__,'phase':state['phase'],
          'model_calls':0,'optimizer_steps':0,'rows':[],
          'limits':[
              'The observations freeze each entry checkpoint across four minibatches; not a replay of the original intervening updates.',
              'Early positive IDs/masks/positions are native; failure denominator counts and printed alpha have recorded approximation error. Iteration81 counts/alpha are original.',
              'Effective parameter/logit gain is directional sensitivity on the actual score vector, not a singular value or a separate causal percentage.',
              'Per-parameter squared norms add over original sharded rank ownership; original native clip returns are retained separately.',
          ]}
labels = ('historical_target_B4','historical_target_global64')
for observation in state.get('observations',[]):
    iteration,mini = observation['iteration'],observation['mini']
    records = [json.loads((DEST/f'iteration{iteration}-mini{mini}-rank{rank}.json').read_bytes()) for rank in (0,1)]
    row = {'iteration':iteration,'checkpoint':observation['checkpoint'],'mini':mini,
           'gradient_statistics':observation['native_gradient_statistics'],
           'native_clip_returns_by_rank':observation['native_clip_returns_by_rank'],
           'output_score':observation['output_score'],'parameter_distribution':{},
           'forward_point_matches_between_weight_views_by_rank':[]}
    for rank in (0,1):
        losses = records[rank]['passes']
        first,second = [losses[label]['microbatch_losses'] for label in labels]
        equal = all(a[k] == b[k] for a,b in zip(first,second)
                    for k in ('weighted_entropy','weighted_kl','ppo_kl','pg_clipfrac','pg_clipfrac_lower') if k in a and k in b)
        row['forward_point_matches_between_weight_views_by_rank'].append(equal)
    for label,mode in zip(labels,('B4','global64')):
        parameters = defaultdict(float)
        for record in records:
            for name,metadata in record['parameters'][label].items():
                assert metadata['statistics_bucket'][0] == 'sharded'
                parameters[name] += metadata['local_gradient_sumsq'] or 0.
        total = sum(parameters.values())
        recorded = observation['native_gradient_statistics']['inner_products'][f'{label}:{label}']
        assert math.isclose(total,recorded,rel_tol=1e-12,abs_tol=1e-15),(iteration,mini,label,total,recorded)
        layers = defaultdict(float)
        for name,value in parameters.items():
            layers[name.split('.layers.')[1].split('.')[0]] += value
        a = sum(value for name,value in parameters.items() if '.lora_A.' in name)
        norm = recorded**.5
        lower,upper = observation['output_score']['groups'][mode]['output_logit_L2_bounds']
        row['parameter_distribution'][mode] = {
            'norm':norm,'LoRA_A_squared_fraction':a/total,
            'top_layers_squared_fraction':sorted([(k,v/total) for k,v in layers.items()],key=lambda p:p[1],reverse=True)[:8],
            'top_parameters_squared_fraction':sorted([(k,v/total) for k,v in parameters.items()],key=lambda p:p[1],reverse=True)[:8],
            'effective_parameter_to_logit_gain_bounds':[norm/upper,norm/lower],
        }
    report['rows'].append(row)
path = OUT/'historical-target-derivative-accounting-v4.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
