"""Audit the completed same-task native target derivatives; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'matched-task-target-derivatives-v11'
manifest=json.loads((DEST/'case-manifest.json').read_bytes())
state=json.loads((DEST/'result.json').read_bytes())
assert state['phase']=='complete_native_gradient_chain_localization_not_repair'
summary={'unix':time.time(),'scope':__doc__,'model_calls':0,'job':json.loads((DEST/'job.json').read_bytes()),
         'cases':[],'case_manifest_SHA256':hashlib.sha256((DEST/'case-manifest.json').read_bytes()).hexdigest(),
         'limits':manifest['limits']+[
             'G/C is an accounting quotient of the measured parameter norm and coefficient norm. It combines relative weighting, score sensitivity and cross-token directions; it is not a singular value or independent causal share.',
             'This target-only historical comparison is not the original per-iteration full-objective logged norm or a fresh-training counterfactual.',
             'Residual gradients under historical reweighting cannot establish the behavior of training from the beginning with another denominator.'],
         'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None}
initial_states=[]
for case,observation in zip(manifest['cases'],state['observations']):
    name=case['name'];assert observation['iteration']==name
    records=[json.loads((DEST/f'iteration{name}-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
    unchanged=[json.loads((DEST/f'iteration{name}-rank{rank}-chain.json').read_bytes()) for rank in (0,1)]
    assert records[0]['gradient_statistics']==records[1]['gradient_statistics']
    assert all(r['trainable_SHA256_before']==r['trainable_SHA256_after'] and
               r['scheduler_before']==r['scheduler_after']==5 and
               not r['passive_boundary_readout_enabled'] for r in unchanged)
    initial_states.append([r['trainable_SHA256_before'] for r in unchanged])
    coefficient_norm=math.sqrt(sum(r['historical_target_dL_dlogp_magnitude']**2*r['original_source']['target_tokens']
                                   for r in case['coefficients']))
    gradient=observation['gradient_statistics']['norms']['native_target']
    diagnostic=[]
    for rank,record in enumerate(records):
        losses=record['passes']['native_target']['microbatch_losses']
        diagnostic.append({'rank':rank,'native_microbatches':len(losses),
            'ppo_kl':[r['ppo_kl'] for r in losses],
            'pg_clipfrac':[r['pg_clipfrac'] for r in losses],
            'pg_clipfrac_lower':[r['pg_clipfrac_lower'] for r in losses],
            'native_clip_return':record['passes']['native_target']['optimizer_boundary']['native_clip_return']})
    summary['cases'].append({'case':name,'checkpoint':5,'rows':64,'matched_initial_tasks':16,
        'native_owners':records[0]['sources'],'effective_configuration':records[0]['effective_config'],
        'native_global_sharded_preclip_target_norm_G':gradient,
        'historical_target_coefficient_norm_C':coefficient_norm,'G_over_C':gradient/coefficient_norm,
        'actual_native_forward_point_diagnostics':diagnostic,
        'original_trainable_digest_before_after_by_rank':[
            {'rank':rank,'before':r['trainable_SHA256_before'],'after':r['trainable_SHA256_after'],
             'scheduler_before':r['scheduler_before'],'scheduler_after':r['scheduler_after']}
            for rank,r in enumerate(unchanged)]})
assert initial_states[0]==initial_states[1]
a,b=summary['cases']
summary['late_over_early']={'G':b['native_global_sharded_preclip_target_norm_G']/a['native_global_sharded_preclip_target_norm_G'],
                          'C':b['historical_target_coefficient_norm_C']/a['historical_target_coefficient_norm_C'],
                          'G_over_C':b['G_over_C']/a['G_over_C']}
summary['conclusions']=[
    'Large late target derivatives remain at the same original checkpoint5 after exact matching of initial task states. Different task composition and evolved weights are not necessary for this saved-input contrast.',
    'The historical target coefficient-vector norm increases by less than the observed parameter-gradient norm. A uniform scalar coefficient increase is not a complete description of this contrast.',
    'Generated histories and relative historical coefficient placement still change together; this does not assign independent causal fractions or establish a unique temporal root cause.',
    'All training loss/backward/accumulation/reduction calls use the original fixed VERL owner; no optimizer/scheduler/DT/rollout/validation/GRPO call is added.'
]
path=DEST/'matched-native-target-derivative-summary-v11.json';path.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':summary}))
