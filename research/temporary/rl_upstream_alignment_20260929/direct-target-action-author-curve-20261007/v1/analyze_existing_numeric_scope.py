"""Separate already-captured finite residuals, context contrasts and coefficient tails.

Descriptive arithmetic only: no new metric, acceptance threshold or credit.
"""
import hashlib
import json
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent;AUDIT=HERE.parents[1]
MEMORY=AUDIT/'direct-target-native-mlp-memory-20261007/v2'
PREFIX=AUDIT/'direct-target-prefix-runtime-20261007/v1/textcraft-credit-cpu-complete-1791373053'
ENDPOINT=AUDIT/'direct-target-extreme-token-endpoint-20261007/v1'


def artifact(path):
    return dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    memory=json.loads((MEMORY/'candidate-analysis-v6/analysis.json').read_bytes())
    endpoint=json.loads((ENDPOINT/'endpoint-analysis.json').read_bytes())['cases'][1]
    interaction=json.loads((AUDIT/'direct-target-reference-interaction-20261007/v1/analysis.json').read_bytes())
    output=dict(scope=__doc__,sources=[artifact(p) for p in (
        MEMORY/'candidate-analysis-v6/analysis.json',ENDPOINT/'endpoint-analysis.json',
        AUDIT/'direct-target-reference-interaction-20261007/v1/analysis.json',
        PREFIX/'token-credit-analysis-complete.json')],single_source_numeric=[])
    for r in memory['ranks']:
        d=r['single_source_eos_diagnostic']['details'];layers=d['layers']
        raw=MEMORY/f'actual-results-v6/results/rank{r["rank"]}-original_gpu_captures.pt'
        saved=torch.load(raw,map_location='cpu',weights_only=False)
        joint=float(saved['signed'][0,7260])
        phase_rows=[dict(layer=int(i),block_type=v['block_type'],
            root_output_effect=v['root_output_effect'],replay_output_effect=v['replay_output_effect'],
            input_effect=v['input_effect'],finite_input_minus_replay=v['input_effect']-v['replay_output_effect'],
            replay_relative_L2=v['replay_relative_L2']) for i,v in layers.items()]
        output['single_source_numeric'].append(dict(rank=r['rank'],
            native_paired_global_root=d['root_effect'],head_and_final_norm_effect=d['seed_effect'],
            signed_global=d['signed_sum'],
            head_plus_final_norm_difference=d['seed_effect']-d['root_effect'],
            decoder_cumulative_difference=d['signed_sum']-d['seed_effect'],
            signed_minus_global_root=d['signed_sum']-d['root_effect'],
            focused_row_root=d['per_sample'][0]['root_effect'],
            focused_row_signed=r['single_source_eos_diagnostic']['candidate_signed'],
            focused_row_difference=r['single_source_eos_diagnostic']['candidate_signed']-d['per_sample'][0]['root_effect'],
            identity_row_root_differences=[v['root_effect'] for v in d['per_sample'][1:]],
            max_native_replay_relative_L2=max(v['replay_relative_L2'] for v in layers.values()),
            max_native_replay_effect_difference=max(abs(v['root_output_effect']-v['replay_output_effect']) for v in layers.values()),
            joint_replay_d=joint,
            original_joint_d=endpoint['saved_DT_d'],
            joint_replay_minus_original=joint-endpoint['saved_DT_d'],
            single_DT_minus_joint_DT=r['single_source_eos_diagnostic']['candidate_signed']-joint,
            full_native_single_delete_d=endpoint['native_d'],
            all_other_EOS_native_marginal=interaction['joint_EOS_context']['d'],
            layers=phase_rows,source=artifact(raw)))
    statistics=json.loads((PREFIX/'token-credit-analysis-complete.json').read_bytes())
    output['textcraft_saved_actor_coefficients']={}
    for key in ('advantages/policy','advantages/prior_source','advantages/self_target'):
        v=statistics['actor_statistics'][key]
        output['textcraft_saved_actor_coefficients'][key]={name:v[name] for name in (
            'tokens','finite_negative','finite_positive','finite_sum','finite_sumsq','tail_counts',
            'one_max_abs_squared_over_finite_sumsq')}
    output['coefficient_scope']='Saved actual whole-batch-whitened coefficients, not policy-gradient norms or gradient shares. One extreme contributes 2.7603% of coefficient sum of squares overall and 9.9031% within saved prior-source positions; this does not prove it dominates the parameter update.'
    output['numeric_scope']='Head and final norm are not separated in this existing trace. Global root sums include three nonzero native identity-row drifts, which are retained, not corrected. Zero replay differences establish this cached replay used its saved native activations; they do not certify the native kernel, FA/FLA tolerance or joint per-token signs. No global residual is used to bound every joint token error.'
    output['interpretation']='This one-source diagnostic stays positive through all native root, finite seed and final signed totals. Its measured residual is distinct from the joint-versus-single attribution difference and the changed-context sign reversal. No production rule or numerical acceptance claim follows from these descriptive totals.'
    assert not torch.cuda.is_initialized()
    (HERE/'existing-numeric-scope.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('layers','source')} for r in output['single_source_numeric']]))


if __name__=='__main__':main()
