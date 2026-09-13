"""Freeze a broader-task check and a deployment identity/cost check."""
import hashlib
import json
from pathlib import Path

own = Path(__file__).resolve().parent
source = (own / 'run_validation.py').read_text(encoding='utf-8')


def replace(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace('"""Screen uniform DT finite-rule variants under the unchanged paper objective."""',
        '"""Frozen broader-task check of the deployed symmetric GDN profile."""')
replace("for name in (Path(__file__).name,'dt_variants.py','run_causal.py'):",
        "for name in (Path(__file__).name,'dt_variants.py','run_causal.py','qwen35_gdn_symmetric.py'):")
replace("        runners=build_uniform_runners(model,finite,memory,options)",
        """        runners=build_uniform_runners(model,finite,memory,options)
        from qwen35_gdn_symmetric import make_qwen35_gdn_symmetric_runner
        prototype=runners['DT_gdn_symmetric']
        runners['DT_gdn_symmetric']=make_qwen35_gdn_symmetric_runner(
            model,finite,memory,dynamic_shapes=True,compiler_options=options)
        report['deployed_profile_sha256']=digest(Path(__file__).with_name('qwen35_gdn_symmetric.py'))""")
replace("            gold=dataset_utils.ruler_gold_prompt_token_indices(ex,tokenizer)",
        "            gold=dataset_utils.ruler_gold_prompt_token_indices(ex,tokenizer)\n            assert not gold,'This check expects tasks without retrieval labels.'")
replace("max_replay_l2=max(x['replay_relative_L2'] for x in detail['layers'].values()),target_offsets=list(range(gl)))",
        """max_replay_l2=max(x['replay_relative_L2'] for x in detail['layers'].values()),target_offsets=list(range(gl)),
                    attribution_seconds=detail['complete_attribution_seconds_with_diagnostics'],
                    peak_allocated_bytes=detail['peak_allocated'])""")
replace("                case['methods'][name]['recall']=float(ft.evaluate_attr_recovery_skip_tokens(torch.tensor(vectors[name])[None],\n                    keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1))",
        "                case['methods'][name]['recall']=None")
replace("            repeated=pull('DT_original_repeat',runners['DT_original'],pair)",
        """            if ordinal==0:
                expected=pull('DT_gdn_symmetric_prototype',prototype,pair)
                actual=vectors['DT_gdn_symmetric_full_sequence']
                case['deployment_relative_l2']=float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),1e-30))
                case['deployment_bitwise']=bool(np.array_equal(expected,actual))
                assert case['deployment_bitwise'],case['deployment_relative_l2']
                del expected,actual
                # Both paths have already run. Warm once more, then alternate
                # five measured pairs on this exact input with the same runner.
                case['profile_cost']={'unit':'Complete attribution with existing CPU checkpoints and diagnostics; no scoring/model loading.',
                    'warmup_pairs':1,'measurement_pairs':5,'rows':[]}
                for rep in range(6):
                    order=plan['methods'] if rep%2==0 else list(reversed(plan['methods']))
                    for method in order:
                        label=method+'_timing';timed=pull(label,runners[method],pair)
                        assert np.array_equal(timed,vectors[method+'_full_sequence'])
                        if rep:
                            d=case['methods'][label]
                            case['profile_cost']['rows'].append(dict(method=method,rep=rep,
                                seconds=d['attribution_seconds'],peak_allocated_bytes=d['peak_allocated_bytes']))
                        del timed
            repeated=pull('DT_original_repeat',runners['DT_original'],pair)""")
replace("vectors['FT_K1']=vv[key+'_FT_K1_prompt'].copy();vectors['FT_K3']=vv[key+'_FT_K3_prompt'].copy()",
        "vectors['FT_K1']=vv[key+'_FT_K1_prompt'].copy()")
replace("                case['FT_K3_recall']=old['FT_K3_needle']", "                case['FT_K3_recall']=None")
replace("                for hops in (1,3):", "                for hops in (1,):")
replace("                case['FT_K3_recall']=float(ft.evaluate_attr_recovery_skip_tokens(torch.tensor(vectors['FT_K3'])[None],\n                    keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1))",
        "                case['FT_K3_recall']=None")
replace("            ft_recall=float(ft.evaluate_attr_recovery_skip_tokens(torch.tensor(vectors['FT_K1'])[None],\n                    keep_prompt_token_indices=keep,gold_prompt_token_indices=gold,top_fraction=.1))",
        "            ft_recall=None")
compile(source, 'run_generalization.py', 'exec')
(own / 'run_generalization.py').write_text(source, encoding='utf-8')
protocol = json.loads((own / 'validation_protocol.json').read_bytes())
protocol.update(version='frozen-gdn-symmetric-broader-task-check-v1',
                selection={task: [10,26,42,58,74,90] for task in ('math','morehopqa')},
                expected_cases=12,
                validation='Fixed original benchmark indices, selected before these candidate results. Previously inspected benchmark; descriptive transfer check, not unseen holdout.',
                metrics='Original signed RISE and positive-part MAS; both tasks lack retrieval gold labels.',
                validation_success='Report all paired RISE/MAS differences and all task regressions. This check does not reselect the frozen candidate.',
                integration='First case deployed factory versus frozen prototype must match bitwise. Every baseline repeat must retain relative L2 <1e-4.',
                cost='First case: once both paths have run, one extra warm pair and five alternating timed pairs. Report full attribution diagnostic time and peak allocated memory; do not extrapolate to batched production.')
protocol.pop('validation_cases_sha256')
protocol['bootstrap']['unit'] = 'Paired cases sampled within each of two tasks; equal task mean weighting. Descriptive intervals.'
(own / 'generalization_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
for name in ('run_generalization.py', 'generalization_protocol.json'):
    print(name, hashlib.sha256((own / name).read_bytes()).hexdigest())
