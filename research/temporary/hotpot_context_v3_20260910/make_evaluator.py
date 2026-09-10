"""Schedule the existing answer-conditioned branch with strict full-input checks."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
V2=HERE.parent/'hotpot_fairness_20260910'
OLD=HERE.parent/'source_v2_gpu_20260910'
source=(V2/'evaluate_answer.py').read_text(encoding='utf-8')
original_math=source[source.index('                    ex = copy.deepcopy'):source.index('                    if index in control_cases:')]
code=source.replace("choices=['fairness']","choices=['context_v3']")
code=code.replace("'fairness':plan['chunks'][args.chunk]","'context_v3':plan['chunks'][args.chunk]")
code=code.replace("'targets':{'hotpotqa_long':'answer_only'}","'targets':{'hotpotqa_long':'answer_conditioned'}")
code=code.replace("r['target_mode']=='answer_only'","r['target_mode']=='answer_conditioned'")
code=code.replace("experiment='hotpot-answer-fairness-v2'","experiment='hotpot-context-cost-v3'")
needle="    assert args.chunk in plan['chunks']"
code=code.replace(needle,needle+"\n    assert sha((FAIR/'full_input_identity.json').read_bytes()) == plan['input_identity_sha256']\n    input_identity = {r['index']:r for r in json.loads((FAIR/'full_input_identity.json').read_bytes())['cases']}\n    assert sha((ROOT/'experiments/official/hotpot_retrieval_v3.py').read_bytes()) == plan['attribution_module_sha256']")
code=code.replace('    from ft_target_control import InitialTargetFT','    from ft_target_control import InitialTargetFT\n    from hotpot_retrieval_v3 import answer_conditioned_weights')
assert original_math==code[code.index('                    ex = copy.deepcopy'):code.index('                    if index in control_cases:')]
check='''                    expected = input_identity[index]
                    assert target_mode == 'answer_conditioned' and ex.target == raw_case['target']
                    assert row['input_sha256'] == expected['input_sha256'], 'Original full input changed'
                    assert row['references']['full'] == expected['reference_sha256'], 'Original reference changed'
                    assert row['target_length'] == expected['target_length'] and row['prompt_length'] == expected['prompt_length']
                    assert row['original_target_sha256'] == expected['original_target_sha256']
                    assert target_weights == answer_conditioned_weights(engine.generation_tokens, ex.indices_to_explain, gen_len-1)
                    expected_weights = [float(answer_start <= j <= answer_end)*w for j,w in enumerate(expected['full_target_weights'])]
                    assert target_weights == expected_weights
                    row['complete_input_and_reference_preserved'] = True
'''
needle="                    seeds = [('DT_target', target_weights)]"
assert code.count(needle)==1
code=code.replace(needle,check+needle)
assert original_math==code[code.index('                    ex = copy.deepcopy'):code.index('                    if index in control_cases:')].replace(check,'')
compile(code,'evaluate_conditioned.py','exec')
(HERE/'evaluate_conditioned.py').write_text(code,encoding='utf-8')
controller=(V2/'run_answer.py').read_text(encoding='utf-8')
controller=controller.replace('answer-only','answer-conditioned').replace('evaluate_answer.py','evaluate_conditioned.py')
controller=controller.replace("'--stage','fairness'","'--stage','context_v3'")
(HERE/'run_conditioned.py').write_text(controller,encoding='utf-8')
print('Unchanged computation block SHA256:',hashlib.sha256(original_math.encode()).hexdigest())
