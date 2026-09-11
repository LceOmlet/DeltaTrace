"""Extend the frozen target evaluator's scheduling, retaining its model math."""
import hashlib
from pathlib import Path

FAIR = Path(__file__).resolve().parent
OLD = FAIR.parent / 'source_v2_gpu_20260910'
source = (OLD / 'evaluate_target_scope.py').read_text(encoding='utf-8')
original_math = source[source.index('                    ex = copy.deepcopy'):source.index("                    row['status'] = 'complete'")]
code = source.replace("PILOT = Path(__file__).resolve().parent", "FAIR = Path(__file__).resolve().parent\nPILOT = FAIR.parent / 'source_v2_gpu_20260910'\nsys.path.insert(0, str(PILOT))")
code = code.replace("choices=['validation']", "choices=['fairness']")
code = code.replace("    parser.add_argument('--choice', type=Path)", "    parser.add_argument('--chunk', required=True)\n    parser.add_argument('--control', type=Path, required=True)")
start = code.index("    if args.stage == 'validation':")
end = code.index('    return args', start)
code = code[:start] + "    assert args.datasets == ['hotpotqa_long']\n" + code[end:]
start = code.index("    split = json.loads((PILOT / 'scope_split.json')")
end = code.index('    caches = {}', start)
code = code[:start] + '''    plan = json.loads((FAIR/'protocol.json').read_bytes())
    assert plan['spec_sha256'] == sha((FAIR/'PROTOCOL.md').read_bytes())
    assert args.chunk in plan['chunks']
    split = {'plan_sha256':plan['spec_sha256'], 'tasks':{'hotpotqa_long':{
        'fairness':plan['chunks'][args.chunk], 'cache_sha256':plan['tasks']['hotpotqa_long']['cache_sha256']}}}
    choice = {'targets':{'hotpotqa_long':'answer_only'}, 'weighted_sources':plan['weighted_sources'],
              'FT_initial_target_adapter_sha256':plan['FT_initial_target_adapter_sha256']}
    assert choice['weighted_sources'] == {n:sha((PILOT/n).read_bytes()) for n in ('weighted_secant.py','weighted_paired.py')}
    assert sha((args.control/'results.json').read_bytes()) == plan['answer_control_results_sha256']
    assert sha((args.control/'vectors.npz').read_bytes()) == plan['answer_control_vectors_sha256']
    control_report = json.loads((args.control/'results.json').read_bytes())
    control_cases = {r['index']:r for r in control_report['cases']
        if r['dataset']=='hotpotqa_long' and r['target_mode']=='answer_only'}
    assert len(control_cases)==8 and control_report['weight_identity']==weight_identity
''' + code[end:]
code = code.replace("    vectors = {}", "    vectors = {}\n    control_vectors = np.load(args.control/'vectors.npz')")
old = "    report.update(experiment='target-scope-v1', stage=args.stage, split_sha256=sha((PILOT / 'scope_split.json').read_bytes()), plan_sha256=split['plan_sha256'], choice=choice)"
new = "    report.update(experiment='hotpot-answer-fairness-v2', stage=args.stage, chunk=args.chunk, indices=plan['chunks'][args.chunk], fair_plan_sha256=sha((FAIR/'protocol.json').read_bytes()), plan_sha256=split['plan_sha256'], choice=choice)"
assert old in code
code = code.replace(old, new)
assert original_math == code[code.index('                    ex = copy.deepcopy'):code.index("                    row['status'] = 'complete'")]
control = '''                    if index in control_cases:
                        old = control_cases[index]
                        for field in ('input_ids','input_sha256','target','target_mode','target_weights','keep','gold','references'):
                            assert row[field] == old[field], field
                        for suffix in ('DT_target_signed_full','FT_K1_prompt','FT_K3_prompt'):
                            assert np.array_equal(vectors[key+'_'+suffix], control_vectors[key+'_'+suffix]), suffix
                        row['development_overlap_bitwise_equal'] = True
'''
code = code.replace("                    row['status'] = 'complete'", control + "                    row['status'] = 'complete'")
compile(code, 'evaluate_answer.py', 'exec')
(FAIR/'evaluate_answer.py').write_text(code, encoding='utf-8')
print('Unchanged attribution/input/FT block SHA256:', hashlib.sha256(original_math.encode()).hexdigest())
