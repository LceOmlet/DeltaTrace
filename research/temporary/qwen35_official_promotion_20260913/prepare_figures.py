"""Freeze three existing illustrative inputs for the new official profile."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / 'qwen35_niah_causal_20260913'
code = (OLD / 'run_generalization.py').read_text(encoding='utf-8')
def replace(a, b):
    global code
    assert code.count(a) == 1, a
    code = code.replace(a, b)

replace('"""Frozen broader-task check of the deployed symmetric GDN profile."""',
        '"""Recompute fixed illustrations and verify the official default on native GPU."""')
replace("            assert not gold,'This check expects tasks without retrieval labels.'\n", '')
replace('from qwen35_gdn_symmetric import make_qwen35_gdn_symmetric_runner',
        'from deltatrace.profiles.official import make_qwen35_runner')
replace("runners['DT_gdn_symmetric']=make_qwen35_gdn_symmetric_runner(",
        "runners['DT_gdn_symmetric']=make_qwen35_runner(")
replace("report['deployed_profile_sha256']=digest(Path(__file__).with_name('qwen35_gdn_symmetric.py'))",
        "report['deployed_profile_sha256']=digest(Path(__file__).with_name('qwen35_gdn_symmetric.py'))\n"
        "        report['official_dispatch_sha256']=digest(Path(__file__).parent/'deltatrace/profiles/official.py')\n"
        "        report['attribution_profile']='gdn-symmetric-v1'")
# No timing benchmark is needed for illustrative scores.
start = code.index('                # Both paths have already run.')
end = code.index("            repeated=pull('DT_original_repeat'", start)
code = code[:start] + code[end:]
replace("for name in (Path(__file__).name,'dt_variants.py','run_causal.py','qwen35_gdn_symmetric.py'):",
        "for name in (Path(__file__).name,'dt_variants.py','run_causal.py','qwen35_gdn_symmetric.py','deltatrace/profiles/official.py'):")
replace("(a.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())",
        "(a.output/Path(name).name).write_bytes((Path(__file__).parent/name).read_bytes())")
compile(code, 'run_figures.py', 'exec')
(HERE/'run_figures.py').write_text(code, encoding='utf-8')
plan = json.loads((OLD/'generalization_fresh_protocol.json').read_bytes())
selection = {'niah_mq_q2': [0], 'morehopqa': [0, 1]}
plan.update(version='qwen35-official-gdn-symmetric-illustrations-v1', selection=selection,
            expected_cases=3, cost='No cost benchmark; quality illustrations only.',
            validation='Existing manuscript examples fixed before recomputation; excluded from the 72-case table.',
            integration='Invoke the official factory without a profile argument. Verify its first source vector bitwise against the frozen prototype.',
            validation_success='Complete all three fixed examples; preserve native input, response, source mapping, and evaluator.')
cases=[]
cache=ROOT.parent/'audit/published_flashtrace/table1-data-v1/extracted/data'
for task, indices in selection.items():
    rows=[json.loads(line) for line in (cache/(task+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    for i in indices:
        row=rows[i]
        cases.append(dict(dataset=task,index=i,prompt=row['prompt'],target=row['target'],metadata=row['metadata']))
raw=(json.dumps(dict(cases=cases),ensure_ascii=False,indent=2)+'\n').encode()
(HERE/'figure_cases.json').write_bytes(raw)
plan['input_cases_sha256']=hashlib.sha256(raw).hexdigest()
(HERE/'figure_protocol.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
for name in ('run_causal.py','dt_variants.py'):
    (HERE/name).write_bytes((OLD/name).read_bytes())
(HERE/'qwen35_gdn_symmetric.py').write_bytes((ROOT/'deltatrace/profiles/qwen35_gdn_symmetric.py').read_bytes())
print(json.dumps({'cases':len(cases),'selection':selection}))
