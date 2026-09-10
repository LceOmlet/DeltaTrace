"""Freeze a task-specific target policy before reserved Recall validation."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
split=json.loads((HERE/'answer_split.json').read_bytes())
split.update(version='target-scope-v1',plan_sha256=sha(HERE/'TARGET_SCOPE_VALIDATION.md'))
targets={t:('full' if t=='hotpotqa_long' else 'answer_only') for t in split['tasks']}
development=json.loads((HERE/'answer_development/analysis.json').read_bytes())
assert development['status']=='verified_development'
assert development['run_results_sha256']==sha(HERE/'answer_dev_corrected_v1/results.json')
choice=dict(status='frozen_for_validation',targets=targets,
    target_policy='task_specific_selected_from_development',
    plan_sha256=split['plan_sha256'],development_results_sha256=development['run_results_sha256'],
    development_vectors_sha256=development['run_vectors_sha256'],
    FT_initial_target_adapter_sha256=sha(HERE/'ft_target_control.py'),
    weighted_sources={n:sha(HERE/n) for n in ('weighted_secant.py','weighted_paired.py')},
    primary_comparisons=['VT_raw','VT_density','HotpotQA_raw','HotpotQA_density'],
    adjusted_interval_quantiles=[.00625,.99375])
for name,data in [('scope_split.json',split)]:
    p=HERE/name
    if p.exists():assert json.loads(p.read_bytes())==data
    else:p.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
choice['split_sha256']=sha(HERE/'scope_split.json')
p=HERE/'scope_choice.json'
if p.exists():assert json.loads(p.read_bytes())==choice
else:p.write_text(json.dumps(choice,indent=2)+'\n',encoding='utf-8')
source=HERE/'evaluate_answer_matched.py'
assert sha(source)=='775f7359a357c00e847b67364b4b6f5f9eea64a8b952d827dbcbbbd751663b8f'
code=source.read_text(encoding='utf-8')
code=code.replace('answer-pilot-v1','target-scope-v1').replace('answer_split.json','scope_split.json').replace('ANSWER_PILOT.md','TARGET_SCOPE_VALIDATION.md')
code=code.replace("choices=['development', 'validation']", "choices=['validation']")
code=code.replace("                modes = ['full', 'answer_conditioned', 'answer_only'] if choice is None else ['full', choice['target_mode']]",
                  "                modes = [choice['targets'][dataset]]")
code=code.replace("        assert choice['status'] == 'frozen_for_validation'", "        assert choice['status'] == 'frozen_for_validation'\n        assert choice['plan_sha256'] == split['plan_sha256']\n        assert choice['targets'] == {t:('full' if t=='hotpotqa_long' else 'answer_only') for t in split['tasks']}\n        assert choice['weighted_sources'] == {n:sha((PILOT/n).read_bytes()) for n in ('weighted_secant.py','weighted_paired.py')}")
compile(code,'evaluate_target_scope.py','exec')
(HERE/'evaluate_target_scope.py').write_text(code,encoding='utf-8')
print(json.dumps({n:sha(HERE/n) for n in ('TARGET_SCOPE_VALIDATION.md','scope_split.json','scope_choice.json','evaluate_target_scope.py')},indent=2))
