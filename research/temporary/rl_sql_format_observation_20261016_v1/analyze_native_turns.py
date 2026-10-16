"""Use the native SQL format function on exact observed environment history."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/sql-native-format-observation-20261016-v1'
prepared=json.loads((ROOT/'candidates/sql-native-format-observation-20261016-v1/prepared.json').read_bytes())
if '--owner' not in sys.argv:
    env=dict(os.environ);env.update(prepared['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--owner'],env=env,cwd=prepared['entry']))

from skyrl_gym.envs.sql.utils import verify_format_and_extract
fn=verify_format_and_extract
source_path=Path(inspect.getsourcefile(fn))
source_lines=source_path.read_text().splitlines()
cases=[]
for path in sorted((AUDIT/'native-turns').glob('*.json')):
    record=json.loads(path.read_bytes())
    for index,turns in enumerate(record['turns']):
        # Exact assistant text and observations already emitted by SQLEnv,
        # in the original env.chat_history order. No decode or SQL parser.
        history=''.join(turn['output_text']+''.join(obs['content'] for obs in (turn['observations'] or []) if obs)
                        for turn in turns)
        returns=[]
        def observe(frame,event,arg):
            if frame.f_code is fn.__code__ and event=='return':
                returns.append(frame.f_lineno)
            return observe
        old=sys.gettrace()
        sys.settrace(observe)
        try:
            valid,_,sql,_=fn(history)
        finally:
            sys.settrace(old)
        assert len(returns)==1
        line=returns[0]
        cases.append(dict(native_artifact=str(path),artifact_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            row=index,turns=len(turns),native_terminal_score=record['terminal_scores'][index],
            native_format_valid=valid,native_return_line=line,
            native_branch_source='\n'.join(f'{i+1}: {source_lines[i]}' for i in range(max(0,line-4),line)),
            sampled_think_starts=sum(turn['output_text'].count('<think>') for turn in turns),
            sampled_think_ends=sum(turn['output_text'].count('</think>') for turn in turns),
            sampled_solution_starts=sum(turn['output_text'].count('<solution>') for turn in turns),
            sampled_solution_ends=sum(turn['output_text'].count('</solution>') for turn in turns),
            first_action=turns[0]['output_text'] if turns else '',last_action=turns[-1]['output_text'] if turns else '',
            stop_reason=record['stop_reasons'][index]))
assert cases,'No completed native generation artifact yet'
from collections import Counter
summary=dict(collected_unix=time.time(),native_format_owner=str(source_path),
    native_format_owner_SHA256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    cases=len(cases),native_return_lines=dict(Counter(case['native_return_line'] for case in cases)),
    terminal_scores=dict(Counter(case['native_terminal_score'] for case in cases)),
    sampled_think_start_counts=dict(Counter(case['sampled_think_starts'] for case in cases)),
    sampled_solution_start_counts=dict(Counter(case['sampled_solution_starts'] for case in cases)),
    source_branches={case['native_return_line']:case['native_branch_source'] for case in cases},
    model_DT_optimizer_calls=0,training_mutations=0)
(AUDIT/'native-format-analysis.json').write_text(json.dumps(dict(summary=summary,cases=cases),indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('FIRST_ACTION_EXAMPLE',cases[0]['first_action'][:1200])
print('LAST_ACTION_EXAMPLE',cases[0]['last_action'][-1200:])
