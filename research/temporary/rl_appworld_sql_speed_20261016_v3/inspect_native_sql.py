"""Read pinned SQL owner source and saved trial transcripts; no model calls."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-sql-native-speed-20261016-v3'
prepared = json.loads((ROOT / 'candidates/appworld-sql-native-speed-20261016-v3/SQL/prepared.json').read_bytes())
if '--owner' not in sys.argv:
    env = dict(os.environ)
    env.update(prepared['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run([env['VENV_PYTHON'], str(Path(__file__).resolve()), '--owner'],
                            cwd=prepared['entry'], env=env, capture_output=True)
    (AUDIT / 'native-SQL-inspection.stdout.log').write_bytes(result.stdout)
    (AUDIT / 'native-SQL-inspection.stderr.log').write_bytes(result.stderr)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    raise SystemExit(result.returncode)

import skyrl_gym.envs.sql.env as sqlenv
import skyrl.train.generators.skyrl_gym_generator as generator
import hashlib
files = {}
for module in (sqlenv, generator):
    path = Path(module.__file__)
    files[str(path)] = dict(SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), source=path.read_text())
for path in Path(sqlenv.__file__).parent.rglob('*.py'):
    if str(path) not in files:
        files[str(path)] = dict(SHA256=hashlib.sha256(path.read_bytes()).hexdigest(), source=path.read_text())
out = dict(files=files, prepared=str(ROOT / 'candidates/appworld-sql-native-speed-20261016-v3/SQL/prepared.json'))
(AUDIT / 'native-SQL-owner-sources.json').write_text(json.dumps(out, indent=2) + '\n')
for path, value in files.items():
    print('SOURCE', path, value['SHA256'])
    lines = value['source'].splitlines()
    if path == sqlenv.__file__:
        print(value['source'])
    elif any('def ' + key in value['source'] for key in ('compute_score', 'format_reward', 'get_reward')):
        print(value['source'][:18000])
    elif path == generator.__file__:
        for i, line in enumerate(lines):
            if any(key in line for key in ('reward', 'response_text', 'think', 'stop_reasons', 'assistant_content')):
                print(f'{i+1}: {line}')

rows = []
for path in sorted((Path(prepared['run']) / 'rollouts').glob('*.jsonl')):
    for index, raw in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(raw)
        text = row['output']
        rows.append(dict(path=str(path), line=index, score=row['score'],
                         tags={tag:text.count(tag) for tag in ('<think>', '</think>', '<sql>', '</sql>', '<solution>', '</solution>')},
                         head=text[:200], tail=text[-1400:]))
(AUDIT / 'native-SQL-saved-transcript-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print('ROW_COUNT',len(rows))
print(json.dumps(rows[:2], indent=2))

from transformers import AutoTokenizer
from skyrl_gym.envs.sql.utils import verify_format_and_extract
model = os.environ.get('DT_TOKENIZER_PATH') or prepared['options']['actor_rollout_ref.model.path']
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
text = '<think>Check.</think><solution>select 1</solution>'
ids = tokenizer.encode(text, add_special_tokens=False)
check = dict(tokenizer_path=model,
             special_tokens={tag:dict(ids=tokenizer.encode(tag, add_special_tokens=False),
                 is_special=[i in tokenizer.all_special_ids for i in tokenizer.encode(tag, add_special_tokens=False)])
                 for tag in ('<think>', '</think>', '<solution>', '</solution>')},
             keep_text=tokenizer.decode(ids, skip_special_tokens=False),
             skip_text=tokenizer.decode(ids, skip_special_tokens=True),
             keep_native_format_valid=verify_format_and_extract(tokenizer.decode(ids, skip_special_tokens=False))[0],
             skip_native_format_valid=verify_format_and_extract(tokenizer.decode(ids, skip_special_tokens=True))[0],
             model_calls=0, optimizer_updates=0)
(AUDIT / 'native-SQL-tokenizer-format-check.json').write_text(json.dumps(check, indent=2) + '\n')
print('TOKENIZER_INTERFACE_CHECK', json.dumps(check, indent=2))
