"""Read a saved rollout with the pinned owner's format checker; no generation.

The original VERL dump may include the prompt in its response column. Match
only exact, uniquely rendered dataset prefixes before calling the owner. The
result is a text diagnostic, not a substitute environment score or token trace.
"""
from stage_environment_entry import remote, ROOT, ENTRY

remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' "$VENV_PYTHON" - <<'PY'
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, sys, time
import pyarrow.parquet as pq
from transformers import AutoTokenizer

root = Path('@ROOT@')
active = json.loads((root / 'active-training.json').read_text())
job = next(j for j in active['jobs'] if j['task'] == 'SkyRL-SQL')
options = json.loads((Path(job['output']) / 'launch.json').read_text())['options']
template = Path(options['+data.sql_chat_template'])
tokenizer = AutoTokenizer.from_pretrained(options['actor_rollout_ref.model.path'], local_files_only=True)
tokenizer.chat_template = template.read_text()
data_path = Path(options['data.train_files'])
rendered = {}
for row in pq.read_table(data_path, columns=['prompt']).to_pylist():
    ids = tokenizer.apply_chat_template(row['prompt'], add_generation_prompt=True,
                                        tokenize=True, return_dict=False)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    rendered[text] = hashlib.sha256(text.encode()).hexdigest()

owner_path = root / 'third_party/skyrl-gym-7d94cc/skyrl_gym/envs/sql/utils.py'
spec = importlib.util.spec_from_file_location('pinned_sql_format_owner', owner_path)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
source_lines = owner_path.read_text().splitlines()
return_lines = []
def trace(frame, event, arg):
    if frame.f_code is owner.verify_format_and_extract.__code__:
        if event == 'return':
            return_lines.append(frame.f_lineno)
        return trace
    return None

dump = Path(job['output']) / 'rollouts/1.jsonl'
rows = [json.loads(line) for line in dump.open()]
counts, samples, unmatched, disagreements = Counter(), {}, [], []
for index, row in enumerate(rows):
    prefixes = [text for text in rendered if row['output'].startswith(text)]
    if len(prefixes) != 1:
        unmatched.append(dict(index=index, prefix_matches=len(prefixes), score=row['score']))
        continue
    prefix = prefixes[0]
    suffix = row['output'][len(prefix):]
    return_lines.clear()
    sys.settrace(trace)
    try:
        valid, _, _, _ = owner.verify_format_and_extract(suffix)
    finally:
        sys.settrace(None)
    line = return_lines[-1]
    key = f'{line}: {source_lines[line-2].strip()}'
    counts[key] += 1
    if valid != (row['score'] != -1.):
        disagreements.append(dict(index=index, stored_score=row['score'], decoded_format_valid=valid))
    if key not in samples:
        samples[key] = dict(index=index, stored_score=row['score'], return_line=line,
                           prompt_sha256=rendered[prefix], suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                           start=suffix[:180], end=suffix[-500:])

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = dict(observed_unix=time.time(), driver_pid=job['pid'], source_dump=str(dump),
              source_dump_sha256=sha(dump), owner_file=str(owner_path), owner_sha256=sha(owner_path),
              template_file=str(template), template_sha256=sha(template), dataset_sha256=sha(data_path),
              rows=len(rows), exact_unique_prompt_matches=len(rows)-len(unmatched),
              unmatched=unmatched, format_agreement_with_stored_score=len(rows)-len(unmatched)-len(disagreements),
              disagreements=disagreements, owner_return_counts=dict(counts), examples=samples,
              scope='Pinned owner checker on saved decoded trajectory suffix, after unique exact owner-rendered prompt removal. No generation, parser rewrite, reward change, or DB execution. Return lines identify the first failed official condition, not all possible failures.')
out = root / 'receipts/owner-b8-dispatch-20260930/sql-formal-format-20261001.json'
out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(dict(path=str(out), sha256=sha(out), **result), ensure_ascii=False))
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY))
