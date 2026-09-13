"""Preserve the fixed broader inputs but measure all baselines in one fresh process.

Two archive-identity attempts stopped before quality scoring because the native
root output already differed across processes. The original guard remains in the
archival driver. This separate protocol supplies the same cached inputs through
its fresh-input path and recomputes both DT and FT. No candidate is reselected.
"""
import hashlib
import json
from pathlib import Path

own = Path(__file__).resolve().parent
source = own.parents[3] / 'audit/published_flashtrace/table1-data-v1/extracted/data'
protocol = json.loads((own / 'generalization_protocol.json').read_bytes())
cases, sources = [], []
for task, indices in protocol['selection'].items():
    file = source / (task + '.jsonl')
    data = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
    sources.append(dict(dataset=task, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    for index in indices:
        row = data[index]
        cases.append(dict(dataset=task, index=index, prompt=row['prompt'], target=row['target'], metadata=row['metadata']))
payload = dict(version='unchanged-broader-inputs-fresh-baselines-v1', cases=cases, sources=sources,
               candidate_receipt_sha256=protocol['candidate_receipt_sha256'],
               scope='Same twelve indices frozen before validation results, with original prompts and targets. Recompute all baselines in one process; do not mix historical vectors with current native outputs.')
case_file = own / 'generalization_fresh_cases.json'
with case_file.open('x', encoding='utf-8') as f:
    f.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
protocol.update(version='gdn-symmetric-broader-fresh-baselines-v1',
                input_cases_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),
                baseline_policy='Fresh DT_original, DT_gdn_symmetric and FT_K1 in the same native process. Keep within-process repeat relative-L2 guard at 1e-4; deployed candidate must match the frozen prototype bitwise. Historical inputs must match, while historical root/vector differences are recorded as separate diagnostics.',
                amendment_reason='Two pre-metric archival checks failed with native root effects 73.5349377861096 and 72.6441776777192 versus historical 73.18754371536602, while each process was bitwise stable and the candidate matched its prototype. No attribution or scoring rule changed.',
                integration='First case deployed factory versus frozen prototype must match bitwise. Every within-process baseline repeat retains relative L2 <1e-4. Same inputs as the fixed original transfer selection.',
                validation_success='Report every task and paired RISE/MAS difference using fresh native baselines. No candidate reselection. Historical drift is descriptive and is not counted as candidate improvement.')
with (own / 'generalization_fresh_protocol.json').open('x') as f:
    f.write(json.dumps(protocol, indent=2) + '\n')
print(json.dumps(dict(cases=len(cases), input_sha256=protocol['input_cases_sha256'])))
