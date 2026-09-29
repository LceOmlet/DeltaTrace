"""Summarize passive native outputs; finite outputs alone are not numerical parity."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
result = dict(scope=__doc__, cases=[])
for name, workers, log in (
    ('before', [900920, 903472], 'probability-native-raw-boundary.log'),
    ('initialized_pages', [987675, 990400], 'probability-native-initialized-pages.log'),
):
    rows, sources = [], []
    for pid in workers:
        path = root/'probability-native-trainer'/f'raw-vllm-{pid}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        rows.extend(row for record in records for row in record['rows'])
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            generate_calls=len(records)))
    text = (root/log).read_text()
    result['cases'].append(dict(name=name, sources=sources, log=str(root/log),
        rows=len(rows), tokens=sum(r['tokens'] for r in rows),
        nonfinite=sum(r['nonfinite'] for r in rows),
        nonfinite_rows=sum(r['nonfinite']>0 for r in rows),
        original_trainer_completed_2_iterations=('step:2 - ' in text and 'Final validation metrics' in text)))
result['mapping_fix'] = json.loads((root/'metax-cumem-initialization.json').read_text())
result['limitations'] = [
    'Numerical inference parity is separate: see vllm-owner-logprobs.json.',
    'The standalone allocator probe passed its three mapping/restore cycles but aborted at process teardown; it is not a fully passing test.',
    'This bounded GRPO reproduction does not validate DT or formal task training.',
]
(root/'vllm-native-nan-reproduction.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
