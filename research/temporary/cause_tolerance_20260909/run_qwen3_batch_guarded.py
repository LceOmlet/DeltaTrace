"""Run the frozen development and transfer cohorts sequentially on one GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
python = '/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'
for name, phase, reference in [
    ('qwen3_batch16_guarded', 'development16', Path('/tmp/codex_clean_development16_20260909_v1/qwen3/results.json')),
    ('qwen3_batch_extra', 'extra', root / 'qwen3_batch_extra_inputs.json'),
]:
    command = [python, '-B', str(root / 'benchmark_qwen3_batch_guarded.py'),
               '--release', '/tmp/codex_deferred_promoted_api_20260909_v1',
               '--environment', '/tmp/codex_clean_development16_20260909_v1/environment.json',
               '--reference', str(reference), '--reference-sha256', hashlib.sha256(reference.read_bytes()).hexdigest(),
               '--phase', phase, '--output', str(root / name)]
    with (root / (name + '.log')).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        receipt = {'pid': process.pid, 'command': command, 'started': time.time()}
        path = root / (name + '_launch.json')
        path.write_text(json.dumps(receipt, indent=2) + '\n')
        receipt.update(exit_code=process.wait(), finished=time.time())
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['exit_code']:
        raise RuntimeError(f'{name} failed; subsequent GPU work was not started')
