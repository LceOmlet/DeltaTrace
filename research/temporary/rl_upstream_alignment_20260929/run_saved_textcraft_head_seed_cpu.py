"""Transfer small saved operands and run the existing owner seed on CPU only."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP


LOCAL = AUDIT / 'textcraft-degradation-20261005/native-layout-20261006/v1'
REMOTE = ROOT + '/receipts/textcraft-matched-layout-20261006-v1/head-seed-cpu'
PYTHON = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
OWNER = ROOT + '/releases/c9cd147/clean/qwen35'
INPUT_NAME = 'native-matched-layout-logits-map.json'
SCRIPT_NAME = 'observe_saved_textcraft_head_seed_cpu.py'
OUTPUT_NAME = 'head-seed-cpu-observation.json'


if __name__ == '__main__':
    input_path = LOCAL / INPUT_NAME
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == '31fa04433a7aef288b85d9ba32fbc13bc53f54a269607fa1b905710a77035e4e'
    script_path = AUDIT / SCRIPT_NAME
    subprocess.run(SSH + ['mkdir', '-p', REMOTE], check=True)
    for path in (input_path, script_path):
        subprocess.run(SCP + [str(path), f'{SSH[-1]}:{REMOTE}/{path.name}'], check=True)
    argv = [PYTHON, REMOTE + '/' + SCRIPT_NAME, '--input', REMOTE + '/' + INPUT_NAME,
        '--owner-dir', OWNER, '--output', REMOTE + '/' + OUTPUT_NAME]
    # Reuse the same already provisioned process environment as the saved
    # native observation. Never print its private environment or credentials.
    saved = json.loads((LOCAL / 'job.json').read_bytes())
    remote_script = """/opt/conda/bin/python - <<'PY'
import json,pathlib,psutil,subprocess
binding=json.loads(__BINDING__)
p=psutil.Process(binding['pid'])
assert abs(p.create_time()-binding['birth'])<.05
env=p.environ()
env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
child=subprocess.run(binding['argv'],env=env)
raise SystemExit(child.returncode)
PY
""".replace('__BINDING__', repr(json.dumps(dict(
        pid=saved['reused_environment_pid'], birth=saved['reused_environment_pid_birth'], argv=argv))))
    old_stdout = LOCAL / 'head-seed-cpu.stdout.txt'
    if old_stdout.exists() and not (LOCAL / OUTPUT_NAME).exists():
        (LOCAL / 'head-seed-cpu.stdout.attempt-bare-env.txt').write_bytes(old_stdout.read_bytes())
    completed = subprocess.run(SSH + ['bash', '-s'], input=remote_script.encode(), capture_output=True)
    (LOCAL / 'head-seed-cpu.stdout.txt').write_bytes(completed.stdout + completed.stderr)
    completed.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{REMOTE}/{OUTPUT_NAME}', str(LOCAL / OUTPUT_NAME)], check=True)
    data = json.loads((LOCAL / OUTPUT_NAME).read_bytes())
    assert not data['cuda_initialized'] and not data['distributed_initialized']
    assert len(data['observations']) == 52
    assert all(data[key] == 0 for key in ('model_forward_calls', 'finite_decoder_calls', 'backward_calls', 'optimizer_steps', 'scheduler_steps'))
    for name, path in ((SCRIPT_NAME, script_path), (INPUT_NAME, input_path)):
        recorded = next(row for row in data['sources'] if Path(row['path']).name == name)
        assert recorded['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    print(completed.stdout.decode(errors='replace'))
