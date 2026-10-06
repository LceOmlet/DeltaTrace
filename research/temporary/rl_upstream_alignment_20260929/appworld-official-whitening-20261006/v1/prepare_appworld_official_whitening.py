"""Stage one isolated AppWorld owner/entry CPU preparation; never submit."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

LOCAL = Path(__file__).resolve().parent
AUDIT = LOCAL.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ENTRY, REPO, ROOT, SCP, SSH

OUT = ROOT + '/receipts/appworld-official-whitening-20261006-v1'
CANDIDATE = ROOT + '/candidates/appworld-official-whitening-20261006-v1'


def main():
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = {
        'deployment-source.json': AUDIT / 'appworld-eval-client-routing-20261005/deployment/source.json',
        'deployment-job.json': AUDIT / 'appworld-eval-client-routing-20261005/deployment/job.json',
        'deployment-launch.json': AUDIT / 'appworld-eval-client-routing-20261005/deployment/launch.json',
        'original-prepared.json': AUDIT / 'appworld-eval-client-routing-20261005/remote-prepared.json',
        'ending-review.json': AUDIT / 'appworld-eval-client-routing-20261005/deployment/ending-review-20261006.json',
    }
    files = {
        'source/experiments/rl/patch_verl_agent2.py': REPO / 'experiments/rl/patch_verl_agent2.py',
        'source/tests/test_dt_official_whitening.py': REPO / 'tests/test_dt_official_whitening.py',
        'source/owner-pristine-trainer.py': AUDIT / 'recipe-sources/verl-agent-20bd331/verl/trainer/ppo/ray_trainer.py',
        'source/inspect_appworld_official_whitening_cpu.py': LOCAL / 'inspect_appworld_official_whitening_cpu.py',
        'source/prepare_appworld_official_whitening.py': Path(__file__).resolve(),
        **{'source/inputs/' + name: path for name, path in inputs.items()},
    }
    clock = subprocess.check_output(['git', 'show', 'afe59dd:experiments/rl/reward_readout.py'], cwd=REPO)
    assert hashlib.sha256(clock).hexdigest() == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
    clock_path = LOCAL / 'accepted-reward-readout-94a7afbc.py'
    if clock_path.exists():
        assert clock_path.read_bytes() == clock
    else:
        clock_path.write_bytes(clock)
    files['source/accepted-reward-readout-94a7afbc.py'] = clock_path
    for name, path in files.items():
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
    payload = dict(root=ROOT, receipt=OUT, candidate=CANDIDATE, provisioned_entry=ENTRY,
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_sha256={name: sha(path) for name, path in files.items()},
        local_sources={name: str(path) for name, path in files.items()},
        context_sources={str(p.relative_to(REPO)): sha(p) for p in
            (REPO/'experiments/rl/PLAN.md', REPO/'experiments/rl/RUNTIME_RECORD.md',
             REPO/'experiments/rl/REMOTE_ENVIRONMENT.md', REPO/'experiments/rl/current_runtime.json')})
    payload_path = LOCAL / 'preparation-input.json'
    payload_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    subprocess.run(SSH + ['mkdir', '-p', OUT+'/source/experiments/rl', OUT+'/source/tests', OUT+'/source/inputs'], check=True)
    for name, path in files.items():
        subprocess.run(SCP + [str(path), f'{SSH[-1]}:{OUT}/{name}'], check=True)
    subprocess.run(SCP + [str(payload_path), f'{SSH[-1]}:{OUT}/preparation-input.json'], check=True)
    script = ('set -e\nsource ' + ENTRY + '/metax-entry.env.sh\n'
        'export PYTHONDONTWRITEBYTECODE=1\nexport CUDA_VISIBLE_DEVICES=""\nexport MACA_VISIBLE_DEVICES=""\n'
        '"$VENV_PYTHON" ' + OUT + '/source/inspect_appworld_official_whitening_cpu.py'
        ' --input ' + OUT + '/preparation-input.json\n')
    (LOCAL/'prepare.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL/'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    names = ('prepared.json', 'candidate-source.json', 'native-interface-inspection.json',
        'native-interface-inspection.stdout.txt', 'focused-owner-tests.stdout.txt',
        'focused-owner-tests.json', 'effective-config.yaml', 'configuration-comparison.json')
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in names] + [str(LOCAL)], check=True)


if __name__ == '__main__':
    main()
