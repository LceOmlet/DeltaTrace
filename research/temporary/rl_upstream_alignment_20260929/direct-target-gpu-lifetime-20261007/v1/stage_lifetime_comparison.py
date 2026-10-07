"""Stage diagnostic composition; reuse existing native worker control."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('_existing_transport', AUDIT / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


if __name__ == '__main__':
    remote = stage.ROOT + '/receipts/direct-target-gpu-lifetime-20261007-v1'
    candidate = stage.ROOT + '/candidates/direct-target-gpu-lifetime-20261007-v1/appworld'
    original = AUDIT / 'direct-target-numerics-20261007/v4'
    files = {name: original / name for name in (
        'diagnose_native_prefix_leases.py', 'run_direct_target_numeric_owner.py',
        'remote_probe_control.py', 'actual-direct-target-inputs.json')}
    files['diagnose_direct_target.py'] = HERE / 'diagnose_direct_target.py'
    files.update({name: AUDIT / name for name in (
        'verify_native_prefix_artifacts.py', 'observe_native_gdn0_operands.py',
        'observe_native_fa3_operands.py')})
    baseline = AUDIT / 'direct-target-causal-prefix-20261007/v3/memory-source/clean/qwen35/qwen35_gdn_finite.py'
    files['baseline_qwen35_gdn_finite.py'] = baseline
    config = dict(base_diagnostic=dict(
        path=stage.ROOT + '/receipts/direct-target-numerics-20261007-v4/diagnose_direct_target.py',
        sha256='0346a466ccf4e7d8d05a67e14fa32f09824d23bcda304c5e3d87457886549a1d'),
        baseline_gdn=dict(path=remote + '/baseline_qwen35_gdn_finite.py',
        sha256='ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'))
    (HERE / 'lifetime-comparison.json').write_text(json.dumps(config, indent=2) + '\n')
    files['lifetime-comparison.json'] = HERE / 'lifetime-comparison.json'
    for name, path in files.items():
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
    archive = HERE / 'comparison-source.tar'
    with tarfile.open(archive, 'w') as bundle:
        for name, path in files.items():
            bundle.add(path, arcname=name)
    subprocess.run(stage.SSH + ['mkdir', remote], check=True)
    subprocess.run(stage.SCP + [str(archive), stage.SSH[-1] + ':' + remote + '/source.tar'], check=True)
    script = ('set -eu\nsource ' + stage.ENTRY + '/metax-entry.env.sh\n'
              'tar -xf ' + remote + '/source.tar -C ' + remote + '\n'
              'CUDA_VISIBLE_DEVICES=\'\' MACA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=1 "$VENV_PYTHON" '
              + remote + '/remote_probe_control.py prepare --candidate ' + candidate + ' --out ' + remote + '\n')
    (HERE / 'comparison-prepare-command.sh').write_text(script)
    result = subprocess.run(stage.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=180)
    (HERE / 'comparison-prepare.stdout.txt').write_bytes(result.stdout)
    (HERE / 'comparison-prepare.stderr.txt').write_bytes(result.stderr)
    receipt = dict(returncode=result.returncode, out=remote, candidate=candidate,
        sources={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for name, path in files.items()},
        scope='Composition/source staging and original CPU inspection only; no model or GPU call')
    (HERE / 'comparison-staging.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
