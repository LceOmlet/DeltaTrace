"""Prepare the verified GDN row-storage binding; never submit or stop a job.

Run on the existing host with VENV_PYTHON and CUDA/MACA hidden. The tested
linked DT tree is reused; no owner loader, environment JSON or launcher is
created. Only the copied entry's lease and original factory binding change.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DIAGNOSTIC = ROOT/'receipts/owner-b8-dispatch-20260930/native-prefix-reuse-boundary-row-storage-offset84-20261007-bc622f3-current-base'
SOURCE_SHA = '3cd90b2db649cd477bc21398e7677dc8ea5534d5230296fc5f63c04037cd3427'
SUMMARY_SHA = '277e697051c23f2a9ab741676ea91cf16c08c0216704ccd470158eb7fad0205d'
PRODUCER_SHA = '2c01c47e699b8b5a05206e881bda596cfbca03d1b89a7c3d3de2ce30ae5995a4'
LEASE_SHA = '6d2aecb0a47c3f4f6bd0e128b6da304ce7d53a602cbb331e18eea8edc745acc8'
ARTIFACT_SHA = '4a461f2168d07b5ce88dbc8468e6a0d3eaae2a2c1d879cd5281347098e2ea5a8'
ARTIFACT_KEY = 'clean/qwen35/qwen35_native_prefix_artifacts.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_factory(raw):
    """Bind the existing optional storage flag with standard functools.partial."""
    bindings = []
    for ending in (b'\n', b'\r\n'):
        old = ending.join((
            b'        from native_prefix_leases import prepare_native_prefix_leases',
            b"        self.readout_options['prefix_lease_factory'] = prepare_native_prefix_leases",
            b''))
        if raw.count(old) == 1:
            bindings.append((old, ending))
    assert len(bindings) == 1, 'Expected the frozen producer factory binding'
    old, ending = bindings[0]
    new = ending.join((
        b'        from functools import partial',
        b'        from native_prefix_leases import prepare_native_prefix_leases',
        b"        self.readout_options['prefix_lease_factory'] = partial(",
        b'            prepare_native_prefix_leases, boundary_row_storage=True)',
        b''))
    patched = raw.replace(old, new, 1)
    before, after = ast.parse(raw), ast.parse(patched)
    cls = next(n for n in after.body if isinstance(n, ast.ClassDef)
               and n.name == 'DeltaTraceRolloutProducer')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    imports = [n for n in init.body if isinstance(n, ast.ImportFrom)
               and n.module == 'functools' and [(a.name, a.asname) for a in n.names] == [('partial', None)]]
    assert len(imports) == 1
    init.body.remove(imports[0])
    assignments = [n for n in init.body if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == 'partial']
    assert len(assignments) == 1
    call = assignments[0].value
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name)
    assert call.args[0].id == 'prepare_native_prefix_leases'
    assert len(call.keywords) == 1 and call.keywords[0].arg == 'boundary_row_storage'
    assert isinstance(call.keywords[0].value, ast.Constant) and call.keywords[0].value.value is True
    assignments[0].value = call.args[0]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    compile(patched, '<prepared-storage-factory>', 'exec')
    return patched


CPU_INSPECTION = '''import hashlib,inspect,json,torch
from pathlib import Path
import deltatrace_rollout as producer
import native_prefix_leases as lease
import qwen35_native_prefix_artifacts as artifact
import qwen35_dense_finite_runner as runner
from transformers.models.qwen3_5 import modeling_qwen3_5 as model
sources={}
for label,module in [('producer',producer),('lease',lease),('artifact',artifact),('runner',runner),('model',model)]:
 p=Path(inspect.getsourcefile(module));sources[label]={'path':str(p),'resolved_path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'sources':sources,'lease_signature':str(inspect.signature(lease.prepare_native_prefix_leases)),
 'cuda_initialized':torch.cuda.is_initialized(),'distributed_initialized':torch.distributed.is_initialized()}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/appworld-fresh-native-conv-canonical-20261007-v2/appworld-dt/source.json')
    parser.add_argument('--runtime-summary', type=Path, default=DIAGNOSTIC/'runtime-summary.json')
    parser.add_argument('--output', type=Path, default=ROOT/'candidates/appworld-native-boundary-row-storage-20261007-v1')
    parser.add_argument('--receipt', type=Path, default=ROOT/'receipts/appworld-efficiency-20261007/native-boundary-row-storage-prepared-v1')
    args = parser.parse_args()
    started = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'CPU preparation only'
    assert os.environ.get('MACA_VISIBLE_DEVICES') in ('', '-1'), 'CPU preparation only'
    assert sha(args.source) == SOURCE_SHA
    assert sha(args.runtime_summary) == SUMMARY_SHA
    prior = json.loads(args.source.read_bytes())
    summary = json.loads(args.runtime_summary.read_bytes())
    assert summary['status'] == 'rank_diagnostics_complete'
    assert prior['checkpoint_restore_requested'] is False and prior['resume_mode'] == 'disable'
    assert sha(prior['prepared_receipt']) == prior['prepared_receipt_sha256']
    previous = json.loads(Path(prior['prepared_receipt']).read_bytes())
    tested = summary['prepared_candidate']
    old_entry, old_dt = Path(previous['entry']), Path(prior['dt_root'])
    dt, verl, loop = Path(tested['dt_root']), Path(prior['verl_root']), Path(prior['loop_root'])
    assert old_dt == Path(tested['current_original_dt_root'])
    assert sha(old_entry/'deltatrace_rollout.py') == PRODUCER_SHA
    for item in (tested['artifact'], tested['lease'], tested['original_artifact'], tested['original_lease'], prior['canonical_HF_owner']):
        assert sha(item['path']) == item['sha256'], item['path']
    assert tested['artifact']['sha256'] == ARTIFACT_SHA and tested['lease']['sha256'] == LEASE_SHA
    for key in ('path', 'sha256'):
        assert tested['canonical_HF_owner'][key] == prior['canonical_HF_owner'][key], key
    for base, hashes in ((old_entry, prior['entry_sha256']), (verl, prior['owner_head_sha256']),
                         (verl, prior['verl_sha256']), (loop, prior['author_sha256']), (old_dt, prior['dt_source_sha256'])):
        for name, expected in hashes.items():
            assert sha(base/name) == expected, (base, name)
    for item in prior['installed_verified_files']:
        assert sha(item['path']) == item['expected_verified_sha256'], item['path']
    dt_hashes = {name: sha(dt/name) for name in prior['dt_source_sha256']}
    changed_dt = [name for name, expected in prior['dt_source_sha256'].items() if dt_hashes[name] != expected]
    assert changed_dt == [ARTIFACT_KEY], changed_dt
    assert dt_hashes[ARTIFACT_KEY] == ARTIFACT_SHA
    environment = Path(prior['resource_environment']['DT_ENVIRONMENT_JSON'])
    assert sha(environment) == prior['candidate_environment']['sha256']
    for key in ('DT_CONV_ISOLATED_IMPORT_ROOT', 'DT_PREFIX_NATIVE_CONV_INITIAL_STATES'):
        assert key not in prior['resource_environment'], 'Keep the canonical HF route without the eager loader'
    assert not args.output.exists() and not args.receipt.exists(), 'Never overwrite a frozen attempt'
    assert not (old_entry/'deltatrace_rollout.py').is_symlink()
    assert not (old_entry/'native_prefix_leases.py').is_symlink()

    args.receipt.mkdir(parents=True)
    entry = args.output/'entry'
    shutil.copytree(old_entry, entry, symlinks=True)
    producer = entry/'deltatrace_rollout.py'
    producer.write_bytes(patch_factory(producer.read_bytes()))
    (entry/'native_prefix_leases.py').write_bytes(Path(tested['lease']['path']).read_bytes())
    entry_hashes = {name: sha(entry/name) for name in prior['entry_sha256']}
    changed_entry = [name for name, expected in prior['entry_sha256'].items() if entry_hashes[name] != expected]
    assert set(changed_entry) == {'deltatrace_rollout.py', 'native_prefix_leases.py'}
    paths = []
    for part in prior['pythonpath'].split(':'):
        if part == str(old_entry):
            part = str(entry)
        elif part == str(old_dt) or part.startswith(str(old_dt) + '/'):
            part = str(dt) + part[len(str(old_dt)):]
        paths.append(part)
    pythonpath = ':'.join(paths)
    assert str(entry) in paths and str(dt/'clean/qwen35') in paths
    assert str(DIAGNOSTIC) not in paths, 'Do not import the diagnostic entry in production'
    resource = dict(prior['resource_environment'], DT_ROOT=str(dt))
    cpu = os.environ.copy()
    cpu.update(resource, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VERL_ROOT=str(verl), DT_ENTRY_ROOT=str(entry), LOOP_ROOT=str(loop), PYTHONPATH=pythonpath)
    for key in ('DT_CONV_ISOLATED_IMPORT_ROOT', 'DT_PREFIX_NATIVE_CONV_INITIAL_STATES'):
        cpu.pop(key, None)
    imported = subprocess.run([sys.executable, '-c', CPU_INSPECTION], env=cpu, cwd=args.receipt,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (args.receipt/'cpu-import.stdout.txt').write_text(imported.stdout)
    (args.receipt/'cpu-import.stderr.txt').write_text(imported.stderr)
    imported.check_returncode()
    identity = json.loads(imported.stdout.splitlines()[-1])
    expected = {'producer': producer, 'lease': entry/'native_prefix_leases.py',
                'artifact': dt/ARTIFACT_KEY, 'runner': dt/'clean/qwen35/qwen35_dense_finite_runner.py',
                'model': Path(prior['canonical_HF_owner']['path'])}
    for label, path in expected.items():
        assert Path(identity['sources'][label]['path']).resolve() == path.resolve(), label
        assert identity['sources'][label]['sha256'] == sha(path), label
    assert identity['cuda_initialized'] is False and identity['distributed_initialized'] is False
    (args.receipt/'cpu-import-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    node = str(entry/'test_owner_entry_launch.py')+'::test_appworld_formal_workload_and_native_validator'
    with (args.receipt/'cpu-validator.log').open('wb') as log:
        checked = subprocess.run([sys.executable, '-m', 'pytest', '-q', node,
                                  '--junitxml='+str(args.receipt/'cpu-validator.xml')],
                                 env=cpu, cwd=args.receipt, stdout=log, stderr=subprocess.STDOUT)
    checked.check_returncode()
    assert sha(args.source) == SOURCE_SHA and sha(environment) == prior['candidate_environment']['sha256']
    for base, hashes in ((old_entry, prior['entry_sha256']), (verl, prior['owner_head_sha256']),
                         (loop, prior['author_sha256']), (dt, dt_hashes)):
        for name, expected_hash in hashes.items():
            assert sha(base/name) == expected_hash, (base, name)
    record = dict(status='prepared_only_not_submitted_not_deployed', prepared_unix=time.time(),
        elapsed_seconds=time.time()-started, preparation_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        base_source=dict(path=str(args.source), sha256=sha(args.source)), previous_preparation=dict(path=prior['prepared_receipt'], sha256=prior['prepared_receipt_sha256']),
        verification_receipt=dict(path=str(args.runtime_summary), sha256=sha(args.runtime_summary)),
        candidate=str(args.output), entry=str(entry), entry_sha256=entry_hashes, changed_entry=changed_entry,
        dt_root=str(dt), dt_source_sha256=dt_hashes, changed_dt=changed_dt,
        boundary_row_storage_candidate=tested, verl_root=str(verl), owner_sha256=prior['owner_head_sha256'],
        owner_head_sha256=prior['owner_head_sha256'], verl_sha256=prior['verl_sha256'], loop_root=str(loop), author_sha256=prior['author_sha256'],
        pythonpath=pythonpath, resource_environment=resource, canonical_HF_owner=prior['canonical_HF_owner'],
        candidate_environment=prior['candidate_environment'], actor_sha256=previous['actor_sha256'], trainer_sha256=previous['trainer_sha256'],
        official_padding_receipt=previous['official_padding_receipt'], official_padding_sha256=previous['official_padding_sha256'],
        native_conv_capacity_receipt=prior['native_conv_capacity_receipt'], checkpoint_resume_mode='disable', checkpoint_restore_requested=False,
        cpu_validator_returncode=checked.returncode, cpu_actual_imports=dict(path=str(args.receipt/'cpu-import-identity.json'), sha256=sha(args.receipt/'cpu-import-identity.json'), details=identity),
        production_import_scope='Existing canonical HF owner and tested linked DT tree; copied original entry binds the verified optional GDN storage interface using functools.partial. No sitecustomize or namespace loader.',
        numerical_scope='Same-capture consumed GDN row bytes and lease mapping verified by the bound B8 receipt. FA/FLA/finite numerical owners, dtype and Q/V/A are unchanged; cross-capture residuals do not define a new tolerance.',
        capacity_scope='The existing native-convolution 32k receipt is retained; this preparation performs no new capacity test and does not claim a new whole-DT numerical acceptance.',
        preservation='VERL whitening/actor/core/optimizer, LOOP async/evaluation/readiness, environment JSON, LoRA8/16, B4, official budgets and original launcher unchanged.',
        zero_operations=dict(model_loads=0, gpu_calls=0, checkpoint_loads=0, training_submissions=0, stopped_jobs=0))
    (args.receipt/'prepared.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key: record[key] for key in ('status', 'entry', 'dt_root', 'changed_entry', 'changed_dt', 'cpu_validator_returncode')}))


if __name__ == '__main__':
    main()
