"""Prepare the accepted cached-convolution owners; never submit a job.

Run with the existing environment's VENV_PYTHON and CUDA/MACA hidden. The
linked owner tree is built by the already exercised isolation preparer.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
SOURCE_SHA = 'dbb3dbf887293680757a44085fdc1c31d310c7b8731bea21c7e321149f2342d5'
PRODUCER_SHA = '5d0f897e47fbf805321b50ebda6abca17dfdf16f6ee86f772a52f17550f7a6ee'
ISOLATION_SHA = 'e73350381a2e0f3683f1e3b93899dd3a2127e636bd7904f3834500496c7eab09'
TRAINER_SHA = 'd35ddd26b7497b153cec22f22e92eb1a08ef70385428dcbdcc37e879d9f16a4f'
ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
DIAGNOSTIC = ROOT/'receipts/owner-b8-dispatch-20260930/native-prefix-reuse-native-conv-initial-states-offset84-20261007-c1a079f-current-base'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_producer(raw):
    """One call-keyword splice; no change to the FLA execution options."""
    text = raw.decode('utf-8')
    original = ast.parse(text)
    cls = next(n for n in original.body if isinstance(n, ast.ClassDef)
               and n.name == 'DeltaTraceRolloutProducer')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'make_qwen35_runner']
    assert len(calls) == 1, 'Expected the actual original runner constructor call'
    call = calls[0]
    keywords = [n for n in call.keywords if n.arg is None
                and isinstance(n.value, ast.Name) and n.value.id == 'execution']
    assert len(keywords) == 1
    keyword = keywords[0]
    lines = raw.splitlines(keepends=True)
    line = lines[keyword.lineno-1]
    assert line.strip() == b'**execution,'
    indentation = line[:len(line)-len(line.lstrip())]
    ending = b'\r\n' if line.endswith(b'\r\n') else b'\n'
    addition = indentation + (
        b"**({'native_conv_initial_states': True} if "
        b"env.get('dt_native_conv_initial_states', False) else {}),"
    ) + ending
    offset = sum(map(len, lines[:keyword.lineno-1]))
    patched = raw[:offset] + addition + raw[offset:]
    candidate = ast.parse(patched.decode('utf-8'))
    candidate_cls = next(n for n in candidate.body if isinstance(n, ast.ClassDef)
                         and n.name == cls.name)
    candidate_init = next(n for n in candidate_cls.body if isinstance(n, ast.FunctionDef)
                          and n.name == init.name)
    candidate_call = next(n for n in ast.walk(candidate_init) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == 'make_qwen35_runner')
    added = [n for n in candidate_call.keywords if n.arg is None and isinstance(n.value, ast.IfExp)]
    assert len(added) == 1
    candidate_call.keywords.remove(added[0])
    assert ast.dump(candidate, include_attributes=False) == ast.dump(original, include_attributes=False)
    for enabled in (False, True):
        expression = ast.Expression(added[0].value)
        value = eval(compile(expression, '<original-runner-optional-keyword>', 'eval'),
                     {}, {'env': {'dt_native_conv_initial_states': enabled}})
        assert value == ({'native_conv_initial_states': True} if enabled else {})
    compile(patched, '<prepared-original-producer>', 'exec')
    return patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/appworld-fresh-official-padding-20261007-v2/appworld-dt/source.json')
    parser.add_argument('--candidate-sources', type=Path, default=DIAGNOSTIC/'candidate_sources')
    parser.add_argument('--isolation-helper', type=Path, default=DIAGNOSTIC/'prepare_isolated_owner_paths.py')
    parser.add_argument('--output', type=Path, default=ROOT/'candidates/appworld-native-conv-initial-states-20261007-v1')
    parser.add_argument('--receipt', type=Path, default=ROOT/'receipts/appworld-efficiency-20261007/native-conv-production-prepared-v1')
    args = parser.parse_args()
    started = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'), 'CPU preparation only'
    assert os.environ.get('MACA_VISIBLE_DEVICES') in ('', '-1'), 'CPU preparation only'
    assert sha(args.source) == SOURCE_SHA
    prior = json.loads(args.source.read_bytes())
    assert prior['checkpoint_restore_requested'] is False and prior['resume_mode'] == 'disable'
    assert sha(args.isolation_helper) == ISOLATION_SHA
    assert not args.output.exists() and not args.receipt.exists(), 'Never overwrite a frozen attempt'
    prepared = json.loads(Path(prior['prepared_receipt']).read_bytes())
    assert sha(prior['prepared_receipt']) == prior['prepared_receipt_sha256']
    old_entry = Path(prepared['entry'])
    verl = Path(prior['verl_root'])
    old_dt = Path(prior['dt_root'])
    loop = Path(prior['loop_root'])
    for base, hashes in ((old_entry, prior['entry_sha256']), (verl, prior['owner_head_sha256']),
                         (loop, prior['author_sha256'])):
        for name, expected in hashes.items():
            assert sha(base/name) == expected, (base, name)
    assert sha(old_entry/'deltatrace_rollout.py') == PRODUCER_SHA
    assert sha(verl/'verl/trainer/ppo/ray_trainer.py') == TRAINER_SHA
    assert sha(verl/'verl/workers/actor/dp_actor.py') == ACTOR_SHA
    for name, expected in prior['dt_source_sha256'].items():
        assert sha(old_dt/name) == expected, name
    for item in prior['installed_verified_files']:
        assert sha(item['path']) == item['expected_verified_sha256'], item['path']

    args.receipt.mkdir(parents=True)
    entry = args.output/'entry'
    shutil.copytree(old_entry, entry, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    producer = entry/'deltatrace_rollout.py'
    before = producer.read_bytes()
    producer.write_bytes(patch_producer(before))
    helper_spec = importlib.util.find_spec('transformers')
    hf = Path(helper_spec.origin).parent/'models/qwen3_5/modeling_qwen3_5.py'
    isolation = args.output/'isolated-owners'
    subprocess.run([sys.executable, str(args.isolation_helper), '--dt-root', str(old_dt),
                    '--hf-model', str(hf), '--candidate-sources', str(args.candidate_sources),
                    '--output', str(isolation)], check=True)
    owner_paths = json.loads((isolation/'isolated-owner-paths.json').read_bytes())
    dt = Path(owner_paths['isolated_dt_root'])
    original_environment = Path(prior['resource_environment']['DT_ENVIRONMENT_JSON'])
    environment = json.loads(original_environment.read_bytes())
    new_environment = copy.deepcopy(environment)
    assert not new_environment['qwen35'].get('dt_native_conv_initial_states', False)
    new_environment['qwen35']['dt_native_conv_initial_states'] = True
    projection = copy.deepcopy(new_environment)
    projection['qwen35'].pop('dt_native_conv_initial_states')
    old_projection = copy.deepcopy(environment)
    old_projection['qwen35'].pop('dt_native_conv_initial_states', None)
    assert projection == old_projection
    environment_path = args.output/'environment.json'
    environment_path.write_text(json.dumps(new_environment, indent=2)+'\n')
    resource = dict(prior['resource_environment'])
    resource.update(owner_paths['env'], DT_ENVIRONMENT_JSON=str(environment_path))
    old_paths = prior['pythonpath'].split(':')
    paths = [str(entry) if p == str(old_entry) else str(dt) if p == str(old_dt)
             else str(dt/'experiments/rl') if p == str(old_dt/'experiments/rl') else p for p in old_paths]
    pythonpath = ':'.join([str(isolation), str(dt/'clean/qwen35'), *paths])
    cpu = os.environ.copy()
    cpu.update(resource, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VERL_ROOT=str(verl), DT_ENTRY_ROOT=str(entry),
               LOOP_ROOT=str(loop), PYTHONPATH=pythonpath)
    # site.py catches a failed sitecustomize import; check actual canonical
    # imported sources in a fresh CPU process rather than treating its guard
    # as evidence that the path composition took effect.
    inspection = '''import hashlib,inspect,json,torch
from pathlib import Path
import deltatrace_rollout
import qwen35_dense_finite_runner as runner
import qwen35_gdn_finite as finite
from transformers.models.qwen3_5 import modeling_qwen3_5 as model
sources={}
for label,module in [('producer',deltatrace_rollout),('runner',runner),('finite',finite),('model',model)]:
 path=Path(inspect.getsourcefile(module));sources[label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({'sources':sources,'runner_signature':str(inspect.signature(runner.Qwen35DenseFiniteRunner)),
 'cuda_initialized':torch.cuda.is_initialized(),'distributed_initialized':torch.distributed.is_initialized()}))
'''
    imported = subprocess.run([sys.executable, '-c', inspection], env=cpu, cwd=args.receipt,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (args.receipt/'cpu-import.stdout.txt').write_text(imported.stdout)
    (args.receipt/'cpu-import.stderr.txt').write_text(imported.stderr)
    imported.check_returncode()
    actual_imports = json.loads(imported.stdout.splitlines()[-1])
    expected_imports = {'producer': producer, 'runner': dt/'clean/qwen35/qwen35_dense_finite_runner.py',
                        'finite': dt/'clean/qwen35/qwen35_gdn_finite.py',
                        'model': Path(owner_paths['hf_candidate']['path'])}
    for label, path in expected_imports.items():
        assert Path(actual_imports['sources'][label]['path']).resolve() == path.resolve(), label
        assert actual_imports['sources'][label]['sha256'] == sha(path), label
    assert actual_imports['cuda_initialized'] is False
    assert actual_imports['distributed_initialized'] is False
    (args.receipt/'cpu-import-identity.json').write_text(json.dumps(actual_imports, indent=2)+'\n')
    node = str(entry/'test_owner_entry_launch.py')+'::test_appworld_formal_workload_and_native_validator'
    with (args.receipt/'cpu-validator.log').open('wb') as log:
        checked = subprocess.run([sys.executable, '-m', 'pytest', '-q', node,
                                  '--junitxml='+str(args.receipt/'cpu-validator.xml')],
                                 env=cpu, cwd=args.receipt, stdout=log, stderr=subprocess.STDOUT)
    checked.check_returncode()
    entry_hashes = {name: sha(entry/name) for name in prior['entry_sha256']}
    changed = [name for name, expected in prior['entry_sha256'].items() if entry_hashes[name] != expected]
    assert changed == ['deltatrace_rollout.py']
    for name, expected in prior['owner_head_sha256'].items():
        assert sha(verl/name) == expected, name
    assert sha(hf) == owner_paths['originals'][str(hf.resolve())]
    assert sha(old_entry/'deltatrace_rollout.py') == PRODUCER_SHA
    record = dict(status='prepared_only_not_submitted_not_deployed', prepared_unix=time.time(),
        elapsed_seconds=time.time()-started, preparation_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        base_source=dict(path=str(args.source), sha256=sha(args.source)), candidate=str(args.output),
        entry=str(entry), entry_sha256=entry_hashes, verl_root=str(verl), owner_sha256=prepared['owner_sha256'],
        owner_head_sha256=prior['owner_head_sha256'], dt_root=str(dt), loop_root=str(loop),
        author_sha256=prior['author_sha256'], pythonpath=pythonpath, resource_environment=resource,
        actor_sha256=ACTOR_SHA, trainer_sha256=TRAINER_SHA, changed_entry=changed,
        changed_owner=[], checkpoint_resume_mode='disable', checkpoint_restore_requested=False,
        official_padding_receipt=prepared['official_padding_receipt'], official_padding_sha256=prepared['official_padding_sha256'],
        cpu_validator_returncode=checked.returncode,
        cpu_actual_imports=dict(path=str(args.receipt/'cpu-import-identity.json'), sha256=sha(args.receipt/'cpu-import-identity.json'), details=actual_imports),
        isolated_owners=dict(receipt=str(isolation/'isolated-owner-paths.json'), sha256=sha(isolation/'isolated-owner-paths.json'), details=owner_paths),
        original_environment=dict(path=str(original_environment), sha256=sha(original_environment)),
        candidate_environment=dict(path=str(environment_path), sha256=sha(environment_path), added_qwen35_key='dt_native_conv_initial_states', other_fields_equal=True),
        production_import_scope='Canonical HF owner module with the already measured minimal cached-convolution branch patch; runner option applies only its native root/replay scopes. No instance forward wrapper, duplicate model or replacement factory; installed and formal owner files are not overwritten.',
        cpu_scope='Original AppWorld launcher/config validator and default-inert full-module AST comparison only; no model, DT, optimizer or new numerical acceptance.',
        zero_operations=dict(model_loads=0, gpu_calls=0, training_submissions=0, checkpoint_loads=0),
        capacity_scope='Prepared only. This preparation does not bind or claim the pending 32k capacity result or authorization to deploy.',
        preservation='Original VERL white trainer/actor/core/worker, LOOP workload/eval, caches/model paths, LoRA8/16, B4, official budgets and PPO settings.')
    (args.receipt/'prepared.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','candidate','entry','verl_root','dt_root','loop_root','cpu_validator_returncode')}))


if __name__ == '__main__':
    main()
