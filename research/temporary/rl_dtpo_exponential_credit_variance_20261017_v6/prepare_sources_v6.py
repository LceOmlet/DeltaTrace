"""Prepare isolated v6 method and native host-memory candidates, without launch."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
SOURCE = AUDIT / 'source'
RESOURCE_VERSION = 'formal-native-host-memory-20261017-v1'
MODE = 'reward_target_exponential_variance_v6'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

spec = importlib.util.spec_from_file_location('accepted_DT_patch_seam', SOURCE / 'patch_verl_agent2.py')
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

SPECS = {
    'TextCraft_DTPO': ('textcraft-dt-reward-target-exponential-20261016-v5',
                      'textcraft-dt-reward-target-exponential-variance-20261017-v6', True),
    'AppWorld_DTPO': ('appworld-dtpo-reward-target-exponential-formal-20261017-v5',
                     'appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6', True),
    'TextCraft_PPO': ('textcraft-ppo-control-fixed-verl-fresh-20261014-v2',
                     'textcraft-PPO-host-memory-recovery-20261017-v3', False),
    'AppWorld_PPO': ('appworld-native-PPO-formal-20261017-v3',
                    'appworld-native-PPO-host-memory-formal-20261017-v4', False),
}


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def resource_patch(verl, *, dt, appworld):
    changes = {}
    helper = verl / 'verl/utils/native_host_memory.py'
    shutil.copyfile(SOURCE / 'native_host_memory.py', helper)
    changes[str(helper)] = dict(before_SHA256=None, after_SHA256=sha(helper))
    for relative in ('verl/trainer/main_ppo.py', 'verl/trainer/ppo/ray_trainer.py',
                     *(['verl/workers/critic/dp_critic.py'] if appworld and not dt else [])):
        path = verl / relative
        before = path.read_text()
        text = before
        if relative.endswith('main_ppo.py'):
            matches = list(re.finditer(r'(?m)^([ \t]*)trainer\.init_workers\(\)$', text))
            assert len(matches) == 1
            match = matches[0]
            indent = match.group(1)
            text = text[:match.start()] + (indent + 'from verl.utils.native_host_memory import host_memory_lease\n'
                + indent + 'with host_memory_lease("native_init_workers"):\n'
                + indent + '    trainer.init_workers()') + text[match.end():]
        elif relative.endswith('ray_trainer.py'):
            text = replace_once(text, 'import verl.utils.torch_functional as verl_F\n',
                'import verl.utils.torch_functional as verl_F\n'
                'from verl.utils.native_host_memory import host_memory_lease\n')
            for phase in ('update_critic', 'update_actor', 'save_checkpoint'):
                old = f'with _timer("{phase}", timing_raw):'
                if old in text:
                    text = replace_once(text, old, old[:-1] + f', host_memory_lease("{phase}"):')
            matches = list(re.finditer(r'(?m)^([ \t]*)self\._load_checkpoint\(\)$', text))
            assert len(matches) == 1
            match = matches[0]
            indent = match.group(1)
            text = text[:match.start()] + (indent + 'with host_memory_lease("native_checkpoint_restore"):\n'
                + indent + '    self._load_checkpoint()') + text[match.end():]
            if dt:
                # The DT boundary and its original collective RPC remain intact.
                # Place one host lease around the driver block, never around a rank.
                anchor = '                        if self.config.algorithm.adv_estimator == AdvantageEstimator.DELTATRACE:\n'
                start = text.index(anchor) + len(anchor)
                end = text.index('                        batch = compute_advantage(', start)
                body = text[start:end]
                text = text[:start] + '                            with host_memory_lease("DT_advantage"):\n' + ''.join(
                    '    ' + line if line.strip() else line for line in body.splitlines(True)) + text[end:]
        else:
            matches = list(re.finditer(r'(?m)^([ \t]*)loss\.backward\(\)$', text))
            assert len(matches) == 1
            match = matches[0]
            indent = match.group(1)
            text = text[:match.end()] + ('\n' + indent +
                'from verl.utils.native_host_memory import release_unused_host_cache\n' + indent +
                'release_unused_host_cache("critic_completed_B4_backward")') + text[match.end():]
        ast.parse(text)
        path.write_text(text, newline='\n')
        changes[str(path)] = dict(before_SHA256=hashlib.sha256(before.encode()).hexdigest(),
                                  after_SHA256=sha(path))
    return changes


def prepare(key, old_version, version, dt):
    old_run = ROOT / 'runs' / old_version
    launch = json.loads((old_run / 'launch.json').read_bytes())
    source = json.loads((old_run / 'source.json').read_bytes())
    old_env = launch['effective_environment']
    destination = ROOT / 'candidates' / version
    assert not destination.exists(), 'Preserve existing candidate and receipts'
    destination.mkdir()
    entry = destination / 'entry'
    old_entry = Path(old_env.get('DT_ENTRY_ROOT', old_env['PYTHONPATH'].split(':')[0]))
    shutil.copytree(old_entry, entry, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    verl = destination / 'verl-owner'
    shutil.copytree(old_env['VERL_ROOT'], verl, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
    if dt:
        for name in ('dt_training_batch.py', 'reward_readout.py', 'textcraft_owner_rollout.py'):
            shutil.copyfile(SOURCE / name, entry / name)
        trainer = verl / 'verl/trainer/ppo/ray_trainer.py'
        trainer.write_text(patch.patch_dt_advantage_preprocessing(trainer.read_text()), newline='\n')
    changes = resource_patch(verl, dt=dt, appworld=key.startswith('AppWorld'))
    env = dict(old_env)
    env['PYTHONPATH'] = env['PYTHONPATH'].replace(str(old_entry), str(entry)).replace(old_env['VERL_ROOT'], str(verl))
    env.update(DT_ENTRY_ROOT=str(entry), VERL_ROOT=str(verl),
               DT_NATIVE_HOST_MEMORY_LEASE='/tmp/deltatrace-four-formal-host-memory-20261017-v1.lock',
               DT_NATIVE_HOST_CACHE_BATCH='1')
    if dt:
        env['DT_CREDIT_MODE'] = MODE
    bindings = {}
    inherited = source.get('source_bindings', source.get('actual_inherited_sources', {}))
    for path, digest in inherited.items():
        actual = str(path).replace(str(old_entry), str(entry)).replace(old_env['VERL_ROOT'], str(verl))
        if Path(actual).is_file():
            bindings[actual] = sha(actual)
    for path in entry.glob('*.py'):
        bindings[str(path)] = sha(path)
    for path in changes:
        bindings[path] = sha(path)
    bindings[str(verl / 'verl/trainer/ppo/ray_trainer.py')] = sha(verl / 'verl/trainer/ppo/ray_trainer.py')
    # Numerical DT, PPO actor/core, checkpoint owner and offload math remain
    # the actual accepted sources. Only the declared resource call is added to
    # the AppWorld critic; its arithmetic is checked separately against baseline.
    preserved = {}
    for relative in ('verl/workers/actor/dp_actor.py', 'verl/trainer/ppo/core_algos.py',
                     'verl/utils/checkpoint/fsdp_checkpoint_manager.py',
                     'verl/utils/activation_offload.py', 'verl/utils/experimental/torch_functional.py'):
        path = verl / relative
        original = Path(old_env['VERL_ROOT']) / relative
        assert sha(path) == sha(original), relative
        preserved[relative] = sha(path)
        bindings[str(path)] = sha(path)
    record = dict(status='sources_prepared_not_tested_not_deployed', key=key, version=version,
        parent_run=str(old_run), parent_launch=str(old_run/'launch.json'),
        parent_launch_SHA256=sha(old_run/'launch.json'), entry=str(entry), verl_root=str(verl),
        effective_environment=env, source_bindings=bindings, preserved_numerical_owners=preserved,
        host_memory_resource_version=RESOURCE_VERSION, host_memory_changes=changes,
        method_credit_mode=env.get('DT_CREDIT_MODE'), method_changed=dt,
        numeric_training_hyperparameters_changed=False, model_calls=0, optimizer_updates=0)
    (destination / 'source-prepared.json').write_text(json.dumps(record, indent=2)+'\n')
    (AUDIT / (key+'-source-prepared.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(key=key, version=version, candidate=str(destination), status=record['status'])))


def repair_prepared_resource_indent():
    """Preserve the failed candidate and repair only the undeployed resource seam."""
    for key, (old_version, version, dt) in SPECS.items():
        destination = ROOT / 'candidates' / version
        path = destination / 'source-prepared.json'
        record = json.loads(path.read_bytes())
        assert record['status'] == 'sources_prepared_not_tested_not_deployed'
        original_env = json.loads((ROOT / 'runs' / old_version / 'launch.json').read_bytes())['effective_environment']
        baseline = Path(original_env['VERL_ROOT'])
        verl = destination / 'verl-owner'
        preserved = AUDIT / 'undeployed-resource-indentation-failure-v1' / key
        preserved.mkdir(parents=True)
        shutil.copyfile(path, preserved / 'source-prepared.json')
        for changed in record['host_memory_changes']:
            current = Path(changed)
            if current.name == 'native_host_memory.py':
                continue
            relative = current.relative_to(verl)
            archive = preserved / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, archive)
            text = (baseline / relative).read_text()
            if dt and current.name == 'ray_trainer.py':
                text = patch.patch_dt_advantage_preprocessing(text)
            current.write_text(text, newline='\n')
        record['host_memory_changes'] = resource_patch(verl, dt=dt, appworld=key.startswith('AppWorld'))
        for current in record['source_bindings']:
            if Path(current).is_file():
                record['source_bindings'][current] = sha(current)
        record['prepared_resource_revision'] = 2
        record['undeployed_failed_preparation_preserved'] = str(preserved)
        for current in (path, AUDIT / (key + '-source-prepared.json')):
            current.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(dict(key=key, status='prepared_resource_indent_repaired_not_deployed')))


if __name__ == '__main__':
    if '--repair-resource-indent' in sys.argv:
        repair_prepared_resource_indent()
    else:
        for key, values in SPECS.items():
            prepare(key, *values)
