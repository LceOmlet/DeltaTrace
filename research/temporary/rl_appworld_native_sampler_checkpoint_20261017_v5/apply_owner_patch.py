"""Add native sampler state APIs and a default-inert VERL checkpoint seam."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='appworld-native-sampler-checkpoint-20261017-v5'
DEST=ROOT/'candidates'/VERSION
AUDIT=ROOT/'receipts'/VERSION
BASE_LOOP=ROOT/'candidates/appworld-environment-readiness-20261017-v4/loop-owner'
BASE_ENTRY=ROOT/'candidates/appworld-dtpo-reward-target-exponential-formal-20261017-v4/entry'
BASE_DT=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
BASE_PPO=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
assert not DEST.exists()
changes={}


def copy_owner(source,target):
    target.mkdir(parents=True)
    for item in source.iterdir():
        output=target/item.name
        if item.name=='data':output.symlink_to(item,target_is_directory=True)
        elif item.is_dir():shutil.copytree(item,output,ignore=shutil.ignore_patterns('__pycache__','.git','.pytest_cache'))
        else:shutil.copy2(item,output)


def add_methods(code,class_name,methods):
    cls=next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    lines=code.splitlines(keepends=True)
    lines.insert(cls.end_lineno,'\n'+textwrap.indent(textwrap.dedent(methods).strip()+'\n','    '))
    return ''.join(lines)


def save_change(relative,old,new,expected):
    assert hashlib.sha256(old.encode()).hexdigest()==expected,relative
    ast.parse(new)
    (DEST/relative).write_text(new)
    changes[relative]=dict(before_SHA256=expected,after_SHA256=hashlib.sha256(new.encode()).hexdigest())


copy_owner(BASE_LOOP,DEST/'loop-owner')
copy_owner(BASE_ENTRY,DEST/'entry')
copy_owner(BASE_DT,DEST/'verl-dt-owner')
copy_owner(BASE_PPO,DEST/'verl-ppo-owner')

relative='loop-owner/phi_agents/rl/appworld_scenario_runner.py'
old=(BASE_LOOP/'phi_agents/rl/appworld_scenario_runner.py').read_text()
new=add_methods(old,'AppWorldScenarioSampler','''
def state_dict(self) -> dict:
    """Export the actual NumPy generator and unconsumed shuffled task order."""
    return dict(dataset_name=self.dataset_name, all_task_ids=copy.copy(self.all_task_ids),
                task_ids=copy.copy(self.task_ids), rng=copy.deepcopy(self.rng.bit_generator.state))

def load_state_dict(self, state: dict) -> None:
    if state['dataset_name'] != self.dataset_name or state['all_task_ids'] != self.all_task_ids:
        raise ValueError('AppWorld sampler checkpoint refers to different task assets')
    self.rng.bit_generator.state = copy.deepcopy(state['rng'])
    self.task_ids = copy.copy(state['task_ids'])
''')
save_change(relative,old,new,'df8481f10f3bda5737160d159763189867ce73ea623a4b464cf899f2d9469eac')

relative='loop-owner/phi_agents/rl/parallel_scenario_sampler.py'
old=(BASE_LOOP/'phi_agents/rl/parallel_scenario_sampler.py').read_text()
new=old.replace('import queue\n','import copy\nimport queue\n',1)
anchor='        self._create_sampler_func = create_sampler_func\n'
assert new.count(anchor)==1
new=new.replace(anchor,anchor+'        self._samplers = {}\n')
anchor='        sampler = self._create_sampler_func()\n'
assert new.count(anchor)==1
new=new.replace(anchor,anchor+'        with self._mutex:\n            self._samplers[thread_idx] = sampler\n')
new=add_methods(new,'ParallelScenarioSampler','''
def state_dict(self) -> dict:
    """Snapshot between rollout batches, including already-prefetched scenarios."""
    with self._mutex:
        while self._n_scenarios_in_progress or len(self._samplers) != len(self._threads):
            self._generation_requested.wait(timeout=1.0)
        with self._queue.mutex:
            return copy.deepcopy(dict(
                samplers=[self._samplers[i].state_dict() for i in range(len(self._threads))],
                scenarios=list(self._queue.queue),
                n_scenarios_requested=self._n_scenarios_requested))

def load_state_dict(self, state: dict) -> None:
    """Restore the same native sampler and prefetch queue before collecting."""
    with self._mutex:
        while self._n_scenarios_in_progress or len(self._samplers) != len(self._threads):
            self._generation_requested.wait(timeout=1.0)
        if len(state['samplers']) != len(self._threads):
            raise ValueError('Sampler checkpoint thread count differs')
        for i, sampler_state in enumerate(state['samplers']):
            self._samplers[i].load_state_dict(sampler_state)
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.queue.extend(copy.deepcopy(state['scenarios']))
        self._n_scenarios_requested = state['n_scenarios_requested']
        self._generation_requested.notify_all()
''')
save_change(relative,old,new,'40a97712e31afcd556e0402ff61e8854653a2bf90242acd5619b61e6b694da97')

relative='entry/loop_owner_worker.py'
old=(BASE_ENTRY/'loop_owner_worker.py').read_text()
anchor="            if command == 'stop':\n                break\n"
assert old.count(anchor)==1
new=old.replace(anchor,anchor+'''
            if command == 'checkpoint_state':
                output.put(('checkpoint_state', rank, sampler.state_dict()))
                continue
            if isinstance(command, tuple) and command[0] == 'restore_checkpoint_state':
                sampler.load_state_dict(command[1])
                output.put(('checkpoint_restored', rank))
                continue
''')
anchor='    def start(self):\n        if not self.ready:'
assert new.count(anchor)==1
new=new.replace(anchor,'    def ensure_ready(self):\n        if not self.ready:',1)
anchor="            self.ready = True\n        for command in self.commands:\n            command.put('collect')"
assert new.count(anchor)==1
new=new.replace(anchor,"            self.ready = True\n\n    def start(self):\n        self.ensure_ready()\n        for command in self.commands:\n            command.put('collect')",1)
new=add_methods(new,'OwnerProcesses','''
def checkpoint_state_dict(self):
    self.ensure_ready()
    for command in self.commands:
        command.put('checkpoint_state')
    states = {}
    while len(states) != len(self.processes):
        event = self.event()
        if event[0] == 'engine_ready':
            continue
        assert event[0] == 'checkpoint_state' and event[1] not in states, event[0]
        states[event[1]] = event[2]
    return [states[i] for i in range(len(self.processes))]

def load_checkpoint_state_dict(self, states):
    self.ensure_ready()
    if len(states) != len(self.processes):
        raise ValueError('Sampler checkpoint rank count differs')
    for command, state in zip(self.commands, states):
        command.put(('restore_checkpoint_state', state))
    restored = set()
    while len(restored) != len(self.processes):
        event = self.event()
        if event[0] == 'engine_ready':
            continue
        assert event[0] == 'checkpoint_restored' and event[1] not in restored, event[0]
        restored.add(event[1])
''')
save_change(relative,old,new,'15aa487af56c42646ad8e617420c889f4a00ce64be91ed7a463fc07a0266fdb6')

relative='entry/loop_owner_rollout.py'
old=(BASE_ENTRY/'loop_owner_rollout.py').read_text()
anchor='        self.processes = None\n'
assert old.count(anchor)==2
new=old.replace(anchor,anchor+'        self._sampler_checkpoint_to_restore = None\n',1)
anchor='                inference_clients)\n'
assert new.count(anchor)==1
new=new.replace(anchor,anchor+'''            if self._sampler_checkpoint_to_restore is not None:
                self.processes.load_checkpoint_state_dict(self._sampler_checkpoint_to_restore)
                self._sampler_checkpoint_to_restore = None
''')
new=add_methods(new,'LoopOwner','''
def checkpoint_state_dict(self):
    if self.processes is None:
        raise RuntimeError('No actual LOOP training sampler exists to checkpoint')
    return self.processes.checkpoint_state_dict()

def load_checkpoint_state_dict(self, state):
    if self.processes is None:
        self._sampler_checkpoint_to_restore = state
    else:
        self.processes.load_checkpoint_state_dict(state)
''')
save_change(relative,old,new,'20addfea31e7dd052bdde0eb13b6a457936d539db7200ce928d766f9c5eee51f')

for kind,base in (('dt',BASE_DT),('ppo',BASE_PPO)):
    relative=f'verl-{kind}-owner/verl/trainer/ppo/ray_trainer.py'
    old=(base/'verl/trainer/ppo/ray_trainer.py').read_text()
    anchor='        # latest checkpointed iteration tracker (for atomic usage)\n'
    assert old.count(anchor)==1
    new=old.replace(anchor,'''        environment_state = getattr(self.envs, "checkpoint_state_dict", None)
        if environment_state is not None:
            torch.save(environment_state(), os.path.join(local_global_step_folder, "loop_sampler.pt"))

'''+anchor)
    tree=ast.parse(new)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_load_checkpoint')
    lines=new.splitlines(keepends=True)
    lines.insert(method.end_lineno,'''        restore_environment_state = getattr(self.envs, "load_checkpoint_state_dict", None)
        if restore_environment_state is not None:
            environment_state_path = os.path.join(global_step_folder, "loop_sampler.pt")
            restore_environment_state(torch.load(environment_state_path, weights_only=False))
''')
    new=''.join(lines)
    save_change(relative,old,new,'b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943')

AUDIT.mkdir(parents=True,exist_ok=True)
record=dict(status='prepared_only_state_and_default_path_verification_pending',version=VERSION,prepared=time.time(),
            owner_roots=dict(loop=str(DEST/'loop-owner'),entry=str(DEST/'entry'),
                             verl_dt=str(DEST/'verl-dt-owner'),verl_ppo=str(DEST/'verl-ppo-owner')),
            baselines=dict(loop=str(BASE_LOOP),entry=str(BASE_ENTRY),verl_dt=str(BASE_DT),verl_ppo=str(BASE_PPO)),
            changes=changes,task_sampler_algorithm_changed=False,PPO_or_DT_math_changed=False,
            checkpoints='Original VERL save/load and completion marker; actual LOOP sampler sidecar saved before marker',
            model_calls=0,optimizer_updates=0,formal_training_changes=0,
            gap_receipt=str(ROOT/'receipts/appworld-formal-training-20261017-v1/AppWorld-native-sampler-checkpoint-state-gap.json'))
(AUDIT/'owner-patch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
