"""Verify actual owner task streams and native checkpoint orchestration on CPU."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-native-sampler-checkpoint-20261017-v5'
PATCH=json.loads((AUDIT/'owner-patch.json').read_bytes())
if '--native' not in sys.argv:
    source=json.loads((ROOT/'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v4/source.json').read_bytes())
    env=dict(os.environ,**source['effective_environment'])
    for key in ('loop','entry','verl_dt'):
        env={k:v.replace(PATCH['baselines'][key],PATCH['owner_roots'][key]) for k,v in env.items()}
    env['CUDA_VISIBLE_DEVICES']=''
    env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

import site
site.addsitedir(os.environ['LOOP_EXTRAS'])
import torch
from omegaconf import OmegaConf
from phi_agents.rl import appworld_scenario_runner as scenario_owner
from phi_agents.rl import parallel_scenario_sampler as parallel_owner
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

BASES=PATCH['baselines']
ROOTS=PATCH['owner_roots']
assert Path(scenario_owner.__file__).is_relative_to(ROOTS['loop'])
assert Path(parallel_owner.__file__).is_relative_to(ROOTS['loop'])
cases=[]


def remove_methods(tree,names):
    for node in ast.walk(tree):
        if isinstance(node,ast.ClassDef):
            node.body=[n for n in node.body if not(isinstance(n,ast.FunctionDef) and n.name in names)]


def module_pair(base,relative):
    return ast.parse((Path(BASES[base])/relative).read_text()),ast.parse((Path(ROOTS[base])/relative).read_text())


original,patched=module_pair('loop','phi_agents/rl/appworld_scenario_runner.py')
remove_methods(patched,{'state_dict','load_state_dict'})
assert ast.dump(original)==ast.dump(patched)
cases.append(dict(case='original_AppWorld_sampling_and_runner_AST_unchanged',passed=True))

original,patched=module_pair('loop','phi_agents/rl/parallel_scenario_sampler.py')
remove_methods(patched,{'state_dict','load_state_dict'})
patched.body=[n for n in patched.body if not(isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='copy')]
for node in ast.walk(patched):
    if isinstance(node,ast.FunctionDef) and node.name=='__init__':
        node.body=[n for n in node.body if not(isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self._samplers')]
    if isinstance(node,ast.FunctionDef) and node.name=='_worker':
        node.body=[n for n in node.body if not(isinstance(n,ast.With) and len(n.body)==1 and isinstance(n.body[0],ast.Assign) and ast.unparse(n.body[0].targets[0])=='self._samplers[thread_idx]')]
assert ast.dump(original)==ast.dump(patched)
cases.append(dict(case='original_parallel_sampling_AST_unchanged',passed=True))

original,patched=module_pair('entry','loop_owner_worker.py')
cls=next(n for n in patched.body if isinstance(n,ast.ClassDef) and n.name=='OwnerProcesses')
ready=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ensure_ready')
start=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='start')
start.body[:1]=copy.deepcopy(ready.body)
remove_methods(patched,{'ensure_ready','checkpoint_state_dict','load_checkpoint_state_dict'})
function=next(n for n in patched.body if isinstance(n,ast.FunctionDef) and n.name=='run_owner')
for node in ast.walk(function):
    if isinstance(node,ast.While) and ast.unparse(node.test)=='True':
        node.body=[n for n in node.body if not(isinstance(n,ast.If) and any(s in ast.unparse(n.test) for s in ('checkpoint_state','restore_checkpoint_state')))]
assert ast.dump(original)==ast.dump(patched)
cases.append(dict(case='original_collect_start_transport_AST_unchanged',passed=True))

original,patched=module_pair('entry','loop_owner_rollout.py')
remove_methods(patched,{'checkpoint_state_dict','load_checkpoint_state_dict'})
for node in ast.walk(patched):
    if isinstance(node,ast.FunctionDef) and node.name=='__init__':
        node.body=[n for n in node.body if not(isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self._sampler_checkpoint_to_restore')]
    if isinstance(node,ast.FunctionDef) and node.name=='collect_native_trajectories':
        for child in ast.walk(node):
            if isinstance(child,ast.If) and ast.unparse(child.test)=='self.processes is None':
                child.body=[n for n in child.body if not(isinstance(n,ast.If) and '_sampler_checkpoint_to_restore' in ast.unparse(n.test))]
assert ast.dump(original)==ast.dump(patched)
cases.append(dict(case='original_LOOP_trajectory_and_credit_AST_unchanged',passed=True))

for kind in ('verl_dt','verl_ppo'):
    original,patched=module_pair(kind,'verl/trainer/ppo/ray_trainer.py')
    for node in ast.walk(patched):
        if isinstance(node,ast.FunctionDef) and node.name in ('_save_checkpoint','_load_checkpoint'):
            node.body=[n for n in node.body if not(
                isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('environment_state','restore_environment_state')
                or isinstance(n,ast.If) and ast.unparse(n.test) in ('environment_state is not None','restore_environment_state is not None'))]
    assert ast.dump(original)==ast.dump(patched)
    cases.append(dict(case=kind+'_original_trainer_and_PPO_AST_unchanged',passed=True))

dataset='train_difficulty_1_and_train_difficulty_2'
Sampler=scenario_owner.AppWorldScenarioSampler
source=Path(BASES['loop'])/'phi_agents/rl/appworld_scenario_runner.py'
old_cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AppWorldScenarioSampler')
namespace=dict(vars(scenario_owner))
exec(compile(ast.Module(body=[old_cls],type_ignores=[]),str(source),'exec'),namespace)
OriginalSampler=namespace['AppWorldScenarioSampler']
a,b=OriginalSampler(dataset,seed=17),Sampler(dataset,seed=17)
count=2*len(a.all_task_ids)+11
assert [next(a).task_id for _ in range(count)]==[next(b).task_id for _ in range(count)]
cases.append(dict(case='actual_original_task_stream_unchanged_across_reshuffles',passed=True,scenarios=count))

with tempfile.TemporaryDirectory(prefix='appworld-native-sampler-state-') as temporary:
    directory=Path(temporary)
    a=Sampler(dataset)
    for _ in range(13):next(a)
    state=a.state_dict()
    path=directory/'sampler-state-only.pt'
    torch.save(state,path)
    b=Sampler(dataset)
    b.load_state_dict(torch.load(path,weights_only=False))
    assert b.state_dict()==state
    count=2*len(a.all_task_ids)+11
    assert [next(a).task_id for _ in range(count)]==[next(b).task_id for _ in range(count)]
    cases.append(dict(case='actual_seed_None_RNG_and_remaining_order_restore_exactly',passed=True,scenarios=count))

    Parallel=parallel_owner.ParallelScenarioSampler
    for pending in (False,True):
        a=Parallel(lambda:Sampler(dataset),num_threads=1)
        b=Parallel(lambda:Sampler(dataset),num_threads=1)
        try:
            a.state_dict()  # Wait for the original worker to construct its sampler.
            if pending:
                with a._mutex:
                    a.ensure_scenarios_requested(31)
                    state=a.state_dict()
                assert state['n_scenarios_requested']==31
            else:
                a.ensure_scenarios_requested(31)
                for _ in range(7):next(a)
                state=a.state_dict()
            torch.save(state,path)
            b.load_state_dict(torch.load(path,weights_only=False))
            count=2*len(state['samplers'][0]['all_task_ids'])+11
            expected=[next(a).task_id for _ in range(count)]
            actual=[next(b).task_id for _ in range(count)]
            assert expected==actual
            cases.append(dict(case='native_parallel_pending_requests_restore' if pending else 'native_parallel_prefetched_queue_restore',
                              passed=True,queued=len(state['scenarios']),pending=state['n_scenarios_requested'],scenarios=count))
        finally:
            a.stop();b.stop()

    trainer_path=Path(ROOTS['verl_ppo'])/'verl/trainer/ppo/ray_trainer.py'
    methods=[n for n in ast.walk(ast.parse(trainer_path.read_text())) if isinstance(n,ast.FunctionDef) and n.name in ('_save_checkpoint','_load_checkpoint')]
    native=dict(os=os,torch=torch,find_latest_ckpt_path=find_latest_ckpt_path)
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(trainer_path),'exec'),native)
    cfg=OmegaConf.create(dict(trainer=dict(default_local_dir=str(directory/'native-checkpoint-orchestration-only'),
        default_hdfs_dir=None,max_actor_ckpt_to_keep=2,max_critic_ckpt_to_keep=2,resume_mode='resume_path',
        resume_from_path=str(directory/'native-checkpoint-orchestration-only/global_step_2'),del_local_ckpt_after_load=False)))
    root=Path(cfg.trainer.default_local_dir);marker=root/'latest_checkpointed_iteration.txt'
    def save_role(local,remote,step,**options):Path(local).mkdir(parents=True,exist_ok=True)
    actor=Mock();actor.save_checkpoint.side_effect=save_role
    critic=Mock();critic.save_checkpoint.side_effect=save_role
    dataloader=Mock();dataloader.state_dict.return_value={'cursor':2}
    environment=Mock();environment.checkpoint_state_dict.return_value=[state,state]
    fixture=SimpleNamespace(config=cfg,global_steps=2,use_critic=True,actor_rollout_wg=actor,critic_wg=critic,
                            train_dataloader=dataloader,envs=environment)
    native['_save_checkpoint'](fixture)
    assert marker.read_text()=='2'
    assert torch.load(root/'global_step_2/loop_sampler.pt',weights_only=False)==[state,state]
    native['_load_checkpoint'](fixture)
    dataloader.load_state_dict.assert_called_once_with({'cursor':2})
    environment.load_checkpoint_state_dict.assert_called_once_with([state,state])
    cases.append(dict(case='native_save_load_delegates_actual_sampler_state_before_completion_marker',passed=True,
                      scope='CPU state orchestration; model/optimizer delegates stubbed, no formal checkpoint'))

    fixture.global_steps=4
    environment.checkpoint_state_dict.side_effect=RuntimeError('observed sampler export failure fixture')
    try:native['_save_checkpoint'](fixture)
    except RuntimeError:pass
    else:raise AssertionError('Sampler export failure must remain visible')
    assert marker.read_text()=='2'
    cases.append(dict(case='sampler_export_failure_does_not_advance_native_completion_marker',passed=True))
    (root/'global_step_2/loop_sampler.pt').unlink()
    try:native['_load_checkpoint'](fixture)
    except FileNotFoundError:pass
    else:raise AssertionError('Missing actual sampler checkpoint must not silently reset its RNG')
    cases.append(dict(case='missing_sampler_state_does_not_silently_fresh_start',passed=True))

    fixture.global_steps=6
    fixture.envs=object()
    native['_save_checkpoint'](fixture)
    assert marker.read_text()=='6' and not(root/'global_step_6/loop_sampler.pt').exists()
    cases.append(dict(case='original_non_LOOP_checkpoint_path_has_no_extra_artifact',passed=True))

assert not torch.cuda.is_initialized()
record=dict(status='passed',collected=time.time(),cases=cases,model_calls=0,DT_calls=0,optimizer_updates=0,
            real_training_checkpoint_create_calls=0,actual_task_assets_used=True,
            source_bindings=PATCH['changes'],native_CPU_transport_verification='not_run_yet')
(AUDIT/'CPU-sampler-checkpoint-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
