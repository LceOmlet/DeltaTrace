"""CPU-only native AppWorld comparison; fixed actions are a test fixture, not an agent.

No model performance, tokenizer, rollout scheduling or numerical parity claim.
Both environment owners execute real AppWorld code and evaluate real tasks.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('owner-root', 'project-root', 'pristine-project-root', 'asset-root', 'run-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.run_root.mkdir(parents=True, exist_ok=True)
    for name in ('data', 'src'):
        link = a.run_root / name
        if not link.exists():
            link.symlink_to(a.asset_root / name, target_is_directory=True)
    os.environ['APPWORLD_ROOT'] = str(a.run_root)
    sys.path.insert(0, str(a.owner_root))
    sys.path.insert(1, str(a.project_root))
    os.chdir(a.owner_root)

    from loop_owner_recipe import environment_configuration, OWNER_COMMIT
    from phi_agents.appworld.interface import AppWorldInterface, load_task_ids
    from phi_agents.rl.appworld_scenario_runner import AppWorldScenarioRunner, AppWorldScenario
    from phi_agents.rl.type_defs import PolicyMessage, PolicyTokenInfo
    from phi_agents.utils.appworld import extract_code_format_output
    from appworld import load_task_ids as project_load_task_ids

    worker_file = Path('agent_system/environments/env_package/appworld/envs.py')
    projection_file = Path('agent_system/environments/env_package/appworld/projection.py')
    worker_module = source_module('comparison_project_appworld', a.project_root / worker_file)
    projection = source_module('comparison_project_projection', a.project_root / projection_file).appworld_projection
    import ast
    def class_ast(path, name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name))
    worker_pristine = class_ast(a.project_root / worker_file, 'AppWorldWorker') == class_ast(a.pristine_project_root / worker_file, 'AppWorldWorker')
    assert worker_pristine
    cfg = environment_configuration(a.owner_root)
    train_name = cfg['training_task_sampler']['dataset_name']
    task_ids = load_task_ids(train_name)
    report = {
        'scope': __doc__, 'loop_commit': OWNER_COMMIT,
        'loop_environment': cfg, 'project_worker_matches_pristine_ast': worker_pristine,
        'splits': {'loop_train': task_ids, 'loop_eval': load_task_ids(cfg['evaluation_task_sampler']['dataset_name']),
                   'project_train': project_load_task_ids('train'), 'project_eval': project_load_task_ids('test_normal')},
        'cases': [], 'parser_comparison': [],
    }
    for text in ('```python\nprint(2)\n```', '<think>check</think><code>print(2)</code>',
                 '```python\nx = 1\n```\n```python\nprint(x + 1)\n```', 'No code'):
        actions, valid = projection([text])
        report['parser_comparison'].append(dict(response=text, loop=extract_code_format_output(text), project=actions[0], project_valid=valid[0]))
    started = time.monotonic()
    for label, task_id in [('completion', task_ids[0]), ('horizon', task_ids[1])]:
        # Fixtures exercise the unchanged native interaction loop and reward path.
        # Token reads are stubbed because these short fixed outputs do not test token budgets.
        llm = Mock()
        llm.special_tokens = {}
        llm.get_tokens.return_value = ([], None, None)
        llm.get_policy_token_info.return_value = PolicyTokenInfo()
        seen_prompts = []
        def generate(messages):
            seen_prompts.append([m.asdict() for m in messages])
            i = len(seen_prompts)
            code = ('apis.supervisor.complete_task()' if label == 'completion' and i == 3
                    else "fixture_counter = globals().get('fixture_counter', 0) + 1; print(fixture_counter)")
            return PolicyMessage('```python\n' + code + '\n```', prompt_tokens=[],
                generated_tokens=[], generated_token_logprobs=[], stopped_by_max_tokens_limit=False)
        llm.generate.side_effect = generate
        runner = AppWorldScenarioRunner(appworld_config=cfg['training_environment']['appworld_config'])
        executed, outputs, evaluations = [], [], []
        original_execute, original_evaluate = runner.world.execute, runner.world.evaluate
        def execute(code):
            output = original_execute(code)
            executed.append(code)
            outputs.append(output)
            return output
        def evaluate():
            result = original_evaluate()
            evaluations.append(asdict(result))
            return result
        with patch.object(runner.world, 'execute', side_effect=execute), \
             patch.object(runner.world, 'evaluate', side_effect=evaluate):
            try:
                rollout = runner.run(AppWorldScenario(task_id=task_id, dataset_name=train_name), llm)
            finally:
                runner.cleanup()
        service = AppWorldInterface(stdout_to_devnull=True)
        worker = worker_module.AppWorldWorker(worker_id=0, max_interactions=50, port=service.port)
        steps = []
        try:
            instruction, info = worker.reset(task_id)
            for code in executed:
                obs, reward, done, details = worker.step(code)
                steps.append(dict(observation=obs, reward=reward, done=done, info=details))
            ev = worker.env.evaluate()
            project_eval = dict(success=bool(ev.success), num_tests=ev.num_tests, passes=ev.passes, failures=ev.failures)
            at_loop_stop = dict(step=worker.current_step_count, done=steps[-1]['done'])
            if label == 'horizon':
                while not steps[-1]['done']:
                    obs, reward, done, details = worker.step(executed[-1])
                    steps.append(dict(observation=obs, reward=reward, done=done, info=details))
        finally:
            worker.close()
            service.close_server()
        assert outputs == [step['observation'] for step in steps[:len(outputs)]]
        le = evaluations[-1]
        assert le['success'] == project_eval['success']
        assert le['num_tests'] == project_eval['num_tests']
        assert len(le['passes']) == len(project_eval['passes'])
        assert len(le['failures']) == len(project_eval['failures'])
        report['cases'].append(dict(label=label, task_id=task_id, loop_return=rollout.ret,
            loop_steps=len(executed), project_steps=len(steps), project_at_loop_stop=at_loop_stop,
            loop_evaluation=le, project_evaluation=project_eval,
            identical_execution_outputs=True, actions=executed, project_steps_detail=steps,
            loop_prompt_message_counts=[len(p) for p in seen_prompts]))
        print(json.dumps(dict(case=label, loop_steps=len(executed), project_steps=len(steps), loop_return=rollout.ret)), flush=True)
    report['seconds'] = time.monotonic() - started
    report['loop_trainer_imported'] = 'phi_agents.rl.train' in sys.modules
    import torch
    report['cuda_initialized'] = torch.cuda.is_initialized()
    assert not report['loop_trainer_imported'] and not report['cuda_initialized']
    report['passed'] = True
    a.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(dict(output=str(a.output), seconds=report['seconds'], passed=True)), flush=True)


if __name__ == '__main__':
    main()
