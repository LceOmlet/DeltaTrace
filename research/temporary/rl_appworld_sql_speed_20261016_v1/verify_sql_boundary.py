"""Reuse the native SQLite/collector owner test and check the new target boundary."""
import json
from pathlib import Path
import subprocess
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-sql-native-speed-20261016-v1'
record=json.loads((ROOT/'candidates/appworld-sql-native-speed-20261016-v1/SQL/prepared.json').read_bytes())
entry=Path(record['entry']);source=(entry/'test_owner_trajectory_batch.py').read_text()
if (AUDIT/'SQL-boundary-tests.xml').exists():
    failures=AUDIT/'failed-CPU-boundary-fixture-r1';failures.mkdir(exist_ok=True)
    for name in ('SQL-boundary-tests.xml','SQL-boundary-tests.stdout.log','SQL-boundary-tests.stderr.log'):
        if (AUDIT/name).exists() and not (failures/name).exists():shutil.copyfile(AUDIT/name,failures/name)
old_group='group = SimpleNamespace(world_size=2, generate_sequences=rollout.generate_sequences)'
assert source.count(old_group)==1
source=source.replace(old_group,'group = SimpleNamespace(world_size=2, generate_sequences=rollout.generate_sequences, begin_rollout_context=lambda *a,**k:None, end_rollout_context=lambda *a,**k:None)')
baseline=AUDIT/'source/test_SQL_original_native_transport.py';baseline.write_text(source)
old="@pytest.mark.parametrize('method', ['dt', 'grpo'])"
assert source.count(old)==1
source=source.replace(old,"@pytest.mark.parametrize('terminal,expected_reward', [('<think>Answer.</think><solution>select name from items</solution>',1.),('<think>Answer.</think><solution>select name from items where 0</solution>',0.),('<think>Answer.</think><solution><sql>invalid</sql></solution>',-1.)])")
source=source.replace('def test_native_trajectory_is_one_training_row_and_observations_are_masked(',
                      'def test_SQL_settlement_target_preserves_native_actions_and_rewards(')
source=source.replace('tokenizer, task, tmp_path, monkeypatch, method):',
                      "tokenizer, task, tmp_path, monkeypatch, terminal, expected_reward):\n    method='dt'")
source=source.replace("        '<think>Answer.</think><solution>select name from items</solution>',","        terminal,")
source=source.replace('assert all(float(x) == 1. for x in output.batch',
                      'assert all(float(x) == expected_reward for x in output.batch')
start=source.index("        if method == 'dt':")
end=source.index('    finally:',start)
source=source[:start]+'''        from executed_target_spans import full_trajectory_artifact
        artifacts=output.non_tensor_batch['dt_direct_target_artifact']
        assert len(artifacts)==10
        for artifact,event,turns in zip(artifacts,train.current,train.rollouts.records):
            trajectory=event.trajectory
            assert artifact['prompt_ids']==list(trajectory.prompt_ids)
            assert artifact['response_ids']==list(trajectory.response_ids)
            assert artifact['policy_mask']==list(trajectory.loss_mask)
            prompt=len(trajectory.prompt_ids)
            last=turns[-1]
            expected_target=[False]*len(trajectory.response_ids)
            for offset in range(last['start']-prompt,last['end']-prompt):
                expected_target[offset]=bool(trajectory.loss_mask[offset])
            assert artifact['target_mask']==expected_target
            assert any(expected_target)
            first=turns[0]
            assert not any(artifact['target_mask'][first['start']-prompt:first['end']-prompt])
            assert all(not t or p for t,p in zip(artifact['target_mask'],artifact['policy_mask']))
            assert len(artifact['retained_response_positions'])==output.batch['responses'].shape[-1]
''' +source[end:]
test=AUDIT/'source/test_SQL_settlement_boundary.py';test.write_text(source)
env=dict(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
result=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',
    str(baseline)+'::test_native_trajectory_is_one_training_row_and_observations_are_masked[grpo]',
    str(test),'--junitxml='+str(AUDIT/'SQL-boundary-tests.xml')],env=env,cwd=entry,capture_output=True)
(AUDIT/'SQL-boundary-tests.stdout.log').write_bytes(result.stdout)
(AUDIT/'SQL-boundary-tests.stderr.log').write_bytes(result.stderr)
print(result.stdout.decode(errors='replace')[-6500:]);print(result.stderr.decode(errors='replace')[-2000:])
result.check_returncode()
