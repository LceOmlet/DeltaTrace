"""One explicit checkpoint5 continuation using the original native controller."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
RUN = ROOT / 'runs/textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v2'
VERSION = RUN.name
PROCESS_OWNER = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
PROCESS_SHA = 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_checkpoint_files(checkpoint):
    assert sha(PROCESS_OWNER) == PROCESS_SHA
    tree = ast.parse(PROCESS_OWNER.read_text())
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and node.lineno in (165, 166)]
    assert sorted(node.lineno for node in assignments) == [165, 166]
    namespace = dict(final=Path(checkpoint), method='dt')
    exec(compile(ast.Module(body=sorted(assignments, key=lambda node: node.lineno),
                            type_ignores=[]), str(PROCESS_OWNER), 'exec'), namespace)
    return namespace['files']


def ready():
    row = json.loads((AUDIT / 'resume5-dt-prepared.json').read_text())
    assert row['run'] == str(RUN) and row['resumed_completed_iteration'] == 5
    assert row['previous_job_identity']['run'] == str(RUN)
    assert row['previous_job_identity']['pid'] == 2654698
    assert row['budget']['total_iterations'] == 330
    assert row['options']['trainer.total_epochs'] == 30
    assert row['options']['trainer.n_gpus_per_node'] == 8
    assert row['options']['trainer.resume_mode'] == 'resume_path'
    assert row['options']['trainer.resume_from_path'] == row['resume_checkpoint']
    assert Path(row['resume_checkpoint']).name == 'global_step_5'
    assert not row['fresh_base_model']
    evidence = json.loads((AUDIT / 'resume5-failure-precondition.json').read_text())
    assert evidence['native_exit_code'] == 1 and evidence['completed_iterations'] == 5
    assert sha(RUN / 'source.json') == evidence['previous_source_SHA256']
    import psutil
    for pid, birth in ((2654663, 1792605592.06), (2654698, 1792605592.43)):
        try:
            process = psutil.Process(pid)
            assert not (process.create_time() == birth and process.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            pass
    assert json.loads((AUDIT / 'resume5-CLI-verification.json').read_text())['status'] == \
        'passed_original_Hydra_parser_and_native_VERL_cfg_only'
    states = json.loads((AUDIT / 'resume5-native-checkpoint-state-verification.json').read_text())
    assert states['status'] == 'verified_native_checkpoint5_prelaunch_metadata'
    assert states['optimizer_step_values'] == [20.0] and states['scheduler_epochs'] == [5]
    files = native_checkpoint_files(row['resume_checkpoint'])
    assert len(files) == 25 and all(path.is_file() and path.stat().st_size for path in files)
    assert (Path(row['checkpoints']) / 'latest_checkpointed_iteration.txt').read_text().strip() == '5'
    for path, digest in row['source_bindings'].items():
        assert sha(path) == digest, path
    for path, digest in row['orchestration_source_bindings'].items():
        assert sha(path) == digest, path
    return row


def controller_source():
    """Reuse the existing 330 adapter, with only explicit resume boundary seams."""
    adapter = AUDIT / 'source/run_campaign.py'
    assert sha(adapter) == '2caba464da5e8fb91044700e6f6d3eea9f4599143cd1e02a4172f623057687b4'
    values = [node.value for node in ast.walk(ast.parse(adapter.read_text()))
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == 'replacements' for target in node.targets)]
    assert len(values) == 1
    replacements = ast.literal_eval(values[0])
    assert sha(PROCESS_OWNER) == PROCESS_SHA
    code = PROCESS_OWNER.read_text()
    for before, after in replacements.items():
        expected = 2 if before == "owner['collect'](str(ROOT), [row['output']])" else 1
        assert code.count(before) == expected
        code = code.replace(before, after)
    resume_seams = {
        'run.mkdir(parents=True, exist_ok=False)': 'run.mkdir(parents=True, exist_ok=True)',
        'output.mkdir()': 'output.mkdir(exist_ok=True)',
        'checkpoint.mkdir(parents=True, exist_ok=False)': 'checkpoint.mkdir(parents=True, exist_ok=True)',
        "with stdout.open('xb') as out, stderr.open('xb') as err:":
            "with stdout.open('ab') as out, stderr.open('ab') as err:",
        '        owner = original_log_reader()':
            "        owner = original_log_reader()\n        from native_resume_chain import collect\n        owner['collect'] = collect",
    }
    for before, after in resume_seams.items():
        assert code.count(before) == 1, before
        code = code.replace(before, after)
    compile(code, str(PROCESS_OWNER), 'exec')
    return code


if __name__ == '__main__':
    code = controller_source()
    assert sha(AUDIT / 'source/composed-native-resume5-resource-owner.py') == \
        hashlib.sha256(code.encode()).hexdigest()
    namespace = dict(__name__='native_same_run_resume5_process_owner', __file__=str(PROCESS_OWNER))
    exec(compile(code, str(PROCESS_OWNER), 'exec'), namespace)
    namespace['main'](audit=AUDIT, version=VERSION, readiness=ready)
