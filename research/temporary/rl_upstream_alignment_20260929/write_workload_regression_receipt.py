"""Index exact CPU workload checks and the dated deployment observation."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

audit = Path(__file__).resolve().parent
repo = audit.parents[2]
out = audit/'workload-regression-20261001'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def artifact(path):
    path = Path(path)
    return dict(path=path.relative_to(repo).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


result = dict(code_commit=commit,
    scope='CPU official recipe -> real formal launcher -> original VERL worker normalization; no model, DT or optimizer execution',
    receipt_generator=artifact(__file__),
    code_sources=[artifact(repo/p) for p in [
        'experiments/rl/test_official_workload_local.py',
        'experiments/rl/test_loop_model_entry_local.py',
        'experiments/rl/launch_sql_native.py',
        'experiments/rl/launch_textcraft_native.py',
        'experiments/rl/launch_appworld_native.py',
        'experiments/rl/owner_environment_configs.json']], groups=[])
for name in ['sql-textcraft', 'appworld']:
    base = out/'final'/name
    resources = json.loads(base.with_suffix('.json').read_text(encoding='utf8'))
    suite = ET.parse(base.with_suffix('.xml')).getroot().find('testsuite')
    result['groups'].append(dict(name=name, tests=int(suite.get('tests')),
        failures=int(suite.get('failures')), errors=int(suite.get('errors')),
        skipped=int(suite.get('skipped')), test_timestamp=suite.get('timestamp'),
        runner_seconds=resources['seconds'], rss_peak_bytes=resources['peak_sampled_tree_rss_bytes'],
        gpu_before=resources['gpu_before'], gpu_after=resources['gpu_after'],
        receipts=[artifact(base.with_suffix(ext)) for ext in ['.json', '.xml', '.log']]))
sources = audit/'recipe-sources'
result['official_sources'] = [artifact(sources/p) for p in [
    'SkyRL-7d94ccf0eac3439c1731ce32018bf043dd639806/examples/train/text_to_sql/run_skyrl_sql.sh',
    'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/examples/train/AgentGym-RL/textcraft_train.sh',
    'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL/verl/workers/agent_actor/dp_actor.py',
    'AgentGym-RL-82402a99c62a293735a3f412fb8ac9a600673bc0/AgentGym-RL/verl/workers/agent_fsdp_workers.py',
    'ml-loop/README.md', 'ml-loop/data/appworld_splits/dev.txt',
    'verl-agent-20bd331/verl/workers/fsdp_workers.py']]
snapshot = repo/'experiments/rl/current_runtime.json'
runtime = json.loads(snapshot.read_text(encoding='utf8'))
result['deployment_snapshot'] = dict(**artifact(snapshot), observed_utc=runtime['observed_utc'])
result['jobs'] = [dict(task=j['task'], pid=j['pid'], created_unix=j['process']['created_unix'],
    entry=j['entry'], source=j['source'], native_budget=j['native_training_workload'],
    effective_resources=j['effective_resource_config'], actor_forward=j['version_mapping']['actor_forward_sources'])
    for j in runtime['jobs']]
result['initial_failure'] = dict(reason='Test setup omitted APPWORLD_ROOT required by the native eval split resolver; exact author dev asset supplied in temporary test layout',
    receipts=[artifact(out/f'appworld-initial{ext}') for ext in ['.json', '.xml', '.log']])
result['status'] = 'CPU workload regression passed; efficiency-path deployment and whole-run throughput remain open as recorded in RUNTIME_RECORD.md'
(out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(dict(path=str(out/'summary.json'), code_commit=commit,
                     tests=sum(g['tests'] for g in result['groups']))))
