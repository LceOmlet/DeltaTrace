"""Record this run using the unchanged native log collector and saved checks."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

from transport import run_python

HERE = Path(__file__).resolve().parent
data = json.loads(run_python(r'''
from pathlib import Path
import hashlib,json,time,runpy,psutil
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=ROOT/'receipts/textcraft-v5-credit-integration-20261020-v1'
s=json.loads((A/'campaign-state.json').read_bytes());j=s['jobs'][0]
p=psutil.Process(j['pid']);assert abs(p.create_time()-j['birth'])<.02
assert abs(psutil.Process(s['controller_pid']).create_time()-s['controller_birth'])<.02
reader=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(reader.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(reader),run_name='v5_20261020_original_collector')
snapshot=owner['collect'](str(ROOT),[j['output']]);current,=snapshot['jobs']
(A/'dt-current-source-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
bindings=[]
resource_overrides=[]
for q in sorted((Path(j['run'])/'runtime-overrides').glob('actual-worker-imports-*.json')):
    value=json.loads(q.read_bytes())
    resource_rows=[dict(rank=r.get('rank'),pid=r.get('pid'),birth=r.get('birth'),
        DT_CREDIT_MODE=r.get('DT_CREDIT_MODE'),
        actor_optimizer_step_values=r.get('actor_optimizer_step_values'),
        **{k:v for k,v in r.items() if k.endswith('resource_override')})
        for r in value.get('results',[])]
    if any(any(k.endswith('resource_override') for k in row) for row in resource_rows):
        assert len(resource_rows)==8 and {r['rank'] for r in resource_rows}==set(range(8))
        resource_overrides.append(dict(path=str(q),SHA256=sha(q),
            classification='completed_runtime_resource_override',
            original_owner_status=value.get('status'),results=resource_rows))
    elif value.get('status')=='completed_read_only_worker_binding_check':
        bindings.append(dict(path=str(q),SHA256=sha(q),record=value))
bindings.sort(key=lambda item:item['record'].get('collected_unix',0))
cpu=json.loads((A/'CPU-behavior-verification.json').read_bytes())
native=json.loads((A/'native-behavior-status.json').read_bytes())
assert cpu['status']=='passed' and native['status']=='completed'
assert json.loads((A/'native-behavior-exit.json').read_bytes())['exit_code']==0
ranks=[]
for rank in range(8):
    q=A/f'native-multi-target-rank{rank}.json';v=json.loads(q.read_bytes())
    assert v['status']=='completed' and all(c['exact_expression_equality'] for c in v['credit_calls'])
    ranks.append(dict(rank=rank,path=str(q),SHA256=sha(q),
        native_attribute_calls=len(v['finite_calls']),credit_calls=len(v['credit_calls']),
        original_unscaled_seed=all(c['original_unscaled_seed'] for c in v['finite_calls']),
        per_event_exponential_exact=True,actual_imports=v['bindings']))
report=dict(collected=time.time(),state=s['status'],completed=owner['completed'](current),
    phase=current.get('phase_label'),checkpoint_step=current.get('checkpoint_step'),
    controller=dict(pid=s['controller_pid'],birth=s['controller_birth']),
    job={k:j.get(k) for k in ['task','method','version','pid','birth','run','output','checkpoints','source_receipt']},
    mode=j['effective_environment']['DT_CREDIT_MODE'],upstream_commit=j['upstream_commit'],
    execution_source_commit=j['local_patch_commit'],method_source_commit=j['method_source_commit'],
    expected_source_bindings=j['source_bindings'],effective_configuration=j['options'],
    effective_config_SHA256=j['effective_config_SHA256'],actual_worker_bindings=bindings,
    completed_runtime_resource_overrides=resource_overrides,
    completed_DT_resource_reuse=(dict(path=str(A/'accepted-DT-resource-reuse-completed.json'),
        SHA256=sha(A/'accepted-DT-resource-reuse-completed.json'),
        classification='completed_runtime_resource_override_not_read_only')
        if (A/'accepted-DT-resource-reuse-completed.json').exists() else None),
    workers=[dict(pid=q.pid,birth=q.create_time(),phase=q.name()) for q in p.children(recursive=True) if 'WorkerDict' in q.name()],
    CPU_verification=dict(status=cpu['status'],tests=cpu['tests'],path=str(A/'CPU-behavior-verification.json')),
    two_eight_rank_transport=json.loads((A/'two-eight-rank-credit-transport.json').read_bytes()),
    native_DT_u_verification=dict(status=native['status'],physical_gpus=native['physical_gpus'],ranks=ranks,
        driver=json.loads((A/'native-multi-target-driver.json').read_bytes()),
        tested_wrapper_SHA256=json.loads((A/'native-behavior-job.json').read_bytes())['verification_wrapper_SHA256'],
        raw_native_returns=str(A/'native-multi-target-native-return.pt'),
        restored_driver_returns=str(A/'native-multi-target-driver-return.pt'),
        initial_failed_comparison=dict(path=str(A/'attempt1-padding-oracle-failure'),
            original_full_return_tensors_preserved=False,
            causal_explanation_established=False,
            oracle_identity_defect_identified=True,
            corrected_original_row_identity_comparison_passed=True),
        saved_tensor_oracle_replay=(json.loads((A/'padding-oracle-saved-tensor-replay.json').read_bytes())
            if (A/'padding-oracle-saved-tensor-replay.json').exists() else None),
        verification_optimizer_updates=0,scope='Eight-rank original real-payload DT/u transport, not a new 32768 capacity or prediction-accuracy certificate'),
    platform_restoration=json.loads((A/'restored-platform-files.json').read_bytes()),
    controller_lock=json.loads((A/'controller-lock-storage-verification.json').read_bytes()),
    service=json.loads((A/'official-service.json').read_bytes()),
    preserved_failed_checks=[str(A/'attempt1-padding-oracle-failure'),str(A/'attempt1-controller-shared-lock-failure')],
    latest_metrics=current['metrics'][-1] if current['metrics'] else None,
    first_complete_rollout_verification=(json.loads((A/'native-first-complete-rollout-verification.json').read_bytes())
        if (A/'native-first-complete-rollout-verification.json').exists() else None),
    first_complete_source_snapshot=(dict(path=str(A/'first-complete-iteration-source-snapshot.json'),
        SHA256=sha(A/'first-complete-iteration-source-snapshot.json'))
        if (A/'first-complete-iteration-source-snapshot.json').exists() else None),
    native_source_snapshot=str(A/'dt-current-source-snapshot.json'),
    original_reader_SHA256=sha(reader),additional_production_model_DT_optimizer_calls=0,
    training_algorithm_source_changes=0,budget=200,prior_histories_spliced=False)
(A/'latest-deployment-observation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
''', timeout=55))

data['recorded_at'] = datetime.fromtimestamp(data['collected'], timezone(timedelta(hours=8))).isoformat()
data['status'] = 'submitted_initialization_pending_first_native_iteration'
if data['actual_worker_bindings']:
    latest = data['actual_worker_bindings'][-1]['record']
    assert len(latest['results']) == 8
    assert all(r['DT_CREDIT_MODE'] == 'reward_target_exponential_v5' and r['DP_world_size'] == 8 for r in latest['results'])
    assert not any(v.get('matches_prepared_source') is False for r in latest['results'] for v in r['modules'].values())
    data['status'] = 'eight_rank_mode_and_loaded_imports_verified'
    data['DT_owner_lazy_imports_pending'] = any(r['modules'].get('reward_readout', {}).get('imported') is False for r in latest['results'])
    if all(r.get('actual_v5_readout_flags') is not None for r in latest['results']):
        for row in latest['results']:
            flags = row['actual_v5_readout_flags']
            assert flags['reward_target_exponential'] is True
            assert flags['clip_exponential_credit'] is False
            assert flags['symmetric_probability_ratio'] is False
            assert flags['weighted_target_credit'] is False
        data['actual_v5_readout_flags_verified'] = True
    if data['completed'] and not data['DT_owner_lazy_imports_pending']:
        data['status'] = 'native_complete_iteration_and_actual_eight_rank_DT_imports_verified'
data['hourly_plot_entry'] = str(HERE/'plot_hourly.py')
data['hourly_plot_entry_SHA256'] = hashlib.sha256((HERE/'plot_hourly.py').read_bytes()).hexdigest()
data['strict_SSH_port'] = 30377
result = Path('experiments/rl/results_textcraft_v5_credit_integration_20261020_v1.json')
result.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
(HERE/'latest-deployment-observation.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
runtime = Path('experiments/rl/current_runtime.json')
state = json.loads(runtime.read_text(encoding='utf-8'))
state['20261020_TextCraft_v5_credit_integration_eight_GPU_current'] = data
runtime.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(dict(result=str(result),status=data['status'],completed=data['completed'],phase=data['phase'],
    actual_binding_receipts=len(data['actual_worker_bindings']),driver=data['job']['pid']), ensure_ascii=False))
