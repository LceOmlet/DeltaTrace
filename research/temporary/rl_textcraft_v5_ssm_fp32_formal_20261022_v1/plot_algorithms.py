"""Compose the accepted three-algorithm plot with the current v5 identity."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'rl_textcraft_v5_credit_integration_20261020_v1'))
from transport import run_python

owner = HERE.parent/'rl_textcraft_v5_eight_gpu_20261019_v1/plot_hourly.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '0b6da9c6e4e59ea8b05d7fce7eecc8d92a75e3d478ce2dff87bddef25aa730fb'
code = owner.read_text(encoding='utf-8')
for before, after in [
 ('textcraft-v5-eight-gpu-20261019-v1', 'textcraft-v5-native-SSM-fp32-20261022-v1'),
 ('textcraft-dtpo-v5-exponential-8gpu-200-20261019-v1', 'textcraft-dtpo-v5-ssm-fp32-8gpu-formal-20261022-v1'),
 ('textcraft-v5-eight-gpu-historical-PPO-GRPO-global-norm-20261019-v1', 'textcraft-v5-ssm-fp32-eight-gpu-vs-historical-PPO-GRPO-20261022-v1'),
 ('DTPO v5（新8卡）', 'v5（本次八卡，SSM FP32）'),
 ('Current eight-GPU v5', 'Current eight-GPU v5'),
 ('新八卡v5为全局实测', '本次v5为全局实测'),
]:
    assert before in code, before
    code = code.replace(before, after)

bridge = r'''
before="snapshot=owner['collect'](str(ROOT),[active['output']])"
assert source.count(before)==1
source=source.replace(before,"snapshot=owner['collect'](str(state_path.parent),[active['output']])")
before="assert current['pid_alive'] and (current['pid'],current['observed_process_created_unix'])==(active['pid'],active['birth'])"
assert source.count(before)==1
source=source.replace(before,"assert current['pid']==active['pid']\nif current['pid_alive']:\n    assert current['observed_process_created_unix']==active['birth']\nsnapshot['native_training_state']=dict(campaign_status=state['status'],source_status=active.get('status'),pid_alive=current['pid_alive'],exit_code=current.get('exit_code'))")

binding=r"""
import psutil
run_source=Path(active['run'])/'source.json'
deployed=json.loads(run_source.read_bytes())
control_path=state_path.parent/'controller-job.json'
controller=json.loads(control_path.read_bytes())
try:
    c=psutil.Process(controller['pid'])
    controller_alive=c.create_time()==controller['birth'] and c.status()!=psutil.STATUS_ZOMBIE
except psutil.Error:
    controller_alive=False
workers=[]
try:
    children=psutil.Process(active['pid']).children(recursive=True)
except psutil.NoSuchProcess:
    children=[]
for child in children:
    try:
        if 'WorkerDict' in child.name():
            workers.append(dict(pid=child.pid,birth=child.create_time(),phase=child.name()))
    except psutil.Error:
        pass
snapshot['active_execution_binding']=dict(run=active['run'],source_manifest=str(run_source),source_manifest_SHA256=hashlib.sha256(run_source.read_bytes()).hexdigest(),execution_source_commit=deployed.get('execution_source_commit'),method_source_commit=deployed.get('method_source_commit'),training_logic_parent_commit=deployed.get('training_logic_parent_commit'),source_bindings=deployed['source_bindings'],credit_mode=deployed['effective_environment'].get('DT_CREDIT_MODE'),native_SSM_cache_dtype=deployed['options'].get('+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype'),controller=dict(pid=controller['pid'],birth=controller['birth'],alive_with_matched_birth=controller_alive),driver=dict(pid=active['pid'],birth=active['birth'],alive_with_matched_birth=current['pid_alive']),live_workers=workers,completed_runtime_overrides=[dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (Path(active['run'])/'runtime-overrides').glob('*.json')])
observer=state_path.parent/'initial-worker-observer-state.json'
snapshot['existing_observer_state']=json.loads(observer.read_bytes()) if observer.exists() else None
snapshot['active_execution_binding']['metadata_scope']='Existing status and completed receipts only; no metadata RPC or recording installation.'
"""
source=source.replace("snapshot['collection_model_calls']=0",binding+"\nsnapshot['collection_model_calls']=0")

# Use the existing strict30377 transport and provisioned VENV directly.
start=source.index("script = ('source ")
end=source.index("snapshot['jobs'].extend([historical_ppo,historical_grpo])",start)
source=source[:start]+"snapshot=json.loads(run_python(remote,timeout=55))\n"+source[end:]

# The collector may serialize devices as CSV; normalize only representation.
anchor='\nimport math\nconversion=[]'
assert source.count(anchor)==1
normalization="\nfor job in snapshot['jobs']:\n    if isinstance(job['devices'],str):\n        job['original_owner_devices']=job['devices']\n        job['devices']=[int(x) for x in job['devices'].split(',') if x.strip()]\n"
source=source.replace(anchor,normalization+anchor)
anchor="stamp = dt.datetime.fromtimestamp(snapshot['collected_at']"
assert source.count(anchor)==1
metadata="""assert {j['method'] for j in snapshot['jobs']}=={'dt','ppo','grpo'}
assert len(historical_ppo['metrics'])==105 and len(historical_grpo['metrics'])==191
snapshot['comparison_scope']='Current v5 and independent frozen original PPO/GRPO controls; no concatenation. Historical sampling protocols differ; descriptive comparison.'
snapshot['primary_window']=max((m['step'] for j in snapshot['jobs'] if j['method']=='dt' for m in j['metrics']),default=0)
snapshot['primary_window_basis']='all_completed_iterations_of_current_v5'
"""
source=source.replace(anchor,metadata+anchor)

# Keep the original three-series common/full figures; omit control-only extras.
start=source.index('code = code.replace("csv_path=output/')
end=source.index('# All series extraction and plotting below remain',start)
source=source[:start]+source[end:]
anchor="exec(compile(code, str(renderer), 'exec'), globals())"
assert source.count(anchor)==1
extra=r"""
code=code.replace('MaxNLocator(integer=True)','MaxNLocator(integer=True,min_n_ticks=1)')
code=code.replace("csv_path=output/'metrics-with-sources.csv'", "if last['dt']:\n    draw(jobs,last['dt'],f'TextCraft · v5 / PPO / GRPO：本次v5全部 {last[chr(100)+chr(116)]} 轮','algorithm-comparison-v5-complete')\ncsv_path=output/'metrics-with-sources.csv'")
code=code.replace("primary_figure=str(output/('algorithm-comparison-common-window.png' if common else 'algorithm-comparison-full.png'))", "primary_figure=str(output/('algorithm-comparison-v5-complete.png' if last['dt'] else 'algorithm-comparison-full.png'))")
code=code.replace('report=dict(gradient_norm_conversion=', "report=dict(primary_window=last['dt'],primary_window_basis='all_completed_iterations_of_current_v5',gradient_norm_conversion=")
code=code.replace('TextCraft · 同全局学习负载：前', 'TextCraft · v5 / PPO / GRPO：共同前')
code=code.replace('TextCraft · 同全局学习负载：各独立 run 完整曲线', 'TextCraft · v5 / PPO / GRPO：各自完整曲线')
if not snapshot['primary_window']:
    code=code.replace('TextCraft · v5 / PPO / GRPO：各自完整曲线','TextCraft · 本次v5尚无完整迭代；历史PPO / GRPO完整曲线')
code=code.replace("(output/'plot-receipt.json').write_text", "(HERE/'latest-algorithm-output.txt').write_text(str(output)+'\\n',encoding='utf-8')\n(output/'plot-receipt.json').write_text")
"""
source=source.replace(anchor,extra+'\n'+anchor)
'''
anchor = "exec(compile(source,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, bridge+'\n'+anchor)
exec(compile(code, str(owner), 'exec'), globals())
