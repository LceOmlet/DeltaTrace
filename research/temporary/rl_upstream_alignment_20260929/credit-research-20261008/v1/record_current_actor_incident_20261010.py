"""Bind the observed native warning, CPU cohort audit, and installed observer."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
RAW=HERE/'direct-credit-records-20261009-v1'
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
latest=max(RAW.glob('native-actor-observer-observation-*.json'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
observation=json.loads(latest.read_bytes())
assert observation['installation']['complete']
assert not observation['query_processes']
expected={987808:1791553850.00,989860:1791553867.51}
for group in observation['installation']['results']:
 assert len(group)==1
 item=group[0];assert expected[item['pid']]==item['birth']
 assert item['observer_sha256']=='8bc4fba4fd7f85cdb07e177b5f6460f374ea6dd24a4b8f172c72e3d52b5bb311'
code=r'''
import hashlib,json,psutil,re,time
from pathlib import Path
root=Path(ROOT);formal=root/'runs/textcraft-formal-stable-20261009-v1'
assert psutil.Process(982372).create_time()==1791553809.84
assert hashlib.sha256((formal/'source.json').read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
trainer=next(ray.glob('*-985585.out'));lines=trainer.read_text(errors='replace').splitlines()
steps=[]
for line in lines:
 match=re.search(r'step:(\d+) - ',line)
 if match:steps.append(dict(step=int(match[1]),original=line[match.start():]))
workers=[];warnings=[]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 p=psutil.Process(pid);assert p.create_time()==birth;m=p.memory_full_info()
 workers.append(dict(pid=pid,birth=birth,phase=p.name(),PSS_bytes=m.pss,RSS_bytes=m.rss))
 path=next(ray.glob('*-'+str(pid)+'.out'));reports=0
 for line in path.read_text(errors='replace').splitlines():
  if line.startswith('[DeltaTrace readout] '):reports+=1
  if 'grad_norm is not finite' in line:
   warnings.append(dict(pid=pid,birth=birth,completed_native_DT_readouts=reports,warning=line,source=str(path)))
latest=steps[-1]['step'];observed=time.time();updates=[]
assert warnings and {x['completed_native_DT_readouts'] for x in warnings}=={14}
observer=json.loads((root/'receipts/textcraft-native-actor-incidents-20261010-v1/installation-results.json').read_bytes())
assert observer['complete']
runtime=dict(observed_unix=observed,formal_pid=982372,formal_birth=1791553809.84,
 observer_source_sha256='8bc4fba4fd7f85cdb07e177b5f6460f374ea6dd24a4b8f172c72e3d52b5bb311',
 installation_result=observer,installed=True,first_input_snapshot_verified=False,
 official_actor_source_sha256='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
 loss_changes=0,configuration_changes=0,model_calls=0,training_limits_added=0)
override=formal/'runtime-overrides/native-actor-incidents-20261010-v1.json'
assert not override.exists()
override.write_text(json.dumps(runtime,indent=2)+'\n')
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=root/name;data=json.loads(path.read_bytes());old_other=[x for x in data['jobs'] if x.get('pid')!=982372]
 jobs=[x for x in data['jobs'] if x.get('pid')==982372];assert len(jobs)==1
 job=jobs[0];old=job.get('status');previous=hashlib.sha256(path.read_bytes()).hexdigest()
 job.update(status='formal_running_native_nonfinite_skip_under_investigation',
  last_status_observed_unix=observed,completed_iterations_observed=latest,total_iterations_observed=330,
  native_nonfinite_warning_iterations=[14],native_nonfinite_warning_ranks=2,
  native_nonfinite_root_cause='not_yet_localized',
  native_incident_observer_runtime_override=str(override),
  native_incident_observer_installed=True,native_incident_first_snapshot_verified=False)
 for key in ['complete_iterations_observed','completed_iterations']:
  if key in job:job[key]=latest
 assert [x for x in data['jobs'] if x.get('pid')!=982372]==old_other
 path.write_text(json.dumps(data,indent=2)+'\n')
 updates.append(dict(path=str(path),previous_status=old,status=job['status'],
  previous_sha256=previous,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(dict(unix=observed,completed_iterations=latest,total_iterations=330,
 latest_original_step=steps[-1],workers=workers,warnings=warnings,
 observer_runtime_override=dict(path=str(override),sha256=hashlib.sha256(override.read_bytes()).hexdigest()),
 authority_status_updates=updates,host_available_bytes=psutil.virtual_memory().available)))
'''.replace('ROOT',repr(transport.ROOT),1)
script='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=45)
(RAW/'native-actor-incident-status.stderr').write_bytes(r.stderr);r.check_returncode()
current=json.loads(r.stdout);status=RAW/'native-actor-incident-current-status.json';status.write_bytes(r.stdout)
def artifact(path):
 return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
audit=json.loads((RAW/'complete-fourteenth-DT-record-audit.json').read_bytes())
receipt=dict(status='Native nonfinite skip at formal iteration14; cause unresolved; passive observer installed',
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8b2e49db118f46e3c05e86944dcf8e293',
 upstream_commit='20bd331',formal_pid=982372,formal_birth=1791553809.84,
 observed_unix=current['unix'],completed_formal_iterations=current['completed_iterations'],total_formal_iterations=330,
 warning_iteration=14,two_rank_warnings_one_global_skip=True,
 native_owner_methods_AST_identical=['update_policy','_optimizer_step'],
 formal_iteration14=dict(DT_record_files=audit['files'],rank_rows=audit['original_rank_rows'],
  unique_trajectories=audit['unique_traj_uids'],CPU_original_credit_seconds=audit['seconds'],
  original_credit_all_finite=True,minimum_raw_source=min(audit['rows'],key=lambda x:x['minimum_prior_source']['raw_advantage']),
  input_old_ref_log_probs_optimizer_RNG_snapshot_available=False),
 observer=dict(installed=True,first_snapshot_verified=False,
  source=artifact(HERE/'capture_current_actor_incident_20261010.py'),
  native_step_count_hook='torch.optim.Optimizer.register_step_post_hook; optimizer.step is not overridden',
  changes='Instance observers delegate to original bound update_policy and _optimizer_step; rolling CPU diagnostic snapshots only.',
  new_model_calls=0,loss_changes=0,configuration_changes=0,training_limits_added=0),
 current=current,raw={p.name:artifact(p) for p in [
  RAW/'nonfinite-log-context-1791598764.json',RAW/'current-formal-nonfinite-inspection-20261010.json',
  RAW/'native-actor-official-skip-source-20261010.json',RAW/'complete-fourteenth-DT-record-audit.json',
  RAW/'fourteenth-DT-complete-observation.json',RAW/'fourteenth-DT-audit-command.sh',
  RAW/'native-actor-observer-stage.json',latest,status]},
 diagnosis='Saved DT credit is finite. Native gradient skip is not yet localized. No repair or full training health claim.',
 controller_incident='Initial metadata write NameError happened after the observer query started; the same PID1511718/birth1791599462.31 was recovered and completed, with no duplicate query.')
path=REPO/'experiments/rl/results_textcraft_native_nonfinite_20261010.json'
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=dict(observed_unix=current['unix'],status=receipt['status'],numerical_version=receipt['numerical_version'],
 numerical_source_commit=receipt['numerical_source_commit'],formal_pid=982372,formal_birth=1791553809.84,
 completed_iterations=current['completed_iterations'],total_iterations=330,warning_iteration=14,
 receipt=artifact(path),observer_installed=True,observer_first_snapshot_verified=False)
runtime=REPO/'experiments/rl/current_runtime.json';raw=runtime.read_bytes();old=json.loads(raw)
key='latest_textcraft_native_nonfinite_20261010';assert key not in old
eol=b'\r\n' if raw.startswith(b'{\r\n') else b'\n';start=b'{'+eol;assert raw.startswith(start)
value=json.dumps({key:entry},ensure_ascii=False,indent=2).encode().replace(b'\n',eol)
new=start+value[len(start):-len(eol+b'}')]+b','+eol+raw[len(start):]
parsed=json.loads(new);assert {k:v for k,v in parsed.items() if k!=key}==old
runtime.write_bytes(new)
local=datetime.datetime.fromtimestamp(current['unix'],datetime.timezone(datetime.timedelta(hours=8)))
note=f'''## {local:%Y-%m-%d %H:%M} TextCraft 第14轮原生梯度跳过，已保存诊断证据并接入被动现场保存

原PID982372/出生1791553809.84及两worker保持，现已完成{current['completed_iterations']}/330轮；
当前正式状态为非有限梯度原因调查中，不能称所有更新健康。第14轮两rank各一条WARN
对应一次全局skip，原step14 grad_norm=nan；其后15、16轮有限。原update_policy及
_optimizer_step与固定VERL-agent20bd331原源码AST一致，非有限时zero_grad并不执行step。
原actor实际SHA3a65e173…；没有改动PPO数学、dtype、mask或超参来规避告警。

原第14轮38个DT批次、152个rank行/149条去重轨迹与原callback路径及UID完全匹配；
CPU复用原信用owner0d3412b8…重算全部优势有限，0.244秒，不作反事实准确性或NaN根因结论。
第14轮当时没有保存actor的old/ref log-prob、更新前LoRA/optimizer/RNG，不能声称精确复现。
已有原始warn、step、DT记录及源码身份已保留；不为重建旧现场重复完整训练。

经原worker通用RPC在原生更新边界安装被动observer8bc4fba4…，两worker完成时间
1791599796.751/1791599796.754；query1511718已退出。仅委托原update_policy/_optimizer_step，
原Optimizer.register_step_post_hook计数，原step方法不覆盖；滚动保存真实输入、可训练
局部参数、optimizer局部state和RNG到CPU；若原skip再现保留该输入。尚未验证首份snapshot
及其实际资源/耗时，不把安装称现场保存已验收。运行时override另记，不修改数值基线。
提交器首版在Popen后写元数据NameError，复用实际同PID恢复记录，未再次提交或重启。
数值版本仍26bef6c8/fla-early-output-scale-20261009-v1，LoRA8/16、每卡B4、原正式预算不变；
无额外模型/DT/反向/optimizer计算、诊断停步或恢复。AppWorld/SQL/GRPO未启动。
回执results_textcraft_native_nonfinite_20261010.json；三份远端authority状态已明确告警未修复。

'''
record=REPO/'experiments/rl/RUNTIME_RECORD.md';raw=record.read_bytes();separator=b'\r\n' if raw.startswith('# 当前运行版本与修复记录\r\n'.encode()) else b'\n'
first_end=raw.index(separator+separator)+2*len(separator)
record.write_bytes(raw[:first_end]+note.encode().replace(b'\n',separator)+raw[first_end:])
print(json.dumps(dict(receipt=str(path),completed_iterations=current['completed_iterations'],
 observer_installed=True,first_snapshot_verified=False,root_cause='unresolved',
 latest_original_step=current['latest_original_step']['original']),ensure_ascii=False))
