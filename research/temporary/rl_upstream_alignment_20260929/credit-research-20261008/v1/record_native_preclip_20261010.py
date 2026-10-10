"""Record the installed passive extension without changing the numerical baseline."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'experiments/rl/current_runtime.json').exists())
RAW=HERE/'direct-credit-records-20261009-v1'
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
if len(sys.argv)==3 and sys.argv[1]=='--bind-source':
    commit=subprocess.check_output(['git','rev-parse',sys.argv[2]],cwd=REPO,text=True).strip()
    relative=(HERE/'capture_native_preclip_20261010.py').relative_to(REPO).as_posix()
    blob=subprocess.check_output(['git','show',commit+':'+relative],cwd=REPO)
    expected='110428a8674e9fb49c27a468a36ae9a78ce030bc184b3d2aca0d4394c737c6c0'
    assert hashlib.sha256(blob).hexdigest()==expected
    code=r'''
import hashlib,json,psutil,time
from pathlib import Path
root=Path(ROOT);assert psutil.Process(982372).create_time()==1791553809.84
formal=root/'runs/textcraft-formal-stable-20261009-v1'
override=formal/'runtime-overrides/native-preclip-preservation-20261010-v1.json'
data=json.loads(override.read_bytes())
assert all(x['observer_sha256']==EXPECTED for x in data['installation']['results'])
source=Path(data['installation']['results'][0]['observer_path'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==EXPECTED
data.update(observer_source_commit=COMMIT,source_uncommitted_at_installation=True,
            committed_source_SHA256_matches_deployed=True)
override.write_text(json.dumps(data,indent=2)+'\n')
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=root/name;content=json.loads(path.read_bytes());jobs=[j for j in content['jobs'] if j.get('pid')==982372]
 assert len(jobs)==1;jobs[0]['native_preclip_observer_source_commit']=COMMIT
 path.write_text(json.dumps(content,indent=2)+'\n')
print(json.dumps(dict(unix=time.time(),observer_source_commit=COMMIT,observer_sha256=EXPECTED,
 source_uncommitted_at_installation=True,committed_source_SHA256_matches_deployed=True,
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
 override_path=str(override),override_sha256=hashlib.sha256(override.read_bytes()).hexdigest())))
'''.replace('ROOT',repr(transport.ROOT),1).replace('EXPECTED',repr(expected)).replace('COMMIT',repr(commit))
    command='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    r=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=45)
    r.check_returncode();binding=json.loads(r.stdout)
    (RAW/'native-preclip-source-commit-binding-20261010.json').write_bytes(r.stdout)
    receipt_path=REPO/'experiments/rl/results_textcraft_native_preclip_20261010.json'
    receipt=json.loads(receipt_path.read_bytes());receipt['observer_source_commit_binding']=binding
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=REPO/'experiments/rl/current_runtime.json';raw=path.read_bytes();old=json.loads(raw)
    key='latest_textcraft_preclip_capture_20261010';entry=dict(old[key],observer_source_commit=commit)
    entry['receipt']=dict(path=str(receipt_path),bytes=receipt_path.stat().st_size,
                          sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    eol=b'\r\n' if raw.startswith(b'{\r\n') else b'\n';start=b'{'+eol
    def fragment(value):return json.dumps({key:value},ensure_ascii=False,indent=2).encode().replace(b'\n',eol)[len(start):-len(eol+b'}')]
    before=fragment(old[key]);assert raw.startswith(start+before+b','+eol)
    after=start+fragment(entry)+raw[len(start+before):]
    assert {k:v for k,v in json.loads(after).items() if k!=key}=={k:v for k,v in old.items() if k!=key}
    path.write_bytes(after)
    path=REPO/'experiments/rl/RUNTIME_RECORD.md';raw=path.read_bytes()
    sep=b'\r\n' if raw.startswith('# 当前运行版本与修复记录\r\n'.encode()) else b'\n'
    end=raw.index(sep+sep)+2*len(sep)
    note=('被动preclip观察扩展源码已绑定提交'+commit+'（SHA110428a8…），\n'
          '部署时未提交，现已验证该提交blob与远端已安装源码SHA完全相同；不是数值基线更新。\n'
          '数值源仍26bef6c8，绑定回执native-preclip-source-commit-binding-20261010.json。\n\n')
    path.write_bytes(raw[:end]+note.encode().replace(b'\n',sep)+raw[end:])
    print(json.dumps(binding));raise SystemExit(0)
snapshot_path=max(RAW.glob('native-input-capture-audit-*.json'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
snapshot=json.loads(snapshot_path.read_bytes())
assert len(snapshot['workers'])==2 and all(w['snapshot_exists'] for w in snapshot['workers'])
assert all(w['trainable_local_parameters']['nonfinite']==w['optimizer_local_state']['nonfinite']==0 for w in snapshot['workers'])
assert all(all(v['all_nonfinite']==0 for v in w['fields'].values()) for w in snapshot['workers'])
code=r'''
import hashlib,json,psutil,re,subprocess,time
from pathlib import Path
root=Path(ROOT);formal=root/'runs/textcraft-formal-stable-20261009-v1'
assert psutil.Process(982372).create_time()==1791553809.84
source_path=formal/'source.json'
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
out=root/'receipts/textcraft-native-actor-incidents-20261010-v1/preclip-v1'
installed=json.loads((out/'query-result-v2.json').read_bytes());assert installed['complete']
assert {x['pid']:x['birth'] for x in installed['results']}=={987808:1791553850.00,989860:1791553867.51}
assert all(x['observer_sha256']=='110428a8674e9fb49c27a468a36ae9a78ce030bc184b3d2aca0d4394c737c6c0' for x in installed['results'])
ray=Path('/tmp/ray/session_2026-10-09_21-50-24_958031_982372/logs')
trainer=next(ray.glob('*-985585.out'))
steps=[dict(step=int(m[1]),original=line[m.start():]) for line in trainer.read_text(errors='replace').splitlines()
 for m in [re.search(r'step:(\d+) - ',line)] if m]
workers=[]
for pid,birth in [(987808,1791553850.00),(989860,1791553867.51)]:
 p=psutil.Process(pid);assert p.create_time()==birth
 folder=next(out.parent.glob('rank*-pid'+str(pid)))
 events=[json.loads(line) for line in (folder/'native-events.jsonl').read_text().splitlines()]
 preclip=[json.loads(line) for line in (folder/'preclip-events.jsonl').read_text().splitlines()]
 native=[e for e in events if e['event']=='native_optimizer_return']
 pairs={(e['update_index'],e['optimizer_index']):e for e in preclip if e['event']=='native_norm_before_clipping'}
 assert native and all((e['update_index'],e['optimizer_index']) in pairs for e in native)
 assert all(e['grad_norm']==pairs[(e['update_index'],e['optimizer_index'])]['local_total_norm'] for e in native)
 assert not any(e['event'].endswith('error') for e in events+preclip)
 workers.append(dict(pid=pid,birth=birth,phase=p.name(),PSS_bytes=p.memory_full_info().pss,
  native_events=events,preclip_events=preclip,completed_native_optimizer_returns=len(native),
  original_optimizer_step_calls=sum(e['native_optimizer_step_calls'] for e in native),
  same_call_norm_return_matches=True))
runtime=dict(version='native-preclip-preservation-20261010-v1',observed_unix=time.time(),
 formal_pid=982372,formal_birth=1791553809.84,installation=installed,
 numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',
 first_input_snapshot_CPU_audit=SNAPSHOT,first_input_snapshot_verified=True,
 primitive_norm_calls='one original PyTorch _get_total_norm call; exact return object retained',
 clipping_or_optimizer_replacement=False,model_calls=0,DT_calls=0,
 root_cause_localized=False,repair_deployed=False,
 actual_native_invocation_verified=True,native_invocations=workers)
override=formal/'runtime-overrides/native-preclip-preservation-20261010-v1.json'
if override.exists():
 previous=json.loads(override.read_bytes())
 assert previous['version']==runtime['version'] and previous['formal_birth']==runtime['formal_birth']
override.write_text(json.dumps(runtime,indent=2)+'\n')
updates=[]
for name in ['formal-training.json','active-training.json','active-source.json']:
 path=root/name;data=json.loads(path.read_bytes());others=[j for j in data['jobs'] if j.get('pid')!=982372]
 jobs=[j for j in data['jobs'] if j.get('pid')==982372];assert len(jobs)==1
 job=jobs[0]
 job.update(native_preclip_preservation_runtime_override=str(override),native_preclip_preservation_installed=True,
  native_preclip_actual_invocation_verified=True,
  native_incident_first_snapshot_verified=True,status='formal_running_native_nonfinite_skip_under_investigation',
  last_status_observed_unix=runtime['observed_unix'],completed_iterations_observed=steps[-1]['step'])
 assert others==[j for j in data['jobs'] if j.get('pid')!=982372]
 path.write_text(json.dumps(data,indent=2)+'\n')
 updates.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=25,check=True).stdout
print(json.dumps(dict(unix=runtime['observed_unix'],installation=installed,workers=workers,
 completed_iterations=steps[-1]['step'],latest_original_step=steps[-1],physical_mx_smi=physical,
 runtime_override=dict(path=str(override),sha256=hashlib.sha256(override.read_bytes()).hexdigest()),
 authority_updates=updates,host_available_bytes=psutil.virtual_memory().available)))
'''.replace('ROOT',repr(transport.ROOT),1).replace('SNAPSHOT',repr(snapshot),1)
command='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
r=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=55)
r.check_returncode();current=json.loads(r.stdout)
status_path=RAW/'native-preclip-recorded-status-20261010.json';status_path.write_bytes(r.stdout)
def artifact(path):
 return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
math_path=RAW/'native-nonfinite-math-source-1791601095.json'
math=json.loads(math_path.read_bytes())
receipt=dict(status='Passive preclip extension installed; first real input snapshot verified; NaN cause unresolved',
 version='native-preclip-preservation-20261010-v1',numerical_version='fla-early-output-scale-20261009-v1',
 numerical_source_commit='26bef6c8b2e49db118f46e3c05e86944dcf8e293',upstream_commit='20bd331',
 observed_unix=current['unix'],formal_pid=982372,formal_birth=1791553809.84,current=current,
 owner_function_AST_checks={f['path']:{n['name']:n['matches_pinned_owner_AST'] for n in f['functions']} for f in math['files']},
 actual_native_fused_backend_log=math['native_flag_lines'],
 input_capture=snapshot,first_snapshot_verified=True,preclip_actual_native_invocation_verified=True,
 root_cause_localized=False,repair_deployed=False,model_calls_added=0,DT_calls_added=0,
 first_sender_failure='Missing frozen PYTHONPATH caused Ray client signature validation failure before worker dispatch. '
  'PID1706148 exited; frozen-env sender1724837 completed both installations. Both attempts retained.',
 raw={p.name:artifact(p) for p in [math_path,snapshot_path,status_path,
  RAW/'native-preclip-stage-20261010.json',RAW/'native-preclip-stage-frozen-env-v2-20261010.json',
  RAW/'native-actor-observer-observation-1791602308.json',
  RAW/'native-curves-20261010-111815/plot-receipt.json',
  RAW/'failed-preclip-sender-v1.py.txt',HERE/'capture_native_preclip_20261010.py',
  HERE/'test_capture_native_preclip_20261010.py',HERE/'retry_native_preclip_frozen_env_20261010.py']})
path=REPO/'experiments/rl/results_textcraft_native_preclip_20261010.json'
path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
key='latest_textcraft_preclip_capture_20261010'
entry=dict(observed_unix=current['unix'],status=receipt['status'],receipt=artifact(path),
 numerical_version=receipt['numerical_version'],formal_pid=982372,formal_birth=1791553809.84,
 completed_iterations=current['completed_iterations'],total_iterations=330,
 first_input_snapshot_verified=True,preclip_actual_invocation_verified=True,NaN_root_cause='unresolved')
runtime=REPO/'experiments/rl/current_runtime.json';raw=runtime.read_bytes();old=json.loads(raw)
eol=b'\r\n' if raw.startswith(b'{\r\n') else b'\n';start=b'{'+eol
value=json.dumps({key:entry},ensure_ascii=False,indent=2).encode().replace(b'\n',eol)
fragment=value[len(start):-len(eol+b'}')]
if key in old:
 old_value=json.dumps({key:old[key]},ensure_ascii=False,indent=2).encode().replace(b'\n',eol)
 old_fragment=old_value[len(start):-len(eol+b'}')]
 assert raw.startswith(start+old_fragment+b','+eol)
 new=start+fragment+raw[len(start+old_fragment):]
else:new=start+fragment+b','+eol+raw[len(start):]
assert {k:v for k,v in json.loads(new).items() if k!=key}=={k:v for k,v in old.items() if k!=key}
runtime.write_bytes(new)
stamp=dt.datetime.fromtimestamp(current['unix'],dt.timezone(dt.timedelta(hours=8)))
note=f'''## {stamp:%Y-%m-%d %H:%M} 原第18轮完整返回，裁剪前观察实际调用已确认

原PID与2/3两worker不变，原日志已完成{current['completed_iterations']}/330轮并进入后续采样。
首份真实输入保存已验证；每rank原4次optimizer调用均执行，8个裁剪前local_total_norm
与同次原_optimizer_step返回的grad_norm逐值相同，未发生observer_error或新非有限跳过。
实际返回是DTensor _NormPartial，各rank原局部值不同；监控仅展示原actor/grad_norm聚合日志，
不将这些标量另称完整全局L2范数，不改变原分布式裁剪行为。输入保存每卡最多1.878秒，
第18轮完整PPO更新已返回；原第14轮NaN根因仍未知，没有数值修复或训练全程健康结论。
原日志奖励/熵/梯度曲线已更新到native-curves-20261010-111815，回执及runtime override
已更新实际调用证据。正式参数、数值基线26bef6c8、预算及其他停止作业保持。

'''
record=REPO/'experiments/rl/RUNTIME_RECORD.md';raw=record.read_bytes()
sep=b'\r\n' if raw.startswith('# 当前运行版本与修复记录\r\n'.encode()) else b'\n'
end=raw.index(sep+sep)+2*len(sep)
record.write_bytes(raw[:end]+note.encode().replace(b'\n',sep)+raw[end:])
print(json.dumps(dict(receipt=str(path),completed_iterations=current['completed_iterations'],
 snapshot_verified=True,preclip_installed=True,root_cause='unresolved',physical=current['physical_mx_smi']),ensure_ascii=False))
