"""Reconcile exact terminal receipts and prepared-only observer status."""
from pathlib import Path
import datetime,hashlib,json
HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]
REPO=AUDIT.parents[2]
MLP=AUDIT/'direct-target-mlp-token-chunk-20261007/v1'

def receipt(path):
    return dict(path=path.resolve().as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)

def load(path):return json.loads(path.read_text(encoding='utf-8'))

terminal=load(HERE/'paused-terminal-1791366952.json')
app=load(MLP/'appworld-terminal-summary-1791366096.json')
audit=load(MLP/'old-stable-to-joint-memory-path-audit-20261007.json')
assert all(not p.get('exists',True) for p in terminal['processes'])
observer=HERE.parent/'v2/observe_token_credit.py'
tests=HERE.parent/'v2/test_observe_token_credit.py'
assert receipt(observer)['sha256']=='81d087ddef613aa5ee6dd64a952e29fd14071fade8245289f5442123c9d45215'
assert receipt(tests)['sha256']=='9e7d1cf7df1df0e1f13195c354f6a18481660f47736fb151f6f8e78cdee20160'
facts=dict(observed_unix=terminal['observed_unix'],
 scope='Passive credit observation preparation and original failure audit only; no algorithm or model update, restart, checkpoint restore/export or new numerical acceptance criterion',
 textcraft=dict(pid=110053,birth=1791344324.6,
  source_sha256='5013ebc878b972a5f52817f7c7de7ae55ddae7f1d61609e943e1ab55bf59b993',
  completed_updates=7,eighth_update_completed=False,status='terminal_after_incorrect_OS_debug_pause',
  chronology=[dict(unix=1791365495.869284,event='Parent SIGSTOP paused exact TextCraft tree before DT8/actor8'),
   dict(unix=1791365862.1221607,event='Infrastructure/workers resumed; driver/TaskRunner still SIGSTOP'),
   dict(local_time='2026-10-07 17:40:04.101 +08:00',event='GCS declared stopped driver unavailable and destroyed TaskRunner'),
   dict(local_time='2026-10-07 17:40:09.101 +08:00',event='Raylet SIGKILL TaskRunner and cascaded owner death to workers'),
   dict(unix=1791366938.4662642,event='Only failed driver resumed to receive native Ray terminal exception/cleanup; all actors already absent')],
  cause='Incorrect parent OS pause stopped Ray driver control threads; original GCS/raylet interval records owner-death cleanup. Not a DT OOM; observer never installed.',
  missing_evidence='Old extreme values cannot be paired to raw d/token positions; eighth in-memory batch lost before observer installation. No saved raw vectors exist.',
  original_terminal=receipt(HERE/'paused-terminal-1791366952.json'),
  owner_death=receipt(HERE/'owner-death-1791366654.json'),
  pause=receipt(HERE/'pause-1791365495.json'),resume_infrastructure=receipt(HERE/'rpc-ready-1791365862.json'),
  failed_driver_cleanup=receipt(HERE/'failed-driver-release.json')),
 appworld=dict(pid=2001805,birth=1791362313.39,source_sha256=app['identity']['source_sha256'],
  status=app['status'],completed_updates=0,failed_batch=29,failed_batch_length=None,failed_layer_index=None,
  error=app['primary_error'],original_terminal=receipt(MLP/'appworld-terminal-original-1791366096.json'),
  terminal_summary=receipt(MLP/'appworld-terminal-summary-1791366096.json'),
  old_to_new_path_audit=receipt(MLP/'old-stable-to-joint-memory-path-audit-20261007.json'),
  cause_scope='Failure before whitening, in native HF/PEFT replay. Resource/offload configuration unchanged from old stable label run; new direct target omits explicit prefix lease wiring; automatic common-prefix remains. Finite MLP chunk does not cover original HF/PEFT MLP temporary. Exact failed shape/dtype/live allocation breakdown absent; omitted lease alone not proven cause.',
  no_restart=True),
 observer=dict(status='v2_prepared_only_not_deployed_or_imported_in_any_training_worker',
  source=receipt(observer),tests=receipt(tests),
  local_verification=dict(command='C:/Users/Administrator/miniconda3/envs/pytorch/python.exe -X utf8 '+tests.as_posix(),
   returncode=0,unittest_passed=8,observed_test_seconds=2.179,
   scope='Original frozen CPU readout interface transport, binding/exception restoration, native FP64 preservation, slot identity, release-file thread hold, partial batch persistence; not DT accuracy, FA/FLA/PPO tolerance, Ray heartbeat or training verification'),
  design='Native trace saved after each completed owner batch; consumed FP32 d and original Q/V/A at unchanged return; original post-scatter/post-whitening actor DataProto saved before original update; explicit file release wait without OS SIGSTOP',
  v1_status='Retained unaccepted candidate; install rejected on missing TaskRunner before any RPC/module import; do not use SIGSTOP hold',
  actual_numeric_capture_completed=False),
 resources=dict(physical_snapshot=receipt(HERE/'paused-terminal-1791366952.json'),
  physical_gpu_mib_each=858,no_gpu_processes=True,cgroup_memory_bytes=terminal['cgroup_memory_bytes'],scope='17:55:52 stage snapshot, not historical peak'))
result_path=REPO/'experiments/rl/results_token_credit_debug_20261007.json'
result_path.write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime_path=REPO/'experiments/rl/current_runtime.json'
runtime=load(runtime_path)
runtime['observed_unix']=terminal['observed_unix']
runtime['observed_utc']=datetime.datetime.fromtimestamp(terminal['observed_unix'],datetime.timezone.utc).isoformat()
runtime['latest_readonly_observation']=dict(receipt=receipt(result_path),
 textcraft='Terminal after incorrect OS debug pause: original driver/TaskRunner/workers absent; completed update7 only; raw token observer not installed',
 appworld='Terminal first DT batch29 native HF/PEFT LoRA MLP OOM, no full DT/PPO update; no restart',
 physical_resources='17:55:52 all physical GPUs858MiB/no processes; phase snapshot only')
for job in runtime['jobs']:
 if job['task']=='TextCraft' and job.get('pid')==110053:
  assert job['source']['sha256']==facts['textcraft']['source_sha256']
  job['declared_job_status']='terminal_after_incorrect_parent_OS_debug_pause'
  job['latest_observation']=facts['textcraft']
  job['debug_observer']=facts['observer']
 elif job['task']=='AppWorld' and job.get('pid')==2001805:
  assert job['source']['sha256']==facts['appworld']['source_sha256']
  job['declared_job_status']='terminal_native_replay_lora_oom_before_first_full_DT'
  job['latest_observation']=facts['appworld']
runtime_path.write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
record_path=REPO/'experiments/rl/RUNTIME_RECORD.md'
old=record_path.read_bytes()
title,body=old.split(b'\n',1)
section='''
## 2026-10-07 17:56 数值检查优先；TextCraft错误暂停导致终止，AppWorld新原生重放OOM

用户要求先查极端信用，不继续拿训练进展替代逐token证据。TextCraft原PID110053/
source5013完成7次更新；第8次在ref阶段尚无DT/actor更新。父代理17:31对原进程树
SIGSTOP，17:37仅恢复基础设施/workers，仍暂停driver/TaskRunner。这是错误的调试暂停：
原GCS17:40:04判停止的driver控制连接不可用并销毁TaskRunner，raylet17:40:09
SIGKILL并级联回收两rank。原区间日志已保存；不是DT OOM，不能称仍安全暂停。
观测器安装前检查即因TaskRunner已不存在退出，未在worker导入或改变数值。
17:55仅恢复已失败driver接收原Ray错误/清理；17:55:52全部原进程已无，未重新提交。
旧−23/−110等汇总极值缺少原位置/d，不能恢复配对；第8批未保存现场也已丢失。

新增被动观测v2仅prepared，CPU8/8接口测试通过，未部署/未证明实际Ray心跳或DT精度。
每个原trace成功返回立即保存FP64 signed、case/UID/原IDs/位置/原detail；完整读出再保存
实际FP32 d与原Q/V/A；原actor更新入口保存scatter及整批官方白化后的DataProto。
显式释放文件前Event.wait等待，不再SIGSTOP Ray进程；原公式/返回/exception/参数不变。
v1错误暂停候选及失败安装记录保留，不在默认启动路径。没有模型重放、额外采样、
信用裁剪/纠偏、检查点导出/恢复或新增数值验收标准；真实向量采集尚未完成。

AppWorld原PID2001805/source24b9本次在第29/29批原HF/PEFT MLP重放up_proj LoRA
乘scaling申请4.49GiB时仅余3.03GiB，完整DT/PPO未完成；失败批length/layer/dtype未记。
此前有限MLP分块628006/1c58仍是该有限边界的已对照实现，不覆盖此次原native MLP。
与旧稳定sourceaa8fac/checkpoint28比对：B4、LoRA8/16、FSDP参数/优化器卸载、
activation_offload与vLLM资源配置相同；官方offload及worker文件SHA未变且sleep有原日志。
旧为辅助类别标签EventRatioReadout，新为真实joint action target；旧显式lease工厂/provider
未接入新DirectActionTargetReadout，但runner自动common-prefix仍存在，不能说前缀全部关闭。
该遗漏和新增原生重放峰值分别调查；未证明仅接lease就能修峰值。错误早于优势白化。
官方activation_offload原eval分支直接原forward，不等于能消除DT eval前向的MLP临时量。
不能用旧标签长期运行、旧短target32k或有限MLP对照宣称新joint整链容量已验收。

本次完整来源/SHA、源码行号、prepared状态和终态见results_token_credit_debug_20261007.json
及old-stable-to-joint-memory-path-audit-20261007.json。17:55:52物理所有GPU858MiB/no processes，
是退出后快照非峰值；两组均未重启，SQL/GRPO未启动，方法PLAN未改。

'''
record_path.write_bytes(title+b'\n'+section.encode('utf-8')+body)
index_path=REPO/'experiments/rl/results_direct_target_mlp_token_chunk_20261007.json'
index=load(index_path)
index['last_completed_textcraft_update_observation']=index.pop('latest_formal_observation',None)
index['last_formal_phase_before_terminal']=index.pop('latest_formal_phase_observation',None)
index['latest_formal_observation']=dict(receipt=receipt(result_path),observed_unix=terminal['observed_unix'],
 textcraft=facts['textcraft']['status'],appworld=facts['appworld']['status'])
index['status']='terminal_native_HF_PEFT_replay_OOM; finite_MLP_comparison_retained; no_restart'
index['current_phase']='AppWorld terminated before full DT; TextCraft terminated after incorrect OS debug pause'
index['terminal_diagnostic']=facts['appworld']
index_path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(result=receipt(result_path),runtime=receipt(runtime_path),record=receipt(record_path))))
