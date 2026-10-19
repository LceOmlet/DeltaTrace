"""Persist the existing native observers and their unchanged phase reduction."""
import datetime as dt
import json
from pathlib import Path

from transport import run_python


report = json.loads(run_python(r'''
import contextlib,hashlib,io,json,psutil,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2'
launch=json.loads((A/'native-observers-launch.json').read_bytes())
physical=json.loads((A/'physical-observer-job.json').read_bytes())
handles=launch['observers']+[dict(role='native_dmon',pid=physical['native_pid'],birth=physical['native_birth'])]
for handle in handles:
 try:
  p=psutil.Process(handle['pid'])
  handle['live_exact_birth']=p.create_time()==handle['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:handle['live_exact_birth']=False
reader=A/'source/report_native_utilization.py'
assert hashlib.sha256(reader.read_bytes()).hexdigest()=='efdcb0b662e74e1cbf57320b2e3c1698942bf32b5c1ce1d705c20f8829b2974f'
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(reader),run_name='__main__')
summary=A/'native-utilization-phase-summary.json'
host_path=R/'receipts/appworld-formal-training-20261017-v1/latest-host-RSS-PSS-XTT-v1.json'
host=json.loads(host_path.read_bytes())
job=next(j for j in host['jobs'] if j['version']=='textcraft-dtpo-v9-clip-u-8gpu-200-20261019-v2')
streams=[dict(path=row['path'],bytes=Path(row['path']).stat().st_size,mtime=Path(row['path']).stat().st_mtime,
              bytes_before_resumption=row['bytes']) for row in launch['stream_paths']]
report=dict(status='original_native_observers_recording_with_actual_current_identity',collected=time.time(),
 resumed_at=launch['created'],observer_launch=launch,actual_handles=handles,
 original_gap_receipt=str(A/'native-observation-gap-and-reuse.json'),
 streams=streams,phase_reduction=dict(path=str(summary),SHA256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                                    record=json.loads(summary.read_bytes())),
 latest_host_observation=dict(path=str(host_path),collected=host['collected'],
   MemAvailable_KiB=host['proc_meminfo_KiB']['MemAvailable'],cgroup_usage_bytes=host['cgroup_usage_bytes'],
   cgroup_oom_control=host['cgroup_oom_control'],driver_xtt_reports_KiB=host['driver_xtt_reports_KiB'],
   current_native_job=job,pss_sample=host['pss_sample'],collection_seconds=host['collection_seconds']),
 checkpoint_observer=json.loads((A/'native-TextCraft-DT-checkpoint5-observer-v1-job.json').read_bytes()),
 training_changes=0,model_calls=0,optimizer_calls=0,new_GPU_profiling_calls=0,
 limitations=[
 'The missing pre-resumption interval is not backfilled, interpolated, or given zero activity.',
 'Original native device activity is not a fraction of peak FLOPS. Memory bandwidth was not recorded.',
 'Only same-phase eight-rank observations bracketed by the original host observer enter pure-phase groups.',
 'The resumed observations start within iteration5, and do not constitute a complete iteration utilization measurement.',
 'XTT card1 and card8 report the same host pool and must not be added or combined with RSS.',
 'The native process tree excludes the external HTTP environment service; host MemAvailable includes the full host.'
 ])
(A/'current-native-observation-and-phase-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''))
report['collected_at_beijing'] = dt.datetime.fromtimestamp(
    report['collected'], dt.timezone(dt.timedelta(hours=8))).isoformat()
path = Path('experiments/rl/results_dtpo_v9_native_observation_20261019_v2.json')
path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
phases = []
for phase in report['phase_reduction']['record']['phases']:
    values = [x['gpu_utilization_percent'] for x in phase['devices'].values()]
    phases.append(dict(identity=phase['identity'],samples_per_card=[x['samples'] for x in values],
        mean_activity_range=[min(x['mean'] for x in values),max(x['mean'] for x in values)]))
print(json.dumps(dict(result=str(path),collected=report['collected_at_beijing'],
    handles=report['actual_handles'],phases=phases,
    checkpoint_observer_status=report['checkpoint_observer']['status']),ensure_ascii=False))
