"""Configure the existing native PPO resume launcher for the released GPU4/5."""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1/resume_native_PPO_after_diagnostic_v55.py'
code = source.read_text()
old = "stop=json.loads((OUT/'PPO-temporary-stop-for-v47-v54.json').read_bytes())\nassert stop['status']=='temporarily_stopped_native_full_restore_required'"
new = """if '--prepare' in sys.argv:
    stop=json.loads((OUT/'PPO-native-pause-for-GPU45-migration-v117.json').read_bytes())
    assert stop['status']=='planned_only_PPO_still_running'
else:
    stop=json.loads((OUT/'PPO-stop-for-GPU45-migration-v117.json').read_bytes())
    assert stop['status']=='temporarily_stopped_native_full_restore_required'"""
assert old in code
code = code.replace(old, new)
changes = {
    'native-PPO-resume-after-v47-v55': 'native-PPO-resume-GPU45-v117',
    "diag=OUT/'native-common-update-feedback-v47'\nresult=json.loads((diag/'result.json').read_bytes())\nassert result['phase']=='failed' or result['phase'].startswith('complete_'), 'Preserve running diagnostic'":
        "grpo_stop=json.loads((ROOT/'receipts/textcraft-grpo-to-ppo-20261015-v1/GRPO-user-stopped.json').read_bytes())\nassert grpo_stop['status']=='stopped_GRPO_at_user_request_checkpoints_and_logs_preserved' and not grpo_stop['survivors']",
    r"r'\|\s*[23]\s+\d+\s+'": r"r'\|\s*[45]\s+\d+\s+'",
    'GPU2/3 still occupied': 'GPU4/5 still occupied',
    '[(1419849,1791977138.31),(3174448,1791793629.73)]':
        '[(3140624,1791993570.92),(2614045,1791949034.31)]',
    "env['CUDA_VISIBLE_DEVICES']='2,3'": "env['CUDA_VISIBLE_DEVICES']='4,5'",
    "    prepared.write_text(json.dumps(report,indent=2)+'\\n')":
        "    report['effective_environment']=dict(report['effective_environment'],CUDA_VISIBLE_DEVICES='4,5')\n    report['physical_gpus']=[4,5]\n    report['execution_change']={'CUDA_VISIBLE_DEVICES':{'before':'2,3','after':'4,5'}}\n    prepared.write_text(json.dumps(report,indent=2)+'\\n')",
    '-v55.stdout.log': '-GPU45-v117.stdout.log',
    '-v55.stderr.log': '-GPU45-v117.stderr.log',
    'before-resume-v55.json': 'before-resume-GPU45-v117.json',
    'resume-v55-job.json': 'resume-GPU45-v117-job.json',
    "job.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),":
        "job.update(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),physical_gpus=[4,5],execution_version='textcraft-PPO-GPU45-migration-20261015-v1',GRPO_stopped_at_user_request=True,",
    "user_authorized_temporary_stop_and_restore=True,hourly_automation='PAUSED'":
        "user_authorized_temporary_stop_and_restore=True,user_authorized_device_migration=True,hourly_automation='ACTIVE_PLOTTING_ONLY'",
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
