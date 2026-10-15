"""Reuse original plotting with an explicitly recorded temporarily paused PPO.

All curves still come from original read_job_logs/parse_logs. The existing
authorized pause affects process status, not validity of completed log points.
"""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('plot_current_approved_DTPO_v63.py')
expected = 'f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
old = """job=json.loads(job_path.read_bytes());p=psutil.Process(job['pid'])
assert p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE"""
new = """job=json.loads(job_path.read_bytes())
try:
    p=psutil.Process(job['pid'])
    PPO_alive=p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:
    PPO_alive=False
if not PPO_alive:
    pause_path=OUT/'PPO-temporary-stop-for-saved-update-v90-v91.json'
    pause=json.loads(pause_path.read_bytes())
    assert pause['status']=='temporarily_stopped_native_full_restore_required'
    assert not pause['survivors']
    assert (pause['job']['pid'],pause['job']['birth'])==(job['pid'],job['birth'])
    snapshot['PPO_temporary_pause']=dict(receipt=str(pause_path),completed_iteration=pause['completed_iteration'],native_full_restore_required=True)"""
assert old in code
code = code.replace(old, new)
old = "pid=job['pid'],observed_process_created_unix=job['birth'],pid_alive=True,"
assert old in code
code = code.replace(old, "pid=job['pid'],observed_process_created_unix=job['birth'],pid_alive=PPO_alive,")
old = "    fig.text(.075,.942,subtitle,color='#475569',fontsize=9.5)"
assert old in code
code = code.replace(old, """    if snapshot.get('PPO_temporary_pause'):
        subtitle+=' · PPO 临停诊断，保留已完成曲线'
    fig.text(.075,.942,subtitle,color='#475569',fontsize=9.5)""")
exec(compile(code, str(source), 'exec'), globals())
