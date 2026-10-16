"""Use the original trainer log parser on this version's isolated runs."""
import hashlib
import json
from pathlib import Path
import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-native-ppo-speed-20261016-v1'
links_file = AUDIT / 'native-Ray-session-links.json'
LINKS = json.loads(links_file.read_bytes()) if links_file.exists() else {}
state = json.loads((AUDIT / 'trial-status.json').read_bytes())
for job in state.get('tasks', []):
    try:
        driver = psutil.Process(job['pid'])
        if driver.create_time() != job['birth']:
            continue
        gcs = next((p for p in driver.children(recursive=True) if p.name() == 'gcs_server'), None)
        if gcs is None:
            continue
        log_dir = next((arg.split('=', 1)[1] for arg in gcs.cmdline() if arg.startswith('--log_dir=')), None)
        if log_dir is None:
            continue
        session = Path(log_dir).parent
        assert session.name.startswith('session_') and Path(log_dir).name == 'logs'
        LINKS[str(job['pid'])] = dict(birth=job['birth'], ray_session=str(session),
                                     source='Exact --log_dir of this driver descendant gcs_server',
                                     gcs_pid=gcs.pid, gcs_birth=gcs.create_time())
    except psutil.NoSuchProcess:
        pass
links_file.write_text(json.dumps(LINKS, indent=2) + '\n')
owner = ROOT / 'receipts/appworld-sql-native-speed-20261016-v1/source/collect_status.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '8e1e9754033645b87aa205b91f632bce48d67702d6d6d54f494878c29a71bad6'
code = owner.read_text().replace('appworld-sql-native-speed-20261016-v1', 'appworld-native-ppo-speed-20261016-v1')
old = "    records,errors=owner['read_job_logs'](current)"
new = """    link=LINKS.get(str(current['pid']))
    if link is not None:
        assert link['birth']==current['birth']
        current=dict(current,ray_session=link['ray_session'])
    records,errors=owner['read_job_logs'](current)"""
assert code.count(old) == 1
code = code.replace(old, new)
old = 'source_log=str(path)))'
assert code.count(old) == 1
code = code.replace(old, "source_log=str(path),ray_session=current.get('ray_session')))")
exec(compile(code, str(owner), 'exec'), globals())
