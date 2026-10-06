"""Observe only this isolated query-wording job, retaining raw artifacts."""
import json
import subprocess
from stage_environment_entry import SSH, SCP
from stage_textcraft_response_clock import OUT, LOCAL
from observe_textcraft_label_encoding import SCRIPT

if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(SSH + ['bash', '-s'], input=SCRIPT.replace('__OUT__', OUT).encode(), capture_output=True)
    result.check_returncode()
    value = json.loads(result.stdout)
    path = LOCAL / f"observation-{int(value['observed_unix'])}.json"
    path.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), alive=value['same_process_alive'], completed=value['completed'],
        host_available_bytes=value['host_available_bytes'], phases=[dict(file=p['path'].rsplit('/',1)[-1],
        phase=p['value']['phase'], cases=len(p['value'].get('cases',[])), calls=p['value'].get('native_forward_calls')) for p in value['phases']], log_tail=value['log_tail'][-8:]), indent=2))
    if value['completed'] or not value['same_process_alive']:
        suffixes = [suffix for suffix in ('.json','.yaml','.log','.txt') if any(p['name'].endswith(suffix) for p in value['artifacts'])]
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/*{suffix}' for suffix in suffixes]+[str(LOCAL)+'/'], capture_output=True, check=True)
