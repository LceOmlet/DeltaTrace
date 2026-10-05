"""Use the existing phase reader for the isolated original-order observation."""
import json
import subprocess

from observe_textcraft_native_readout import SCRIPT
from stage_textcraft_fla_orders import OUT, LOCAL
from stage_environment_entry import SSH, SCP


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(SSH + ['bash', '-s'],
        input=SCRIPT.replace('__OUT__', OUT).encode(), capture_output=True)
    result.check_returncode()
    observation = json.loads(result.stdout)
    path = LOCAL / f"observation-{int(observation['observed_unix'])}.json"
    path.write_text(json.dumps(observation, indent=2)+'\n')
    print(json.dumps(dict(path=str(path), completed=observation['completed'],
        alive=observation['same_process_alive'], host_available_bytes=observation['host_available_bytes'],
        ranks=[{key: row[key] for key in ('rank', 'phase', 'completed_cases', 'native_forward_calls')}
            for row in observation['ranks']], log_tail=observation['log_tail'][-8:]), indent=2))
    if observation['completed'] or not observation['same_process_alive']:
        names = ['job.json', 'prepared.json', 'native-owner-inspection.json', 'effective-config.yaml',
                 'diagnostic.log', 'rank0-readout.json', 'rank1-readout.json', 'physical-before-start.txt']
        if observation['completed']:
            names.append('completed.json')
        for name in names:
            fetch = subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], capture_output=True)
            if fetch.returncode:
                print(json.dumps(dict(unavailable=name, error=fetch.stderr.decode(errors='replace'))))
