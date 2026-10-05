"""Reuse original read-only job observation for matched root diagnostics."""
import json
import subprocess

import observe_textcraft_native_readout as existing
from stage_textcraft_matched_layout import LOCAL, OUT


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(existing.SSH + ['bash', '-s'],
        input=existing.SCRIPT.replace('__OUT__', OUT).encode(), capture_output=True)
    result.check_returncode()
    data = json.loads(result.stdout)
    path = LOCAL / f"observation-{int(data['observed_unix'])}.json"
    path.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), completed=data['completed'], alive=data['same_process_alive'],
        host_available=data['host_available_bytes'],
        ranks=[dict(rank=row['rank'], phase=row['phase'],
            completed_root_observations=row['completed_cases'], native_forward_calls=row['native_forward_calls'])
            for row in data['ranks']], log_tail=data['log_tail'][-8:]), indent=2))
    if data['completed']:
        for name in ['job.json', 'prepared.json', 'completed.json', 'native-owner-inspection.json',
                     'effective-config.yaml', 'rank0-readout.json', 'rank1-readout.json', 'diagnostic.log',
                     'physical-before-start.txt']:
            subprocess.run(existing.SCP + [f'{existing.SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
