"""Persist original bounded-diagnostic observations before parsing/reporting."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH

CASES = {
    'queries':'native-conditional-queries',
    'memory-limit':'conditional-memory-limit',
    'memory-finite':'conditional-memory-finite',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=tuple(CASES))
    args = parser.parse_args()
    prefix = CASES[args.case]
    folder = HERE/(prefix+'-observations')
    folder.mkdir(exist_ok=True)
    path = folder/(str(int(time.time()))+'.json')
    template = (HERE/'native-context-profile-observe-command.sh').read_bytes()
    old = b'/receipts/native-context-profile-20261009-v1'
    assert template.count(old) == 1
    command = template.replace(old, ('/receipts/'+prefix+'-20261009-v1').encode())
    (HERE/(prefix+'-observe-command.sh')).write_bytes(command)
    try:
        result = subprocess.run(SSH+['bash', '-s'], input=command, capture_output=True, timeout=35)
        path.with_suffix('.stdout').write_bytes(result.stdout)
        path.with_suffix('.stderr').write_bytes(result.stderr)
        result.check_returncode()
        observation = json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        path.write_text(json.dumps(dict(local_observation_error=str(error),
            read_only=True, remote_diagnostic_outcome='unknown'), indent=2)+'\n')
        raise
    path.write_text(json.dumps(observation, indent=2)+'\n')
    outcome = observation.get('result')
    if outcome:
        (HERE/(prefix+'-result.json')).write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps(dict(observation=str(path), alive=observation['driver_alive'],
        outcome=outcome, log_tail=observation['log'][-2000:])))


if __name__ == '__main__':
    main()
