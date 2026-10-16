"""Thin reuse of the recorded strict SSH/SCP environment entry."""
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('entry', HERE.parents[2] / 'research/temporary/rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
AUDIT = entry.ROOT + '/receipts/PPO-32k-native-efficiency-20261016-v1'

if __name__ == '__main__':
    if sys.argv[1] == 'upload':
        entry.remote('mkdir -p ' + AUDIT + '/source\n')
        names = sys.argv[2:] or [p.name for p in HERE.glob('*.py')]
        for name in names:
            assert Path(name).name == name
            subprocess.run(entry.SCP + [str(HERE / name), entry.SSH[-1] + ':' + AUDIT + '/source/' + name], check=True)
    elif sys.argv[1] == 'run':
        name = sys.argv[2]
        assert Path(name).name == name and name.endswith('.py')
        entry.remote('source ' + entry.ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" ' + AUDIT + '/source/' + name + ' ' + ' '.join(shlex.quote(v) for v in sys.argv[3:]) + '\n')
    elif sys.argv[1] == 'download':
        dest = HERE / 'receipts'
        dest.mkdir(exist_ok=True)
        for name in sys.argv[2:]:
            assert Path(name).name == name
            subprocess.run(entry.SCP + [entry.SSH[-1] + ':' + AUDIT + '/' + name, str(dest / name)], check=True)
    else:
        raise ValueError(sys.argv[1])
