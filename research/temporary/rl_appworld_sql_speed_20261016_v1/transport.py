"""Use the existing strict SSH/SCP transport for isolated speed-trial sources."""
import importlib.util
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('recorded_entry',REPO/'research/temporary/rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
AUDIT=entry.ROOT+'/receipts/appworld-sql-native-speed-20261016-v1'
if __name__=='__main__':
    if sys.argv[1]=='upload':
        entry.remote('mkdir -p '+AUDIT+'/source\n')
        for path in HERE.glob('*.py'):
            subprocess.run(entry.SCP+[str(path),entry.SSH[-1]+':'+AUDIT+'/source/'+path.name],check=True)
    elif sys.argv[1]=='run':
        name=sys.argv[2];assert Path(name).name==name and name.endswith('.py')
        import shlex
        args=' '.join(shlex.quote(v) for v in sys.argv[3:])
        entry.remote('source '+entry.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" '+AUDIT+'/source/'+name+' '+args+'\n')
    elif sys.argv[1]=='download':
        dest=HERE/'receipts';dest.mkdir(exist_ok=True)
        for name in sys.argv[2:]:
            assert Path(name).name==name
            subprocess.run(entry.SCP+[entry.SSH[-1]+':'+AUDIT+'/'+name,str(dest/name)],check=True)
    else:raise ValueError(sys.argv[1])
