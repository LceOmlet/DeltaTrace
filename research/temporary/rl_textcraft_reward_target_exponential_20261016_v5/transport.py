"""Use the recorded strict SSH/SCP entry for this version's small source files."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
AUDIT=ROOT+'/receipts/textcraft-reward-target-exponential-20261016-v5'
spec=importlib.util.spec_from_file_location('recorded_entry',REPO/'research/temporary/rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='upload':
        archive=HERE/'source-v5.tar'
        with tarfile.open(archive,'w') as tar:
            for path in sorted(HERE.glob('*.py')):tar.add(path,arcname=path.name)
            for path in sorted((HERE/'candidate-source').glob('*.py')):tar.add(path,arcname='candidate-source/'+path.name)
        entry.remote('mkdir -p '+AUDIT+'/source\n')
        subprocess.run(entry.SCP+[str(archive),entry.SSH[-1]+':'+AUDIT+'/source-v5.tar'],check=True)
        entry.remote('tar -xf '+AUDIT+'/source-v5.tar -C '+AUDIT+'/source\n')
    elif action=='prepare':
        entry.remote('source '+entry.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" '+AUDIT+'/source/prepare_remote_v5.py\n')
    elif action=='run':
        # The file chooses CPU or native GPU resources; launch env is merged
        # only after the recorded provisioned environment has been sourced.
        filename=sys.argv[2]
        assert Path(filename).name==filename and filename.endswith('.py')
        args=sys.argv[3:]
        import shlex
        script='import json,os,subprocess\nfrom pathlib import Path\np=json.loads(Path('+repr(AUDIT+'/prepared-launch.json')+').read_bytes())\nenv=dict(os.environ);env.update(p["effective_environment"])\nenv["CUDA_VISIBLE_DEVICES"]=""\nsubprocess.run([env["VENV_PYTHON"],'+repr(AUDIT+'/source/'+filename)+']+'+repr(args)+',env=env,check=True)\n'
        entry.remote('source '+entry.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+script+'\nPY\n')
    elif action=='download':
        destination=HERE/'receipts';destination.mkdir(exist_ok=True)
        for name in sys.argv[2:]:
            assert Path(name).name==name
            subprocess.run(entry.SCP+[entry.SSH[-1]+':'+AUDIT+'/'+name,str(destination/name)],check=True)
    else:raise ValueError(action)
