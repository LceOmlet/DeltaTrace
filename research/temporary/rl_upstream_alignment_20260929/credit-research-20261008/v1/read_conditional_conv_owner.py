"""Read the provisioned owners without GPU work or runtime changes."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH, SCP, ROOT, ENTRY


def main():
    remote = ROOT+'/receipts/conditional-conv-owner-20261009-v1'
    subprocess.run(SSH+['mkdir', '-p', remote], check=True, timeout=25)
    subprocess.run(SCP+[str(HERE/'inspect_conditional_conv_owner.py'), SSH[-1]+':'+remote+'/'],
                   check=True, timeout=30)
    bootstrap = '''import hashlib,json,os,re,subprocess
from pathlib import Path
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
device=next(i for i in (4,5) if not re.search(r'^\\|\\s*'+str(i)+r'\\s+\\d+\\s+\\S',physical,re.M))
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']=str(device)
env['PYTHONPATH']=':'.join([s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
out=Path(OUT)
(out/'inspection-environment.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 device=device,python=env['VENV_PYTHON'],pythonpath=env['PYTHONPATH'],physical_before=physical,
 model_calls=0,DT_calls=0,optimizer=0),indent=2)+'\\n')
subprocess.run([env['VENV_PYTHON'],str(out/'inspect_conditional_conv_owner.py'),
 '--root',ROOT,'--output',str(out/'owner.json')],env=env,check=True)
'''
    command = ('source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
               +'ROOT='+repr(ROOT)+'\nOUT='+repr(remote)+'\n'+bootstrap+'\nPY\n')
    (HERE/'conditional-conv-owner-command.sh').write_bytes(command.encode())
    try:
        result = subprocess.run(SSH+['bash', '-s'], input=command.encode(), capture_output=True, timeout=45)
    except subprocess.TimeoutExpired as error:
        (HERE/'conditional-conv-owner.stdout.txt').write_bytes(error.stdout or b'')
        (HERE/'conditional-conv-owner.stderr.txt').write_bytes(error.stderr or b'')
        (HERE/'conditional-conv-owner-timeout.json').write_text(
            '{"remote_outcome":"unknown","reason":"local SSH timeout; inspect remote before relaunch"}\n')
        raise
    (HERE/'conditional-conv-owner.stdout.txt').write_bytes(result.stdout)
    (HERE/'conditional-conv-owner.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    print(result.stdout.decode())
    subprocess.run(SCP+[SSH[-1]+':'+remote+'/owner.json',str(HERE/'conditional-conv-owner.json')],
                   check=True, timeout=30)


if __name__ == '__main__':
    main()
