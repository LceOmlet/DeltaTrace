"""Bounded default parity and original convolution graph check; no model/DT runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH, SCP, ROOT, ENTRY

OUT = ROOT+'/receipts/conditional-owner-seams-20261009-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    if args.observe:
        template = (HERE/'native-context-profile-observe-command.sh').read_bytes()
        command = template.replace((ROOT+'/receipts/native-context-profile-20261009-v1').encode(), OUT.encode())
        name = 'conditional-owner-seams-observe-'+str(int(time.time()))
    else:
        names = ['check_conditional_owner_seams.py','conditional_gdn_context.py',
                 'conditional_conv_windows.py','check_conditional_conv_windows.py',
                 'tiled_conditional_memory.py','conditional_window_memory.py','native_conditional_queries.py',
                 'conditional-gdn-owner-prepared/qwen35_gdn_finite.py','conditional-gdn-owner-prepared/finite_fla_gpu.py']
        subprocess.run(SSH+['mkdir', '-p', OUT], check=True, timeout=20)
        subprocess.run(SCP+[str(HERE/name) for name in names]+[SSH[-1]+':'+OUT+'/'], check=True, timeout=30)
        files = {Path(name).name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        body = '''import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(OUT)
for name,digest in FILES.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
assert not (out/'launch.json').exists(), 'Inspect the original job; do not duplicate it'
physical=subprocess.run(['mx-smi'],text=True,capture_output=True,check=True).stdout
device=next(i for i in (4,5) if not re.search(r'^\\|\\s*'+str(i)+r'\\s+\\d+\\s+\\S',physical,re.M))
source=Path(ROOT)/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
s=json.loads(source.read_bytes())
env=dict(os.environ,**s['environment']);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']=str(device)
env['PYTHONPATH']=':'.join([str(out),s['pythonpath'],str(Path(s['dt_root'])/'clean/qwen35')])
argv=['timeout','--signal=TERM','240',env['VENV_PYTHON'],str(out/'check_conditional_owner_seams.py'),
 '--original_fla',str(Path(s['dt_root'])/'clean/qwen35/finite_fla_gpu.py'),
 '--original_query',ROOT+'/receipts/native-conditional-queries-20261009-v1',
 '--original_conv',ROOT+'/receipts/conditional-conv-windows-20261009-v1',
 '--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),argv=argv,script_sha256=FILES,
 base_commit=COMMIT,source_json_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 devices=[device],launched_unix=time.time(),wall_bound_seconds=240,
 model_calls=0,DT_calls=0,optimizer=0,formal_restart=False,production_modified=False,physical_before=physical)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''
        command = ('source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
                   +'ROOT='+repr(ROOT)+'\nOUT='+repr(OUT)+'\nFILES='+repr(files)+'\nCOMMIT='+repr(commit)+'\n'
                   +body+'\nPY\n').encode()
        name = 'conditional-owner-seams-launch'
    (HERE/(name+'-command.sh')).write_bytes(command)
    result = subprocess.run(SSH+['bash', '-s'], input=command, capture_output=True, timeout=35)
    (HERE/(name+'.stdout')).write_bytes(result.stdout)
    (HERE/(name+'.stderr')).write_bytes(result.stderr)
    result.check_returncode()
    value = json.loads(result.stdout)
    (HERE/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    if args.observe:
        if value.get('result'):
            (HERE/'conditional-owner-seams-result.json').write_text(json.dumps(value['result'], indent=2)+'\n')
        print(json.dumps(dict(alive=value['driver_alive'], log=value['log'][-2000:],
                              result=value.get('result'))))
    else:
        print(json.dumps({k:value[k] for k in ('pid', 'birth', 'devices', 'script_sha256')}))


if __name__ == '__main__':
    main()
