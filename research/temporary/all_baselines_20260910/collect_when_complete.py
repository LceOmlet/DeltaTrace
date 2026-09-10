"""One-off collector: wait for this run, transfer it, then verify and report it."""
import argparse
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT.parent/'.analysis-deps'))
import paramiko

BASE='/tmp/codex_source_v2_gpu_20260910_v1'
PYTHON='/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python'
SCRIPTS=BASE+'/repo/research/temporary/all_baselines_20260910'

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);a=p.parse_args()
    client=paramiko.SSHClient();client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
    password=getpass.getpass('SSH password: ')
    client.connect('ssh.v5000-prod-gw.nhss.zhejianglab.com',port=a.port,username='root',password=password,
        look_for_keys=False,allow_agent=False,timeout=20,banner_timeout=30)
    del password
    client.get_transport().set_keepalive(15)
    def command(args,timeout=60):
        _,stdout,stderr=client.exec_command(shlex.join(args),timeout=timeout)
        output=stdout.read().decode();errors=stderr.read().decode();code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError(f'Remote command failed ({code}): {errors[-4000:]} {output[-4000:]}')
        return json.loads(output)
    try:
        prior=-1
        while True:
            state=command([PYTHON,SCRIPTS+'/progress.py','--base',BASE])
            if state['completed']!=prior:
                print(json.dumps(dict(stage='attribution',**state)),flush=True);prior=state['completed']
            if state['status']=='complete':break
            if state['status']=='failed' or not state['alive']:raise RuntimeError('Attribution requires attention: '+json.dumps(state))
            time.sleep(60)
        assert state['completed']==2240
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        name='baseline_full_'+stamp+'.zip';remote=BASE+'/'+name
        print(json.dumps(dict(stage='exporting')),flush=True)
        transfer=command([PYTHON,SCRIPTS+'/export_results.py','--base',BASE,'--output',remote])
        target=HERE.parent/'source_v2_gpu_20260910/all_baselines_transfer'/name
        target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        print(json.dumps(dict(stage='downloading',bytes=transfer['bytes'])),flush=True)
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(60)
            sftp.get(remote,str(target),max_concurrent_prefetch_requests=16)
        assert target.stat().st_size==transfer['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==transfer['sha256']
    finally:client.close()
    env=dict(os.environ,PYTHONPATH=str(HERE.parent/'needle_gap_20260910/.deps'))
    commands=[['import_results.py','--archive',str(target)],['analyze_all.py','--run',str(HERE/'raw')],
        ['build_report.py'],['verify_outputs.py']]
    for args in commands:
        print(json.dumps(dict(stage=args[0])),flush=True)
        subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]],cwd=ROOT,env=env,check=True)
    receipt=dict(status='all_baselines_collected_scored_reported_verified',completed=time.time(),archive=target.name,
        archive_sha256=transfer['sha256'],verification_sha256=hashlib.sha256((HERE/'verification.json').read_bytes()).hexdigest())
    (HERE/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
