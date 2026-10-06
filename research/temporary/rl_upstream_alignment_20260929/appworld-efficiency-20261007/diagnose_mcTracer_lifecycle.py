"""Bounded profiler-lifecycle reproduction; no model, RL, DT or checkpoint.

The official tracer is attached to only one of two identical tiny CUDA API
clients. This diagnoses the observed formal-worker exit; it is not a training,
numerical-tolerance or throughput test. GPU2 is checked before invoking this.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


AUDIT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("entry", AUDIT / "stage_environment_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

REMOTE = r'''
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
/opt/conda/bin/python - <<'PY'
import os,pty,termios,fcntl,select,subprocess,time,pathlib,json,hashlib
stamp=int(time.time())
base=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-efficiency-20261007')
out=base/('mcTracer-lifecycle-'+str(stamp));out.mkdir()
code="""import time,os,torch,json
x=torch.ones((128,128),device='cuda',dtype=torch.bfloat16)
torch.cuda.synchronize()
print(json.dumps(dict(ready_unix=time.time(),pid=os.getpid())),flush=True)
end=time.time()+45
while time.time()<end:
 y=x@x
 torch.cuda.synchronize()
 time.sleep(.02)
print(json.dumps(dict(completed_unix=time.time(),pid=os.getpid())),flush=True)
"""
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2')
children=[]
for name in ['control','attached']:
 f=(out/(name+'.stdout.txt')).open('w')
 e=(out/(name+'.stderr.txt')).open('w')
 p=subprocess.Popen([os.environ['VENV_PYTHON'],'-u','-c',code],env=env,stdout=f,stderr=e)
 children.append((name,p,f,e))
deadline=time.time()+25
while time.time()<deadline:
 if all((out/(name+'.stdout.txt')).stat().st_size for name,_,_,_ in children):break
 if any(p.poll() is not None for _,p,_,_ in children):break
 time.sleep(.2)
ready={name:(out/(name+'.stdout.txt')).read_text() for name,_,_,_ in children}
trace=None;failed_attempt=None
if all(ready.values()) and all(p.poll() is None for _,p,_,_ in children):
 pid=children[1][1].pid
 if FAILED_OUTPUT_FIRST:
  cmd=['/opt/maca/bin/mcTracer','--odname',str(out/'failed-absolute'),'--name','attached','--attach',str(pid)]
  tick=time.time();r=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=10)
  failed_attempt=dict(command=cmd,started_unix=tick,ended_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  # The formal observations had about 20s between the first failed init and retry.
  time.sleep(19)
 master,slave=pty.openpty()
 a=termios.tcgetattr(slave);a[3]&=~(termios.ICANON|termios.ECHO);termios.tcsetattr(slave,termios.TCSANOW,a)
 def setup():
  os.setsid();fcntl.ioctl(slave,termios.TIOCSCTTY,0)
 cmd=['/opt/maca/bin/mcTracer','--odname','trace','--name','attached','--attach',str(pid)]
 p=subprocess.Popen(cmd,cwd=out,stdin=slave,stdout=slave,stderr=slave,preexec_fn=setup)
 os.close(slave);os.set_blocking(master,False);buf=bytearray();begin=time.time();sent=None
 while p.poll() is None and time.time()-begin<18:
  if time.time()-begin>=5 and sent is None:os.write(master,b'\x14');sent=time.time()
  r,_,_=select.select([master],[],[],.1)
  if r:
   try:buf.extend(os.read(master,65536))
   except OSError:pass
 while True:
  try:
   b=os.read(master,65536)
   if not b:break
   buf.extend(b)
  except (BlockingIOError,OSError):break
 trace=dict(command=cmd,tracer_pid=p.pid,returncode=p.poll(),started_unix=begin,stop_sent_unix=sent,finished_unix=time.time(),console=buf.decode(errors='replace'))
 (out/'trace-console.txt').write_text(trace['console'])
 if p.poll() is not None:os.close(master)
observations=[];deadline=time.time()+50
while any(p.poll() is None for _,p,_,_ in children) and time.time()<deadline:
 observations.append(dict(unix=time.time(),returncodes={name:p.poll() for name,p,_,_ in children}))
 time.sleep(.5)
result=[]
for name,p,f,e in children:
 f.close();e.close()
 result.append(dict(name=name,pid=p.pid,returncode=p.poll(),stdout=(out/(name+'.stdout.txt')).read_text(),stderr=(out/(name+'.stderr.txt')).read_text()))
report=dict(observed_unix=time.time(),started_unix=stamp,scope='Two identical bounded tiny original Torch CUDA clients; official mcTracer attach/stop on one only. No model/DT/RL/optimizer/checkpoint or training-acceptance claim.',device=2,shape=[128,128],dtype='bfloat16',failed_attempt=failed_attempt,trace=trace,clients=result,observations=observations)
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
PY
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-output-first", action="store_true")
    args = parser.parse_args()
    script = REMOTE.replace("FAILED_OUTPUT_FIRST", repr(args.failed_output_first))
    result = subprocess.run(entry.SSH + ["bash", "-s"], input=script.encode(),
                            capture_output=True, timeout=110)
    if result.returncode:
        print(result.stderr.decode(errors="replace"))
        raise SystemExit(result.returncode)
    report = json.loads(result.stdout)
    target = Path(__file__).resolve().parent / "continuation-1791314906" / (
        "mcTracer-lifecycle-%d.json" % report["started_unix"])
    target.write_bytes(result.stdout)
    print(target)
    print(json.dumps({k: report[k] for k in ("trace", "clients")}, ensure_ascii=False))
