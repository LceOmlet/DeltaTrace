source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-native-identity-operators-appworld-20261009-v2'
SHA='ac22648a404497ecc785ca54685dabe9565ed864363152927e64410ae22c8b2d'
COMMIT='632037447306ae4d55a3782004801f4044f3f48c'
import json,hashlib,subprocess,os,psutil,time
from pathlib import Path
out=Path(OUT);script=out/'check_native_linear_roundoff.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==SHA
assert (out/'results/completed.json').exists()
assert not (out/'linear-check-launch.json').exists()
argv=[os.environ['VENV_PYTHON'],'-u',str(script),'--folder',str(out/'results'),'--output',str(out/'results/linear-roundoff.json')]
with (out/'linear-check.log').open('xb') as stream:
 process=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=process.pid,birth=psutil.Process(process.pid).create_time(),unix=time.time(),script_sha256=SHA,code_commit=COMMIT,argv=argv,GPU=False,model=False,production_modified=False)
(out/'linear-check-launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
