/opt/conda/bin/python - <<'PY'
import hashlib,json,os,pathlib,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-official-whitening-20261006-v1');p=out/'source/experiments/rl/test_rollout_credit.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='ec491fc4db8a3f4d295c22d99a0515ba4bf2b2e61f7112c1456c691365071319'
x=json.loads((out/'recovered-runtime-environment.json').read_bytes());e=dict(os.environ);e.update(x['environment'])
source=json.loads((out/'candidate-source.json').read_bytes());e.update(VERL_ROOT=source['candidate'],CUDA_VISIBLE_DEVICES='')
e['PYTHONPATH']=e['VERL_ROOT']+':'+e['PYTHONPATH']
for k in ('RAY_ADDRESS','MACA_VISIBLE_DEVICES','RAY_TMPDIR'):e.pop(k,None)
t=time.monotonic()
with (out/'collector-owner-tests.stdout.txt').open('wb') as log:
 q=subprocess.run([e['VENV_PYTHON'],'-m','pytest','-q',str(p),'--junitxml='+str(out/'collector-owner-tests.xml')],cwd=out,env=e,stdout=log,stderr=subprocess.STDOUT)
y=dict(scope='Existing collector/trainer credit transport unit tests; fixtures are interface data only, not attribution/learning-quality validation.',source={'path':str(p),'sha256':'ec491fc4db8a3f4d295c22d99a0515ba4bf2b2e61f7112c1456c691365071319'},returncode=q.returncode,elapsed_seconds=time.monotonic()-t,cuda_visible_devices='',VERL_ROOT=e['VERL_ROOT'],sources=source['source_sha256'])
(out/'collector-owner-test-receipt.json').write_text(json.dumps(y,indent=2)+'\n')
print(json.dumps(y));print((out/'collector-owner-tests.stdout.txt').read_text()[-4000:]);assert q.returncode==0
PY
