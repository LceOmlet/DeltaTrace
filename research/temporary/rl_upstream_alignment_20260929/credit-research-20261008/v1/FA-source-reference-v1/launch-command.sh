set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-FA-source-reference-20261009-v1'); hashes={'inspect_saved_FA_source_reference.py': 'c8abce2df7822ae0a41a3bed26385cef4042cbe3ab3aec2f826bad47f4fd9499', 'protocol.json': '20fe106339e7cc4eac6f916266b90b49f201837e19cfbfe1ea7d39e60ac445b5'}
assert not (out/'launch.json').exists(),'Do not duplicate diagnostic'
for name,digest in hashes.items():
 path=out/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 if name.endswith('.py'):ast.parse(path.read_bytes())
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s*4\s+\d+\s+\S',physical,re.M),'GPU4 occupied'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
env.pop('MACA_VISIBLE_DEVICES',None)
env['PYTHONPATH']='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35'
argv=['timeout','600',env['VENV_PYTHON'],'-u',str(out/'inspect_saved_FA_source_reference.py'),
 '--directory','/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-pv-textcraft-20261009-v2','--owner','/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1/deltatrace/clean/qwen35/vendor_fa_finite_bf16_d256.py','--library','/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/libfinite_row_query_starts.so','--protocol',str(out/'protocol.json'),'--output',str(out/'result.json')]
with (out/'driver.log').open('xb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit='97039cf00020764b0119a040352ec83b49894271',
 files=hashes,argv=argv,devices=[4],physical_before=physical,
 host_available_bytes=psutil.virtual_memory().available,production_modified=False,training_candidate=False)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

PY
