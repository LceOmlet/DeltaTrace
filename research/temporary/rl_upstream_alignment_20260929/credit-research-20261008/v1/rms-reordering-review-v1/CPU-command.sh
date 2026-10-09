exec /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python - <<'PY'
from pathlib import Path
import os,json,hashlib,subprocess,psutil,time
target=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1')
assert not (target/'launch.json').exists()
files={'check_rms_reordering.py': '59c94a1082e5ddb582339e6792b72f57be9e2001a4d132130f12e41d76eea233', 'original-signed_secant_rules.py': '056d576e31b7076e7a08c89a5f25fde86897cfdf9daa3988d0a20acd6401030d', 'candidate-signed_secant_rules.py': 'b03b8d7761d9d24a303a87d7dea894b21187ed7b5438351140ae619816b3689a', 'prepared.json': '190f1fa5b84fb21dae9874adc6ada005b1d53a1adf836e40181a4b5f870a49d9'}
for name,h in files.items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==h
assert psutil.virtual_memory().available>4*(1<<30)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (target/'stdout').open('xb') as out,(target/'stderr').open('xb') as err:
 p=subprocess.Popen(['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python', '-u', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/check_rms_reordering.py', '--original', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/original-signed_secant_rules.py', '--candidate', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/candidate-signed_secant_rules.py', '--precast', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt', '--output', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/result.json'],env=env,stdout=out,stderr=err)
 record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit='12a1b24e6715c6be6b5448b7f870daa4ffea5cce',files=files,argv=['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python', '-u', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/check_rms_reordering.py', '--original', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/original-signed_secant_rules.py', '--candidate', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/candidate-signed_secant_rules.py', '--precast', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-single-background-nonfinite-appworld-20261009-v3/results/rank0-precast-seed.pt', '--output', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/rms-normalized-secant-review-20261009-v1/result.json'],operations=dict(model=0,DT=0,FA=0,FLA=0,GPU=0,optimizer=0),production_modified=False)
 (target/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
 record['returncode']=p.wait(timeout=90)
 record['elapsed_seconds']=time.time()-record['unix']
 (target/'completion.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
assert record['returncode']==0

PY
