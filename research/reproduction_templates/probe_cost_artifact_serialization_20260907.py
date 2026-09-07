"""Measure standard JSON audit checkpoint overhead; zero model/GPU calls."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=Path('${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1/results.json')
raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
assert digest=='f0bf315cbe7422bc5827182eb12905b545f2b34356cddcf0e33af39ac69ec679'
d=json.loads(raw);assert d['status']=='complete';del raw
out={'scope':'Single final-report standard JSON encode+atomic-write comparison on the experiment host. Does not time model computation, alter study results, or directly apportion all historical elapsed overhead.','source_sha256':digest,'model_calls':0,'records':[]}
for name,options in [('pretty',{'indent':2}),('compact',{'separators':(',',':')})]:
    started=time.perf_counter();value=json.dumps(d,ensure_ascii=False,**options);encoded=time.perf_counter()
    target=HERE/(name+'.json');temp=HERE/(name+'.partial')
    temp.write_text(value,encoding='utf-8');temp.replace(target);written=time.perf_counter()
    assert json.loads(target.read_text())==d
    out['records'].append({'format':name,'encoding_seconds':encoded-started,'atomic_write_seconds':written-encoded,'total_seconds':written-started,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'decoded_payload_identical':True})
    del value
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
(HERE/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
