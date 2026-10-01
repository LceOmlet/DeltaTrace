"""Summarize the original vLLM trace without adding nested timings together."""
from stage_environment_entry import ROOT, ENTRY, AUDIT, SSH, SCP, remote
import subprocess

remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,re
root=Path('@ROOT@');out=root/'receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/native-profiler'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def union_seconds(rows):
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in rows)
    if not intervals:return 0.
    total=0.;start,end=intervals[0]
    for a,b in intervals[1:]:
        if a>end:total+=end-start;start,end=a,b
        else:end=max(end,b)
    return (total+end-start)/1e6
def top(rows,count=15):
    totals=defaultdict(float)
    for e in rows:totals[e['name']]+=e['dur']/1e6
    return [dict(name=k,inclusive_seconds=v) for k,v in sorted(totals.items(),key=lambda x:x[1],reverse=True)[:count]]
result=dict(scope='First 16 original worker iterations in one existing formal batch/rank. Native GPU annotations and nested CPU operations are not added to kernel timings. No full-iteration or unprofiled speedup claim.',ranks=[])
for rank in (0,1):
    receipt=out/f'rank{rank}.json';record=json.loads(receipt.read_text())
    assert record['restored'] and all(x['restored'] and not x['running_after_stop'] for x in record['finish'])
    trace=next((out/f'rank{rank}').glob('*.json.gz'))
    events=json.loads(gzip.decompress(trace.read_bytes()))['traceEvents']
    rows=[e for e in events if e.get('ph')=='X' and 'dur' in e]
    category=lambda name:[e for e in rows if e.get('cat')==name]
    kernels=category('kernel');copies=category('gpu_memcpy');runtime=category('cuda_runtime')
    phases=[]
    for e in category('gpu_user_annotation'):
        match=re.fullmatch(r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)',e['name'])
        if match:
            context,context_tokens,generation,generation_tokens=map(int,match.groups())
            phases.append(dict(name=e['name'],context_requests=context,context_tokens=context_tokens,
                generation_requests=generation,generation_tokens=generation_tokens,gpu_range_seconds=e['dur']/1e6))
    phases.sort(key=lambda x:(-x['context_tokens'],x['name']))
    native_gpu_ranges=category('gpu_user_annotation')
    host_copies=[e for e in rows if e.get('cat')=='cpu_op' and e['name']=='aten::copy_']
    result['ranks'].append(dict(rank=rank,receipt=str(receipt),receipt_sha256=sha(receipt),
        trace=str(trace),trace_sha256=sha(trace),trace_bytes=trace.stat().st_size,
        trace_event_count=len(events),native_phases=phases,
        kernel_sum_seconds=sum(e['dur'] for e in kernels)/1e6,
        kernel_union_seconds=union_seconds(kernels),gpu_copy_union_seconds=union_seconds(copies),
        gpu_compute_and_copy_union_seconds=union_seconds(kernels+copies),
        native_gpu_range_union_seconds=union_seconds(native_gpu_ranges),
        host_copy_sum_seconds=sum(e['dur'] for e in host_copies)/1e6,
        host_copy_union_across_threads_seconds=union_seconds(host_copies),
        copy_event_count=len(copies),copy_event_argument_example=copies[0].get('args') if copies else None,
        kernel_top=top(kernels),runtime_top=top(runtime),runtime_union_across_threads_seconds=union_seconds(runtime),
        copy_top=top(copies),cpu_top_inclusive=top(category('cpu_op'))))
    del events,rows
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
local=AUDIT/'appworld-rollout-scope-20261001'/'native-profiler';local.mkdir(exist_ok=True)
base=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/native-profiler'
for name in ['preflight.json','submitted.json','installed.json','rank0.json','rank1.json','summary.json',
             'rank0/profiler_out_0.txt','rank1/profiler_out_0.txt']:
    target=local/name;target.parent.mkdir(exist_ok=True)
    subprocess.run(SCP+[f'{SSH[-1]}:{base}/{name}',str(target)],check=True)
