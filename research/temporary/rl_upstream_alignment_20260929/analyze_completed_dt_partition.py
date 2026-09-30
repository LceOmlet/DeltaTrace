"""Compare native partitioning on lengths from completed formal DT reports."""
from stage_environment_entry import remote, ROOT, ENTRY

remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import importlib.util, json, hashlib
import torch
torch.set_num_threads(1)
root=Path('@ROOT@');active=json.loads((root/'active-training.json').read_text())
out=root/'receipts/owner-b8-dispatch-20260930/dt-wait-profile'
records=[]
for job in active['jobs']:
    if job['task'] not in ('AppWorld','TextCraft'):continue
    path=Path(job['verl_root'])/'verl/utils/seqlen_balancing.py'
    spec=importlib.util.spec_from_file_location('owner_balance',path)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    ranks=[];sources=[]
    for log in sorted(Path('/tmp/ray').glob(f"session_*_{job['pid']}/logs/worker*.out")):
        reports=[]
        with log.open(errors='replace') as stream:
            for line in stream:
                if line.startswith('[DeltaTrace readout] '):
                    r=json.loads(line.split('] ',1)[1])
                    if r.get('traces'):reports.append(r)
        if reports:ranks.append(reports);sources.append(str(log))
    if len(ranks)!=2:continue
    for idx in range(min(map(len,ranks))):
        reports=[rank[idx] for rank in ranks]
        lengths=[[t['context_tokens'] for t in r['traces']] for r in reports]
        if len(lengths[0]) != len(lengths[1]) or len(lengths[0])%4:continue
        flat=sorted(sum(lengths,[]))
        partitions=owner.get_seqlen_balanced_partitions(flat,2,True)
        balanced=[sorted(flat[i] for i in indices) for indices in partitions]
        def cost(rows):
            maxima=[[max(rank[i:i+4]) for i in range(0,len(rank),4)] for rank in rows]
            return dict(rank_tokens=[sum(rank) for rank in rows],
                padded_tokens=sum(sum(rank)*4 for rank in maxima),
                synchronized_length_sum=sum(max(pair) for pair in zip(*maxima)),
                largest_batch_length=[max(rank) for rank in maxima])
        before,after=cost(lengths),cost(balanced)
        records.append(dict(task=job['task'],driver_pid=job['pid'],group_index=idx,
            sources=sources,owner_file=str(path),owner_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows_per_rank=len(lengths[0]),seconds=[r['seconds'] for r in reports],
            before=before,official_balanced=after,
            synchronized_length_reduction=1-after['synchronized_length_sum']/before['synchronized_length_sum'],
            scope='Actual completed request lengths, CPU owner partition only; length sum is a workload proxy, not measured speedup. Current training unchanged.'))
(out/'completed-partition-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY))
