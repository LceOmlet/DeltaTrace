"""Collect the completed bounded observer; summarize original output metadata."""
import hashlib
import json
from pathlib import Path
import subprocess
from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote

receipt=ROOT+'/receipts/owner-b8-dispatch-20260930/appworld-rollout-scope/formal-cache'
remote(r'''source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,psutil
p=Path('@RECEIPT@');record=json.loads((p/'installed.json').read_text())
driver=psutil.Process(record['driver_pid'])
assert driver.create_time()==record['driver_created_unix']
for path in sorted(p.glob('rank*.json')):
 value=json.loads(path.read_text())
 assert value['restored'] and len(value['calls'])==8 and 'observation_error' not in value,value
 print(json.dumps(dict(rank=value['rank'],calls=len(value['calls']),restored=value['restored'])))
PY
'''.replace('@ENTRY@',ENTRY).replace('@RECEIPT@',receipt))
out=AUDIT/'appworld-rollout-scope-20261001/formal-cache';out.mkdir(exist_ok=True)
names=['submitted.json','installed.json','rank0.json','rank1.json',
       'client-signature-failure.json','client-signature-failure-keywords.json']
for name in names:
    subprocess.run(SCP+[f'{SSH[-1]}:{receipt}/{name}',str(out/name)],check=True)
art=lambda p:dict(path=p.relative_to(REPO).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
ranks=[json.loads((out/f'rank{n}.json').read_text()) for n in range(2)]
summary=dict(scope='Eight original formal AppWorld engine.generate calls per rank; no new generation or sampling changes. Generated token counts include original VERL DP padding.',
    sources=[art(out/n) for n in names],ranks=[],paired_call_diagnostics=[])
for rank in ranks:
    calls=rank['calls'];prompt=sum(c['prompt_tokens'] for c in calls);cached=sum(c['cached_tokens'] for c in calls)
    summary['ranks'].append(dict(rank=rank['rank'],pid=rank['pid'],calls=len(calls),restored=rank['restored'],
        prompt_tokens=prompt,cached_tokens=cached,cache_hit_fraction=cached/prompt,
        generated_tokens=sum(c['generated_tokens'] for c in calls),
        engine_seconds=sum(c['engine_seconds'] for c in calls),
        lora_ids=[list(ids) for ids in sorted({tuple(c['lora_ids']) for c in calls})],
        scope_open_in_all_calls=all(c['owner_scope_open'] for c in calls),
        requests_per_call=[c['requests'] for c in calls]))
for index,(a,b) in enumerate(zip(ranks[0]['calls'],ranks[1]['calls'])):
    summary['paired_call_diagnostics'].append(dict(index=index,
        inferred_start_difference_seconds=abs((a['unix']-a['engine_seconds'])-(b['unix']-b['engine_seconds'])),
        requests_per_rank=[a['requests'],b['requests']],
        max_rank_engine_seconds=max(a['engine_seconds'],b['engine_seconds']),
        min_rank_engine_seconds=min(a['engine_seconds'],b['engine_seconds'])))
rows=summary['paired_call_diagnostics'];single=[r for r in rows if r['requests_per_rank']==[1,1]]
summary['single_request_diagnostic']=dict(
    scope='Pairs follow observed order; start-time differences above expose alignment. Max rank engine time is not full RPC/rollout time.',
    calls=len(single),max_rank_engine_seconds=sum(r['max_rank_engine_seconds'] for r in single),
    all_max_rank_engine_seconds=sum(r['max_rank_engine_seconds'] for r in rows))
s=summary['single_request_diagnostic'];s['fraction']=s['max_rank_engine_seconds']/s['all_max_rank_engine_seconds']
summary['interpretation']='Real prefix cache reuse is present with stable per-rank LoRA IDs. Singleton batches and unequal per-rank completion remain measurable costs. These observations are not a controlled before/after speedup or a full-iteration result.'
summary['observer_restored']=all(r['restored'] for r in ranks)
summary['scripts']=[art(Path(__file__).resolve()),art(AUDIT/'observe_formal_rollout_cache.py')]
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:summary[k] for k in ['ranks','single_request_diagnostic','observer_restored']},indent=2))
