"""Validate serialized CSV values and switch only the local current-result pointer."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

payload=read(OUT/'prepared_tables.json')
checks=[]
for name,table in payload.items():
    with (OUT/name).open(newline='',encoding='utf-8') as f:actual=list(csv.reader(f))
    assert actual[0]==table['columns'] and len(actual)-1==len(table['rows'])
    for row,expected in zip(actual[1:],table['rows']):
        assert len(row)==len(expected)
        for got,want in zip(row,expected):
            if isinstance(want,(float,int)) and not isinstance(want,bool):assert float(got)==want
            else:assert got==('' if want is None else str(want))
    checks.append(dict(file=name,rows=len(actual)-1,sha256=sha(OUT/name)))

def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for name in ('task_metrics.csv','per_case.csv'):
    before=rows(BASE/'hotpot_positive_sum_v1'/name);after=rows(OUT/name)
    assert len(before)==len(after)
    for a,b in zip(before,after):
        if a['dataset']=='hotpotqa_long' and a['metric']=='recall':
            assert b['metric']=='complete_support'
            if not b['scope'].endswith('_budget_curve'):assert float(b['budget_fraction'])==.3
        else:assert a==b
all_methods=rows(OUT/'hotpotqa_all_methods.csv');cases=rows(OUT/'hotpotqa_per_case.csv')
for s in all_methods:
    sample=[r for r in cases if (r['model'],r['method'])==(s['model'],s['method'])]
    assert len(sample)==int(s['cases'])==48
    assert {int(r['index']) for r in sample}==set(range(48))
    count=sum(int(r['complete_support']) for r in sample)
    assert count==int(s['complete_cases']) and count/48==float(s['complete_support'])
    assert float(s['budget_fraction'])==.3
    assert all(int(r['spent_tokens'])<=int(r['budget_tokens']) for r in sample)

pointer=dict(version='hotpot-complete-support-30pct-v1',
    report='hotpot_complete_support_30_v1/hotpotqa_all_methods.csv',
    tables='hotpot_complete_support_30_v1/task_metrics.csv',
    per_case='hotpot_complete_support_30_v1/per_case.csv',
    hotpot_all_methods='hotpot_complete_support_30_v1/hotpotqa_all_methods.csv',
    hotpot_per_case='hotpot_complete_support_30_v1/hotpotqa_per_case.csv',
    receipt='hotpot_complete_support_30_v1/receipt.json',
    hotpot_metric='complete_support',hotpot_metric_display_name='Full-support Recall@30%',hotpot_primary_budget_fraction=.3,
    note='Local CSV records only. Hotpot reports complete support at 30% full body-token budget with token-positive sentence sums. Prior versions retained; paper unchanged.')
current=BASE/'latest_results.json'
assert read(current)==read(OUT/'previous_latest_results.json') or read(current)==pointer
dump(OUT/'latest_results.json',pointer)
temporary=BASE/'latest_results.json.tmp'
dump(temporary,pointer);temporary.replace(current)
receipt=dict(status='complete',created_at_utc=datetime.now(timezone.utc).isoformat(),
    model_calls=0,method_rows=14,method_case_rows=672,methods_each=dict(qwen3=9,qwen35=5),
    csv_roundtrip='all cells verified; other task/metric rows unchanged',
    checks=checks,protocol_sha256=sha(OUT/'protocol.json'),selections_sha256=sha(OUT/'selections.json'),
    latest_pointer_sha256=sha(current),previous_pointer_sha256=sha(OUT/'previous_latest_results.json'),
    history='User selected this metric and budget after examining the prior curves; historical files retained.')
dump(OUT/'receipt.json',receipt)
print(json.dumps(dict(status='complete',method_rows=14,per_case_rows=672,current_pointer=pointer),ensure_ascii=False))
