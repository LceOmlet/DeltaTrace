"""Locate saved prefix-layout differences without another model call."""
import argparse
import subprocess
from stage_environment_entry import AUDIT, ENTRY, ROOT, SCP, SSH, remote

parser=argparse.ArgumentParser()
parser.add_argument('receipt_name')
args=parser.parse_args()
assert args.receipt_name.startswith('native-dt-gathers-') and '/' not in args.receipt_name
path=ROOT+'/receipts/owner-b8-dispatch-20260930/'+args.receipt_name
remote(fr'''source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,torch
from pathlib import Path
p=Path('{path}')
records=json.loads((p/'completed.json').read_text())['modes']['prefix_diagnostic']
load=lambda n:torch.load(n,map_location='cpu',weights_only=True)
bykey={{(r['rank'],r['label']):r for r in records}}
def difference(a,b):
 a=a.double();b=b.double();assert a.shape==b.shape
 return dict(exact_equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()),
  relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)))
out=dict(scope='Same saved B4/rank, three calls in one RPC. Layer observations are diagnostics, not a new tolerance.',cases=[])
for rank in (0,1):
 first=bykey[rank,'native_first']
 for label in ['observed_cut_320','native_repeat']:
  actual=bykey[rank,label]
  row=dict(rank=rank,reference='native_first',actual=label,
   prefix=[first['prefix_length'],actual['prefix_length']],
   outputs={{k:difference(v,load(actual['output_path'])[k]) for k,v in load(first['output_path']).items()}},layers=[])
  a,b=load(first['root_samples']),load(actual['root_samples'])
  row['layers']=[dict(layer=int(k),**difference(a[k],b[k])) for k in sorted(a,key=int)]
  row['first_changed_root_layer']=next((x['layer'] for x in row['layers'] if not x['exact_equal']),None)
  row['target_logp0_difference']=difference(torch.tensor(first['dt_info'][0]['target_logp0']),torch.tensor(actual['dt_info'][0]['target_logp0']))
  row['target_logp1_difference']=difference(torch.tensor(first['dt_info'][0]['target_logp1']),torch.tensor(actual['dt_info'][0]['target_logp1']))
  out['cases'].append(row)
out['same_inputs_across_ranks']=[]
for label in ['native_first','observed_cut_320','native_repeat']:
 a,b=(bykey[i,label] for i in (0,1))
 roots=[load(r['root_samples']) for r in (a,b)]
 out['same_inputs_across_ranks'].append(dict(label=label,
  outputs={{k:difference(v,load(b['output_path'])[k]) for k,v in load(a['output_path']).items()}},
  layers=[dict(layer=int(k),**difference(roots[0][k],roots[1][k])) for k in sorted(roots[0],key=int)]))
(p/'prefix-summary.json').write_text(json.dumps(out,indent=2)+'\n')
for row in out['cases']:
 print(json.dumps({{k:v for k,v in row.items() if k!='layers'}}))
print('cross-rank',json.dumps([dict(label=r['label'],outputs=r['outputs'],first_changed_layer=next((x['layer'] for x in r['layers'] if not x['exact_equal']),None)) for r in out['same_inputs_across_ranks']]))
PY
''')
out=AUDIT/args.receipt_name;out.mkdir(exist_ok=True)
for name in ['completed.json','prefix-summary.json']:
 subprocess.run(SCP+[f'{SSH[-1]}:{path}/{name}',str(out/name)],check=True)
