import json,pathlib,numpy as np
b=pathlib.Path(r'C:/Users/Chen/Documents/ChatGPT/credit/audit/qwen35_existing_runtime_20260913/full_dynamic_with_ifr')
counts={m:dict(cases=0,cases_negative=0,min=0.,negative_abs_mass=0.,total_abs_mass=0.) for m in ('FT_K1','FT_K3')}
for f in b.glob('*/vectors.npz'):
 with np.load(f,allow_pickle=False) as v:
  for k in v.files:
   for m in counts:
    if not k.endswith('_'+m+'_prompt'):continue
    x=v[k];c=counts[m];c['cases']+=1;c['cases_negative']+=int(np.any(x<0));c['min']=min(c['min'],float(x.min()));c['negative_abs_mass']+=float(np.abs(x[x<0]).sum());c['total_abs_mass']+=float(np.abs(x).sum())
print(json.dumps(counts,indent=2))

