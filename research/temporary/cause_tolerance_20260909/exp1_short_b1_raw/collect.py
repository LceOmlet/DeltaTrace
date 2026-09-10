"""Wait for this task's completion, then archive its bounded evidence tree."""
from pathlib import Path
import hashlib,json,subprocess,time,zipfile
p=Path(__file__).resolve().parent
while True:
    q=[json.loads((p/n).read_bytes()) for n in ['queue_v2.json','queue_v3.json','postflight.json']]
    if all(x['status']=='complete' for x in q[:2]) and q[2]['status'] in ['verified','failed']:break
    time.sleep(5)
files={}
for f in p.iterdir():
    if f.is_file() and (f.suffix in ['.py','.log'] or f.name.startswith(('benchmark_protocol','queue','restore','native_restore','postflight')) and f.suffix=='.json'):
        files[f.name]=f
    elif f.is_dir() and (f/'results.json').exists():
        for n in ['results.json','time_curve_runs.jsonl','time_curve_summary.csv','vectors.npz']:
            if (f/n).exists():files[f.name+'/'+n]=f/n
files['release_environment.json']=p/'release/environment.json'
files['official_exp1/run_time_curve.py']=p/'release/official_exp1/run_time_curve.py'
gpu=subprocess.run(['mx-smi'],capture_output=True,text=True)
(p/'final_gpu_state.txt').write_text(gpu.stdout+gpu.stderr);files['final_gpu_state.txt']=p/'final_gpu_state.txt'
excluded={n:'Initial Qwen3.5 environment omitted the already accepted native FLA compatibility files; superseded by the hash-gated v2 run.' for n in ['qwen35_ifr_multi_hop_both','qwen35_ifr_multi_hop']}
for family in ['qwen3','qwen35']:
    for method in ['perturbation_all','perturbation_CLP','perturbation_REAGENT']:
        excluded[family+'_'+method+'_v2']='Original constructor auxiliary Longformer model absent; superseded by v3 with the exact official dependency restored and verified.'
selection={'excluded_attempts':excluded,'rule':'Exclude only eight identified incomplete-environment attempts; retain all failures after restoration. No runtime outlier removal.'}
(p/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');files['selection.json']=p/'selection.json'
manifest={name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for name,f in sorted(files.items())}
(p/'raw_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files['raw_manifest.json']=p/'raw_manifest.json'
with zipfile.ZipFile(p/'final_raw.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name,f in sorted(files.items()):z.write(f,name)
print(json.dumps({'status':'complete','files':len(files),'zip_bytes':(p/'final_raw.zip').stat().st_size,'zip_sha256':hashlib.sha256((p/'final_raw.zip').read_bytes()).hexdigest()}),flush=True)
