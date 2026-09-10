"""Bounded complete evidence for the rematerialization/whole-graph/SAC study."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
folders=['remat_pilot','whole_remat','whole_profile','whole_sac','sac_fa_head','sac_released','sac_lifetime','sac_lifetime_v2','sac_lifetime_v3','sac_lifetime_v4','memory_production_pilot_v2']
for folder in folders:assert json.loads((root/folder/'results.json').read_bytes())['status'] in ['complete','failed'],folder
prior=json.loads((root/'evidence_benchmark_manifest.json').read_bytes()) if (root/'evidence_benchmark_manifest.json').exists() else None
if prior is None:
    with zipfile.ZipFile(root/'evidence_benchmark.zip') as z:prior=json.loads(z.read('archive_manifest.json'))
selected=[path for path in root.iterdir() if path.is_file() and path.suffix in ['.py','.json','.log','.so'] and path.name not in prior['files'] and not path.name.startswith(('postflight','author','check_','collect_author','memory_confirmation','benchmark_memory_confirmation','run_memory_confirmation','memory_author'))]
for folder in folders:selected.extend(path for path in (root/folder).rglob('*') if path.is_file())
assert (root/'memory_production_pilot').is_dir() and not (root/'memory_production_pilot/results.json').exists()
assert 'FileNotFoundError' in (root/'memory_production_pilot.log').read_text()
selected.extend(path for path in (root/'memory_production_release').rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc')
sha=lambda data:hashlib.sha256(data).hexdigest();members={};total=0
for path in sorted(set(selected)):
    data=path.read_bytes();name=path.relative_to(root).as_posix();assert len(data)<50_000_000,name
    total+=len(data);members[name]={'sha256':sha(data),'bytes':len(data)}
assert total<150_000_000,total
manifest={'files_count':len(members),'uncompressed_bytes':total,'files':members}
target=root/'evidence_optimization_v1.zip';assert not target.exists()
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(root/name,name)
    z.writestr('archive_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'archive':str(target),'sha256':sha(target.read_bytes()),'bytes':target.stat().st_size,'files':len(members),'uncompressed_bytes':total}))
