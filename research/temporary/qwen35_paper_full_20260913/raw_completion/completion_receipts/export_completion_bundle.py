"""Archive only the completed recovery, export and repeated cost records."""
import hashlib,json,pathlib,zipfile
ROOT=pathlib.Path('/mnt/geogpt-doc-new/deepresearch/lzq/deltatrace_qwen35_20260912')
OWN=pathlib.Path(__file__).resolve().parent
DEST=OWN.parent/'qwen35_paper_completion_20260913.zip'
read=lambda p:json.loads(p.read_bytes())
assert read(ROOT/'paper_comparison_status.json')['status']=='complete'
assert read(OWN/'cost_controller_status.json')['status']=='complete'
assert read(ROOT/'paper_recovery_dynamic/results.json')['status']=='complete'
assert read(ROOT/'export_dynamic_with_ifr/verification.json')['cases']==1243
assert read(OWN/'cost_v1/results.json')['status']=='complete'
records=[]
with zipfile.ZipFile(DEST,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for parent,name in [(ROOT,'paper_recovery_dynamic'),(ROOT,'export_dynamic_with_ifr'),(OWN,'cost_v1')]:
        for path in sorted((parent/name).rglob('*')):
            if not path.is_file():continue
            data=path.read_bytes();relative=path.relative_to(parent).as_posix()
            archive.writestr(relative,data)
            records.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    for path in [ROOT/'paper_comparison_status.json',OWN/'cost_controller_status.json',OWN/'cost.log',
                 OWN/'cost_controller.log',OWN/'continue_cost.py',OWN/'export_completion_bundle.py']:
        data=path.read_bytes();relative='completion_receipts/'+path.name
        archive.writestr(relative,data)
        records.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    archive.writestr('bundle_manifest.json',json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(path=str(DEST),bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),files=len(records))))
