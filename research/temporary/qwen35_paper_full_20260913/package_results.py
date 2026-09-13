"""Create a portable record bundle after all independent verification passes."""
import hashlib,json,pathlib,zipfile
OWN=pathlib.Path(__file__).resolve().parent
REPO=OWN.parents[2]
TRANSFER=OWN.parent/'source_v2_gpu_20260910'
AUDIT=OWN.parent/'qwen35_paper_audit_20260913'
result=json.loads((OWN/'results/verification.json').read_bytes())
assert result['status']=='complete' and result['quality_cases']==1243 and result['cost_calls']==192
output=OWN/'QWEN35_DT_PAPER_RESULTS.zip'
manifest=[]
def add(archive,path,relative,stored=False):
    data=path.read_bytes()
    archive.writestr(relative,data,compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    manifest.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in sorted((OWN/'results').iterdir()):
        if path.is_file():add(archive,path,'results/'+path.name)
    for name in ['qwen35_existing_review_20260913.zip','qwen35_existing_runtime_20260913.zip',
                 'qwen35_main_completion_20260913.zip','qwen35_paper_completion_20260913.zip']:
        add(archive,TRANSFER/name,'raw_snapshots/'+name,stored=True)
    for name in ['protocol.json','benchmark_complete_calls.py','verify_and_report.py','continue_cost.py',
                 'export_completion_bundle.py','package_results.py']:
        add(archive,OWN/name,'completion_code/'+name)
    for name in ['audit_protocol.py','audit_prepared_inputs.py','audit_saved_curves.py','protocol_audit.json',
                 'prepared_inputs_audit.json','saved_curves_audit_full.json','composite_protocol.json']:
        add(archive,AUDIT/name,'independent_audits/'+name)
    for name in ['evaluation.tex','appendix.tex','method.tex']:
        add(archive,REPO/'paper/iclr2027/sections'/name,'dt_manuscript/'+name)
    frozen_inputs=json.loads((AUDIT/'protocol_audit.json').read_bytes())['paper_source_files']
    for relative,digest in frozen_inputs.items():
        path=REPO/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        add(archive,path,'dt_paper_fixtures/'+path.name)
    readme='''# Qwen3.5-9B completed DT-paper comparison

Start with results/RESULTS.md. The quality CSVs contain all 13 tasks and 1,243 examples,
including the 448 paper-matched VT/HotpotQA recovery evaluations. The repeated cost
CSVs contain 16 fixed development examples and 192 complete calls.

raw_snapshots contains the original code/runtime archives and the final recovery,
combined export and cost archive. Extract the runtime snapshot first, then overlay
the main-completion snapshot to add the final MoreHopQA records. The code snapshot
and paper-completion snapshot have distinct top-level directories.

independent_audits contains the fixed-protocol, tokenizer/input and full saved-curve
audits. Static-audit limitations describe the scope of that earlier audit; the final
results/verification.json records the completed model execution and cost verification.

completion_code contains the frozen extension protocol, actual cost driver and
final independent verifier. dt_manuscript records the evaluation authority, from
the DeltaTrace manuscript at commit 4cbcbc8127acab3146e0256d204f7acb47a885fd.
dt_paper_fixtures preserves the original matched inputs, candidates and labels.

Model weights, compiled accelerator libraries and the Python environment are not
duplicated in this record bundle. Their identities and paths are recorded in the
runtime receipts. Re-running GPU inference requires those original dependencies.

Quality intervals use 10,000 paired within-task bootstrap resamples with seed 73;
they are not adjusted for comparisons across tasks. They are recomputed separately
from the original export, whose random stream also included one-pass cost metrics.
The primary cost table uses the separate repeated complete-call experiment.
'''
    archive.writestr('README.md',readme)
    archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
digest=hashlib.sha256(output.read_bytes()).hexdigest()
(OWN/'bundle_receipt.json').write_text(json.dumps(dict(path=output.name,bytes=output.stat().st_size,
    sha256=digest,files=len(manifest)),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=digest,files=len(manifest))))
