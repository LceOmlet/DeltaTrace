"""One explicit upstream-layout repair verification after the preserved v1 failure."""
import ast
import base64
import hashlib
import json
import shlex
import sys
import zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
sys.path.insert(0,str(R/'research/runtime'))
from fla_wy_upstream_layout_backport import transformed_source
upstream=(A/'fla_wy_upstream_20260908/wy_fast.py').read_bytes()
old_wy=(A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/lib/python3.12/site-packages/fla/ops/gated_delta_rule/wy_fast.py').read_bytes()
new_wy=transformed_source(old_wy,upstream)
source=(A/'qwen35_fla_backward_20260908.py').read_text(encoding='utf-8')
source=source.replace('def source_receipt(scheduled=False):','def source_receipt(scheduled=False, wy_fixed=False):')
line="                        if scheduled: expected['fla/ops/common/chunk_delta_h.py'] = p['fla_scheduled_chunk_sha256']"
assert source.count(line)==1
source=source.replace(line,line+"\n                        if wy_fixed: expected['fla/ops/gated_delta_rule/wy_fast.py'] = p['wy_backported_sha256']")
old="    assert count == 2853 and set(changed) == ({'fla/utils.py', 'fla/ops/common/chunk_delta_h.py'} if scheduled else {'fla/utils.py'})"
assert old in source
source=source.replace(old,"    expected_changes = ({'fla/utils.py', 'fla/ops/common/chunk_delta_h.py'} if scheduled else {'fla/utils.py'})\n    if wy_fixed: expected_changes.add('fla/ops/gated_delta_rule/wy_fast.py')\n    assert count == 2853 and set(changed) == expected_changes")
old="    r['sources_during'] = r['sources_before']"
new="""    assert sha((HERE/'fla_wy_upstream_layout_backport.py').read_bytes())==p['layout_runtime_sha256']
    assert sha((Path(p['prior_failure_directory'])/'results.json').read_bytes())==p['prior_failure_raw_sha256']
    from fla_wy_upstream_layout_backport import apply_backport
    r['layout_backport']=apply_backport(p['isolated_environment'],Path(p['isolated_site'])/'fla/ops/gated_delta_rule/wy_fast.py',
                                      HERE/'upstream_wy_fast.py',HERE/'layout_backport_receipt.json')
    assert r['layout_backport']['after_sha256']==p['wy_backported_sha256']
    r['sources_during'] = source_receipt(scheduled=True,wy_fixed=True)"""
assert old in source;source=source.replace(old,new,1)
old="                gradients=torch.autograd.grad(output,[gpu[key] for key in names],grad_outputs=seed)"
assert old in source
source=source.replace(old,"                # Public engine control keeps the passive Python hook on this thread.\n                with torch.autograd.set_multithreading_enabled(False):\n                    gradients=torch.autograd.grad(output,[gpu[key] for key in names],grad_outputs=seed)")
# Save every completed stage even if a later native stage fails.
old="        capture.clear();r['native_stage_references_after_release']=len(capture.records);save()"
new="""        for index,row in enumerate(capture.records):
            try:save_arrays('completed_stage_'+str(index)+'_'+row['stage'],row['values'])
            except Exception:r.setdefault('partial_capture_errors',[]).append(traceback.format_exc())
        capture.clear();r['native_stage_references_after_release']=len(capture.records);save()"""
assert old in source;source=source.replace(old,new)
source=source.replace("r['sources_after']=source_receipt(scheduled=True)","r['sources_after']=source_receipt(scheduled=True,wy_fixed=True)")
source=source.replace("assert r['sources_before']==r['sources_after']","assert r['sources_during']==r['sources_after']")
source=source.replace("'native_fla_stage_capture.py','results.json']", "'native_fla_stage_capture.py','fla_wy_upstream_layout_backport.py','upstream_wy_fast.py','results.json']+[x.name for x in HERE.glob('layout_backport_receipt.json')]")
ast.parse(source)
study=A/'qwen35_fla_backward_20260908_v2.py';study.write_text(source,encoding='utf-8')
p=json.loads((A/'qwen35_fla_backward_protocol_20260908.json').read_bytes())
runtime=(R/'research/runtime/fla_wy_upstream_layout_backport.py').read_bytes()
p.update(study_sha256=sha(study.read_bytes()),layout_runtime_sha256=sha(runtime),wy_backported_sha256=sha(new_wy),
    prior_failure_directory='${ARTIFACT_ROOT}/codex_qwen35_fla_backward_20260908_v1',
    prior_failure_raw_sha256=sha((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_fla_backward_20260908_v1/results.json').read_bytes()),
    family_FLA_forward_attempt_ceiling=2,family_FLA_backward_attempt_ceiling=2,family_CPU_reference_ceiling=1,
    schedule_policy='Existing state schedule is unchanged. Only two upstream WY backward expressions are backported before FLA import.',
    runtime_control='torch.autograd.set_multithreading_enabled(False) only around native backward, to observe actual stages with a thread-local passive hook. No arithmetic change; timings remain diagnostic.',
    stop='One repair verification only, after the preserved v1 compiler error. No additional source adaptation or retry in this protocol. Preserve completed stages before release even on failure.')
p['native_stage_source_sha256']['wy']=sha(new_wy)
protocol=json.dumps(p,indent=2).encode();(A/'qwen35_fla_backward_protocol_20260908_v2.json').write_bytes(protocol)
files={'study.py':study.read_bytes(),'protocol.json':protocol,'fla_wy_upstream_layout_backport.py':runtime,'upstream_wy_fast.py':upstream,
       'official_fixed_text_inputs.py':(R/'research/runtime/official_fixed_text_inputs.py').read_bytes(),
       'native_fla_stage_capture.py':(R/'research/runtime/native_fla_stage_capture.py').read_bytes()}
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib; d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_fla_backward_20260908_v2"); d.mkdir(exist_ok=False); '
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+'))); [(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()]; '
        'f=(d/"driver.log").open("w"); j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True); '
        '(d/"pid").write_text(str(j.pid)); print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_fla_backward_20260908_v2.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'study_sha256':p['study_sha256'],'protocol_sha256':sha(protocol),'backported_wy_sha256':sha(new_wy),
                  'cumulative_forward_backward_attempts_maximum':2,'CPU_references_maximum':1}))
