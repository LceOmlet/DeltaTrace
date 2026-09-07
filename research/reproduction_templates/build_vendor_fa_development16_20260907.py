"""Freeze all original sixteen development cases after verified integration pilot."""
import hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
summary=json.loads((A/'vendor_fa_end_to_end_summary_20260907.json').read_text());assert summary['status']=='verified_complete' and summary['curves_verified']==6
p=json.loads((A/'vendor_fa_end_to_end_protocol_20260907.json').read_text())
study=A/'vendor_fa_development16_20260907.py';study.write_bytes((A/'vendor_fa_end_to_end_probe_20260907.py').read_bytes())
p.update(purpose='All sixteen original NI0-7/MH0-7 development cases. Same fixed traceable-FA finite P1 versus same-job dense P1, fresh original32deletion curves, same-job ordinary FA backward memory references, full attribution costs. Native model FA unchanged. Original FT0-3 controls retained with verified parent hashes as historical controls. This is development, not independent confirmation, batching or long-input validation.',
 study_sha256=sha(study),selection=[[ds,i] for ds in ['niah_mq_q2','morehopqa'] for i in range(8)],
 wait_for_pid=123093,wait_for_script='codex_vendor_fa_end_to_end_20260907_v1',
 required_integration_pilot={'raw_sha256':summary['raw_sha256'],'summary_sha256':sha(A/'vendor_fa_end_to_end_summary_20260907.json'),
 'scope':'Verified three-case integration prerequisite; no pilot quality reused for candidate or dense curves in this job.'},
 budget={'native_root_forwards':818,'native_vjps':16,'evaluation_forwards':672,'ft_attribution_forwards':0,
 'ordinary_reference_forwards':16,'native_attribution_forwards':130,'manual_passes':130,'extra_layer_replay_calls':4680,
 'extra_native_fa_attention_calls':4680,'native_attribution_endpoint_trajectories':260,
 'extra_layer_replay_endpoint_trajectories':9360,'finite_FA_calls_attempted':2340,'finite_FA_calls_enqueued':2340},
 repeats='Each of16original cases/dense or finite mode:1warm+3measured rotated. NI0 extra complete shape/memory profile per mode.16ordinary native backward references. All130attributions frozen before32fresh original21-point curves.')
p['predeclared_review']['quality']='Fresh original RISE/MAS/needle for dense and finite on all16development examples. Original FT0-3 pinned parent controls retained as historical comparison. Evaluate unchanged joint development gate; no independent confirmation or full cross-task claim.'
protocol=A/'vendor_fa_development16_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_development16_20260907_v1',
 f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in p['sources']],
 '--request',str(A/'vendor_fa_development16_launch_20260907.json')],check=True)
request=json.loads((A/'vendor_fa_development16_launch_20260907.json').read_text());assert len(request['cmd'])<128000
print('Frozen original development16:',len(request['cmd']),'characters;818rootF,16VJP,130finite passes;32fresh original curves.')
