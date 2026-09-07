"""One source-supported collector correction after preserving v1 failure."""
from pathlib import Path
import ast,hashlib,json
A=Path(__file__).resolve().parent
failed=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v1/results.json'
r=json.loads(failed.read_bytes())
assert r['status']=='failed' and "KeyError: 'hidden_states'" in r['error']
assert r['root_forward_attempts']==1 and r['root_forwards_completed']==r['finite_attempts']==0
code=(A/'build_qwen35_gdn_integration_20260908.py').read_text(encoding='utf-8')
code=code.replace('codex_qwen35_gdn_integration_20260908_v1','codex_qwen35_gdn_integration_20260908_v2')
code=code.replace('qwen35_gdn_integration_protocol_20260908.json','qwen35_gdn_integration_protocol_20260908_v2.json')
code=code.replace('launch_qwen35_gdn_integration_20260908.json','launch_qwen35_gdn_integration_20260908_v2.json')
insertion="""p.update(prior_failed_result_sha256=%r,
    collector_correction='Read pinned accelerate closure forward_func for passive observation; do not replace native wrapped forward.',
    combined_family_budget='At most two model loads/root attempts including v1 wiring failure; one completed root, three finite calls and one native local backward. No other retry is planned.',
    stop='One corrected capture after source inspection; preserve both outcomes. Stop on any further failure, no precision or quality sweep.')
""" % hashlib.sha256(failed.read_bytes()).hexdigest()
code=code.replace("files={'study.py':study.read_bytes(),",insertion+"files={'study.py':study.read_bytes(),")
ast.parse(code);exec(compile(code,str(A/'build_qwen35_gdn_integration_20260908.py'),'exec'))
