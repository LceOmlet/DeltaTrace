"""Preserve a failed exact-parent check, its actual magnitude, and spent budget."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_finite_actual_20260907_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert d['status']=='failed' and not d['operator_attempts'] and d['manual_passes']==1
assert not d['records'][0]['signed_vector_exact_to_parent'] and len(d['captured_layers'])==1
assert sha(F/'study.py')==p['study_sha256']==sha(A/'vendor_fa_finite_actual_probe_20260907.py')
for name,h in p['sources'].items():assert sha(F/name)==sha(A/name)==h
parent_path=A/'snapshot'/p['required_parent'].lstrip('/');assert sha(parent_path)==p['required_parent_sha256']
parent=json.loads(parent_path.read_text());old=next(r for r in parent['records'] if r['dataset']=='niah_mq_q2' and r['idx']==0)['native']['strong_secant_pv_content_P1']
new=d['records'][0]['original_result']
assert d['checkpoint_before']==parent['checkpoint_before'] and d['native_sources_before']==parent['native_sources_before']
base=(F/'qwen_signed_secant_pv_rules.py').read_text();actual=(F/'qwen_signed_secant_pv_operand_capture.py').read_text()
tree=ast.parse(actual);assign=tree.body.pop(0);assert assign.targets[0].id=='ATTENTION_OBSERVER'
class RemoveObserver(ast.NodeTransformer):
    count=0
    def visit_If(self,node):
        if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='ATTENTION_OBSERVER':
            self.count+=1;return None
        return self.generic_visit(node)
remove=RemoveObserver();tree=remove.visit(tree);assert remove.count==1 and ast.dump(tree)==ast.dump(ast.parse(base))
assert new['endpoint_scores32']==old['endpoint_scores32']
assert all(r['native_input_exact'] and r['native_output_exact'] for r in new['native_layer_boundary_checks']['paired_batch'])
assert all(r['public_output_exact_to_actual_model_FA'] for r in new['public_FA_capture_checks'])
for meta in d['captured_layers'][0]['operands'].values():
    path=F/meta['file'];assert sha(path)==meta['sha256'];x=np.load(path,allow_pickle=False)
    assert list(x.shape)==meta['shape'] and str(x.dtype)==meta['dtype'].removeprefix('torch.') and np.isfinite(x).all()
x=np.asarray(new['signed_full_sequence']);y=np.asarray(old['signed_full_sequence'])
out={'status':'verified_failed_capture_not_kernel_failure','raw_sha256':sha(F/'results.json'),
 'parent_vector_guard':'failed_preserved','cause':'Unresolved. Callback source is passive; source arithmetic unchanged after removal. Allocation/compiler/warmup explanations are hypotheses, not established causes.',
 'vector':{'relative_l2':float(np.linalg.norm(x-y)/np.linalg.norm(y)),'maximum_absolute':float(abs(x-y).max()),'changed_elements':int((x!=y).sum()),'sign_changes':int((np.sign(x)!=np.sign(y)).sum())},
 'endpoint_scores_exact':True,'native_layer_replay_exact':True,'public_FA_outputs_exact':True,
 'source_arithmetic_unchanged_after_removing_observer':True,'checkpoint_and_native_sources_before_match_parent':True,
 'post_run_checkpoint_and_source_checks':'Not reached because guard failed; do not infer their completion.',
 'new_unassigned':new['unassigned_total'],'parent_unassigned':old['unassigned_total'],
 'budget':{k:d[k] for k in p['capture_budget']},'finite_operator_calls':0,'captured_real_layers':1,
 'use_limit':'Captured layer35 operands may define a standalone local numerical control, without treating the original end-to-end capture guard as passed.'}
for key,value in p['capture_budget'].items():assert d[key]==value
(A/'vendor_fa_capture_failure_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
