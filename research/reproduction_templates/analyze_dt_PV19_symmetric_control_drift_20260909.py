"""Existing-data cross-process C drift and same-process two-mask-family comparison."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV_layer19_remaining_regression_20260909_s0_v1';new=A/'snapshot${ARTIFACT_ROOT}/codex_dt_PV19_symmetric_whole_pilot_20260909_v1'
o,n=read(old/'results.json'),read(new/'results.json');audit=read(A/'dt_PV19_symmetric_whole_pilot_summary_20260909.json')
assert audit['results_sha256']==sha(new/'results.json') and audit['status']=='independent_PV19_symmetric_whole_pilot_audit_passed'
assert o['status']=='layer19_PV_content0_remaining_segment_4DT100FLA84score_complete'
for d,r in [(old,o),(new,n)]:
 rec=read(d/'terminal_receipt.json');assert rec['proc_exists'] is False
 for file,item in rec['files'].items():assert sha(d/file)==item['sha256']
 assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
for k in ['native_model_sha256','installed_FA_interface_sha256','runtime_source_sha256','expected_weight_stats','checkpoint_config_tokenizer_sha256','cache_hashes','official_source_blob_sha1']:
 assert o['protocol'][k]==n['protocol'][k]
class CurrentOnly(ast.NodeTransformer):
 def visit_If(self,node):
  if ast.dump(node.test)==ast.dump(ast.parse("pv_rule=='symmetric'",mode='eval').body):return None
  return self.generic_visit(node)
 def visit_Tuple(self,node):
  if all(isinstance(x,ast.Constant) for x in node.elts) and [x.value for x in node.elts]==['content1','content0','symmetric']:node.elts=node.elts[:2]
  return self.generic_visit(node)
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
  return self.generic_visit(node)
source={}
for file in ['qwen35_decoder_finite.py','qwen35_dense_finite_runner.py']:
 x=ast.dump(CurrentOnly().visit(ast.parse((old/file).read_bytes())),include_attributes=False)
 y=ast.dump(CurrentOnly().visit(ast.parse((new/file).read_bytes())),include_attributes=False)
 assert x==y;source[file]={'old_sha256':sha(old/file),'tested_sha256':sha(new/file),'current_P1_path_AST_equal_after_removing_only_new_symmetric_branch_and_docstrings':True}
vo,vn=np.load(old/'vectors.npz',allow_pickle=False),np.load(new/'vectors.npz',allow_pickle=False)
out={'status':'symmetric_pilot_two_mask_and_historical_C_drift_audited','analyzer_sha256':sha(Path(__file__)),
 'results_sha256':sha(new/'results.json'),'old_results_sha256':sha(old/'results.json'),'paired_audit_sha256':sha(A/'dt_PV19_symmetric_whole_pilot_summary_20260909.json'),
 'default_source_equivalence':source,'cases':{}}
for key in n['cases']:
 assert o['cases'][key]['input']==n['cases'][key]['input'] and o['cases'][key]['gold']==n['cases'][key]['gold']
 ro=next(x for x in o['runs'] if x['case']==key and x['method']=='control');rn=next(x for x in n['runs'] if x['case']==key and x['method']=='control')
 before,after=vo[key+'_control_full'],vn[key+'_control_full'];delta=after-before
 native={}
 for k in ['target_logp0','target_logp1']:
  a,b=np.asarray(ro['details'][k]),np.asarray(rn['details'][k]);native[k]={'max_absolute':float(np.abs(b-a).max()),'sum_delta':float((b-a).sum()),'relative_L2':float(np.linalg.norm(b-a)/np.linalg.norm(a))}
 co,cn=o['cases'][key]['curves']['control'],n['cases'][key]['curves']['control'];old_inputs={x['input_sha256']:(i,float(co['scores'][i])) for i,x in enumerate(co['input_receipts'])};shared=[]
 for i,rec in enumerate(cn['input_receipts']):
  if rec['input_sha256'] in old_inputs:
   oi,score=old_inputs[rec['input_sha256']];shared.append({'old_step':oi,'new_step':i,'input_sha256':rec['input_sha256'],'old_score':score,'new_score':cn['scores'][i],'delta':cn['scores'][i]-score})
 same=audit['cases'][key];out['cases'][key]={'same_process_candidate_minus_control':same['candidate_minus_control'],
  'same_process_two_mask_families':{bg:{m:z['MAE_AUC'] for m,z in rows.items()} for bg,rows in same['both_actual_mask_families'].items()},
  'same_process_native_root_drift':same['same_case_native_root_drift'],'same_process_score_endpoint_deltas':same['candidate_minus_control_native_score_endpoints'],
  'old_vs_new_C_metrics':{'old':co['return_metrics'],'new':cn['return_metrics'],'old_needle':co['needle'],'new_needle':cn['needle']},
  'old_vs_new_C_vector':{'relative_L2':float(np.linalg.norm(delta)/np.linalg.norm(before)),'max_absolute':float(np.abs(delta).max()),'signed_sum_delta':float(delta.sum())},
  'old_vs_new_C_native_root':native,'identical_scored_inputs_across_processes':shared,
  'cost_single_observation_only':{m:{'seconds':run['outer_attribute_seconds'],'peak_allocated':run['details']['peak_allocated'],'finite_FA_calls':run['counts']['finite_FA_backend_calls']} for m in ['control','candidate'] for run in n['runs'] if run['case']==key and run['method']==m}}
out['decision']='Symmetric PV19 is not promoted or expanded: both original RISE regress, NI0 MAS regresses, needle unchanged, MH2 MAS improvement below.001. Preserve actual two-mask error tradeoffs and cost. No further continuous PV weight tuning.'
out['limits']=['Default P1 arithmetic/source AST equality is not proof of compiled or native execution equality. Real original native root/scorer differences across processes are measured, not declared harmless BF16 error or a proved backend bug.',
 'Only same-process C/candidate comparisons support this method decision. Prior FT and prior C are historical references, not contemporaneous controls.',
 'Raw weight contents were not rehashed in either original job; shared weight file size/mtime and config/tokenizer/source receipts agree. No native framework repair is justified by this audit alone.',
 'One timing observation per method/case includes diagnostics and order/shape effects. No warm speed or true batch conclusion.']
path=A/'dt_PV19_symmetric_control_drift_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path),'cases':{k:{'two_mask':v['same_process_two_mask_families'],'native':v['old_vs_new_C_native_root'],'vector':v['old_vs_new_C_vector'],'common_score_deltas':[x['delta'] for x in v['identical_scored_inputs_across_processes']],'cost':v['cost_single_observation_only']} for k,v in out['cases'].items()}}))
