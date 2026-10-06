"""Prepare an isolated fixed compact+row B8 then exact32k diagnostic entry."""
import ast,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'native-representation-candidate/diagnose_native_prefix_leases.py'
def replace_once(text,old,new):
 assert text.count(old)==1,(old[:120],text.count(old))
 return text.replace(old,new)
def main():
 raw=ORIGINAL.read_bytes();assert hashlib.sha256(raw).hexdigest()=='cdfe43a52ad5e6bdd7f51404f5094b0ac602332719b643604282bb6c41404dc4'
 text=raw.decode()
 text=replace_once(text,'def diagnose(runner, producer, out, save, *, cache_tensors=None):','def _diagnose_combined_stage(runner, producer, out, save, *, cache_tensors=None, capacity=False):')
 text=replace_once(text,"    boundary_row_storage=os.environ.get('DT_PREFIX_BOUNDARY_ROW_STORAGE')=='1'", "    boundary_row_storage=os.environ.get('DT_PREFIX_BOUNDARY_ROW_STORAGE')=='1'\n    exact_capacity=bool(capacity)\n    assert individual_rows and base_prefetch\n    assert not any(os.environ.get(key)=='1' for key in (\n        'DT_PREFIX_NATIVE_CONV_CAPACITY','DT_PREFIX_NATIVE_CONV_INITIAL_STATES',\n        'DT_PREFIX_BOUNDARY_ROW_STORAGE','DT_PREFIX_NATIVE_BACKWARD',\n        'DT_PREFIX_HOT_PROFILE','DT_PREFIX_REVERSE_PREFETCH','DT_PREFIX_ROOT_TAPE'))\n    assert not os.environ.get('DT_PREFIX_CHECKPOINT')")
 for old in ('    if conv_capacity:\n        capacity_source_offset=offset', '    if conv_capacity:\n        rows=[episode[0] for episode in capacity_episodes]', '        if conv_capacity:\n            assert len(pending)==1', '            if conv_capacity:\n                result=[row for episode in readout.episodes(capacity_episodes,complete_returns=capacity_returns)'):
  text=replace_once(text,old,old.replace('if conv_capacity:','if conv_capacity or exact_capacity:'))
 text=replace_once(text,"            if conv_capacity else 'Current frozen formal owner", "            if (conv_capacity or exact_capacity) else 'Current frozen formal owner")
 text=replace_once(text,"        labels=('shared_cold','shared_warm','shared_individual_rows_cold','shared_individual_rows_warm',\n                'shared_individual_rows_observed')", "        labels=('shared_individual_rows_cold','shared_individual_rows_warm')")
 text=replace_once(text,"    if current_owner:reference_label='shared_warm'", "    if current_owner:reference_label='shared_individual_rows_warm'")
 old="                        **({'individual_prefixes':True} if individual_rows and label.startswith('shared_individual_rows') else {}))"
 new="                        **({'individual_prefixes':True,'boundary_row_storage':True} if individual_rows and label.startswith('shared_individual_rows') else {}))\n                if individual_rows and label.startswith('shared_individual_rows'):\n                    consumed=[(source,row,n) for lease in leases if lease is not None\n                              for (source,row),n in zip(lease.sources,lease.prefix_lengths)]\n                    assert consumed and all(getattr(source,'boundary_rows',None) is not None for source,row,n in consumed)\n                    assert all(row in source.boundary_rows[n] for source,row,n in consumed)\n                    save('combined_storage_row_bank_enabled',variant=label,\n                        individual_prefixes=True,boundary_row_storage=True,\n                        consumed_source_slots=len(consumed),\n                        distinct_artifacts=len({id(source) for source,row,n in consumed}),\n                        original_preparation=preparation)"
 text=replace_once(text,old,new)
 text+='\n\n'+WRAPPER
 tree=ast.parse(text);original_tree=ast.parse(raw)
 def helper(t):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_native_conv_capacity_inputs')
 assert ast.dump(helper(tree),include_attributes=False)==ast.dump(helper(original_tree),include_attributes=False)
 for node in original_tree.body:
  if isinstance(node,ast.FunctionDef) and node.name!='diagnose':
   other=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==node.name)
   assert ast.dump(node,include_attributes=False)==ast.dump(other,include_attributes=False),node.name
 target=HERE/'diagnose_combined_storage_row_capacity.py';target.write_text(text)
 (HERE/'combined-diagnostic-vs-frozen-row.patch').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),text.splitlines(True),fromfile='frozen-row-cdfe43a5',tofile='isolated-combined-row-capacity')))
 receipt=dict(status='prepared_only_no_execution',original=dict(path=str(ORIGINAL.resolve()),sha256=hashlib.sha256(raw).hexdigest()),candidate=dict(path=str(target.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),complete_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()),capacity_helper_AST_unchanged=True,all_other_original_top_level_functions_AST_unchanged=True,scope='Fixed real B8 compact+row cold/warm then exact32768 cold/warm in the original worker/model. No convolution toggles, observers/reference calls, CP or optimizer steps; existing helper and original readout/finite formulas unchanged.')
 (HERE/'combined-diagnostic-source-prepared.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
WRAPPER=r'''def diagnose(runner, producer, out, save, *, cache_tensors=None):
    """Run only the two explicitly approved compact+row resource stages."""
    import gc,json
    out=Path(out)
    prepared=json.loads((out/'prepared.json').read_bytes())
    assert prepared['combined_storage_row_capacity_entry'] is True
    assert prepared['checkpoint'] is None and prepared['base_model_initialization'] is True
    completed={}
    for stage,capacity in (('actual_b8',False),('exact32768_b8',True)):
        stage_out=out/stage
        assert stage_out.is_dir()
        def stage_save(phase,**values):
            if phase=='native_prefix_lease_diagnostic_complete':
                completed[stage]=dict(**values)
            save(phase,stage=stage,**values)
        save('combined_storage_row_stage_start',stage=stage,exact32768_capacity=capacity,
             checkpoint_loads=0,optimizer_steps=0,extra_observer_calls=0)
        _diagnose_combined_stage(runner,producer,stage_out,stage_save,
                                cache_tensors=cache_tensors,capacity=capacity)
        if not capacity:
            rank=str(torch.distributed.get_rank())
            previous=prepared['previous_uncompressed_row_vectors'][rank]
            source=Path(previous['path'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==previous['sha256']
            actual_path=stage_out/f'prefix-lease-vectors-rank{rank}.pt'
            actual=torch.load(actual_path,map_location='cpu',weights_only=False)
            reference=torch.load(source,map_location='cpu',weights_only=False)
            comparisons=[]
            for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates'):
                left=actual['shared_individual_rows_warm'][key]
                right=reference['shared_individual_rows_warm'][key]
                assert left.shape==right.shape and left.dtype==right.dtype
                comparisons.append(dict(key=key,shape=list(left.shape),dtype=str(left.dtype),
                    literal_bytes_equal=bool(torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8))),
                    numerical_equal=bool(torch.equal(left,right)),
                    maximum_absolute_difference=float((left.double()-right.double()).abs().max())))
            completed[stage]['previous_uncompressed_row_comparison']=dict(
                source=previous,actual=dict(path=str(actual_path),sha256=hashlib.sha256(actual_path.read_bytes()).hexdigest()),
                values=comparisons,scope='Literal output comparison only; no new floating-point tolerance or numerical correction')
            save('combined_storage_row_previous_output_compared',stage=stage,
                 comparison=completed[stage]['previous_uncompressed_row_comparison'])
            del actual,reference,left,right
        gc.collect()
    save('combined_storage_row_capacity_complete',completed_stages=completed,
         checkpoint_loads=0,optimizer_steps=0,
         scope='Actual compact+row B8 and exact32768 resource calls only; no task-quality or new kernel tolerance claim')
    return None
'''
if __name__=='__main__':main()
