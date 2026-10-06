"""Select a bounded row-layout comparison in the existing real-ID diagnostic.

Only diagnostic dispatch, owner injection and reporting differ. The original
actor loader, readout, finite arithmetic and Q/V/A consumers are called intact.
No model or remote process is launched by this preparation script.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
SOURCE = AUDIT / 'diagnose_native_prefix_leases.py'
EXPECTED = '6a0af68922d6ef9bd75bbb7c909333497b75ec09a1019f41624aa56e8fe0f349'


def main():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED
    before = raw.decode('utf8').replace('\r\n', '\n')
    text = before
    changes = []

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old[:100]
        text = text.replace(old, new)
        changes.append(dict(original=old, candidate=new))

    replace("    base_prefetch=os.environ.get('DT_PREFIX_BASE_MODEL_PREFETCH')=='1'",
            "    base_prefetch=os.environ.get('DT_PREFIX_BASE_MODEL_PREFETCH')=='1'\n"
            "    individual_rows=os.environ.get('DT_PREFIX_INDIVIDUAL_ROW_CANDIDATE')=='1'")
    replace("assert (os.environ.get('DT_PREFIX_REVERSE_PREFETCH')=='1' or conv_initial_states or boundary_row_storage",
            "assert (individual_rows or os.environ.get('DT_PREFIX_REVERSE_PREFETCH')=='1' or conv_initial_states or boundary_row_storage")
    replace("    if ledger_only:\n        labels=('shared_ledger',)", """    if individual_rows:
        assert current_owner and base_prefetch and phase_only and warm_phase
        assert not os.environ.get('DT_PREFIX_CHECKPOINT')
        assert not any(os.environ.get(key)=='1' for key in (
            'DT_PREFIX_HOT_PROFILE','DT_PREFIX_NATIVE_BACKWARD','DT_PREFIX_REVERSE_PREFETCH',
            'DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_PREFIX_NATIVE_CONV_CAPACITY',
            'DT_PREFIX_BOUNDARY_ROW_STORAGE','DT_PREFIX_LEDGER_ONLY','DT_PREFIX_ROOT_TAPE'))
        import inspect,importlib,json,sys
        prepared=json.loads((Path(out)/'prepared.json').read_bytes())
        owner=prepared['individual_row_candidate']
        modules={}
        for name in ('runner','finite_wrapper'):
            item=owner[name];source=Path(item['path'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
            spec=importlib.util.spec_from_file_location('_isolated_individual_row_'+name,source)
            value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value
            spec.loader.exec_module(value);modules[name]=value
        row_module=modules['runner'];row_finite=modules['finite_wrapper']
        row_module.RightPaddedLengths=row_finite.RightPaddedLengths
        row_operation=row_finite.VendorFAFiniteP1BF16D256(
            owner['finite_library']['path'],owner['finite_library']['sha256'])
        for name,value in (('artifact',importlib.import_module('qwen35_native_prefix_artifacts')),
                           ('lease',prepare_native_prefix_leases),
                           ('answer',importlib.import_module('qwen35_answer_finite'))):
            source=Path(inspect.getsourcefile(value));item=owner[name]
            assert source.resolve()==Path(item['path']).resolve()
            assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
        original_finite=runner.finite_fa
        labels=('shared_cold','shared_warm','shared_individual_rows_cold','shared_individual_rows_warm',
                'shared_individual_rows_observed')
        save('individual_row_candidate_sources',sources=owner,
             scope='Isolated actual B8 dispatch and representation comparison; no checkpoint, optimizer, new full-network tolerance or formal deployment')
    if ledger_only:
        labels=('shared_ledger',)""")
    replace("                    leases,preparation=prepare_native_prefix_leases(args[0],all_requests,**kwargs)",
            "                    leases,preparation=prepare_native_prefix_leases(args[0],all_requests,**kwargs,\n"
            "                        **({'individual_prefixes':True} if individual_rows and label.startswith('shared_individual_rows') else {}))")
    replace("    for label in labels:\n        if boundary_row_storage and label=='shared_boundary_rows_cold':",
            "    for label in labels:\n"
            "        if individual_rows and label=='shared_individual_rows_cold':\n"
            "            shared_bank=None\n"
            "            import gc\n"
            "            gc.collect()\n"
            "        if boundary_row_storage and label=='shared_boundary_rows_cold':")
    replace("        previous_class=runner.__class__\n        previous_capture_backend=runner.capture_backend",
            "        previous_class=runner.__class__\n"
            "        if individual_rows and label.startswith('shared_individual_rows'):\n"
            "            runner.__class__=row_module.Qwen35DenseFiniteRunner\n"
            "            runner.finite_fa=row_operation\n"
            "        previous_capture_backend=runner.capture_backend")
    replace("                runner.attribute=native_attribute\n            else:",
            "                runner.attribute=(types.MethodType(row_module.Qwen35DenseFiniteRunner.attribute,runner)\n"
            "                    if individual_rows and label.startswith('shared_individual_rows') else native_attribute)\n"
            "            else:")
    replace("            runner.__class__=previous_class\n            if conv_initial_states:",
            "            runner.__class__=previous_class\n"
            "            if individual_rows:runner.finite_fa=original_finite\n"
            "            if conv_initial_states:")
    replace("            context=(torch.profiler.profile(", """            actual_row_finite=None
            row_fa_records=None
            if individual_rows and label=='shared_individual_rows_observed':
                from observe_native_gdn0_operands import NativeGDN0Operands
                from observe_native_fa3_varlen_operands import make_attention_capture_observer
                from observe_actual_row_finite_fa import ActualRowFiniteFA
                globals_=selected_attribute.__func__.__globals__
                def row_capture_class(name):
                    return globals_[name] if saved_capture_backend is None else getattr(saved_capture_backend,name)
                gdn0_audit=NativeGDN0Operands(runner.model.model.language_model.layers[0].linear_attn,
                    Path(out)/'actual-gdn0-operands',variant=label,rank=torch.distributed.get_rank())
                row_fa_records=[]
                runner.capture_backend=types.SimpleNamespace(
                    NativeDecoderCapture=row_capture_class('NativeDecoderCapture'),
                    NativeGDNCapture=gdn0_audit.capture_type(row_capture_class('NativeGDNCapture')),
                    NativeDenseAttentionCapture=make_attention_capture_observer(
                        row_capture_class('NativeDenseAttentionCapture'),
                        target_module=runner.model.model.language_model.layers[3].self_attn,
                        records=row_fa_records))
                actual_row_finite=ActualRowFiniteFA(row_operation,
                    Path(out)/f'actual-row-finite-fa3-rank{torch.distributed.get_rank()}.pt')
                runner.finite_fa=actual_row_finite
            context=(torch.profiler.profile(""")
    replace("                    runner.capture_backend=saved_capture_backend\n", """                    runner.capture_backend=saved_capture_backend
                    if actual_row_finite is not None:
                        runner.finite_fa=row_operation
""")
    replace("            if gdn0_audit is not None:\n", """            if individual_rows and label=='shared_individual_rows_observed':
                import json
                assert len(gdn0_audit.records)==len(row_fa_records)==1
                assert actual_row_finite.calls==8 and actual_row_finite.saved is not None
                fa_path=Path(out)/f'actual-row-native-fa3-rank{torch.distributed.get_rank()}.pt'
                with fa_path.open('xb') as stream:torch.save(row_fa_records[0],stream)
                save('actual_individual_row_operands_saved',
                     actual_finite=actual_row_finite.saved,
                     native_gdn0=gdn0_audit.records[0],
                     native_fa3=dict(path=str(fa_path),sha256=hashlib.sha256(fa_path.read_bytes()).hexdigest()),
                     scope='Passive actual operands including original finite LSE/upstream; copy/export overhead belongs to this observed variant only. Official numerical checks run after model release.')
                gdn0_audit=None
            if gdn0_audit is not None:
""")
    # Persist only exact baseline/candidate changes; no new numerical threshold.
    ast.parse(text)
    output = HERE/'native-representation-candidate'/'diagnose_native_prefix_leases.py'
    output.write_bytes(text.encode('utf8'))
    patch = output.with_suffix('.patch')
    patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),
        fromfile='original-diagnostic',tofile='isolated-row-diagnostic')),encoding='utf8')
    receipt = dict(role=__doc__, original=dict(path=str(SOURCE),sha256=EXPECTED),
        candidate=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        changes=changes, syntactically_valid=True, executed=False,
        numerical_threshold_added=False, checkpoint_load=False, optimizer_update=False)
    (output.parent/'real-b8-diagnostic-prepared.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(candidate_sha256=receipt['candidate']['sha256'],changes=len(changes))))


if __name__ == '__main__':
    main()
