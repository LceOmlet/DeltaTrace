"""Bind a research-only primitive result without promoting it to a repair.

This script performs local receipt accounting only. It never selects another
sample, changes PLAN, calls a model, deploys a candidate or releases an update.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
COLLECTION = HERE.parent
REPO = HERE.parents[5]


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def main():
    compiled = read(HERE/'compiled-owner.json')
    verified = read(HERE/'original-fa-check.json')
    numerical = verified['numerical']
    assert compiled['returncode'] == verified['returncode'] == 0
    assert numerical['finite_library_sha256'] == compiled['library_sha256']
    assert numerical['status'] == 'passed'
    assert all(numerical['unchanged_default'].values())
    assert compiled['formal_textcraft']['release_exists'] == [False, False]
    prepared = read(HERE/'prepared-owner.json')
    assert prepared['header_sha256'] == ref(HERE/'conditional_attention_rows.cuh')['sha256']
    assert prepared['patch_sha256'] == ref(HERE/'owner.patch')['sha256']
    for name, expected in prepared['generated_sha256'].items():
        assert compiled['source_sha256'][name] == expected
    assert compiled['source_sha256']['conditional_attention_rows.cuh'] == prepared['header_sha256']
    test_source = ref(REPO/'experiments/rl/verify_saved_fa_dtypes.py')
    assert verified['launch']['source_sha256']['verify_saved_fa_dtypes.py'] == test_source['sha256']
    source_names = ['conditional_attention_rows.cuh', 'prepare_owner.py',
                    'stage_compile_owner.py', 'check_original_fa.py',
                    'record_owner_candidate.py', 'owner.patch', 'prepared-owner.json']
    evidence_names = ['compiled-owner.json', 'original-fa-check.json',
                      'compiled-owner-first-failure.json',
                      'compiled-owner-first-numerical-failure.json',
                      'compiled-owner-row-mapping-edit-failure.json',
                      'original-fa-first-failure.json',
                      'original-fa-nonfinite-diagnosis.json',
                      'native-row-reduction-owner.json', 'native-softmax-row-owner.json']
    result = dict(
        status='Research-only primitive verified; no collection-quality claim or production repair',
        observed_unix=time.time(),
        prepared_from_git=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        actual_candidate=dict(remote_directory=compiled['remote_directory'],
                              library_sha256=compiled['library_sha256'],
                              generated_source_sha256=compiled['source_sha256'],
                              original_owner=prepared['original_cuda'],
                              original_wrapper=prepared['original_wrapper']),
        source=[ref(HERE/name) for name in source_names]+[test_source],
        evidence=[ref(HERE/name) for name in evidence_names],
        method=ref(REPO/'experiments/rl/PLAN.md'),
        frozen_evidence=[ref(REPO/'experiments/rl/results_credit_collection_scope_review_20261008.json'),
                         ref(COLLECTION/'conditional-attention-feasibility.json'),
                         ref(COLLECTION/'author-collection-summary.json')],
        owner_change=dict(
            placement='Default-inert extension of the existing finite-FA owner; original public FA remains the query/LSE owner.',
            unchanged_default=numerical['unchanged_default'],
            enabled_path='Complete local conditional Q/K/V identity with own-key interaction; fused row statistics and stable sigmoid secant.',
            no_output_correction=True, no_credit_clipping=True, no_extra_model_forward=True,
            runner_connected=False, gate_connected=False,
            full_generated_owner_copies='Local/remote compilation artifacts only; repository stores the differential patch and generator.'),
        test=dict(
            original_source_sha256='a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9',
            original_assertions=numerical['assertions'], checks=numerical['checks'],
            actual_query_shape=numerical['query_shape'], actual_key_shape=numerical['key_shape'],
            operand_dtypes=numerical['operand_dtypes'],
            finite_nonfinite_debug=numerical['finite_nonfinite_debug'],
            scope='Saved actual operands, coincident-endpoint derivative. Not nonzero finite accuracy, frozen collection quality, 32k capacity or a whole-model tolerance.'),
        failure_history=[
            dict(stage='First compilation', cause='Owner Allreduce requires an lvalue reduction functor; temporary functors did not compile.'),
            dict(stage='First numerical check', cause='Candidate used CUDA xor2/1 row grouping instead of this pinned MetaX owner quad_allreduce_ xor48/32/16 grouping; mixed rows produced invalid normalizers and NaNs.',
                 scope='Candidate implementation defect; not evidence about the original training degradation.',
                 resolution='Use the pinned native owner Allreduce<64> mapping. No clipping, rescaling or tolerance change.'),
            dict(stage='Row-mapping edit compilation', cause='An escaped newline in a comment swallowed the max_op declaration; fixed the text edit.')],
        measured_resources=dict(
            compilation_seconds=compiled['finished_unix']-compiled['started_unix'],
            compilation_sampled_tree_PSS_bytes=compiled['sampled_peak_tree_PSS_bytes'],
            verification_seconds=verified['elapsed_seconds'],
            verification_sampled_tree_PSS_bytes=verified['sampled_peak_tree_PSS_bytes'],
            verification_peak_allocated_bytes=numerical['peak_allocated_bytes'],
            verification_peak_reserved_bytes=numerical['peak_reserved_bytes'],
            physical_before_after_receipt=ref(HERE/'original-fa-check.json'),
            scope='One primitive process on an idle physical GPU4; no model allocation or capacity extrapolation.'),
        scientific_scope=dict(
            one_point_generalization_retracted=True,
            grouping='Keep task/state/trajectory/exposure/minibatch and separate uniform cohort from predicted-tail census; retain joint DT/native ratio bins.',
            primary='Original author cumulative deletion, signed RISE and positive MAS; per-state aggregation and paired UID comparisons remain unchanged.',
            no_pooled_heavy_tail_moments=True,
            no_new_quality_scores=True,
            no_production_repair_selected=True,
            next_research_claim='Nonzero local semantics and complete runner/gate composition must be checked before any frozen-development quality comparison. This receipt does not justify deploying the candidate or calling full attention/GDN the main cause.',
            existing_cost_warning='The algebra/cost receipt compares against both current repeated-LSE work and potential native-LSE reuse. No speed claim follows from contraction counts.'),
        formal_state=dict(textcraft=compiled['formal_textcraft'], appworld_restarted=False,
                          production_modified=False, optimizer_steps=0,
                          model_calls=0, rollouts=0, checkpoint_restores=0))
    destination=REPO/'experiments/rl/results_conditional_attention_owner_20261008.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    runtime_path=REPO/'experiments/rl/current_runtime.json'
    runtime=read(runtime_path)
    runtime['latest_conditional_attention_owner_research']=dict(
        receipt=ref(destination), status=result['status'],
        deployed=False, library_sha256=compiled['library_sha256'],
        unchanged_default=True, official_coincident_derivative='passed',
        collection_quality='Not measured for this candidate',
        runner_gate_composition='Not connected', formal_state=result['formal_state'])
    runtime_path.write_bytes((json.dumps(runtime,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps(dict(receipt=ref(destination), candidate=compiled['library_sha256'],
                         official_primitive=numerical['status'], deployed=False),ensure_ascii=False))


if __name__=='__main__':
    main()
