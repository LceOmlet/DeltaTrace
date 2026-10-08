"""Record completed primitive checks without promoting a prepared runner."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    local = read(HERE / 'composition-local.json')
    derivative = read(HERE / 'composition-derivative.json')
    prepared = read(HERE / 'composition-prepared.json')
    assert local['returncode'] == derivative['returncode'] == 0
    assert derivative['numerical']['status'] == 'passed'
    assert all(derivative['numerical']['unchanged_default'].values())
    assert local['formal_textcraft']['release_exists'] == [False, False]
    assert derivative['formal_textcraft']['release_exists'] == [False, False]
    assert prepared['helper_sha256'] == ref(HERE / 'conditional_attention_endpoints.py')['sha256']
    assert prepared['patch_sha256'] == ref(HERE / 'composition.patch')['sha256']
    for receipt in [local, derivative]:
        assert receipt['launch']['library_sha256'] == prepared['library_sha256']
        assert receipt['launch']['source_sha256']['conditional_attention_endpoints.py'] == prepared['helper_sha256']
        assert receipt['launch']['source_sha256']['qwen35_decoder_finite.py'] == prepared['tasks']['textcraft'][0]['generated_sha256']
    verifier = REPO / 'experiments/rl/verify_saved_fa_dtypes.py'
    assert derivative['launch']['source_sha256'][verifier.name] == ref(verifier)['sha256']
    n = local['numerical']
    result = dict(
        observed_unix=time.time(),
        prepared_from_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        status='Primitive composition verified; complete model runner prepared only; no method-quality or production claim',
        source=[ref(HERE / name) for name in ['conditional_attention_endpoints.py', 'prepare_composition.py', 'composition.patch', 'composition-prepared.json', 'verify_conditional_finite_local.py', 'check_nonzero_local.py', 'record_composition.py']] + [ref(verifier)],
        evidence=[ref(HERE / name) for name in ['native-padding-owner.json', 'nonzero-local.json', 'composition-local.json', 'composition-derivative.json', 'nonzero-local-interface-failure.json', 'nonzero-local-interface-failure-worker.py', 'nonzero-local-dtype-failure.json', 'nonzero-local-dtype-failure-worker.py']],
        prepared=prepared,
        official_derivative=dict(assertions=derivative['numerical']['assertions'], checks=derivative['numerical']['checks'], unchanged_default=derivative['numerical']['unchanged_default'], source=derivative['numerical']['source'], query_shape=derivative['numerical']['query_shape'], key_shape=derivative['numerical']['key_shape'], scope=derivative['numerical']['candidate_scope']),
        nonzero_local=dict(selection=n['selected'], actual_shape=n['actual_shape'], original_reference=n['original_reference'], abs_error=n['abs_error'], relative_L2=n['relative_L2'], sign_crossings=n['sign_crossings'], nonfinite=n['nonfinite'], composition=n['composition'], scope='Complete first 64 eligible positions in each of four saved operand rows. Diagnostic only; not frozen-collection quality and no new finite-effect tolerance.'),
        resources={name: dict(elapsed_seconds=receipt['elapsed_seconds'], sampled_peak_tree_PSS_bytes=receipt['sampled_peak_tree_PSS_bytes'], peak_allocated_bytes=receipt['numerical']['peak_allocated_bytes'], peak_reserved_bytes=receipt['numerical']['peak_reserved_bytes'], remote_directory=receipt['remote_directory'], launch=receipt['launch']) for name, receipt in [('local_composition', local), ('official_derivative', derivative)]},
        failures=[dict(stage='Nonzero worker boundary', cause='Worker initially used query_start=433 rather than the existing 64-aligned 384 boundary; corrected the worker and retained the original coefficient cut.'), dict(stage='Nonzero worker dtype', cause='Worker initially passed BF16 to the original FP32 upstream ABI; corrected the worker, preserving the original kernel BF16 conversion.')],
        scope=dict(public_FA_and_padding='Original installed functions; no replacement inference or training implementation.', candidate_change='Default-inert extension of the existing finite owner; compiled own-key terms and public varlen FA share a conditional background.', generated_copies='Prepared artifacts only; generator and differential patch are the recorded source.', gate_and_complete_runner='Prepared, not executed or deployed.', collection_quality='Not measured for this candidate; frozen task/state/trajectory split, ratio bins and original cumulative-deletion/RISE/MAS remain unchanged.', GDN_main_cause='Not established; prior single-point generalization remains retracted.', no_credit_correction=True, no_clipping=True, no_model_calls=True, no_optimizer_steps=True),
        formal_state=dict(textcraft=derivative['formal_textcraft'], appworld_restarted=False, deployed=False, checkpoint_restores=0))
    destination = REPO / 'experiments/rl/results_conditional_attention_composition_20261008.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    runtime_path = REPO / 'experiments/rl/current_runtime.json'
    runtime = read(runtime_path)
    runtime['latest_conditional_attention_composition_research'] = dict(receipt=ref(destination), observed_unix=result['observed_unix'], status=result['status'], library_sha256=prepared['library_sha256'], official_derivative='passed', nonzero_local_relative_L2=n['relative_L2'], full_runner='Prepared only', collection_quality='Not measured', formal_state=result['formal_state'])
    runtime_path.write_bytes((json.dumps(runtime, ensure_ascii=False, indent=2) + '\n').replace('\n', '\r\n').encode())
    print(json.dumps(dict(receipt=ref(destination), deployed=False, official_derivative='passed'), ensure_ascii=False))


if __name__ == '__main__':
    main()
