"""Persist already-observed CPU/source evidence; never imports Torch or uses SSH."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
VENV = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env'


def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def official(name, url, expected):
    raw = urlopen(url, timeout=15).read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Official source identity changed: {name}: {actual}')
    path = HERE / name
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(f'Refusing to overwrite source: {path}')
    if not path.exists():
        path.write_bytes(raw)
    return dict(identity(path), url=url, exact_upstream_bytes=True), raw.decode('utf-8')


def main():
    interface, interface_text = official(
        'causal_conv1d_interface_v150.py',
        'https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.5.0/causal_conv1d/causal_conv1d_interface.py',
        '7286f93996561532dac8452de1e75bd44a31baf5acbb3ad4fd48b87fbc7fdc94')
    tests, test_text = official(
        'test_causal_conv1d_v150.py',
        'https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.5.0/tests/test_causal_conv1d.py',
        'c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf')
    test_ast = ast.parse(test_text)
    test_fn = next(n for n in test_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'test_causal_conv1d')
    assertions = [dict(line=n.lineno, source=ast.get_source_segment(test_text, n))
                  for n in ast.walk(test_fn) if isinstance(n, ast.Assert)]
    owner_bytes = subprocess.check_output(['git', 'show', '8e7dd71:deltatrace/clean/qwen35/qwen35_gdn_finite.py'], cwd=REPO)
    owner_sha = hashlib.sha256(owner_bytes).hexdigest()
    if owner_sha != 'fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca':
        raise RuntimeError('Pinned GDN owner identity differs')
    refs = [
        'experiments/rl/PLAN.md', 'experiments/rl/RUNTIME_RECORD.md',
        'experiments/rl/REMOTE_ENVIRONMENT.md',
        'research/temporary/rl_upstream_alignment_20260929/appworld-efficiency-20261006/retained-action-transition-v2/current_runtime.json',
        'research/temporary/rl_upstream_alignment_20260929/observe_native_gdn0_operands.py',
        'experiments/rl/results_native_root_tape_gdn0_20261004.json',
        'experiments/rl/results_native_root_tape_gdn0_operands_20261004.json',
        'research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/owner-sources/modeling_qwen3_5.py',
        'research/temporary/rl_upstream_alignment_20260929/textcraft-degradation-20261005/owner-sources/source.json']
    report = dict(
        status='cpu_source_audit_complete_awaiting_actual_conv_operands',
        scope='Installed owner interface/source and existing actual-asset inventory only; not a method blocker or operator acceptance.',
        operations=dict(gpu_calls=0, model_loads=0, extra_model_forwards=0, dt_calls=0,
                        optimizer_steps=0, package_installs=0, production_edits=0),
        receipt_writer=identity(__file__), source_receipts=[identity(REPO / p) for p in refs],
        installed=dict(
            torch='2.8.0+metax3.5.3.9', causal_conv_distribution='1.5.0.post8+metax3.5.3.9torch2.8',
            interface=dict(path='/opt/conda/lib/python3.12/site-packages/causal_conv1d/causal_conv1d_interface.py',
                sha256=interface['sha256'], exact_python_bytes_equal_official_v150=True),
            extension=dict(path='/opt/conda/lib/python3.12/site-packages/causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so',
                sha256='ba1a7ad3889327b10f04a6a722cdfda8a0b453ce4f9796c43f8d05d3bdee36a8'),
            package_metadata_sha256='d3a33e5e6b461e3971d6dbbf4c870dd91f0e21bb9a0dfe303e4965609cb7293c',
            wheel_sha256='ec33c297f2e5bf08fcbde4120192b09c442bb247941cd551a43f76fbaf571f2d',
            probe_scope='Earlier read-only CPU import, CUDA_VISIBLE_DEVICES empty, MACA_VISIBLE_DEVICES=-1; torch.cuda.is_initialized false before/after. Not a live worker function inspection.'),
        official_sources=dict(interface=interface, tests=tests),
        public_contract=dict(
            signature='causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None, return_final_states=False, final_states_out=None, activation=None)',
            initial_states_shape='[batch, dim, width-1]',
            implementation='Original CausalConv1dFn.apply and causal_conv1d_cuda fwd/bwd; original causal_conv1d_ref retained for comparison.',
            constraints=['seq_idx excludes initial_states/final-state return',
                         'Original test covers initial_states only in channel-last layout; copy/stride handling remains owner code.'],
            original_bf16_tolerance=dict(rtol=0.01, atol=0.05, source_lines=[43, 44],
                applies_to=['forward output', 'input gradient', 'initial-state gradient']),
            original_weight_gradient_tolerance=dict(rtol=0.001, atol=0.001, source_lines=[45, 100]),
            original_test_function=dict(name=test_fn.name, line=test_fn.lineno, end_line=test_fn.end_lineno,
                assertions=assertions, executed=False),
            limits='Python interface byte equality does not validate MetaX native binary numerics. No causal-conv operator check has run in this task; do not borrow FA/FLA tolerances.'),
        native_model_branch=dict(
            path=VENV + '/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py',
            sha256='f7e1a804fa12684bd1cc225c85cdf5f0b5996f30d66263f11de13449f53272be',
            cache_read_lines=[457, 462], projection_lines=[464, 465],
            cached_multi_cat_line=488, original_cache_update_lines=[490, 491],
            public_fn_lines=[492, 499], current_suffix_crop_line=503,
            single_token_update_lines=[473, 481],
            context='Original HF cached multi-token branch prepends conv_state, then crops the output to seq_len. This is the short convolution state, not the entire 12k token history.',
            import_mask_caveat='is_causal_conv1d_available requires torch CUDA availability. A hidden-device CPU import reporting fn=None/fast_path=False cannot establish that the running workers use fallback F.conv1d.',
            import_utils=dict(path=VENV + '/lib/python3.12/site-packages/transformers/utils/import_utils.py',
                sha256='f3c64422e98644451a5c5a4eb034b91dece3e915bff77a914f691c916ef6d3a6')),
        dt_left_window_contract=dict(
            git_source='8e7dd71:deltatrace/clean/qwen35/qwen35_gdn_finite.py', sha256=owner_sha,
            deployed_event_source=ROOT + '/releases/c9cd147/clean/qwen35/qwen35_gdn_finite.py',
            accelerated_capture=ROOT + '/releases/c9cd147/accelerated/qwen35/qwen35_code_local_capture.py',
            accelerated_capture_sha256='993ae72a49dc90901bfcbb7817e5fd32ad6004fbe630a190b775c6f1e77fbe72',
            conv_call_lines=[88, 95], conv_context_start_line=49,
            finite_preactivation_autograd_lines=[268, 282],
            actual_dependence='Capture derives cached_conv_context from concatenated x minus current input T, retains the needed left window, crops native conv_output; finite calls the same public fn with activation=None, zeros leading seed positions and returns current-token input gradients.',
            initial_states_conversion='The public initial_states width is K-1, while HF stores K conv-state positions. Any future seam must preserve the exact original crop and compact finite left window, rather than pass the entire cache directly or change cache updates.',
            capture_disables_native_conv=False, source_reason='Capture only observes the original public fn; the finite preactivation and autograd also call that fn.'),
        existing_actual_asset_audit=dict(
            directory=ROOT + '/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-local-prefix-root-tape-gdn0-20261004-a2dbbc5/actual-gdn0-operands',
            observed_files=6,
            inspected_serialized_asset=dict(name='gdn0-shared_warm-rank0-call1.pt', bytes=467050864,
                sha256='a039303ded9bf7a5a169e00e2eae40b0e166830b1ea3afd37346001e8de0200e',
                data_pickle_sha256='a49fcdb6506610c68cb4b4319e06e589426f499ac55752ff1e3948d470d8b7b0',
                observed_unix=1791302236.3781693, verification='Stream SHA with 1MiB reads; ZipFile data.pkl pickletools string inspection; no tensor unpickle or Torch import.',
                peak_rss_kib=20136),
            saved_tensor_fields=['raw_q', 'raw_k', 'raw_v', 'raw_g', 'raw_beta', 'fla_initial_state',
                'fla_cu_seqlens', 'initial_state', 'cu_seqlens', 'stage_o', 'stage_final_state',
                'stage_k', 'stage_w', 'stage_u', 'stage_g', 'stage_h', 'native_o', 'native_ht', 'model_boundary_o'],
            missing=['native projected_qkv/conv x', 'actual convolution left state', 'actual conv weight/bias',
                     'original finite preactivation autograd seed'],
            warning='capture_calls.conv=1 is an executed-call count, not a saved convolution operand. The limited filename scan was not an exhaustive server inventory.',
            decision='Do not fabricate inputs, load another model, stop formal training, or run an operator diagnosis from the saved FLA-only assets.'),
        prior_physical_observation=dict(
            observed_unix=1791302079.4298244, evidence_origin='Earlier bounded read-only tool observation in this task, not refreshed by this writer.',
            app_pid=3592468, expected_birth=1791291997.14, actual_birth=1791291997.14,
            gpu2=dict(total_mib=65536, used_mib=54272, free_mib=11264),
            gpu3=dict(total_mib=65536, used_mib=54272),
            requested_minimum_free_gib=8, requested_maximum_increment_gib=2,
            run_skipped_reason='Real convolution input/seed absent; resource observation alone does not authorize synthetic inputs.',
            not_current_reservation=True),
        passive_capture_design=dict(
            status='Design only; no observer written or attached to live jobs.',
            seam='Extend the existing isolated GDN0 capture_backend collector NativeGDNCapture.event; call original event first and return unchanged.',
            native_event='At its original conv call, save only the first actual paired-B8 GDN0 x, weight, bias/activation/seq_idx metadata and original input_shape/coefficient_start/cached_conv_context. Retain exact dtype/stride/bytes and call identity.',
            finite_seed_event='At the already-executing public torch.autograd.grad call inside the original GDN finite owner, observe its caller projected/pre/seed and compact context; save those actual operands once. Do not invoke another forward/backward.',
            followup='Use the same installed causal_conv1d_fn and original causal_conv1d_ref/assertions on the actual saved operands. Record concat/crop versus initial_states outputs and input gradients, hot boundary cost and physical/allocated peak separately.',
            no_changes=['No cache/state algorithm copy', 'No HF default patch', 'No live monkeypatch',
                        'No DT/PPO/FA/FLA/LoRA/microbatch/config change', 'No fabricated fixture'],
            interpretation='A future successful primitive check would establish this owner seam only, not full DT acceptance or formal wall-clock speed.'),
        limitations=['No convolution outputs, input gradients or costs were measured in this task.',
                     'Existing source and observed cat cost motivate a candidate seam, not a proved end-to-end speedup.',
                     'Operand capture is a next actual-call diagnostic observation, not a blocker of the accepted method or ongoing formal jobs.'])
    output = HERE / 'cpu-readonly-receipt.json'
    if output.exists():
        raise RuntimeError(f'Refusing to replace a frozen receipt: {output}')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(identity(output)))


if __name__ == '__main__':
    main()
