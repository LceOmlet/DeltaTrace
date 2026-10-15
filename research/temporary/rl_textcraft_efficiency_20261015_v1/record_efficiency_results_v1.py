"""Summarize measured cost, exact owner parity, and actual deployment evidence."""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
raw=json.loads((HERE/'actual-DT-breakdown-and-head-backward.json').read_bytes())
bench=json.loads((HERE/'FA-native-LSE-source.json').read_bytes())['benchmark']['head-forward-benchmark.json']
latest=json.loads((HERE/'head-deployment-progress.json').read_bytes())
profiles=[]
for name,value in raw['files'].items():
    if 'existing_owner_batches' not in value:continue
    totals=collections.defaultdict(float)
    for batch in value['existing_owner_batches']:
        for call in batch['owner_info']['calls']:
            key=call['kind']
            if key.startswith('native_replay_'):key='native_layer_replay'
            elif key.startswith('finite_decoder_'):
                family=batch['owner_info']['layers'][key.rsplit('_',1)[1]]['block_type']
                key='finite_FA_decoder' if family=='full_attention' else 'finite_GDN_decoder'
            elif key.startswith('public_FA_LSE_'):key='nested_public_FA_LSE'
            elif key.startswith('restore_mixer_captures_'):key='nested_restore_FA_captures'
            totals[key]+=call.get('stream_elapsed_seconds',call.get('seconds',0))
    profiles.append(dict(rank=value['rank'],pid=value['pid'],birth=value['birth'],
        completed_iteration_before_DT=value['scheduler_before_DT'],native_B4_calls=len(value['existing_owner_batches']),
        wall_seconds=value['existing_readout_report']['seconds'],
        prefix=value['existing_readout_report']['shared_native_prefix'],CUDA_stream_sections_seconds=dict(totals),
        nested_times_are_included_in_finite_FA_decoder=True,
        CUDA_stream_elapsed_includes_waits_not_just_kernel_execution=True,
        observer_restored=value['original_attribute_restored'] and value['original_trajectories_restored'],
        source_receipt=ROOT+'/receipts/textcraft-efficiency-20261015-v1/'+name))
dimensions=latest['model_dimensions']
mlp_nominal_bytes=2*8*32768*dimensions['intermediate_size']*2*3*dimensions['num_hidden_layers']
deployments={label:latest['deployments'].get(label+'-frozen-head-deployment.json') for label in ('DTPO','PPO')}
status='both_current_actors_verified_deployed' if all(deployments.values()) else 'DTPO_deployed_PPO_waiting_existing_RPC_boundary'
report=dict(recorded_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),status=status,
    scope='Current DTPO v3 and shared original VERL policy-update efficiency; paused gradient research not resumed.',
    version='textcraft-native-frozen-head-efficiency-20261015-v1',patch_commit='7a33e61a',
    baseline_owner_SHA256='e285c3353bddbffae38df346b44014ee5038b606531874ddbeb10ee1241f77de',
    patched_owner_SHA256='bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2',
    observed_defect='Original fused-head chunk helper computed a dense frozen vocabulary-weight gradient; its original autograd caller discarded that result.',
    minimal_owner_fix='Pass the existing vocab_weights.requires_grad condition into the original compiled helper and skip only the unused weight-gradient matrix product.',
    original_forward_and_needed_hidden_gradient_expressions_unchanged=True,
    original_actor_core_unchanged=True,no_shadow_head_or_PPO_implementation=True,
    tolerance=dict(kind='exact_value_equality',rtol=0,atol=0,
        reason='Removed output has no consumer; all needed expressions and their dtype/rounding boundaries remain unchanged.',
        actual_native_hidden_fixture=True,fixed_nonzero_operator_cotangents=True,
        tested_public_autograd_branches=bench['public_autograd_checks'],
        needed_hidden_gradient_equal=bench['hidden_gradient_equal'],maximum_abs_difference=bench['hidden_gradient_max_abs_diff'],
        no_generic_project_tolerance_imported=True,
        limitation='These are exact operator-fixture comparisons, not a universal floating-point error bound or full future-training bit-identity claim.'),
    isolated_head_cost=dict(physical_GPU=2,median_seconds_per_chunk=bench['median_seconds_per_chunk'],
        chunk_backward_speed_ratio=bench['chunk_backward_speed_ratio'],
        removed_unused_matrix_bytes=dimensions['vocab_size']*dimensions['hidden_size']*2,
        full_update_or_training_speedup_measured=False),
    DT_existing_timing_profiles=profiles,
    no_repeated_per_action_DT='Actual current adapter deduplicates trajectory identities and issues one joint true-target attribution per required trajectory, packed B4.',
    forward_reuse_boundaries=[
        'DT needs paired reference/factual contexts and intermediate finite-rule operands; old-logprob tensors do not provide those artifacts.',
        'The reference policy disables LoRA; it is a different conditional probability from the active actor.',
        'Original PPO forward/backward after each optimizer update cannot be replaced by old-policy forward outputs.'],
    DT_replay_memory_constraint=dict(model_dimensions=dimensions,nominal_endpoint_rows_per_rank=8,context_limit=32768,
        all_32_layer_gate_up_silu_nominal_BF16_bytes_both_ranks=mlp_nominal_bytes,
        cgroup_limit_bytes=966367641600,
        scope='A naive retain-all strategy for the current three MLP captures alone exceeds the existing host limit at 32k; this is not a proof that bounded retention cannot improve it.'),
    further_findings=[
        'The duplicate public FA LSE calls cost approximately 2.68 seconds inside a 509-second DT RPC; they are not the large bottleneck.',
        'Major DT costs are native root capture, native per-layer replay, and the existing finite decoder propagation. No second large removable duplicate has been verified.',
        'LoRA reference computation calculates an unused entropy; no deployment or saving claim is made for this separate candidate.'],
    training_hyperparameter_credit_changes=0,optimizer_scheduler_calls_added=0,
    training_restarts=0,GRPO_restarts=0,gradient_rootcause_goal_resumed=False,
    deployments=deployments,
    evidence_files={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        HERE/'actual-DT-breakdown-and-head-backward.json',HERE/'FA-native-LSE-source.json',HERE/'head-deployment-progress.json')})
selection=json.loads((HERE/'selection-progress.json').read_bytes())
checks=selection['receipts']['head-selection-verification.json']
boundary=selection['receipts']['selected-head-worker-verification.json']
assert checks['phase']=='completed' and checks['all_needed_outputs_gradients_exact']
assert boundary['RNG_restored'] and boundary['scheduler_before']==boundary['scheduler_after']
actual,_,full=checks['checks']
report['head_selection_candidate']=dict(
    version='textcraft-native-head-selection-20261015-v1',
    status='verified_not_deployed_no_measured_current_TextCraft_benefit',
    deployed=False,
    observed_gap='Pinned Torch wrapper ignored the existing actor logits_to_keep metadata.',
    owner_extension='Visit only the original512-token chunks intersecting the requested rows, retaining original formulas, shapes, rounding and order for every visited chunk.',
    actual_native_fixture=actual,
    full32768_head_capacity_fixture=full,
    full32768_fixture_is_not_a_natural_task_or_full_training_speedup=True,
    tolerance=checks['tolerance'],
    source_SHA256=checks['sources'],
    verification_worker=boundary,
    default_trainable_head_compatibility=checks['checks'][1],
    decision='Actual B4 context5376/keep5057 still visits42 forward and42 backward chunks: median0.759679 versus0.760457 seconds. The context32768/keep10241 fixture reduces256 to84 chunks and4.626371 to1.520755 seconds, but this long-prompt saving does not describe the actual315-token TextCraft prompts.',
    receipt=ROOT+'/receipts/textcraft-efficiency-20261015-v1/head-selection-verification.json')
prior=json.loads((REPO/'experiments/rl/results_textcraft_native_graph_deployment_20261012.json').read_bytes())
report['context32768_feasibility']=dict(
    DT_and_actor_existing_full_capacity=prior['capacity'],
    full_capacity_source='experiments/rl/results_textcraft_native_graph_deployment_20261012.json',
    unchanged_actual_DT_memory_binding=prior['memory_integration']['gate_capture_host_fix']['changed_files'],
    current_patch_only_removes_unused_head_weight_gradient=True,
    no_context_microbatch_or_32k_memory_tradeoff_changed=True,
    current_head_B4_32768_needed_outputs_and_VJP_exact=True,
    new_full_DT_actor_capacity_run_this_turn=False,
    native_PPO_critic_full32768_capacity_claimed=False)
report['further_findings']=[
    'The duplicate public FA LSE calls cost approximately2.68 seconds inside a509-second DT RPC; they are not the large bottleneck.',
    'Major DT costs are native root capture, native per-layer replay, and existing finite decoder propagation. No second large removable duplicate has been verified.',
    'Current TextCraft prompts are315 tokens. The chunk-preserving logits_to_keep candidate has exact outputs/VJP at32768 but no measured speed gain on the actual full B4 head artifact, and remains outside the launch path.',
    'No all-layer retention or modified LoRA finite arithmetic was introduced to obtain a benchmark improvement.']
report['head_verification_completed_at']=datetime.fromtimestamp(boundary['completed_unix'],ZoneInfo('Asia/Shanghai')).isoformat()
report['head_selection_candidate']['source_patch_commit']=subprocess.run(
    ['git','log','-1','--format=%H','--','experiments/rl/patch_actor_head_selection.py'],
    cwd=REPO,check=True,capture_output=True,text=True).stdout.strip()
report['pinned_VERL_upstream_commit']='20bd331b'
report['unchanged_DTPO_method_source_patch_commit']='f726ee72'
report['unchanged_DTPO_method_deployment_commit']='e05e23cb1bc1cccade732ad192ce15b7e9dc6a1e'
report['evidence_files'][str((HERE/'selection-progress.json').relative_to(REPO))]=hashlib.sha256((HERE/'selection-progress.json').read_bytes()).hexdigest()
final=json.loads((HERE/'final-efficiency-readout.json').read_bytes())
assert final['no_candidate_in_current_launch_path']
for path,sha in report['context32768_feasibility']['unchanged_actual_DT_memory_binding'].items():
    assert final['DT_actual_file_SHA256'][path]==sha
report['final_actual_binding_readout']=dict(unix=final['unix'],current_jobs=final['current_jobs'],
    DT_actual_file_SHA256=final['DT_actual_file_SHA256'],
    resource_receipt=str((HERE/'final-efficiency-readout.json').relative_to(REPO)))
report['evidence_files'][str((HERE/'final-efficiency-readout.json').relative_to(REPO))]=hashlib.sha256((HERE/'final-efficiency-readout.json').read_bytes()).hexdigest()
path=REPO/'experiments/rl/results_textcraft_efficiency_20261015_v1.json'
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(path=str(path),status=status,deployments=[k for k,v in deployments.items() if v],
    native_MLP_capture_nominal_GiB_both_ranks=mlp_nominal_bytes/2**30),ensure_ascii=False))
