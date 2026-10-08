"""Bind saved operator verification and isolated collection to exact sources."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def file(path):
    path=Path(path)
    raw=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-kind',choices=('conditional_gdn','conditional_mixers'),default='conditional_gdn')
    args=parser.parse_args()
    label=args.candidate_kind
    folder=HERE/(label.replace('_','-')+'-collection-v1')
    mixers=label=='conditional_mixers'
    tiled=json.loads((HERE/'tiled-conditional-memory-result.json').read_bytes())
    seam=json.loads((HERE/'conditional-owner-seams-v2-result.json').read_bytes())
    launch=json.loads((folder/'textcraft-launch.json').read_bytes())
    source_names=['tiled_conditional_memory.py','conditional_gdn_context.py',
        'conditional_conv_windows.py','conditional_gdn_candidate.py',
        'conditional-gdn-owner-prepared/qwen35_gdn_finite.py',
        'conditional-gdn-owner-prepared/finite_fla_gpu.py',
        'conditional-attention-owner-v1/inspect_conditional_collection.py',
        'conditional-attention-owner-v1/submit_conditional_collection.py']
    if mixers:
        source_names += ['prepare_conditional_mixers.py','conditional-mixers-derivation.json',
            'conditional-attention-owner-v1/conditional_attention_endpoints.py',
            'conditional-attention-owner-v1/composition-prepared/textcraft/qwen35_decoder_finite.py',
            'conditional-attention-owner-v1/composition-prepared/textcraft/qwen35_dense_finite_runner.py']
    observations=sorted(folder.glob('textcraft-observe-*.json'))
    value=json.loads(observations[-1].read_bytes())
    records=value['records']
    complete=len(records)==2 and all(r['phase']=='complete' for r in records.values())
    analysis_path=folder/'textcraft-analysis.json'
    analysis=json.loads(analysis_path.read_bytes()) if complete and analysis_path.is_file() else None
    result=dict(status='isolated_conditional_GDN_completed_insufficient_repair_not_deployed' if complete else 'isolated_conditional_GDN_full_frozen_collection_running_not_accepted',
        observed_unix=value['unix'],PPO_debug=file(REPO/'experiments/rl/results_preserved_actor_debug_20261008.json'),
        PPO_original_NaN_repaired=False,
        prior_joint_background_evidence=file(REPO/'experiments/rl/results_single_background_completed_20261009.json'),
        tiled=dict(launch=file(HERE/'tiled-conditional-memory-launch.json'),
            result=file(HERE/'tiled-conditional-memory-result.json'),
            original_FLA_derivative_checks=sum(len(c['derivative_checks']) for c in tiled['cases']),
            all_original_derivative_checks_passed=all(d['passed'] for c in tiled['cases'] for d in c['derivative_checks']),
            finite_scalar='Changes reported against prior untiled values, not called an official finite-scalar tolerance.'),
        seams=dict(launch=file(HERE/'conditional-owner-seams-v2-launch.json'),
            result=file(HERE/'conditional-owner-seams-v2-result.json'),
            default_FLA_bitwise=seam['unchanged_fla_default_bitwise'],
            conv_cases=seam['cases'],peak_allocated=seam['peak_allocated_bytes'],
            previous_failed_layout='v1 original-reference initial_states.stride(1) was wrong; source, command and failure remain preserved, as does the remote v1 candidate source. No threshold changed.'),
        sources={name:file(HERE/name) for name in source_names},
        collection=dict(launch=file(folder/'textcraft-launch.json'),
            comparison_inputs=file(folder/'comparison-inputs-textcraft.json'),
            latest_observation=file(observations[-1]),pid=launch['pid'],birth=launch['birth'],
            alive_at_snapshot=value['driver_alive'],devices=launch['devices'],
            code_commit_at_launch=launch['code_commit'],
            actual_launch_source_SHA256=launch['scripts'],
            rank_phases={rank:{k:r.get(k) for k in ('phase','batch','elapsed_seconds','operations','traceback')}
                for rank,r in records.items()},
            candidate_label=label,frozen_primary_trajectories=32,initial_states=16,
            extra_tail_trajectories=13,original_baseline_reused=True,
            native_forward_calls_per_rank=launch['native_calls_per_rank'],
            expected_DT_calls_per_rank=launch['diagnostic_DT_B4_calls_per_rank'],
            complete=complete,whole_DT_repair_accepted=False,
            physical_resources=value['physical'],host_available_bytes=value['host']['available']),
        official_owners='Original installed convolution/l2norm/FLA readout and actual runner/model/capture/answer owners; default-inert GDN extension only in research imports.',
        method_unchanged='Q/V/A, joint actual action target, observation mask, whole-batch whitening, upstream PPO, LoRA8/16, per-card B4.',
        formal_training_started=False,checkpoint_restore=False,production_modified=False,
        scope='Operator seam acceptance is separate from the frozen collection author curves and tail accuracy; neither proves a whole-method repair.')
    if mixers:
        result['status']='isolated_conditional_mixers_'+('completed_unaccepted_not_deployed' if complete else 'frozen_collection_running_not_accepted')
        result['FA_verification']=file(REPO/'experiments/rl/results_conditional_attention_composition_20261008.json')
        result['official_owners']='The same previously checked conditional FA and GDN seams, actual native HF/PEFT/capture owners, and unchanged pointwise/head rules; only the explicit research runner binds both mixer callbacks.'
    if analysis is not None:
        result['collection'].update(analysis=file(analysis_path),
            primary_metrics={k:{n:v for n,v in m.items() if n in ('paired_finite','available_equal_state_means','state_delta_standard_error')}
                for k,m in analysis['primary_metrics'].items()},
            paired_factual_scores=analysis['paired_factual_scores'],
            robust_tail_counts={k:v for k,v in analysis['robust_diagnostic_points'].items() if k in
                ('original_spurious_tail_points','still_spurious_tail','still_opposite_sign','original_uniform_missed_tail_points','missed_tail_still_outside_tail')},
            DT_cost_ratio=analysis['DT_cost']['total_ratio'],conclusion=analysis['conclusion'],
            AppWorld_candidate_run=False,heldout_candidate_run=False)
        preserved=folder/'textcraft-preserved.json'
        if preserved.is_file():result['collection']['preserved_vectors_and_phase_logs']=file(preserved)
    receipt=REPO/('experiments/rl/results_'+label+'_20261009.json')
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    snapshot=REPO/'experiments/rl/current_runtime.json'
    current=json.loads(snapshot.read_bytes())
    current['observed_unix']=value['unix']
    current['observed_utc']=datetime.fromtimestamp(value['unix'],timezone.utc).isoformat()
    current['latest_readonly_observation']=dict(textcraft='Formal training stopped',appworld='Formal training stopped',
        diagnostic=result['status'],receipt=file(receipt))
    current['isolated_'+('conditional_mixers' if mixers else 'conditional_GDN')+'_diagnostic']=result['collection']
    snapshot.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(receipt=file(receipt),status=result['status'],derivative_checks=result['tiled']['original_FLA_derivative_checks'])))


if __name__=='__main__':main()
