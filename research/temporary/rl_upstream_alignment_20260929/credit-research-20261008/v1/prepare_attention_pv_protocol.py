"""Derive a passive PV ledger from the measured finite FA core discrepancy.

No numerical rule is proposed here. Native mixed products use the original
public FA implementation and actual dense arguments on the frozen collection.
"""
import json
from pathlib import Path
import sympy as sp

from prepare_attention_gate_protocol import ref

HERE=Path(__file__).resolve().parent


def main():
    result=json.loads((HERE/'attention-input-protocol.json').read_bytes())
    cq,ck,cv,of,od,ofr,odr,ofd,u=sp.symbols('Cq Ck Cv O_F O_D O_FR O_DR O_FD u')
    terms=dict(joint_QK_softmax_residual=cq+ck-u*(ofr-odr),
        factual_P_value_residual=cv-u*(of-ofd),
        PV_background_residual=u*((ofr-odr)-(ofd-od)))
    residual=sp.expand(sum(terms.values())-(cq+ck+cv-u*(of-od)))
    assert residual==0
    result.update(scope=__doc__,read_attention_input=True,read_attention_pv=True,
        arithmetic_PV=dict(terms={k:str(v) for k,v in terms.items()},symbolic_residual=str(residual),
            native_pair='F and single-delete D from the same original native call',
            reference='Original joint reference V_R captured by the unchanged DT',
            original_FA_calls='1: paired native Q/K and repeated V_R. 2: repeated factual Q/K and paired native V_D/V_F.',
            rounding='Ledger uses actual BF16 FA outputs. PV background includes applicable native arithmetic; '
                'OFF versus original native OF is retained without an invented pass threshold.',
            factual_drift='Report matched DT-factual core minus this native-pair core separately; do not correct it.'),
        reason='The completed input probe left finite_FA_core largest in every one of the nine historical-paired-unchanged '
            'spurious tail positions across five states. Distinguish QK/softmax propagation from PV background '
            'and factual-P value arithmetic before proposing another owner rule.',
        prepared_only=True,launched=False,candidate=False,method_changed=False,production_modified=False,
        original_FA_FLA_tolerance_changed=False)
    paths=[Path(x['path']) for x in result['inputs']]
    paths += [HERE/'passive_attention_pv.py',Path(__file__),HERE/'attention-input-textcraft/analysis.json',
        HERE/'finite-FA-source.json',HERE/'stage_layer_factual_controls.py',HERE/'launch_layer_factual_controls.py']
    result['inputs']=[ref(p) for p in dict.fromkeys(paths)]
    cfg=json.loads((HERE/'gdn-owner-readonly.json').read_bytes())['model_config']['text_config']
    plan=json.loads((HERE/'layer-collection-inputs.json').read_bytes())
    H,K,D=cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim']
    for task,cost in result['costs'].items():
        N=cost['longest_actual_frozen_context']
        cost.update(additional_public_FA_calls_per_original_native_forward=2,
            additional_public_FA_calls_per_rank_upper=2*max(cost['original_native_calls_per_rank']),
            additional_whole_model_calls=0,additional_DT_calls=0,
            additional_live_device_shape_upper_bytes=8*N*D*2*((H+2*K)+(H+K)+K+H),
            additional_mixed_output_host_upper_bytes=2*8*N*H*D*2,
            additional_repeated_reference_value_host_upper_bytes=8*N*K*D*2,
            extra_memory_scope='Upper for retained native Q/K/V, repeated factual Q/K, repeated reference V, '
                'and one mixed output; excludes native FA workspace. No N-by-N allocation. '
                'Same data is released after each native round; actual peak and phase time must be recorded.',
            cost_basis='Two original dense FA calls per already scheduled native comparison, not two whole-model calls. '
                'No new sample, per-source model loop or changed diagnostic worker deadline.')
        entries={e['traj_uid']:e for e in plan['tasks'][task]['entries']}
        batches=plan['tasks'][task]['batches']
        joint_bytes=native_bytes=0
        for i,batch in enumerate(batches):
            extent=max(entries[uid]['selected_tokens'] for uid in batch['uids'])
            rounds=max(batches[j]['native_paired_forwards'] for j in (i//2*2,i//2*2+1))
            joint_bytes += 8*(extent+64)*D*(2*H+2*K)*2+4*(extent+64)*H*D*4*4
            native_bytes += rounds*8*extent*D*(2*H+2*K)*2
        cost.update(all_ranks_joint_operand_storage_upper_bytes=joint_bytes,
            all_ranks_native_operand_storage_upper_bytes=native_bytes,
            artifact_scope='Preserve actual original Q/K/V, attention outputs and original DT coefficients for '
                'every scheduled frozen query. No model weights, optimizer checkpoint or mixed-output tensor dump. '
                'Storage bounds exclude small serialization metadata; write/hash time is measured separately.')
    result['observations'] += ['Preserve the actual dense owner and its original recorded arguments; do not copy mask/unpadding/attention.',
        'All165 points compared with both fixed historical baseline and immediately preceding input probe; keep any drift separate.',
        'Use native F/D for this decomposition and retain the DT/native factual endpoint residual separately.',
        'Default passive gate path remains inert; new PV readouts are isolated behind the diagnostic-only flag.']
    path=HERE/'attention-pv-protocol.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=ref(path),symbolic_residual=str(residual),costs=result['costs'])))


if __name__=='__main__':
    main()
