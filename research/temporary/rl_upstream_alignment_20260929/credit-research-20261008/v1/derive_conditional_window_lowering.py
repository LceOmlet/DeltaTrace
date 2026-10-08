"""Derive a bounded conditional GDN-memory composition, not a new RL method.

No model/kernel import or execution. Arbitrary noncommuting symbols establish
the two compact window identities; the readout ledger names existing native
owner calls. This is not empirical attribution quality or a deployed patch.
"""
import hashlib
import json
from pathlib import Path
import time

import sympy as sp

HERE = Path(__file__).resolve().parent


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def derive(width):
    def nc(name):
        return sp.Symbol(name, commutative=False)

    checks = []

    def check(name, value):
        residue = sp.expand(value)
        if residue != 0:
            raise AssertionError((name, str(residue)))
        checks.append(dict(name=name, expanded_residual='0'))

    af = sp.symbols('af0:'+str(width))
    ac0 = sp.Symbol('ac0')
    ac = (ac0, *af[1:])
    beta = sp.symbols('beta0:'+str(width))
    kf = [nc('kf'+str(i)) for i in range(width)]
    kc = [nc('kc'+str(i)) for i in range(width)]
    kft = [nc('kfT'+str(i)) for i in range(width)]
    kct = [nc('kcT'+str(i)) for i in range(width)]
    uf = [nc('ufT'+str(i)) for i in range(width)]
    uc = [nc('ucT'+str(i)) for i in range(width)]
    h0 = nc('H0')

    # u is the native residual v_new, not the uncorrected v. Defining it from
    # beta*(v-alpha*Hprev^T*k) is the original memory state identity. This
    # low-rank difference proof needs no independence assumption about u.
    hf = hc = h0
    scalar = 0
    state_factors = []
    state_ranks = []
    for j in range(width):
        hf = af[j]*hf+kf[j]*uf[j]
        hc = ac[j]*hc+kc[j]*uc[j]
        scalar = ac0-af[0] if j == 0 else af[j]*scalar
        if j:
            state_factors = [(af[j]*u, v) for u, v in state_factors]
        state_factors += [(kc[j], uc[j]), (-kf[j], uf[j])]
        compact = scalar*h0+sum(u*v for u, v in state_factors)
        check('Conditional minus factual state, through position '+str(j), hc-hf-compact)
        state_ranks.append(len(state_factors))

    # P_j = future state cotangent + scale*q_j*do_j^T. alpha/beta differ
    # only at the first input position. P_0 excludes that first transition,
    # so its difference has no full-rank scalar term.
    qf = [nc('qf'+str(i)) for i in range(width)]
    qc = [nc('qc'+str(i)) for i in range(width)]
    dot = [nc('doT'+str(i)) for i in range(width)]
    scale = sp.Symbol('scale')
    future = nc('SharedFuture')
    pf = future+scale*qf[-1]*dot[-1]
    pc = future+scale*qc[-1]*dot[-1]
    reverse_factors = [(scale*(qc[-1]-qf[-1]), dot[-1])]
    reverse_ranks = [None]*width
    reverse_ranks[-1] = 1
    check('Conditional reverse state at last window position',
          pc-pf-sum(u*v for u, v in reverse_factors))
    for j in reversed(range(width-1)):
        n = j+1
        mf = af[n]*(1-beta[n]*kf[n]*kft[n])
        mc = af[n]*(1-beta[n]*kc[n]*kct[n])
        reverse_factors = [(mc*u, v) for u, v in reverse_factors]
        reverse_factors += [(af[n]*beta[n]*kf[n], kft[n]*pf),
                            (-af[n]*beta[n]*kc[n], kct[n]*pf),
                            (scale*(qc[j]-qf[j]), dot[j])]
        pf = mf*pf+scale*qf[j]*dot[j]
        pc = mc*pc+scale*qc[j]*dot[j]
        check('Conditional reverse difference at position '+str(j),
              pc-pf-sum(u*v for u, v in reverse_factors))
        reverse_ranks[j] = len(reverse_factors)

    # The reverse factors' left vectors stay in the listed basis: applying
    # Mc only multiplies an existing vector by alpha and adds a multiple of
    # kc. This is why the first-alpha trace needs seven additional past
    # queries, rather than one dense state or ten unrelated queries.
    basis = (['qc'+str(i)+'-qf'+str(i) for i in range(width)]
             +['kf'+str(i) for i in range(1, width)]
             +['kc'+str(i) for i in range(1, width)])
    return dict(sympy_version=sp.__version__, checks=checks,
        state_difference='Hc_j-Hf_j = dc_j*H0 + sum_m U_jm V_jm^T',
        state_rank_by_position=state_ranks,
        state_scalar='dc_0=ac_0-af_0; dc_j=af_j*dc_(j-1), j>0',
        reverse_difference='Pc_j-Pf_j = sum_m U_jm V_jm^T',
        reverse_rank_by_position=reverse_ranks,
        reverse_basis=basis,
        reverse_first_full_rank_term=False,
        small_decay_divisions=0,
        proof_scope='Arbitrary noncommuting symbols; not native dtype accuracy, a whole-model single-deletion oracle, or an accepted credit repair.')


def main():
    tick = time.perf_counter()
    algebra_path = HERE/'gdn-context-algebra-cost.json'
    previous = json.loads(algebra_path.read_bytes())
    width = previous['actual_owner_binding']['conv_width']
    chunk = previous['capacity_geometry']['chunk']
    assert width == 4 and chunk == 64
    # Every native readout here is the existing chunk_fwd_o with its required
    # contiguous operands, captured h/dh_end and strict boundary semantics.
    # A future matrix query uses two reversed native readouts, as verified
    # for original dv in results_native_context_readouts_20261009.json.
    ledger = [
        dict(orientation='factual future / conditional past',
             quantity='H0^T kc_(i+l)', lags=list(range(width)),
             calls=width, owner='past native chunk_fwd_o'),
        dict(orientation='factual future / conditional past',
             quantity='H0 do_(i+l)', lags=list(range(width)),
             calls=width, owner='transposed past native chunk_fwd_o'),
        dict(orientation='factual future / conditional past',
             quantity='H0 Lf_(i+l)', lags=list(range(width)),
             calls=width, owner='transposed past native chunk_fwd_o'),
        dict(orientation='factual future / conditional past',
             quantity='Pf_(i+l) uc_(i+l)', lags=list(range(width)),
             calls=2*width, owner='two transposed reversed native chunk_fwd_o per lag'),
        dict(orientation='conditional future / factual past',
             quantity='Pf_(i+l)^T kc_(i+l)', lags=list(range(width)),
             calls=2*width, owner='two reversed native chunk_fwd_o per lag'),
        dict(orientation='conditional future / factual past',
             quantity='Hf_(j-1) Wf_j to recover Pf_j uf_j from original dk',
             lags='all factual positions once', calls=1,
             owner='transposed past native chunk_fwd_o; original coincident mixed dk'),
        dict(orientation='conditional future / factual past',
             quantity='Hf_(i+l-1) Lc_(i+l)', lags=list(range(width)),
             calls=width, owner='transposed past native chunk_fwd_o'),
        dict(orientation='conditional future / factual past',
             quantity='H0^T kf_(i+l)', lags=list(range(1, width)),
             calls=width-1, owner='past native chunk_fwd_o'),
        dict(orientation='conditional future / factual past',
             quantity='H0^T (qc_(i+l)-qf_(i+l))', lags=list(range(width)),
             calls=width, owner='past native chunk_fwd_o'),
    ]
    assert sum(item['calls'] for item in ledger) == 40
    B, T, H, K = 4, 32768, 8, 128
    # A full-T eager factor/query bank is rejected by this arithmetic. A
    # source tile of 64 native chunks is an internal lowering proposal only,
    # never a rollout/minibatch/LoRA or experiment-setting change.
    source_tile = chunk*chunk
    halo_tile = source_tile+chunk
    vector_bytes = B*T*H*K*4
    tile_vector_bytes = B*halo_tile*H*K*4
    terms = {'all_40_readout_results':40,
             'conditional_state_factors_and_reverse_triples':2*width+3*(1+3*(width-1)),
             'conditional_q_k_v_all_four_positions':3*width,
             'finite_q_k_v_outputs_all_four_positions':3*width,
             'temporary_native_vector_inputs_and_outputs':8}
    memory = dict(
        scope='Conservative tensor arithmetic for the proposed head8 lowering, not measured total GPU/host peak. No lifetime reuse credited.',
        geometry=dict(batch=B, total_length=T, head_group=H, dimension=K,
                      source_tile=source_tile, native_chunk_halo=chunk),
        full_sequence_FP32_vector_bytes=vector_bytes,
        rejected_full_sequence_query_and_rank_bank_bytes=vector_bytes*(40+terms['conditional_state_factors_and_reverse_triples']),
        proposed_FP32_tile_vector_bytes=tile_vector_bytes,
        proposed_vector_terms=terms,
        proposed_vector_bytes=sum(terms.values())*tile_vector_bytes,
        proposed_h_and_dh_FP32_bytes=2*B*(halo_tile//chunk)*H*K*K*4,
        excluded='Existing model, all original endpoint captures, original coincident finite owner temporaries, norm/gate, linear projections, allocator reserve and host offload. Must measure actual combined residency before capacity/efficiency acceptance.')
    out = HERE/'conditional-window-lowering.json'
    result = dict(status='conditional_memory_core_prepared_not_full_credit_repair_or_accepted',
        source=ref(Path(__file__)), previous_algebra=ref(algebra_path),
        owner=previous['actual_owner_binding'],
        original_readout_evidence=ref(HERE.parents[4]/'experiments/rl/results_native_context_readouts_20261009.json'),
        prepared_core=ref(HERE/'conditional_window_memory.py'),
        native_query_composition=ref(HERE/'native_conditional_queries.py'),
        algebra=derive(width),
        conditional_state_evaluation='Evaluate uc sequentially through the four positions from H0^T kc_l and already formed rank updates; retain cross-position interactions. Do not treat the four changed positions as independent interventions.',
        ordered_finite_coefficients=previous['root_aligned_possibility']['conditional_finite_coefficients'],
        native_readout_ledger=ledger, readouts_per_source_tile_per_head_group=40,
        proposed_readouts_per_layer_at_32768=40*(T//source_tile)*4,
        coincident_original_mixed_owner_calls=1, coincident_original_adjoint_owner_calls=1,
        first_alpha=dict(
            forward='alpha_finite_f = alpha_coincident_f + beta_f*(kf-kc)^T*H0*Lf',
            reverse='alpha_finite_rev = alpha_coincident_f + beta_f*kf^T*H0*Lf - beta_c*kf^T*H0*Lc + trace((Pc-Pf)^T H0)',
            trace='Use the reverse low-rank basis and the already required H0^T kc queries plus H0^T kf_(i+1:i+4) and H0^T delta_q_(i:i+4). Never divide original r0 by alpha.'),
        memory_arithmetic=memory,
        boundary_rules=['All source positions including native 64-token boundaries remain in the same frozen evaluation.',
            'Past queries refer to H_(i-1), including captured chunk-start h for the first position.',
            'Future queries refer to Pf_(i+l); shifted queries crossing a chunk consume the next original chunk and dh_end, not zero future.',
            'The final incomplete chunk uses the original finite owner identity padding: zero q/k/v/beta/do/raw_g, carry last cumulative g.',
            'An end-of-sequence window has only the existing positions, with no fabricated action or observation.'],
        remaining_owner_composition=['Conditional q/k/v must come from the same selected layer-input row through the original linear projection and width4 convolution, with existing finite SiLU/L2 owners.',
            'A shared ordinary conv backward is insufficient when finite coefficients depend on both source and lag. Accumulate the four original linear conv weights with the corresponding conditional coefficients inside the existing DT owner; do not copy the native conv state machine.',
            'Original norm/gate finite propagation and upstream hidden-layer approximation remain separate. This memory identity alone cannot establish full original-token exactness.',
            'The research-only memory core and native query composition are prepared; the measured derivative-limit call count is40. The complete GDN owner composition, real live allocations, nonzero finite effects and unchanged frozen author cumulative deletion/RISE/MAS remain required before acceptance or deployment.'],
        evidence_limits='The collection supports joint-background allocation error, not GDN dominance. The 0.9ms original readout and 46.5ms original symmetric FLA primitive timings do not measure this complete 40-readout composition.',
        operations=dict(new_model=0, new_DT=0, new_native_operator=0,
                        new_backward=0, new_optimizer=0, production_modified=False,
                        training_restarted=False, credit_repair_accepted=False))
    result['cpu_seconds'] = time.perf_counter()-tick
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(out), identities=len(result['algebra']['checks']),
        readouts=result['readouts_per_source_tile_per_head_group'],
        rejected_global_bank_GiB=memory['rejected_full_sequence_query_and_rank_bank_bytes']/2**30,
        conservative_tile_extra_GiB=(memory['proposed_vector_bytes']+memory['proposed_h_and_dh_FP32_bytes'])/2**30,
        seconds=result['cpu_seconds'], credit_repair_accepted=False)))


if __name__ == '__main__':
    main()
