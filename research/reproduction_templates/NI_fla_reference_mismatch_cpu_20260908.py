"""CPU64 reference-content ledger on captured native chunks; no native replay.

The unchanged mixed_chunk is attribution algebra, not a model or state forward.
All L/D/Z values are supplied actual native/diagnostic captures. No L recovery,
new A-transpose product, per-token KxV state, or full-sequence square tensor.
"""
import time
import numpy as np
from finite_fla_chunk_reference import mixed_chunk, exp_secant


def _np(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    assert x.device.type == 'cpu', 'GPU tensors prohibited in CPU audit'
    return x.detach().double().numpy()


def _bytes(x):
    if isinstance(x, dict):
        return sum(_bytes(v) for v in x.values())
    if isinstance(x, np.ndarray):
        return x.nbytes
    return x.numel()*x.element_size() if hasattr(x, 'numel') else 0


def _stats(x):
    return {'net': float(x.sum()), 'positive': float(np.maximum(x, 0).sum()),
            'negative': float(np.minimum(x, 0).sum()), 'absolute': float(np.abs(x).sum())}


def _dot(a, b):
    assert a.shape == b.shape
    return (a*b).sum(-1) if a.ndim == 4 else a*b


def audit(paired_e, points_e, saved_coeff6, replay_coeff6, adjointsCPU,
          diagCPU, scale, input_info, priorpoints_receipts):
    started = time.perf_counter()
    E, T, H, K = paired_e['q'].shape
    assert E == 2 and K == 128 and T == input_info['total_length']
    assert set(points_e) == {'0', '1', '10', '20'}
    B, C, N = 1, 64, (T+63)//64
    assert tuple(diagCPU['L'].shape) == (B*H*N, C, K)
    assert tuple(adjointsCPU['dh_end'].shape) == (B, N, H, K, K)
    # A reshape/view of saved L, not a recomputation or division by beta.
    native_L = diagCPU['L'].reshape(B, H, N, C, K)
    array_rows = {s:{} for s in ['1','10','20','B2']}
    calls = 0

    def put(step, name, start, stop, value):
        assert value.shape == (B,H,stop-start)
        if name not in array_rows[step]:
            array_rows[step][name] = np.zeros((B,T,H), dtype=np.float64)
        array_rows[step][name][:,start:stop] = value.transpose(0,2,1)

    for block, start in enumerate(range(0,T,C)):
        stop = min(start+C,T)
        length = stop-start
        def token(e, name, ep=None):
            x = e[name] if ep is None else e[name][ep::2]
            return _np(x[:,start:stop]).swapaxes(1,2)
        def state(e, ep=None):
            x = e['h'] if ep is None else e['h'][ep::2]
            return _np(x[:,block])
        Z = token(adjointsCPU, 'do')
        D = _np(adjointsCPU['dh_end'][:,block])
        L = _np(native_L[:,:,block,:length])
        q1,k1,beta1,G1,g1 = [token(paired_e,n,1) for n in ['q','k','beta','g','raw_g']]
        W = beta1[...,None]*L
        lower = np.tril(np.ones((length,length)), -1)

        def reference(e, ep=None):
            nonlocal calls
            k,v,u,g,G = [token(e,n,ep) for n in ['k','v','v_new','raw_g','g']]
            h = state(e,ep)
            m = mixed_chunk(q1,k,k1,v,u,beta1,g,G,G1,h,D,Z,L,scale)
            calls += 1
            ER = np.exp(np.minimum(G[..., :,None]-G[...,None,:],0))*lower
            # Exact two additive source groups of the existing dk formula.
            read = -np.exp(G)[...,None]*(W@h.swapaxes(-1,-2)) - ((W@u.swapaxes(-1,-2))*ER)@k
            m['key_read'] = read
            m['key_write'] = m['k']-read
            return m

        m01 = reference(paired_e,0)
        e01 = exp_secant(token(paired_e,'raw_g',0),g1)
        m01['g'] = m01['alpha']*e01
        saved = {k:token(saved_coeff6,k) for k in ['q','k','v','beta','g']}
        replay = {k:token(replay_coeff6,k) for k in saved}
        clean_delta = {k:token(points_e['0'],'raw_g' if k=='g' else k)-token(paired_e,'raw_g' if k=='g' else k,1) for k in saved}
        clean_transfer = sum(_dot(saved[k],clean_delta[k]) for k in saved)-_dot(Z,token(points_e['0'],'o')-token(paired_e,'o',1))

        for step in ['1','10','20','B2']:
            a, ep = (paired_e,0) if step=='B2' else (points_e[step],None)
            ma = m01 if step=='B2' else reference(a)
            # v is independent of the reference content for fixed endpoint1.
            assert np.array_equal(ma['v'],m01['v'])
            delta = {k:token(paired_e,'raw_g' if k=='g' else k,1)-token(a,'raw_g' if k=='g' else k,ep) for k in saved}
            da = np.exp(g1)-np.exp(token(a,'raw_g',ep))
            do = token(paired_e,'o',1)-token(a,'o',ep)
            terms = {}
            for k in saved:
                terms['saved_to_replayed_GPU_'+k] = _dot(saved[k]-replay[k],delta[k])
                terms['replayed_GPU_to_CPU01_'+k] = _dot(replay[k]-m01[k],delta[k])
            terms['query_reference_content'] = _dot(m01['q']-ma['q'],delta['q'])
            terms['key_write_reference_content'] = _dot(m01['key_write']-ma['key_write'],delta['k'])
            terms['key_read_reference_content'] = _dot(m01['key_read']-ma['key_read'],delta['k'])
            terms['beta_reference_correction_content'] = _dot(m01['beta']-ma['beta'],delta['beta'])
            terms['v_reference_difference'] = _dot(m01['v']-ma['v'],delta['v'])
            terms['decay_reference_content'] = (m01['alpha']-ma['alpha'])*da
            terms['exp_secant_conditional_curvature'] = m01['alpha']*(e01*delta['g']-da)
            actual = _dot(Z,do)
            pair_prediction = sum(_dot(ma[k],delta[k]) for k in ['q','k','v','beta'])+ma['alpha']*da
            terms['matched_pair_native_and_fixed_adjoint_closure'] = pair_prediction-actual
            terms['B1clean_minus_B2input_saved_coefficient_transfer'] = np.zeros_like(actual) if step=='B2' else clean_transfer
            expected = sum(_dot(saved[k],delta[k]) for k in saved)-actual+terms['B1clean_minus_B2input_saved_coefficient_transfer']
            reconstructed = sum(terms.values())
            assert np.max(np.abs(reconstructed-expected),initial=0)<1e-7
            for name,value in terms.items(): put(step,name,start,stop,value)
            put(step,'saved_error',start,stop,expected)
            put(step,'B2input_minus_A_actual_output',start,stop,actual)
            put(step,'MA1_primitive_prediction',start,stop,pair_prediction)
            put(step,'closure_residual',start,stop,reconstructed-expected)

    assert calls == N*4
    summary = {'scope':'CPU64 attribution algebra on captured native h/k/g/v_new, fixed actual L/D/Z and original endpoint1 routing. No model, new native adjoint, new state trajectory, candidate, or metric call.',
        'sign_convention':'prediction_minus_actual', 'primitive_alpha':'CPU exp(actual raw_g), a theoretical primitive; native cumulative-g/default-precision differences remain in matched-pair closure.',
        'residual_scope':'Matched-pair residual is measured, not assumed zero: it includes fixed captured native adjoint/default-precision chunk consistency. B1clean/B2input transfer is separate. Coordinate groups label hidden positions, not independent original token causes.',
        'counts':{'mixed_chunk_CPU64_calls':calls,'extra_key_read_CPU_matmuls':calls*3,
                  'A_transpose_du_recomputations':0,'native_calls':0,'model_calls':0,'GPU_calls':0},
        'input_logical_tensor_bytes':sum(_bytes(x) for x in [paired_e,points_e,saved_coeff6,replay_coeff6,adjointsCPU,diagCPU]),
        'points':{}}
    arrays = {}
    keep, P = set(input_info['keep']),input_info['prompt_length']
    for step, row in array_rows.items():
        deleted = keep if step=='B2' else set(priorpoints_receipts[step]['deleted_positions'])
        assert deleted <= keep
        groups = {'deleted':sorted(deleted),'kept':sorted(keep-deleted),
                  'other_prompt':sorted(set(range(P))-keep),'response':list(range(P,T))}
        assert sorted(sum(groups.values(),[])) == list(range(T))
        summaries = {}
        for name,value in row.items():
            assert np.isfinite(value).all()
            summaries[name] = {'total':_stats(value),
                'heads':[_stats(value[:,:,h]) for h in range(H)],
                'groups':{group:{'count':len(idx),**_stats(value[:,idx])} for group,idx in groups.items()}}
            arrays[step+'_'+name] = value
        summary['points'][step] = {'saved_error':_stats(row['saved_error']),
            'closure_max_absolute':float(np.max(np.abs(row['closure_residual']),initial=0)),
            'contractions':summaries}
    summary['output_array_bytes'] = sum(x.nbytes for x in arrays.values())
    summary['CPU_seconds'] = time.perf_counter()-started
    return summary,arrays
