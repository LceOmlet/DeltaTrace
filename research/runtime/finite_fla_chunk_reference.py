"""CPU64 mixed contractions on captured native FLA states and adjoints.

This is an attribution-algebra reference, NOT a model/attention forward or a
production GPU backend. It reuses supplied native WY/state tensors, handles at
most 64x64 interaction tiles, and never stores per-token KxV states. Coefficients
are for normalized q/k, v, beta and alpha=exp(g); surrounding finite pullbacks
are not supplied by this module. Source/numeric validation belongs to its caller.
"""
import numpy as np


def transpose(x):
    return np.swapaxes(x, -1, -2)


def rowdot(x, y):
    return np.sum(x * y, axis=-1)


def exp_secant(g0, g1):
    """Stable exponential divided difference, including coincident endpoints."""
    distance = np.abs(g1 - g0)
    ratio = np.ones_like(distance)
    np.divide(-np.expm1(-distance), distance, out=ratio, where=distance != 0)
    return np.exp(np.maximum(g0, g1)) * ratio


def mixed_chunk(q1, k0, k1, v0, u0, beta1, g0, G0, G1,
                H0, dh_end, do, lambda_u, scale):
    """One chunk, [B,H,C,D] operands; H0/dh_end [B,H,K,V].

    q1 is normalized but unscaled; u0 is native v_new, not WY's transformed
    beta*v before subtracting the chunk-start state. lambda_u is A1^T*dU_WY.
    """
    C = q1.shape[-2]
    assert 1 <= C <= 64 and k0.shape == k1.shape == q1.shape
    assert np.all(np.diff(G0, axis=-1) <= 1e-5) and np.all(np.diff(G1, axis=-1) <= 1e-5)
    lower = np.tril(np.ones((C, C)), -1)
    causal = lower + np.eye(C)
    upper = np.triu(np.ones((C, C)))
    E0 = np.exp(np.minimum(G0[..., :, None] - G0[..., None, :], 0)) * causal
    E1 = np.exp(np.minimum(G1[..., None, :] - G1[..., :, None], 0)) * upper
    Gprev = np.concatenate([np.zeros_like(G0[..., :1]), G0[..., :-1]], axis=-1)
    Eprev = np.exp(np.minimum(Gprev[..., :, None] - G0[..., None, :], 0)) * lower
    eend = np.exp(G1[..., -1:] - G1)
    q = q1 * scale
    w = beta1[..., None] * lambda_u
    lr = -w

    dq = scale * (np.exp(G0)[..., None] * (do @ transpose(H0))
                  + ((do @ transpose(u0)) * E0) @ k0)
    dk = (eend[..., None] * (u0 @ transpose(dh_end))
          + ((u0 @ transpose(do)) * E1) @ q
          - ((u0 @ transpose(w)) * E1) @ k1
          + beta1[..., None] * k1 * rowdot(u0, lambda_u)[..., None]
          + np.exp(G0)[..., None] * (lr @ transpose(H0))
          + ((lr @ transpose(u0)) * (E0 * lower)) @ k0)
    r0 = np.exp(G0)[..., None] * (k0 @ H0) + ((k0 @ transpose(k0)) * (E0 * lower)) @ u0
    dbeta = rowdot(v0 - r0, lambda_u)

    S = np.sum(H0 * dh_end, axis=(-2, -1))
    b = rowdot(do, q @ H0) - rowdot(w, k1 @ H0)
    d = rowdot(u0, k0 @ dh_end)
    M = (k0 @ transpose(q)) * (u0 @ transpose(do)) - (k0 @ transpose(k1)) * (u0 @ transpose(w))
    # Stable weighted prefix scan. This serial CPU reference can be implemented
    # by the compiler's associative affine scan; it does not invert tiny decay.
    past = np.zeros_like(M)
    accumulated = np.zeros_like(M[..., 0, :])
    for j in range(C):
        past[..., j, :] = accumulated
        accumulated = M[..., j, :] + np.exp(g0[..., j, None]) * accumulated
    dalpha = (eend * (np.exp(Gprev) * S[..., None] + (Eprev @ d[..., None])[..., 0])
              + np.exp(Gprev) * (E1 @ b[..., None])[..., 0]
              + np.sum(past * E1, axis=-1))
    return dict(q=dq, k=dk, v=w, beta=dbeta, alpha=dalpha)


def coefficients_from_native_capture(endpoints, adjoints, scale):
    """Interleaved [EOS0,input0,EOS1,input1,...] endpoints, input-only adjoints."""
    B, T, H, K = endpoints['q'].shape
    assert B % 2 == 0 and adjoints['do'].shape[:3] == (B//2,T,H)
    result = {k: np.empty((B//2,T,H,K) if k in ['q','k','v'] else (B//2,T,H), dtype=np.float64)
              for k in ['q','k','v','beta','alpha','g']}
    lambdas = np.empty_like(result['v'])
    for block, start in enumerate(range(0,T,64)):
        stop = min(start+64,T);C=stop-start
        def token(name, endpoint):
            a=np.asarray(endpoints[name][endpoint::2,start:stop],dtype=np.float64)
            return np.swapaxes(a,1,2)
        def adj(name):
            return np.swapaxes(np.asarray(adjoints[name][:,start:stop],dtype=np.float64),1,2)
        A1=token('A',1)[...,:C]
        lu=transpose(A1) @ adj('dU_WY')
        values=mixed_chunk(q1=token('q',1),k0=token('k',0),k1=token('k',1),v0=token('v',0),
            u0=token('v_new',0),beta1=token('beta',1),g0=token('raw_g',0),G0=token('g',0),G1=token('g',1),
            H0=np.asarray(endpoints['h'][0::2,block],dtype=np.float64),
            dh_end=np.asarray(adjoints['dh_end'][:,block],dtype=np.float64),do=adj('do'),lambda_u=lu,scale=scale)
        values['g']=values['alpha'] * exp_secant(token('raw_g',0),token('raw_g',1))
        for key,value in values.items():result[key][:,start:stop]=np.swapaxes(value,1,2)
        lambdas[:,start:stop]=np.swapaxes(lu,1,2)
    return result,lambdas
