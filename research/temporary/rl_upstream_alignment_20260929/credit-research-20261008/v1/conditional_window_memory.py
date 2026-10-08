"""Research-only conditional finite coefficients for the existing GDN owner.

Compose NativeStateQueries and original coincident FLA coefficients. No native
state solver/backward is copied. Both finite endpoint orders retain the coupled
width4 convolution window. The incoming cotangent is supplied by original DT.
This is NOT a standalone model attribution or a deployed training estimator.
"""
import torch
import torch.nn.functional as F


def conditional_memory_coefficients(factual, base, L, r0, queries, conditional,
                                    alpha0, beta0, scale):
    """All source rows in one native-chunk-aligned tile, with original halo.

    conditional is q/k/v, each [4,B,S,H,K], at actual native operand dtype.
    alpha0/beta0 change only at the selected source row. Factual captures and
    base coefficients cover S plus the original right halo. No source is
    selected by credit magnitude. This function never constructs dense states.
    """
    width = 4
    B, length, H, K = factual['q'].shape
    assert conditional['q'].shape[0] == width
    _, _, sources, _, _ = conditional['q'].shape
    assert sources <= length
    assert all(value.shape == (width, B, sources, H, K)
               and value.dtype == factual['q'].dtype for value in conditional.values())

    def dot(a, b):
        return (a*b).sum(-1, keepdim=True)

    def ahead(value, lag, fill=0.):
        selected = value[:, lag:lag+sources]
        return F.pad(selected, (0, 0)*(selected.ndim-2)+(0, sources-selected.shape[1]), value=fill).float()

    def placed(value, lag=0):
        value = F.pad(value, (0, 0, 0, 0, lag, length-lag-sources)) if lag+sources <= length else \
                F.pad(value[:, :length-lag], (0, 0, 0, 0, lag, 0))
        return value.to(factual['q'].dtype).contiguous()

    def past(value, transpose=False):
        return queries.past(placed(value), transpose=transpose)[:, :sources].float()

    def past_at(value, lag):
        return ahead(queries.past(placed(value, lag), transpose=True), lag)

    def future_at(value, lag, transpose=False):
        return ahead(queries.future(placed(value, lag), transpose=transpose), lag)

    qf = [ahead(factual['q'], j) for j in range(width)]
    kf = [ahead(factual['k'], j) for j in range(width)]
    vf = [ahead(factual['v'], j) for j in range(width)]
    uf = [ahead(factual['v_new'], j) for j in range(width)]
    do = [ahead(queries.do[:, :length], j) for j in range(width)]
    lf = [ahead(L, j) for j in range(width)]
    af = [ahead(factual['raw_g'].float().exp(), j, 1.)[..., None] for j in range(width)]
    bf = [ahead(factual['beta'], j)[..., None] for j in range(width)]
    ac, bc = [alpha0.float()[..., None], *af[1:]], [beta0.float()[..., None], *bf[1:]]
    valid = [torch.arange(sources, device=factual['q'].device)+j < length for j in range(width)]
    qc, kc, vc = [[conditional[key][j].float()*valid[j][None, :, None, None]
                  for j in range(width)] for key in ('q', 'k', 'v')]
    # Original native past, queried at the same source position for every lag.
    h0kc = [past(kc[j]) for j in range(width)]
    h0do = [past(do[j], transpose=True) for j in range(width)]
    h0lf = [past(lf[j], transpose=True) for j in range(width)]
    uc, forward = [], []
    state_terms = []  # (key, residual, subsequent scalar decay), rank <= 4.
    prefix = torch.ones_like(ac[0])
    for j in range(width):
        previous_k = prefix*h0kc[j]
        previous_l = prefix*h0lf[j]
        for key, residual, decay in state_terms:
            previous_k = previous_k+decay*residual*dot(key, kc[j])
            previous_l = previous_l+decay*key*dot(residual, lf[j])
        uj = bc[j]*(vc[j]-ac[j]*previous_k)
        uc.append(uj)
        state_terms = [(key, residual, ac[j]*decay) for key, residual, decay in state_terms]
        state_terms.append((kc[j], uj, torch.ones_like(ac[j])))
        prefix = ac[j]*prefix
        current_do = prefix*h0do[j]
        for key, residual, decay in state_terms:
            current_do = current_do+decay*key*dot(residual, do[j])
        forward.append(dict(q=scale*current_do,
            k=future_at(uj, j, transpose=True)-af[j]*bf[j]*previous_l,
            v=bf[j]*lf[j]))
    beta_forward = dot(vc[0]-ac[0]*h0kc[0], lf[0]).squeeze(-1)
    alpha_forward = ahead(base['alpha'], 0)+bf[0].squeeze(-1)*dot(kf[0]-kc[0], h0lf[0]).squeeze(-1)

    # Conditional future differs only inside this fixed window. Its low-rank
    # factors include H0^T U, so the first-alpha trace needs no dense state.
    pkc = [future_at(kc[j], j) for j in range(width)]
    factual_W = factual['beta'].float()[..., None]*L
    previous_W = queries.past(factual_W.to(factual['q'].dtype), transpose=True).float()
    reverse, factors = [None]*width, []
    h0lc0 = lc0 = None
    for j in reversed(range(width)):
        if j < width-1:
            n = j+1
            updated = []
            for u, v, h0u in factors:
                ku = dot(kc[n], u)
                updated.append((af[n]*u-af[n]*bf[n]*kc[n]*ku, v,
                                af[n]*h0u-af[n]*bf[n]*h0kc[n]*ku))
            factors = updated
            h0kf = past(kf[n])
            factors += [(af[n]*bf[n]*kf[n], lf[n], af[n]*bf[n]*h0kf),
                        (-af[n]*bf[n]*kc[n], pkc[n], -af[n]*bf[n]*h0kc[n])]
        delta_q = qc[j]-qf[j]
        factors.append((scale*delta_q, do[j], scale*past(delta_q)))
        lc = pkc[j]
        pu = ahead(base['k'], j)+af[j]*ahead(previous_W, j)
        for u, v, _ in factors:
            lc = lc+v*dot(u, kc[j])
            pu = pu+u*dot(v, uf[j])
        previous_lc = past_at(lc, j)
        reverse[j] = dict(q=ahead(base['q'], j),
                          k=pu-ac[j]*bc[j]*previous_lc, v=bc[j]*lc)
        if j == 0:
            lc0, h0lc0 = lc, previous_lc
    beta_reverse = dot(vf[0]-ahead(r0, 0), lc0).squeeze(-1)
    trace = sum(dot(v, h0u).squeeze(-1) for _, v, h0u in factors)
    alpha_reverse = (ahead(base['alpha'], 0)
                     +bf[0].squeeze(-1)*dot(kf[0], h0lf[0]).squeeze(-1)
                     -bc[0].squeeze(-1)*dot(kf[0], h0lc0).squeeze(-1)+trace)
    result = {name:torch.stack([(forward[j][name]+reverse[j][name])*0.5
                               for j in range(width)]) for name in ('q', 'k', 'v')}
    result.update(beta=(beta_forward+beta_reverse)*0.5,
                  alpha=(alpha_forward+alpha_reverse)*0.5)
    return result
