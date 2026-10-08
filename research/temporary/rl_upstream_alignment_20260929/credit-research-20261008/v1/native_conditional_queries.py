"""Default-inert composition of the original FLA chunk readout interface.

This supplies state queries for a prospective DT finite operator. It does not
solve the state recurrence, run a model, compute credit, or replace native FLA.
The factual native capture and native adjoints remain the sole state owners.
"""
import torch
import torch.nn.functional as F
from fla.ops.common.chunk_o import chunk_fwd_o


class NativeStateQueries:
    def __init__(self, factual, adjoints, L, scale):
        self.length = factual['q'].shape[1]
        self.chunk = 64  # Exact pinned native h/WY chunk geometry.
        self.chunks = (self.length+self.chunk-1)//self.chunk
        self.scale = scale
        self.dtype = factual['q'].dtype
        self.readout_calls = 0
        self.f = {key:self.pad(factual[key]) for key in ('q', 'k', 'v_new')}
        self.g = self.pad(factual['g'], cumulative=True)
        self.h = factual['h'].contiguous()
        self.ht = self.h.transpose(-1, -2).contiguous()
        self.do = self.pad(adjoints['do'])
        self.W = self.pad((factual['beta'].float()[..., None]*L).to(factual['q'].dtype))
        self.D = (adjoints['dh_end']/scale).contiguous()
        self.Dt = self.D.transpose(-1, -2).contiguous()
        self.zero_D = torch.zeros_like(self.D)
        G = self.g.reshape(self.g.shape[0], self.chunks, self.chunk, self.g.shape[-1])
        self.reverse_g = (G[:, :, -1:]-G).flip(2).flatten(1, 2).contiguous()
        # Representation-only views/copies, not a second state propagation.
        self.rev = {key:self.reverse(value) for key, value in
                    dict(q=self.f['q'], k=self.f['k'], do=self.do, W=self.W).items()}

    def pad(self, value, cumulative=False):
        padding = self.chunks*self.chunk-self.length
        if padding:
            if cumulative:
                value = torch.cat((value, value[:, -1:].expand(-1, padding, -1)), dim=1)
            else:
                value = F.pad(value, (0, 0)*(value.ndim-2)+(0, padding))
        return value.contiguous()

    def reverse(self, value):
        shape = value.shape
        return value.reshape(shape[0], self.chunks, self.chunk, *shape[2:]).flip(2).reshape(shape).contiguous()

    def readout(self, **kwargs):
        self.readout_calls += 1
        return chunk_fwd_o(**kwargs)

    def current(self, query, transpose=False):
        q = self.pad(query.to(self.dtype))
        k, v, h = ((self.f['v_new'], self.f['k'], self.ht) if transpose
                   else (self.f['k'], self.f['v_new'], self.h))
        return self.readout(q=q, k=k, v=v, h=h, g=self.g, scale=1.)[:, :self.length]

    def past(self, query, transpose=False):
        """H_(j-1)^T query_j, or H_(j-1) query_j, without alpha division."""
        q = self.pad(query.to(self.dtype))
        B, _, H, K = q.shape
        block = q.reshape(B, self.chunks, self.chunk, H, K)
        shifted = torch.cat((block[:, :, 1:], torch.zeros_like(block[:, :, :1])), dim=2)
        k, v, h = ((self.f['v_new'], self.f['k'], self.ht) if transpose
                   else (self.f['k'], self.f['v_new'], self.h))
        inner = self.readout(q=shifted.flatten(1, 2).contiguous(), k=k, v=v,
                             h=h, g=self.g, scale=1.)
        inner = inner.reshape(B, self.chunks, self.chunk, H, K)
        # For the first position in each native chunk, the original captured
        # h is exactly the preceding state. Never shift across chunk identity.
        boundary = torch.einsum('bnhk,bnhkv->bnhv', block[:, :, 0].float(), h.float())
        result = torch.cat((boundary[:, :, None], inner[:, :, :-1].float()), dim=2)
        return result.flatten(1, 2)[:, :self.length].contiguous()

    def future(self, query, transpose=False):
        """Pf_j^T query_j, or Pf_j query_j, from original dh_end/WY."""
        q = self.pad(query.to(self.dtype))
        if transpose:
            key, value, h = self.rev['do'], self.rev['q'], self.Dt
            update_key, update_value = self.rev['W'], self.rev['k']
            diagonal_key, diagonal_value = self.W, self.f['k']
        else:
            key, value, h = self.rev['q'], self.rev['do'], self.D
            update_key, update_value = self.rev['k'], self.rev['W']
            diagonal_key, diagonal_value = self.f['k'], self.W
        rq = self.reverse(q)
        output = self.reverse(self.readout(q=rq, k=key, v=value, h=h,
                                          g=self.reverse_g, scale=self.scale))
        update = self.reverse(self.readout(q=rq, k=update_key, v=update_value,
                                          h=self.zero_D, g=self.reverse_g, scale=1.))
        diagonal = (q.float()*diagonal_key.float()).sum(-1, keepdim=True)*diagonal_value.float()
        return (output.float()-update.float()+diagonal)[:, :self.length].contiguous()
