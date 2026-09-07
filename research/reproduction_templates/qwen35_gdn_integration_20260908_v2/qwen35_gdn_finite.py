"""Finite pullback for the actual Qwen3.5 GDN module, without replacing forward.

No cache/packed-sequence support yet. Consume native paired endpoint captures.
FLA uses the documented content1 recurrence; output norm*SiLU(z) allocates the
interaction to normalized content (input gate1, baseline normalized content0).
RMS/L2 use the existing symmetric finite rule. Conv preactivations and the
linear transpose use the installed causal-conv primitive, with all costs paid.
"""
import inspect
import importlib
import sys
import torch
import torch.nn.functional as F
from signed_secant_rules import rmsnorm_secant_pullback
from finite_fla_gpu import _mm


def resolve_native_gdn_forward(module_type):
    """Find the existing body hidden by the pinned accelerate decorator.

    force_accelerate_hooks closes over forward_func without functools.wraps.
    This only selects a code object for passive observation; it never calls or
    installs the unwrapped body as a replacement model forward.
    """
    function=inspect.unwrap(module_type.forward)
    if 'hidden_states' not in inspect.signature(function).parameters:
        function=inspect.getclosurevars(function).nonlocals.get('forward_func')
    assert inspect.isfunction(function)
    assert function.__qualname__==module_type.__qualname__+'.forward'
    assert function.__code__.co_filename==inspect.getfile(module_type)
    assert 'hidden_states' in inspect.signature(function).parameters
    return function


class NativeGDNCapture:
    """Passive, single-module capture; never wraps a native operator or forward."""
    def __init__(self, module, device='cpu'):
        self.module=module;self.device=device;self.values={};self.endpoints={}
        self.active=False;self.calls={};self.scale=None
        chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')
        self.codes={inspect.unwrap(f).__code__:label for label,f in [
            ('module',resolve_native_gdn_forward(type(module))),('conv',module.causal_conv1d_fn),
            ('FLA',module.chunk_gated_delta_rule),('stage',chunk.chunk_gated_delta_rule_fwd)]}

    def copy(self,x):
        return None if x is None else x.detach().to(self.device,copy=True)

    def event(self,frame,kind,value):
        label=self.codes.get(frame.f_code);f=frame.f_locals
        if label=='module' and kind=='call' and f['self'] is self.module:
            assert not self.active and not self.values
            assert f.get('cache_params') is None and not f.get('kwargs',{}).get('cu_seq_lens_q')
            self.active=True
            self.values['input']=self.copy(f['hidden_states'])
            self.values['mask']=self.copy(f.get('attention_mask'))
        if not self.active:return
        if kind=='call' and label:self.calls[label]=self.calls.get(label,0)+1
        if kind=='call' and label=='conv':self.values['projected_qkv']=self.copy(f['x'])
        if kind=='return' and label=='conv' and value is not None:self.values['conv_output']=self.copy(value)
        if kind=='call' and label=='FLA':
            for name in ['q','k']:self.values['raw_'+name]=self.copy(f[name])
            self.endpoints['raw_g']=self.copy(f['g'])
        if kind=='return' and label=='stage' and value is not None:
            assert f['initial_state'] is None and f['cu_seqlens'] is None
            self.scale=float(f['scale'])
            for name in ['q','k','v','g','beta','A','w','v_new','o','h']:self.endpoints[name]=self.copy(f[name])
        if kind=='return' and label=='module' and f['self'] is self.module:
            assert value is not None
            b,t,_=f['hidden_states'].shape
            for name in ['a','b']:self.values[name]=self.copy(f[name])
            self.values['z']=self.copy(f['z'].reshape(b,t,self.module.num_v_heads,self.module.head_v_dim))
            self.values['norm_output']=self.copy(f['core_attn_out'])
            self.values['output']=self.copy(value);self.active=False

    def __enter__(self):
        assert sys.getprofile() is None;sys.setprofile(self.event);return self

    def __exit__(self,*args):
        sys.setprofile(None)


def _linear_transpose(upstream,weight):
    shape=upstream.shape
    return _mm(upstream.reshape(1,-1,shape[-1]),weight.unsqueeze(0)).reshape(*shape[:-1],weight.shape[-1])


def _scalar_secant(x0,x1,y0,y1,derivative0):
    delta=x1-x0;nonzero=delta!=0
    return torch.where(nonzero,(y1-y0)/torch.where(nonzero,delta,torch.ones_like(delta)),derivative0)


def _silu_derivative(x):
    s=x.sigmoid();return s*(1+x*(1-s))


def _l2_pullback(x0,x1,upstream):
    dimension=x0.shape[-1]
    return rmsnorm_secant_pullback(x0.float(),x1.float(),dimension**-.5,upstream,1e-6/dimension)


def gdn_finite_pullback(module,values,endpoints,upstream,scale,fla_pullback,diagnostics=False):
    """Return input finite coefficients; actual input rows are 1,3,... .

    The supplied FLA callback is the traceable finite extension. The native
    convolution's autograd is used only for its linear preactivation operator;
    it also computes discarded weight gradients. That work is not free.
    """
    assert module.activation=='silu' and module.norm.activation=='silu'
    assert module.conv1d.bias is None and module.norm.bias is None
    assert type(module.norm).__module__=='fla.modules.fused_norm_gate'
    batch,length,width=upstream.shape;assert values['input'].shape==(2*batch,length,width)
    left=lambda x:x[0::2].float();right=lambda x:x[1::2].float()
    e=endpoints;c=values;terms={}
    mnorm=_linear_transpose(upstream,module.out_proj.weight)
    m=mnorm.reshape(batch,length,module.num_v_heads,module.head_v_dim)
    o0,o1=left(e['o']),right(e['o']);z0,z1=left(c['z']),right(c['z'])
    weight=module.norm.weight.float();eps=module.norm.eps
    n0=o0*torch.rsqrt(o0.square().mean(-1,keepdim=True)+eps)*weight
    s0,s1=F.silu(z0),F.silu(z1)
    mo=rmsnorm_secant_pullback(o0,o1,weight,m*s1,eps)
    mz=m*n0*_scalar_secant(z0,z1,s0,s1,_silu_derivative(z0))
    # Native FLA's BF16 dot path is explicit here; retain the rounding boundary
    # in diagnostics rather than pretending FP32 upstreams remained unchanged.
    native_mo=mo.to(e['o'].dtype)
    coeff=fla_pullback(e,native_mo,scale)
    mq=_l2_pullback(c['raw_q'][0::2],c['raw_q'][1::2],coeff['q'])
    mk=_l2_pullback(c['raw_k'][0::2],c['raw_k'][1::2],coeff['k'])
    repeat=module.num_v_heads//module.num_k_heads
    mq=mq.reshape(batch,length,module.num_k_heads,repeat,module.head_k_dim).sum(3)
    mk=mk.reshape(batch,length,module.num_k_heads,repeat,module.head_k_dim).sum(3)
    b0,b1=left(c['b']),right(c['b']);a0,a1=left(c['a']),right(c['a'])
    sigmoid=b0.sigmoid()
    mb=coeff['beta']*_scalar_secant(b0,b1,left(e['beta']),right(e['beta']),sigmoid*(1-sigmoid))
    ga0=-module.A_log.float().exp()*(a0+module.dt_bias.float()).sigmoid()
    ma=coeff['g']*_scalar_secant(a0,a1,left(e['raw_g']),right(e['raw_g']),ga0)
    mconv=torch.cat([mq.flatten(2),mk.flatten(2),coeff['v'].flatten(2)],dim=-1).transpose(1,2)
    # One real public paired preactivation call, one real public autograd bwd.
    # Preserve fused model SiLU outputs as endpoints. Rounding can make equal
    # saved preactivations have different fused outputs; report those cases.
    with torch.enable_grad():
        projected=c['projected_qkv'].detach().requires_grad_(True)
        pre=module.causal_conv1d_fn(projected,module.conv1d.weight.squeeze(1),activation=None)
        p0,p1=left(pre.detach()),right(pre.detach())
        y0,y1=left(c['conv_output']),right(c['conv_output'])
        mpre=mconv*_scalar_secant(p0,p1,y0,y1,_silu_derivative(p0))
        seed=torch.zeros_like(pre);seed[1::2]=mpre.to(pre.dtype)
        mprojected,=torch.autograd.grad(pre,projected,seed)
    mqkv=mprojected[1::2].transpose(1,2)
    mx=_linear_transpose(mqkv,module.in_proj_qkv.weight)
    mx=mx+_linear_transpose(mz.flatten(2),module.in_proj_z.weight)
    mx=mx+_linear_transpose(mb,module.in_proj_b.weight)
    mx=mx+_linear_transpose(ma,module.in_proj_a.weight)
    mask=c['mask']
    if mask is not None and mask.shape[0]>1 and mask.shape[1]>1:mx=mx*mask[1::2,:,None]
    if diagnostics:
        # Returned tensors are for separate analysis and are not retained by
        # the normal propagation route. No synchronization in the route itself.
        terms={'mnorm':mnorm,'mo_before_cast':mo,'mo_native':native_mo,'mz':mz,'coeff':coeff,
            'mq':mq,'mk':mk,'mb':mb,'ma':ma,'mconv':mconv,'pre':pre.detach(),'mpre':mpre,
            'mprojected':mprojected.detach(),'pre_collision':(p0==p1)&(y0!=y1)}
    return mx,terms
