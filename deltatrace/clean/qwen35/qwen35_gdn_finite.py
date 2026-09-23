"""Finite pullback for the actual Qwen3.5 GDN module, without replacing forward.

No cache/packed-sequence support yet. Consume native paired endpoint captures.
FLA uses the documented content1 recurrence; output norm*SiLU(z) defaults to
allocating interaction to normalized content (gate1, normalized content0).
Its optional symmetric rule splits only this output product's interaction.
RMS/L2 use the existing symmetric finite rule. Conv preactivations and the
linear transpose use the installed causal-conv primitive, with all costs paid.
"""
import inspect
import importlib
import sys
import torch
import torch.nn.functional as F
from signed_secant_rules import rmsnorm_secant_pullback
from qwen35_decoder_finite import _linear_transpose, _linear_weights
from native_attention_capture import copy_capture_tensor


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
    def __init__(self, module, device='cpu', *, copy_tensors=True, preserve_strides=False, capture_module_outputs=True, pinned_host=False, capture_input=True):
        self.module=module;self.device=device;self.values={};self.endpoints={}
        self.copy_tensors=copy_tensors
        self.preserve_strides=preserve_strides
        self.capture_module_outputs=capture_module_outputs
        self.pinned_host=pinned_host
        self.capture_input=capture_input;self.input_shape=None
        self.active=False;self.calls={};self.scale=None
        chunk=importlib.import_module('fla.ops.gated_delta_rule.chunk')
        self.codes={inspect.unwrap(f).__code__:label for label,f in [
            ('module',resolve_native_gdn_forward(type(module))),('conv',module.causal_conv1d_fn),
            ('FLA',module.chunk_gated_delta_rule),('stage',chunk.chunk_gated_delta_rule_fwd)]}

    def copy(self,x):
        return None if x is None else copy_capture_tensor(x,self.device,copy=self.copy_tensors,
                                                          preserve_strides=self.preserve_strides,pinned_host=self.pinned_host)

    def event(self,frame,kind,value):
        label=self.codes.get(frame.f_code);f=frame.f_locals
        if label=='module' and kind=='call' and f['self'] is self.module:
            assert not self.active and not self.values
            assert f.get('cache_params') is None and not f.get('kwargs',{}).get('cu_seq_lens_q')
            self.active=True
            self.input_shape=tuple(f['hidden_states'].shape)
            if self.capture_input:self.values['input']=self.copy(f['hidden_states'])
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
            if self.capture_module_outputs:
                # These two outputs belong only to the full diagnostic trace;
                # finite propagation consumes neither of them.
                self.values['norm_output']=self.copy(f['core_attn_out'])
                self.values['output']=self.copy(value)
            self.active=False

    def __enter__(self):
        assert sys.getprofile() is None;sys.setprofile(self.event);return self

    def __exit__(self,*args):
        sys.setprofile(None)
        if self.pinned_host and torch.device(self.device).type=='cpu':
            torch.cuda.current_stream().synchronize()


def _scalar_secant(x0,x1,y0,y1,derivative0):
    delta=x1-x0;nonzero=delta!=0
    return torch.where(nonzero,(y1-y0)/torch.where(nonzero,delta,torch.ones_like(delta)),derivative0)


def _silu_derivative(x):
    s=x.sigmoid();return s*(1+x*(1-s))


def _l2_pullback(x0,x1,upstream):
    dimension=x0.shape[-1]
    return rmsnorm_secant_pullback(x0.float(),x1.float(),dimension**-.5,upstream,1e-6/dimension)


def _norm_gate_finite_rule(o,z,m,weight,eps,norm_gate_rule):
    """Existing finite RMS/SiLU formula, factored for the owner compiler."""
    o0,o1=o[0::2].float(),o[1::2].float();z0,z1=z[0::2].float(),z[1::2].float()
    weight=weight.float()
    n0=o0*torch.rsqrt(o0.square().mean(-1,keepdim=True)+eps)*weight
    s0,s1=F.silu(z0),F.silu(z1)
    if norm_gate_rule=='content1':
        mo=rmsnorm_secant_pullback(o0,o1,weight,m*s1,eps)
        mz=m*n0*_scalar_secant(z0,z1,s0,s1,_silu_derivative(z0))
    else:
        n1=o1*torch.rsqrt(o1.square().mean(-1,keepdim=True)+eps)*weight
        s_mean=(s0+s1)*0.5;n_mean=(n0+n1)*0.5
        mo=rmsnorm_secant_pullback(o0,o1,weight,m*s_mean,eps)
        mz=m*n_mean*_scalar_secant(z0,z1,s0,s1,_silu_derivative(z0))
    return mo,mz


def _conv_silu_finite_rule(pre,output,upstream):
    """Existing convolution SiLU secant, exposed to the same owner compiler."""
    p0,p1=pre[0::2].float(),pre[1::2].float()
    y0,y1=output[0::2].float(),output[1::2].float()
    return upstream*_scalar_secant(p0,p1,y0,y1,_silu_derivative(p0))


def gdn_finite_pullback(module,values,endpoints,upstream,scale,fla_pullback,diagnostics=False,*,norm_gate_rule='content1',key_norm_pullback=None,offload_endpoints=False,fla_head_batch_size=None,norm_gate_pullback=None,conv_silu_pullback=None,input_shape=None):
    """Return input finite coefficients; actual input rows are 1,3,... .

    The supplied FLA callback is the traceable finite extension. The native
    convolution's autograd is used only for its linear preactivation operator;
    it also computes discarded weight gradients. That work is not free.

    With n=RMS(o) and s=SiLU(z), content1 uses delta(n*s)=s1*delta(n)
    +n0*delta(s); symmetric uses s_mean*delta(n)+n_mean*delta(s). Both
    preserve the theoretical endpoint identity with the same finite RMS and
    SiLU operators. This option does not change the native forward or FLA.
    """
    if norm_gate_rule not in ('content1','symmetric'):
        raise ValueError(f'Unsupported norm_gate_rule: {norm_gate_rule!r}')
    assert module.activation=='silu' and module.norm.activation=='silu'
    assert module.conv1d.bias is None and module.norm.bias is None
    assert type(module.norm).__module__=='fla.modules.fused_norm_gate'
    # Only the actual input's dimensions are consumed here. Full diagnostic
    # captures still retain its values; normal propagation carries metadata.
    batch,length,width=upstream.shape
    if input_shape is None:input_shape=values['input'].shape
    assert tuple(input_shape)==(2*batch,length,width)
    left=lambda x:x[0::2].float();right=lambda x:x[1::2].float()
    e=endpoints;c=values;terms={}
    def restore(capture,*names):
        if offload_endpoints:
            for name in names:
                capture[name]=copy_capture_tensor(capture[name],'cuda',preserve_strides=True)
    def head_group(value,start):
        part=value[:,:,start:start+fla_head_batch_size]
        if part.device.type=='cpu' and part.is_pinned():
            # Torch copies the strided pinned source directly into the same
            # contiguous GPU layout. Do not repack into pageable host memory.
            return torch.empty(part.shape,device='cuda',dtype=part.dtype).copy_(part,non_blocking=True)
        return copy_capture_tensor(part.contiguous(),'cuda',preserve_strides=True)
    restore(e,'o');restore(c,'z')
    mnorm=_linear_transpose(upstream,_linear_weights(module.out_proj))
    m=mnorm.reshape(batch,length,module.num_v_heads,module.head_v_dim)
    norm_gate=_norm_gate_finite_rule if norm_gate_pullback is None else norm_gate_pullback
    mo,mz=norm_gate(e['o'],c['z'],m,module.norm.weight,module.norm.eps,norm_gate_rule)
    # Native FLA's BF16 dot path is explicit here; retain the rounding boundary
    # in diagnostics rather than pretending FP32 upstreams remained unchanged.
    native_mo=mo.to(e['o'].dtype)
    if offload_endpoints:
        del e['o'],c['z'],mnorm,m,mo
    if fla_head_batch_size is not None:
        # Head-local state recurrences are independent. Keep every trajectory
        # and time step in each owner call. The callback still owns both
        # endpoint orientations. This is separate from capture transport so
        # copying and kernel-shape roundoff can be checked independently.
        parts=[]
        for start in range(0,module.num_v_heads,fla_head_batch_size):
            group={name:head_group(value,start)
                   for name,value in e.items() if name!='o'}
            parts.append(fla_pullback(group,native_mo[:,:,start:start+fla_head_batch_size].contiguous(),scale))
            del group
        coeff={name:torch.cat([part[name] for part in parts],dim=2) for name in parts[0]}
        del parts
    else:
        if offload_endpoints:restore(e,*tuple(e))
        coeff=fla_pullback(e,native_mo,scale)
    if offload_endpoints:del native_mo
    restore(c,'raw_q','raw_k')
    mq=_l2_pullback(c['raw_q'][0::2],c['raw_q'][1::2],coeff['q'])
    key_norm=_l2_pullback if key_norm_pullback is None else key_norm_pullback
    mk=key_norm(c['raw_k'][0::2],c['raw_k'][1::2],coeff['k'])
    if offload_endpoints:
        del c['raw_q'],c['raw_k'],coeff['q'],coeff['k']
    repeat=module.num_v_heads//module.num_k_heads
    mq=mq.reshape(batch,length,module.num_k_heads,repeat,module.head_k_dim).sum(3)
    mk=mk.reshape(batch,length,module.num_k_heads,repeat,module.head_k_dim).sum(3)
    restore(c,'a','b');restore(e,'beta','raw_g')
    b0,b1=left(c['b']),right(c['b']);a0,a1=left(c['a']),right(c['a'])
    sigmoid=b0.sigmoid()
    mb=coeff['beta']*_scalar_secant(b0,b1,left(e['beta']),right(e['beta']),sigmoid*(1-sigmoid))
    ga0=-module.A_log.float().exp()*(a0+module.dt_bias.float()).sigmoid()
    ma=coeff['g']*_scalar_secant(a0,a1,left(e['raw_g']),right(e['raw_g']),ga0)
    mconv=torch.cat([mq.flatten(2),mk.flatten(2),coeff['v'].flatten(2)],dim=-1).transpose(1,2)
    if offload_endpoints:
        del mq,mk,coeff,b0,b1,a0,a1,sigmoid,ga0
        e.clear()
    restore(c,'projected_qkv','conv_output')
    # One real public paired preactivation call, one real public autograd bwd.
    # Preserve fused model SiLU outputs as endpoints. Rounding can make equal
    # saved preactivations have different fused outputs; report those cases.
    with torch.enable_grad():
        projected=c['projected_qkv'].detach().requires_grad_(True)
        pre=module.causal_conv1d_fn(projected,module.conv1d.weight.squeeze(1),activation=None)
        conv_silu=_conv_silu_finite_rule if conv_silu_pullback is None else conv_silu_pullback
        mpre=conv_silu(pre.detach(),c['conv_output'],mconv)
        seed=torch.zeros_like(pre);seed[1::2]=mpre.to(pre.dtype)
        mprojected,=torch.autograd.grad(pre,projected,seed)
    if offload_endpoints:
        del c['projected_qkv'],c['conv_output'],pre,mpre,seed,projected,mconv
    mqkv=mprojected[1::2].transpose(1,2)
    mx=_linear_transpose(mqkv,_linear_weights(module.in_proj_qkv))
    if offload_endpoints:del mprojected,mqkv
    mx=mx+_linear_transpose(mz.flatten(2),_linear_weights(module.in_proj_z))
    mx=mx+_linear_transpose(mb,_linear_weights(module.in_proj_b))
    mx=mx+_linear_transpose(ma,_linear_weights(module.in_proj_a))
    if offload_endpoints and c['mask'] is not None:restore(c,'mask')
    mask=c['mask']
    if mask is not None and mask.shape[0]>1 and mask.shape[1]>1:mx=mx*mask[1::2,:,None]
    if diagnostics:
        p0,p1=left(pre.detach()),right(pre.detach())
        y0,y1=left(c['conv_output']),right(c['conv_output'])
        # Returned tensors are for separate analysis and are not retained by
        # the normal propagation route. No synchronization in the route itself.
        terms={'mnorm':mnorm,'mo_before_cast':mo,'mo_native':native_mo,'mz':mz,'coeff':coeff,
            'mq':mq,'mk':mk,'mb':mb,'ma':ma,'mconv':mconv,'pre':pre.detach(),'mpre':mpre,
            'mprojected':mprojected.detach(),'pre_collision':(p0==p1)&(y0!=y1)}
    return mx,terms
