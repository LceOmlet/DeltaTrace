"""Finite attribution for original Qwen3.5 attention boundaries and decoders.

No model forward/backward is replaced. Consume actual paired endpoint captures;
reuse the vendor FA finite core, existing GDN finite callback, native BF16 GEMMs,
the established symmetric RMS/SwiGLU rules and the official Torch compiler.
"""
import torch
from signed_secant_rules import rmsnorm_secant_pullback
from compiled_swiglu_secant import swiglu_finite_rule
from finite_fla_gpu import _mm


class NativeDecoderCapture:
    """Module hooks only, so the mixer can use its existing passive observer."""
    def __init__(self, layer, destination='cpu'):
        self.layer=layer;self.destination=destination;self.values={};self.handles=[];self.calls={}

    def retain(self,name,value):
        self.values[name]=value.detach().to(self.destination,copy=True)

    def __enter__(self):
        targets={'input_norm':self.layer.input_layernorm,'post_norm':self.layer.post_attention_layernorm,
            'gate':self.layer.mlp.gate_proj,'up':self.layer.mlp.up_proj,'silu':self.layer.mlp.act_fn,
            'down':self.layer.mlp.down_proj,'mlp':self.layer.mlp}
        assert all(isinstance(module,torch.nn.Module) for module in targets.values())
        for name,module in targets.items():
            def observe(_module,args,output,name=name):
                self.calls[name]=self.calls.get(name,0)+1
                # Shared inputs already have a retained norm/gate endpoint;
                # avoid copying them once again at every projection.
                if name in ('input_norm','post_norm','down'):self.retain(name+'_input',args[0])
                if name!='down':self.retain(name+'_output',output)
            self.handles.append(module.register_forward_hook(observe))
        def output(_module,_args,value):
            self.calls['decoder']=self.calls.get('decoder',0)+1;self.retain('output',value)
        self.handles.append(self.layer.register_forward_hook(output))
        return self

    def __exit__(self,*_exc):
        for handle in self.handles:handle.remove()
        self.handles.clear()


def _linear_transpose(upstream,weight):
    shape=upstream.shape
    return _mm(upstream.reshape(1,-1,shape[-1]),weight.unsqueeze(0)).reshape(*shape[:-1],weight.shape[-1])


def _secant(x0,x1,y0,y1,derivative):
    delta=x1-x0;nonzero=delta!=0
    return torch.where(nonzero,(y1-y0)/torch.where(nonzero,delta,torch.ones_like(delta)),derivative)


def _partial_rotation_transpose(multiplier,cos,sin):
    # Native Qwen applies RoPE only to the first rotary_dim coordinates.
    # R^T m = m*cos - rotate_half(m*sin); remaining coordinates pass through.
    d=cos.shape[-1];rot=multiplier[...,:d];rest=multiplier[...,d:]
    c=cos.float().unsqueeze(1);s=sin.float().unsqueeze(1)
    first,second=(rot*s).chunk(2,dim=-1)
    return torch.cat((rot*c-torch.cat((-second,first),dim=-1),rest),dim=-1)


def _mlp_input_rule(g0,g1,u0,u1,s0,s1,upstream,down_weight,up_weight,gate_weight):
    product=_linear_transpose(upstream,down_weight)
    mu,mg=swiglu_finite_rule(g0,g1,u0,u1,s0.float(),s1.float(),product)
    return _linear_transpose(mu,up_weight)+_linear_transpose(mg,gate_weight)


def _norm_residual_rule(x0,x1,raw_weight,upstream,residual,eps):
    return residual+rmsnorm_secant_pullback(x0.float(),x1.float(),1+raw_weight.float(),upstream,eps)


def _attention_gate_rule(qproj0,qproj1,attention0,upstream,out_weight,heads,dim):
    b,t,_=upstream.shape
    g0=qproj0.view(b,t,heads,2*dim)[...,dim:]
    g1=qproj1.view(b,t,heads,2*dim)[...,dim:]
    # Same installed elementwise sigmoid as the native module, at its native
    # BF16 operand/output dtype. This is a finite-rule scalar evaluation.
    s0=g0.sigmoid().float();s1=g1.sigmoid().float()
    g0f=g0.float();g1f=g1.float();derivative=g0f.sigmoid()
    multiplier=_linear_transpose(upstream,out_weight).view(b,t,heads,dim)
    # content1: Δ(n*s)=s1*Δn+n0*Δs; gate remains a separate signed branch.
    mcontent=multiplier*s1
    mgate=multiplier*attention0.float()*_secant(g0f,g1f,s0,s1,derivative*(1-derivative))
    return mcontent.transpose(1,2).contiguous(),mgate


def _attention_input_rule(dq,dk,dv,mgate,q0,q1,k0,k1,qweight,kweight,cos,sin,
                          qproj_weight,kproj_weight,vproj_weight,eps,groups):
    b,h,t,d=dq.shape;kh=dk.shape[1]//groups
    mq=dq.float();mk=dk.float().reshape(b,kh,groups,t,d).sum(2)
    mv=dv.float().reshape(b,kh,groups,t,d).sum(2).transpose(1,2).reshape(b,t,kh*d)
    mq=_partial_rotation_transpose(mq,cos,sin).transpose(1,2)
    mk=_partial_rotation_transpose(mk,cos,sin).transpose(1,2)
    mq=rmsnorm_secant_pullback(q0.float(),q1.float(),1+qweight.float(),mq,eps)
    mk=rmsnorm_secant_pullback(k0.float(),k1.float(),1+kweight.float(),mk,eps)
    # q_proj interleaves query and gate WITHIN each head, not globally.
    q_and_gate=torch.cat((mq,mgate),dim=-1).reshape(b,t,h*2*d)
    return (_linear_transpose(q_and_gate,qproj_weight)
            +_linear_transpose(mk.reshape(b,t,kh*d),kproj_weight)
            +_linear_transpose(mv,vproj_weight))


class FiniteBoundaryOps:
    """Four composite graphs; native/vendor operations stay externally visible.

    Compiles only finite attribution. Cold compiler/default tuning costs must
    be counted; disabling max_autotune does not disable all compiler tuning.
    No hidden eager fallback is installed when a graph fails to compile.
    """
    def __init__(self,compiled=True):
        self.compiled=compiled
        for name,fn in [('mlp',_mlp_input_rule),('norm_residual',_norm_residual_rule),
                        ('attention_gate',_attention_gate_rule),('attention_input',_attention_input_rule)]:
            op=torch.compile(fn,fullgraph=True,dynamic=False,
                options={'triton.cudagraphs':False,'max_autotune':False}) if compiled else fn
            setattr(self,name,op)


def attention_finite_pullback(module,values,lse,cos,sin,upstream,finite_fa,layout,boundaries,diagnostics=False):
    """Input coefficients for the actual standard-attention module.

    lse is the publicly returned paired FA LSE, [2B,H,T]. Endpoints share
    positions and mask; the caller validates that contract once when captured.
    """
    c=values;b,t,width=upstream.shape;heads=module.config.num_attention_heads;dim=module.head_dim
    assert c['input'].shape==(2*b,t,width) and lse.shape==(2*b,heads,t)
    assert cos.shape==sin.shape and cos.shape[0]==2*b and cos.shape[1]==t
    assert module.q_norm.eps==module.k_norm.eps and module.attention_dropout==0
    mcontent,mgate=boundaries.attention_gate(c['q_proj_output'][0::2],c['q_proj_output'][1::2],
        c['attention_output'][0::2],upstream,module.o_proj.weight,heads,dim)
    ops={'q0':c['query'][0::2],'q1':c['query'][1::2],'k0':c['key'][0::2],'k1':c['key'][1::2],
         'v0':c['value'][0::2],'u':mcontent,'lse0':lse[0::2],'lse1':lse[1::2]}
    activity={} if diagnostics else None
    coeff=finite_fa(ops,module.scaling,layout,activity)
    mx=boundaries.attention_input(coeff['dq'],coeff['dk'],coeff['dv'],mgate,
        c['q_norm_input'][0::2],c['q_norm_input'][1::2],c['k_norm_input'][0::2],c['k_norm_input'][1::2],
        module.q_norm.weight,module.k_norm.weight,cos[1::2],sin[1::2],
        module.q_proj.weight,module.k_proj.weight,module.v_proj.weight,module.q_norm.eps,module.num_key_value_groups)
    diagnostics_out={'mcontent':mcontent,'mgate':mgate,'coeff':coeff,'finite_FA_activity':activity} if diagnostics else {}
    return mx,diagnostics_out


def decoder_finite_pullback(layer,values,upstream,mixer_pullback,boundaries,diagnostics=False):
    """Both original decoder families: symmetric MLP + two residual/norm paths.

    mixer_pullback receives the actual mixer-output cotangent and returns its
    input coefficients, using the traceable FA or GDN finite implementation.
    """
    c=values
    mnorm=boundaries.mlp(c['gate_output'][0::2],c['gate_output'][1::2],
        c['up_output'][0::2],c['up_output'][1::2],c['silu_output'][0::2],c['silu_output'][1::2],
        upstream,layer.mlp.down_proj.weight,layer.mlp.up_proj.weight,layer.mlp.gate_proj.weight)
    mmixer=boundaries.norm_residual(c['post_norm_input'][0::2],c['post_norm_input'][1::2],
        layer.post_attention_layernorm.weight,mnorm,upstream,layer.post_attention_layernorm.eps)
    mmixer_input,mixer_diagnostics=mixer_pullback(mmixer)
    mx=boundaries.norm_residual(c['input_norm_input'][0::2],c['input_norm_input'][1::2],
        layer.input_layernorm.weight,mmixer_input,mmixer,layer.input_layernorm.eps)
    return mx,({'m_mlp_norm_output':mnorm,'m_mixer_output':mmixer,'m_mixer_input':mmixer_input,
                'mixer':mixer_diagnostics} if diagnostics else {})
