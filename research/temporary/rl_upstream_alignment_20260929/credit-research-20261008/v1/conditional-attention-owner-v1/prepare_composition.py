"""Prepare default-inert runner/decoder seams in isolated owner copies.

The generator preserves each task's actual owner, including its memory patch.
It never rewrites PLAN, production files, model operations or official kernels.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
COLLECTION=HERE.parent


def replace_once(text,old,new):
    assert text.count(old)==1,(old[:150],text.count(old))
    return text.replace(old,new,1)


# These new scalar contractions belong to the finite boundary owner. Their
# fused gather/reductions avoid expanded GQA inputs and full attention maps.
DIAGONAL_RULES='''
def _attention_own_kv(value,starts,time):
    b,heads,length,dim=value.shape
    positions=starts[:,None]+torch.arange(time,device=value.device)[None,:]
    # Only padding indices exceed length. They never enter the public query
    # mask or a valid finite coefficient. This is indexing, not credit clipping.
    positions=positions.clamp_max(length-1)
    return value.gather(2,positions[:,None,:,None].expand(b,heads,time,dim))


def _attention_own_scores_rule(q0,q1,k0,starts,scale):
    b,heads,time,dim=q0.shape;kh=k0.shape[1];groups=heads//kh
    own=_attention_own_kv(k0,starts,time).float().unsqueeze(2)
    score0=(q0.float().reshape(b,kh,groups,time,dim)*own).sum(-1).reshape(b,heads,time)*scale
    score1=(q1.float().reshape(b,kh,groups,time,dim)*own).sum(-1).reshape(b,heads,time)*scale
    return score0,score1


def _attention_own_uv_rule(upstream,v0,starts):
    b,heads,time,dim=upstream.shape;kh=v0.shape[1];groups=heads//kh
    own=_attention_own_kv(v0,starts,time).float().unsqueeze(2)
    # The existing finite FA ABI stores U in BF16 before contraction.
    u=upstream.to(torch.bfloat16).float().reshape(b,kh,groups,time,dim)
    return (u*own).sum(-1).reshape(b,heads,time)


def _attention_own_endpoint_rule(past,lse,own_score,v0,starts):
    b,time,heads,dim=past.shape;kh=v0.shape[1];groups=heads//kh
    logz=torch.logaddexp(lse,own_score)
    old_weight=torch.exp(lse-logz).transpose(1,2).unsqueeze(-1)
    own_weight=torch.exp(own_score-logz).reshape(b,kh,groups,time,1)
    own=_attention_own_kv(v0,starts,time).float().unsqueeze(2)
    own=(own*own_weight).reshape(b,heads,time,dim).transpose(1,2)
    # Keep the combined finite-rule value in FP32; do not add a second BF16
    # rounding after the native strict-past output has already been stored.
    return past.float()*old_weight+own,logz


'''


def decoder_patch(text):
    text=replace_once(text,'class FiniteBoundaryOps:',DIAGONAL_RULES+'class FiniteBoundaryOps:')
    text=replace_once(text,"            'attention_gate':[(i,1) for i in range(4)],",
        "            'attention_gate':[(i,1) for i in range(4)],\n"
        "            'attention_own_scores':[(0,2),(1,2),(2,2),(3,0)],\n"
        "            'attention_own_uv':[(0,2),(1,2),(2,0)],\n"
        "            'attention_own_endpoint':[(0,1),(1,2),(2,2),(3,2),(4,0)],")
    text=replace_once(text,"                        ('attention_gate',_attention_gate_rule),('attention_input',_attention_input_rule),",
        "                        ('attention_gate',_attention_gate_rule),('attention_input',_attention_input_rule),\n"
        "                        ('attention_own_scores',_attention_own_scores_rule),\n"
        "                        ('attention_own_uv',_attention_own_uv_rule),\n"
        "                        ('attention_own_endpoint',_attention_own_endpoint_rule),")
    text=replace_once(text,"pv_rule='content1',consume_captures=False,input_shape=None):",
        "pv_rule='content1',consume_captures=False,input_shape=None,conditional=False):")
    text=replace_once(text,"    assert tuple(input_shape)==(2*b,t,width) and lse.shape==(2*b,heads,t)",
        "    assert tuple(input_shape)==(2*b,t,width)\n"
        "    if not conditional:assert lse.shape==(2*b,heads,t)")
    text=replace_once(text,"    mcontent,mgate=boundaries.attention_gate(c['q_proj_output'][0::2],c['q_proj_output'][1::2],\n        c['attention_output'][0::2],upstream,_linear_weights(module.o_proj),heads,dim)",
'''    endpoints=None
    if conditional:
        if pv_rule!='content1':raise ValueError('Conditional identity uses the fixed content1 gate ordering.')
        from conditional_attention_endpoints import prepare_conditional_endpoints
        endpoints=prepare_conditional_endpoints(c,layout,module.scaling,boundaries)
    attention0=c['attention_output'][0::2] if endpoints is None else endpoints['attention0']
    mcontent,mgate=boundaries.attention_gate(c['q_proj_output'][0::2],c['q_proj_output'][1::2],
        attention0,upstream,_linear_weights(module.o_proj),heads,dim)
    del attention0''')
    text=replace_once(text,"        for name in ('q_proj_output','attention_output'):\n            del c[name]",
        "        del c['q_proj_output']\n"
        "        if conditional:c.pop('attention_output',None)\n"
        "        else:del c['attention_output']")
    text=replace_once(text,"         'v0':c['value'][0::2],'u':mcontent,'lse0':lse[0::2],'lse1':lse[1::2]}",
'''         'v0':c['value'][0::2],'u':mcontent,
         'lse0':lse[0::2] if endpoints is None else endpoints['lse0'],
         'lse1':lse[1::2] if endpoints is None else endpoints['lse0']}
    if endpoints is not None:
        ops.update(v1=c['value'][1::2],own_q0k0=endpoints['own_q0k0'],own_q1k0=endpoints['own_q1k0'],
            own_uv0=boundaries.attention_own_uv(mcontent,c['value'][0::2],endpoints['query_starts']))
        del endpoints''')
    text=replace_once(text,'    coeff=finite_fa(ops,module.scaling,layout,activity)',
        '    coeff=(finite_fa(ops,module.scaling,layout,activity,conditional=True) if conditional\n'
        '           else finite_fa(ops,module.scaling,layout,activity))')
    return text


def runner_patch(text):
    text=replace_once(text,'native_conv_initial_states=False):','native_conv_initial_states=False,conditional_attention=False):')
    text=replace_once(text,'        self.native_conv_initial_states=native_conv_initial_states',
        '        self.native_conv_initial_states=native_conv_initial_states\n        self.conditional_attention=conditional_attention')
    text=replace_once(text,"                'dense_q','dense_k','dense_v'}\n            backend=self.capture_backend",
'''                'dense_q','dense_k','dense_v'}
            if self.conditional_attention and observer is None:
                attention_needed={'q_proj_output','query','key','value','q_norm_input','k_norm_input'}
            backend=self.capture_backend''')
    text=replace_once(text,'                if not offload_mixer:\n                    with torch.no_grad():aux,lse,unused=',
        '                if not offload_mixer and not self.conditional_attention:\n                    with torch.no_grad():aux,lse,unused=')
    text=replace_once(text,'                    if offload_mixer:\n                        aux,lse,unused=',
        '                    if offload_mixer and not self.conditional_attention:\n                        aux,lse,unused=')
    # AppWorld already consumes captures without offload, while TextCraft's
    # pinned owner ties consumption to offload. Preserve each actual choice.
    text=replace_once(text,",input_shape=mixer_input_shape)\n                finite_fla=",
        ",input_shape=mixer_input_shape,conditional=self.conditional_attention)\n                finite_fla=")
    return text


def main():
    decoder_owners=json.loads((COLLECTION/'attention-owner-sources.json').read_bytes())['tasks']
    runner_owners=json.loads((COLLECTION/'conditional-owner-lifetime.json').read_bytes())['tasks']
    helper=(HERE/'conditional_attention_endpoints.py').read_bytes()
    compiled=json.loads((HERE/'compiled-owner.json').read_bytes())
    metadata=dict(status='Prepared-only complete conditional attention composition; not executed or deployed',
        native_FA_replaced=False,model_implementation_replaced=False,production_modified=False,
        helper_sha256=hashlib.sha256(helper).hexdigest(),library_sha256=compiled['library_sha256'],tasks={})
    all_patch=[]
    for task in ('textcraft','appworld'):
        decoder=next(f for f in decoder_owners[task]['files'] if Path(f['path']).name=='qwen35_decoder_finite.py')
        runner=runner_owners[task]['files'][0]
        entries=[];dest=HERE/'composition-prepared'/task;dest.mkdir(parents=True,exist_ok=True)
        for owner,patcher in [(decoder,decoder_patch),(runner,runner_patch)]:
            original=owner['text'];assert hashlib.sha256(original.encode()).hexdigest()==owner['sha256']
            modified=patcher(original);ast.parse(modified)
            name=Path(owner['path']).name;(dest/name).write_bytes(modified.encode())
            diff=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),
                fromfile='a/'+task+'/'+name,tofile='b/'+task+'/'+name))
            all_patch.append(diff)
            entries.append(dict(original={k:v for k,v in owner.items() if k!='text'},
                generated_name=name,generated_sha256=hashlib.sha256(modified.encode()).hexdigest()))
        metadata['tasks'][task]=entries
    patch=''.join(all_patch).encode();(HERE/'composition.patch').write_bytes(patch)
    metadata['patch_sha256']=hashlib.sha256(patch).hexdigest()
    (HERE/'composition-prepared.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
