"""Contract actual paired differences with saved DT coefficients, without rescaling."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
s = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
n, b, a = s['native'], s['boundaries'], s['attention_values']

def dot(c, delta):
    return float((c.double()*delta.double()).sum())

def effect(c, pair):
    return dot(c, pair[1::2].double()-pair[0::2].double())

up = s['upstream']
mnorm = b['mlp'][0]['output']
mmixer = b['norm_residual'][0]['output']
mi = b['attention_input'][0]['output']
mc, mg = b['attention_gate'][0]['output']
residual = effect(mmixer, n['input_norm']['input'])
content = effect(mc, a['attention_output'].transpose(1, 2))
batch, heads, time, dim = mc.shape
qproj = a['q_proj_output'].reshape(2*batch, time, heads, 2*dim)
gate = effect(mg, qproj[..., dim:])
fa, terms = s['fa'], {}
ops, coeff = fa['operands'], fa['coefficients']
terms['dq'] = dot(coeff['dq'], ops['q1'].double()-ops['q0'].double())
start = fa['query_start']
groups = heads//ops['k0'].shape[1]
for name, paired in [('dk', a['key']), ('dv', a['value'])]:
    delta = (paired[1::2].double()-paired[0::2].double())[:,:,start:]
    terms[name] = dot(coeff[name], delta.repeat_interleave(groups, 1))
stages = [
    ('decoder_output', effect(up, n['decoder']['output'])),
    ('decoder_residual_addition', effect(up, n['post_norm']['input'])+effect(up,n['mlp']['output'])),
    ('mlp_and_output_residual', effect(up, n['post_norm']['input'])+effect(mnorm, n['post_norm']['output'])),
    ('post_norm', effect(mmixer, n['post_norm']['input'])),
    ('attention_and_input_residual', residual+effect(mmixer, n['attention']['output'])),
    ('attention_gate_and_output_projection', residual+content+gate),
    ('finite_attention', residual+sum(terms.values())+gate),
    ('attention_input_projection_rope_norm', residual+effect(mi, n['input_norm']['output'])),
    ('input_norm', effect(s['input_coefficients'], n['input_norm']['input'])),
]
result = dict(scope=__doc__, source=str(args.source), layer=s['layer'],
    stages=[dict(name=name, effect=value,
                 previous_minus_current=stages[i-1][1]-value if i else None)
            for i,(name,value) in enumerate(stages)],
    total_residual=stages[0][1]-stages[-1][1], fa_terms=terms,
    fa_operand_dtypes={k:str(v.dtype) for k,v in ops.items()},
    fa_coefficient_dtypes={k:str(v.dtype) for k,v in coeff.items()},
    native_residual_addition_exact={
        'decoder':torch.equal(n['decoder']['output'],n['post_norm']['input']+n['mlp']['output']),
        'attention':torch.equal(n['post_norm']['input'],n['input_norm']['input']+n['attention']['output'])},
    native_shapes={k:dict(input=list(v['input'].shape),output=list(v['output'].shape)) for k,v in n.items()},
    attention_shapes={k:list(v.shape) for k,v in a.items() if isinstance(v,torch.Tensor)})
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
