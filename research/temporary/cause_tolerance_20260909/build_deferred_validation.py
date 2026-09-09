"""Derive an explicit DT validation-scheduling candidate from frozen sources.

Changes only diagnostic predicates and collection. Generated modules do not
replace any native forward, attention, Torch function or native backward.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path


root=Path(__file__).resolve().parents[3]
here=Path(__file__).resolve().parent
source=root/'deltatrace/clean/qwen3'
manifest=json.loads((root/'deltatrace/clean/sources.json').read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
receipt=[]

def derive(name,outname,replacements):
    path=source/name;raw=path.read_bytes()
    assert sha(raw)==manifest['models']['qwen3']['files'][path.relative_to(root).as_posix()]['sha256']
    original=raw.decode().replace('\r\n','\n');modified=original
    for old,new in replacements:
        assert modified.count(old)==1,(name,old,modified.count(old))
        modified=modified.replace(old,new)
    ast.parse(modified)
    output=here/outname;output.write_text(modified,encoding='utf-8',newline='\n')
    receipt.append({'source':path.relative_to(root).as_posix(),'source_sha256':sha(raw),
                    'output':outname,'output_sha256':sha(output.read_bytes()),
                    'diff':''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=name,tofile=outname))})

derive('qwen_public_fa_layer_replay.py','qwen3_deferred_replay.py',[
 ('def __init__(self,model,checkpoint,reference=None,activity=None):','def __init__(self,model,checkpoint,reference=None,activity=None,validation=None):\n        assert validation is not None\n        self.validation=validation'),
 ('assert lse.dtype==torch.float32 and torch.isfinite(lse).all()',"assert lse.dtype==torch.float32\n        self.validation.finite(lse,f'layer{index}.public_lse')"),
 ("output_exact=bool(torch.equal(auxiliary_out,values['fa_out']))\n        assert output_exact, 'Public metadata invocation changed endpoint attention output'", "output_exact=self.validation.equal(auxiliary_out,values['fa_out'],f'layer{index}.public_output')"),
 ("bool(torch.equal(values['x'],self.checkpoint['layers'][index]['x']))", "self.validation.equal(values['x'],self.checkpoint['layers'][index]['x'],f'layer{index}.replay_input')"),
 ("bool(torch.equal(values['out'],expected_output))", "self.validation.equal(values['out'],expected_output,f'layer{index}.replay_output')"),
 ("        assert check['native_input_exact'] and check['native_output_exact'],f'Original decoder boundary replay mismatch at layer{index}'", "        # All predicates are checked before the public DT API returns.")])

derive('qwen_signed_secant_vendor_fa.py','qwen3_deferred_finite.py',[
 ('finite_activity=None):','finite_activity=None,validation=None):\n    assert validation is not None'),
 ("assert torch.equal(reconstructed_q, qs.transpose(1, 2)) and torch.equal(reconstructed_k, ks.transpose(1, 2))", "validation.equal(reconstructed_q,qs.transpose(1,2),f'layer{layer.self_attn.layer_idx}.rope_q')\n        validation.equal(reconstructed_k,ks.transpose(1,2),f'layer{layer.self_attn.layer_idx}.rope_k')"),
 ("assert torch.equal(values['v'].view(1, n, -1, h), vs)","validation.equal(values['v'].view(1,n,-1,h),vs,f'layer{layer.self_attn.layer_idx}.value')"),
 ("assert torch.equal(values['fa_out'].reshape_as(values['concat']), values['concat'])","validation.equal(values['fa_out'].reshape_as(values['concat']),values['concat'],f'layer{layer.self_attn.layer_idx}.concat')"),
 ('assert torch.isfinite(lse).all()',"validation.finite(lse,f'layer{layer.self_attn.layer_idx}.lse')"),
 ("assert torch.equal(product, raw['product'].to(device)), 'Native SwiGLU product mismatch'","validation.equal(product,raw['product'].to(device),f'layer{li}.swiglu')"),
 ('assert all(torch.isfinite(value).all() for value in outputs.values())',"for key,value in outputs.items():validation.finite(value,f'layer{li}.finite_FA_{key}')"),
 ("assert (outputs['tau']>0).all()","validation.positive(outputs['tau'],f'layer{li}.tau')"),
 ('assert torch.isfinite(m).all()',"validation.finite(m,f'layer{li}.multiplier')"),
 ('float(m.abs().max())','validation.max_abs(m)')])

derive('qwen_signed_secant_paired_vendor_fa.py','qwen3_deferred_pair.py',[
 ('from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews,EndpointReplay',
  'from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair,PairedReplayViews as FrozenPairedReplayViews,EndpointReplay\nfrom qwen3_deferred_replay import NativeLayerReplay\nfrom deferred_validation import DeferredValidation\n\nclass PairedReplayViews(FrozenPairedReplayViews):\n    def __init__(self,model,master,activity=None,validation=None):\n        self.native=NativeLayerReplay(model,master,activity=activity,validation=validation)\n        self.index=None;self.values=None;self.layout_records=[]'),
 ('from qwen_signed_secant_vendor_fa import propagate_signed_secant','from qwen3_deferred_finite import propagate_signed_secant'),
 ('master=before[\'paired_checkpoint\'];paired=PairedReplayViews(model,master,activity=activity)',
  "validation=DeferredValidation()\n    master=before['paired_checkpoint'];paired=PairedReplayViews(model,master,activity=activity,validation=validation)"),
 ('finite_attention=finite_attention,finite_activity=finite_activity)','finite_attention=finite_attention,finite_activity=finite_activity,validation=validation)'),
 ('        return result','        return validation.finish(result)')])
(here/'deferred_validation_derivation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'generated_modules':len(receipt),'native_functions_changed':0,'finite_arithmetic_changed':False}))
