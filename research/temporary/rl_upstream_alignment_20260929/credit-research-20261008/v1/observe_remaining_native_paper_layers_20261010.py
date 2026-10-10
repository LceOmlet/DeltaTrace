"""Extend the existing passive observer to the remaining native paper layers.

Original public module hooks and LocalCaptureEvents only. The existing
OperatorObservation owns comparison, first-difference selection and operand
capture. All model/operator calls and their results remain untouched.
"""
import hashlib
import inspect
import json
from pathlib import Path

import torch

from inspect_native_identity_operators import OperatorObservation


class FullLayerObservation(OperatorObservation):
    def __enter__(self):
        super().__enter__()
        try:
            from accelerated.native_capture_events import LocalCaptureEvents
            from transformers.integrations.flash_attention import flash_attention_forward

            for i in range(3,len(self.text.layers)):
                layer = self.text.layers[i]

                def begin(module,args,kwargs,i=i):
                    value = args[0] if args else kwargs['hidden_states']
                    self.current = (dict(layer=i,length=value.shape[1])
                                    if value.shape[0]==8 and len(self.first)<4 else None)

                def end(module,args,value):
                    self.current = None

                self.handles.append(layer.register_forward_pre_hook(begin,with_kwargs=True))
                for name,module in layer.named_modules():
                    if isinstance(module,torch.nn.Linear):
                        self.linear_names[id(module)] = (name or 'decoder_output')+'.base'
                        continue  # Existing original Linear return observer.
                    if name and name not in (
                        'input_layernorm','post_attention_layernorm','mlp',
                        'linear_attn','linear_attn.norm','self_attn',
                        'self_attn.q_norm','self_attn.k_norm'):
                        continue

                    def returned(module,args,output,name=name):
                        if self.current is None or len(self.first)==4:
                            return
                        value = output[0] if isinstance(output,tuple) else output
                        if not isinstance(value,torch.Tensor):
                            return
                        layout = 'norm' if name=='linear_attn.norm' else 'time'
                        x = args[0] if args and isinstance(args[0],torch.Tensor) else None
                        self.observe(name or 'decoder_output',value,layout=layout,inputs={'input':x})

                    self.handles.append(module.register_forward_hook(returned))
                self.handles.append(layer.register_forward_hook(end))

            self.fa_pending = {}
            self.fa_code = flash_attention_forward.__code__
            path=Path(self.fa_code.co_filename)
            self.owners.append(dict(operation='native_FA_interface',path=str(path),
                resolved=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                qualname=self.fa_code.co_qualname))
            self.fa_capture = LocalCaptureEvents([self.fa_code],self.fa_event)
            self.fa_capture.__enter__()
            return self
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise

    def fa_event(self,frame,kind,value):
        if self.current is None or len(self.first)==4:
            return
        if kind=='call':
            self.fa_pending[frame.f_code] = {
                name:frame.f_locals[name] for name in ('query','key','value','attention_mask',
                    'dropout','scaling','sliding_window','softcap','is_causal')
                if name in frame.f_locals}
            return
        args = self.fa_pending.pop(frame.f_code)
        output = value[0] if isinstance(value,tuple) else value
        self.observe('native_FA_interface',output,inputs=args,artifact=args)

    def __exit__(self,*exc):
        capture = getattr(self,'fa_capture',None)
        if capture is not None:
            capture.__exit__(*exc)
        result = super().__exit__(*exc)
        if 'operator_observation' in self.active:
            self.active['operator_observation']['scope'] = (
                'Original B8 four paper examples. Existing first-three-layer observer plus '
                'public hooks on all later decoders and native FA interface; recording '
                'stops after all four pairs have a first observed difference. No native result replaced.')
            path=self.out/f"rank{self.active['rank']}-batch{self.active['index']}-operators.json"
            path.write_text(json.dumps(self.active['operator_observation'],indent=2)+'\n')
        return result
