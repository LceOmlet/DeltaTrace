"""Observe original operations inside the already-localized native decoders.

This is an operator diagnosis of original batches 1/11 plus matched 0/10,
not a new quality sample. Public module hooks and LocalCaptureEvents observe
the unchanged native call. No forward, reference, kernel or result is replaced.
Only the first unequal operation per row saves operands, before owner cache
updates. The original diagnostic still stops before finite propagation.
"""
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import inspect_native_identity_roots as roots


def cpu(value):
    if isinstance(value,torch.Tensor):
        return value.detach().to('cpu',copy=True)
    if isinstance(value,(list,tuple)):
        return type(value)(cpu(v) for v in value)
    if isinstance(value,dict):
        return {k:cpu(v) for k,v in value.items()}
    return value


class OperatorObservation:
    def __init__(self,runner,out,active):
        self.runner,self.out,self.active=runner,Path(out),active
        self.handles=[];self.current=None;self.pending={};self.events=[]
        self.first={};self.artifacts=[];self.seconds=0.0
        self.text=runner.model.model.language_model

    def paired(self,value,layout='time'):
        if layout=='conv':return value.transpose(1,2)
        if layout=='norm':
            return value.reshape(8,self.current['length'],-1,value.shape[-1])
        return value

    def stats(self,value):
        assert value.shape[0]==8, value.shape
        rows=[]
        for row in range(4):
            a,b=value[2*row],value[2*row+1]
            equal=torch.equal(a,b)
            item=dict(equal=equal)
            if not equal:
                delta=(a.float()-b.float()).reshape(-1)
                mask=delta.ne(0)
                first=int(mask.to(torch.uint8).argmax())
                coord=[];remainder=first
                for n in reversed(a.shape):coord.append(remainder%n);remainder//=n
                item.update(maxabs=float(delta.abs().max()),changed_elements=int(mask.sum()),
                    first_coordinate=list(reversed(coord)),
                    nonfinite=int((~torch.isfinite(delta)).sum()))
            rows.append(item)
        return dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()),rows=rows)

    def observe(self,name,value,*,layout='time',inputs=None,artifact=None,record_first=True):
        tick=time.perf_counter()
        value=self.paired(value,layout)
        event=dict(layer=self.current['layer'],operation=name,output=self.stats(value))
        if inputs is not None:
            event['inputs']={k:self.stats(v) for k,v in inputs.items()
                             if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==8}
        new=[r for r,x in enumerate(event['output']['rows']) if record_first and not x['equal'] and r not in self.first]
        if new:
            for row in new:self.first[row]=dict(layer=event['layer'],operation=name,event=len(self.events))
            if artifact is not None:
                # Persist full original batch/time/head dimensions, not a
                # cropped recurrence with an invented initial state.
                path=self.out/f"rank{self.active['rank']}-batch{self.active['index']}-layer{event['layer']}-{name}.pt"
                saved=dict(operation=name,layer=event['layer'],new_rows=new,
                    output=cpu(value),operands=cpu(artifact() if callable(artifact) else artifact),output_metadata=event['output'])
                torch.save(saved,path)
                with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=digest,
                    first_unequal_rows=new,operation=name,layer=event['layer'])
                self.artifacts.append(receipt);event['artifact']=receipt
                del saved
        self.events.append(event)
        self.seconds+=time.perf_counter()-tick

    def __enter__(self):
        from accelerated.native_capture_events import LocalCaptureEvents
        from qwen35_gdn_finite import resolve_native_gdn_forward
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
        codes={}
        for i in range(3):
            layer=self.text.layers[i]
            assert layer.block_type!='full_attention'
            mixer=layer.linear_attn
            codes[resolve_native_gdn_forward(type(mixer)).__code__]='GDN'
            codes[inspect.unwrap(mixer.causal_conv1d_fn).__code__]='conv'
            codes[inspect.unwrap(mixer.chunk_gated_delta_rule).__code__]='FLA'
            codes[chunk_gated_delta_rule_fwd.__code__]='FLA_stage'

            def begin(module,args,kwargs,i=i):
                value=args[0] if args else kwargs['hidden_states']
                self.current=(dict(layer=i,length=value.shape[1]) if value.shape[0]==8 else None)
            self.handles.append(layer.register_forward_pre_hook(begin,with_kwargs=True))

            def end(module,args,value):self.current=None
            modules=dict(input_norm=layer.input_layernorm,post_attention_norm=layer.post_attention_layernorm,
                in_proj_qkv=mixer.in_proj_qkv,in_proj_z=mixer.in_proj_z,in_proj_b=mixer.in_proj_b,
                in_proj_a=mixer.in_proj_a,gated_norm=mixer.norm,out_proj=mixer.out_proj,
                mlp_gate=layer.mlp.gate_proj,mlp_up=layer.mlp.up_proj,mlp_down=layer.mlp.down_proj,
                GDN_output=mixer,MLP_output=layer.mlp,decoder_output=layer)
            for name,module in modules.items():
                def hook(module,args,output,name=name):
                    if self.current is None:return
                    value=output[0] if isinstance(output,tuple) else output
                    layout='norm' if name=='gated_norm' else 'time'
                    x=self.paired(args[0],layout) if args and isinstance(args[0],torch.Tensor) else None
                    # Capture a module only if it is the first divergence.
                    # state_dict retains the actual PEFT/base parameters.
                    self.observe(name,value,layout=layout,inputs={'input':x},
                        artifact=lambda:dict(args=args,state_dict=module.state_dict(),module_type=str(type(module))))
                self.handles.append(module.register_forward_hook(hook))
            self.handles.append(layer.register_forward_hook(end))
        self.codes=codes
        self.owners=[]
        for code,label in codes.items():
            path=Path(code.co_filename)
            self.owners.append(dict(operation=label,path=str(path),resolved=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),qualname=code.co_qualname))
        self.capture=LocalCaptureEvents(codes,self.event)
        self.capture.__enter__()
        return self

    def event(self,frame,kind,value):
        if self.current is None:return
        label=self.codes[frame.f_code];f=frame.f_locals
        if label=='GDN':return
        if kind=='call':
            names=(('x','weight','bias','activation','seq_idx','initial_states','return_final_states') if label=='conv'
                   else ('q','k','v','g','beta','scale','initial_state','output_final_state',
                         'cu_seqlens','use_qk_l2norm_in_kernel'))
            self.pending[frame.f_code]={n:f[n] for n in names if n in f}
            return
        args=self.pending.pop(frame.f_code)
        if label=='FLA_stage':
            # Only describe this internal return. The public FLA return saves
            # original raw inputs (including its own L2-normalization flag).
            self.observe(label,value[1],inputs={k:v for k,v in args.items() if k!='initial_state'},record_first=False)
        elif label=='FLA':
            self.observe(label,value[0],inputs=args,artifact=args)
        else:
            output=value[0] if isinstance(value,tuple) else value
            self.observe(label,output,layout='conv',
                inputs={k:(v.transpose(1,2) if k in ('x','initial_states') else v) for k,v in args.items()},
                artifact=args)

    def __exit__(self,*exc):
        try:self.capture.__exit__(*exc)
        finally:
            for handle in self.handles:handle.remove()
            self.handles.clear();self.pending.clear()
            self.active['operator_observation']=dict(events=self.events,
                first_unequal_by_row=self.first,artifacts=self.artifacts,observer_seconds=self.seconds,
                owners=self.owners,
                scope='First three decoders; original batches 0/1/10/11. Operator diagnosis, not population accuracy or a new numerical gate.')
            path=self.out/f"rank{self.active['rank']}-batch{self.active['index']}-operators.json"
            path.write_text(json.dumps(self.active['operator_observation'],indent=2)+'\n')


if __name__=='__main__':
    roots.initializer.make_worker=lambda:roots.make_worker(
        operator_observer=OperatorObservation,batch_indices=(0,1,10,11))
    roots.initializer.main()
