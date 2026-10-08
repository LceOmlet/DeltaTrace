"""Locate the first nonfinite original finite callback, without changing its math.

Only the explicitly requested diagnostic uses this observer. Finite returns are
returned as the original object. A bad return is saved and stops the one-call
replay, instead of propagating it through the remaining layers.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch


class SeedCaptured(Exception):
    """The requested precast operand was saved; no full DT result is claimed."""


class PassiveNonfinite:
    def __init__(self, runner, out, rank):
        self.runner, self.out, self.rank = runner, Path(out), rank
        self.restore, self.layer, self.calls = [], None, []
        self.layers = {id(v):i for i,v in enumerate(runner.model.model.language_model.layers)}
        self.capture_seed=os.environ.get('DT_CAPTURE_FLA_PRECAST')=='1'

    @staticmethod
    def tensors(value, path='return'):
        if isinstance(value,torch.Tensor):yield path,value
        elif isinstance(value,dict):
            for key,item in value.items():yield from PassiveNonfinite.tensors(item,path+'.'+str(key))
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):yield from PassiveNonfinite.tensors(item,path+'.'+str(i))

    @staticmethod
    def freeze(value):
        if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
        if isinstance(value,dict):return {k:PassiveNonfinite.freeze(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return type(value)(PassiveNonfinite.freeze(v) for v in value)
        if value is None or isinstance(value,(str,int,float,bool)):return value
        return dict(type=type(value).__qualname__,identity=id(value),representation=repr(value)[:1000])

    @staticmethod
    def bad(value):
        result=[]
        for path,tensor in PassiveNonfinite.tensors(value):
            if not tensor.is_floating_point():continue
            flat=tensor.reshape(-1)
            for first in range(0,flat.numel(),1048576):
                part=flat[first:first+1048576]
                valid=torch.isfinite(part)
                if not bool(valid.all()):
                    slots=(~valid).nonzero().flatten()
                    result.append(dict(path=path,shape=list(tensor.shape),dtype=str(tensor.dtype),
                        chunk_start=first,bad_count_in_chunk=int(slots.numel()),
                        first_flat_indices=(slots[:8]+first).cpu().tolist(),
                        first_values=part[slots[:8]].cpu().tolist()))
                    break
        return result

    def wrap(self,name,owner):
        def call(*args,**kwargs):
            previous=self.layer
            if name=='decoder':self.layer=self.layers[id(args[0])]
            try:
                value=owner(*args,**kwargs)
                if self.capture_seed and name=='gdn_norm_gate' and self.layer==4:
                    artifact=self.out/f'rank{self.rank}-precast-seed.pt'
                    torch.save(dict(mo=self.freeze(value[0]),output_dtype=args[0].dtype,
                        callback=name,layer=self.layer),artifact)
                    record=dict(path=str(artifact),bytes=artifact.stat().st_size,
                        sha256=hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest(),
                        scope='Original FP32 norm-gate cotangent before BF16 and FP16 conversion; intentional early stop, not a completed DT.')
                    (self.out/f'rank{self.rank}-precast-seed.json').write_text(json.dumps(record,indent=2)+'\n')
                    raise SeedCaptured('Original layer4 cotangent saved before native dtype conversion.')
                if self.capture_seed:return value
                bad=self.bad(value)
                self.calls.append(dict(callback=name,layer=self.layer,finite=not bad))
                if bad:
                    artifact=self.out/f'rank{self.rank}-first-nonfinite.pt'
                    torch.save(dict(callback=name,layer=self.layer,args=self.freeze(args),
                        kwargs=self.freeze(kwargs),output=self.freeze(value)),artifact)
                    record=dict(callback=name,layer=self.layer,bad=bad,calls=self.calls,unix=time.time(),
                        artifact=dict(path=str(artifact),bytes=artifact.stat().st_size,
                            sha256=hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest()),
                        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
                        free=torch.cuda.mem_get_info()[0],original_object_returned_on_finite=True,
                        source=inspect.getsourcefile(owner if inspect.isfunction(owner) or inspect.ismethod(owner) else type(owner)))
                    (self.out/f'rank{self.rank}-first-nonfinite.json').write_text(json.dumps(record,indent=2)+'\n')
                    raise ValueError(f'Passive observer: first nonfinite return in {name}, layer {self.layer}; exact operands saved.')
                return value
            finally:self.layer=previous
        return call

    def bind(self,obj,key,name,mapping=False):
        original=obj[key] if mapping else getattr(obj,key)
        self.restore.append((obj,key,original,mapping))
        if mapping:obj[key]=self.wrap(name,original)
        else:setattr(obj,key,self.wrap(name,original))

    def __enter__(self):
        namespace=self.runner.attribute.__func__.__globals__
        gdn_namespace=namespace['gdn_finite_pullback'].__globals__
        self.bind(gdn_namespace,'_l2_pullback','GDN_qk_normalization',True)
        self.bind(namespace,'decoder_finite_pullback','decoder',True)
        self.bind(namespace,'gdn_finite_pullback','GDN',True)
        self.bind(namespace,'attention_finite_pullback','attention',True)
        self.bind(self.runner,'finite_fa','finite_FA')
        self.bind(self.runner,'finite_fla','finite_FLA')
        for layer in self.runner.finite_fla_by_layer:
            self.bind(self.runner.finite_fla_by_layer,layer,'finite_FLA_layer_'+str(layer),True)
        self.bind(self.runner,'answer','answer')
        for name in ('norm_residual','gdn_norm_gate','gdn_conv_silu'):
            self.bind(self.runner.boundaries,name,name)
        return self

    def __exit__(self,*_exc):
        for obj,key,original,mapping in reversed(self.restore):
            if mapping:obj[key]=original
            else:setattr(obj,key,original)
        (self.out/f'rank{self.rank}-finite-callbacks.json').write_text(json.dumps(self.calls,indent=2)+'\n')
