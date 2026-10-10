"""Observe native old/ref/old calls; never replace a model or loss computation.

Clone the second B4 decoder outputs on their existing GPU stream, then save
them after the native RPC returns. Clones change allocation/timing; this is
localization data, not a reconstruction of the missing failed runtime.
"""
import hashlib
import inspect
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class NativeLayerAudit:
    def __init__(self, model, output, rank):
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

        self.model = model
        self.output = Path(output)
        self.rank = rank
        self.active = False
        self.tensors = {}
        self.counts = {}
        self.handles = []
        self.errors = []
        texts = [m for m in model.modules() if isinstance(m, Qwen3_5TextModel)]
        assert len(texts) == 1, len(texts)
        text = texts[0]
        self.native_source = Path(inspect.getsourcefile(Qwen3_5TextModel))
        self.text = text
        self.layer_count = len(text.layers)
        assert self.layer_count == text.config.num_hidden_layers

        def observe(name, module, args, kwargs, result):
            if not self.active:
                return
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            if index != 1:
                return
            try:
                self.tensors[name] = result.detach().clone()
                if name == 'layer00':
                    self.tensors['layer00_input'] = args[0].detach().clone()
                    for i, value in enumerate(kwargs.get('position_embeddings', ())):
                        self.tensors[f'rotary_{i}'] = value.detach().clone()
                    value = kwargs.get('position_ids')
                    if isinstance(value, torch.Tensor):
                        self.tensors['position_ids'] = value.detach().clone()
                if getattr(module, 'block_type', None) == 'full_attention' and 'full_attention_mask' not in self.tensors:
                    value = kwargs.get('attention_mask')
                    if isinstance(value, torch.Tensor):
                        self.tensors['full_attention_mask'] = value.detach().clone()
            except Exception as error:
                self.errors.append(dict(name=name, error=repr(error)))

        for i, layer in enumerate(text.layers):
            name = f'layer{i:02d}'
            self.handles.append(layer.register_forward_hook(
                lambda module, args, kwargs, result, n=name: observe(n, module, args, kwargs, result),
                with_kwargs=True))
        self.handles.append(text.norm.register_forward_hook(
            lambda module, args, kwargs, result: observe('final_norm', module, args, kwargs, result),
            with_kwargs=True))

    def begin(self, label):
        assert not self.tensors
        self.label = label
        self.counts = {}
        self.errors = []
        self.active = True

    def flush(self):
        import torch

        self.active = False
        folder = self.output / f'rank{self.rank}-{self.label}-layers'
        folder.mkdir(exist_ok=False)
        files = []
        for name in list(self.tensors):
            value = self.tensors.pop(name)
            path = folder / f'{name}.pt'
            tensor = value.detach().to('cpu', copy=True)
            torch.save(tensor, path)
            files.append(dict(name=name, path=str(path), sha256=sha(path),
                bytes=path.stat().st_size, shape=list(tensor.shape), dtype=str(tensor.dtype),
                finite=bool(torch.isfinite(tensor).all())))
            del value, tensor
        return dict(label=self.label, counts=self.counts, errors=self.errors,
            files=files, expected_decoder_layers=self.layer_count,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())

    def fingerprint(self):
        import torch

        # Read completed CPU-offloaded state; do not hash an in-flight D2H copy.
        # This synchronization is an explicit diagnostic timing perturbation.
        torch.cuda.synchronize()
        fields = []
        for kind, items in [('parameter', self.model.named_parameters()), ('buffer', self.model.named_buffers())]:
            for name, value in items:
                local = value.to_local() if hasattr(value, 'to_local') else value
                cpu = local.detach().cpu().contiguous()
                digest = hashlib.sha256(memoryview(cpu.reshape(-1).view(torch.uint8).numpy())).hexdigest()
                fields.append(dict(kind=kind, name=name, shape=list(value.shape),
                    local_shape=list(local.shape), dtype=str(local.dtype), device=str(local.device),
                    requires_grad=value.requires_grad, sha256=digest,
                    placements=[str(x) for x in getattr(value, 'placements', ())]))
        return dict(fields=fields, text_attention_implementation=self.text.config._attn_implementation,
            model_training=self.model.training, float32_matmul_precision=torch.get_float32_matmul_precision(),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            native_layer_sources={str(self.native_source):sha(self.native_source)})

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
