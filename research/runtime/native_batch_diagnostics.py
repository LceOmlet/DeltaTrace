"""Passive CPU diagnostics for two real forwards; never substitutes model output."""
import hashlib
import json
from pathlib import Path
import numpy as np


def metrics(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = a-b
    norm = np.linalg.norm(a.ravel())
    return {'shape': list(a.shape), 'relative_L2': float(np.linalg.norm(delta.ravel()) / max(norm, 1e-30)),
        'max_abs': float(np.abs(delta).max()), 'mean_abs': float(np.abs(delta).mean()),
        'max_abs_over_reference_RMS': float(np.abs(delta).max() / max(norm/np.sqrt(a.size), 1e-30)),
        'changed_elements': int(np.count_nonzero(delta)), 'elements': int(a.size)}


class NativeBatchDiagnostics:
    def __init__(self, directory, valid_length, report):
        self.directory = Path(directory)
        self.valid_length = valid_length
        self.report = report
        self.baseline = {}
        self.prefixes = [{}, {}]
        self.fa = None
        self.raw = [{}, {}]
        self.report.update(module_comparisons=[], module_receipts=[[], []], artifacts=[])

    @staticmethod
    def numpy(tensor):
        return tensor.detach().float().cpu().numpy()

    def observe(self, name, tensor, call_index, persist_prefix=False):
        value = tensor[0, :self.valid_length].detach().cpu().contiguous()
        array = self.numpy(value)
        self.report['module_receipts'][call_index].append({'name': name, 'shape': list(array.shape),
            'sha256_float32_values': hashlib.sha256(array.tobytes()).hexdigest()})
        if persist_prefix:
            self.raw[call_index][name] = array[:129].copy()
        if call_index == 0:
            assert name not in self.baseline
            self.baseline[name] = value
        else:
            reference = self.baseline.pop(name)
            self.report['module_comparisons'].append({'name': name, **metrics(self.numpy(reference), array)})

    def capture_fla(self, local, output, call_index):
        if self.prefixes[call_index]: return
        assert local['initial_state'] is None and local['use_qk_l2norm_in_kernel']
        values = {k: local[k][:1, :129].detach().cpu().contiguous() for k in ['q','k','v','g','beta']}
        values['output'] = output[0][:1, :129].detach().cpu().contiguous()
        self.prefixes[call_index] = values
        self.persist('FLA_prefix_call' + str(call_index), {k:self.numpy(v) for k,v in values.items()},
                     {k:str(v.dtype) for k,v in values.items()})

    def capture_fa(self, local, output, call_index):
        if call_index != 0 or self.fa is not None: return
        import torch
        assert isinstance(output, torch.Tensor)
        assert local['dropout_p'] == 0 and local['causal'] and not local['return_attn_probs']
        assert local['alibi_slopes'] is None and local['window_size'] == (-1, -1)
        assert local.get('softcap', 0) == 0
        self.fa = {k:local[k].detach().cpu().contiguous() for k in ['q','k','v']}
        self.fa.update(output=output.detach().cpu().contiguous(), softmax_scale=local['softmax_scale'])
        self.persist('FA_dense_actual_call0', {k:self.numpy(v) for k,v in self.fa.items() if k!='softmax_scale'},
                     {'dtype':'torch.bfloat16', 'softmax_scale':self.fa['softmax_scale']})

    def persist(self, name, arrays, metadata=None):
        path = self.directory / (name + '.npz')
        temp = path.with_suffix('.partial')
        with temp.open('wb') as f: np.savez_compressed(f, **arrays)
        temp.replace(path)
        receipt = {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'keys':list(arrays),'metadata':metadata}
        self.report['artifacts'].append(receipt)
        # Make diagnostic arrays recoverable even if a later model call fails.
        manifest = self.directory / 'diagnostic_artifacts.json'
        manifest.write_text(json.dumps(self.report, indent=2))
        return receipt

    def flush_probes(self, call_index):
        if self.raw[call_index]:
            self.persist('native_module_prefixes_call' + str(call_index), self.raw[call_index])
