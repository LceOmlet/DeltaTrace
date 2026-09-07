"""Passive access to actual FLA intermediates for finite-extension development.

No forward/backward function is replaced. Captured tensors share native storage;
the caller must not mutate them and must release this per-invocation state.
This is a pinned-source development interface, not a complete attribution method
or an interface promised to work unchanged with arbitrary FLA releases.
"""
import hashlib
import importlib
import inspect
import sys
from pathlib import Path


class NativeFLAStageCapture:
    def __init__(self, expected_sources):
        chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
        state = importlib.import_module('fla.ops.common.chunk_delta_h')
        wy = importlib.import_module('fla.ops.gated_delta_rule.wy_fast')
        modules = {'chunk':chunk, 'state':state, 'wy':wy}
        if set(expected_sources) != set(modules):
            raise ValueError('All three native source hashes are required.')
        self.sources = {}
        for name, module in modules.items():
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if digest != expected_sources[name]:
                raise ValueError('Native FLA source mismatch: ' + name)
            self.sources[name] = digest
        assert chunk.chunk_gated_delta_rule_fwd_h is state.chunk_gated_delta_rule_fwd_h
        assert chunk.chunk_gated_delta_rule_bwd_dhu is state.chunk_gated_delta_rule_bwd_dhu
        assert chunk.prepare_wy_repr_bwd is wy.prepare_wy_repr_bwd
        self.functions = {
            'forward':chunk.chunk_gated_delta_rule_fwd,
            'backward':chunk.chunk_gated_delta_rule_bwd,
            'state_forward':state.chunk_gated_delta_rule_fwd_h,
            'state_backward':state.chunk_gated_delta_rule_bwd_dhu,
            'wy_backward':wy.prepare_wy_repr_bwd}
        self.names = {inspect.unwrap(fn).__code__:name for name,fn in self.functions.items()}
        self.records = []
        self.counts = {}
        self.active = False

    def _event(self, frame, event, result):
        label = self.names.get(frame.f_code)
        if label is None:
            return
        if event == 'call':
            self.counts[label] = self.counts.get(label,0) + 1
        if event != 'return' or result is None:
            return
        import torch
        # Explicitly retain only named native values used by the finite contract.
        local = frame.f_locals
        keys = {
            'forward':['q','k','v','g','beta','A','w','u','h','v_new','o'],
            'backward':['q','k','v','g','beta','A','w','u','h','v_new','do','dh','dq','dk','dv','db','dg'],
            'state_forward':['k','w','u','g'],
            'state_backward':['q','k','w','g','do','dv'],
            'wy_backward':['k','v','beta','g','A','dw','du']}[label]
        values = {key:local[key].detach() for key in keys if isinstance(local.get(key),torch.Tensor)}
        if label == 'state_forward':
            values.update(h=result[0].detach(), v_new=result[1].detach())
        elif label == 'state_backward':
            values.update(dh=result[0].detach(), dU_WY=result[2].detach())
        elif label == 'wy_backward':
            values.update(dk=result[0].detach(), dV=result[1].detach(), db=result[2].detach(), dg=result[3].detach())
        self.records.append({'stage':label,'values':values,
                             'versions':{key:value._version for key,value in values.items()}})

    def __enter__(self):
        if self.active or self.records or sys.getprofile() is not None:
            raise RuntimeError('A fresh capture and unused thread profiling hook are required.')
        self.active = True
        sys.setprofile(self._event)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.setprofile(None)
        self.active = False

    def only(self, name):
        matches = [row for row in self.records if row['stage']==name]
        if len(matches) != 1:
            raise ValueError('Expected one actual native stage: ' + name)
        row = matches[0]
        if any(value._version != row['versions'][key] for key,value in row['values'].items()):
            raise RuntimeError('Native intermediate was mutated after capture: ' + name)
        return row['values']

    def retained_storage_bytes(self):
        storages = {}
        for row in self.records:
            for value in row['values'].values():
                storage = value.untyped_storage()
                storages[(str(value.device),storage.data_ptr())] = storage.nbytes()
        return sum(storages.values())

    def clear(self):
        if self.active:
            raise RuntimeError('Release capture after exiting its context.')
        self.records.clear()
