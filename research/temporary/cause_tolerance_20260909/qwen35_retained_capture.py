"""Retain actual native replay tensors; optional copy-at-capture mutation audit.

Subclasses only DT's passive capture helpers. Original model, FA, FLA, layer
calls and finite propagation are inherited unchanged. Root checkpoints are
outside these helpers and retain their original copies.
"""
import torch
from qwen35_decoder_finite import NativeDecoderCapture as _Decoder
from native_dense_attention_capture import NativeDenseAttentionCapture as _Attention
from qwen35_gdn_finite import NativeGDNCapture as _GDN

AUDIT = False
AUDIT_RECORDS = []


class _Retain:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._snapshots = []

    def _actual(self, name, value, device):
        retained = value.detach().to(device)
        if AUDIT:
            self._snapshots.append((name, retained, retained.clone()))
        return retained

    def __exit__(self, *args):
        result = super().__exit__(*args)
        if AUDIT and args[0] is None:
            predicates = [torch.eq(actual, snapshot).all() for _, actual, snapshot in self._snapshots]
            all_equal = bool(torch.stack(predicates).all())
            record = {'capture': type(self).__name__, 'tensors': len(predicates),
                      'all_values_equal_capture_time': all_equal,
                      'changed': [] if all_equal else [name for name, actual, snapshot in self._snapshots
                                                     if not torch.equal(actual, snapshot)]}
            AUDIT_RECORDS.append(record)
            self._snapshots.clear()
            assert all_equal, record
        return result


class NativeDecoderCapture(_Retain, _Decoder):
    def retain(self, name, value):
        self.values[name] = self._actual(name, value, self.destination)


class NativeDenseAttentionCapture(_Retain, _Attention):
    def retain(self, name, value):
        self.values[name] = self._actual(name, value, self.destination)


class NativeGDNCapture(_Retain, _GDN):
    def copy(self, value):
        return None if value is None else self._actual('operand_' + str(len(self._snapshots)), value, self.device)
