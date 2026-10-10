"""Capture existing old/train forwards; defer CPU reads until native update ends.

This diagnostic retains GPU clones and therefore changes allocation/timing.
It does not change native model/loss/gradient/optimizer computations or claim
to recreate the missing original persistent runtime.
"""
import inspect
from pathlib import Path

from capture_native_pre_dt_layers_20261011 import NativeLayerAudit, sha
from capture_native_loss_backward_20261010 import observe_call


class NativeTrainingAudit:
    def __init__(self, model, head, record, output, rank):
        import torch

        self.layers = NativeLayerAudit(model, output, rank)
        self.record = record
        self.output = Path(output)
        self.rank = rank
        self.stage = 'old'
        self.head_values = None
        self.head_errors = []
        self.layers.begin('old_training_comparison')
        self.native_head = head.FusedLinearForPPO.forward
        self.head = head

        def capture(result, arguments):
            if self.stage != 'training' or len(record['microbatches']) != 1 or self.head_values is not None:
                return
            self.head_values = {key:arguments[key].detach().clone()
                for key in ['hidden_states', 'vocab_weights', 'input_ids']}
            self.head_values.update(token_log_probs=result[0].detach().clone(),
                                    entropy=result[1].detach().clone())
            self.head_metadata = dict(microbatch_index=1, temperature=arguments['temperature'],
                autocast_enabled=torch.is_autocast_enabled('cuda'),
                autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                original_strides={key:list(arguments[key].stride()) for key in
                                  ['hidden_states', 'vocab_weights', 'input_ids']},
                head_source_sha256=sha(inspect.getsourcefile(head)),
                scope='Native second B4 head return; GPU clones now, CPU reads after update')

        head.FusedLinearForPPO.forward = observe_call(self.native_head, lambda:True, capture,
                                                     lambda error:self.head_errors.append(repr(error)))

    def hold_old(self):
        self.layers.active = False
        self.old = (self.layers.label, self.layers.tensors, self.layers.counts, self.layers.errors)
        self.layers.tensors = {}
        self.stage = 'inactive'

    def begin_training(self):
        self.stage = 'training'
        self.layers.begin('training_second_B4')

    def finish(self):
        import torch

        self.stage = 'inactive'
        self.layers.close()
        self.head.FusedLinearForPPO.forward = self.native_head
        train = self.layers.flush()
        self.layers.label, self.layers.tensors, self.layers.counts, self.layers.errors = self.old
        old = self.layers.flush()
        self.old = None
        head_record = None
        if self.head_values is not None:
            path = self.output / f'rank{self.rank}-native-training-head-second-B4.pt'
            values = {}
            for key in list(self.head_values):
                value = self.head_values.pop(key)
                values[key] = value.to('cpu', copy=True)
                del value
            torch.save(dict(tensors=values, metadata=self.head_metadata), path)
            head_record = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                               metadata=self.head_metadata)
        return dict(scope=__doc__, layers=[old, train], head=head_record,
                    head_errors=self.head_errors, CPU_reads_deferred_until_update_ended=True)
