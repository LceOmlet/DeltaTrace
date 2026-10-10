"""Passive native-head return capture before root FSDP releases full weights."""
import hashlib
import inspect
from pathlib import Path


def install(head, record, microbatch_index, output, rank, notify):
    import torch
    from capture_native_loss_backward_20261010 import observe_call

    native = head.FusedLinearForPPO.forward
    if microbatch_index is None:
        return lambda: None

    def observed(result, arguments):
        index = len(record['microbatches'])
        if index != microbatch_index or record.get('live_head_capture'):
            return
        path = Path(output)/f'rank{rank}-native-head-live-microbatch{index}.pt'
        values = {key:arguments[key].detach().to('cpu',copy=True)
                  for key in ['hidden_states','vocab_weights','input_ids']}
        values.update(token_log_probs=result[0].detach().to('cpu',copy=True),
                      entropy=result[1].detach().to('cpu',copy=True))
        metadata = dict(microbatch_index=index,temperature=arguments['temperature'],
            autocast_enabled=torch.is_autocast_enabled('cuda'),
            autocast_dtype=str(torch.get_autocast_dtype('cuda')),
            original_strides={key:list(arguments[key].stride()) for key in
                              ['hidden_states','vocab_weights','input_ids']},
            head_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(head)).read_bytes()).hexdigest(),
            scope='Native head return, before root FSDP post-forward; no extra model/head call')
        torch.save(dict(tensors=values,metadata=metadata),path)
        record['live_head_capture']=dict(path=str(path),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=metadata)
        notify('native_live_head_captured')

    head.FusedLinearForPPO.forward=observe_call(native,lambda:True,observed,
        lambda error:notify('optional_live_head_capture_error',error=repr(error)))

    def remove():
        head.FusedLinearForPPO.forward=native

    return remove
