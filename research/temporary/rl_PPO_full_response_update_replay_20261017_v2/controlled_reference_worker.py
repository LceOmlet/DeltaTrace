"""Test-only reference setup and removable native-operand observation."""


def worker_setup():
    import metax_fsdp_event_wait
    from torch.backends import cudnn

    metax_fsdp_event_wait.worker_setup()
    cudnn.deterministic = True


def install_actual_conv_capture(worker, audit):
    import hashlib
    import inspect
    import json
    from pathlib import Path
    import torch
    from torch.backends import cudnn

    layer = worker.critic_module.model.language_model.layers[0].linear_attn
    function = layer.causal_conv1d_fn
    worker._actual_conv_binding = dict(
        causal_conv1d_fn=None if function is None else repr(function),
        Conv1d_class=type(layer.conv1d).__qualname__,
        Conv1d_owner=inspect.getfile(type(layer.conv1d)),
        test_only_cudnn_deterministic=cudnn.deterministic,
        test_helper_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    handles = []

    def cpu_copy(value):
        value = value.detach()
        if hasattr(value, 'to_local'):
            value = value.to_local()
        return value.cpu().clone()

    def capture_native_call(module, args, output):
        if not output.requires_grad:
            return
        record = dict(worker._actual_conv_binding)
        record.update(rank=worker.rank, input_shape=list(args[0].shape),
                      input_stride=list(args[0].stride()), input_dtype=str(args[0].dtype),
                      weight_dtype=str(module.weight.dtype), output_shape=list(output.shape),
                      output_dtype=str(output.dtype),
                      kernel_size=module.kernel_size, stride=module.stride,
                      padding=module.padding, dilation=module.dilation, groups=module.groups)
        tensors = dict(input=cpu_copy(args[0]), weight=cpu_copy(module.weight),
                       bias=None if module.bias is None else cpu_copy(module.bias))

        def capture_native_output_gradient(gradient):
            tensors['output_gradient'] = cpu_copy(gradient)
            record['output_gradient_stride'] = list(gradient.stride())
            record['output_gradient_dtype'] = str(gradient.dtype)
            torch.save(tensors, audit / f'actual-native-conv-operands-rank{worker.rank}.pt')
            (audit / f'actual-native-conv-operands-rank{worker.rank}.json').write_text(json.dumps(record, indent=2))
            return gradient

        output.register_hook(capture_native_output_gradient)
        handles[0].remove()

    handles.append(layer.conv1d.register_forward_hook(capture_native_call))
