"""Read original FSDP/offload Python metadata without reading tensor values.

Diagnostic only. No state setters, collectives, synchronization, tensor copies,
model calls, or alternative lifecycle are introduced. Values are observations,
not acceptance gates; pending native operations are not automatically defects.
"""
import inspect

import torch


def describe_native_runtime(model):
    groups = []
    contexts = {}
    activation_handlers = {}
    for name, module in model.named_modules():
        get_state = getattr(module, '_get_fsdp_state', None)
        if get_state is not None:
            state = get_state()
            group = state._fsdp_param_group
            context = state._comm_ctx
            context_id = str(id(context))
            if context_id not in contexts:
                contexts[context_id] = dict(
                    post_forward_order_count=len(context.post_forward_order),
                    post_forward_distinct_groups=len({id(g) for g in context.post_forward_order}),
                    all_gather_state_pending=context.all_gather_state is not None,
                    reduce_scatter_state_pending=context.reduce_scatter_state is not None,
                    iteration_forward_root_present=state._state_ctx.iter_forward_root is not None,
                    is_last_backward=state._state_ctx.is_last_backward,
                    post_backward_callback_queued=state._state_ctx.post_backward_final_callback_queued,
                    post_optimizer_event_present=state._state_ctx.post_optim_event is not None,
                    streams={key: getattr(getattr(context, key, None), 'cuda_stream', None)
                             for key in ['all_gather_copy_in_stream', 'all_gather_stream',
                                         'reduce_scatter_stream', 'all_reduce_stream']},
                )
            row = dict(name=name, context_id=context_id,
                       training_state=str(state._training_state),
                       auto_reshard_after_forward=state._auto_reshard_after_forward)
            if group is not None:
                row.update(
                    group_training_state=str(group._training_state),
                    sharded_state=str(group._sharded_state),
                    parameter_count=len(group.fsdp_params),
                    parameter_offload_policy=type(group.offload_policy).__name__,
                    post_forward_indices_count=len(group._post_forward_indices),
                    all_gather_result_pending=group._all_gather_result is not None,
                    post_reduce_event_present=group._post_reduce_event is not None,
                    reshard_after_forward_event_present=group._reshard_after_forward_event is not None,
                    gradient_offload_events=sum(p.grad_offload_event is not None for p in group.fsdp_params),
                    reshard_after_forward=group.post_forward_mesh_info is not None,
                )
            groups.append(row)
        forward = getattr(module.forward, '__func__', module.forward)
        if inspect.isfunction(forward) and forward.__closure__:
            closure = dict(zip(forward.__code__.co_freevars,
                               (cell.cell_contents for cell in forward.__closure__)))
            handler = closure.get('handler')
            if handler is not None and hasattr(handler, '_offload_ctx'):
                offload = handler._offload_ctx.offload_handler
                activation_handlers[str(id(offload))] = dict(
                    type=type(offload).__name__,
                    current_group=offload.current_group,
                    offloaded_group_count=offload.offloaded_group_count,
                    tensor_state_count=len(offload.tensor_tag_to_state),
                    tensor_buffer_count=len(offload.tensor_tag_to_buf),
                    group_offload_mapping_count=len(offload.group_offload_mapping),
                    hook_inside_context=handler._offload_ctx.inside_context,
                    checkpoint_enabled=handler._enable_ckpt,
                )
    return dict(groups=groups, contexts=contexts,
                activation_handlers=activation_handlers,
                current_stream=torch.cuda.current_stream().cuda_stream,
                grad_enabled=torch.is_grad_enabled(),
                autocast_cuda_enabled=torch.is_autocast_enabled('cuda'),
                autocast_cuda_dtype=str(torch.get_autocast_dtype('cuda')),
                tensor_value_reads=0, synchronization_calls=0,
                numerical_state_mutations=0)
