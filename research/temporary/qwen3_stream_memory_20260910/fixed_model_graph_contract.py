"""Reject model changes that would invalidate captured original native kernels."""
import json


def tensor_state(value):
    return (id(value),value._version,value.data_ptr(),tuple(value.shape),tuple(value.stride()),str(value.dtype),str(value.device),value.requires_grad)


def module_state(module):
    return (id(module),type(module),type(module).forward,module.__dict__.get('forward'),module.training,
        tuple(module._forward_pre_hooks.items()),tuple(module._forward_hooks.items()),
        tuple(module._forward_pre_hooks_with_kwargs),tuple(module._forward_hooks_with_kwargs),tuple(module._forward_hooks_always_called))


def config_state(model):
    config=model.config
    values={name:getattr(config,name,None) for name in ['_attn_implementation','hidden_act','attention_bias','attention_dropout','rope_theta','rope_scaling','max_position_embeddings','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim']}
    layers=[]
    for layer in model.model.layers:
        attention=layer.self_attn
        layers.append((attention.scaling,attention.sliding_window,attention.head_dim,attention.num_key_value_groups,
            layer.input_layernorm.variance_epsilon,layer.post_attention_layernorm.variance_epsilon,
            attention.q_norm.variance_epsilon,attention.k_norm.variance_epsilon))
    return (json.dumps(values,sort_keys=True),tuple(layers),model.model.norm.variance_epsilon)


class FixedModelGraphContract:
    def __init__(self,model):
        self.model=model
        assert not model.training and model.config._attn_implementation=='flash_attention_2'
        assert all(not p.requires_grad for p in model.parameters())
        self.parameters=tuple((name,tensor_state(p)) for name,p in model.named_parameters())
        self.buffers=tuple((name,tensor_state(p)) for name,p in model.named_buffers())
        self.modules=tuple((name,module_state(m)) for name,m in model.named_modules())
        self.config=config_state(model)

    def validate(self):
        model=self.model
        if self.parameters!=tuple((name,tensor_state(p)) for name,p in model.named_parameters()):
            raise ValueError('Captured model parameter identity, version, storage or layout changed; create a new controller.')
        if self.buffers!=tuple((name,tensor_state(p)) for name,p in model.named_buffers()):
            raise ValueError('Captured model buffer identity, version, storage or layout changed; create a new controller.')
        if self.modules!=tuple((name,module_state(m)) for name,m in model.named_modules()):
            raise ValueError('Captured module, forward callable, hooks or training state changed; create a new controller.')
        if self.config!=config_state(model):
            raise ValueError('Captured model configuration changed; create a new controller.')
