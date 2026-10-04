"""Passive ranges around the current DT owner's parameter callbacks.

This diagnostic context calls the same original bound methods. It supplies no
unshard, reshard, collective, tensor transport, prefetch or finite computation.
Use around one existing attribute invocation and export its original profiler.
"""
import types


class NativeFiniteParameterRanges:
    """Label existing ``prepare_finite_layer(layer)``/release calls only."""

    def __init__(self, model):
        self.model = model
        self.indices = {id(layer): index for index, layer in
                        enumerate(model.model.language_model.layers)}
        self.saved = []
        self.active = False

    def _restore(self):
        try:
            for name, had_instance, instance_value in reversed(self.saved):
                if had_instance:
                    setattr(self.model, name, instance_value)
                else:
                    delattr(self.model, name)
        finally:
            self.saved.clear()
            self.active = False

    def __enter__(self):
        if self.active:
            raise RuntimeError('NativeFiniteParameterRanges is already active')
        import torch

        def wrapper(original, prefix):
            # Actual owner callback signatures are (self, layer). Preserve
            # the layer identity, return object, and any original exception.
            def labelled(_model, layer):
                index = self.indices[id(layer)]
                with torch.profiler.record_function(prefix + str(index)):
                    return original(layer)
            return types.MethodType(labelled, self.model)

        methods = [(name, getattr(self.model, name), prefix) for name, prefix in (
            ('prepare_finite_layer', 'DT_native_prepare_finite_layer_'),
            ('release_finite_layer', 'DT_native_release_finite_layer_'))]
        if any(not callable(original) for _, original, _ in methods):
            raise TypeError('Current DT owner requires both original layer callbacks')
        self.active = True
        try:
            for name, original, prefix in methods:
                had_instance = name in vars(self.model)
                instance_value = vars(self.model).get(name)
                setattr(self.model, name, wrapper(original, prefix))
                self.saved.append((name, had_instance, instance_value))
        except BaseException:
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._restore()
        return False

