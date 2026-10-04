"""Prepared-only owner scheduling: retain original Qwen3.5 root captures.

The unchanged runner supplies its capture factory. No model, capture body,
finite operator, cache, parameter lifetime or tensor transport is implemented
here. Captures belong to one attribute invocation and are consumed once.
"""
from contextlib import ExitStack
import sys


class NativeQwen35RootTape:
    def __init__(self):
        self.entries = {}
        self.versions = {}
        self.active = None
        self.handles = []
        self.captured_layers = 0
        self.consumed_layers = 0
        self.version_checks = 0

    @staticmethod
    def _tensors(dc, mc):
        for scope, values in (("decoder", dc.values), ("mixer", mc.values),
                              ("endpoints", getattr(mc, "endpoints", {}))):
            for name, value in values.items():
                if value is not None and hasattr(value, "untyped_storage"):
                    yield scope + "." + name, value

    @staticmethod
    def _release(dc, mc):
        dc.values.clear()
        mc.values.clear()
        getattr(mc, "endpoints", {}).clear()

    def _finish_layer(self, exc_info):
        if self.active is None:
            return
        index, stack, dc, mc, offload_mixer, post_handle = self.active
        self.active = None
        retained = False
        try:
            stack.__exit__(*exc_info)
            if exc_info[0] is None:
                # Same version-counter check used by the existing Qwen3 root
                # tape. It reads metadata only; no mutation-audit clones.
                self.versions[index] = {name: value._version
                                        for name, value in self._tensors(dc, mc)}
                self.entries[index] = (dc, mc, offload_mixer)
                self.captured_layers += 1
                retained = True
        finally:
            if post_handle is not None:
                post_handle.remove()
            if not retained:
                self._release(dc, mc)

    def _start_layer(self, index, layer, factory):
        if self.active is not None or index in self.entries:
            raise RuntimeError("Root capture tape requires one original sequential layer call.")
        dc, mc, offload_mixer = factory(index, layer)
        stack = ExitStack()
        self.active = (index, stack, dc, mc, offload_mixer, None)
        try:
            stack.enter_context(dc)
            stack.enter_context(mc)
            # Owner dc registers its decoder-output hook during __enter__.
            # Register after that hook so all original fields/calls are ready.
            def after(_layer, _args, _kwargs, _output):
                self._finish_layer(sys.exc_info())
                return None
            post = layer.register_forward_hook(after, with_kwargs=True, always_call=True)
            self.active = (index, stack, dc, mc, offload_mixer, post)
        except BaseException:
            self._finish_layer(sys.exc_info())
            raise
        return None

    def capture_scope(self, layers, factory):
        tape = self
        class RootScope:
            def __enter__(self):
                try:
                    for index, layer in enumerate(layers):
                        def before(module, _args, _kwargs, index=index):
                            return tape._start_layer(index, module, factory)
                        tape.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
                except BaseException:
                    self.__exit__(*sys.exc_info())
                    raise
                return tape

            def __exit__(self, *exc_info):
                try:
                    tape._finish_layer(exc_info)
                finally:
                    for handle in reversed(tape.handles):
                        handle.remove()
                    tape.handles.clear()
                return False
        return RootScope()

    def pop(self, index):
        dc, mc, offload_mixer = self.entries[index]
        for name, value in self._tensors(dc, mc):
            if value._version != self.versions[index][name]:
                raise RuntimeError(f"Original root capture mutated before finite consumption: layer {index} {name}")
            self.version_checks += 1
        self.versions.pop(index)
        self.consumed_layers += 1
        return self.entries.pop(index)

    def report(self):
        return {"prepared_only": True, "captured_layers": self.captured_layers,
                "consumed_layers": self.consumed_layers,
                "unconsumed_layers": len(self.entries), "version_checks": self.version_checks,
                "native_decoder_replay_calls": 0, "snapshot_copies_added": 0}

    def clear(self):
        try:
            self._finish_layer((RuntimeError, RuntimeError("Root tape cleanup"), None))
        finally:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            for dc, mc, _offload_mixer in self.entries.values():
                self._release(dc, mc)
            self.entries.clear()
            self.versions.clear()
