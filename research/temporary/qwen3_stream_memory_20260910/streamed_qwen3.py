"""Refresh pinned finite graph inputs as original native layers finish.

The first call for a geometry uses the unchanged graph builder. Warm calls
retain actual operands for one decoder at a time, copy each original storage
once into the existing graph, and release source references after validation.
No original model, finite operator, precision boundary or graph is replaced.
"""
import time
import weakref
import torch
from qwen3_finite_graph import FiniteGraphQwen3, payload
from qwen3_root_retained import NativeRootTape
from qwen3_graph_storage_inputs import at_path, storage_key, refresh
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair


def memory():
    return {'allocated_bytes': torch.cuda.memory_allocated(),
            'reserved_bytes': torch.cuda.memory_reserved(),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


class StreamTape(NativeRootTape):
    def __init__(self, model, program, *, mutation_audit=False):
        super().__init__(model, mutation_audit=mutation_audit)
        self.program = program
        self.by_layer = [[] for _ in self.layers]
        self.remaining_paths = []
        self.root_representatives = set()
        for item in program.paths:
            path, destination, representative, size = item
            if path[0] in ('layers', 'arguments'):
                self.by_layer[path[1]].append(item)
                self.root_representatives.add(representative)
            else:
                self.remaining_paths.append(item)
        self.remaining_groups = [g for g in program.storage_groups
                                 if g[0] not in self.root_representatives]
        assert all(p[2] not in self.root_representatives for p in self.remaining_paths)
        self.copied = {}
        self.copy_checks = self.mutation_checks = self.copy_bytes = 0
        self.layer_copy_cpu_seconds = 0.0
        self.live_versions = []
        self.completed_layers = 0

    def retain(self, index, name, value):
        super().retain(index, name, value)
        # Weak references do not prolong operand lifetimes. Check original
        # tensors and their view owners again after the whole root finishes.
        self.live_versions.append((weakref.ref(value), value._version, index, name))
        if value._base is not None:
            self.live_versions.append((weakref.ref(value._base), value._base._version, index, name))

    def finish_layer(self, index):
        started = time.perf_counter()
        values = self.layers[index]
        for name, value in values.items():
            if value is None:
                continue
            assert value._version == self.versions.pop((index, name)), (index, name)
            if self.mutation_audit:
                assert torch.equal(value, self.snapshots.pop((index, name))), (index, name)
                self.mutation_checks += 1
        source = {'layers': self.layers, 'arguments': self.arguments}
        for path, destination, representative, size in self.by_layer[index]:
            current = at_path(source, path)
            assert current.shape == destination.shape and current.dtype == destination.dtype
            assert current.device == destination.device
            assert current.stride() == destination.stride()
            assert current.storage_offset() == destination.storage_offset()
            assert current.untyped_storage().nbytes() == size * current.element_size()
            key = storage_key(current)
            if representative not in self.copied:
                flat = destination.detach().as_strided((size,), (1,), 0)
                flat.copy_(current.detach().as_strided((size,), (1,), 0), non_blocking=True)
                self.copied[representative] = key
                self.copy_bytes += size * current.element_size()
            else:
                assert self.copied[representative] == key, (path, 'source alias changed')
            if self.mutation_audit:
                assert torch.equal(destination, current), path
                self.copy_checks += 1
        # Only the graph-owned views survive. In particular, discard the
        # native FA argument dictionary, which otherwise retains q/k/v.
        self.layers[index] = self.program.inputs['layers'][index]
        self.arguments[index] = self.program.inputs['arguments'][index]
        self.storage.clear()
        self.completed_layers += 1
        self.layer_copy_cpu_seconds += time.perf_counter() - started

    def __enter__(self):
        super().__enter__()
        try:
            for index, layer in enumerate(self.model.model.layers):
                def finish(_module, _args, _output, index=index):
                    self.finish_layer(index)
                self.handles.append(layer.register_forward_hook(finish))
            return self
        except BaseException:
            import sys
            self.__exit__(*sys.exc_info())
            raise

    def finish_root(self, source):
        assert self.completed_layers == len(self.layers) == 36
        for reference, version, index, name in self.live_versions:
            value = reference()
            if value is not None:
                assert value._version == version, (index, name, 'late native mutation')
        checks = refresh(source, self.remaining_paths, self.remaining_groups,
                         audit=self.mutation_audit)
        assert len(self.copied) + len(self.remaining_groups) == len(self.program.storage_groups)
        return self.copy_checks + checks

    def clear(self):
        super().clear()
        self.live_versions.clear()


class StreamedFiniteGraphQwen3(FiniteGraphQwen3):
    def attribute(self, before_ids, after_ids, mask, prompt_len, *, mutation_audit=False):
        geometry = (tuple(after_ids.shape), prompt_len)
        if self.program is None or self.signature[:2] != geometry:
            result = super().attribute(before_ids, after_ids, mask, prompt_len,
                                       mutation_audit=mutation_audit)
            result['streamed_root'] = {'enabled': False, 'reason': 'original cold geometry builder'}
            return result
        assert not self.busy and not self.model.training
        self.busy = True
        tape = StreamTape(self.model, self.program, mutation_audit=mutation_audit)
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            start_memory = memory()
            with tape:
                before, after = capture_checkpoint_pair(self.model, before_ids, after_ids, mask, prompt_len)
            root_done = time.perf_counter()
            root_memory = memory()
            assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']
            assert torch.equal(before['target'], after['target'])
            assert torch.equal(before['cos'], after['cos']) and torch.equal(before['sin'], after['sin'])
            assert before['mask'] is None and after['mask'] is None
            signature = (tuple(after_ids.shape), prompt_len, tuple(after['logits'].shape),
                         str(after_ids.device), str(next(self.model.parameters()).dtype))
            assert signature == self.signature
            source = payload(before, after, tape)
            copy_checks = tape.finish_root(source)
            copy_cpu_done = time.perf_counter()
            copies = len(tape.copied) + len(tape.remaining_groups)
            streamed_bytes = tape.copy_bytes
            copy_cpu = tape.layer_copy_cpu_seconds
            mutation_checks = tape.mutation_checks
            tape.clear()
            del source
            refresh_memory = memory()
            result = self.program.replay(before, after, started)
            replay_done = time.perf_counter()
            result['root_retention_mutation_audit'] = {'enabled': mutation_audit, 'predicates': mutation_checks}
            result['graph_input_copy_audit'] = {'enabled': mutation_audit, 'predicates': copy_checks, 'all_passed': True}
            result['native_graph_execution'] = {'graph_replays': 1, 'geometry_cache_entries': 1,
                'static_input_tensors': len(self.program.paths), 'static_input_storage_copies': copies,
                'every_input_tensor_refreshed': True, 'attribution_results_reused': False,
                'build_this_call': None, 'finite_program_warmups_this_call': 0,
                'public_auxiliary_FA_GPU_operations_per_replay': 36,
                'finite_FA_GPU_operations_per_replay': 36, 'public_auxiliary_FA_Python_calls_per_replay': 0,
                'native_model_root_calls': 1, 'native_model_root_endpoint_batch': 2}
            result['streamed_root'] = {'enabled': True, 'completed_layers': 36,
                'root_storage_copy_bytes': streamed_bytes,
                'host_sections_seconds': {'root_including_layer_copies': root_done-started,
                    'endpoint_checks_and_refresh_enqueue': copy_cpu_done-root_done,
                    'cleanup_replay_and_complete_cpu_return': replay_done-copy_cpu_done},
                'layer_copy_host_seconds_included_in_root': copy_cpu,
                'timing_scope': 'Host sections partition the API through full CPU resolution. CUDA copies are asynchronous; these are not GPU kernel durations. Complete external synchronized timing remains authoritative.',
                'memory': {'entry': start_memory, 'after_root': root_memory,
                           'after_refresh': refresh_memory, 'after_resolve': memory()}}
            result['recomputation_scope'] = 'Fresh unchanged B2 root; stream each original storage once at decoder completion; replay the unchanged finite graph and resolve every strict check before full return.'
            return result
        finally:
            tape.clear()
            self.busy = False
