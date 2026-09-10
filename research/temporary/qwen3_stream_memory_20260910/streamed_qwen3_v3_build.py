"""Stream first-call native operands too; adopt their storage into the graph."""
import time
import weakref
import torch
from streamed_qwen3_v3_base import StreamTape, StreamedFiniteGraphQwen3, memory
from qwen3_finite_graph import Program, payload
from qwen3_root_retained import NativeRootTape
from qwen3_graph_storage_inputs import clone_tree, at_path, storage_key
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair


class BuildingStreamTape(StreamTape):
    def __init__(self, model, *, mutation_audit=False):
        NativeRootTape.__init__(self, model, mutation_audit=mutation_audit)
        self.live_versions = []
        self.output_refs = {}
        self.previous_output = None
        self.flat_buffers = []
        self.copy_checks = self.mutation_checks = self.copy_bytes = 0
        self.layer_copy_cpu_seconds = 0.0
        self.completed_layers = 0

    def retain(self, index, name, value):
        super().retain(index, name, value)
        if name == 'out':
            self.output_refs[index] = weakref.ref(value)

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
        memo = {'storage_groups': {}}
        shared_key = None
        if self.previous_output is not None:
            reference, original_key, flat, size = self.previous_output
            # The preceding original decoder output must still be the actual
            # next decoder input. This also disambiguates allocator address reuse.
            assert reference() is not None
            assert storage_key(values['x']) == original_key
            shared_key = original_key
            memo['storage_groups'][shared_key] = (('layers', index-1, 'out'), flat, size)
        paths = []
        row = clone_tree(values, memo, paths, ('layers', index))
        arguments = clone_tree(self.arguments[index], memo, paths, ('arguments', index))
        source = {'layers': self.layers, 'arguments': self.arguments}
        if self.mutation_audit:
            for path, destination, representative, size in paths:
                assert torch.equal(destination, at_path(source, path)), path
                self.copy_checks += 1
        for key, group in memo['storage_groups'].items():
            if key != shared_key:
                self.flat_buffers.append(group)
                self.copy_bytes += group[2] * group[1].element_size()
        original_key = storage_key(values['out'])
        _, flat, size = memo['storage_groups'][original_key]
        self.previous_output = (self.output_refs.pop(index), original_key, flat, size)
        self.layers[index] = row
        self.arguments[index] = arguments
        self.storage.clear()
        self.maximum_retained_bytes = self.copy_bytes
        self.completed_layers += 1
        self.layer_copy_cpu_seconds += time.perf_counter()-started

    def validate_end(self):
        assert self.completed_layers == 36 and not self.versions and not self.snapshots
        for reference, version, index, name in self.live_versions:
            value = reference()
            if value is not None:
                assert value._version == version, (index, name, 'late native mutation')

    def clear(self):
        super().clear()
        self.output_refs.clear()
        self.previous_output = None
        self.flat_buffers.clear()


class AdoptedProgram(Program):
    def __init__(self, model, source, retained_bytes, finite, flat_buffers):
        self.model = model
        self.finite = finite
        self.retained_bytes = retained_bytes
        self.paths = []
        # Pre-existing graph-owned native operand storage is adopted unchanged.
        # Endpoint/checkpoint tensors still use the original clone_tree behavior.
        memo = {'storage_groups': {storage_key(flat): (path, flat, size)
                for path, flat, size in flat_buffers}}
        assert len(memo['storage_groups']) == len(flat_buffers)
        self.inputs = clone_tree(source, memo, self.paths)
        self.storage_groups = list(memo['storage_groups'].values())
        self.graph = self.packed = self.build_info = None

    def validate_adoption(self, source, *, audit=False):
        fresh = {path: storage_key(at_path(source, path))
                 for path, destination, size in self.storage_groups}
        count = 0
        for path, destination, representative, size in self.paths:
            current = at_path(source, path)
            assert current.shape == destination.shape and current.dtype == destination.dtype
            assert current.device == destination.device and current.stride() == destination.stride()
            assert current.storage_offset() == destination.storage_offset()
            assert current.untyped_storage().nbytes() == size * current.element_size()
            assert storage_key(current) == fresh[representative]
            if audit:
                assert torch.equal(destination, current), path
                count += 1
        return count


class StreamedBuildFiniteGraphQwen3(StreamedFiniteGraphQwen3):
    cold_tape_type=BuildingStreamTape

    def initialize_storage_metadata(self,tape):
        pass

    def attribute(self, before_ids, after_ids, mask, prompt_len, *, mutation_audit=False):
        geometry = (tuple(after_ids.shape), prompt_len)
        if self.program is not None and self.signature[:2] == geometry:
            return super().attribute(before_ids, after_ids, mask, prompt_len,
                                     mutation_audit=mutation_audit)
        assert not self.busy and not self.model.training
        self.busy = True
        tape = self.cold_tape_type(self.model, mutation_audit=mutation_audit)
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            start_memory = memory()
            # Release the previous geometry before executing the new root.
            self.program = None
            self.signature = None
            with tape:
                before, after = capture_checkpoint_pair(self.model, before_ids, after_ids, mask, prompt_len)
            root_done = time.perf_counter()
            root_memory = memory()
            tape.validate_end()
            assert before['length'] == after['length'] and before['prompt_len'] == after['prompt_len']
            assert torch.equal(before['target'], after['target'])
            assert torch.equal(before['cos'], after['cos']) and torch.equal(before['sin'], after['sin'])
            assert before['mask'] is None and after['mask'] is None
            signature = (tuple(after_ids.shape), prompt_len, tuple(after['logits'].shape),
                         str(after_ids.device), str(next(self.model.parameters()).dtype))
            source = payload(before, after, tape)
            self.program = AdoptedProgram(self.model, source, tape.maximum_retained_bytes,
                                          self.finite, tape.flat_buffers)
            self.initialize_storage_metadata(tape)
            copy_checks = self.program.validate_adoption(source, audit=mutation_audit)
            copy_bytes = tape.copy_bytes
            copy_cpu = tape.layer_copy_cpu_seconds
            mutation_checks = tape.mutation_checks
            extra_clone_checks = tape.copy_checks
            tape.clear()
            del source
            adopted = time.perf_counter()
            adopted_memory = memory()
            build = self.program.build()
            self.signature = signature
            built = time.perf_counter()
            built_memory = memory()
            result = self.program.replay(before, after, started)
            resolved = time.perf_counter()
            result['root_retention_mutation_audit'] = {'enabled': mutation_audit, 'predicates': mutation_checks}
            result['graph_input_copy_audit'] = {'enabled': mutation_audit, 'predicates': copy_checks, 'all_passed': True}
            result['native_graph_execution'] = {'graph_replays': 1, 'geometry_cache_entries': 1,
                'static_input_tensors': len(self.program.paths),
                'static_input_storage_copies': len(self.program.storage_groups),
                'every_input_tensor_refreshed': True, 'attribution_results_reused': False,
                'build_this_call': build, 'finite_program_warmups_this_call': 2,
                'public_auxiliary_FA_GPU_operations_per_replay': 36,
                'finite_FA_GPU_operations_per_replay': 36, 'public_auxiliary_FA_Python_calls_per_replay': 0,
                'native_model_root_calls': 1, 'native_model_root_endpoint_batch': 2}
            result['streamed_root'] = {'enabled': True, 'completed_layers': 36,
                'cold_storage_adoption': True, 'root_storage_copy_bytes': copy_bytes,
                'extra_root_clone_audit_checks': extra_clone_checks,
                'host_sections_seconds': {'root_including_layer_copies': root_done-started,
                    'checks_adoption_and_cleanup': adopted-root_done,
                    'finite_program_warmup_and_graph_capture': built-adopted,
                    'replay_and_complete_cpu_return': resolved-built},
                'layer_copy_host_seconds_included_in_root': copy_cpu,
                'timing_scope': 'All graph warmups, capture and replay are inside the complete call. Host sections are not GPU kernel durations.',
                'memory': {'entry': start_memory, 'after_root': root_memory,
                           'after_adoption': adopted_memory, 'after_build': built_memory,
                           'after_resolve': memory()}}
            result['recomputation_scope'] = 'Fresh unchanged B2 root; clone each native storage once as decoders finish and adopt into graph inputs; original finite-program warmups/capture/replay fully charged.'
            return result
        except BaseException:
            self.program = None
            self.signature = None
            raise
        finally:
            tape.clear()
            self.busy = False
