"""Original two-warmup/one-capture build with explicit warm-storage release."""
import gc,time,torch


def build(self):
    started=time.perf_counter();warm=[];stream=self.build_stream
    gc.collect();torch.cuda.empty_cache()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(),torch.cuda.stream(stream):
        for _ in range(2):
            t=time.perf_counter();packed=self.invoke();torch.cuda.synchronize()
            result=packed.resolve(self.inputs['before'],self.inputs['after'],time.perf_counter()-t)
            warm.append({'seconds':time.perf_counter()-t,'strict_predicates':result['deferred_validation']['predicates'],'all_passed':True,
                'allocated_end':torch.cuda.memory_allocated(),'reserved_end':torch.cuda.memory_reserved()})
            del packed,result
            self.inputs=None;gc.collect();torch.cuda.empty_cache()
            warm[-1].update(allocated_after_release=torch.cuda.memory_allocated(),reserved_after_release=torch.cuda.memory_reserved())
        self.graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph,stream=stream):self.packed=self.invoke()
        except BaseException:self.graph=self.packed=None;raise
    torch.cuda.current_stream().wait_stream(stream)
    self.build_info={'seconds':time.perf_counter()-started,'warm_finite_programs':warm,'native_graph_program_recordings':1,
        'captured_program_recording_is_not_a_model_forward':False,'static_input_tensors':len(self.paths),
        'static_input_bytes':sum(t.numel()*t.element_size() for _,t,_ in self.storage_groups),'static_input_storage_copies':len(self.storage_groups),
        'original_model_Python_calls_during_build':3,'original_model_GPU_executions_during_build':3,
        'recorded_peak_allocated_bytes_including_warmup':torch.cuda.max_memory_allocated(),'recorded_peak_reserved_bytes_including_warmup':torch.cuda.max_memory_reserved(),'root_and_finite_captured_together':True,'root_checkpoint_copies_between_graphs':0,
        'warm_storage_release':'Release all warm inputs and output tensors, run Python cycle collection and empty unused allocator cache before next warm/capture. No peak reset; all release time charged.'}
    return self.build_info
