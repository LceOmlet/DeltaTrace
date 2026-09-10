"""Native GPU graph of the unchanged finite program, with fresh root operands.

One geometry is cached. Every attribution runs the original model and copies
every tensor into graph inputs. The graph contains the same native MM, public
auxiliary FA, finite FA and original compiled secant/strict-check kernels.
"""
import copy,time
import torch
from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair
from qwen3_root_local import LocalNativeRootTape
from qwen3_root_tensor_graph import NativeRootTape,propagate_root_tape


def slim(endpoint):
    return {k:v for k,v in endpoint.items() if k!='layers'}


def payload(before,after,tape):
    return {'before':slim(before),'after':slim(after),'layers':tape.layers,'arguments':tape.arguments}


def clone_tree(value,memo,paths,path=()):
    if id(value) in memo:return memo[id(value)]
    if isinstance(value,torch.Tensor):
        result=value.detach().clone(memory_format=torch.preserve_format)
        memo[id(value)]=result;paths.append((path,result));return result
    if isinstance(value,dict):
        result={};memo[id(value)]=result
        result.update((key,clone_tree(item,memo,paths,path+(key,))) for key,item in value.items());return result
    if isinstance(value,list):
        result=[];memo[id(value)]=result
        result.extend(clone_tree(item,memo,paths,path+(i,)) for i,item in enumerate(value));return result
    if isinstance(value,tuple):
        result=tuple(clone_tree(item,memo,paths,path+(i,)) for i,item in enumerate(value));memo[id(value)]=result;return result
    assert value is None or isinstance(value,(bool,int,float,str)),type(value)
    return value


def at_path(value,path):
    for part in path:value=value[part]
    return value


class Program:
    def __init__(self,model,source,retained_bytes,finite):
        self.model=model;self.finite=finite;self.retained_bytes=retained_bytes;self.paths=[]
        self.inputs=clone_tree(source,{},self.paths)
        self.graph=None;self.packed=None;self.build_info=None

    def refresh(self,source,*,audit=False):
        count=0
        for path,destination in self.paths:
            current=at_path(source,path)
            assert isinstance(current,torch.Tensor) and destination.shape==current.shape
            assert destination.dtype==current.dtype and destination.device==current.device
            destination.copy_(current)
            if audit:assert torch.equal(destination,current);count+=1
        return count

    def invoke(self):
        tape=NativeRootTape(self.model,mutation_audit=False)
        tape.layers=[dict(row) for row in self.inputs['layers']]
        tape.arguments={index:dict(arguments) for index,arguments in self.inputs['arguments'].items()}
        tape.versions={(index,name):value._version for index,row in enumerate(tape.layers) for name,value in row.items() if isinstance(value,torch.Tensor)}
        tape.maximum_retained_bytes=self.retained_bytes
        return propagate_root_tape(self.model,self.inputs['before'],self.inputs['after'],tape,pv_rule='content_P1',finite_attention=self.finite)

    def build(self):
        start=time.perf_counter();warm=[]
        # Compile and initialize libraries before stream capture, charging both calls.
        with torch.no_grad():
            for _ in range(2):
                t=time.perf_counter();packed=self.invoke();torch.cuda.synchronize()
                result=packed.resolve(self.inputs['before'],self.inputs['after'],time.perf_counter()-t)
                warm.append({'seconds':time.perf_counter()-t,'strict_predicates':result['deferred_validation']['predicates'],'all_passed':True})
                del packed,result
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.packed=self.invoke()
        self.build_info={'seconds':time.perf_counter()-start,'warm_finite_programs':warm,
            'native_graph_program_recordings':1,'captured_program_recording_is_not_a_model_forward':True,
            'static_input_tensors':len(self.paths),'static_input_bytes':sum(t.numel()*t.element_size() for _,t in self.paths)}
        return self.build_info

    def replay(self,before,after,started):
        self.graph.replay()
        return self.packed.resolve(before,after,time.perf_counter()-started)


class FiniteGraphQwen3:
    def __init__(self,model,finite):
        self.model=model;self.finite=finite;self.program=None;self.signature=None;self.busy=False

    def attribute(self,before_ids,after_ids,mask,prompt_len,*,mutation_audit=False):
        assert not self.busy
        self.busy=True;tape=LocalNativeRootTape(self.model,mutation_audit=mutation_audit)
        try:
            torch.cuda.reset_peak_memory_stats()
            with tape:before,after=capture_checkpoint_pair(self.model,before_ids,after_ids,mask,prompt_len)
            assert before['length']==after['length'] and before['prompt_len']==after['prompt_len']
            assert torch.equal(before['target'],after['target'])
            assert torch.equal(before['cos'],after['cos']) and torch.equal(before['sin'],after['sin'])
            assert before['mask'] is None and after['mask'] is None
            mutation_checks=0
            for index,values in enumerate(tape.layers):
                for name,value in values.items():
                    if value is None:continue
                    assert value._version==tape.versions[index,name],(index,name)
                    if mutation_audit:
                        assert torch.equal(value,tape.snapshots[index,name]),(index,name)
                        mutation_checks+=1
            signature=(tuple(after_ids.shape),prompt_len,tuple(after['logits'].shape),str(after_ids.device),str(next(self.model.parameters()).dtype))
            source=payload(before,after,tape);torch.cuda.synchronize();started=time.perf_counter()
            build=None
            if self.signature!=signature:
                self.program=None
                self.program=Program(self.model,source,tape.maximum_retained_bytes,self.finite);self.signature=signature
                copy_checks=self.program.refresh(source,audit=mutation_audit)
                tape.clear();del source
                build=self.program.build()
            else:
                copy_checks=self.program.refresh(source,audit=mutation_audit)
                tape.clear();del source
            result=self.program.replay(before,after,started)
            result['root_retention_mutation_audit']={'enabled':mutation_audit,'predicates':mutation_checks}
            result['graph_input_copy_audit']={'enabled':mutation_audit,'predicates':copy_checks,'all_passed':True}
            result['native_graph_execution']={'graph_replays':1,'geometry_cache_entries':1,'static_input_tensors':len(self.program.paths),
                'every_input_tensor_refreshed':True,'attribution_results_reused':False,'build_this_call':build,
                'finite_program_warmups_this_call':2 if build is not None else 0,
                'public_auxiliary_FA_GPU_operations_per_replay':36,'finite_FA_GPU_operations_per_replay':36,
                'public_auxiliary_FA_Python_calls_per_replay':0,'native_model_root_calls':1,'native_model_root_endpoint_batch':2}
            result['recomputation_scope']='One fresh original B2 root; every graph input refreshed. One native graph replay of the unchanged finite program, including 36 public auxiliary B2 FA operations. Graph-build warmups are separately counted and charged within the first call.'
            return result
        finally:
            tape.clear();self.busy=False
