"""Materialize fresh public metadata in C; every attribution value stays live."""
import marshal
import torch
from batched_validation_rope import _Scalar
from qwen3_graph_validation import GraphValidation


class TemplateGraphValidation(GraphValidation):
    def finish(self,result):
        self.flush()
        return TemplatePackedResult(result,torch.cat(self.flag_batches),torch.cat(self.stat_batches),
            tuple(self.flags),tuple(self.stats),tuple(self.labels),self.flushes)


class TemplatePackedResult:
    def __init__(self,result,flags,statistics,flag_handles,stat_handles,labels,groups):
        self.result=result;self.flags=flags;self.statistics=statistics
        self.flag_handles=flag_handles;self.stat_handles=stat_handles;self.labels=labels;self.groups=groups
        handles={id(s):('flag',i) for i,s in enumerate(flag_handles)}
        handles.update({id(s):('stat',i) for i,s in enumerate(stat_handles)})
        self.dynamic_paths=[]
        def template(value,path):
            kind=type(value)
            if kind in (bool,int,float,str,type(None)):return value
            if kind is dict:return {k:template(v,path+(k,)) for k,v in value.items()}
            if kind is list:return [template(v,path+(i,)) for i,v in enumerate(value)]
            if kind is tuple:return tuple(template(v,path+(i,)) for i,v in enumerate(value))
            if isinstance(value,_Scalar):
                assert id(value) in handles
                self.dynamic_paths.append((path,*handles[id(value)]));return None
            raise TypeError('Unsupported static public metadata: '+str(kind))
        live={'signed_full_sequence','signed_sum','target_delta_score32_sum64','target_delta_score16','unassigned_total','seconds','peak_allocated_bytes'}
        static={k:v for k,v in result.items() if k not in live}
        # This is private, locally generated data containing only primitive
        # metadata. No externally supplied serialization or prior output is read.
        # Construct a tree before serializing so mutable container aliases are
        # expanded just as in the original recursive public-return conversion.
        self.template_bytes=marshal.dumps(template(static,()))

    @staticmethod
    def put(container,path,value):
        key=path[0];child=value if len(path)==1 else TemplatePackedResult.put(container[key],path[1:],value)
        if type(container) is tuple:
            result=list(container);result[key]=child;return tuple(result)
        container[key]=child;return container

    def resolve(self,before,after,seconds):
        flags=self.flags.cpu().tolist();statistics=self.statistics.cpu().tolist()
        assert len(flags)==len(self.flag_handles)==len(self.labels) and len(statistics)==len(self.stat_handles)
        failures=[label for label,flag in zip(self.labels,flags) if not flag]
        if failures:raise ValueError('Graph DT validation failed before return: '+', '.join(failures))
        for scalar,value in zip(self.flag_handles,flags):scalar.value=bool(value)
        for scalar,value in zip(self.stat_handles,statistics):scalar.value=float(value)
        signed=self.result['signed_full_sequence']
        assert bool(torch.isfinite(signed).all()), 'Nonfinite signed attribution'
        total=float(self.result['signed_sum']);delta=after['score32_sum64']-before['score32_sum64']
        result=marshal.loads(self.template_bytes)
        for path,kind,index in self.dynamic_paths:
            self.put(result,path,bool(flags[index]) if kind=='flag' else float(statistics[index]))
        result.update(signed_full_sequence=signed.cpu().tolist(),signed_sum=total,
            target_delta_score32_sum64=delta,target_delta_score16=after['score16']-before['score16'],
            unassigned_total=delta-total,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        result['deferred_validation']={'predicates':len(flags),'all_passed':True,'statistics':len(statistics),
            'compiled_groups':self.groups,'synchronization':'before API return','predicate_rule':'unchanged exact checks'}
        result['output_materialization']={'static_metadata_template_bytes':len(self.template_bytes),
            'dynamic_scalar_paths':len(self.dynamic_paths),'fresh_mutable_containers_each_return':True,
            'attribution_values_reused':False,'source':'Private primitive metadata prepared during charged build; current GPU predicates, statistics, scores and complete signed vector resolved on every call.'}
        return result
