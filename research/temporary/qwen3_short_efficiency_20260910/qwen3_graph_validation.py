"""Keep original strict predicate kernels in a native GPU graph, resolve on return."""
import torch
from batched_validation_rope import BatchedValidation,_Scalar


class GraphValidation(BatchedValidation):
    def finish(self,result):
        self.flush()
        return PackedResult(result,torch.cat(self.flag_batches),torch.cat(self.stat_batches),
            tuple(self.flags),tuple(self.stats),tuple(self.labels),self.flushes)


class PackedResult:
    def __init__(self,result,flags,statistics,flag_handles,stat_handles,labels,groups):
        self.result=result;self.flags=flags;self.statistics=statistics
        self.flag_handles=flag_handles;self.stat_handles=stat_handles;self.labels=labels;self.groups=groups

    def resolve(self,before,after,seconds):
        flags=self.flags.cpu().tolist();statistics=self.statistics.cpu().tolist()
        assert len(flags)==len(self.flag_handles)==len(self.labels) and len(statistics)==len(self.stat_handles)
        failures=[label for label,flag in zip(self.labels,flags) if not flag]
        if failures:raise ValueError('Graph DT validation failed before return: '+', '.join(failures))
        for scalar,value in zip(self.flag_handles,flags):scalar.value=bool(value)
        for scalar,value in zip(self.stat_handles,statistics):scalar.value=float(value)
        result=dict(self.result);signed=result['signed_full_sequence']
        assert bool(torch.isfinite(signed).all()), 'Nonfinite signed attribution'
        total=float(result['signed_sum']);delta=after['score32_sum64']-before['score32_sum64']
        result.update(signed_full_sequence=signed.cpu().tolist(),signed_sum=total,
            target_delta_score32_sum64=delta,target_delta_score16=after['score16']-before['score16'],
            unassigned_total=delta-total,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        def walk(x):
            kind=type(x)
            if kind in (bool,int,float,str,type(None)):return x
            if kind is dict:return {k:walk(v) for k,v in x.items()}
            if kind is list:return [walk(v) for v in x]
            if kind is tuple:return tuple(walk(v) for v in x)
            if isinstance(x,_Scalar):
                assert x.value is not None;return x.value
            if isinstance(x,torch.Tensor):raise ValueError('Unregistered graph tensor in public result')
            return x
        result=walk(result)
        result['deferred_validation']={'predicates':len(flags),'all_passed':True,'statistics':len(statistics),
            'compiled_groups':self.groups,'synchronization':'before API return','predicate_rule':'unchanged exact checks'}
        return result
