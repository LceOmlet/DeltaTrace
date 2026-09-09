"""Collect exact device predicates, synchronize and fail before API return.

Only validation scheduling changes. No Torch/model/FA function is replaced.
Predicates remain strict; this does not weaken checks to tolerance thresholds.
"""
import torch


def _equal(a,b):return (a==b).all()
def _finite(a):return torch.isfinite(a).all()
def _positive(a):return (a>0).all()
def _maximum(a):return a.abs().max()

def _compile(fn):
    return torch.compile(fn,fullgraph=True,dynamic=True,
                         options={'triton.cudagraphs':False,'max_autotune':False})

equal_device=_compile(_equal)
finite_device=_compile(_finite)
positive_device=_compile(_positive)
maximum_device=_compile(_maximum)


class DeferredValidation:
    def __init__(self):
        self.predicates=[];self.labels=[];self.statistics=[]

    def add(self,predicate,label):
        assert predicate.ndim==0 and predicate.dtype==torch.bool
        self.predicates.append(predicate);self.labels.append(label)
        return predicate

    def equal(self,a,b,label):
        if a.shape!=b.shape or a.device!=b.device:
            raise ValueError('Validation shape/device mismatch: '+label)
        return self.add(equal_device(a,b),label)

    def finite(self,a,label):return self.add(finite_device(a),label)
    def positive(self,a,label):return self.add(positive_device(a),label)

    def max_abs(self,a):
        value=maximum_device(a);self.statistics.append(value);return value

    def finish(self,result):
        flags=torch.stack(self.predicates).cpu().tolist()
        failures=[label for label,flag in zip(self.labels,flags) if not flag]
        if failures:raise ValueError('Deferred DT validation failed before return: '+', '.join(failures))
        stats=torch.stack(self.statistics).cpu().tolist() if self.statistics else []
        resolved={id(t):bool(v) for t,v in zip(self.predicates,flags)}
        resolved.update({id(t):float(v) for t,v in zip(self.statistics,stats)})
        def walk(x):
            if isinstance(x,torch.Tensor):
                if id(x) not in resolved:raise ValueError('Unregistered tensor in public diagnostic record')
                return resolved[id(x)]
            if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
            if isinstance(x,list):return [walk(v) for v in x]
            if isinstance(x,tuple):return tuple(walk(v) for v in x)
            return x
        result=walk(result)
        result['deferred_validation']={'predicates':len(flags),'all_passed':True,'statistics':len(stats),
                                       'synchronization':'before API return','predicate_rule':'unchanged exact checks'}
        return result
