"""Run the same exact DT predicates in one compiled group per layer.

Retain pending operands only until the layer's existing max_abs diagnostic.
Every predicate and maximum is resolved before returning the public result.
No sampling, tolerance relaxation, removed checks, or custom reduction kernel.
"""
import torch


def _group(equal_pairs,finite_values,positive_values,maximum_values):
    predicates=[(a==b).all() for a,b in equal_pairs]
    predicates += [torch.isfinite(a).all() for a in finite_values]
    predicates += [(a>0).all() for a in positive_values]
    maxima=[a.abs().max() for a in maximum_values]
    return torch.stack(predicates),torch.stack(maxima)


compiled_group=torch.compile(_group,fullgraph=True,dynamic=True,
  options={'triton.cudagraphs':False,'max_autotune':False})


class _Scalar:
    __slots__=('value',)
    def __init__(self):self.value=None


class BatchedValidation:
    def __init__(self):
        self.pending=[];self.flag_batches=[];self.stat_batches=[];self.flags=[];self.stats=[]
        self.labels=[];self.flushes=0

    def equal(self,a,b,label):
        if a.shape!=b.shape or a.device!=b.device:raise ValueError('Validation shape/device mismatch: '+label)
        out=_Scalar();self.pending.append(('equal',(a,b),label,out));return out

    def finite(self,a,label):
        out=_Scalar();self.pending.append(('finite',a,label,out));return out

    def positive(self,a,label):
        out=_Scalar();self.pending.append(('positive',a,label,out));return out

    def max_abs(self,a):
        out=_Scalar();self.pending.append(('maximum',a,None,out));self.flush();return out

    def flush(self):
        if not self.pending:return
        groups=[[x for x in self.pending if x[0]==kind] for kind in ['equal','finite','positive','maximum']]
        assert groups[3], 'Each finite layer supplies its unchanged maximum diagnostic.'
        predicates,statistics=compiled_group(*(tuple(x[1] for x in group) for group in groups))
        self.flag_batches.append(predicates);self.stat_batches.append(statistics)
        for group in groups[:3]:
            self.flags.extend(x[3] for x in group);self.labels.extend(x[2] for x in group)
        self.stats.extend(x[3] for x in groups[3]);self.pending.clear();self.flushes+=1

    def finish(self,result):
        self.flush()
        flags=torch.cat(self.flag_batches).cpu().tolist();stats=torch.cat(self.stat_batches).cpu().tolist()
        assert len(flags)==len(self.flags)==len(self.labels) and len(stats)==len(self.stats)
        failures=[label for label,value in zip(self.labels,flags) if not value]
        if failures:raise ValueError('Deferred DT validation failed before return: '+', '.join(failures))
        for scalar,value in zip(self.flags,flags):scalar.value=bool(value)
        for scalar,value in zip(self.stats,stats):scalar.value=float(value)
        def walk(x):
            kind=type(x)
            if kind in (bool,int,float,str,type(None)):return x
            if kind is dict:return {k:walk(v) for k,v in x.items()}
            if kind is list:return [walk(v) for v in x]
            if kind is tuple:return tuple(walk(v) for v in x)
            if isinstance(x,_Scalar):
                assert x.value is not None;return x.value
            if isinstance(x,torch.Tensor):raise ValueError('Unregistered tensor in public diagnostic record')
            if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
            if isinstance(x,list):return [walk(v) for v in x]
            if isinstance(x,tuple):return tuple(walk(v) for v in x)
            return x
        result=walk(result)
        result['deferred_validation']={'predicates':len(flags),'all_passed':True,'statistics':len(stats),
          'synchronization':'before API return','predicate_rule':'unchanged exact checks','compiled_groups':self.flushes}
        self.flag_batches.clear();self.stat_batches.clear();self.flags.clear();self.stats.clear();self.labels.clear()
        return result


def verify_rejection_and_statistics(device='cuda'):
    """Exercise genuine failures and maxima against eager Torch on a small input."""
    records=[]
    for bad_kind in [None,'equal','finite','positive']:
        a=torch.tensor([1.,2.,3.],device=device);b=a.clone()
        finite=a.clone();positive=a.clone()
        if bad_kind=='equal':b[1]=7
        if bad_kind=='finite':finite[1]=float('inf')
        if bad_kind=='positive':positive[1]=0
        checks=BatchedValidation();e=checks.equal(a,b,'equality');f=checks.finite(finite,'finite');p=checks.positive(positive,'positive')
        maximum=checks.max_abs(a)
        expected={'equality':bool(torch.equal(a,b)),'finite':bool(torch.isfinite(finite).all()),'positive':bool((positive>0).all()),'maximum':float(a.abs().max())}
        try:
            actual=checks.finish({'equality':e,'finite':f,'positive':p,'maximum':maximum})
            assert bad_kind is None and all(actual[k]==v for k,v in expected.items())
            records.append({'case':'valid','all_fields_match_eager':True})
        except ValueError as exc:
            assert bad_kind is not None and ('equality' if bad_kind=='equal' else bad_kind) in str(exc)
            records.append({'case':bad_kind,'rejected_before_return':True})
    return records
