"""Focused CPU storage and real-root late-mutation/cleanup controls."""
import gc,time
from types import SimpleNamespace
import torch


def storage_probes(model):
    from qwen3_streamed_graph import StreamTape
    records=[]
    for fault in [None,'alias','stride','version']:
        source=torch.arange(16,dtype=torch.float32).reshape(4,4)
        destination=torch.empty(16,dtype=torch.float32)
        left=destination.view(4,4);right=destination.view(4,4)
        representative=('layers',0,'x')
        layers=[{'x':left,'out':right}]+[{} for _ in range(35)]
        program=SimpleNamespace(inputs={'layers':layers,'arguments':{0:{}}},
            paths=[(('layers',0,'x'),left,representative,16),
                   (('layers',0,'out'),right,representative,16)],
            storage_groups=[(representative,destination,16)])
        tape=StreamTape(model,program,mutation_audit=True)
        tape.retain(0,'x',source)
        other=source.clone() if fault=='alias' else source.t() if fault=='stride' else source.view(4,4)
        tape.retain(0,'out',other);tape.arguments[0]={}
        if fault=='version':source.add_(0)
        start=time.perf_counter();rejected=False
        try:
            tape.finish_layer(0)
            if fault is None:
                assert torch.equal(left,source) and torch.equal(right,source)
                assert len(tape.copied)==1 and tape.copy_bytes==64
                assert tape.copy_checks==2 and tape.mutation_checks==2
        except AssertionError:
            if fault is None:raise
            rejected=True
        finally:tape.clear()
        assert fault is None or rejected, (fault, 'fault escaped rejection')
        records.append({'case':fault or 'valid_shared_storage','passed':fault is None or rejected,
                        'rejected':rejected,'CPU_only':True,'seconds':time.perf_counter()-start})
    return records


def late_mutation_probes(model,runner,base,ids,mask,prompt_len):
    """Increment a real root output's version after it was streamed.

    add_(0) keeps values identical: only the late mutation guard can reject.
    Every fault call and successful recovery runs a complete fresh root.
    """
    counts=lambda:[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
    baseline_counts=counts();records=[]
    for cold,field in [(False,'out'),(True,'out'),(False,'a'),(True,'a')]:
        if cold:runner.close()
        saved={};seen=[]
        def save_output(_module,_args,output):saved['output']=output
        def mutate_late(_module,_args):
            saved['output'].add_(0);seen.append(True)
        source_module=model.model.layers[0] if field=='out' else model.model.layers[0].input_layernorm
        h0=source_module.register_forward_hook(save_output)
        h1=model.model.layers[1].register_forward_pre_hook(mutate_late)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        rejected=False;error=None
        try:
            runner.attribute(base,ids,mask,prompt_len,mutation_audit=False)
        except AssertionError as exc:
            error=str(exc);assert 'late native mutation' in error,error;rejected=True
        finally:
            h0.remove();h1.remove();saved.clear()
        torch.cuda.synchronize()
        records.append({'case':('cold' if cold else 'hot')+'_late_version_'+field,
            'rejected_before_return':rejected,'error':error,'fault_injections':len(seen),
            'seconds':time.perf_counter()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        assert rejected and seen==[True] and not runner.busy and counts()==baseline_counts
        if cold:assert runner.program is None and runner.signature is None
        gc.collect();torch.cuda.synchronize();t=time.perf_counter()
        recovered=runner.attribute(base,ids,mask,prompt_len,mutation_audit=True)
        torch.cuda.synchronize()
        records[-1]['recovery_seconds']=time.perf_counter()-t
        records[-1]['recovery_vector']=recovered['signed_full_sequence']
        records[-1]['recovery_math']={k:recovered[k] for k in ['target_delta_score32_sum64','target_delta_score16','signed_sum','unassigned_total','layer_checks']}
        assert counts()==baseline_counts
    return records
