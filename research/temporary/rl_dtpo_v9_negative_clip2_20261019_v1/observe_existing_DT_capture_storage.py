"""Read existing replay-capture storage metadata during one native readout.

The original capture __exit__ methods, runner calls and return objects are
retained. No tensor contents are read, copied or moved. Root checkpoints are
outside this observation. Every temporary class method restores at readout
exit; no resource option, numerical operator, CUDA event or RNG is changed.
"""
import ast
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_existing_DT_metadata.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '20c0de1e0d2b7f89082e372fbdac292cab0bea41866612f8f0555d3afe94649c'
tree = ast.parse(owner.read_text())
observation = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'observation' for t in n.targets))
for old, new in (
    ('_existing_DT_timing_scope_installed', '_existing_DT_capture_storage_scope_installed'),
    ('native_DT_existing_timing', 'native_DT_capture_storage'),
    ('existing_DT_timing_observer_', 'existing_DT_capture_storage_observer_'),
):
    assert old in observation
    observation = observation.replace(old, new)
anchor = '        native_attribute=runner.attribute'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
        import torch,inspect,hashlib
        from pathlib import Path
        backend=runner.capture_backend
        assert backend is not None
        capture_classes=[(name,getattr(backend,name)) for name in (
            'NativeDecoderCapture','NativeDenseAttentionCapture','NativeGDNCapture')]
        capture_sources={name:dict(path=inspect.getfile(cls),SHA256=hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest())
                         for name,cls in capture_classes}''')
anchor = '            count=[0]'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
            active_storage=[None]
            saved_exits=[]
            def observe_capture_exit(original_exit,kind):
                def exit_with_storage(capture,*exit_args):
                    returned_exit=original_exit(capture,*exit_args)
                    current=active_storage[0]
                    if current is not None and exit_args[0] is None:
                        try:
                            seen={};cpu_bytes=0;gpu_bytes=0;tensors=[]
                            for group in ('values','endpoints'):
                                for name,value in getattr(capture,group,{}).items():
                                    if not isinstance(value,torch.Tensor):continue
                                    storage=value.untyped_storage()
                                    key=(str(value.device),storage.data_ptr(),storage.nbytes())
                                    new=key not in seen
                                    if new:
                                        seen[key]=len(seen)
                                        if value.device.type=='cpu':cpu_bytes+=storage.nbytes()
                                        elif value.is_cuda:gpu_bytes+=storage.nbytes()
                                    tensors.append(dict(group=group,name=name,shape=list(value.shape),
                                        stride=list(value.stride()),dtype=str(value.dtype),device=str(value.device),
                                        element_bytes=value.element_size(),numel=value.numel(),
                                        storage_bytes=storage.nbytes(),storage_group=seen[key],
                                        counted_as_unique_storage=new))
                            current['capture_exit_counts'][kind]=current['capture_exit_counts'].get(kind,0)+1
                            prior=current['maximum_CPU_bytes_by_capture'].get(kind,-1)
                            if cpu_bytes>prior:
                                current['maximum_CPU_bytes_by_capture'][kind]=cpu_bytes
                                current['maximum_CPU_capture_details'][kind]=dict(CPU_bytes=cpu_bytes,GPU_bytes=gpu_bytes,tensors=tensors)
                        except Exception as error:
                            current['metadata_errors'].append(type(error).__name__+': '+str(error))
                    return returned_exit
                return exit_with_storage
            for kind,cls in capture_classes:
                native_exit=cls.__exit__
                saved_exits.append((cls,'__exit__' in vars(cls),native_exit))
                cls.__exit__=observe_capture_exit(native_exit,kind)
            scheduler_epoch=actor.actor_lr_scheduler.last_epoch''')
anchor = '                began=time.time()'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                storage_report=dict(capture_exit_counts={},maximum_CPU_bytes_by_capture={},
                    maximum_CPU_capture_details={},metadata_errors=[])
                active_storage[0]=storage_report''')
anchor = '                ended=time.time()'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                active_storage[0]=None
                maxima=storage_report['maximum_CPU_bytes_by_capture']
                storage_report['conservative_extra_replay_storage_bytes']=(maxima.get('NativeDecoderCapture',0)
                    +max(maxima.get('NativeDenseAttentionCapture',0),maxima.get('NativeGDNCapture',0)))
                storage_report['complete_layer_capture_counts']=(storage_report['capture_exit_counts'].get('NativeDecoderCapture',0)==len(returned[1]['layers'])
                    and sum(storage_report['capture_exit_counts'].get(k,0) for k in ('NativeDenseAttentionCapture','NativeGDNCapture'))==len(returned[1]['layers']))''')
anchor = '                        started=began,finished=ended,paired_shape=list(paired.shape),'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                        replay_capture_storage=storage_report,capture_sources=capture_sources,
                        training_iteration=scheduler_epoch+1,root_checkpoints_included=False,
                        metadata_only=True,new_CUDA_calls=0,new_collectives=0,new_tensor_copies=0,''')
anchor = '                runner.attribute=native_attribute'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                active_storage[0]=None
                for cls,had_own_exit,native_exit in reversed(saved_exits):
                    if had_own_exit:cls.__exit__=native_exit
                    else:delattr(cls,'__exit__')
''' + anchor)
observation = observation.replace('restored_original_methods=True,additional_DT_calls=0)',
    'restored_original_methods=True,additional_DT_calls=0,capture_class_methods_restored=True,training_iteration=scheduler_epoch+1)')

wrapper = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'native-capture-storage-readonly-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"
                    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(wrapper), 'exec'), globals())
