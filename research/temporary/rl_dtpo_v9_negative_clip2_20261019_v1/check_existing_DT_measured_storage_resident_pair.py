"""Use actual retention-source storage for one native long-B4 A-B-A check.

Compose the existing metadata observation and native owner comparison. Source
storage is counted per retention event, without pointer deduplication: an
allocator may reuse an address after an offloaded source dies. No tensor or
storage reference is retained by the measurement. Production receives A0.
"""
import builtins
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def existing_observation(name, expected):
    source = HERE / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    namespace = {'__name__': 'original_observation_source_only', '__file__': str(source)}
    stopped = []

    def source_exec(compiled, globals_dict):
        if compiled.co_filename == str(HERE / 'inspect_native_workers.py'):
            stopped.append(compiled.co_filename)
        else:
            builtins.exec(compiled, globals_dict)

    namespace['exec'] = source_exec
    builtins.exec(compile(source.read_text(), str(source), 'exec'), namespace)
    assert len(stopped) == 1
    return namespace['observation']


observation = existing_observation('observe_existing_DT_capture_storage.py',
    'fb8bd249645ab03231e96e43a0ef80f90d065800105d9741fb594159b934523a')
paired_observation = existing_observation('check_existing_DT_long_gpu_resident_pair.py',
    'a339ff751ae8f18120d6e50e22ed4d7bd6e91ecfb78aefb6474f863c63a02404')
start = paired_observation.index('                pair_info=None')
end = paired_observation.index('                try:\n                    detail=returned[1]', start)
control = paired_observation[start:end]
for old, new in (
    ('and not getattr(runner,\'fa_replay_capture_cuda\',False)):',
     "and not getattr(runner,'fa_replay_capture_cuda',False)\n                            and storage_report['source_retention_coverage_complete']):"),
    ('extra_bound=48*paired.shape[0]*paired.shape[1]*max(hidden,heads*head_dim)*runner.model.lm_head.weight.element_size()',
     "source_maxima=storage_report['maximum_source_bytes_by_capture']\n"
     "                            extra_bound=source_maxima.get('NativeDecoderCapture',0)+max(source_maxima.get('NativeGDNCapture',0),source_maxima.get('NativeDenseAttentionCapture',0))\n"
     "                            kv_heads=int(getattr(config,'num_key_value_heads',heads))\n"
     "                            full_layers=sum(layer.block_type=='full_attention' for layer in runner.model.model.language_model.layers)\n"
     "                            cache_bound=4*paired.shape[0]*paired.shape[1]*kv_heads*head_dim*runner.model.lm_head.weight.element_size()*full_layers"),
    ('required=workspace+extra_bound+2*2**30', 'required=workspace+extra_bound+cache_bound+2*2**30'),
    ('required_with_margin=required,hidden_size=hidden,attention_width=heads*head_dim)',
     "required_with_margin=required,hidden_size=hidden,attention_width=heads*head_dim,\n"
     "                                two_native_KV_cache_bound=cache_bound,KV_heads=kv_heads,full_attention_layers=full_layers,source_storage_coverage=True)"),
):
    assert old in control, old
    control = control.replace(old, new)

for old, new in (
    ('_existing_DT_capture_storage_scope_installed', '_measured_source_resident_pair_scope_installed'),
    ('native_DT_capture_storage', 'native_DT_measured_source_resident_pair'),
    ('existing_DT_capture_storage_observer_', 'measured_source_resident_pair_observer_'),
):
    assert old in observation
    observation = observation.replace(old, new)
anchor = '        import torch,inspect,hashlib'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
        original_native=runner._bounded_native_mixer_storage['original_attribute']
        native_file=Path(original_native.__func__.__code__.co_filename)
        assert hashlib.sha256(native_file.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
        assert producer.direct_readout.trajectories.__name__=='trajectories' ''')
# Path must be imported before it is used by the added native-source check.
observation = observation.replace('        import torch,inspect,hashlib\n',
    '        from pathlib import Path\n        import torch,inspect,hashlib\n', 1)
anchor = '        def observe_one_existing_readout(*args,**kwargs):'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''        retention_owners=[]
        for kind,cls in capture_classes:
            method='copy' if kind=='NativeGDNCapture' else 'retain'
            native_method=getattr(cls,method)
            assert hashlib.sha256(Path(inspect.getfile(native_method)).read_bytes()).hexdigest()=='5ad0bf9b8f9ec1868a20d964ad7a4a41c328f4da40c5d993f7ef92e8a5eaa810'
            retention_owners.append((cls,method,native_method))
''' + anchor)
anchor = '            count=[0]'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
            validation_calls=[0]
            guard_calls=[0]
            saved_retention_methods=[]
            def source_metadata(retained,source,capture,label):
                current=active_storage[0]
                if current is None:return
                try:
                    if not isinstance(retained,torch.Tensor) or retained.device.type!='cpu':return
                    if not isinstance(source,torch.Tensor):
                        current['metadata_errors'].append('CPU retention has no tensor source')
                        return
                    target=retained.untyped_storage();origin=source.untyped_storage()
                    record=vars(capture).setdefault('_native_source_storage_metadata',dict(CPU_targets=set(),source_event_bytes=0,events=[]))
                    record['CPU_targets'].add((target.data_ptr(),target.nbytes()))
                    bound=max(origin.nbytes(),target.nbytes())
                    record['source_event_bytes']+=bound
                    record['events'].append(dict(label=label,source_shape=list(source.shape),
                        source_stride=list(source.stride()),source_dtype=str(source.dtype),
                        source_device=str(source.device),source_storage_bytes=origin.nbytes(),
                        CPU_destination_storage_bytes=target.nbytes(),counted_bound_bytes=bound))
                except Exception as error:
                    current['metadata_errors'].append(type(error).__name__+': '+str(error))
            def wrap_retention(native_method,method):
                def retain_with_source(capture,*args,**kwargs):
                    returned=native_method(capture,*args,**kwargs)
                    if active_storage[0] is not None:
                        try:
                            if method=='retain':
                                label=args[0] if args else kwargs['name']
                                value=args[1] if len(args)>1 else kwargs['value']
                                target=capture.values.get(label)
                            else:
                                label='native_copy'
                                value=args[0] if args else kwargs['value']
                                target=returned
                            source_metadata(target,value,capture,label)
                        except Exception as error:
                            active_storage[0]['metadata_errors'].append(type(error).__name__+': '+str(error))
                    return returned
                return retain_with_source''')
anchor = '                    current=active_storage[0]'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                    source_record=vars(capture).pop('_native_source_storage_metadata',None)''')
anchor = "                            seen={};cpu_bytes=0;gpu_bytes=0;tensors=[]"
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                            CPU_targets=set()''')
anchor = "                                        if value.device.type=='cpu':cpu_bytes+=storage.nbytes()"
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                                        if value.device.type=='cpu':
                                            cpu_bytes+=storage.nbytes()
                                            CPU_targets.add((storage.data_ptr(),storage.nbytes()))''')
anchor = "                            current['capture_exit_counts'][kind]=current['capture_exit_counts'].get(kind,0)+1"
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                            covered=source_record['CPU_targets'] if source_record else set()
                            missing=CPU_targets-covered
                            current['uncovered_CPU_storage_count']+=len(missing)
                            source_bytes=source_record['source_event_bytes'] if source_record else 0
                            if source_bytes>current['maximum_source_bytes_by_capture'].get(kind,-1):
                                current['maximum_source_bytes_by_capture'][kind]=source_bytes
                                current['maximum_source_details'][kind]=(source_record['events'] if source_record else [])
''' + anchor)
anchor = '            scheduler_epoch=actor.actor_lr_scheduler.last_epoch'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''            for cls,method,native_method in retention_owners:
                saved_retention_methods.append((cls,method,method in vars(cls),native_method))
                setattr(cls,method,wrap_retention(native_method,method))
''' + anchor)
anchor = '                    maximum_CPU_capture_details={},metadata_errors=[])'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                    maximum_CPU_capture_details={},metadata_errors=[],
                    maximum_source_bytes_by_capture={},maximum_source_details={},uncovered_CPU_storage_count=0)''')
anchor = '                try:\n                    detail=returned[1]'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                storage_report['source_retention_coverage_complete']=(storage_report['complete_layer_capture_counts']
                    and not storage_report['metadata_errors'] and storage_report['uncovered_CPU_storage_count']==0)
''' + control + anchor)
anchor = '                        replay_capture_storage=storage_report,capture_sources=capture_sources,'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, anchor + '''
                        optional_same_input_measured_source_resident_pair=pair_info,
                        measured_source_memory_guard=memory_guard,''')
observation = observation.replace('metadata_only=True,new_CUDA_calls=0,new_collectives=0,new_tensor_copies=0,',
    'capture_metadata_only=True,metadata_added_CUDA_calls=0,metadata_added_collectives=0,metadata_added_tensor_copies=0,')
observation = observation.replace('additional_DT_calls=0,new_CUDA_events=0,original_return_objects_unchanged=True,',
    "additional_DT_calls=(pair_info['additional_DT_calls'] if pair_info else 0),metadata_added_CUDA_events=0,original_return_objects_unchanged=True,")
anchor = '                for cls,had_own_exit,native_exit in reversed(saved_exits):'
assert observation.count(anchor) == 1
observation = observation.replace(anchor, '''                for cls,method,had_own,native_method in reversed(saved_retention_methods):
                    if had_own:setattr(cls,method,native_method)
                    else:delattr(cls,method)
''' + anchor)
observation = observation.replace('restored_original_methods=True,additional_DT_calls=0,capture_class_methods_restored=True,training_iteration=scheduler_epoch+1)',
    'restored_original_methods=True,additional_DT_calls=validation_calls[0],capture_class_methods_restored=True,retention_methods_restored=True,validation_guard_collectives=guard_calls[0],training_iteration=scheduler_epoch+1)')
observation = observation.replace("result['measured_source_resident_pair_observer_additional_calls']=0",
    "result['measured_source_resident_pair_observer_max_validation_calls_per_rank']=2")

wrapper = HERE / 'inspect_native_workers.py'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'measured-source-resident-pair-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"
                    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(wrapper), 'exec'), globals())
