"""Read native allocator segments; do not allocate, clear or move tensors."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1', 'native-allocator-storage-readonly-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    import torch,inspect
    device=torch.cuda.current_device()
    segments=torch.cuda.memory_snapshot()
    free,total=torch.cuda.mem_get_info()
    selected=[s for s in segments if s.get('device')==device]
    pools={}
    reusable=[]
    for segment in selected:
        pool=segment.get('segment_pool_id')
        key=str(pool)
        entry=pools.setdefault(key,dict(segments=0,total_size=0,active_size=0,allocated_size=0))
        for name in ('total_size','active_size','allocated_size'):entry[name]+=segment.get(name,0)
        entry['segments']+=1
        if (pool is not None and tuple(pool)==(0,0) and segment.get('active_size')==0 and segment.get('blocks')
                and all(b.get('state')=='inactive' for b in segment.get('blocks',[]))):
            reusable.append(dict(bytes=segment['total_size'],pool=list(pool),blocks=len(segment.get('blocks',[]))))
    source=Path(inspect.getfile(torch.cuda.memory_snapshot))
    result['native_allocator_storage']=dict(free_bytes=free,total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        pools=pools,fully_inactive_default_segments=reusable,
        fully_inactive_default_bytes=sum(s['bytes'] for s in reusable),
        unknown_pool_segments=sum(s.get('segment_pool_id') is None for s in selected),
        snapshot_source=str(source),snapshot_source_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        CUDA_memory_queries=True,new_tensor_allocations=0,new_model_calls=0,cache_clears=0,resource_changes=0,
        scope='Native default-pool metadata only. Private/missing pool IDs are excluded from reusable-byte credit. No capacity or speed claim.')
    return result'''
code = code.replace(anchor, "assert code.count('    return result')==1\ncode=code.replace('    return result',"
                    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
