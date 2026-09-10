"""Refresh graph inputs once per original storage, preserving every native view."""
import torch


def at_path(value,path):
    for part in path:value=value[part]
    return value


def storage_key(value):
    return (str(value.device),value.untyped_storage().data_ptr(),value.dtype)


def clone_tree(value,memo,paths,path=()):
    if id(value) in memo:return memo[id(value)]
    if isinstance(value,torch.Tensor):
        groups=memo.setdefault('storage_groups',{})
        key=storage_key(value);size=value.untyped_storage().nbytes()//value.element_size()
        if key not in groups:
            flat=value.detach().as_strided((size,),(1,),0).clone()
            groups[key]=(path,flat,size)
        representative,flat,size=groups[key]
        result=flat.as_strided(value.shape,value.stride(),value.storage_offset())
        memo[id(value)]=result
        paths.append((path,result,representative,size))
        return result
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


def refresh(source,paths,groups,*,audit=False):
    count=0;fresh={}
    for representative,destination,size in groups:
        current=at_path(source,representative)
        assert current.dtype==destination.dtype and current.device==destination.device
        assert current.untyped_storage().nbytes()==size*current.element_size()
        fresh[representative]=storage_key(current)
        destination.copy_(current.detach().as_strided((size,),(1,),0),non_blocking=True)
    for path,destination,representative,size in paths:
        current=at_path(source,path)
        assert isinstance(current,torch.Tensor) and destination.shape==current.shape
        assert destination.dtype==current.dtype and destination.device==current.device
        assert destination.stride()==current.stride() and destination.storage_offset()==current.storage_offset()
        assert storage_key(current)==fresh[representative]
        if audit:assert torch.equal(destination,current);count+=1
    return count
