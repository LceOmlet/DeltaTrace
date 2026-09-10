"""Reject changes before graph execution; validate actual root lifetime guards."""
import torch

def run(model,new,base,ids,mask,prompt_len,old,fa):
    controls=[]
    def reject(name,change,restore,expected):
        change();error=None
        try:new.attribute(base,ids,mask,prompt_len,mutation_audit=True)
        except (AssertionError,ValueError) as exc:error=str(exc)
        finally:restore()
        assert error is not None and (not expected or expected in error),(name,error)
        assert not new.busy
        controls.append({'case':name,'rejected_before_graph_execution':True,'error':error})
    reject('training',lambda:model.train(),lambda:model.eval(),'')
    previous=model.config.attention_dropout
    reject('configuration',lambda:setattr(model.config,'attention_dropout',previous+0.1),lambda:setattr(model.config,'attention_dropout',previous),'configuration changed')
    handle=[]
    reject('new_forward_hook',lambda:handle.append(model.register_forward_pre_hook(lambda *a:None)),lambda:handle.pop().remove(),'hooks or training state changed')
    module=model.model.layers[0].input_layernorm;parameter=module.weight
    replacement=torch.nn.Parameter(parameter.detach().clone(),requires_grad=False)
    reject('parameter_identity',lambda:setattr(module,'weight',replacement),lambda:setattr(module,'weight',parameter),'parameter identity')
    del replacement
    # Check root endpoint guards directly. Adding zero preserves values and only
    # changes an actual retained activation version after layer 0 has finished.
    from qwen3_whole_graph_core import root_endpoints
    held=[];handles=[]
    def hold(_module,args):held.append(args[0])
    def mutate(_module,args):held[0].add_(0)
    handles.append(model.model.layers[0].register_forward_pre_hook(hold))
    handles.append(model.model.layers[1].register_forward_pre_hook(mutate))
    error=None
    try:
        with torch.no_grad():root_endpoints(model,torch.cat((base,ids)),prompt_len)
    except AssertionError as exc:error=str(exc)
    finally:
        for handle in handles:handle.remove()
        held.clear()
    assert error is not None
    controls.append({'case':'original_root_late_activation_version','rejected':True,'values_unchanged_add_zero':True})
    def add_zero():
        with torch.no_grad():parameter.add_(0)
    reject('parameter_version',add_zero,lambda:None,'parameter identity')
    return controls
