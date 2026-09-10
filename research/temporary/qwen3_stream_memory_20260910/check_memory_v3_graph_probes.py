"""Charged, small GPU controls for replay-time strict validation and storage views."""
def run():
    import time,torch
    from qwen3_output_template import TemplateGraphValidation as GraphValidation
    from qwen3_graph_storage_inputs import clone_tree,refresh
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    started=time.perf_counter();device='cuda';records=[]
    a=torch.tensor([1.,2.,3.],device=device);b=a.clone();f=a.clone();p=a.clone()
    seed=torch.tensor(True,device=device);signed=a.double()
    q=torch.ones((1,5,2,8),device=device,dtype=torch.float16);k=q.clone()
    cos=torch.ones((1,5,8),device=device,dtype=torch.float16);sin=cos*.25
    aq,ak=apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin)
    aq=aq.transpose(1,2).clone();ak=ak.transpose(1,2).clone();oq=aq.clone();ok=ak.clone()
    endpoint={'score16':0.,'score32_sum64':0.}
    def invoke():
        checks=GraphValidation();checks.predicate(seed,'seed')
        eq=checks.equal(a,b,'equal');checks.finite(f,'finite');checks.positive(p,'positive')
        checks.rope(q,k,aq,ak,cos,sin,'rope_q','rope_k');maximum=checks.max_abs(a)
        return checks.finish({'signed_full_sequence':signed,'signed_sum':signed.sum(),'equal':eq,'maximum':maximum})
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(),torch.cuda.stream(stream):
        warm=invoke();assert warm.resolve(endpoint,endpoint,0.)['maximum']==3.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):packed=invoke()
    torch.cuda.current_stream().wait_stream(stream)
    labels={'equal':'equal','finite':'finite','positive':'positive','seed':'seed','rope_q':'rope_q','rope_k':'rope_k','signed':'Nonfinite signed attribution'}
    for bad in [None,'equal','finite','positive','seed','rope_q','rope_k','signed']:
        b.copy_(a);f.copy_(a);p.copy_(a);seed.fill_(True);signed.copy_(a);aq.copy_(oq);ak.copy_(ok)
        if bad=='equal':b[1]=7
        if bad=='finite':f[1]=float('inf')
        if bad=='positive':p[1]=0
        if bad=='seed':seed.fill_(False)
        if bad=='rope_q':aq[0,0,0,0]+=1
        if bad=='rope_k':ak[0,0,0,0]+=1
        if bad=='signed':signed[0]=float('nan')
        graph.replay()
        rejected=False
        try:
            result=packed.resolve(endpoint,endpoint,0.)
            assert bad is None and result['equal'] and result['maximum']==3.
            assert result['signed_full_sequence']==[1.,2.,3.] and result['signed_sum']==6.
        except (ValueError,AssertionError) as exc:
            assert bad is not None and labels[bad] in str(exc),(bad,str(exc));rejected=True
        records.append({'case':bad or 'valid','passed':True,'rejected_before_return':rejected})
    raw=torch.arange(24,device=device,dtype=torch.float32).reshape(4,6)
    source={'whole':raw,'transpose':raw.t(),'offset':raw[1:,2:],'repeat':raw}
    memo={};paths=[];cloned=clone_tree(source,memo,paths);groups=list(memo['storage_groups'].values())
    assert len(groups)==1 and cloned['whole'] is cloned['repeat']
    newer=raw+7;fresh={'whole':newer,'transpose':newer.t(),'offset':newer[1:,2:],'repeat':newer}
    assert refresh(fresh,paths,groups,audit=True)==3
    assert all(torch.equal(cloned[key],fresh[key]) for key in source)
    fresh['offset']=fresh['offset'].clone();rejected=False
    try:refresh(fresh,paths,groups)
    except AssertionError:rejected=True
    assert rejected
    torch.cuda.synchronize()
    return {'seconds':time.perf_counter()-started,'strict_graph_cases':records,
        'strict_graph_warm_programs':1,'strict_graph_recordings':1,'strict_graph_replays':len(records),
        'storage_views_exact':True,'changed_storage_topology_rejected':True,
        'model_calls':0,'FA_calls':0,'attribution_calls':0}
