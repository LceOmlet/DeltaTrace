"""Batch scheduling only; execute the unchanged original FlashTrace evaluator."""
from collections import defaultdict

def evaluate_requests(evaluator, requests, batch_size):
    """requests: unique key -> ([1,prompt_length], [1,response_length]).

    Bucket exact lengths; no padding, token replacement, model implementation,
    score equation or forward override lives here. Each request is independent.
    """
    import torch
    if not isinstance(batch_size,int) or batch_size<1:raise ValueError('positive batch_size required')
    groups=defaultdict(list)
    for key,(prompt,response) in requests.items():
        assert prompt.ndim==response.ndim==2 and prompt.shape[0]==response.shape[0]==1
        assert prompt.device==response.device and prompt.dtype==response.dtype
        groups[(prompt.shape[1],response.shape[1],prompt.device,prompt.dtype)].append(key)
    result={};physical_calls=0;trajectories=0;actual_batch_sizes=[]
    with torch.no_grad():
        for keys in groups.values():
            for start in range(0,len(keys),batch_size):
                chunk=keys[start:start+batch_size]
                prompt=torch.cat([requests[k][0] for k in chunk],dim=0)
                response=torch.cat([requests[k][1] for k in chunk],dim=0)
                # This is the original bound method, not a copied evaluator.
                value=evaluator.compute_logprob_response_given_prompt(prompt,response)
                physical_calls+=1;trajectories+=len(chunk);actual_batch_sizes.append(len(chunk))
                assert value.shape==response.shape and torch.isfinite(value).all()
                # Sum each response separately in the original dtype, matching
                # original batch=1 value.sum(); never sum across examples.
                sums=torch.stack([value[i:i+1].sum() for i in range(len(chunk))]).cpu().tolist()
                for key,score in zip(chunk,sums):result[key]=float(score)
    assert set(result)==set(requests)
    return result,{'physical_evaluation_forwards':physical_calls,
                   'evaluation_trajectories':trajectories,'actual_batch_sizes':actual_batch_sizes}
