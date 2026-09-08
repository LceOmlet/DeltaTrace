"""Unchanged audit/count helpers extracted from the preserved production study."""
import torch,hashlib
sha=lambda b:hashlib.sha256(b).hexdigest()

class CountFinite:
    """Count attribution callback entries/returns; never wrap a native operator."""
    def __init__(self,operation):self.operation=operation;self.entered=0;self.returned=0
    def __call__(self,*args,**kwargs):
        self.entered+=1;out=self.operation(*args,**kwargs);self.returned+=1;return out

def deletion_audit(score,ids,keep,eos):
    """Input/order audit only; the original function computes any reported metric."""
    w=score[None].sum(0);keep=sorted(set(keep));assert len(keep)>=20
    local=torch.argsort(w[torch.tensor(keep,dtype=torch.long)],descending=True)
    order=[keep[int(i)] for i in local];size,remainder=divmod(len(keep),20)
    groups=[];receipts=[];changed=set();perturbed=ids.clone()
    receipts.append({'input_sha256':sha(perturbed[None].numpy().tobytes()),'deleted_positions':[]})
    start=0
    for step in range(20):
        group=order[start:start+size+(step<remainder)];start+=len(group);groups.append(group)
        changed.update(group);perturbed[group]=eos
        actual=(perturbed!=ids).nonzero().flatten().tolist();assert actual==sorted(changed)
        receipts.append({'input_sha256':sha(perturbed[None].numpy().tobytes()),'deleted_positions':actual})
    return {'sorted_keep':order,'groups':groups,'input_receipts':receipts}

def check_counts(details,fa_delta,fla_delta):
    kinds=[q['kind'] for q in details['calls']]
    counts={'native_root':kinds.count('native_root_with_CPU_checkpoints'),
        'native_decoder_replays':sum(q.startswith('native_replay_') for q in kinds),
        'finite_decoder_calls':sum(q.startswith('finite_decoder_') for q in kinds),
        'public_FA_auxiliary_calls':sum(q.startswith('public_FA_LSE_') for q in kinds),
        'finite_FA_calls':fa_delta,'finite_FLA_calls':fla_delta}
    assert counts=={'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,
        'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24},counts
    assert len(details['layers'])==32
    for q in details['layers'].values():
        assert q['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','up','silu','down','mlp','decoder')}
        expected=({'module':1,'interface':1,'native_varlen':0,'native_dense':1}
            if q['block_type']=='full_attention' else {'module':1,'conv':1,'FLA':1,'stage':1})
        assert q['mixer_calls']==expected
    return counts
