"""Explicit native-sentence retrieval with full cost and nested rank prefixes.

The prior hotpot_evidence module is frozen for v2 reproduction. Mapping and
set-metric helpers remain reusable; v3 cost and selection rules are separate.
"""
import numpy as np
from hotpot_evidence import native_units, token_groups, supporting_fact_metrics


def answer_conditioned_weights(full_target_tokens, answer_span, eos_position):
    """Change seeds only; this API never constructs or shortens model input."""
    start,end=map(int,answer_span)
    if not (0<=start<=end<eos_position==len(full_target_tokens)-1):
        raise ValueError('Answer span must lie before the existing terminal EOS')
    weights=[float(start<=i<=end and str(token).strip() not in ('',',','.'))
             for i,token in enumerate(full_target_tokens)]
    if not any(weights):raise ValueError('No answer target seeds')
    assert not any(weights[:start]) and not any(weights[end+1:])
    return weights


def all_token_groups(text, offsets, units, *, coordinate_shift=1):
    """Cost every source-contained tokenizer token, independent of stop masks."""
    lo,hi=units[0]['start']+coordinate_shift,units[-1]['end']+coordinate_shift
    indices=[i for i,(a,b) in enumerate(offsets) if lo<=a<b<=hi]
    return token_groups(text,offsets,indices,units,coordinate_shift=coordinate_shift)


def rank_sentences(scores, units, all_groups, eligible_groups, *, pooling):
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Invalid scores')
    if pooling not in ('signed_sum','positive_mean_eligible'):raise ValueError('Unknown pooling')
    candidates=[i for i,u in enumerate(units) if u['kind']=='sentence' and all_groups[i]]
    pooled={}
    for i in candidates:
        if pooling=='signed_sum':pooled[i]=float(np.sum(values[all_groups[i]],dtype=np.float64))
        else:
            if not eligible_groups[i]:raise ValueError('Native sentence without old eligible scores')
            # Preserve v2's float32 positive view, then sum in float64.
            v=np.maximum(np.asarray(scores,dtype=np.float32),0)
            pooled[i]=float(np.sum(v[eligible_groups[i]],dtype=np.float64)/len(eligible_groups[i]))
    return sorted(candidates,key=lambda i:(-pooled[i],units[i]['start'])),pooled


def select_prefix(order, all_groups, *, token_budget=None, sentence_budget=None):
    if (token_budget is None)==(sentence_budget is None):raise ValueError('Choose one budget unit')
    budget=token_budget if token_budget is not None else sentence_budget
    if isinstance(budget,bool) or not isinstance(budget,(int,np.integer)) or budget<0:
        raise ValueError('Budget must be a nonnegative integer')
    if len(set(order))!=len(order) or any(not all_groups[i] for i in order):raise ValueError('Invalid order')
    selected=[];tokens=[]
    for i in order:
        if sentence_budget is not None:
            if len(selected)==sentence_budget:break
        elif len(tokens)+len(all_groups[i])>token_budget:break
        selected.append(i);tokens.extend(all_groups[i])
    if len(tokens)!=len(set(tokens)):raise ValueError('A token is charged to more than one sentence')
    return selected,tokens
