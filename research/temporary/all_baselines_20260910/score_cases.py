"""Score the frozen retrieval policy; selectors receive no answer or gold."""
import math
import numpy as np
from hotpot_retrieval_v3 import rank_sentences,select_prefix
from hotpot_evidence import supporting_fact_metrics,fact_recall_ceiling
from retrieval_views import sentence_density_order
from recovery_diagnostics import recovery_diagnostics

HP_METRICS=('precision','recall','f1','exact_match','complete_support','token_recall','fact_recall_ceiling',
    'spent_tokens','unused_tokens','selected_sentences','empty_selection')

def score_case(item,candidate,label,method,values,*,aggregation='signed_sum'):
    # Native HotpotQA v3 retains the stored signed vector's precision before
    # float64 sentence sums; only the frozen VT token view converts to float32.
    scores=np.asarray(values,dtype=np.float64 if item['dataset']=='hotpotqa_long' else np.float32)
    assert scores.shape==(len(item['user_positions']),) and np.isfinite(scores).all()
    base=dict(dataset=item['dataset'],index=item['index'],method=method,target_mode=item['target_mode'],aggregation=aggregation)
    rows=[];selections=[]
    if item['dataset']=='hotpotqa_long':
        units=candidate['units'];groups=candidate['all_groups'];body=set(candidate['all_body_tokens'])
        assert body=={token for i,u in enumerate(units) if u['kind']=='sentence' for token in groups[i]}
        order,pooled=rank_sentences(scores,units,groups,candidate['eligible_groups'],pooling='signed_sum')
        goldtokens=set(label['gold_all_body_tokens']);previous=set()
        for unit,budgets in [('all_body_tokens',[.05,.1,.2]),('sentences',[2,4,8])]:
            for fraction in budgets:
                budget=math.ceil(len(body)*fraction) if unit=='all_body_tokens' else fraction
                kw={'token_budget':budget} if unit=='all_body_tokens' else {'sentence_budget':budget}
                selected,tokens=select_prefix(order,groups,**kw)
                if unit=='all_body_tokens':assert previous<=set(selected);previous=set(selected)
                row=dict(base,view='native_sentence',budget_unit=unit,fraction=fraction,budget=budget,
                    all_body_tokens=len(body),**supporting_fact_metrics(selected,units,label['official_keys']),
                    token_recall=len(set(tokens)&goldtokens)/len(goldtokens),spent_tokens=len(tokens),
                    unused_tokens=budget-len(tokens) if unit=='all_body_tokens' else 0,
                    selected_sentences=len(selected),empty_selection=int(not selected),
                    fact_recall_ceiling=fact_recall_ceiling(units,groups,label['official_keys'],**kw))
                rows.append(row)
                selections.append(dict(base,view=row['view'],budget_unit=unit,fraction=fraction,budget=budget,
                    ranked_units=order,ranked_scores=[pooled[i] for i in order],selected_units=selected,selected_tokens=tokens))
    else:
        keep=candidate['keep'];positive=np.maximum(scores,0)
        order=sentence_density_order(' '+item['prompt'],candidate['offsets'],positive,keep)
        rank=np.zeros_like(positive);rank[order]=np.arange(len(order),0,-1)
        for view,values in [('raw',positive),('density',rank)]:
            for fraction in [.05,.1,.2]:
                metrics=recovery_diagnostics(values,keep,label['gold'],fraction)
                selected=metrics.pop('selected')
                rows.append(dict(base,view=view,budget_unit='eligible_body_tokens',fraction=fraction,**metrics))
                selections.append(dict(base,view=view,budget_unit='eligible_body_tokens',fraction=fraction,
                    budget=metrics['budget'],selected_tokens=selected))
    return rows,selections
