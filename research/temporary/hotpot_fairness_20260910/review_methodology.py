"""Audit experimental meaning beyond agreement with the implemented scorer."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from common import HERE,sha,read_run
from hotpot_evidence import token_groups

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));plan=json.loads((HERE/'protocol.json').read_bytes())
    assert sha(a.cache)==plan['tasks']['hotpotqa_long']['cache_sha256']
    cache=[json.loads(s) for s in a.cache.read_bytes().splitlines()]
    candidates={r['index']:r for r in json.loads((HERE/'candidates.json').read_bytes())['cases']}
    targets=[]
    for chunk in plan['chunks']:
        r,v,_=read_run(HERE/'raw/answer_v2'/chunk)
        for row in r['cases']:
            i=row['index'];ex=cache[i];lo,hi=row['original_answer_char_span']
            assert row['target']==ex['target'][lo:hi] and lo>0
            enc=tok.encode(ex['target'],add_special_tokens=False)
            start,end=ex['indices_to_explain'];assert lo==enc.offsets[start][0] and hi==enc.offsets[end][1]
            targets.append(dict(index=i,removed_prefix_tokens=start,removed_prefix_characters=lo,
                original_target_tokens=len(enc.ids),replacement_target_tokens=row['target_length']-1,
                original_context_preserved=False))
        v.close()
    assert len(targets)==48
    predictions=json.loads((HERE/'selections.json').read_bytes())
    allgroups={};allbody={}
    for i,ex in enumerate(cache):
        c=candidates[i];units=c['units'];text=' '+ex['prompt'];offsets=tok.encode(text,add_special_tokens=False).offsets
        eligible=[t for t,(lo,hi) in enumerate(offsets) if units[0]['start']+1<=lo<hi<=units[-1]['end']+1]
        allgroups[i]=token_groups(text,offsets,eligible,units)
        allbody[i]=sum(len(g) for u,g in zip(units,allgroups[i]) if u['kind']=='sentence')
    costs=[]
    for row in predictions:
        if row['budget_unit']!='body_tokens' or row['budget_value']!=.1:continue
        i=row['index'];tokens={t for j in row['selected_units'] for t in allgroups[i][j]}
        costs.append(dict(index=i,target_mode=row['target_mode'],method=row['method'],
            declared_eligible_cost=len(row['selected_tokens']),all_body_token_cost=len(tokens),
            declared_eligible_budget=row['budget'],all_body_token_budget=math.ceil(.1*allbody[i])))
    metrics=list(csv.DictReader((HERE/'native_per_case.csv').open(encoding='utf-8')))
    bykey={(int(r['index']),r['target_mode'],r['method'],float(r['budget_value'])):r for r in metrics
        if r['labels']=='official_restored' and r['budget_unit']=='body_tokens'}
    ps={(int(r['index']),r['target_mode'],r['method'],r['budget_value']):set(r['selected_units']) for r in predictions if r['budget_unit']=='body_tokens'}
    transitions=[]
    for i in range(48):
        for target in ('full','answer_only'):
            for method in ('DT_target','FT_K1','FT_K3'):
                for lo,hi in ((.05,.1),(.1,.2)):
                    k=(i,target,method);r0=bykey[k+(lo,)];r1=bykey[k+(hi,)]
                    if not ps[k+(lo,)]<=ps[k+(hi,)]:
                        transitions.append(dict(index=i,target_mode=target,method=method,from_fraction=lo,to_fraction=hi,
                            recall_before=float(r0['recall']),recall_after=float(r1['recall']),
                            selected_sets_nested=False,recall_decreased=float(r1['recall'])<float(r0['recall'])))
    # A larger budget can admit an earlier long non-gold sentence, displacing
    # two later gold sentences; this is a property of the chosen packing rule.
    from hotpot_evidence import select_sentences
    groups=[list(range(6)),[6,7,8],[9,10]]
    before,_=select_sentences([0,1,2],groups,token_budget=5)
    after,_=select_sentences([0,1,2],groups,token_budget=6)
    assert before==[1,2] and after==[0]
    summary=dict(answer_context_removed_cases=len(targets),removed_prefix_tokens_min=min(r['removed_prefix_tokens'] for r in targets),
        removed_prefix_tokens_max=max(r['removed_prefix_tokens'] for r in targets),
        removed_prefix_tokens_mean=statistics.fmean(r['removed_prefix_tokens'] for r in targets),
        budget_transitions_checked=576,non_nested_selection_transitions=len(transitions),
        fact_recall_decrease_transitions=sum(r['recall_decreased'] for r in transitions),
        selected_rows_at_10_percent=len(costs),rows_with_uncharged_punctuation_or_whitespace=sum(r['all_body_token_cost']>r['declared_eligible_cost'] for r in costs),
        rows_exceeding_all_body_token_budget=sum(r['all_body_token_cost']>r['all_body_token_budget'] for r in costs),
        average_declared_eligible_cost=statistics.fmean(r['declared_eligible_cost'] for r in costs),
        average_all_body_token_cost=statistics.fmean(r['all_body_token_cost'] for r in costs))
    out=dict(status='implementation_reproducible_protocol_claims_require_narrowing',reviewer='same assistant; no independent reviewer',
        code_sha256=sha(Path(__file__)),prior_analysis_sha256=sha(HERE/'analysis.json'),
        source_cache_sha256=sha(a.cache),tokenizer_sha256=sha(a.tokenizer),summary=summary,
        target_context_audit=sorted(targets,key=lambda r:r['index']),budget_transitions=transitions,cost_audit=costs,
        interpretation='Prefix removal changes p(answer | prompt, reasoning) to p(answer | prompt). Non-nested packing and eligible-token costs are explicit design choices, not necessarily code bugs; their symmetric application does not establish measurement neutrality.')
    (HERE/'methodology_review.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
