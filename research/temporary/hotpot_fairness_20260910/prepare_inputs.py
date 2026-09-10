"""Construct source-only units first, then independently attach scoring labels."""
import argparse
import collections
import json
import re
import numpy as np
from common import HERE, ROOT, TASK, full_cases, sha
from evidence_protocol import source_span
from hotpot_evidence import native_units, token_groups

def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','cache','tokenizer'):p.add_argument('--'+n,type=__import__('pathlib').Path,required=True)
    a=p.parse_args()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(a.tokenizer));plan=json.loads((HERE/'protocol.json').read_bytes())
    assert sha(a.source)==plan['source_sha256'] and sha(a.cache)==plan['tasks'][TASK]['cache_sha256']
    source=json.loads(a.source.read_bytes());cache=[json.loads(s) for s in a.cache.read_bytes().splitlines()]
    byq=collections.defaultdict(list)
    for r in source:byq[r['question'].strip()].append(r)
    runs,receipts=full_cases(); candidates=[];matched=[]
    for i,ex in enumerate(cache):
        prompt=ex['prompt'];question=re.sub(r'\s*Answer:\s*$','',prompt.rsplit('Question:',1)[1]).strip()
        assert len(byq[question])==1
        original=byq[question][0];matched.append(original)
        span=source_span(TASK,prompt);enc=tok.encode(' '+prompt,add_special_tokens=False)
        row=runs[i][0];keep=row['keep'];units=native_units(prompt,original['context'],span['start'],span['end'])
        groups=token_groups(' '+prompt,enc.offsets,keep,units)
        assert len(enc.ids)==len(row['user_positions'])
        assert all(groups[j] for j,u in enumerate(units) if u['kind']=='sentence')
        body_tokens=sorted(t for j,u in enumerate(units) if u['kind']=='sentence' for t in groups[j])
        candidates.append(dict(index=i,source_id=original['_id'],question=question,prompt_sha256=span['prompt_sha256'],
            context_count=len(original['context']),units=units,groups=groups,keep=keep,body_tokens=body_tokens))
    # Commit the candidate hash before reading or constructing supporting-fact labels.
    write('candidates.json',dict(source_sha256=sha(a.source),cache_sha256=sha(a.cache),tokenizer_sha256=sha(a.tokenizer),
        builder_sha256=sha(__import__('pathlib').Path(__file__)),unit_code_sha256=sha(ROOT/'experiments/official/hotpot_evidence.py'),
        full_vector_origins=receipts,cases=candidates))
    review=json.loads((HERE/'semantic_review.json').read_bytes());labels=[];changes=[];stats=[];review_docs=[];crossings=[]
    for c,ex,original in zip(candidates,cache,matched):
        i=c['index'];units=c['units'];groups=c['groups'];gold=list(map(tuple,original['supporting_facts']))
        old=ex['metadata']['needle_spans'];assert set(gold)=={(r['title'],r['sentence_index']) for r in old}
        unit_by_key={(u['title'],u['sentence_index']):j for j,u in enumerate(units) if u['kind']=='sentence'}
        restored=[]
        for r in old:
            u=units[unit_by_key[r['title'],r['sentence_index']]]
            assert r['sentence']==ex['prompt'][u['start']:u['end']]
            if r['span']!=[u['start'],u['end']]:changes.append(dict(index=i,title=r['title'],sentence_index=r['sentence_index'],old_span=r['span'],restored_span=[u['start'],u['end']]))
            restored.append(dict(title=u['title'],sentence_index=u['sentence_index'],span=[u['start'],u['end']]))
        revised=list(gold)
        for replacement in review['replacements']:
            if replacement['index']==i:
                key=(replacement['title'],replacement['remove']);assert key in revised
                revised[revised.index(key)]=(replacement['title'],replacement['add'])
        text=' '+ex['prompt'];enc=tok.encode(text,add_special_tokens=False);offsets=enc.offsets
        for j,u in enumerate(units):
            for t in groups[j]:
                lo,hi=offsets[t]
                if hi>u['end']+1 and text[u['end']+1:hi].strip():
                    crossings.append(dict(index=i,token=t,text=text[lo:hi],unit=j,title=u['title'],sentence_index=u['sentence_index']))
        # Exact old overlap projection, changing coordinates only in this diagnostic.
        def project(spans):return sorted(t for t in c['keep'] if any(offsets[t][0]<b+1 and offsets[t][1]>a+1 for a,b in spans))
        old_projected=project([x['span'] for x in old]);assert old_projected==sorted(set(runs[i][0]['gold']) & set(c['keep']))
        labels.append(dict(index=i,official_restored=gold,review_corrected=revised,official_spans=restored,
            cached_gold_tokens=old_projected,restored_overlap_gold_tokens=project([x['span'] for x in restored])))
        boundaries=[m.end() for m in re.finditer(r'(?<=[.!?])[ \t]+|\n+',text)]
        old_groups=np.searchsorted(boundaries,[a for a,b in offsets],side='right')
        anchors=[next((k for k in range(a,b) if not text[k].isspace()),a) for a,b in offsets]
        new_groups=np.searchsorted(boundaries,anchors,side='right')
        costs=[len(groups[unit_by_key[k]]) for k in gold]
        shifted=sum(any(old_groups[t]!=new_groups[t] for t in groups[unit_by_key[k]]) for k in gold)
        stats.append(dict(index=i,gold_facts=len(gold),gold_costs=costs,max_min_cost_ratio=max(costs)/min(costs),
            shifted_eligible_tokens=int(sum(old_groups[t]!=new_groups[t] for t in c['keep'])),gold_facts_with_shift=int(shifted),
            eligible_body_tokens=len(c['body_tokens']),eligible_metadata_tokens=len(c['keep'])-len(c['body_tokens'])))
        if i in plan['review_indices']+plan['ambiguity_indices']:
            review_docs.append(dict(index=i,question=c['question'],answer=original['answer'],gold=gold,
                documents={t:ss for t,ss in original['context'] if t in {x[0] for x in gold}}))
    assert len(candidates)==48 and sum(c['context_count'] for c in candidates)==480
    assert [x['index'] for x in changes]==[8]
    write('labels.json',dict(candidate_sha256=sha(HERE/'candidates.json'),semantic_review_sha256=sha(HERE/'semantic_review.json'),cases=labels))
    write('source_audit.json',dict(status='all_480_documents_and_48_token_maps_verified',candidate_sha256=sha(HERE/'candidates.json'),
        labels_sha256=sha(HERE/'labels.json'),coordinate_changes=changes,crossing_tokens=crossings,per_case=stats,
        summary=dict(cases=48,documents=480,native_sentences=sum(sum(u['kind']=='sentence' for u in c['units']) for c in candidates),
            gold_facts=sum(s['gold_facts'] for s in stats),shifted_eligible_tokens=sum(s['shifted_eligible_tokens'] for s in stats),
            gold_facts_with_shift=sum(s['gold_facts_with_shift'] for s in stats),median_gold_length_ratio=float(np.median([s['max_min_cost_ratio'] for s in stats])),
            maximum_gold_length_ratio=max(s['max_min_cost_ratio'] for s in stats))))
    write('review_source_excerpt.json',dict(source_sha256=sha(a.source),cases=review_docs))
    print(json.dumps(json.loads((HERE/'source_audit.json').read_bytes())['summary']))

if __name__=='__main__':main()
