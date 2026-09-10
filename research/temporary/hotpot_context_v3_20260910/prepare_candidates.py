"""Build native candidates and full token costs before attaching official gold."""
import argparse
import collections
import json
from pathlib import Path
import re
from artifacts_v3 import HERE,ROOT,V2,sha,byte_sha
from hotpot_evidence import native_units,token_groups
from hotpot_retrieval_v3 import all_token_groups
from evidence_protocol import source_span

def write(name,data,compact=False):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=None if compact else 2,
        separators=(',',':') if compact else None)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','cache','tokenizer'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    from tokenizers import Tokenizer
    plan=json.loads((HERE/'protocol.json').read_bytes());prior=json.loads((V2/'analysis.json').read_bytes())
    assert sha(a.source)==plan['source_sha256'] and sha(a.cache)==plan['tasks']['hotpotqa_long']['cache_sha256']
    assert sha(V2/'candidates.json')==prior['candidate_sha256']
    old_candidates=json.loads((V2/'candidates.json').read_bytes())
    assert sha(a.tokenizer)==old_candidates['tokenizer_sha256']
    assert sha(ROOT/'experiments/official/hotpot_evidence.py')==old_candidates['unit_code_sha256']
    old={r['index']:r for r in old_candidates['cases']}
    source=json.loads(a.source.read_bytes());cache=[json.loads(l) for l in a.cache.read_bytes().splitlines()]
    byq=collections.defaultdict(list)
    for row in source:byq[row['question'].strip()].append(row)
    tok=Tokenizer.from_file(str(a.tokenizer));candidates=[];matched=[];crossings=[]
    for i,ex in enumerate(cache):
        prompt=ex['prompt'];q=re.sub(r'\s*Answer:\s*$','',prompt.rsplit('Question:',1)[1]).strip()
        assert len(byq[q])==1
        original=byq[q][0];matched.append(original);span=source_span('hotpotqa_long',prompt)
        units=native_units(prompt,original['context'],span['start'],span['end'])
        assert units==old[i]['units'] and span['prompt_sha256']==old[i]['prompt_sha256']
        text=' '+prompt;enc=tok.encode(text,add_special_tokens=False)
        eligible_groups=token_groups(text,enc.offsets,old[i]['keep'],units)
        assert eligible_groups==old[i]['groups']
        groups=all_token_groups(text,enc.offsets,units)
        for j,u in enumerate(units):
            assert set(eligible_groups[j])<=set(groups[j])
            if u['kind']=='sentence':assert groups[j] and eligible_groups[j]
            for t in groups[j]:
                lo,hi=enc.offsets[t]
                if hi>u['end']+1 and text[u['end']+1:hi].strip():
                    crossings.append(dict(index=i,token=t,text=text[lo:hi],unit=j))
        body_tokens=sorted(t for j,u in enumerate(units) if u['kind']=='sentence' for t in groups[j])
        candidates.append(dict(index=i,source_id=original['_id'],prompt_sha256=span['prompt_sha256'],question=q,
            documents=len(original['context']),units=units,all_groups=groups,eligible_groups=eligible_groups,
            old_keep=old[i]['keep'],all_body_tokens=body_tokens))
    assert len(candidates)==48 and sum(c['documents'] for c in candidates)==480
    write('candidates.json',dict(version=plan['version'],protocol_sha256=sha(HERE/'protocol.json'),
        source_sha256=sha(a.source),cache_sha256=sha(a.cache),tokenizer_sha256=sha(a.tokenizer),
        builder_sha256=sha(Path(__file__)),retrieval_module_sha256=sha(ROOT/'experiments/official/hotpot_retrieval_v3.py'),
        mapping_module_sha256=sha(ROOT/'experiments/official/hotpot_evidence.py'),cases=candidates),compact=True)
    # Gold is attached only after the candidate and all-cost sidecar is fixed.
    labels=[];changed=[]
    for c,original,ex in zip(candidates,matched,cache):
        units=c['units'];keys=original['supporting_facts'];unit_by_key={(u['title'],u['sentence_index']):j for j,u in enumerate(units) if u['kind']=='sentence'}
        assert set(map(tuple,keys))=={(s['title'],s['sentence_index']) for s in ex['metadata']['needle_spans']}
        spans=[]
        for title,sid in keys:
            u=units[unit_by_key[title,sid]];spans.append([u['start'],u['end']])
            oldspan=next(s['span'] for s in ex['metadata']['needle_spans'] if (s['title'],s['sentence_index'])==(title,sid))
            if oldspan!=spans[-1]:changed.append(dict(index=c['index'],title=title,sentence_index=sid,old=oldspan,restored=spans[-1]))
        gold_ids=[unit_by_key[tuple(k)] for k in keys]
        offsets=tok.encode(' '+ex['prompt'],add_special_tokens=False).offsets
        overlap=sorted(t for t in c['old_keep'] if any(offsets[t][0]<hi+1 and offsets[t][1]>lo+1 for lo,hi in spans))
        labels.append(dict(index=c['index'],official_keys=keys,official_spans=spans,
            gold_all_body_tokens=sorted(t for j in gold_ids for t in c['all_groups'][j]),
            restored_overlap_gold_old_eligible=overlap))
    assert [r['index'] for r in changed]==[8]
    write('labels.json',dict(candidate_sha256=sha(HERE/'candidates.json'),official_source_sha256=sha(a.source),cases=labels),compact=True)
    summary=dict(cases=48,documents=480,native_sentences=sum(sum(u['kind']=='sentence' for u in c['units']) for c in candidates),
        all_body_tokens=sum(len(c['all_body_tokens']) for c in candidates),
        old_eligible_body_tokens=sum(sum(len(g) for u,g in zip(c['units'],c['eligible_groups']) if u['kind']=='sentence') for c in candidates),
        official_facts=sum(len(r['official_keys']) for r in labels),gold_coordinate_changed_cases=[8],crossing_token_count=len(crossings))
    write('source_audit.json',dict(status='verified_all_48_source_maps_and_full_token_costs',summary=summary,
        coordinate_changes=changed,crossing_tokens=crossings,candidate_sha256=sha(HERE/'candidates.json'),labels_sha256=sha(HERE/'labels.json')))
    print(json.dumps(summary))

if __name__=='__main__':main()
