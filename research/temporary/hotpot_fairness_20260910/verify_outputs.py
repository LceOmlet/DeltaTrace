"""Independent set/count and exhaustive-budget audit of every saved selection."""
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import unittest
from common import HERE,ROOT,sha
from test_evidence import EvidenceTests

def close(x,y):assert math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-12),(x,y)

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful() and result.testsRun==8
    analysis=json.loads((HERE/'analysis.json').read_bytes())
    for path,digest in analysis['code_sha256'].items():assert sha(ROOT/path)==digest
    candidates={r['index']:r for r in json.loads((HERE/'candidates.json').read_bytes())['cases']}
    labels={r['index']:r for r in json.loads((HERE/'labels.json').read_bytes())['cases']}
    predictions=json.loads((HERE/'selections.json').read_bytes())
    cols=('index','target_mode','method','budget_unit','budget_value')
    def key(row):return (int(row['index']),row['target_mode'],row['method'],row['budget_unit'],float(row['budget_value']))
    selected={key(r):r for r in predictions};assert len(selected)==len(predictions)==1728
    rows=list(csv.DictReader((HERE/'native_per_case.csv').open(encoding='utf-8')))
    assert len(rows)==3456 and {int(r['index']) for r in rows}==set(range(48))
    for r in rows:
        i=int(r['index']);c=candidates[i];p=selected[key(r)];units=c['units'];groups=c['groups']
        ids=p['selected_units'];keys=set(map(tuple,p['selected_keys']));gold=set(map(tuple,labels[i][r['labels']]))
        assert len(ids)==len(set(ids)) and all(units[j]['kind']=='sentence' for j in ids)
        assert keys=={(units[j]['title'],units[j]['sentence_index']) for j in ids}
        tokens=[t for j in ids for t in groups[j]]
        assert p['selected_tokens']==tokens and len(tokens)==len(set(tokens))
        assert set(tokens)<=set(c['body_tokens'])
        tp=len(keys&gold);fp=len(keys-gold);fn=len(gold-keys)
        oracle={'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn),
            'f1':2*tp/(2*tp+fp+fn),'exact_match':int(fp+fn==0),'complete_support':int(fn==0),
            'true_positive':tp,'predicted':len(keys),'gold':len(gold),'spent_tokens':len(tokens),'selected_sentences':len(ids)}
        for metric,value in oracle.items():close(r[metric],value)
        budget=int(r['budget'])
        if r['budget_unit']=='body_tokens':
            assert budget==math.ceil(float(r['budget_value'])*len(c['body_tokens'])) and len(tokens)<=budget
            close(r['unused_tokens'],budget-len(tokens))
        else:
            assert len(ids)==budget==int(float(r['budget_value']))
        gold_unit_ids=[j for j,u in enumerate(units) if (u['title'],u['sentence_index']) in gold]
        gold_tokens={t for j in gold_unit_ids for t in groups[j]}
        close(r['gold_tokens'],len(gold_tokens));close(r['token_recall'],len(set(tokens)&gold_tokens)/len(gold_tokens))
        # Exhaust all subsets of the 2..4 gold facts, independently of the greedy ceiling implementation.
        costs=[len(groups[j]) for j in gold_unit_ids]
        affordable=[len(subset) for size in range(len(costs)+1) for subset in itertools.combinations(costs,size)
                    if (sum(subset)<=budget if r['budget_unit']=='body_tokens' else len(subset)<=budget)]
        close(r['fact_recall_ceiling'],max(affordable)/len(gold))
    revised=[i for i in labels if labels[i]['official_restored']!=labels[i]['review_corrected']]
    assert revised==[36,40]
    summary=list(csv.DictReader((HERE/'native_summary.csv').open(encoding='utf-8')))
    grouping=('target_mode','labels','method','budget_unit','budget_value')
    metrics=('precision','recall','f1','exact_match','complete_support','token_recall','fact_recall_ceiling','spent_tokens','unused_tokens','selected_sentences')
    for s in summary:
        matching=[r for r in rows if all(str(r[k])==str(s[k]) for k in grouping)
                  and (s['subset']=='all48' or int(r['index']) not in (15,16,30))]
        assert len(matching)==int(s['n'])
        for metric in metrics:close(s[metric],statistics.fmean(float(r[metric]) for r in matching))
    assert len(summary)==144
    primary=list(csv.DictReader((HERE/'primary_recall10.csv').open(encoding='utf-8')));assert len(primary)==4
    for p in primary:
        pair={m:next(r for r in summary if r['target_mode']==p['target_mode'] and r['labels']==p['labels'] and r['subset']=='all48'
            and r['budget_unit']=='body_tokens' and float(r['budget_value'])==.1 and r['method']==m) for m in ('DT_target','FT_K3')}
        close(p['dt_recall'],pair['DT_target']['recall']);close(p['ft_k3_recall'],pair['FT_K3']['recall'])
        close(p['dt_minus_ft'],float(p['dt_recall'])-float(p['ft_k3_recall']))
        assert float(p['ci_low'])<0<float(p['ci_high']) and float(p['confidence'])==.9875
    frozen=json.loads((ROOT/'deltatrace/clean/sources.json').read_bytes());count=0
    for model in frozen['models'].values():
        for path,record in model['files'].items():assert sha(ROOT/path)==record['sha256'];count+=1
    assert count==27
    old=json.loads((HERE/'protocol.json').read_bytes())
    assert sha(HERE.parent/'source_v2_gpu_20260910/full_recall/analysis.json')==old['prior_full_analysis_sha256']
    artifacts=['analysis.json','candidates.json','labels.json','source_audit.json','selections.json','native_per_case.csv','diagnostic_per_case.csv',
        'primary_recall10.csv','native_summary.csv','diagnostic_summary.csv','RESULTS.md','semantic_review.json','review_source_excerpt.json','raw/answer_v2/manifest.json']
    receipt=dict(status='passed',regression_tests=8,native_score_rows=3456,independent_prediction_sets=1728,
        exhaustive_gold_subset_budget_checks=3456,native_summary_rows=144,primary_comparisons=4,
        frozen_method_files=count,prior_full_analysis_unchanged=True,review_label_changes_only=[36,40],
        verifier_sha256=sha(Path(__file__)),builder_sha256=sha(HERE/'build_report.py'),output_sha256={p:sha(HERE/p) for p in artifacts})
    (HERE/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='output_sha256'},indent=2))

if __name__=='__main__':main()
