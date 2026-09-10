"""Expose only the independently verified full comparison in the repository index."""
import json
from common import HERE,ROOT,METHODS,sha

def main():
    verification=json.loads((HERE/'verification.json').read_bytes());assert verification['status']=='passed'
    for name,digest in verification['output_sha256'].items():assert sha(HERE/name)==digest
    analysis=json.loads((HERE/'analysis.json').read_bytes());rows={r['method']:r for r in analysis['primary']}
    assert analysis['new_method_cases']==2240
    names={'DT':'DeltaTrace','FT_K3':'FlashTrace K3','AttnLRP':'AttnLRP †'}
    table='\n'.join(['| Method | VT macro Recall@10% | HotpotQA Recall@10% |','| --- | ---: | ---: |',
        *[f"| {names.get(m,m)} | {100*rows[m]['vt_macro']:.2f}% | {100*rows[m]['hotpotqa_long']:.2f}% |" for m in METHODS]])
    start='<!-- frozen-all-baselines:start -->';end='<!-- frozen-all-baselines:end -->'
    section=f'''{start}
### Frozen VT and HotpotQA comparison across all seven algorithms

All five remaining baselines have been remeasured on the same **448 inputs**,
yielding **2,240 new method-case records**. DT and FT reuse their verified vectors.
The fixed VT policy reconstructs the stored generated answer and ranks positive
eligible body-token scores. The HotpotQA policy retains the entire response,
sums signed scores over native body sentences, and charges every body token in
the longest affordable ranking prefix.

{table}

VT macro equally weights four 100-case tasks; HotpotQA uses official
supporting-fact Recall over 48 cases. Their metric denominators differ.
Perturbation/REAGENT/CLP use the author's 20-segment approximation. † AttnLRP
includes a documented FP16 zero-ratio repair and lossless saved-tensor offload.
This reporting scope was chosen retrospectively after DT/FT results; it does not
establish measurement neutrality or independent holdout evidence. See the
[complete results, adjusted paired intervals, technical history and raw data](research/temporary/all_baselines_20260910/RESULTS.md)
and [independent verification](research/temporary/all_baselines_20260910/verification.json).
{end}
'''
    path=ROOT/'README.md';text=path.read_bytes().decode('utf-8')
    if start in text:
        assert text.count(start)==text.count(end)==1
        lo=text.index(start);hi=text.index(end)+len(end);text=text[:lo]+section.rstrip()+text[hi:]
    else:
        anchor='## Results\n';assert text.count(anchor)==1
        text=text.replace(anchor,anchor+'\n'+section,1)
    text=text.replace('The new protocol has not yet produced GPU quality results.',
        'For the completed frozen VT/HotpotQA baseline comparison, use the [dedicated protocol and runner](research/temporary/all_baselines_20260910/RESULTS.md).')
    path.write_bytes(text.encode('utf-8'))
    (HERE/'RUN_STATUS.md').write_text('# Complete and independently verified\n\nAll 2,240 new method-case results on the 448 frozen inputs are complete. DT/FT vectors were reused.\n\nSee [results](RESULTS.md), [verification](verification.json), and [raw records](raw).\n',encoding='utf-8',newline='\n')
    receipt=dict(status='verified_results_indexed',verification_sha256=sha(HERE/'verification.json'),readme_sha256=sha(path),
        builder_sha256=sha(HERE/'update_result_index.py'),run_status_sha256=sha(HERE/'RUN_STATUS.md'))
    (HERE/'index_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
