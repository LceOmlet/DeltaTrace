"""Freeze extra-task input identities from the released Qwen3 paper archive.

VT h2 c3 indices0/1; HotpotQA indices0..3. Selection uses index and length only,
before any execution of this batch candidate on these tasks. These are transfer
checks, not an independent method-selection holdout or new FT paper table.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sha=lambda b:hashlib.sha256(b).hexdigest();cases=[];refs=[]
    for task,n,digest in [('vt_h2_c3',2,'6ea377cd05d5bd66ac9baf302ceea4d5246c059caa34eecfc6b2efe29f179c71'),
                          ('hotpotqa_long',4,'16893f8021639312634d9adc2118862a5f2076b2ca5fe8cc0f089a1f1add885c')]:
        path=args.paper/'raw'/(task+'.results.json.gz');raw=path.read_bytes();assert sha(raw)==digest
        decoded=gzip.decompress(raw);r=json.loads(decoded);assert r['status']=='complete' and r['family']=='qwen3'
        selected=sorted(r['cases'],key=lambda c:c['index'])[:n];assert [c['index'] for c in selected]==list(range(n))
        for c in selected:
            cases.append({k:c[k] for k in ('dataset','index','input_ids','input_sha256','prompt_length','target_length','user_positions','keep','gold')})
        refs.append({'task':task,'source_archive_sha256':digest,'source_result_sha256':sha(decoded),'indices':list(range(n))})
    report={'preparation_status':'frozen','kind':'original-author-input-transfer-check','family':'qwen3','cases':cases,'references':refs,
            'new_model_calls':0,'new_metric_calls':0,'method_selection_holdout':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n',newline='\n');print('Frozen six original author inputs; no scores used for selection.')


if __name__=='__main__':main()
