"""CPU-only author fixed-text/span mapping, NI0 and MH1; no model load."""
import os
os.environ.update(MACA_PATH='/opt/maca', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((HERE/'protocol.json').read_bytes())
r={'status':'running','protocol':p,'model_loads':0,'model_forwards':0,'generation_calls':0,'quality_queries':0,'cases':[]}
start=time.perf_counter()
def save():
    path=HERE/'results.partial';path.write_text(json.dumps(r,ensure_ascii=False,indent=2));path.replace(HERE/'results.json')
try:
    for name,digest in p['files_sha256'].items():assert sha((HERE/name).read_bytes())==digest
    from transformers import AutoTokenizer
    from official_fixed_text_inputs import load_author_preparer
    from official_span_mapping import load_author_span_helpers, remap_cached_example
    root=Path(p['author_root']);cp=Path(p['checkpoint'])
    for name,digest in p['span_source_sha256'].items():
        assert sha((root/name).read_bytes().replace(b'\r\n',b'\n'))==digest
    for name,digest in p['checkpoint_config_tokenizer_sha256'].items():assert sha((cp/name).read_bytes())==digest
    oldcp=Path(p['old_checkpoint'])
    r['old_tokenizer_source_sha256']={x.name:sha(x.read_bytes()) for x in oldcp.iterdir()
        if x.name in ['tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','chat_template.jinja']}
    assert 'tokenizer.json' in r['old_tokenizer_source_sha256']
    old=AutoTokenizer.from_pretrained(oldcp,local_files_only=True,trust_remote_code=False)
    new=AutoTokenizer.from_pretrained(cp,local_files_only=True,trust_remote_code=False)
    old.pad_token=old.eos_token;new.pad_token=new.eos_token
    assert old.eos_token_id==151645 and new.eos_token_id==248046
    helpers=load_author_span_helpers(root,p['span_source_sha256'])
    preparer=load_author_preparer(root,p['author_source_sha256'])
    parent_raw=(Path(p['input_parent_directory'])/'results.json').read_bytes()
    assert sha(parent_raw)==p['input_parent_sha256'];prior=json.loads(parent_raw)
    for dataset,index in p['selection']:
        raw=(root/'exp/exp2/data'/f'{dataset}.jsonl').read_bytes();assert sha(raw)==p['cache_sha256'][dataset]
        record=json.loads(raw.decode().splitlines()[index])
        # Only the author's text-only methods are called. No fake model object.
        engine=preparer.__new__(preparer);engine.tokenizer=new;engine.device='cpu'
        example,case,mapping=remap_cached_example(record,old,new,engine,helpers)
        meta=case['metadata'];meta.update(dataset=dataset,index=index)
        expected=next(x for x in prior['input_metadata'] if (x['dataset'],x['index'])==(dataset,index))
        assert {k:v for k,v in meta.items() if k!='gold_and_cached_generation_indices_remapped'}=={
            k:v for k,v in expected.items() if k!='gold_and_cached_generation_indices_remapped'}
        r['cases'].append({'dataset':dataset,'index':index,'input_metadata':meta,'mapping':mapping})
        save()
    r['status']='official_spans_remapped'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc()
finally:
    r['seconds']=time.perf_counter()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','official_span_mapping.py','official_fixed_text_inputs.py','results.json']:
            z.write(HERE/name,name)
    print(json.dumps({'status':r['status'],'cases':len(r['cases']),'error':r.get('error')}),flush=True)
