"""CPU audit of every frozen input and native perturbation sink partition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    env=json.loads(a.environment.read_bytes())['qwen3'];official=Path(env['official_root'])
    source=json.loads((HERE/'baseline_source_identity.json').read_bytes())
    for name,r in source['files'].items():
        assert hashlib.sha256((official/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==r['normalized_sha256']
    os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false');os.environ.setdefault('MACA_PATH','/opt/maca')
    sys.path.insert(0,str(official))
    import torch
    from transformers import AutoTokenizer
    import llm_attr,perturbation_fast,shared_utils
    from baseline_adapters import validate_target
    torch.set_num_threads(4)
    class ShapeOnlyModel:
        device=torch.device('cpu')
        def eval(self):return self
    tokenizer=AutoTokenizer.from_pretrained(env['checkpoint'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
    tracer=llm_attr.LLMAttribution(ShapeOnlyModel(),tokenizer)
    inputs=json.loads((HERE/'inputs.json').read_bytes())['cases'];checks=[];failures=[]
    for item in inputs:
        try:
            tracer.target_response(item['prompt'],item['target'])
            ids=torch.cat((tracer.prompt_ids,tracer.generation_ids),dim=1)[0].tolist()
            assert ids==item['input_ids'] and list(tracer.user_prompt_indices)==item['user_positions']
            validate_target(item['target_weights'],tracer.generation_tokens)
            assert perturbation_fast._resolve_indices_to_explain_from_stack() is None
            sentences=shared_utils.create_sentences(''.join(tracer.generation_tokens),tokenizer)
            masks=shared_utils.create_sentence_masks(tracer.generation_tokens,sentences)
            indices=[torch.where(mask==1)[0].tolist() for mask in masks]
            flat=[i for group in indices for i in group]
            assert flat==list(range(len(flat))),('Non-contiguous native sink partition',flat)
            selected=[i for i,w in enumerate(item['target_weights']) if w]
            assert set(selected)<=set(flat),('Missing target positions',sorted(set(selected)-set(flat)))
            checks.append(dict(dataset=item['dataset'],index=item['index'],input_sha256=item['input_sha256'],
                native_sink_groups=indices,covered_target_count=len(selected)))
        except Exception as error:failures.append(dict(dataset=item['dataset'],index=item['index'],error=f'{type(error).__name__}: {error}'))
    receipt_path=Path(env['checkpoint_receipt']);assert sha(receipt_path)==env['checkpoint_receipt_sha256']
    checkpoint=json.loads(receipt_path.read_bytes());weight_files=[]
    for item in checkpoint['files']:
        path=Path(env['checkpoint'])/item['name'];assert path.stat().st_size==item['bytes']
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(8*1024*1024):h.update(block)
        assert h.hexdigest()==item['sha256'];weight_files.append(dict(item,mtime_ns=path.stat().st_mtime_ns))
    receipt=dict(status='passed' if not failures else 'failed',case_count=len(checks),cases=checks,failures=failures,
        inputs_sha256=sha(HERE/'inputs.json'),source_identity_sha256=sha(HERE/'baseline_source_identity.json'),
        preflight_sha256=sha(Path(__file__)),environment_sha256=sha(a.environment),
        checkpoint_receipt_sha256=sha(receipt_path),weight_files=weight_files,
        spacy_pipeline=shared_utils.nlp.pipe_names,torch_version=torch.__version__,time=time.time())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],verified=len(checks),failures=failures),indent=2))
    assert not failures and len(checks)==448

if __name__=='__main__':main()
