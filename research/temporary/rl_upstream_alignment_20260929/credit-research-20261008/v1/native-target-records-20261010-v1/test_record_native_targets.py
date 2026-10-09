"""Compare passive serialization on an existing real native B4 capture only."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OLD_SHA = '2609d93f91e9e16b8963ed5ca2226a00e28190005dbd51e56a214d3d35696eb2'
NEW_SHA = 'f0fbae610f1a8c22b65c6486cefb73eb5ff04c154178e0f822e679b600fa33bc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(x,y) for x,y in zip(left,right))
    return left == right


def helper(path):
    tree = ast.parse(path.read_bytes())
    owner = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name=='DirectActionTargetReadout')
    node = next(x for x in owner.body if isinstance(x, ast.FunctionDef) and x.name=='_record_joint_dt_batch')
    namespace = dict(torch=torch,json=json,time=time,__file__=str(path))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name]


def main():
    directory=Path(sys.argv[1]);old=ROOT/'receipts/direct-credit-records-20261009-v1/reward_readout.py'
    new=directory/'reward_readout.py'
    assert digest(old)==OLD_SHA and digest(new)==NEW_SHA
    # Exactly one data dictionary and one torch.save keyword are the entire change.
    tree=ast.parse(new.read_bytes())
    owner=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='DirectActionTargetReadout')
    node=next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name=='_record_joint_dt_batch')
    node.body=[x for x in node.body if not (isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='native_targets' for y in x.targets))]
    removed=0
    for call in ast.walk(node):
        if isinstance(call,ast.Call):
            before=len(call.keywords)
            call.keywords=[x for x in call.keywords if x.arg!='native_target_diagnostics']
            removed+=before-len(call.keywords)
    assert removed==1
    assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(old.read_bytes()),include_attributes=False)
    source=ROOT/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank1-readout-native-batch-21.pt'
    assert digest(source)=='19b18c5f4204ec4d88e40d72d350a06954f64452824a8662e95497319f89a37a'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    batch=[]
    # Same original scatter used by the existing persistence test, using its real native vector.
    for index,row in enumerate(sorted(saved['rows'],key=lambda row:row['batch_row'])):
        item=dict(row,index=row['trajectory_index'],reward=float(row['row']['dt_direct_reward']))
        item['ratios']=torch.zeros(row['width'],dtype=torch.float32)
        values=saved['native_signed'][index,row['prompt_length']:row['selected'].numel()]
        prior=row['prior'][row['suffix_positions']]
        item['ratios'][row['suffix_positions'][prior]]=values[prior].to(item['ratios'].dtype)
        batch.append(item)
    assert len(batch)==4
    frozen=[{k:v.clone() for k,v in x.items() if isinstance(v,torch.Tensor)} for x in batch]
    tokenizer=AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',local_files_only=True)
    state=SimpleNamespace(task='TextCraft',tokenizer=tokenizer)
    helper(new)(state,batch,saved['detail']) # Disabled path remains a no-op.
    records=[];timings=[]
    for label,path in [('original',old),('extended',new)]:
        out=directory/('CPU-'+label);out.mkdir(exist_ok=False)
        state._diagnostic_directory=str(out)
        started=time.perf_counter();helper(path)(state,batch,saved['detail']);timings.append(time.perf_counter()-started)
        files=list(out.glob('joint-*.pt'));assert len(files)==1
        records.append((torch.load(files[0],map_location='cpu',weights_only=True),files[0]))
    previous=records[0][0];current=records[1][0]
    native=current.pop('native_target_diagnostics')
    current['readout_source']=previous['readout_source']
    assert equal(previous,current)
    assert native and all(equal(value,saved['detail'][key]) for key,value in native.items())
    assert all(equal(old,{k:item[k] for k in old}) for old,item in zip(frozen,batch))
    assert not torch.cuda.is_initialized()
    receipt=dict(status='passed_existing_real_B4_CPU_serialization_only',unix=time.time(),
      previous_source_sha256=OLD_SHA,candidate_sha256=NEW_SHA,test_sha256=digest(Path(__file__)),
      capture_path=str(source),capture_sha256=digest(source),original_math_AST_unchanged=True,
      original_payload_exact=True,original_input_tensors_unchanged=True,disabled_path_noop=True,
      native_fields_exact=list(native),records=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for _,p in records],
      original_and_extended_write_seconds=timings,CUDA_initialized=False,model_DT_backward_optimizer_calls=0,
      scope='Tests only passive CPU persistence of already-returned native diagnostics, not numerical accuracy or attribution quality. No copied training/DT algorithm.')
    (directory/'cpu-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
