"""Check captured Linear outputs with the owning PyTorch operator/assertion.

FP64 CPU F.linear checks every nonidentical output coordinate using the
exact original BF16 inputs and weights, without repeating the full GEMM. The
unchanged public torch.testing.assert_close uses its actual-dtype defaults.
Full paired outputs are compared; FP64 checks concern all differing cells,
not the equal cells. This is not FA/FLA, DT or PPO correctness validation.
No model, GPU, rollout, update, method change or compensation is performed.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import torch
import torch.nn.functional as F
from torch.testing._comparison import default_tolerances


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check(actual,expected):
    result=dict(rtol_atol=default_tolerances(actual,expected))
    try:
        torch.testing.assert_close(actual,expected)
        result['status']='passed'
    except AssertionError as error:
        result.update(status='failed',error=str(error))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Preserve original check; do not repeat'
    assert (args.folder/'completed.json').exists(),'Read only a completed capture'
    torch.set_num_threads(8)
    started=time.perf_counter()
    result=dict(scope=__doc__,pid=os.getpid(),birth=psutil.Process().create_time(),
        script_sha256=sha(__file__),torch_version=torch.__version__,cases=[],
        owners=[dict(path=inspect.getsourcefile(f),sha256=sha(inspect.getsourcefile(f)),name=f.__qualname__)
                for f in (torch.nn.Linear.forward,torch.testing.assert_close,default_tolerances)],
        operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0,checkpoint_restore=0),
        production_modified=False,official_thresholds_changed=False)
    artifacts=[]
    for path in sorted(args.folder.glob('*-operators.json')):
        observation=json.loads(path.read_bytes())
        artifacts.extend(observation['artifacts'])
    assert artifacts
    for artifact in artifacts:
        path=Path(artifact['path'])
        assert sha(path)==artifact['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        op=saved['operands']
        assert saved['operation'].endswith('.base') and 'input' in op,'Use the corresponding owner for a non-Linear artifact'
        x,w,b,y=op['input'],op['weight'],op['bias'],saved['output']
        assert type(x) is torch.Tensor and type(w) is torch.Tensor
        assert x.device.type==w.device.type==y.device.type=='cpu'
        assert x.dtype==w.dtype==y.dtype==torch.bfloat16
        assert x.shape[0]==y.shape[0]==8 and w.shape==(y.shape[-1],x.shape[-1])
        examples=[];pair_checks=[];scalar_checks=[];maximum=0.0
        for row in range(4):
            assert torch.equal(x[2*row],x[2*row+1])
            a,z=y[2*row],y[2*row+1]
            pair_checks.append(check(a,z))
            expected_values=[];actual0=[];actual1=[]
            for t,c in (a!=z).nonzero().tolist():
                # Original F.linear, limited to an exact requested scalar.
                expected=F.linear(x[2*row,t].double(),w[c:c+1].double(),
                                  None if b is None else b[c:c+1].double())[0]
                lo=float(a[t,c]);hi=float(z[t,c])
                maximum=max(maximum,abs(lo-float(expected)),abs(hi-float(expected)))
                expected_values.append(expected);actual0.append(a[t,c]);actual1.append(z[t,c])
                examples.append(dict(row=row,token=t,channel=c,native_pair=[lo,hi],
                    FP64=float(expected),FP64_rounded_BF16=float(expected.to(y.dtype)),
                    native_absolute_errors=[abs(lo-float(expected)),abs(hi-float(expected))]))
            if expected_values:
                expected=torch.stack(expected_values).to(y.dtype)
                scalar_checks.append(dict(row=row,endpoints=[check(torch.stack(values),expected)
                                                           for values in (actual0,actual1)]))
        result['cases'].append(dict(artifact=artifact,operation=saved['operation'],
            shape=list(x.shape),weight_shape=list(w.shape),dtype=str(x.dtype),
            all_identical_input_pairs=True,
            all_unequal_cells_vs_FP64_rounded_BF16_checks=scalar_checks,native_pair_checks=pair_checks,
            maximum_native_absolute_error_vs_FP64_on_unequal_cells=maximum,unequal_elements=examples,
            sampled_PSS_bytes=psutil.Process().memory_full_info().pss))
        del saved,op,x,w,b,y
    result.update(status='complete',seconds=time.perf_counter()-started,cuda_initialized=torch.cuda.is_initialized(),
        sampled_terminal_PSS_bytes=psutil.Process().memory_full_info().pss,
        claims_excluded=['FA/FLA tolerance','DT accuracy','PPO correctness','whole-model tolerance','attribution repair'])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),seconds=result['seconds'],
        unequal_elements=[len(c['unequal_elements']) for c in result['cases']],
        paired_check_statuses=[[k['status'] for k in c['native_pair_checks']] for c in result['cases']])),flush=True)


if __name__=='__main__':main()
