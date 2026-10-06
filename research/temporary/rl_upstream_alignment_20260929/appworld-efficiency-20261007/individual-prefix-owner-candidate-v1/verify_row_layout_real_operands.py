"""Test-only owner injection plus real saved finite-FA layout comparisons.

The original saved-operand script executes unchanged pinned FA assertions at
coincident endpoints. Nonzero finite outputs are compared only as exact layout
transport; no invented finite-error tolerance or training-accuracy conclusion.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def differences(expected,actual):
    return {name:dict(equal=torch.equal(expected[name],actual[name]),
                      max_abs=float((expected[name].float()-actual[name].float()).abs().max()),
                      dtype=str(actual[name].dtype),shape=list(actual[name].shape))
            for name in expected}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out
    out.mkdir(exist_ok=True)
    root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    candidate=Path(__file__).parent
    data=root/'receipts/upstream-alignment-20260929/actual-dt-layer3-boundaries.pt'
    saved_script=root/'releases/c9cd147/experiments/rl/verify_saved_fa_dtypes.py'
    sources=root/'receipts/training-setup/official-kernel-tests'
    original_wrapper=root/'candidates/dt-minibatch/native-prefix/vendor_fa_finite_bf16_d256.py'
    original_library=root/'candidates/dt-minibatch/native-prefix/libfinite_cached_queries.so'
    library=candidate/'libfinite_row_query_starts.so'
    assert hashlib.sha256(saved_script.read_bytes()).hexdigest()=='7ff11d9d800a2c233b019213ae0fa9a5b603dfea51dfffabe6b9296a74aab9a9'
    assert hashlib.sha256(original_wrapper.read_bytes()).hexdigest()=='f5ea2f67af2a47b2a736545b22ff6683336be14be505060674fce6f01de3f15c'
    assert hashlib.sha256(original_library.read_bytes()).hexdigest()=='5d2af760abb2684401682029e41ac68aefc58ed2796082d435874a2b74bbcebb'
    compiled=json.loads((candidate/'build.json').read_text())
    assert hashlib.sha256(library.read_bytes()).hexdigest()==compiled['library_sha256']
    baseline=module('original_scalar_finite_owner',original_wrapper)
    newer=module('row_layout_finite_owner',candidate/'vendor_fa_finite_bf16_d256.py')
    native_layout=newer.RightPaddedLengths
    native_operation=newer.VendorFAFiniteP1BF16D256

    class RowLayout(native_layout):
        """Only change storage representation for the original real test."""
        def __init__(self,lengths,padded_length,device,**kwargs):
            start=kwargs.pop('query_start',0)
            coefficient_starts=kwargs.pop('coefficient_starts',None)
            if coefficient_starts is None:coefficient_starts=[start]*len(lengths)
            super().__init__(lengths,padded_length,device,
                coefficient_starts=coefficient_starts,query_starts=[start]*len(lengths),
                query_padded_length=padded_length-start,**kwargs)

    # Explicit test-only injection into the unchanged saved-operand verifier.
    newer.RightPaddedLengths=RowLayout
    newer.VendorFAFiniteP1BF16D256=lambda _library,_sha:native_operation(library,compiled['library_sha256'])
    sys.modules['vendor_fa_finite_bf16_d256']=newer
    sys.path.insert(0,str(saved_script.parent))
    os.environ['DT_ENVIRONMENT_JSON']=str(root/'candidates/dt-minibatch/native-prefix/environment.json')
    original_argv=sys.argv
    sys.argv=[str(saved_script),'--operands',str(data),'--sources',str(sources),
              '--output',str(out/'original-official-saved-operand-check.json')]
    print('phase=unchanged_official_assertions_real_coincident_operands',flush=True)
    runpy.run_path(str(saved_script),run_name='__main__')
    sys.argv=original_argv
    official=json.loads((out/'original-official-saved-operand-check.json').read_text())
    # Restore the unmodified candidate classes for direct explicit calls below.
    newer.RightPaddedLengths=native_layout
    newer.VendorFAFiniteP1BF16D256=native_operation
    del sys.modules['vendor_fa_finite_bf16_d256']

    print('phase=real_nonzero_finite_layout_transport',flush=True)
    saved=torch.load(data,map_location='cpu',weights_only=True,mmap=True)
    fa=saved['fa']
    ops={name:value.to('cuda') for name,value in fa['operands'].items()}
    actual_owner=baseline.VendorFAFiniteP1BF16D256(original_library,'5d2af760abb2684401682029e41ac68aefc58ed2796082d435874a2b74bbcebb')
    candidate_owner=native_operation(library,compiled['library_sha256'])
    old_layout=baseline.RightPaddedLengths(list(fa['lengths']),fa['padded_length'],'cuda',
        coefficient_starts=list(fa['coefficient_starts']),query_start=fa['query_start'])
    scalar_layout=native_layout(list(fa['lengths']),fa['padded_length'],'cuda',
        coefficient_starts=list(fa['coefficient_starts']),query_start=fa['query_start'])
    row_layout=native_layout(list(fa['lengths']),fa['padded_length'],'cuda',
        coefficient_starts=list(fa['coefficient_starts']),query_starts=[fa['query_start']]*4,
        query_padded_length=ops['q0'].shape[2])
    with torch.no_grad():
        original=actual_owner(ops,fa['scale'],old_layout)
        scalar=candidate_owner(ops,fa['scale'],scalar_layout)
        row=candidate_owner(ops,fa['scale'],row_layout)
        same=dict(candidate_scalar_vs_original=differences(original,scalar),
                  candidate_row_vs_original=differences(original,row))

        # Supplementary row-index transport only. Preserve real operand values;
        # retain only coefficients at/after each selected cut. This is NOT the
        # original training action range (433); it cannot prove full credit.
        starts=[384,448,512,576]
        coefficient_starts=[max(fa['coefficient_starts'][i],start) for i,start in enumerate(starts)]
        stride=ops['q0'].shape[2]
        packed=dict(ops)
        for name in ['q0','q1','u','lse0','lse1']:
            packed[name]=torch.zeros_like(ops[name])
            for i,start in enumerate(starts):
                offset=start-fa['query_start']
                count=fa['lengths'][i]-start
                if name.startswith('lse'):packed[name][i,:,:count]=ops[name][i,:,offset:offset+count]
                else:packed[name][i,:,:count,:]=ops[name][i,:,offset:offset+count,:]
        mixed_layout=native_layout(list(fa['lengths']),fa['padded_length'],'cuda',
            coefficient_starts=coefficient_starts,query_starts=starts,query_padded_length=stride)
        mixed=candidate_owner(packed,fa['scale'],mixed_layout)
        common_layout=baseline.RightPaddedLengths(list(fa['lengths']),fa['padded_length'],'cuda',
            coefficient_starts=coefficient_starts,query_start=fa['query_start'])
        common=actual_owner(ops,fa['scale'],common_layout)
        expected={name:torch.zeros_like(value) for name,value in mixed.items()}
        for name,value in common.items():
            for i,start in enumerate(starts):
                offset=start-fa['query_start'];count=fa['lengths'][i]-start
                if name in ['tau','center']:expected[name][i,:,:count]=value[i,:,offset:offset+count]
                else:expected[name][i,:,:count,:]=value[i,:,offset:offset+count,:]
        mixed_differences=differences(expected,mixed)
    torch.cuda.synchronize()
    report=dict(scope=__doc__,status='measured_not_deployed',observed_unix=time.time(),
        original_test_source=dict(path=str(saved_script),sha256=hashlib.sha256(saved_script.read_bytes()).hexdigest()),
        candidate_library=dict(path=str(library),sha256=compiled['library_sha256']),
        saved_operands=dict(path=str(data),bytes=data.stat().st_size,sha256='0ade21d748c0fa37bd46a08ee3c1452d6cb298dac36b0fed6ad7c7be4b9978aa',hash_source='Previously completed read-only original-artifact audit; same immutable saved artifact, mmap load'),
        actual_shapes={name:list(value.shape) for name,value in ops.items()},
        actual_dtypes={name:str(value.dtype) for name,value in ops.items()},
        original_official_coincident_check_status=official['status'],
        same_actual_complete_range=same,
        mixed_index_transport=dict(query_starts=starts,coefficient_starts=coefficient_starts,query_stride=stride,
            differences=mixed_differences,
            scope='Supplementary ABI transport on real tensors, preserving only the declared shortened coefficient ranges. Not actual heterogeneous training cuts or full-credit equivalence.'),
        resource=dict(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        model_loads=0,checkpoint_loads=0,backward_optimizer_steps=0,production_changes=0)
    (out/'real-layout-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
