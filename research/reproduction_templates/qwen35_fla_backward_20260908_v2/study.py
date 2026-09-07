"""One native FLA forward/backward on captured real inputs; no model or quality calls."""
import os
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import contextlib
import hashlib
import inspect
import json
import signal
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent
p = json.loads((HERE / 'protocol.json').read_text())
os.environ['TRITON_CACHE_DIR'] = p.get('compiler_cache', str(HERE / 'triton_cache'))
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha((HERE / 'study.py').read_bytes()) == p['study_sha256']
assert sha((HERE / 'official_fixed_text_inputs.py').read_bytes()) == p['input_runtime_sha256']
sys.path.insert(0, p['author_root'])
r = {'status': 'running', 'protocol': p, 'model_load_attempts': 0, 'model_loads': 0,
     'root_forward_attempts': 0, 'root_forwards_entered': 0, 'root_forwards_completed': 0,
     'quality_queries': 0, 'generation_calls': 0, 'attributions': 0, 'calls': []}
start = time.time()


def save():
    temp = HERE / 'results.partial'
    temp.write_text(json.dumps(r, ensure_ascii=False, separators=(',', ':')))
    temp.replace(HERE / 'results.json')


def source_receipt(scheduled=False, wy_fixed=False):
    site = Path(p['isolated_site'])
    count = 0; changed = []; h = hashlib.sha256()
    for path, digest in p['wheels'].items():
        raw = Path(path).read_bytes(); assert sha(raw) == digest
        with zipfile.ZipFile(__import__('io').BytesIO(raw)) as z:
            for name in z.namelist():
                if name.endswith('.py') and name.startswith(('fla/', 'transformers/')):
                    actual = (site / name).read_bytes(); original = z.read(name)
                    if actual != original:
                        expected = {'fla/utils.py': p['fla_mapped_utils_sha256']}
                        if scheduled: expected['fla/ops/common/chunk_delta_h.py'] = p['fla_scheduled_chunk_sha256']
                        if wy_fixed: expected['fla/ops/gated_delta_rule/wy_fast.py'] = p['wy_backported_sha256']
                        assert name in expected and sha(actual) == expected[name]
                        changed.append(name)
                    h.update(name.encode() + bytes.fromhex(sha(actual))); count += 1
    expected_changes = ({'fla/utils.py', 'fla/ops/common/chunk_delta_h.py'} if scheduled else {'fla/utils.py'})
    if wy_fixed: expected_changes.add('fla/ops/gated_delta_rule/wy_fast.py')
    assert count == 2853 and set(changed) == expected_changes
    return {'files_compared': count, 'explicit_changes': changed, 'source_tree_sha256': h.hexdigest()}


try:
    r['sources_before'] = source_receipt(scheduled=True)
    assert sha((HERE/'fla_wy_upstream_layout_backport.py').read_bytes())==p['layout_runtime_sha256']
    assert sha((Path(p['prior_failure_directory'])/'results.json').read_bytes())==p['prior_failure_raw_sha256']
    from fla_wy_upstream_layout_backport import apply_backport
    r['layout_backport']=apply_backport(p['isolated_environment'],Path(p['isolated_site'])/'fla/ops/gated_delta_rule/wy_fast.py',
                                      HERE/'upstream_wy_fast.py',HERE/'layout_backport_receipt.json')
    assert r['layout_backport']['after_sha256']==p['wy_backported_sha256']
    r['sources_during'] = source_receipt(scheduled=True,wy_fixed=True)
    root = Path(p['author_root'])
    for name, digest in p['author_source_sha256'].items():
        assert sha((root / name).read_bytes().replace(b'\r\n', b'\n')) == digest
    checkpoint = Path(p['checkpoint'])
    r['checkpoint_stats_before'] = {x.name: {'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns}
        for x in checkpoint.glob('*.safetensors')}
    assert {k: v['bytes'] for k, v in r['checkpoint_stats_before'].items()} == p['verified_shard_sizes']
    for name, digest in p['checkpoint_config_tokenizer_sha256'].items():
        assert sha((checkpoint / name).read_bytes()) == digest
    import torch
    import triton
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from fla import utils as fla_utils
    import flash_attn.flash_attn_interface as fa
    import causal_conv1d
    from official_fixed_text_inputs import prepare_fixed_text, right_pad_batch, load_author_preparer
    AuthorTextPreparer = load_author_preparer(p['author_root'], p['author_source_sha256'])
    r['text_preparation_source'] = 'Unchanged hash-verified author LLMAttribution class and two constants extracted from original ASTs; unrelated top-level imports omitted. No model implementation copied or replaced.'
    assert sha(Path(native.__file__).read_bytes()) == p['native_model_sha256']
    assert native.chunk_gated_delta_rule is chunk_gated_delta_rule
    assert native.fused_recurrent_gated_delta_rule is fused_recurrent_gated_delta_rule
    assert native.is_fast_path_available and fla_utils.device_platform == 'maca'
    assert not fla_utils.IS_NVIDIA and not hasattr(torch, 'maca')
    r['versions'] = {'torch': torch.__version__, 'triton': triton.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name(), 'real_triton_backend': triton.runtime.driver.active.get_current_target().backend}
    assert sha((HERE/'native_fla_stage_capture.py').read_bytes()) == p['stage_runtime_sha256']
    from native_fla_stage_capture import NativeFLAStageCapture
    import numpy as np
    parent=Path(p['required_parent_directory'])
    assert sha((parent/'results.json').read_bytes())==p['required_parent_raw_sha256']
    assert sha((parent/p['input_npz']).read_bytes())==p['input_npz_sha256']
    with np.load(parent/p['input_npz'],allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    names=['q','k','v','g','beta']
    dtypes={key:torch.float32 if key=='g' else torch.bfloat16 for key in names}
    cpu={key:torch.from_numpy(arrays[key].copy()).to(dtypes[key]).requires_grad_() for key in names}
    gpu={key:cpu[key].detach().to('cuda').requires_grad_() for key in names}
    cpu_seed=torch.from_numpy(arrays['output'].copy()).to(torch.bfloat16)
    seed=cpu_seed.to('cuda')
    assert cpu['q'].shape==cpu['k'].shape==cpu['v'].shape==(1,129,32,128)
    assert cpu['g'].shape==cpu['beta'].shape==(1,129,32)
    torch.set_num_threads(4)
    r.update(native_forward_attempts=0,native_forwards_completed=0,native_backward_attempts=0,native_backwards_completed=0,
             CPU_forward_attempts=0,CPU_forwards_completed=0,CPU_backward_attempts=0,CPU_backwards_completed=0,artifacts=[])
    def save_arrays(name,values):
        arrays={key:value.detach().float().cpu().numpy() for key,value in values.items()}
        file=HERE/(name+'.npz');temp=file.with_suffix('.partial')
        with temp.open('wb') as f:np.savez_compressed(f,**arrays)
        temp.replace(file)
        r['artifacts'].append({'file':file.name,'sha256':sha(file.read_bytes()),
                              'dtypes':{key:str(value.dtype) for key,value in values.items()}})
        save()
    capture=NativeFLAStageCapture(p['native_stage_source_sha256'])
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    profiler=None
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with capture:
                r['native_forward_attempts']+=1;save()
                output,_=chunk_gated_delta_rule(**gpu,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
                r['native_forwards_completed']+=1
                save_arrays('native_output',{'output':output})
                r['native_backward_attempts']+=1;save()
                # Public engine control keeps the passive Python hook on this thread.
                with torch.autograd.set_multithreading_enabled(False):
                    gradients=torch.autograd.grad(output,[gpu[key] for key in names],grad_outputs=seed)
                r['native_backwards_completed']+=1
            torch.cuda.synchronize()
        r['native_stage_counts']=capture.counts
        assert capture.counts=={'forward':1,'state_forward':2,'backward':1,'state_backward':1,'wy_backward':1}
        save_arrays('native_gradients',dict(zip(names,gradients)))
        for stage in ['forward','state_backward','wy_backward']:
            values=capture.only(stage)
            save_arrays('native_stage_'+stage,values)
        r['retained_native_stage_storage_bytes']=capture.retained_storage_bytes()
    finally:
        r['native_diagnostic_seconds']=time.perf_counter()-tick
        r['native_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        r['native_stage_counts']=capture.counts
        r['captured_stage_names']=[row['stage'] for row in capture.records]
        if profiler is not None:
            try:
                trace=HERE/'native_operator_profile.json';profiler.export_chrome_trace(str(trace))
                r['native_profile']={'file':trace.name,'sha256':sha(trace.read_bytes())}
            except Exception:r['profile_export_error']=traceback.format_exc()
        for index,row in enumerate(capture.records):
            try:save_arrays('completed_stage_'+str(index)+'_'+row['stage'],row['values'])
            except Exception:r.setdefault('partial_capture_errors',[]).append(traceback.format_exc())
        capture.clear();r['native_stage_references_after_release']=len(capture.records);save()
    for value in gradients:assert bool(torch.isfinite(value).all())
    del gradients,output,gpu,seed,profiler
    torch.cuda.synchronize()
    r['CPU_forward_attempts']+=1;save();tick=time.perf_counter()
    reference,_=native.torch_recurrent_gated_delta_rule(cpu['q'],cpu['k'],cpu['v'],cpu['g'],cpu['beta'],
        initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True)
    r['CPU_forwards_completed']+=1;save_arrays('official_CPU_output',{'output':reference})
    r['CPU_backward_attempts']+=1;save()
    reference_grad=torch.autograd.grad(reference,[cpu[key] for key in names],grad_outputs=cpu_seed)
    r['CPU_backwards_completed']+=1;save_arrays('official_CPU_gradients',dict(zip(names,reference_grad)))
    r['CPU_reference_seconds']=time.perf_counter()-tick
    for value in reference_grad:assert bool(torch.isfinite(value).all())
    r['sources_after']=source_receipt(scheduled=True,wy_fixed=True)
    assert r['sources_during']==r['sources_after']
    assert r['root_forward_attempts']==r['model_load_attempts']==0
    r['status']='native_FLA_backward_and_official_CPU_reference_executed'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc();raise
finally:
    r['job_seconds']=time.time()-start;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','official_fixed_text_inputs.py','native_fla_stage_capture.py','fla_wy_upstream_layout_backport.py','upstream_wy_fast.py','results.json']+[x.name for x in HERE.glob('layout_backport_receipt.json')]+[x.name for x in HERE.glob('*.npz')]+[x.name for x in HERE.glob('*_profile.json')]:
            z.write(HERE/name,name)
