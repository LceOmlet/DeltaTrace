"""Three remaining B4 API attempts after two preserved harness failures."""
import ast
import base64
import hashlib
import json
import shlex
import zlib
from pathlib import Path

A = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
source = (A / 'fa_shared_mean_integration_20260907.py').read_text()
marker = "report.update(integration_groups=[],fresh_attributions=0,quality_queries=0)"
assert source.count(marker) == 1
source = source[:source.index(marker)]
# Install SAC inside measured(), after its zero-hook entry audit. Remove it
# before that context's zero-hook exit audit; setup/cleanup remain timed.
source = source.replace('def batch_run(examples,expected):', 'def batch_run(examples,expected,reuse=None):')
start = source.index('            before,after=capture_batch(')
end = source.index('\n        finished=True', start)
block = source[start:end]
source = source[:start] + "            reuse_context=NativeMLPSelectiveReuse(model,reuse,first_cached_layer=4) if reuse is not None else contextlib.nullcontext()\n            with reuse_context:\n" + '\n'.join('    '+line for line in block.splitlines()) + source[end:]
start = source.index('def batch_run(')
end = source.index('\nfrom qwen_signed_secant_paired_vendor_fa_gqa', start)
block = source[start:end]
block = block.replace("    try:\n        with measured() as cost:", "    cost={k:0 for k in ['native_forwards','native_forward_trajectories','extra_replay_calls','extra_replay_trajectories','peak_allocated_bytes']}\n    measurement_entered=False\n    try:\n        with measured() as cost:\n            measurement_entered=True")
block = block.replace("    finally:\n        cost['public_FA_activity']", "    except Exception:\n        report['last_attempt_original_error']=traceback.format_exc()\n        raise\n    finally:\n        cost['measurement_entered']=measurement_entered\n        cost['public_FA_activity']")
source = source[:start] + block + source[end:]
source += r'''
from native_mlp_selective_checkpoint_reuse import NativeMLPSelectiveReuse
import torch.utils.checkpoint as torch_checkpoint
assert hashlib.sha256(Path(torch_checkpoint.__file__).read_bytes()).hexdigest()==p['torch_checkpoint_source_sha256']
report.update(integration_groups=[],fresh_attributions=0,quality_queries=0)
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    selection=p['integration_groups'][0]
    examples=[];rows=[]
    for dataset,index in selection:
        data=ROOT/f'exp/exp2/data/{dataset}.jsonl'
        assert hashlib.sha256(data.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        examples.append(runner.ds_utils.load_cached(data)[index])
        rows.append(next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index)))
    assert len(rows)==4
    group={'selection':selection,'example_batch_size':4,'runs':[],'comparisons':[]}
    report['integration_groups'].append(group)
    extension=compact_extension
    assert isinstance(extension,VendorFAFiniteP1SharedMeanReuse)
    finite_capture=partial(compact_pair,finite_attention=extension)
    propagate_batch=compact_batch
    for repeat in range(2):
        pair={}
        for mode in (['shared_mean_sac32'] if repeat==0 else ['shared_mean','shared_mean_sac32']):
            assert report.get('manual_attempts',0)<3
            report['active']=[repeat,mode];save();native_method_audit(True)
            reuse={}
            # Count cache context construction, cleanup and synchronization for both methods.
            def invoke():
                torch.cuda.synchronize();tick=time.perf_counter()
                value=batch_run(examples,rows,reuse=reuse if mode=='shared_mean_sac32' else None)
                torch.cuda.synchronize()
                value['all_in_seconds']=time.perf_counter()-tick
                value['native_MLP_reuse']=reuse
                return value
            if repeat==0 and mode=='shared_mean_sac32':
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as profiler:
                    value=invoke()
                trace=HERE/'sac32_B4_warm_profile.json';profiler.export_chrome_trace(str(trace))
                report['candidate_profile']={'file':trace.name,'sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                    'scope':'Budgeted candidate warm only; not steady wall timing.'}
                del profiler
            else:
                value=invoke()
            report['fresh_attributions']+=1;pair[mode]=value
            if mode=='shared_mean_sac32':
                assert reuse['installed_source_sha256']==p['torch_checkpoint_source_sha256']
                assert reuse['saved_mm_outputs']==reuse['reused_mm_outputs']==96
                assert reuse['retained_tensor_bytes']==0
                assert len(reuse['regions'])==32 and all(r['completed'] and r['input_exact'] for r in reuse['regions'])
            group['runs'].append({'mode':mode,'repeat':repeat,'warm':repeat==0,'result':value})
            save();print('SHARED_FA_MLP_B4',repeat,mode,value['all_in_seconds'],value['peak_allocated_bytes'],flush=True)
        if repeat==0:
            continue
        old=pair['shared_mean'];new=pair['shared_mean_sac32']
        a=torch.tensor(old['signed_full_sequence'],dtype=torch.float64)
        b=torch.tensor(new['signed_full_sequence'],dtype=torch.float64)
        check={'repeat':repeat,'full_signed_exact':bool(torch.equal(a,b)),
            'max_abs_difference':float((a-b).abs().max()),'sign_flips':int(((a*b)<0).sum()),
            'endpoints_exact':old['endpoint_scores32']==new['endpoint_scores32']}
        group['comparisons'].append(check);save()
        assert check['full_signed_exact'] and check['endpoints_exact']
    assert report['fresh_attributions']==report['native_root_forwards']==report['manual_attempts']==3
    assert report['native_vjps']==report['evaluation_forwards']==report['ft_attribution_forwards']==0
    assert report['extra_layer_replay_calls']==report['extra_native_fa_attention_calls']==report['finite_FA_calls_enqueued']==108
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['active']=None;report['status']='shared_FA_native_MLP_B4_three_call_screen_complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['job_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]+[x.name for x in HERE.glob('*_profile.json')]:z.write(HERE/name,name)
'''
ast.parse(source)
study = A / 'shared_fa_mlp_B4_20260908.py'
study.write_text(source, encoding='utf-8')
p = json.loads((A / 'fa_shared_mean_integration_protocol_20260907.json').read_text())
p.update(study_sha256=sha(study.read_bytes()),
         runtime_source_parent='${ARTIFACT_ROOT}/codex_fa_shared_mean_integration_20260907_v1',
         integration_groups=[p['integration_groups'][1]],
         wait_for_pid=183703, wait_for_script='${ARTIFACT_ROOT}/codex_fa_phase_profile_20260908_v1/study.py',
         maximum_queue_seconds=60,
         purpose='Three remaining original B4 API attempts after three previous attempts: v1 failed partial baseline, v2 completed warm baseline, v2 candidate rejected before model execution. This run: candidate warm, measured baseline/candidate. No FT, generation or quality queries. One measured pair cannot establish stable speedup.',
         first_cached_layer=4, maximum_fresh_attributions=3,
         total_API_attempt_budget_including_failed_parents=6,
         total_root_call_budget_including_failed_parents=5,
         prior_failed_attempt='${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v1',
         prior_failed_counts={'native_root_forwards':1,'decoder_replays':1,'auxiliary_FA_calls':1,'finite_kernels_enqueued':0},
         second_failed_attempt='${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v2',
         second_failed_counts={'API_attempts':2,'completed_attributions':1,'native_root_forwards':1,'decoder_replays':36,'auxiliary_FA_calls':36,'finite_operators_enqueued':36},
         torch_checkpoint_source_sha256='255ec2eccfa184eb3e13ef4791955d1f6f010d8ef1f59e7a8eccb9da26161314',
         promotion='No automatic broad promotion. Stop on source/input/output failure. Report all-in measured pairs and peak capacity tradeoff. If median candidate/baseline >=1, stop expansion; no cache-layer tuning from this screen.',
         memory_policy='No old fixed72.5MB limit; record matched baseline/candidate peaks and capacity. No global N-by-N matrices.',
         timing_policy='All-in includes SAC context construction and cleanup; profiled warm excluded. One measured baseline/candidate pair only; preliminary. No cross-job multiplication of speed ratios.')
p['sources']['native_mlp_selective_checkpoint_reuse.py'] = sha((A / 'native_mlp_selective_checkpoint_reuse.py').read_bytes())
protocol = A / 'shared_fa_mlp_B4_protocol_20260908.json'
protocol.write_text(json.dumps(p, indent=2))
files = {'study.py': study.read_bytes(), 'protocol.json': protocol.read_bytes(),
         'native_mlp_selective_checkpoint_reuse.py': (A / 'native_mlp_selective_checkpoint_reuse.py').read_bytes()}
payload = {name: base64.b64encode(data).decode() for name, data in files.items()}
python = '${PYTHON}'
packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
loader = ('import base64,pathlib,subprocess,json,zlib; d=pathlib.Path("${ARTIFACT_ROOT}/codex_shared_fa_mlp_B4_20260908_v3"); '
          'd.mkdir(exist_ok=False); files=json.loads(zlib.decompress(base64.b64decode(' + repr(packed) + '))); '
          '[(d/n).write_bytes(base64.b64decode(b)) for n,b in files.items()]; '
          'f=(d/"driver.log").open("w"); j=subprocess.Popen([' + repr(python) + ',str(d/"study.py")],'
          'stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (d/"pid").write_text(str(j.pid)); '
          'print(json.dumps({"pid":j.pid,"directory":str(d)}))')
command = python + ' -c ' + shlex.quote(loader)
assert len(command) < 100000
(A / 'launch_shared_fa_mlp_B4_20260908.json').write_text(json.dumps({'cmd': command, 'timeout': 10}))
print(json.dumps({'script_sha256': sha(study.read_bytes()), 'command_bytes': len(command),
                  'maximum_attributions': 3, 'total_attempt_budget': 6, 'quality_queries': 0}))
