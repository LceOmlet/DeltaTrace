"""Replay saved multi-turn requests with the completed native diagnostic owner.

No training restart, model/task download, optimizer, or formal import change.
Uses the two explicitly selected physical GPUs (default 2/3).
"""
import argparse
import ast
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SSH, SCP, remote


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices',type=int,nargs=2,default=[2,3],metavar=('FIRST','SECOND'),
        help='Two physical GPU indices for the existing B4-per-rank diagnosis; default 2 3.')
    parser.add_argument('--ledger-only',action='store_true',
        help='With current formal owner and phase/warm flags, run one original shared B4 and save its existing scalar layer ledger only.')
    parser.add_argument('--phase-only',action='store_true',
        help='Two DT calls on the largest recorded residual B4, retaining the full factual capture bank.')
    parser.add_argument('--components-only',action='store_true',
        help='Locate the first changed native operator of the recorded peak B4; no full DT replay.')
    parser.add_argument('--component-layer',type=int,
        help='With --components-only, observe this previously recorded native layer directly; skip automatic layer discovery.')
    parser.add_argument('--warm-phases',action='store_true',
        help='For --phase-only, retain cold and warm B4 calls so phase costs use warm results.')
    parser.add_argument('--hot-profile',action='store_true',
        help='With --phase-only --warm-phases, instrument the existing original/shared warm B4 calls; no extra forward.')
    parser.add_argument('--local-prefix-branch',action='store_true',
        help='With --phase-only --warm-phases, patch only the frozen owner prefix-branch method; no formal deployment.')
    parser.add_argument('--projection-inputs',action='store_true',
        help='Observe root/replay inputs of actual base MLP projections in the existing shared warm B4; no replacement computation.')
    parser.add_argument('--reverse-prefetch',action='store_true',
        help='Compare one extra shared warm B4 using the official FSDP next-layer prefetch setter only during reverse replay.')
    parser.add_argument('--native-conv-initial-states',action='store_true',
        help='Current-base same-bank cold/warm OFF and warm ON for the original public cached-convolution initial_states API; isolated owner paths only.')
    parser.add_argument('--native-conv-capacity',action='store_true',
        help='With --native-conv-initial-states, call the unchanged capacity fixture at exact32768/response512; no environment reward or actor-update claim.')
    parser.add_argument('--root-capture-inventory',action='store_true',
        help='Observe one layer at a time in the existing root forward with original capture APIs; release immediately, no retained tape or replacement computation.')
    parser.add_argument('--root-tape',action='store_true',
        help='Compare the isolated default-off owner root-capture reuse seam on the same B4 and immutable prefix bank.')
    parser.add_argument('--root-tape-cpu',action='store_true',
        help='With --root-tape, use existing pinned CPU capture transport and the original accelerated capture bodies; no formal deployment.')
    parser.add_argument('--request-offset',type=int,
        help='Select this original context-sorted B4 offset instead of the recorded numerical residual B4.')
    parser.add_argument('--root-tape-capacity',action='store_true',
        help='With --root-tape, substitute the original exact32768 fixture for the saved real B4; no actor update.')
    parser.add_argument('--root-tape-gdn0',action='store_true',
        help='With --root-tape, save actual cached-suffix GDN0 operands and execute the original FLA assertions; copies invalidate speed timing.')
    parser.add_argument('--root-tape-fa3',action='store_true',
        help='With --root-tape, save complete actual FA3 operands and execute the original FA output assertions after attribute returns; diagnostic timing only.')
    parser.add_argument('--root-tape-hot',action='store_true',
        help='Profile only the existing real root-tape warm B4, including original parameter prepare/release callbacks; no scheduling change.')
    parser.add_argument('--checkpoint',
        help='Completed global_step directory passed once to the original VERL checkpoint loader.')
    parser.add_argument('--current-formal-owner',action='store_true',
        help='Use the current frozen AppWorld entry/VERL/DT and launch options, not historical diagnostic owner overrides.')
    parser.add_argument('--base-model',action='store_true',
        help='With current formal owner and prefetch/convolution diagnosis, initialize its original base actor without loading a checkpoint; replay current owner queries on saved literal rows.')
    parser.add_argument('--native-backward',action='store_true',
        help='After current-owner DT, time one first and one warm original actor backward on the same factual B4 event target; no optimizer update.')
    args=parser.parse_args()
    if min(args.devices)<0 or len(set(args.devices))!=2:
        parser.error('--devices requires two distinct nonnegative physical GPU indices')
    if args.ledger_only and not (args.current_formal_owner and args.phase_only and args.warm_phases):
        parser.error('--ledger-only requires --current-formal-owner --phase-only --warm-phases')
    if args.ledger_only and any((args.hot_profile,args.native_backward,args.local_prefix_branch,
            args.projection_inputs,args.reverse_prefetch,args.root_capture_inventory,args.root_tape,
            args.root_tape_cpu,args.root_tape_capacity,args.root_tape_gdn0,args.root_tape_fa3,
            args.root_tape_hot,args.components_only,args.component_layer is not None)):
        parser.error('--ledger-only does not run profiling, backward, operator observers or owner variants')
    if args.base_model and not (args.current_formal_owner and (args.reverse_prefetch or args.native_conv_initial_states)
            and args.phase_only and args.warm_phases):
        parser.error('--base-model requires --current-formal-owner, --reverse-prefetch or --native-conv-initial-states, --phase-only --warm-phases')
    if args.native_conv_initial_states and not args.base_model:
        parser.error('--native-conv-initial-states requires explicit --base-model current-owner diagnosis')
    if args.native_conv_capacity and not args.native_conv_initial_states:
        parser.error('--native-conv-capacity requires --native-conv-initial-states')
    if args.native_conv_initial_states and any((args.reverse_prefetch,args.ledger_only,args.hot_profile,
            args.projection_inputs,args.local_prefix_branch,args.root_capture_inventory,args.root_tape,
            args.components_only,args.native_backward)):
        parser.error('Measure initial_states independently of prefetch, profiles, observers and other owner variants')
    if args.base_model and (args.checkpoint or args.native_backward):
        parser.error('--base-model does not load checkpoints or run actor backward')
    if args.current_formal_owner and (not (args.checkpoint or args.base_model)
            or not args.phase_only or not args.warm_phases):
        parser.error('--current-formal-owner requires --checkpoint or --base-model, plus --phase-only --warm-phases')
    if args.current_formal_owner and any((args.local_prefix_branch,args.projection_inputs,
            args.root_capture_inventory,args.root_tape,args.components_only)):
        parser.error('Current formal owner mode does not load historical scheduling/capture/source variants')
    if args.current_formal_owner and args.reverse_prefetch and not args.base_model:
        parser.error('Current-owner reverse prefetch is scoped to the explicit --base-model diagnosis')
    if args.native_backward and not args.current_formal_owner:
        parser.error('--native-backward uses --current-formal-owner and its restored checkpoint')
    if args.components_only and args.phase_only:
        parser.error('Select either the native operator diagnosis or the DT phase replay')
    if args.warm_phases and not args.phase_only:
        parser.error('--warm-phases requires --phase-only')
    if args.component_layer is not None and (not args.components_only or args.component_layer < 0):
        parser.error('--component-layer requires --components-only and a nonnegative recorded layer index')
    if args.hot_profile and not (args.phase_only and args.warm_phases):
        parser.error('--hot-profile requires --phase-only --warm-phases')
    if args.local_prefix_branch and not (args.phase_only and args.warm_phases):
        parser.error('--local-prefix-branch requires --phase-only --warm-phases')
    if args.projection_inputs and not (args.phase_only and args.warm_phases):
        parser.error('--projection-inputs requires --phase-only --warm-phases')
    if args.projection_inputs and args.hot_profile:
        parser.error('Keep copying input observations separate from device-time profiling')
    if args.reverse_prefetch and not (args.phase_only and args.warm_phases):
        parser.error('--reverse-prefetch requires --phase-only --warm-phases')
    if args.reverse_prefetch and (args.projection_inputs or args.hot_profile):
        parser.error('Measure prefetch without input copies or trace export in the timed call')
    if args.root_capture_inventory and not (args.phase_only and args.warm_phases):
        parser.error('--root-capture-inventory requires --phase-only --warm-phases')
    if args.root_capture_inventory and (args.projection_inputs or args.hot_profile or args.reverse_prefetch):
        parser.error('Observe capture storage independently of copying, trace export and scheduling candidates')
    if args.root_tape and not (args.phase_only and args.warm_phases):
        parser.error('--root-tape requires --phase-only --warm-phases')
    if args.root_tape_cpu and not args.root_tape:
        parser.error('--root-tape-cpu requires --root-tape')
    if args.request_offset is not None and (not args.phase_only or args.request_offset<0 or args.request_offset%4):
        parser.error('--request-offset requires --phase-only and an original nonnegative B4 offset')
    if args.root_tape and (args.root_capture_inventory or args.projection_inputs or args.hot_profile or args.reverse_prefetch):
        parser.error('Compare root tape without separate observers or scheduling candidates')
    if args.root_tape_capacity and not args.root_tape:
        parser.error('--root-tape-capacity requires --root-tape')
    if args.root_tape_gdn0 and (not args.root_tape or args.root_tape_capacity):
        parser.error('--root-tape-gdn0 requires the separate real-input --root-tape probe')
    if args.root_tape_fa3 and (not args.root_tape or args.root_tape_capacity or args.root_tape_gdn0):
        parser.error('--root-tape-fa3 requires a separate real-input --root-tape probe')
    if args.root_tape_hot and (not args.root_tape or args.root_tape_capacity or args.root_tape_gdn0 or args.root_tape_fa3):
        parser.error('--root-tape-hot requires a separate real-input --root-tape profile')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    parent = ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-dt-leases-20261003-v2'
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-'+(
        'local-prefix-' if args.local_prefix_branch else '')+('cpu-' if args.root_tape_cpu else '')+(
        ('components' if args.component_layer is None else 'components-layer'+str(args.component_layer)) if args.components_only else
        'ledger' if args.ledger_only else 'native-conv-capacity' if args.native_conv_capacity else 'native-conv-initial-states' if args.native_conv_initial_states else 'root-tape-hot' if args.root_tape_hot else 'root-tape-fa3' if args.root_tape_fa3 else 'root-tape-gdn0' if args.root_tape_gdn0 else 'root-tape-capacity' if args.root_tape_capacity else 'root-tape' if args.root_tape else 'root-capture-inventory' if args.root_capture_inventory else 'reverse-prefetch' if args.reverse_prefetch else 'projection-inputs' if args.projection_inputs else 'hot-phase' if args.hot_profile else 'warm-phase' if args.warm_phases else
        'phase' if args.phase_only else 'workload')+('-offset'+str(args.request_offset) if args.request_offset is not None else '')+('-20261007-' if args.native_conv_initial_states else '-20261005-' if args.root_tape_cpu or args.ledger_only else '-20261004-')+commit[:7]
    if args.current_formal_owner:
        out+='-current-'+('base' if args.base_model else args.checkpoint.rstrip('/').split('/')[-1])
    files = {
        REPO/'experiments/rl/native_prefix_leases.py': 'native_prefix_leases.py',
        REPO/'experiments/rl/test_native_prefix_leases.py': 'test_native_prefix_leases.py',
        AUDIT/'diagnose_native_prefix_leases.py': 'diagnose_native_prefix_leases.py',
        AUDIT/'diagnose_native_prefix_components.py': 'diagnose_native_prefix_components.py',
        Path(__file__): 'run_native_prefix_reuse_workload.py',
    }
    if args.checkpoint or args.current_formal_owner:
        files[AUDIT/'verify_native_prefix_artifacts.py']='verify_native_prefix_artifacts.py'
    if args.current_formal_owner:
        # These production modules come only from the frozen formal entry.
        del files[REPO/'experiments/rl/native_prefix_leases.py']
    if args.projection_inputs:
        files[AUDIT/'diagnose_native_projection_inputs.py'] = 'diagnose_native_projection_inputs.py'
    if args.reverse_prefetch:
        files[AUDIT/'native_reverse_prefetch_candidate.py'] = 'native_reverse_prefetch_candidate.py'
    if args.native_conv_initial_states:
        conv=AUDIT/'gdn-cached-conv-interface-20261007/native-initial-states-candidate-v1'
        files[conv/'prepare_isolated_owner_paths.py']='prepare_isolated_owner_paths.py'
        for relative in ('transformers/models/qwen3_5/modeling_qwen3_5.py',
                         'deltatrace/clean/qwen35/qwen35_gdn_finite.py',
                         'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'):
            files[conv/'candidate_sources'/relative]='candidate_sources/'+relative
    if args.native_conv_capacity:
        files[REPO/'experiments/rl/verify_dt_context_capacity.py']='verify_dt_context_capacity.py'
        files[AUDIT/'test_capacity_fixture_readout_interface.py']='test_capacity_fixture_readout_interface.py'
        files[AUDIT/'test_native_conv_capacity_interface.py']='test_native_conv_capacity_interface.py'
    if args.root_capture_inventory:
        for name in ('diagnose_qwen35_root_capture_inventory.py',
                     'native_root_capture_inventory_factory.py',
                     'test_root_capture_inventory_cpu_structure.py',
                     'test_native_root_inventory_decoder_owner.py',
                     'test_prefix_component_diagnostic_selection.py'):
            files[AUDIT/name] = name
    if args.root_tape:
        for name in ('prepare_native_root_tape_owner_20261004.py',
                     'native_qwen35_root_tape.py', 'test_native_root_tape_owner.py',
                     'test_native_root_inventory_decoder_owner.py'):
            files[AUDIT/name] = name
    if args.root_tape_cpu:
        for name in ('prepare_native_root_tape_cpu_owner_20261005.py',
                     'prepare_native_root_capture_transport_owner_20261005.py',
                     'test_native_root_tape_cpu_owner.py',
                     'test_native_root_capture_transport_owner.py'):
            files[AUDIT/name]=name
    if args.root_tape_capacity:
        files[AUDIT/'diagnose_native_root_tape_capacity.py']='diagnose_native_root_tape_capacity.py'
        files[REPO/'experiments/rl/verify_dt_context_capacity.py']='verify_dt_context_capacity.py'
        files[AUDIT/'test_capacity_fixture_readout_interface.py']='test_capacity_fixture_readout_interface.py'
    if args.root_tape_gdn0:
        for name in ('observe_native_gdn0_operands.py','test_observe_native_gdn0_operands.py'):
            files[AUDIT/name]=name
    if args.root_tape_fa3:
        for name in ('observe_native_fa3_operands.py','test_observe_native_fa3_operands.py'):
            files[AUDIT/name]=name
    if args.root_tape_hot:
        for name in ('native_finite_parameter_ranges.py','test_native_finite_parameter_ranges.py'):
            files[AUDIT/name]=name
    branch_reference = None
    branch_commit = branch_baseline_commit = old_method_ast = new_method_ast = None
    if args.local_prefix_branch:
        owner_path = 'experiments/rl/deltatrace_rollout.py'
        branch_commit = subprocess.check_output(
            ['git', 'log', '-1', '--format=%H', '--', owner_path], cwd=REPO, text=True).strip()
        branch_baseline_commit = subprocess.check_output(
            ['git', 'rev-parse', branch_commit+'^'], cwd=REPO, text=True).strip()
        branch_reference = subprocess.check_output(
            ['git', 'show', branch_commit+':'+owner_path], cwd=REPO)
        baseline_reference = subprocess.check_output(
            ['git', 'show', branch_baseline_commit+':'+owner_path], cwd=REPO)
        def method_ast(source):
            owner = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.ClassDef) and node.name == '_Qwen35CausalOwnerView')
            return ast.dump(next(node for node in owner.body
                                if isinstance(node, ast.FunctionDef) and node.name == 'synchronize_prefix_start'))
        old_method_ast, new_method_ast = method_ast(baseline_reference), method_ast(branch_reference)
        assert old_method_ast != new_method_ast, 'Expected the recorded prefix-branch-only change'
        files[REPO/'experiments/rl/test_prefix_branch_owner.py'] = 'test_prefix_branch_owner.py'
    bundle = AUDIT/('native-prefix-reuse-workload-'+commit[:7]+'.tar')
    with tarfile.open(bundle, 'w') as archive:
        for path, name in files.items():
            archive.add(path, arcname=name)
        if branch_reference is not None:
            info = tarfile.TarInfo('prefix_branch_owner_candidate.py')
            info.size = len(branch_reference)
            archive.addfile(info, io.BytesIO(branch_reference))
    remote(f'test ! -e {out}/prepared.json && test ! -e {out}/job.json && mkdir -p {out}\n')
    subprocess.run(SCP+[str(bundle), f'{SSH[-1]}:{out}/overlay.tar'], check=True)
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,re,shutil,subprocess,sys,time
root=pathlib.Path('@ROOT@');parent=pathlib.Path('@PARENT@');out=pathlib.Path('@OUT@')
previous=json.loads((parent/'prepared.json').read_bytes())
assert (parent/'result.json').exists(), 'Reuse only the completed original comparison'
assert not (out/'prepared.json').exists() and not (out/'job.json').exists()
physical=subprocess.check_output(['mx-smi'],text=True)
devices=@DEVICES@
assert not any(re.search(r'^\|\s+'+str(device)+r'\s+\d+\s+',physical.split('| Process:')[-1],re.M)
 for device in devices), f'GPUs{devices} are occupied'
manifest=json.loads((root/'active-training.json').read_bytes())
formal_owner=None
if @CURRENT_FORMAL_OWNER@:
 app=next(j for j in manifest['jobs'] if j['task']=='AppWorld')
 source_path=pathlib.Path(app.get('source_receipt',str(pathlib.Path(app['output'])/'source.json')))
 source=json.loads(source_path.read_bytes())
 formal_entry=pathlib.Path(app['entry']);formal_verl=pathlib.Path(app['verl_root']);formal_dt=pathlib.Path(source['dt_root'])
 source_bindings={}
 for base,bindings in ((formal_entry,source['entry_sha256']),(formal_verl,source['owner_head_sha256'])):
  for name,expected in bindings.items():
   p=base/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
   assert actual==expected,p
   source_bindings[str(p)]=actual
 prepared_path=pathlib.Path(source['prepared_receipt'])
 prepared=json.loads(prepared_path.read_bytes())
 assert hashlib.sha256(prepared_path.read_bytes()).hexdigest()==source['prepared_receipt_sha256']
 dt_bindings=source['dt_source_sha256'] if @BASE_MODEL@ else prepared['dt_source_sha256']
 for name in ('clean/qwen35/qwen35_dense_finite_runner.py','clean/qwen35/qwen35_native_prefix_artifacts.py',
              'experiments/rl/deltatrace_credit.py'):
  p=formal_dt/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
  assert actual==dt_bindings[name],p
  source_bindings[str(p)]=actual
 run_env=dict(os.environ)
 run_env.update(VERL_ROOT=str(formal_verl),DT_ROOT=str(formal_dt),DT_ENTRY_ROOT=str(formal_entry),
                LOOP_ROOT=source['loop_root'],APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
 if @BASE_MODEL@:
  p=pathlib.Path(source['resource_environment']['DT_ENVIRONMENT_JSON'])
  source_bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  run_env['DT_ENVIRONMENT_JSON']=str(p)
 formal_owner=dict(source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
  entry=str(formal_entry),verl_root=str(formal_verl),dt_root=str(formal_dt),
  source_bindings=source_bindings,prepared_path=str(prepared_path),
  prepared_sha256=source['prepared_receipt_sha256'],pythonpath=source['pythonpath'])
else:
 text=next(j for j in manifest['jobs'] if j['task']=='TextCraft')
 live=psutil.Process(text['pid'])
 run_env={k.decode():v.decode() for k,v in (x.split(b'=',1) for x in
  pathlib.Path('/proc',str(live.pid),'environ').read_bytes().split(b'\0') if b'=' in x)}
for key in ('MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR'):
 run_env.pop(key,None)
# Frozen original numerical owner, validated diagnostic and literal inputs.
for path in parent.iterdir():
 if @CURRENT_FORMAL_OWNER@:
  if path.is_file() and path.name.startswith('actual-requests-rank'):shutil.copy2(path,out/path.name)
 elif path.is_dir() and path.name=='verl-root': shutil.copytree(path,out/path.name)
 elif path.is_file() and (path.suffix=='.py' or path.name=='native-launch-options.json'
                         or path.name.startswith('actual-requests-rank')):
  shutil.copy2(path,out/path.name)
for source,expected in previous['source_files'].items():
 if source.startswith(str(parent)) and not @CURRENT_FORMAL_OWNER@:
  destination=out/pathlib.Path(source).relative_to(parent)
  assert hashlib.sha256(destination.read_bytes()).hexdigest()==expected, destination
subprocess.run(['tar','-xf',str(out/'overlay.tar'),'-C',str(out)],check=True)
if @CURRENT_FORMAL_OWNER@:
 launch_path=pathlib.Path(app['output'])/'launch.json'
 launch=json.loads(launch_path.read_bytes())
 (out/'native-launch-options.json').write_text(json.dumps(launch['options'],indent=2)+'\n')
 formal_owner['launch_path']=str(launch_path)
 formal_owner['launch_sha256']=hashlib.sha256(launch_path.read_bytes()).hexdigest()
if @ROOT_TAPE_CAPACITY@:
 p=out/'verify_native_prefix_artifacts.py';before=p.read_text()
 needle='from diagnose_native_prefix_leases import diagnose'
 assert before.count(needle)==1
 p.write_text(before.replace(needle,'from diagnose_native_root_tape_capacity import diagnose'))
root_tape_patch=None
if @ROOT_TAPE@:
 import importlib.util
 sys.path.insert(0,str(out))
 path=out/('prepare_native_root_tape_cpu_owner_20261005.py' if @ROOT_TAPE_CPU@ else 'prepare_native_root_tape_owner_20261004.py')
 spec=importlib.util.spec_from_file_location('_isolated_root_tape_preparer',path)
 preparer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=preparer;spec.loader.exec_module(preparer)
 baseline=out/'qwen35_dense_finite_runner_candidate.py'
 if @ROOT_TAPE_CPU@:
  root_tape_patch=preparer.prepare(baseline,root/'releases/c9cd147/clean/qwen35',out/'root-tape-owner',hashlib.sha256(baseline.read_bytes()).hexdigest())
 else:
  root_tape_patch=preparer.prepare(baseline,out/'root-tape-owner',hashlib.sha256(baseline.read_bytes()).hexdigest())
prefix_branch_patch=None
if @LOCAL_PREFIX_BRANCH@:
 # Keep every other byte of the actual frozen original entry. The candidate
 # reference is provenance only; the original module remains the import owner.
 p=pathlib.Path(previous['source_formal_entry'])/'deltatrace_rollout.py'
 before=p.read_bytes()
 reference=(out/'prefix_branch_owner_candidate.py').read_bytes()
 assert hashlib.sha256(reference).hexdigest()==@BRANCH_REFERENCE_SHA@
 def owner_method(source):
  owner=next(node for node in ast.parse(source).body
   if isinstance(node,ast.ClassDef) and node.name=='_Qwen35CausalOwnerView')
  return next(node for node in owner.body
   if isinstance(node,ast.FunctionDef) and node.name=='synchronize_prefix_start')
 original_method=owner_method(before);candidate_method=owner_method(reference)
 assert ast.dump(original_method)==@OLD_METHOD_AST@, 'Frozen owner is not the recorded full-value MIN method'
 assert ast.dump(candidate_method)==@NEW_METHOD_AST@, 'Candidate is not the committed branch-only method'
 original_lines=before.splitlines(keepends=True);candidate_lines=reference.splitlines(keepends=True)
 prefix=b''.join(original_lines[:original_method.lineno-1])
 suffix=b''.join(original_lines[original_method.end_lineno:])
 replacement=b''.join(candidate_lines[candidate_method.lineno-1:candidate_method.end_lineno])
 after=prefix+replacement+suffix
 assert after[:len(prefix)]==prefix and after[-len(suffix):]==suffix
 assert ast.dump(owner_method(after))==ast.dump(candidate_method)
 destination=out/'deltatrace_rollout.py';destination.write_bytes(after)
 prefix_branch_patch=dict(original_path=str(p),original_sha256=hashlib.sha256(before).hexdigest(),
  patched_path=str(destination),patched_sha256=hashlib.sha256(after).hexdigest(),
  reference_path=str(out/'prefix_branch_owner_candidate.py'),reference_sha256=hashlib.sha256(reference).hexdigest(),
  method_source_commit=@BRANCH_COMMIT@,baseline_source_commit=@BRANCH_BASELINE_COMMIT@,
  original_method_ast_sha256=hashlib.sha256(ast.dump(original_method).encode()).hexdigest(),
  patched_method_ast_sha256=hashlib.sha256(ast.dump(candidate_method).encode()).hexdigest(),
  all_other_source_bytes_preserved=True,formal_deployment=False)
if @COMPONENTS_ONLY@:
 # The completed frozen diagnostic predates the cache-field callback. Keep
 # that owner/initialization and pass its existing tensors observer only.
 p=out/'verify_native_prefix_artifacts.py'
 before=p.read_text();needle='diagnose(runner, producer, OUT, save)'
 assert before.count(needle)==1, 'Inspect the frozen diagnostic call before changing it'
 p.write_text(before.replace(needle,'diagnose(runner, producer, OUT, save, cache_tensors=tensors)'))
options=json.loads((out/'native-launch-options.json').read_bytes())
run_env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),DT_PREFIX_PROBE_ROOT=str(out),VERL_ROOT=str(out/'verl-root'),
 DT_PREFIX_DT_LEASE_DIAGNOSTIC='1',DT_PREFIX_DIAGNOSTIC_ROWS='88',
 DT_TASK='AppWorld',DT_MAX_STEPS=str(options['env.max_steps']),DT_MAX_LENGTH='32768',
 DT_SAMPLING_JSON=json.dumps(dict(temperature=options['actor_rollout_ref.rollout.temperature'],
  max_tokens=options['data.max_response_length'])),
 DT_PREFIX_OWNER_SOURCE=str(out/'qwen35_dense_finite_runner_candidate.py'),
 DT_PREFIX_ARTIFACT_SOURCE=str(out/'qwen35_native_prefix_artifacts.py'))
run_env['PYTHONPATH']=':'.join([str(out),previous['source_formal_entry'],str(out/'verl-root'),run_env.get('PYTHONPATH','')])
if @CURRENT_FORMAL_OWNER@:
 run_env['VERL_ROOT']=str(formal_verl)
 run_env['DT_ROOT']=str(formal_dt)
 run_env['PYTHONPATH']=':'.join([str(out),str(formal_dt/'clean/qwen35'),formal_owner['pythonpath']])
 run_env['DT_PREFIX_CURRENT_FORMAL_OWNER']='1'
 # The formal producer imports the current owner directly; no older source
 # override or diagnostic artifact module can enter this path.
 run_env.pop('DT_PREFIX_OWNER_SOURCE',None)
 run_env.pop('DT_PREFIX_ARTIFACT_SOURCE',None)
else:
 run_env.pop('DT_PREFIX_CURRENT_FORMAL_OWNER',None)
conv_isolated=None
if @NATIVE_CONV_INITIAL_STATES@:
 # Standard package namespace path plus a linked original DT tree. Nothing
 # installed or used by the running formal job is overwritten.
 import importlib.util
 hf_package=pathlib.Path(importlib.util.find_spec('transformers').origin).parent
 isolation=out/'conv-isolated-owners'
 subprocess.run([sys.executable,str(out/'prepare_isolated_owner_paths.py'),
  '--dt-root',str(formal_dt),'--hf-model',str(hf_package/'models/qwen3_5/modeling_qwen3_5.py'),
  '--candidate-sources',str(out/'candidate_sources'),'--output',str(isolation)],check=True)
 conv_isolated=json.loads((isolation/'isolated-owner-paths.json').read_bytes())
 run_env.update(conv_isolated['env'])
 run_env['PYTHONPATH']=':'.join([str(isolation),str(pathlib.Path(conv_isolated['isolated_dt_root'])/'clean/qwen35'),run_env['PYTHONPATH']])
else:
 run_env.pop('DT_PREFIX_NATIVE_CONV_INITIAL_STATES',None)
 run_env.pop('DT_CONV_ISOLATED_IMPORT_ROOT',None)
if @BASE_MODEL@:run_env['DT_PREFIX_BASE_MODEL_PREFETCH']='1'
else:run_env.pop('DT_PREFIX_BASE_MODEL_PREFETCH',None)
if @NATIVE_CONV_CAPACITY@:
 run_env['DT_PREFIX_NATIVE_CONV_CAPACITY']='1'
 run_env['DT_CAPACITY_FIXTURE_SOURCE']=str(out/'verify_dt_context_capacity.py')
else:run_env.pop('DT_PREFIX_NATIVE_CONV_CAPACITY',None)
if @CHECKPOINT@ is not None:run_env['DT_PREFIX_CHECKPOINT']=@CHECKPOINT@
else:run_env.pop('DT_PREFIX_CHECKPOINT',None)
if @NATIVE_BACKWARD@:run_env['DT_PREFIX_NATIVE_BACKWARD']='1'
else:run_env.pop('DT_PREFIX_NATIVE_BACKWARD',None)
if @LEDGER_ONLY@:run_env['DT_PREFIX_LEDGER_ONLY']='1'
else:run_env.pop('DT_PREFIX_LEDGER_ONLY',None)
if @ROOT_CAPTURE_INVENTORY@ or @ROOT_TAPE@:
 run_env['DT_ROOT_INVENTORY_DECODER_SOURCE']=str(root/'releases/c9cd147/clean/qwen35/qwen35_decoder_finite.py')
if @ROOT_TAPE@:
 run_env['DT_ROOT_TAPE_OWNER_SOURCE']=str(out/'qwen35_dense_finite_runner_candidate.py')
if @ROOT_TAPE_CAPACITY@:
 run_env['DT_CAPACITY_FIXTURE_SOURCE']=str(out/'verify_dt_context_capacity.py')
selected_observation=None
component_layer=@COMPONENT_LAYER@
run_env.pop('DT_PREFIX_COMPONENT_LAYER',None)
run_env.pop('DT_PREFIX_HOT_PROFILE',None)
run_env.pop('DT_PREFIX_PROJECTION_INPUTS',None)
run_env.pop('DT_PREFIX_REVERSE_PREFETCH',None)
run_env.pop('DT_PREFIX_ROOT_CAPTURE_INVENTORY',None)
run_env.pop('DT_PREFIX_ROOT_TAPE',None)
run_env.pop('DT_PREFIX_ROOT_TAPE_CPU',None)
run_env.pop('DT_PREFIX_ROOT_TAPE_GDN0',None)
run_env.pop('DT_PREFIX_ROOT_TAPE_FA3',None)
run_env.pop('DT_PREFIX_ROOT_TAPE_HOT',None)
if @BOUNDED_ONLY@:
 import torch
 maxima=[]
 peak=None
 if not @CURRENT_FORMAL_OWNER@:
  completed=root/'receipts/owner-b8-dispatch-20260930/native-prefix-reuse-workload-20261004-712795d'
  for rank in (0,1):
   p=completed/f'prefix-lease-vectors-rank{rank}.pt'
   values=torch.load(p,map_location='cpu',weights_only=False)
   original=values['original_warm']['dt_token_advantages']
   delta=(values['shared_warm']['dt_token_advantages']-original).abs()
   index=int(delta.argmax());row,column=divmod(index,delta.shape[1])
   maxima.append(dict(rank=rank,row=row,column=column,residual=float(delta[row,column]),
    vector_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  peak=max(maxima,key=lambda item:item['residual']);offset=peak['row']//4*4
 else:offset=0
 request_offset=@REQUEST_OFFSET@
 if request_offset is not None:
  offset=request_offset
 run_env.update(DT_PREFIX_PHASE_ONLY='1',DT_PREFIX_DIAGNOSTIC_OFFSET=str(offset),DT_PREFIX_DIAGNOSTIC_ROWS='4')
 if @COMPONENTS_ONLY@:
  run_env.update(DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC='1',DT_PREFIX_COMPONENT_REQUEST_INDEX=str(peak['row']))
  if component_layer is not None:
   run_env['DT_PREFIX_COMPONENT_LAYER']=str(component_layer)
 if @WARM_PHASES@:
  run_env['DT_PREFIX_PHASE_WARM']='1'
 if @HOT_PROFILE@:
  run_env['DT_PREFIX_HOT_PROFILE']='1'
 if @PROJECTION_INPUTS@:
  run_env['DT_PREFIX_PROJECTION_INPUTS']='1'
 if @REVERSE_PREFETCH@:
  run_env['DT_PREFIX_REVERSE_PREFETCH']='1'
 if @ROOT_CAPTURE_INVENTORY@:
  run_env['DT_PREFIX_ROOT_CAPTURE_INVENTORY']='1'
 if @ROOT_TAPE@:
  run_env['DT_PREFIX_ROOT_TAPE']='1'
 if @ROOT_TAPE_CPU@:
  run_env['DT_PREFIX_ROOT_TAPE_CPU']='1'
 if @ROOT_TAPE_GDN0@:
  run_env['DT_PREFIX_ROOT_TAPE_GDN0']='1'
 if @ROOT_TAPE_FA3@:
  run_env['DT_PREFIX_ROOT_TAPE_FA3']='1'
 if @ROOT_TAPE_HOT@:
  run_env['DT_PREFIX_ROOT_TAPE_HOT']='1'
 selected_observation=dict(peak=peak,all_ranks=maxima,offset=offset,component_layer=component_layer,
  scope='Same original B4 stream and full 88-row capture bank; bounded operator/phase instrumentation, not a repeat of the workload benchmark')
receipt=dict(role='Isolated original B4 DT replay; no formal deployment or acceptance of a new numerical core',
 diagnostic_commit='@COMMIT@',stager_sha256='@SHA@',devices=devices,rows_per_rank=int(run_env['DT_PREFIX_DIAGNOSTIC_ROWS']),
 original_scalar_ledger_only=@LEDGER_ONLY@,
 instrumented_hot_profile=@HOT_PROFILE@,
 projection_input_observation=@PROJECTION_INPUTS@,
 native_reverse_prefetch_candidate=@REVERSE_PREFETCH@,
 native_conv_initial_states_candidate=conv_isolated,
 native_conv_exact32768_capacity=@NATIVE_CONV_CAPACITY@,
 root_capture_inventory=@ROOT_CAPTURE_INVENTORY@,
 native_root_tape_candidate=root_tape_patch,
 root_tape_existing_cpu_transport=@ROOT_TAPE_CPU@,
 root_tape_exact32768_fixture=@ROOT_TAPE_CAPACITY@,
 root_tape_actual_gdn0_operand_check=@ROOT_TAPE_GDN0@,
 root_tape_actual_fa3_operand_check=@ROOT_TAPE_FA3@,
 root_tape_original_parameter_phase_profile=@ROOT_TAPE_HOT@,
 prefix_branch_patch=prefix_branch_patch,
 selected_observation=selected_observation,
 parent_prepared=dict(path=str(parent/'prepared.json'),sha256=hashlib.sha256((parent/'prepared.json').read_bytes()).hexdigest()),
 live_environment_source=(dict(pid=live.pid,pid_birth=live.create_time(),task='TextCraft',
  use='Recorded provisioning/cache/DT flags only; AppWorld task and sampling come from the original frozen launch')
  if not @CURRENT_FORMAL_OWNER@ else None),
 current_formal_owner=formal_owner,checkpoint=@CHECKPOINT@,base_model_initialization=@BASE_MODEL@,
 native_backward_reference=@NATIVE_BACKWARD@,
 baseline_dt_release='c9cd147',baseline_dt_reference='fc2e6c2',baseline_verl_upstream='20bd331',
 source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*out.glob('*.py'),*out.glob('root-tape-owner/*.py')]},
 literal_request_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('actual-requests-rank*.pt')},
 configuration=dict(rank=8,alpha=16,actor_microbatch=4,dt_minibatch=4,max_length=32768,
  official_launch_sha256=hashlib.sha256((out/'native-launch-options.json').read_bytes()).hexdigest(),
  event_sampling=json.loads(run_env['DT_SAMPLING_JSON']),
  numerical_environment=dict(path=run_env['DT_ENVIRONMENT_JSON'],sha256=hashlib.sha256(pathlib.Path(run_env['DT_ENVIRONMENT_JSON']).read_bytes()).hexdigest())),
 observed_before_start=time.time(),physical_before_start=physical,
 available_host_bytes=psutil.virtual_memory().available)
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
test_env=dict(run_env,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
if @CURRENT_FORMAL_OWNER@:
 # Existing CPU owner-interface tests read this explicit source path; the
 # GPU producer itself has no historical override and imports the frozen DT.
 test_env['DT_PREFIX_ARTIFACT_SOURCE']=str(formal_dt/'clean/qwen35/qwen35_native_prefix_artifacts.py')
if @ROOT_TAPE_FA3@:
 test_env['DT_OFFICIAL_FA_TEST_SOURCE']=str(pathlib.Path('@ROOT@')/'receipts/training-setup/official-kernel-tests/test_flash_attn_v263.py')
if @ROOT_TAPE_HOT@:
 test_env['DT_FINITE_PARAMETER_RANGE_OWNER_SOURCE']=str(out/'deltatrace_rollout.py') if @LOCAL_PREFIX_BRANCH@ else str(pathlib.Path(previous['source_formal_entry'])/'deltatrace_rollout.py')
with (out/'cpu-tests.log').open('wb') as log:
 test_paths=[str(out/'test_native_prefix_leases.py')]
 if @LOCAL_PREFIX_BRANCH@:test_paths.append(str(out/'test_prefix_branch_owner.py'))
 if @ROOT_CAPTURE_INVENTORY@:
  test_paths.extend(str(out/name) for name in ('test_root_capture_inventory_cpu_structure.py',
   'test_native_root_inventory_decoder_owner.py','test_prefix_component_diagnostic_selection.py'))
 if @ROOT_TAPE@:test_paths.append(str(out/'test_native_root_tape_owner.py'))
 if @ROOT_TAPE_CPU@:
  test_paths.extend(str(out/name) for name in ('test_native_root_tape_cpu_owner.py','test_native_root_capture_transport_owner.py'))
 if @ROOT_TAPE_CAPACITY@:test_paths.append(str(out/'test_capacity_fixture_readout_interface.py'))
 if @NATIVE_CONV_CAPACITY@:
  test_paths.extend(str(out/name) for name in ('test_capacity_fixture_readout_interface.py',
   'test_native_conv_capacity_interface.py'))
 if @ROOT_TAPE_GDN0@:test_paths.append(str(out/'test_observe_native_gdn0_operands.py'))
 if @ROOT_TAPE_FA3@:test_paths.append(str(out/'test_observe_native_fa3_operands.py'))
 if @ROOT_TAPE_HOT@:test_paths.append(str(out/'test_native_finite_parameter_ranges.py'))
 selection='not moved_artifact'
 if @ROOT_CAPTURE_INVENTORY@:
  # The original Git-byte assertion ran locally; the frozen remote receipt is
  # not a Git checkout. Other diagnostic/actual-owner CPU contracts still run.
  selection+=' and not original_reference_selection_and_assertion_execution_are_unchanged'
 p=subprocess.run([run_env['VENV_PYTHON'],'-m','pytest',*test_paths,
  '-k',selection,'-q','--junitxml='+str(out/'cpu-tests.xml')],env=test_env,cwd=out,
  stdout=log,stderr=subprocess.STDOUT)
receipt['cpu_tests_returncode']=p.returncode
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert p.returncode==0, (out/'cpu-tests.log').read_text()
with (out/'probe.log').open('wb') as log:
 p=subprocess.Popen([run_env['VENV_PYTHON'],'-u',str(out/'verify_native_prefix_artifacts.py')],
  cwd=out,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt.update(pid=p.pid,pid_birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),log=str(out/'probe.log'))
(out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(out=str(out),pid=p.pid,pid_birth=receipt['pid_birth'],cpu_tests_returncode=0,devices=devices)))
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@PARENT@',parent)
        .replace('@OUT@',out).replace('@COMMIT@',commit).replace('@PHASE_ONLY@',str(args.phase_only))
        .replace('@BOUNDED_ONLY@',str(args.phase_only or args.components_only))
        .replace('@COMPONENTS_ONLY@',str(args.components_only)).replace('@WARM_PHASES@',str(args.warm_phases))
        .replace('@COMPONENT_LAYER@',repr(args.component_layer))
        .replace('@HOT_PROFILE@',str(args.hot_profile))
        .replace('@PROJECTION_INPUTS@',str(args.projection_inputs))
        .replace('@REVERSE_PREFETCH@',str(args.reverse_prefetch))
        .replace('@NATIVE_CONV_INITIAL_STATES@',str(args.native_conv_initial_states))
        .replace('@NATIVE_CONV_CAPACITY@',str(args.native_conv_capacity))
        .replace('@ROOT_CAPTURE_INVENTORY@',str(args.root_capture_inventory))
        .replace('@ROOT_TAPE@',str(args.root_tape))
        .replace('@ROOT_TAPE_CPU@',str(args.root_tape_cpu))
        .replace('@REQUEST_OFFSET@',repr(args.request_offset))
        .replace('@ROOT_TAPE_CAPACITY@',str(args.root_tape_capacity))
        .replace('@ROOT_TAPE_GDN0@',str(args.root_tape_gdn0))
        .replace('@ROOT_TAPE_FA3@',str(args.root_tape_fa3))
        .replace('@ROOT_TAPE_HOT@',str(args.root_tape_hot))
        .replace('@LOCAL_PREFIX_BRANCH@',str(args.local_prefix_branch))
        .replace('@CURRENT_FORMAL_OWNER@',str(args.current_formal_owner))
        .replace('@BASE_MODEL@',str(args.base_model))
        .replace('@DEVICES@',repr(args.devices)).replace('@LEDGER_ONLY@',str(args.ledger_only))
        .replace('@CHECKPOINT@',repr(args.checkpoint)).replace('@NATIVE_BACKWARD@',str(args.native_backward))
        .replace('@BRANCH_REFERENCE_SHA@',repr(hashlib.sha256(branch_reference).hexdigest()) if branch_reference is not None else 'None')
        .replace('@BRANCH_COMMIT@',repr(branch_commit)).replace('@BRANCH_BASELINE_COMMIT@',repr(branch_baseline_commit))
        .replace('@OLD_METHOD_AST@',repr(old_method_ast)).replace('@NEW_METHOD_AST@',repr(new_method_ast))
        .replace('@SHA@',hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
