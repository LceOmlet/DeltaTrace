"""Read the measured owners and saved score schema, without loading a model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ENTRY, SSH


def main():
    measured = json.loads((HERE.parents[3].parent / 'experiments/rl/results_credit_layer_localization_20261008.json').read_bytes())
    # Recorded imported modules, not a similarly named preliminary checkout.
    bindings = {}
    for task, result in measured['tasks'].items():
        bindings[task] = dict(source=result['launch']['source_path'],
            source_sha256=result['launch']['source_sha256'],
            finite=result['ranks']['0']['actual_finite_owner'])
    body = 'BINDINGS=' + repr(bindings) + '\n' + r'''
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import psutil
import torch
torch.set_num_threads(1)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
out = dict(unix=time.time(), tasks={}, scope='CPU source/schema read only, no candidate or numerical test')
base_path = sys.path.copy()
for task, binding in BINDINGS.items():
    source_path = Path(binding['source'])
    assert sha(source_path) == binding['source_sha256']
    source = json.loads(source_path.read_bytes())
    os.environ.update(source['environment'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path(os.environ['DT_ROOT']).resolve()
    env_path = Path(os.environ.get('DT_ENVIRONMENT_JSON', root/'environment.json'))
    env = json.loads(env_path.read_bytes())['qwen35']
    sys.path[:] = [str(root), os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],
        str(root/'clean/qwen35'), *source['pythonpath'].split(':'), env['ft_extension_root'], *base_path]
    files = []
    for name in ('qwen35_decoder_finite', 'vendor_fa_finite_bf16_d256',
                 'qwen35_answer_finite', 'compiled_logprob_seed', 'compiled_finite_rules'):
        spec = importlib.util.find_spec(name)
        assert spec is not None, name
        path = Path(spec.origin)
        files.append(dict(module=name, path=str(path), resolved_path=str(path.resolve()),
                          sha256=sha(path), text=path.read_text()))
    decoder = files[0]
    assert decoder['sha256'] == binding['finite']['decoder_sha256'], (task, decoder)
    original = Path(binding['finite']['decoder_path'])
    assert sha(original) == decoder['sha256']
    saved_path = next(iter(sorted((Path(binding['source']).parents[4]/'receipts'/
        'direct-target-prefix-runtime-20261007-v1'/(task+'-first-dt')).glob('rank*-readout-native-batch-*.pt'))))
    saved = torch.load(saved_path, map_location='cpu', weights_only=False)
    detail = saved['detail']
    library = Path(env['finite_library'])
    out['tasks'][task] = dict(source_sha256=binding['source_sha256'], DT_ROOT=str(root),
        environment_json=dict(path=str(env_path), sha256=sha(env_path)), files=files,
        recorded_decoder_match=True,
        finite_library=dict(path=str(library), sha256=sha(library), expected_sha256=env['finite_library_sha256']),
        capture_schema=dict(path=str(saved_path), sha256=sha(saved_path), keys=list(saved),
            detail_keys=list(detail), first_sample=detail.get('per_sample', [None])[0]))
    assert out['tasks'][task]['finite_library']['sha256'] == env['finite_library_sha256']
    del saved
public = importlib.util.find_spec('flash_attn')
assert public is not None
interface = Path(public.origin).parent/'flash_attn_interface.py'
out['native_FA_public_interface'] = dict(path=str(interface),sha256=sha(interface),text=interface.read_text())
hold = Path(BINDINGS['textcraft']['source']).parents[4]/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
out['formal_textcraft'] = dict(pid=2833207,expected_birth=1791370325.16,
    actual_birth=psutil.Process(2833207).create_time(),
    release_exists=[(hold/('rank'+str(rank)+'-release-update')).exists() for rank in (0,1)])
assert not torch.cuda.is_initialized()
out['cuda_initialized'] = False
out['model_calls'] = out['DT_calls'] = out['optimizer_steps'] = 0
print(json.dumps(out, ensure_ascii=False))
'''
    command = 'set -eu\nsource ' + ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n'
    (HERE/'attention-owner-command.sh').write_text(command, encoding='utf-8', newline='\n')
    run = subprocess.run(SSH+['bash', '-s'], input=command.encode(), capture_output=True, timeout=60)
    (HERE/'attention-owner.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    out = json.loads(run.stdout)
    out['inspector_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'attention-owner-sources.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cuda_initialized=out['cuda_initialized'], tasks={task:dict(
        recorded_decoder_match=value['recorded_decoder_match'],
        files=[{k:v for k,v in row.items() if k!='text'} for row in value['files']],
        finite_library=value['finite_library'], capture_schema=value['capture_schema'])
        for task,value in out['tasks'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
