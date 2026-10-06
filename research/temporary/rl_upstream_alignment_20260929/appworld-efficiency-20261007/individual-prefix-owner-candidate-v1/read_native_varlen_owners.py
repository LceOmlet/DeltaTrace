"""Read installed FA/HF owner source bytes without importing model/GPU code."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('entry', HERE.parents[1] / 'stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def main():
    code = r'''
import ast,base64,hashlib,importlib.machinery,json,os,pathlib,sys,sysconfig
site=pathlib.Path(sysconfig.get_paths()['purelib'])
rows=[]
for name,names in [
 ('transformers/modeling_flash_attention_utils.py',['_upad_input','_get_unpad_data','_flash_attention_forward']),
 ('transformers/masking_utils.py',['create_recurrent_attention_mask','create_causal_mask']),
 ('flash_attn/flash_attn_interface.py',['_flash_attn_varlen_forward','FlashAttnVarlenFunc','flash_attn_varlen_func'])]:
 p=site/name
 if not p.exists():
  package,relative=name.split('/',1)
  owner=importlib.machinery.PathFinder.find_spec(package,sys.path)
  assert owner is not None and owner.origin is not None
  p=pathlib.Path(owner.origin).parent/relative
 raw=p.read_bytes(); source=raw.decode('utf8');tree=ast.parse(source)
 selected={node.name:ast.get_source_segment(source,node) for node in tree.body
           if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
 rows.append(dict(path=str(p),relative=name,sha256=hashlib.sha256(raw).hexdigest(),
                  bytes=len(raw),selected=selected,raw_base64=base64.b64encode(raw).decode()))
print(json.dumps(dict(scope='Installed owner source only; no Torch, FA/HF imports, model, GPU, RPC, checkpoint, or configuration changes',files=rows)))
'''
    compile(code, '<read installed owner files>', 'exec')
    script = ('source ' + entry.ENTRY + '/metax-entry.env.sh\n'
              + '"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n')
    result = subprocess.run(entry.SSH + ['bash', '-s'], input=script.encode(),
                            capture_output=True, timeout=35)
    stderr_path = HERE / 'read-installed-owners.stderr.txt'
    if stderr_path.exists() and stderr_path.read_bytes():
        previous=stderr_path.read_bytes()
        (HERE / ('read-installed-owners.stderr-' + hashlib.sha256(previous).hexdigest()[:12] + '.txt')).write_bytes(previous)
    stderr_path.write_bytes(result.stderr)
    result.check_returncode()
    report = json.loads(result.stdout)
    out = HERE / 'installed-varlen-owner-source'
    out.mkdir(exist_ok=True)
    for row in report['files']:
        raw = base64.b64decode(row.pop('raw_base64'))
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        local = out / row['relative']
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(raw)
        row['local_snapshot'] = str(local)
    (out / 'source-readonly.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(scope=report['scope'], files=[
        {k:v for k,v in row.items() if k!='selected'} for row in report['files']]), indent=2))


if __name__ == '__main__':
    main()
