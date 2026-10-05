"""Fetch only saved native metadata and the original tokenizer template source."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location('original_stage_paths', AUDIT / 'stage_environment_entry.py')
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)
mapping = json.loads((HERE.parents[1] / 'native-minibatch-v4/native-minibatch-readout-mapping.json').read_text())
decoder = json.loads((HERE / 'native-prefix-decoding.json').read_text())
credit_path = mapping['inputs']['credits']['path']
model = decoder['tokenizer']['path']
files = {
    'original-native-collect-metadata.json': credit_path.rsplit('/', 1)[0] + '/metadata.json',
    'original-tokenizer-config.json': model + '/tokenizer_config.json',
    'original-tokenizer-chat-template.jinja': model + '/chat_template.jinja',
}
program = 'import pathlib,hashlib,json,os\nfiles=' + repr(files) + '''
out={'pid':os.getpid(),'scope':'read exact saved metadata and original tokenizer source only','files':{}}
for name,path in files.items():
 p=pathlib.Path(path)
 if p.is_file():
  b=p.read_bytes();out['files'][name]={'path':path,'exists':True,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 else:out['files'][name]={'path':path,'exists':False}
print(json.dumps(out,sort_keys=True))
'''
result = subprocess.run(STAGE.SSH + ['/opt/conda/bin/python -c ' + shlex.quote(program)], capture_output=True, text=True, check=True, timeout=60)
(HERE / 'readout-history-contract-fetch.stdout.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
receipt = json.loads(result.stdout)
for name, row in receipt['files'].items():
    if row['exists']:
        local = HERE / name
        subprocess.run(STAGE.SCP + [STAGE.SSH[-1] + ':' + row['path'], str(local)], capture_output=True, check=True, timeout=60)
        assert hashlib.sha256(local.read_bytes()).hexdigest() == row['sha256']
        row['local_path'] = str(local)
        if name == 'original-tokenizer-config.json':
            original = next(x for x in decoder['sources']['tokenizer_files'] if x['path'].endswith('/tokenizer_config.json'))
            assert row['sha256'] == original['sha256']
receipt['operations'] = {'models':0,'forwards':0,'backward':0,'sampling':0,'optimizer':0,'source_edits':0}
(HERE / 'readout-history-contract-fetch.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k: {'exists':v['exists'],'sha256':v.get('sha256'),'bytes':v.get('bytes')} for k,v in receipt['files'].items()},sort_keys=True))
