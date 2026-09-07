"""CPU-only tokenizer identity and exact official-cache-text roundtrip preflight."""
import os
os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = json.loads((HERE / 'protocol.json').read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha((HERE / 'study.py').read_bytes()) == p['study_sha256']
start = time.perf_counter()
r = {'status': 'running', 'protocol': p, 'files': {}, 'records': [],
     'model_loads': 0, 'model_forwards': 0, 'quality_queries': 0, 'generation_calls': 0}
try:
    checkpoint = Path(p['checkpoint'])
    for name, expected in p['official_files'].items():
        raw = (checkpoint / name).read_bytes()
        actual = {'bytes': len(raw), 'sha256': sha(raw),
                  'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}
        assert actual['bytes'] == expected['size'], name
        if 'lfs' in expected:
            assert actual['sha256'] == expected['lfs']['sha256'], name
        else:
            assert actual['git_blob_sha1'] == expected['blobId'], name
        r['files'][name] = actual
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    r['tokenizer'] = {'class': type(tokenizer).__name__, 'fast': tokenizer.is_fast,
        'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id, 'vocabulary_size': len(tokenizer),
        'transformers_version': importlib.metadata.version('transformers')}
    config = json.loads((checkpoint / 'config.json').read_text())['text_config']
    assert tokenizer.is_fast and tokenizer.eos_token_id == config['eos_token_id'] == 248044
    assert len(tokenizer) == config['vocab_size']
    assert r['tokenizer']['transformers_version'] == '5.13.0'
    for dataset, index in p['selection']:
        path = Path(p['official_cache_root']) / (dataset + '.jsonl')
        raw = path.read_bytes()
        assert sha(raw) == p['official_cache_sha256'][dataset]
        record = json.loads(raw.decode('utf-8').splitlines()[index])
        out = {'dataset': dataset, 'index': index, 'fields': {}}
        for key in ['prompt', 'target']:
            text = record[key]
            ids = tokenizer.encode(text, add_special_tokens=False)
            restored = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            assert text == restored, (dataset, index, key)
            out['fields'][key] = {'text_sha256': sha(text.encode()), 'text_characters': len(text),
                'new_token_count': len(ids), 'ids_json_sha256': sha(json.dumps(ids, separators=(',', ':')).encode()),
                'exact_text_roundtrip': True}
        out['scope'] = 'Raw official prompt and target tokenized separately only. No chat-input assembly, target boundary/gold remapping or model execution claimed; old cached token indices are not reused.'
        r['records'].append(out)
    assert 'torch' not in sys.modules
    r.update(status='official_tokenizer_identity_and_two_cache_text_roundtrips_passed',
             torch_imported=False, model_input_contract_complete=False)
except Exception:
    r.update(status='failed', error=traceback.format_exc())
    raise
finally:
    r['seconds'] = time.perf_counter() - start
    (HERE / 'results.json').write_text(json.dumps(r, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(HERE / 'review_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for n in ['study.py', 'protocol.json', 'results.json']:
            z.write(HERE / n, n)
    print(json.dumps({'status': r['status'], 'seconds': r['seconds']}), flush=True)
