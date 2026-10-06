"""Import only the prepared canonical interfaces, with all CUDA devices hidden."""
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

OUT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('MACA_VISIBLE_DEVICES') == '-1'
    prepared = json.loads((OUT/'prepared.json').read_bytes())
    assert sha(OUT/'prepared.json') == '6c65c8c17f80575cbb643505e06cd8e6c8150d90ccdb23c0d49d4320aace2103'
    formal = json.loads(Path(prepared['formal_source']['path']).read_bytes())
    assert sha(prepared['formal_source']['path']) == prepared['formal_source']['sha256']
    dt = Path(prepared['dt_root'])
    entry = Path(prepared['entry'])
    numerical = json.loads(Path(prepared['environment']['path']).read_bytes())['qwen35']
    os.environ.update(numerical.get('runtime_environment', {}))
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('MACA_VISIBLE_DEVICES') == '-1'
    formal_entry = str(Path(prepared['import_resolution']['reward_readout']['resolved_path']).parent)
    inherited = [p.replace(formal['dt_root'],str(dt),1)
        if p==formal['dt_root'] or p.startswith(formal['dt_root']+'/')
        else p.replace(formal_entry,str(entry),1)
        for p in formal['pythonpath'].split(':')]
    # Same paths prepended by the unchanged producer; entry precedes inherited entry.
    search = [str(dt), numerical['official_root'], str(dt/'clean/qwen35'), str(entry), *inherited]
    sys.path[:0] = search
    import torch
    initialized_before = torch.cuda.is_initialized()
    assert not initialized_before
    started = time.time()
    imported = {}
    for name, expected in prepared['import_resolution'].items():
        module = importlib.import_module(name)
        path = Path(inspect.getsourcefile(module))
        assert path.resolve() == Path(expected['resolved_path']).resolve(), (name, str(path))
        assert sha(path) == expected['sha256'], name
        imported[name] = dict(path=str(path), resolved_path=str(path.resolve()), sha256=sha(path))
    factory = importlib.import_module('profiles.official')
    profile = importlib.import_module('profiles.qwen35_gdn_symmetric')
    capture = importlib.import_module('accelerated.qwen35.qwen35_code_local_capture')
    for name,module in [('profiles/official.py',factory), ('profiles/qwen35_gdn_symmetric.py',profile),
                        ('accelerated/qwen35/qwen35_code_local_capture.py',capture)]:
        path = Path(inspect.getsourcefile(module))
        assert path.resolve() == (dt/name).resolve()
        assert sha(path) == prepared['unchanged_factories'][name]['sha256']
        imported[name] = dict(path=str(path),resolved_path=str(path.resolve()),sha256=sha(path))
    leases = importlib.import_module('native_prefix_leases')
    producer = importlib.import_module('deltatrace_rollout')
    original_factory = leases.prepare_native_prefix_leases
    producer_tree = ast.parse(Path(producer.__file__).read_bytes())
    branch = next(n for n in ast.walk(producer_tree) if isinstance(n,ast.If) and n.lineno==396)
    branch_code = compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                          str(producer.__file__)+'::callback-configuration-only', 'exec')
    interface_cases = []
    for env in ({}, {'individual_prefixes':True}, {'boundary_row_storage':True},
                {'individual_prefixes':True,'boundary_row_storage':True}):
        instance = SimpleNamespace(readout_options={'prefix_lease_factory':original_factory})
        scope = dict(env=env,self=instance,prepare_native_prefix_leases=original_factory)
        exec(branch_code,scope)
        configured = instance.readout_options['prefix_lease_factory']
        if not env:
            assert configured is original_factory
            kws = {}
        else:
            assert configured.func is original_factory
            kws = configured.keywords
            assert kws == {'individual_prefixes':env.get('individual_prefixes',False),
                           'boundary_row_storage':env.get('boundary_row_storage',False)}
        inspect.signature(original_factory).bind_partial(None, [],minibatch_size=4,eos_token_id=248046,**kws)
        interface_cases.append(dict(env=env,owner_kwargs=kws,owner_callable_same=True,called=False))
    initialized_after = torch.cuda.is_initialized()
    assert not initialized_after
    assert sha(prepared['formal_source']['path']) == prepared['formal_source']['sha256']
    mem = dict(line.split(':',1) for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:] if ':' in line)
    result = dict(status='CPU_import_interface_passed_prepared_only',pid=os.getpid(),observed_unix=time.time(),
        imports_seconds=time.time()-started,prepared=dict(path=str(OUT/'prepared.json'),sha256=sha(OUT/'prepared.json')),
        actual_imported_sources=imported,factory=dict(path=inspect.getsourcefile(factory.make_qwen35_runner),
        sha256=sha(inspect.getsourcefile(factory.make_qwen35_runner)),called=False),
        callback_configuration_cases=interface_cases,owner_signature=str(inspect.signature(original_factory)),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],MACA_VISIBLE_DEVICES=os.environ['MACA_VISIBLE_DEVICES'],
        cuda_initialized_before=initialized_before,cuda_initialized_after=initialized_after,
        PSS_bytes=int(mem['Pss'].split()[0])*1024,RSS_bytes=int(mem['Rss'].split()[0])*1024,
        model_instantiated=False,DT_called=False,GPU=False,checkpoint=False,production_changed=False,
        numerical_acceptance=False,scope='Real original imports and callback owner parameter binding only. No factory, model, capture, finite, lease or reward computation executed.')
    with (OUT/'actual-imports-cpu.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],receipt=str(OUT/'actual-imports-cpu.json'),
        sha256=sha(OUT/'actual-imports-cpu.json'),cuda_initialized_after=initialized_after,
        imports=len(imported),PSS_bytes=result['PSS_bytes'],RSS_bytes=result['RSS_bytes'])))


if __name__ == '__main__':
    main()
