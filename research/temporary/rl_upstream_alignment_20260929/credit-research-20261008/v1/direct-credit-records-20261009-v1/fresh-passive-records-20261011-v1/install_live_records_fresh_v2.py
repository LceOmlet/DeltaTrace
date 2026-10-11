"""Attach passive owner logging at the original VERL worker RPC boundary."""
import __future__
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray


ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
FORMAL = ROOT + '/runs/textcraft-dt-formal-eventwait-fresh-20261011-v1'
SOURCE = ROOT + '/receipts/direct-credit-records-20261009-v1/reward_readout.py'
BASE_SHA = '814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b'
NEW_SHA = '2609d93f91e9e16b8963ed5ca2226a00e28190005dbd51e56a214d3d35696eb2'


class RemoveRecorder(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '_record_joint_dt_batch':
            return None
        return self.generic_visit(node)


def install_worker(worker):
    import reward_readout as module
    # The DT numerical module is lazy; verify its accepted file without importing it early.
    gdn_path = Path(ROOT) / 'releases/fla-early-output-scale-20261009-v1/textcraft/qwen35_gdn_finite.py'

    assert os.getpid() in [2646857, 2648285]
    assert abs(psutil.Process().create_time() - {2646857:1791730217.11, 2648285:1791730232.84}[os.getpid()]) < .05
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == BASE_SHA
    assert hashlib.sha256(gdn_path.read_bytes()).hexdigest() == '7c06d5e0a4d6c00c13483dadd27d6389e25eee7868a05666d2d1faeaa2d62656'
    candidate = Path(SOURCE)
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == NEW_SHA
    baseline = ast.parse(Path(module.__file__).read_bytes())
    tree = ast.parse(candidate.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectActionTargetReadout')
    helper = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_record_joint_dt_batch')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'trajectories')
    selected = ast.Module(body=[helper, method], type_ignores=[])
    # Compare the full owner AST after removing precisely this passive seam.
    tree = ast.parse(candidate.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DirectActionTargetReadout')
    cls.body = [n for n in cls.body if not (
        isinstance(n, ast.FunctionDef) and n.name == '_record_joint_dt_batch')]
    tree = RemoveRecorder().visit(tree)
    assert ast.dump(tree, include_attributes=False) == ast.dump(baseline, include_attributes=False)
    owners = [o for o in worker.worker_dict.values() if getattr(o, '_is_actor', False)]
    assert len(owners) == 1
    owner = owners[0]
    # Install identical passive methods before the original owner's lazy constructor.
    # No constructor is copied, wrapped or called by this diagnostic.
    readout = module.DirectActionTargetReadout
    assert not hasattr(readout, '_diagnostic_directory')
    assert (owner.config.model.lora_rank, owner.config.model.lora_alpha) == (8, 16)
    assert owner.config.actor.ppo_micro_batch_size_per_gpu == 4
    namespace = dict(module.__dict__, __file__=str(candidate))
    exec(compile(selected, str(candidate), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    assert inspect.signature(namespace['trajectories']) == inspect.signature(readout.trajectories)
    assert namespace['trace_token_attribution'] is module.trace_token_attribution
    assert namespace['reward_event_token_credit'] is module.reward_event_token_credit
    directory = Path(FORMAL) / 'credit-records' / f'rank{owner.rank}-pid{os.getpid()}'
    directory.mkdir(parents=True, exist_ok=False)
    readout._record_joint_dt_batch = namespace['_record_joint_dt_batch']
    readout.trajectories = namespace['trajectories']
    readout._diagnostic_directory = str(directory)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=owner.rank, unix=time.time(),
                original_module_file=module.__file__, original_module_sha256=BASE_SHA,
                active_method_source=str(candidate), active_method_source_sha256=NEW_SHA,
                record_directory=str(directory), numerical_GDN_sha256=hashlib.sha256(gdn_path.read_bytes()).hexdigest(),
                numerical_version='fla-early-output-scale-20261009-v1', method_math_AST_unchanged=True,
                lora_rank=8, lora_alpha=16, per_rank_microbatch=4, patches='Passive owner recording only',
                model_calls=0, DT_calls=0, optimizer=0)


if __name__ == '__main__':
    assert abs(psutil.Process(2638949).create_time() - 1791730174.47) < .05
    actual_environment = json.loads((Path(FORMAL) / 'source.json').read_bytes())['environment']
    for name in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'MACA_PATH'):
        if name in actual_environment:
            assert os.environ.get(name) == actual_environment[name], f'Wrong formal client environment: {name}'
    source = Path(__file__)
    verified = json.loads((Path(ROOT) / 'receipts/direct-credit-records-20261009-v1/cpu-result.json').read_bytes())
    assert verified['status'] == 'passed_CPU_data_persistence_only'
    assert verified['candidate_sha256'] == NEW_SHA
    output = Path(FORMAL) / 'runtime-overrides' / 'credit-records-20261011-fresh-v2.json'
    output.parent.mkdir(exist_ok=True)
    record = dict(started_unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(), complete=False,
                  script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), formal_pid=2638949,
                  formal_birth=1791730174.47, scope=__doc__, source_commit=sys.argv[1])
    output.write_text(json.dumps(record, indent=2)+'\n')
    ray.init(address='10.200.112.15:58998', log_to_driver=False, ignore_reinit_error=False)
    try:
        named = [x for x in ray.util.list_named_actors(all_namespaces=True)
                 if 'WorkerDict_' in x['name'] and 'register_center' not in x['name']]
        assert len(named) == 2
        refs = [ray.get_actor(x['name'], namespace=x['namespace']).execute_with_func_generator.remote(install_worker)
                for x in named]
        record.update(named_actors=named)
        output.write_text(json.dumps(record, indent=2)+'\n')
        results = ray.get(refs)
        record.update(complete=True, completed_unix=time.time(), results=results)
        output.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    finally:
        ray.shutdown()
