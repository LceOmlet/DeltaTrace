"""CPU-only probe of the patched original owner against installed vLLM APIs.

Stop init_engine at its native argument object, before engine/config creation.
Exercise the original vLLM Ray RPC and WorkerWrapperBase with a CPU actor.
No model, inference, training, tolerance or default launch path is involved.
"""
import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.candidate))
    import psutil
    import ray
    from omegaconf import OmegaConf
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm.v1.executor.ray_executor import RayDistributedExecutor
    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
    from vllm.entrypoints.openai.models.protocol import BaseModelPath

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result = dict(started_unix=time.time(), helper_pid=os.getpid(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='CPU API/transport probe only. No engine, model, generation or training.',
        sources={}, checks=[])
    source = lambda path: dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    try:
        module = importlib.import_module('verl.workers.rollout.vllm_rollout.vllm_async_server')
        assert module._VLLM_015
        assert Path(module.__file__).resolve().is_relative_to(args.candidate.resolve())
        for obj in [module, AsyncEngineArgs, WorkerWrapperBase, RayDistributedExecutor,
                    module.OpenAIServingChat, module.OpenAIServingModels]:
            path = inspect.getfile(obj)
            result['sources'][path] = source(path)
        assert module.ExternalRayDistributedExecutor.collective_rpc is RayDistributedExecutor.collective_rpc
        result['checks'].append('Candidate imports installed native 0.15 types and reuses original Ray RPC by identity')

        active = json.loads((args.root/'active-training.json').read_text())
        job = next(j for j in active['jobs'] if j['task'] == 'AppWorld')
        driver = psutil.Process(job['pid'])
        assert abs(driver.create_time()-job['observed_process_created_unix']) < .02
        result['formal_identity'] = dict(pid=driver.pid, created_unix=driver.create_time())
        launch = Path(job['output'])/'launch.json'
        config = OmegaConf.load(args.candidate/'verl/trainer/config/ppo_trainer.yaml')
        for key, value in json.loads(launch.read_text())['options'].items():
            OmegaConf.update(config, key.lstrip('+'), value, force_add=True)
        native_args = []

        class ArgumentsCaptured(BaseException):
            pass

        class RecordedNativeArgs(AsyncEngineArgs):
            def create_engine_config(self, *unused, **unused_kw):
                native_args.append(self)
                raise ArgumentsCaptured

        original_args = module.AsyncEngineArgs
        module.AsyncEngineArgs = RecordedNativeArgs
        cls = module.AsyncvLLMServer.__ray_metadata__.modified_class
        server = object.__new__(cls)
        server.config = config.actor_rollout_ref
        server.vllm_dp_rank = 0
        try:
            asyncio.run(server.init_engine())
        except ArgumentsCaptured:
            pass
        finally:
            module.AsyncEngineArgs = original_args
        assert len(native_args) == 1
        parsed = native_args[0]
        names = ['model','dtype','max_model_len','max_num_seqs','max_num_batched_tokens',
                 'enable_lora','max_loras','max_lora_rank','enforce_eager',
                 'gpu_memory_utilization','enable_prefix_caching','mm_processor_cache_gb']
        result['native_engine_arguments'] = {key:getattr(parsed,key) for key in names}
        assert parsed.enable_lora and parsed.max_lora_rank == 8
        assert parsed.max_model_len == 32768
        assert parsed.max_num_seqs == config.actor_rollout_ref.rollout.max_num_seqs
        result['formal_launch'] = source(launch)
        result['checks'].append('Original init_engine constructs real native arguments with recorded model/context/resources/LoRA')

        base_paths = [BaseModelPath(name='owner-model', model_path=parsed.model)]
        inspect.signature(module.OpenAIServingModels).bind(object(),base_paths)
        inspect.signature(module.OpenAIServingChat).bind(object(),object(),'assistant',
            request_logger=None,chat_template=None,chat_template_content_format='auto')
        error = ErrorResponse(error=ErrorInfo(message='interface probe',type='BadRequestError',code=400))
        assert error.error.code == 400
        result['checks'].append('Installed serving constructor bindings and native error status match')

        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        port = next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
        ray.init(address=f'127.0.0.1:{port}',namespace=f'async-owner-interface-{os.getpid()}',ignore_reinit_error=True)

        class CPUValue:
            def echo(self, value):
                return value

        @ray.remote(num_cpus=0)
        class NativeWrapperActor:
            def __init__(self):
                self.wrapper = WorkerWrapperBase()
                self.wrapper.worker = CPUValue()
            def execute_method(self, *values, **options):
                return self.wrapper.execute_method(*values, **options)

        actor = NativeWrapperActor.remote()
        try:
            executor = object.__new__(module.ExternalRayDistributedExecutor)
            executor.workers = [actor]
            value = {'request':'native-cpu-rpc','tokens':[7,8,9]}
            assert executor.collective_rpc('echo',args=(value,),timeout=30) == [value]
            assert executor.collective_rpc('echo',args=(value,),non_block=True).result(timeout=30) == [value]
            result['checks'].append('Original native Ray RPC and WorkerWrapperBase return exact artifacts, including native future')
        finally:
            ray.kill(actor, no_restart=True)
            ray.shutdown()
        result['passed'] = True
    except BaseException as error:
        result['passed'] = False
        result['error'] = repr(error)
        raise
    finally:
        result['finished_unix'] = time.time()
        result['seconds'] = result['finished_unix']-result['started_unix']
        result['helper_pss_bytes'] = psutil.Process().memory_full_info().pss
        (out/'interfaces.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
