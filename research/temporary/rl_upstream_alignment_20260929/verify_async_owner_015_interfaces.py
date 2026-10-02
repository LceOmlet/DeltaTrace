"""CPU-only probe of the patched original owner against installed vLLM APIs.

By default stop init_engine at its native arguments; --config-only validates
the actual native configuration and stops before engine creation.
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
    parser.add_argument('--config-only', action='store_true',
                        help='Validate the native engine config, stop before engine creation, and do not join Ray.')
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
        driver = None
        if not args.config_only:
            driver = psutil.Process(job['pid'])
            assert abs(driver.create_time()-job['observed_process_created_unix']) < .02
            result['formal_identity'] = dict(pid=driver.pid, created_unix=driver.create_time())
        else:
            result['scope']='Native engine configuration validation only; stops before engine/model creation and does not join Ray.'
            result['formal_identity']=dict(pid=job['pid'],created_unix=job['observed_process_created_unix'],
                                          identity_source='Original launch of the initialization-failed job')
        launch = Path(job['output'])/'launch.json'
        config = OmegaConf.load(args.candidate/'verl/trainer/config/ppo_trainer.yaml')
        for key, value in json.loads(launch.read_text())['options'].items():
            OmegaConf.update(config, key.lstrip('+'), value, force_add=True)
        if args.config_only:
            # Compose the pending launch itself. Checking only the old launch
            # did not exercise the async manager's required scheduler config.
            import runpy
            source_receipt = json.loads(Path(job['source_receipt']).read_text())
            entry = args.candidate.parent/'entry'
            os.environ.update(LOOP_ROOT=source_receipt['loop_root'],
                VERL_ROOT=str(args.candidate),DT_ROOT=source_receipt['dt_root'],
                DT_ENTRY_ROOT=str(entry))
            sys.path[:0]=[str(entry),source_receipt['loop_root']]
            pending_launcher=entry/'launch_appworld_native.py'
            pending=runpy.run_path(str(pending_launcher))
            options,sampling=pending['options_for'](Path(job['output']),resume_from=Path(job['resume_from']))
            old_options=json.loads(launch.read_text())['options']
            comparable=dict(options)
            comparable['data.custom_cls.path']=old_options['data.custom_cls.path']
            result['pending_configuration_changes']={key:dict(before=old_options.get(key),after=comparable.get(key))
                for key in old_options.keys()|comparable.keys() if old_options.get(key)!=comparable.get(key)}
            result['pending_launcher']=source(pending_launcher)
            for key,value in options.items():
                OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
        native_args = []
        native_configs = []

        class ArgumentsCaptured(BaseException):
            pass

        class RecordedNativeArgs(AsyncEngineArgs):
            def create_engine_config(self, *unused, **unused_kw):
                native_args.append(self)
                if args.config_only:
                    native_config=super().create_engine_config()
                    native_configs.append(native_config)
                    result['native_engine_config']=dict(
                        model=native_config.model_config.model,
                        dtype=str(native_config.model_config.dtype),
                        max_model_len=native_config.model_config.max_model_len,
                        swap_space=native_config.cache_config.swap_space,
                        max_num_seqs=native_config.scheduler_config.max_num_seqs,
                        max_lora_rank=native_config.lora_config.max_lora_rank)
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

        if args.config_only:
            # Execute the actual owner method, including its native-version
            # branch. Stop at the native worker entry before any device/model
            # initialization; constructing the wrapper alone missed NameError.
            from unittest.mock import patch
            spmd = importlib.import_module('verl.workers.rollout.vllm_rollout.vllm_rollout_spmd')
            assert spmd.WorkerWrapperBase is WorkerWrapperBase
            result['sources'][spmd.__file__] = source(spmd.__file__)
            result['worker_init_boundary'] = []

            class WorkerArgumentsCaptured(BaseException):
                pass

            def capture_worker_arguments(wrapper, values):
                assert type(wrapper) is WorkerWrapperBase
                assert values[0]['vllm_config'] is native_configs[0]
                assert values[0]['local_rank'] == 0
                result['worker_init_boundary'].append(dict(
                    rank=values[0]['rank'], local_rank=values[0]['local_rank'],
                    native_config_identity_preserved=True))
                raise WorkerArgumentsCaptured

            for rank in (0, 1):
                rollout = spmd.vLLMAsyncRollout()
                with patch.dict(os.environ, {'RANK': str(rank)}), \
                     patch.object(WorkerWrapperBase, 'init_worker', capture_worker_arguments):
                    try:
                        rollout.execute_method('init_worker', [dict(vllm_config=native_configs[0])])
                    except WorkerArgumentsCaptured:
                        pass
                assert result['worker_init_boundary'][-1]['rank'] == rank
            result['checks'].append('Original execute_method/init_worker reaches native WorkerWrapperBase.init_worker for both recorded ranks; no device/model initialized')
            # Exercise the original manager's thread target with the configured
            # original scheduler. No server request or engine is constructed.
            import threading
            manager_module=importlib.import_module('verl.workers.rollout.async_server')
            manager=object.__new__(manager_module.AsyncLLMServerManager)
            manager.config=config.actor_rollout_ref
            manager.scheduler_kwargs={}
            manager.server_addresses=['127.0.0.1:1']
            manager.chat_scheduler_ready=threading.Event()
            manager.chat_scheduler_loop=None
            thread=threading.Thread(target=manager._init_chat_scheduler,daemon=True)
            thread.start()
            try:
                assert manager.chat_scheduler_ready.wait(45), 'Original scheduler did not signal ready'
                assert type(manager.chat_scheduler) is manager_module.ChatCompletionScheduler
                result['sources'][manager_module.__file__]=source(manager_module.__file__)
                result['checks'].append('Original manager _init_chat_scheduler constructed the original configured ChatCompletionScheduler and signaled ready; no inference request made')
            finally:
                if manager.chat_scheduler_loop is not None:
                    manager.chat_scheduler_loop.call_soon_threadsafe(manager.chat_scheduler_loop.stop)
                thread.join(5)
                assert not thread.is_alive()
                if manager.chat_scheduler_loop is not None:
                    manager.chat_scheduler_loop.close()
            result['passed']=True
            result['checks'].append('Original create_engine_config completed; no AsyncLLM/model/Ray actor created')
            return

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
