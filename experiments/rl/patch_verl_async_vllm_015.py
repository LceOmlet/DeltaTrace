"""Adapt pinned VERL's existing async owner to installed vLLM 0.15 APIs.

Apply explicitly to a separate candidate tree. This does not select async
rollout, launch an engine, copy a scheduler, or change the verified sync path.
The owner imports, constructor arguments and RPC implementation are reused.
"""
from pathlib import Path

from patch_verl_environment_entry import replace_once


def patch_server(source):
    old = '''from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels'''
    new = '''from verl.utils.vllm_utils import is_version_ge

_VLLM_015 = (is_version_ge(pkg="vllm", minver="0.15.0")
             and not is_version_ge(pkg="vllm", minver="0.16.0"))
if _VLLM_015:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.v1.executor.ray_executor import RayDistributedExecutor
else:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels'''
    source = replace_once(source, old, new)
    source = replace_once(source,
        'from vllm.worker.worker_base import WorkerWrapperBase\n',
        'if _VLLM_015:\n'
        '    from vllm.v1.worker.worker_base import WorkerWrapperBase\n'
        'else:\n'
        '    from vllm.worker.worker_base import WorkerWrapperBase\n')
    # vLLM now passes non_block to collective_rpc. Its Ray implementation
    # already owns serialization, futures and timeout handling; do not copy it.
    source = replace_once(source, '    def check_health(self):\n',
        '    if _VLLM_015:\n'
        '        collective_rpc = RayDistributedExecutor.collective_rpc\n\n'
        '    def check_health(self):\n')
    source = replace_once(source,
        '            disable_mm_preprocessor_cache=True,\n',
        '            **({"mm_processor_cache_gb": 0} if _VLLM_015 else\n'
        '               {"disable_mm_preprocessor_cache": True}),\n')
    # The original sync worker already supplies these same model/resource
    # inputs. The old async server silently omitted them.
    source = replace_once(source, '        engine_args = AsyncEngineArgs(\n',
        '        lora_rank = int(self.config.model.get("lora_rank", 0))\n'
        '        lora_kwargs = ({"enable_lora": True, "max_loras": 1,\n'
        '                        "max_lora_rank": lora_rank} if lora_rank and _VLLM_015 else {})\n'
        '        engine_args = AsyncEngineArgs(\n')
    source = replace_once(source, '            model=local_path,\n',
        '            model=local_path,\n'
        '            **lora_kwargs,\n'
        '            **({"max_num_seqs": config.max_num_seqs} if _VLLM_015 else {}),\n')
    source = replace_once(source,
        '        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)\n',
        '        models = (OpenAIServingModels(self.engine, BASE_MODEL_PATHS) if _VLLM_015\n'
        '                  else OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS))\n')
    source = replace_once(source,
        '            self.engine,\n            model_config,\n            models,\n',
        '            self.engine,\n'
        '            *([models] if _VLLM_015 else [model_config, models]),\n')
    # ErrorResponse moved the status into its native ErrorInfo; preserve the
    # official response payload and use its existing status code.
    source = source.replace('generator.code',
        '(generator.error.code if _VLLM_015 else generator.code)')
    compile(source, '<patched vllm_async_server>', 'exec')
    return source


def patch_worker(source):
    old = '        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)\n'
    new = ('        self.inference_engine = (WorkerWrapperBase()\n'
           '            if (is_version_ge(pkg="vllm", minver="0.15.0")\n'
           '                and not is_version_ge(pkg="vllm", minver="0.16.0"))\n'
           '            else WorkerWrapperBase(vllm_config=self.vllm_config))\n')
    source = replace_once(source, old, new)
    compile(source, '<patched vllm_rollout_spmd>', 'exec')
    return source


def patch_sharding(source):
    old = '                self.inference_engine.llm_engine.add_lora(lora_reqest)\n'
    new = ('                # Async owns a native WorkerWrapperBase, sync owns LLM.\n'
           '                from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import WorkerWrapperBase\n'
           '                engine = (self.inference_engine\n'
           '                          if isinstance(self.inference_engine, WorkerWrapperBase)\n'
           '                          else self.inference_engine.llm_engine)\n'
           '                engine.add_lora(lora_reqest)\n')
    source = replace_once(source, old, new)
    compile(source, '<patched fsdp_vllm>', 'exec')
    return source


def apply(root):
    names = [('verl/workers/rollout/vllm_rollout/vllm_async_server.py', patch_server),
             ('verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py', patch_worker),
             ('verl/workers/sharding_manager/fsdp_vllm.py', patch_sharding)]
    for name, patch in names:
        path = Path(root)/name
        path.write_text(patch(path.read_text()), newline='\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_root', type=Path)
    apply(parser.parse_args().candidate_root)
