"""Adapt pinned VERL's existing async owner to installed vLLM 0.15 APIs.

Apply explicitly to a separate candidate tree. This does not select async
rollout, launch an engine, copy a scheduler, or change the verified sync path.
The owner imports, constructor arguments and RPC implementation are reused.
"""
from pathlib import Path
from textwrap import dedent, indent

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
        '        self.owner_sampling_params = SamplingParams(**kwargs)\n'
        '        engine_args = AsyncEngineArgs(\n')
    source = replace_once(source, '            model=local_path,\n',
        '            model=local_path,\n'
        '            **lora_kwargs,\n'
        '            **({"max_num_seqs": config.max_num_seqs} if _VLLM_015 else {}),\n')
    source = replace_once(source, '            seed=self.vllm_dp_rank,\n',
        '            seed=config.get("seed", 0) if _VLLM_015 else self.vllm_dp_rank,\n'
        '            **({key: value for key, value in config.get("engine_kwargs", {}).get("vllm", {}).items()\n'
        '                if value is not None}\n'
        '               if _VLLM_015 else {}),\n')
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
    source = replace_once(source, '    async def wake_up(self):\n        await self.engine.wake_up()\n',
        '''    async def generate_tokens(self, prompt_ids, sampling_overrides, request_id):
        # Raw token API composed from the same native AsyncLLM. No chat
        # rendering, tokenizer round trip, decoder or scheduler is introduced.
        from owner_environment_transport import owner_sampling_params
        params = owner_sampling_params(self.owner_sampling_params, [sampling_overrides], None)[0]
        async for output in self.engine.generate(
            prompt={"prompt_token_ids": prompt_ids}, sampling_params=params,
            request_id=request_id, lora_request=self.owner_lora_request):
            if output.finished:
                return output
        raise RuntimeError("Native AsyncLLM ended without its final RequestOutput")

    async def abort_request(self, request_id):
        await self.engine.abort(request_id)

    async def wake_up(self):
        await self.engine.wake_up()
        if _VLLM_015:
            from vllm.lora.request import LoRARequest
            ids = await self.engine.collective_rpc("list_loras")
            self.owner_lora_request = None
            if int(self.config.model.get("lora_rank", 0)):
                adapter_id = next(iter(ids[0]))
                self.owner_lora_request = LoRARequest(
                    lora_name=str(adapter_id), lora_int_id=adapter_id,
                    lora_path="/simon-stub-path")
''')
    # Match the sync owner's initial sleep. The first manager wake must run
    # the existing FSDP/LoRA synchronization instead of skipping it as awake.
    source = replace_once(source,
        '            chat_template_content_format="auto",\n        )\n\n    async def chat_completion',
        '            chat_template_content_format="auto",\n        )\n'
        '        if _VLLM_015:\n'
        '            await self.sleep()\n\n    async def chat_completion')
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


def patch_output_processor(source):
    """Expose the existing owner's output conversion to its async transport.

    Move the original statements; do not implement a second token/mask/position
    conversion. The sync owner calls precisely the same statements afterwards.
    """
    if '\ndef process_request_outputs(' in source:
        return source
    first = source.index('            response = []\n')
    middle = source.index('\n        response_length = response.size(1)', first)
    last = source.index('\n        # free vllm cache engine', middle)
    body = dedent(source[first:middle]) + '\n' + dedent(source[middle:last])
    for before, after in (
        ('self.sampling_params', 'sampling_params'),
        ('self.pad_token_id', 'pad_token_id'),
        ('self.config.response_length', 'max_response_length'),
    ):
        body = body.replace(before, after)
    body += '\nreturn DataProto(batch=batch, non_tensor_batch=non_tensor_batch)\n'
    function = (\
        '\ndef process_request_outputs(outputs, idx, attention_mask, position_ids,\n'
        '                            non_tensor_batch, *, eos_token_id, sampling_params,\n'
        '                            pad_token_id, max_response_length, owner_overrides,\n'
        '                            active_rows=None, active_mask=None, do_sample=True):\n'
        '    """Original owner conversion, shared by sync and async transports."""\n'
        '    batch_size = idx.size(0)\n'
        + indent(body, '    ') + '\n')
    call = (\
        '            batch = process_request_outputs(\n'
        '                outputs, idx, attention_mask, position_ids, non_tensor_batch,\n'
        '                eos_token_id=eos_token_id, sampling_params=self.sampling_params,\n'
        '                pad_token_id=self.pad_token_id, max_response_length=self.config.response_length,\n'
        '                owner_overrides=owner_overrides, active_rows=active_rows,\n'
        '                active_mask=active_mask, do_sample=do_sample)\n')
    source = source[:first] + call + source[last:]
    source = replace_once(source,
        '        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)\n',
        '        return batch\n')
    source = replace_once(source, '\nclass vLLMAsyncRollout:', function + '\nclass vLLMAsyncRollout:')
    compile(source, '<shared native output converter>', 'exec')
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
             ('verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
              lambda source: patch_output_processor(patch_worker(source))),
             ('verl/workers/sharding_manager/fsdp_vllm.py', patch_sharding),
             ('verl/trainer/ppo/ray_trainer.py', patch_trainer)]
    for name, patch in names:
        path = Path(root)/name
        path.write_text(patch(path.read_text()), newline='\n')


def patch_trainer(source):
    anchor = '''            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )
'''
    return replace_once(source,anchor,anchor+
        '            self.traj_collector.async_rollout_manager = self.async_rollout_manager\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_root', type=Path)
    apply(parser.parse_args().candidate_root)
