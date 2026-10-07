"""Original AgentGym rollout with a transport to the existing VERL workers.

No environment state machine, retry, parser, reward or training loss is defined
here. AgentGym owns the entire episode and final tensors. Hooks only retain
action boundaries for DT; VERL owns engine calls and the resulting updates.
"""
from pathlib import Path
import json
import os
import site
import sys
from types import ModuleType
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from textcraft_environment_entry import owner_module


class TextCraftOwner:
    def __init__(self, configuration, tokenizer, evaluation):
        self.config, self.tokenizer, self.evaluation = configuration, tokenizer, evaluation
        if os.environ.get('TEXTCRAFT_EXTRAS'):
            site.addsitedir(os.environ['TEXTCRAFT_EXTRAS'])
        root = Path(configuration.env.textcraft.owner_root)/'AgentGym-RL'
        import verl.utils
        for namespace, path in [(verl.utils, root/'verl/utils')]:
            if str(path) not in namespace.__path__:
                namespace.__path__.append(str(path))
        # Isolate the author's identically named schemas module from VERL's.
        # Namespace packages load the original files, not alternate classes.
        for name, path in [('agentgym_environment', root/'verl/workers/rollout'),
                           ('agentgym_environment.agent_vllm_rollout', root/'verl/workers/rollout/agent_vllm_rollout')]:
            if name not in sys.modules:
                package = ModuleType(name)
                package.__path__ = [str(path)]
                sys.modules[name] = package
        self.module = owner_module(root/'verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py',
            'agentgym_environment.agent_vllm_rollout.' + ('eval_entry' if evaluation else 'train_entry'))
        self.dataset_module = owner_module(root/'verl/utils/agent_dataset/rl_dataset.py',
                                          'agentgym_native_dataset')
        self.native = json.loads(Path(__file__).with_name('owner_environment_configs.json').read_text())['TextCraft']
        self.dataset_client = None
        self.handlers = []

    def close(self):
        if self.dataset_client is not None:
            self.dataset_client.close()
            self.dataset_client = None

    def collect_native_trajectories(self, collector, gen_batch, actor_rollout_wg, is_train):
        from agent_system.reward_manager.episode import EpisodeRewardManager
        from verl.utils.agentgym.client import init_env_client
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from inspect import signature
        from vllm import SamplingParams
        owner = self
        native = self.native['eval' if self.evaluation else 'train']
        rollout_config = native['rollout'] if self.evaluation else native['actor_rollout_ref']['rollout']
        env_config = self.config.env.textcraft.eval_client if self.evaluation else self.config.env.textcraft.client
        data_config = native['data']
        n = rollout_config['n']
        if self.dataset_client is None:
            self.dataset_client = init_env_client(env_config)
        # Call the original dataset row renderer/tokenizer without reloading the
        # task collection already sampled by VERL's existing dataloader.
        dataset = self.dataset_module.RLHFDataset.__new__(self.dataset_module.RLHFDataset)
        dataset.dataframe = [dict(item_id=f"textcraft_{task['item_id']}")
                             for task in gen_batch.non_tensor_batch['env_kwargs']]
        dataset.env_client, dataset.tokenizer = self.dataset_client, self.tokenizer
        dataset.prompt_key, dataset.max_prompt_length = data_config['prompt_key'], data_config['max_prompt_length']
        dataset.return_raw_chat, dataset.truncation = True, 'error'
        prompts = DataProto.from_single_dict(self.dataset_module.collate_fn(
            [dataset[i] for i in range(len(dataset.dataframe))]))
        prompts.meta_info = dict(gen_batch.meta_info, max_rounds=self.config.env.max_steps)
        groups = [str(uuid.uuid4()) for _ in range(len(prompts))]
        ids = [str(uuid.uuid4()) for _ in range(len(prompts)*n)]
        records = [[] for _ in ids]
        handlers = []
        module = self.module
        original_handler = module.RolloutHandler
        original_client = module.init_env_client
        execution_clients = []

        class RecordedHandler(original_handler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.transport_index = len(handlers)
                handlers.append(self)

            def add_assistant_message(self, *args, **kwargs):
                before = len(self.input_ids)
                super().add_assistant_message(*args, **kwargs)
                # Read the owner's newly emitted mask, not a second ChatML parser.
                positions = [i for i in range(before, len(self.input_ids)) if self.loss_mask[i]]
                start, end = positions[0], positions[-1]+1
                records[self.transport_index][-1].update(start=start, end=end,
                    native_prompt_ids=self.input_ids[:start].copy(),
                    native_response_ids=self.input_ids[start:end].copy(),
                    native_content=(args[1] if len(args) > 1 else kwargs['content']))

            def truncate_output_ids(self):
                self.dt_full_input_ids = self.input_ids.copy()
                self.dt_full_policy_mask = self.loss_mask.copy()
                return super().truncate_output_ids()

        def recorded_client(*args, **kwargs):
            client = original_client(*args, **kwargs)
            index = len(execution_clients)
            execution_clients.append(client)
            def record_payload(metadata):
                records[index][-1]['executed_payload_metadata'] = metadata
            # The pinned owner's step calls this optional sink only after its
            # own parser and original POST have actually returned.
            client._executed_payload_sink = record_payload
            return client

        class EngineTransport:
            # VERL's existing RPC manages wake/sync/sleep for each generation.
            def init_cache_engine(self):
                pass

            def free_cache_engine(self):
                pass

            def generate(self, *, prompt_token_ids, sampling_params, **kwargs):
                if not prompt_token_ids:
                    return (torch.empty((0, 0), dtype=torch.long),)
                active = [h for h in handlers if not h.done]
                sampling = {k: getattr(sampling_params, k)
                            for k in signature(SamplingParams.from_optional).parameters}
                transport_input = DataProto.from_dict(tensors=dict(
                    input_ids=torch.zeros(len(active), 1, dtype=torch.long)),
                    non_tensors=dict(data_source=np.array(['textcraft']*len(active), dtype=object)),
                    meta_info=gen_batch.meta_info)
                batch = collector.preprocess_batch(transport_input, dict(
                    raw_prompt_ids=prompt_token_ids, sampling_kwargs=[sampling]*len(active)))
                batch.meta_info.update(eos_token_id=owner.tokenizer.eos_token_id,
                                       pad_token_id=owner.tokenizer.pad_token_id)
                batch, padding = pad_dataproto_to_divisor(batch, actor_rollout_wg.world_size)
                output = unpad_dataproto(actor_rollout_wg.generate_sequences(batch), padding)
                for row, h in enumerate(active):
                    length = int(output.non_tensor_batch['owner_response_length'][row])
                    records[h.transport_index].append(dict(data=output.select_idxs([row]),
                        output_ids=output.batch['responses'][row, :length].tolist(),
                        prompt_ids=list(prompt_token_ids[row])))
                return (output.batch['responses'],)

        module.RolloutHandler = RecordedHandler
        module.init_env_client = recorded_client
        try:
            runner = module.vLLMRollout(None, OmegaConf.create(rollout_config), env_config,
                self.tokenizer, None, inference_engine=EngineTransport())
            native_output = runner.generate_sequences(prompts)
        finally:
            module.RolloutHandler = original_handler
            module.init_env_client = original_client
        self.handlers, self.records = handlers, records
        output = native_output
        # VERL uses rm_scores for a native reward tensor. All token arrays are
        # passed directly from AgentGym, including observations and its EOS.
        output.batch['rm_scores'] = output.batch.pop('scores')
        output.batch['loss_mask'] = torch.cat((torch.zeros_like(output.batch['prompts']),
                                              output.batch['response_mask']), -1)
        scores = output.batch['rm_scores'].sum(-1).cpu().numpy()
        output.non_tensor_batch.update(data_source=np.array(['textcraft']*len(output), dtype=object),
            uid=np.repeat(np.array(groups, dtype=object), n), traj_uid=np.array(ids, dtype=object),
            episode_rewards=scores, episode_lengths=output.batch['task_rounds'].cpu().numpy(),
            tool_callings=output.batch['task_rounds'].cpu().numpy())
        output.meta_info['multi_turn'] = True
        if self.config.algorithm.adv_estimator != 'deltatrace':
            return output
        from executed_target_spans import encoded_payload_mask, full_trajectory_artifact
        direct_artifacts = []
        for h, turns in zip(handlers, records):
            full_ids, full_policy = h.dt_full_input_ids, h.dt_full_policy_mask
            target_mask = [False] * len(full_ids)
            parser_metadata = []
            suffix_ids = self.tokenizer.encode(h.format_config['qwen']['assistat_suffix_msg'],
                                               add_special_tokens=False)
            for turn in turns:
                metadata = turn.get('executed_payload_metadata')
                if metadata is None:
                    continue  # The original owner rejected/failed this execution.
                content = turn['native_content']
                if turn['native_response_ids'][-len(suffix_ids):] != suffix_ids:
                    raise ValueError('Original TextCraft assistant suffix differs from its owner format')
                content_ids = turn['native_response_ids'][:-len(suffix_ids)]
                mask, mapping = encoded_payload_mask(self.tokenizer, content, content_ids,
                                                     metadata['source_spans'])
                start = turn['start']
                for offset, selected in enumerate(mask):
                    target_mask[start + offset] = selected
                parser_metadata.append(dict(start=start, **metadata, token_mapping=mapping))
            artifact = full_trajectory_artifact(h.prompt_ids, full_ids, full_policy, target_mask,
                                                h.response_ids)
            artifact['retained_response_positions'] += [-1] * (
                output.batch['responses'].shape[1] - len(h.response_ids))
            artifact['payload_metadata'] = parser_metadata
            direct_artifacts.append(artifact)
        direct_array = np.empty(len(direct_artifacts), dtype=object)
        direct_array[:] = direct_artifacts
        output.non_tensor_batch['dt_direct_target_artifact'] = direct_array
        sources, maps = [], []
        for index, (h, turns) in enumerate(zip(handlers, records)):
            mapping = []
            for step, turn in enumerate(turns):
                source_index = len(sources)
                start = turn['start']-len(h.prompt_ids)
                length = min(len(turn['native_response_ids']), len(h.response_ids)-start)
                sources.append(dict(prompt_ids=turn['native_prompt_ids'], response_ids=turn['native_response_ids'],
                    traj_uid=ids[index], data_source='textcraft', env_step=step, active_masks=True,
                    episode_rewards=scores[index], episode_lengths=len(turns),
                    rewards=scores[index] if step == len(turns)-1 else 0.))
                if length > 0:
                    assert h.input_ids[:turn['start']] == turn['native_prompt_ids']
                    assert h.response_ids[start:start+length] == turn['native_response_ids'][:length]
                    mapping.append((source_index, start, length))
            maps.append(mapping)
        # AgentGym itself decodes/re-encodes model output and appends EOS. DT
        # must target those exact official training tokens; generated IDs may
        # differ (e.g. stripped special tokens). Do not alter the owner's arrays.
        from owner_trajectory_batch import native_credit_response_batch
        self.credit_responses = native_credit_response_batch(owner.tokenizer, sources)
        array = np.empty(len(maps), dtype=object)
        array[:] = maps
        output.non_tensor_batch['dt_response_slices'] = array
        return output
