"""Default-inert owner seams for exact external environment artifacts.

Apply after the verified 20bd331 distributed runtime patches. The official
collector, vLLM generation, reward manager and algorithms retain ownership.
This patch adds an env factory, raw token input, exact output metadata and a
callback. It does not copy an alternative rollout loop or training algorithm.
"""
from pathlib import Path
import argparse


def replace_once(source, old, new):
    if new in source:
        return source
    if source.count(old) != 1:
        raise ValueError(f'Expected one pinned owner entry: {old[:100]!r}')
    return source.replace(old, new, 1)


def patch_main(source):
    source = replace_once(source,
        '        envs, val_envs = make_envs(config)\n',
        '        if not config.env.get("factory"):\n'
        '            envs, val_envs = make_envs(config)\n')
    anchor = '        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)\n'
    source = replace_once(source, anchor, anchor +
        '        if config.env.get("factory"):\n'
        '            from hydra.utils import instantiate\n'
        '            envs, val_envs = instantiate(config.env.factory, configuration=config,\n'
        '                                         tokenizer=tokenizer, _recursive_=False)\n')
    return replace_once(source,
        '        trainer.init_workers()\n        trainer.fit()\n',
        '        from contextlib import ExitStack\n'
        '        with ExitStack() as environment_cleanup:\n'
        '            if config.env.get("factory"):\n'
        '                environment_cleanup.callback(envs.close)\n'
        '                environment_cleanup.callback(val_envs.close)\n'
        '            trainer.init_workers()\n'
        '            trainer.fit()\n')


def patch_collector(source):
    anchor = "        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]\n"
    source = replace_once(source, anchor,
        '        if "raw_prompt_ids" in obs:\n'
        '            from owner_environment_transport import preprocess_owner_tokens\n'
        '            return preprocess_owner_tokens(self, item, gen_batch, obs)\n\n' + anchor)
    anchor = '            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]\n'
    source = replace_once(source, anchor, anchor +
        '            if "owner_sampling_kwargs" in batch.non_tensor_batch:\n'
        '                non_tensor_batch_keys_to_pop.append("owner_sampling_kwargs")\n')
    anchor = '            batch_input.meta_info = gen_batch.meta_info\n'
    source = replace_once(source, anchor, anchor +
        '            if "owner_sampling_kwargs" in batch_input.non_tensor_batch:\n'
        '                batch_input.non_tensor_batch["rollout_active_mask"] = active_masks\n')
    anchor = ('            if self.config.env.get("context_budget_action", "error") == "end_episode":\n'
              '                next_obs, rewards, dones, infos = envs.step(text_actions, active_masks=active_masks)')
    source = replace_once(source, anchor,
        '            if hasattr(envs, "step_policy_outputs"):\n'
        '                next_obs, rewards, dones, infos = envs.step_policy_outputs(batch, active_masks)\n'
        '            elif self.config.env.get("context_budget_action", "error") == "end_episode":\n'
        '                next_obs, rewards, dones, infos = envs.step(text_actions, active_masks=active_masks)')
    anchor = '        success: Dict[str, np.ndarray] = envs.success_evaluator(\n'
    source = replace_once(source, anchor,
        '        if hasattr(envs, "finalize_trajectory_metadata"):\n'
        '            envs.finalize_trajectory_metadata(total_batch_list)\n\n' + anchor)
    return source


def patch_vllm(source):
    anchor = '        active_mask = non_tensor_batch.pop("rollout_active_mask", None)\n'
    source = replace_once(source, anchor,
        '        owner_overrides = non_tensor_batch.pop("owner_sampling_kwargs", None)\n' + anchor)
    anchor = '        with self.update_sampling_params(**kwargs):\n'
    source = replace_once(source, anchor, anchor +
        '            request_sampling = self.sampling_params\n'
        '            if owner_overrides is not None:\n'
        '                from owner_environment_transport import owner_sampling_params\n'
        '                request_sampling = owner_sampling_params(self.sampling_params, owner_overrides, active_rows)\n')
    source = replace_once(source, '                sampling_params=self.sampling_params,\n',
                                     '                sampling_params=request_sampling,\n')
    anchor = '            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)\n'
    source = replace_once(source, anchor,
        '            if owner_overrides is not None:\n'
        '                non_tensor_batch["owner_response_length"] = np.array([len(r) for r in response])\n'
        '                texts = np.full(batch_size, "", dtype=object)\n'
        '                reasons = np.full(batch_size, "", dtype=object)\n'
        '                rows = range(batch_size) if active_rows is None else active_rows\n'
        '                for row, output in zip(rows, outputs):\n'
        '                    texts[row] = output.outputs[0].text\n'
        '                    reasons[row] = output.outputs[0].finish_reason\n'
        '                non_tensor_batch["owner_response_text"] = texts\n'
        '                non_tensor_batch["owner_finish_reason"] = reasons\n\n' + anchor)
    anchor = '        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)\n'
    source = replace_once(source, anchor, anchor +
        '        if owner_overrides is not None:\n'
        '            lengths = torch.as_tensor(non_tensor_batch["owner_response_length"], device=response.device)\n'
        '            response_attention_mask = (torch.arange(response_length, device=response.device)[None, :] < lengths[:, None]).to(attention_mask.dtype)\n')
    return source


def patch_dataset(source):
    # TF5 returns BatchEncoding by default, whereas this owner measures ID count.
    return replace_once(source,
        'tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)',
        'tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, return_dict=False)')


def apply(root):
    paths = {'verl/trainer/main_ppo.py': patch_main,
             'agent_system/multi_turn_rollout/rollout_loop.py': patch_collector,
             'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py': patch_vllm,
             'verl/utils/dataset/rl_dataset.py': patch_dataset}
    for name, patch in paths.items():
        path = root / name
        source = patch(path.read_text(encoding='utf-8'))
        compile(source, str(path), 'exec')
        path.write_text(source, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_root', type=Path)
    apply(parser.parse_args().owner_root)
