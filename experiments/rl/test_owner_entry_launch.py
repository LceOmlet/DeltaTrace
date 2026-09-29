"""Run the actual pinned trainer's config validator before allocating workers."""
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from hydra import initialize_config_dir, compose
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from launch_sql_native import command
from launch_owner_entry_check import options_for
from owner_runtime_options import owner_command


@pytest.mark.parametrize('task,method,phase',[
    ('SkyRL-SQL','dt','bounded'), ('SkyRL-SQL','grpo','bounded'),
    ('SkyRL-SQL','dt','formal'), ('SkyRL-SQL','grpo','formal'),
    ('TextCraft','dt','bounded'), ('AppWorld','dt','bounded'),
])
def test_two_rank_config_satisfies_native_validator(task,method,phase,tmp_path):
    if task=='SkyRL-SQL':
        argv,_=command(SimpleNamespace(method=method,phase=phase,data=str(tmp_path),output=str(tmp_path)))
    else:
        options,_=options_for(task,tmp_path,tmp_path)
        argv=owner_command(options)
    with initialize_config_dir(config_dir=str(Path(os.environ['VERL_ROOT'])/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='ppo_trainer',overrides=argv[3:])
    RayPPOTrainer._validate_config(SimpleNamespace(config=cfg,use_reference_policy=False,use_critic=False))
    assert cfg.actor_rollout_ref.actor.entropy_coeff==.001
    assert cfg.actor_rollout_ref.actor.clip_ratio_c==3.
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==1
    expected_mini = 1280 if task=='SkyRL-SQL' and phase=='formal' else 64
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==expected_mini
    assert cfg.actor_rollout_ref.model.lora_rank==8
    assert cfg.actor_rollout_ref.model.lora_alpha==16


def test_textcraft_renderer_uses_qwen_official_non_thinking_option():
    from transformers import AutoTokenizer
    from textcraft_environment_entry import configure_textcraft_tokenizer, owner_module
    tokenizer=AutoTokenizer.from_pretrained(os.environ['DT_TOKENIZER_PATH'],local_files_only=True)
    schemas=owner_module(Path(os.environ['AGENTGYM_RL_ROOT'])/'AgentGym-RL/verl/workers/rollout/schemas.py',
                         'owner_textcraft_render_test')
    messages=[schemas.Message('user','Craft one blue dye.'),
              schemas.Message('assistant','Thought: get the ingredient.\nAction: get 1 lapis lazuli'),
              schemas.Message('user','Got 1 lapis lazuli.')]
    expected=tokenizer.apply_chat_template([m.to_dict() for m in messages],
        add_generation_prompt=True,tokenize=True,return_dict=False,enable_thinking=False)
    configure_textcraft_tokenizer(tokenizer)
    actual=schemas.RolloutHandler.get_generation_prompt(SimpleNamespace(messages=messages),tokenizer)
    assert actual==expected
