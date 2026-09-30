"""Inject the existing engine into AgentGym's original rollout, not its trainer.

The original loop, exception policy, masking, truncation and reward assembly
remain verbatim. Engine creation stays unchanged when no dependency is passed.
"""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch_rollout(source):
    source = replace_once(source,
        'from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs',
        'from ..schemas import RolloutHandler, Message, _pre_process_inputs')
    source = replace_once(source,
        'from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length',
        'from verl.utils.torch_functional import pad_sequence_to_length')
    source = replace_once(source,
        'from verl.third_party.vllm import LLM, vllm_version\nfrom verl.third_party.vllm import parallel_state as vllm_ps\n',
        '# The constructor imports the owner engine only when no engine is injected.\n')
    start = '        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \\\n'
    end = '        self.inference_engine.offload_model_weights()\n'
    if 'if "inference_engine" in kwargs:' not in source:
        a, b = source.index(start), source.index(end) + len(end)
        original = source[a:b]
        source = source[:a] + ('        if "inference_engine" in kwargs:\n'
            '            from vllm import __version__ as vllm_version\n'
            '            self.inference_engine = kwargs.pop("inference_engine")\n'
            '        else:\n'
            '            from verl.third_party.vllm import LLM, vllm_version\n'
            '            from verl.third_party.vllm import parallel_state as vllm_ps\n' +
            ''.join('    '+line if line.strip() else line for line in original.splitlines(True))) + source[b:]
    # The existing engine lives in VERL workers; this caller is the driver.
    source = source.replace('torch.distributed.get_rank()',
        '(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)') if 'if torch.distributed.is_initialized()' not in source else source
    return source


def apply(root):
    path = Path(root)/'AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py'
    source = patch_rollout(path.read_text())
    compile(source, str(path), 'exec')
    path.write_text(source, newline='\n')
    schema = path.parents[1]/'schemas.py'
    source = replace_once(schema.read_text(),
        'tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)',
        'tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True, return_dict=False)')
    schema.write_text(source, newline='\n')


if __name__ == '__main__':
    import sys
    apply(sys.argv[1])
