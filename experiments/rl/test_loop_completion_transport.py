"""Original LOOP generate/runner calls through the exact completion boundary."""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest
from phi_agents.rl.type_defs import UserMessage

from loop_environment_entry import LoopEpisode, CompletionRequest, EpisodeResult
from owner_environment_transport import PolicyReply


def make_llm():
    from phi_agents.rl.llm import qwen_3 as owner
    from patch_loop_tokenizer_entry import patch
    # Works against both a pristine local owner and the deployed patched owner.
    # The independent tokenizer-entry tests check the exact three-method patch.
    namespace = dict(vars(owner))
    exec(compile(patch(Path(owner.__file__).read_text()), owner.__file__, 'exec'), namespace)
    assets = Path(os.environ.get('DT_TOKENIZER_PATH', str(Path(__file__).resolve().parents[2]
        / 'research/temporary/rl_upstream_alignment_20260929/qwen35-entry-assets')))
    return namespace['VLLMQwen3'](host='127.0.0.1', port=1, base_model_path=assets,
        model_id=None, temperature=1., max_new_tokens=1500, top_p=None,
        top_k=None, min_p=None, frequency_penalty=None, max_model_len=32768)


class RunnerFixture:
    """Only environment result is a fixture; original LLM generates the messages."""
    def run(self, scenario, llm):
        first = llm.generate([UserMessage('Read inventory.')])
        second = llm.generate([UserMessage('Read inventory.'), first, UserMessage('wood')])
        return SimpleNamespace(messages=[first, second], ret=.5)


def test_native_client_keeps_exact_ids_history_and_completion_parameters():
    llm = make_llm()
    episode = LoopEpisode(RunnerFixture(), object(), llm)
    request = episode.start()
    try:
        assert isinstance(request, CompletionRequest)
        assert request.prompt_ids == llm.get_tokens([UserMessage('Read inventory.')])[0] + list(llm.generation_prompt_tokens)
        assert request.sampling_kwargs['max_tokens'] == 1500
        assert request.sampling_kwargs['skip_special_tokens'] is False
        assert set(request.sampling_kwargs['stop_token_ids']) == {248044, 248045, 248046}
        text = 'Check.\n</think>\n```python\nprint(1)\n```'
        ids = llm.tokenizer.encode(text, add_special_tokens=False) + [llm.tokenizer.eos_token_id]
        reply = PolicyReply(ids, [-.25]*len(ids), 'unused native-client text', 'stop')
        second_request = episode.advance(reply)
        assert isinstance(second_request, CompletionRequest)
        assert any(second_request.prompt_ids[i:i+len(ids)] == ids for i in range(len(second_request.prompt_ids)))
        result = episode.advance(reply)
        assert isinstance(result, EpisodeResult) and result.rollout.ret == .5
        assert all(m.generated_tokens == ids for m in result.rollout.messages)
        assert all(m.generated_token_logprobs == reply.logprobs for m in result.rollout.messages)
        assert all(m.content == text for m in result.rollout.messages)
    finally:
        episode.cancel()
    assert not episode.thread.is_alive()


def test_owner_error_is_not_converted_to_zero_reward():
    class BrokenRunner:
        def run(self, *_):
            raise ValueError('recorded environment error')
    llm = SimpleNamespace(_vllm=SimpleNamespace())
    episode = LoopEpisode(BrokenRunner(), object(), llm)
    with pytest.raises(ValueError, match='recorded environment error'):
        episode.start()
    episode.thread.join(timeout=2)
    assert not episode.thread.is_alive()
