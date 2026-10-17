"""Configure the original native engine's ngram option in the existing bounded probe."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('probe_native_tp_state10.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '3d92b76ba18dd2c85652c5c54105c40306317234980cd99708f948f5acd4fc0c'
wrapper = source.read_text()
anchor = "try:\n    exec(compile(code, str(source), 'exec'), globals())"
assert wrapper.count(anchor) == 1
configure = '''code = code.replace('-state10-v2', '-ngram5-state10-v1')
code = code.replace('/tmp/dttp{TP}s10v2', '/tmp/dtng5{TP}s10v1')
anchor = 'cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = TP\\n'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + "assert TP == 1\\nOmegaConf.update(cfg, 'actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config', dict(method='ngram', num_speculative_tokens=5), force_add=True)\\n")
anchor = '    ids = list(engine.llm_engine.list_loras())\\n'
assert code.count(anchor) == 1
code = code.replace(anchor, "    native_spec = engine.llm_engine.vllm_config.speculative_config\\n    assert native_spec.method == 'ngram' and native_spec.num_speculative_tokens == 5\\n" + anchor)
'''
wrapper = wrapper.replace(anchor, configure + anchor)
exec(compile(wrapper, str(source), 'exec'), globals())
