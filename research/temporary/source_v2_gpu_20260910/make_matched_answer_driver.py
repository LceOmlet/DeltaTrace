"""Use the verified initial-seed FT adapter without restricting reasoning hops."""
from pathlib import Path
import hashlib

HERE=Path(__file__).resolve().parent
source=HERE/'evaluate_answer.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='04b61e0245ce1fc917700acddfc83bd6ceb47e6076959904366ef0aec2a26600'
code=source.read_text(encoding='utf-8')
code=code.replace('    import ft_ifr_improve as ft','    import ft_ifr_improve as ft\n    from ft_target_control import InitialTargetFT')
code=code.replace("                    elif target_mode == 'answer_conditioned':\n                        ex.thinking_span = list(ex.sink_span)\n",'')
code=code.replace("        report['weighted_sources'] =", "        report['FT_initial_target_adapter'] = {'sha256': sha((PILOT/'ft_target_control.py').read_bytes()), 'addendum_sha256': sha((PILOT/'ANSWER_HOPS_ADDENDUM.md').read_bytes())}\n        if choice:\n            assert choice['FT_initial_target_adapter_sha256'] == report['FT_initial_target_adapter']['sha256']\n        report['weighted_sources'] =")
code=code.replace('                                    original = ft.LLMIFRAttributionBoth(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32, show_progress=False)',
    "                                    tracer_class = InitialTargetFT if target_mode == 'answer_conditioned' else ft.LLMIFRAttributionBoth\n                                    original = tracer_class(model, tokenizer, chunk_tokens=128, sink_chunk_tokens=32, show_progress=False)\n                                    seed_kwargs = {'initial_target_mask': target_weights} if target_mode == 'answer_conditioned' else {}")
code=code.replace("thinking_span=tuple(ex.thinking_span) if ex.thinking_span is not None else None, n_hops=hops)",
    "thinking_span=tuple(ex.thinking_span) if ex.thinking_span is not None else None, n_hops=hops, **seed_kwargs)")
code=code.replace("                            assert aggregation_calls, 'No actual FT target aggregation was observed'",
    "                            assert aggregation_calls, 'No actual FT target aggregation was observed'\n                            assert all(item['start'] == 0 and item['end'] == gen_len-2 for item in aggregation_calls), 'Reasoning-hop support was reduced'")
assert 'ex.thinking_span = list(ex.sink_span)' not in code
assert code.count('**seed_kwargs')==1 and code.count('tracer_class =')==1
compile(code,'evaluate_answer_matched.py','exec')
path=HERE/'evaluate_answer_matched.py'
path.write_text(code,encoding='utf-8')
print(hashlib.sha256(path.read_bytes()).hexdigest())
