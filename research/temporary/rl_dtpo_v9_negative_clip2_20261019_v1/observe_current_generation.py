"""Reuse the bounded passive generation observer for execution v4.

Read native resolved scheduling configuration and eight existing calls per
rank. No new generation, model, environment or optimizer invocation.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_native_generation.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '69fbc86619652aa68eddb263caffffc67cbaadf473a002d5beee6b85e8b26f29'
code = owner.read_text()
anchor = 'code=wrapper.read_text()'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + ".replace('v9-negative2-eight-GPU-initial-v1','current-generation-passive-v4')")
code = code.replace('_native_efficiency_observer_installed', '_native_efficiency_observer_v4_installed')
anchor = "    result['generation_observer_installed']=bool(engine is not None)"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    native_engine=getattr(engine,'llm_engine',None)
    resolved=getattr(native_engine,'vllm_config',None)
    scheduler=getattr(resolved,'scheduler_config',None)
    compilation=getattr(resolved,'compilation_config',None)
    result['generation_resolved_native_options']={
        'async_scheduling':getattr(scheduler,'async_scheduling',None),
        'max_num_seqs':getattr(scheduler,'max_num_seqs',None),
        'max_num_batched_tokens':getattr(scheduler,'max_num_batched_tokens',None),
        'cudagraph_mode':str(getattr(compilation,'cudagraph_mode',None))}
''' + anchor)
exec(compile(code, str(owner), 'exec'), globals())
