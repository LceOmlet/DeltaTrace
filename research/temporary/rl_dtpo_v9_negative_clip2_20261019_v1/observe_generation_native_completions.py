"""Add native completion timestamps to two existing generation calls per rank.

The original vLLM step outputs and generate outputs pass through unchanged.
No model, scheduler, environment, RNG or synchronization operation is added.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_native_generation.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '69fbc86619652aa68eddb263caffffc67cbaadf473a002d5beee6b85e8b26f29'
code = owner.read_text()
code = code.replace('_native_efficiency_observer_installed', '_native_completion_observer_v4_installed')
code = code.replace('native_generation_efficiency', 'native_generation_completions')
code = code.replace("code=wrapper.read_text()", "code=wrapper.read_text().replace('v9-negative2-eight-GPU-initial-v1','generation-native-completions-v4')")
code = code.replace('if count[0]>=8:', 'if count[0]>=2:')
code = code.replace("result['generation_observer_max_existing_calls']=8", "result['generation_observer_max_existing_calls']=2")
anchor = '            outputs=native_generate(*args,**kwargs)'
assert code.count(anchor) == 1
code = code.replace(anchor, '''            llm=engine.llm_engine
            native_step=llm.step
            completion_events=[]
            def observe_native_step(*step_args,**step_kwargs):
                returned=native_step(*step_args,**step_kwargs)
                stamp=time.time()
                for output in returned:
                    if getattr(output,'finished',False):
                        completion_events.append(dict(
                            request_id=output.request_id,finished=stamp,
                            generated_token_lengths=[len(sample.token_ids) for sample in output.outputs]))
                return returned
            llm.step=observe_native_step
            try:
                outputs=native_generate(*args,**kwargs)
            except BaseException:
                engine.generate=native_generate
                raise
            finally:
                llm.step=native_step''')
anchor = '                    additional_generation_calls=0,original_outputs_returned_unchanged=True)'
assert code.count(anchor) == 1
code = code.replace(anchor, '''                    native_completion_events=completion_events,
                    native_step_method_restored=True,
                    scope='CPU timestamps of original vLLM RequestOutput.finished; no new inference or CUDA event',
''' + anchor)
exec(compile(code, str(owner), 'exec'), globals())
