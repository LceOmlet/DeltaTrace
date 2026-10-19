"""Keep optional completion metadata failures outside the generation call."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('observe_generation_native_completions.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'fea05c2465ad23f4ef46c0b73ba58cae05519e61042e0e110ca686260e60494e'
code = owner.read_text()
code = code.replace('_native_completion_observer_v4_installed', '_native_completion_observer_v4_v2_installed')
code = code.replace('generation-native-completions-v4', 'generation-native-completions-v4-v2')
code = code.replace('            completion_events=[]', '            completion_events=[]\n            completion_errors=[]')
before = '''                for output in returned:
                    if getattr(output,'finished',False):
                        completion_events.append(dict(
                            request_id=output.request_id,finished=stamp,
                            generated_token_lengths=[len(sample.token_ids) for sample in output.outputs]))'''
assert code.count(before) == 1
after = '                try:\n' + '\n'.join('    ' + line for line in before.splitlines()) + '''
                except Exception as error:
                    completion_errors.append(repr(error))'''
code = code.replace(before, after)
code = code.replace('native_completion_events=completion_events,',
                    'native_completion_events=completion_events,native_completion_observation_errors=completion_errors,')
anchor = "exec(compile(code, str(owner), 'exec'), globals())"
assert code.count(anchor) == 1
unwrap = '''        pending=engine.generate
        if getattr(engine,'_native_completion_observer_v4_installed',False) and pending.__name__=='observe_existing_generate':
            cells=dict(zip(pending.__code__.co_freevars,[c.cell_contents for c in pending.__closure__]))
            engine.generate=cells['native_generate']
            result['superseded_completion_observer_calls']=cells['count'][0]
        native_generate=engine.generate'''
code = code.replace(anchor,
    "assert code.count('        native_generate=engine.generate') == 1\n"
    "code=code.replace('        native_generate=engine.generate'," + repr(unwrap) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
