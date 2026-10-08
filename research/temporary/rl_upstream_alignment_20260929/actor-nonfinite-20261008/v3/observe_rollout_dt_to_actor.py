"""Reuse the previous read-only observer for this distinct diagnostic tree."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
previous = HERE.parent / 'v2/observe_dt_to_actor.py'
source = previous.read_text(encoding='utf-8')
source = source.replace("transport.ROOT+'/receipts/textcraft-DT-to-actor-nonfinite-20261008-v2'",
                        "transport.ROOT+'/receipts/textcraft-rollout-DT-actor-nonfinite-20261008-v3'")
source = source.replace("last_microbatch=v.get('microbatches',[])[-1:]",
                        "last_microbatch=[{k:m.get(k) for k in ('index','phase','input_shape')} for m in v.get('microbatches',[])[-1:]]")
exec(compile(source, str(previous), 'exec'), {'__file__': str(HERE / previous.name), '__name__': '__main__'})
