"""Use the same bounded nonblocking observer for two engines and their caller."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('profile_current_DT.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'eeb265f6705330b1102768b011c9a04770be629812577aed9a25477da1df184c'
code = source.read_text()


def replace(before, after):
    global code
    assert code.count(before) == 1, before
    code = code.replace(before, after)


replace('assert len(targeted)==2', """assert len(targeted)==2
driver=psutil.Process(active['jobs'][0]['pid'])
caller,=[p for p in driver.children(recursive=True) if 'TaskRunner' in p.name()]
targeted[caller.pid]=caller.create_time()""")
replace("if 'WorkerDict' not in child.name() or child.pid not in targeted:continue",
        'if child.pid not in targeted:continue')
replace('bounded-DT-phase-', 'bounded-generation-phase-')
exec(compile(code, str(source), 'exec'), globals())
