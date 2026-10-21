"""Use the original metadata owner with this run's actual eight-rank identity."""
from prepare_campaign import ROOT, AUDIT, VERSION, sha

owner = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/inspect_current_workers.py'
assert sha(owner) == 'dc09a24a80a06e6c4c8f6363d02e82f47309851d898b14049d39d7ade2637abc'
code = owner.read_text()
for before, after in {
    "AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'":
        "AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'",
    "(ROOT / 'formal-training.json')": "(AUDIT / 'formal-training.json')",
    "j.get('task') == 'AppWorld'": "j.get('task') == 'TextCraft'",
    "task='AppWorld'": "task='TextCraft'",
    "else 'formal-v1')": "else 'v10-eight-initial-v2')",
    "source = json.loads(Path(job['source_receipt']).read_bytes())":
        "assert job['version']==" + repr(VERSION) + "\nsource = json.loads(Path(job['source_receipt']).read_bytes())",
    'assert len(named) == 2': 'assert len(named) == 8',
}.items():
    assert code.count(before) == 1, before
    code = code.replace(before, after)
exec(compile(code, str(owner), 'exec'), globals())
