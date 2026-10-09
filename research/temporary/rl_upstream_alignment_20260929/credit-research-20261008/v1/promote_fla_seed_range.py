"""Apply the already verified AppWorld seed-representation diff to exact bases.

This prepares source only. No alternate GDN/FLA algorithm or model is added.
The three verified edits are reused verbatim, preserving each base's lifecycle.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    app_original=(HERE/'fla-seed-range-preserved/original-AppWorld-GDN.py').read_bytes()
    app_repaired=(HERE/'native-fla-range-owner-prepared/qwen35_gdn_finite.py').read_bytes()
    assert sha(app_original)=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
    assert sha(app_repaired)=='33b169b3fb660eb8ce57a6bda6ecb04c7f7235a029aa04038ff447b2faddf6fd'
    original=app_original.decode().splitlines(True);repaired=app_repaired.decode().splitlines(True)
    edits=[]
    for tag,i,j,a,b in difflib.SequenceMatcher(None,original,repaired,autojunk=False).get_opcodes():
        if tag=='equal':continue
        if tag=='insert':i-=1;a-=1
        old=''.join(original[i:j]);new=''.join(repaired[a:b])
        assert old and app_original.decode().count(old)==1
        edits.append((old,new))
    assert len(edits)==3
    target=REPO/'deltatrace/clean/qwen35/qwen35_gdn_finite.py'
    raw=target.read_bytes()
    assert sha(raw)=='ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'
    text=raw.decode()
    for old,new in edits:
        assert text.count(old)==1
        text=text.replace(old,new)
    back=text
    for old,new in reversed(edits):
        assert back.count(new)==1
        back=back.replace(new,old)
    assert back.encode()==raw
    for content,consumes in [(raw,False),(text.encode(),False),(app_original,True),(app_repaired,True)]:
        tree=ast.parse(content)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gdn_finite_pullback')
        assert ('consume_captures' in {n.arg for n in fn.args.kwonlyargs})==consumes
    folder=HERE/'accepted-fla-seed-range-v1';folder.mkdir(exist_ok=True)
    (folder/'original-TextCraft-GDN.py').write_bytes(raw)
    (folder/'TextCraft-owner.patch').write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),text.splitlines(True),
        fromfile='TextCraft-ef55ce08',tofile='accepted-TextCraft-seed-range-v1')),encoding='utf8')
    target.write_bytes(text.encode())
    record=dict(status='prepared_for_user_authorized_deployment',version='fla-seed-range-20261009-v1',
        source=__file__,edits_reused_verbatim_from_verified_AppWorld=3,inverse_patch_restores_exact_base=True,
        textcraft=dict(base_sha256=sha(raw),repaired_sha256=sha(text.encode()),source=str(target),consume_captures=False),
        appworld=dict(base_sha256=sha(app_original),repaired_sha256=sha(app_repaired),
            source=str(HERE/'native-fla-range-owner-prepared/qwen35_gdn_finite.py'),consume_captures=True),
        verification_receipt='experiments/rl/results_fla_range_owner_20261009.json',
        default_norm_gate_selector_changed=False,finite_FLA_callback_count_changed=False,
        QVA_or_whitening_or_PPO_changed=False,formal_training_restart=False,
        historical_candidate_files_modified=False)
    (folder/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
