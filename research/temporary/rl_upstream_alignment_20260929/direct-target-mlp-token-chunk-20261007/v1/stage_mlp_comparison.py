"""Compose the pinned staging owner; stage/CPU-inspect only when invoked.

The existing lifetime candidate, native controller and real-ID loader are
unchanged. There is no GPU launch or formal submission in this helper.
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
OWNER = AUDIT / 'direct-target-gpu-lifetime-20261007/v1/stage_lifetime_comparison.py'
OWNER_SHA256 = '8428ddece033362a6429d641bd5aef18f347d8848ebcf6a1a2c4fa1807e5aecc'
LIFETIME_DIAGNOSTIC_SHA256 = '2829969c633630605ea4aae78f64a7ee29efce0d4a6a4b063c63b197f044ddfe'
CHUNK_SHA256 = '1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'


def composed_source():
    assert hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA256
    lifetime = OWNER.with_name('diagnose_direct_target.py')
    assert hashlib.sha256(lifetime.read_bytes()).hexdigest() == LIFETIME_DIAGNOSTIC_SHA256
    chunk = HERE / 'candidate/qwen35_decoder_finite.py'
    assert hashlib.sha256(chunk.read_bytes()).hexdigest() == CHUNK_SHA256
    source = OWNER.read_text(encoding='utf-8')

    def replace_once(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)

    replace_once("remote = stage.ROOT + '/receipts/direct-target-gpu-lifetime-20261007-v1'",
                 "remote = stage.ROOT + '/receipts/direct-target-mlp-token-chunk-20261007-v1'")
    replace_once("files['diagnose_direct_target.py'] = HERE / 'diagnose_direct_target.py'",
                 "files['diagnose_direct_target.py'] = HERE / 'diagnose_direct_target.py'\n"
                 "    files['lifetime_diagnostic_owner.py'] = AUDIT / 'direct-target-gpu-lifetime-20261007/v1/diagnose_direct_target.py'")
    replace_once("baseline = AUDIT / 'direct-target-causal-prefix-20261007/v3/memory-source/clean/qwen35/qwen35_gdn_finite.py'",
                 "baseline = HERE / 'candidate/qwen35_decoder_finite.py'")
    source = source.replace('baseline_qwen35_gdn_finite.py', 'chunk_qwen35_decoder_finite.py')
    replace_once('baseline_gdn=dict(path=remote', 'chunk_owner=dict(path=remote')
    replace_once("sha256='ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9'))",
                 "sha256='" + CHUNK_SHA256 + "'), token_chunk_size=2048)")
    ast.parse(source, filename=str(HERE / 'composed-staging-owner.py'))
    return source


if __name__ == '__main__':
    exec(compile(composed_source(), str(OWNER), 'exec'),
         dict(__name__='__main__', __file__=str(__file__)))
