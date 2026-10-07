"""Prepare only the isolated finite-MLP token-chunk candidate.

Reuse the frozen lifetime preparation's owner renderer and transport main.
The renderer changes only candidate source/identity seams; original submit
validation, CPU imports and configuration comparison remain the owners.
Running this helper prepares only. It does not submit, stop, restart or deploy.
Preparation is not numerical, memory-capacity or throughput verification.
"""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
LIFETIME_PREPARE = AUDIT / 'direct-target-gpu-lifetime-20261007/v1/prepare_appworld_gpu_lifetime.py'
LIFETIME_PREPARE_SHA = '2c0cd90395b2efa4734cf1b2f050b3a10b20b9bbc3a2c88240b4d6cd9e424d3c'
PRIOR_SOURCE_SHA = '942c2d686a701317e8441f5b299dd86dfb55deb7356c7427e62bc4cbdcd36488'
RETAINED_GDN_SHA = '448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
REPLACEMENTS = {
    'clean/qwen35/qwen35_dense_finite_runner.py': (
        'ba639b2876827cc250f4806af278065181ef78d28c8b22da97377c34b9e168d7',
        '628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'),
    'clean/qwen35/qwen35_decoder_finite.py': (
        '047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197',
        '1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lifetime_owner():
    assert sha(LIFETIME_PREPARE) == LIFETIME_PREPARE_SHA
    spec = importlib.util.spec_from_file_location('frozen_lifetime_preparation_owner', LIFETIME_PREPARE)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


def render_owner_code(owner, root, out, commit):
    """Compose the frozen renderer, retaining its original preparation owners."""
    lifetime = lifetime_owner()
    code = lifetime.render_owner_code(owner, root, out, commit)
    once = lifetime.replace_once
    code = once(code,
        "prior_path=R/'runs/direct-target-causal-prefix-20261007-v2/appworld/appworld-dt/source.json'",
        "prior_path=R/'runs/direct-target-gpu-lifetime-20261007-v1/appworld/appworld-dt/source.json'")
    code = once(code, lifetime.PRIOR_SOURCE_SHA, PRIOR_SOURCE_SHA)
    code = code.replace('996278', '1468126').replace('1791352696.13', '1791357190.77')
    code = once(code, 'replacements=' + repr(lifetime.REPLACEMENTS),
                'replacements=' + repr(REPLACEMENTS))
    code = once(code,
        "output=R/'runs/direct-target-gpu-lifetime-20261007-v1/appworld/appworld-dt'",
        "output=R/'runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt'")
    code = once(code,
        "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite']:",
        "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite','qwen35_decoder_finite']:")
    code = once(code,
        "assert imports['qwen35_dense_finite_runner']['sha256']=='" + lifetime.REPLACEMENTS['clean/qwen35/qwen35_dense_finite_runner.py'][1] + "'",
        "assert imports['qwen35_dense_finite_runner']['sha256']=='" + REPLACEMENTS['clean/qwen35/qwen35_dense_finite_runner.py'][1] + "'\n"
        "assert imports['qwen35_decoder_finite']['path']==str(dt/'clean/qwen35/qwen35_decoder_finite.py')\n"
        "assert imports['qwen35_decoder_finite']['resolved_path']==str(dt/'clean/qwen35/qwen35_decoder_finite.py')\n"
        "assert imports['qwen35_decoder_finite']['sha256']=='" + REPLACEMENTS['clean/qwen35/qwen35_decoder_finite.py'][1] + "'")
    code = once(code,
        " if name=='qwen35_dense_finite_runner':continue\n"
        " assert after['sha256']==before['sha256'],name\n"
        " if name!='qwen35_answer_finite':\n"
        "  assert after['path']==before['path'] and after['resolved_path']==before['resolved_path'],name",
        " if name in ('qwen35_dense_finite_runner','qwen35_decoder_finite'):continue\n"
        " assert after['sha256']==before['sha256'],name\n"
        " assert after['path']==remap(before['path']),name\n"
        " assert after['resolved_path']==before['resolved_path'],name")
    code = once(code,
        "('qwen35_dense_finite_runner.py','qwen35_gdn_finite.py','prepare_appworld_gpu_lifetime.py','submit_prepared_direct_targets.py')",
        "('qwen35_dense_finite_runner.py','qwen35_decoder_finite.py','prepare_appworld_mlp_token_chunk.py','submit_prepared_direct_targets.py')")
    code = code.replace('prepare_appworld_gpu_lifetime.py', 'prepare_appworld_mlp_token_chunk.py')
    code = code.replace('gpu_lifetime_preparation=', 'mlp_token_chunk_preparation=')
    code = once(code,
        'Real executed joint action targets; isolated last-use release independent of capture transport;',
        'Real executed joint action targets; isolated finite-MLP token chunking at2048;')
    code = once(code,
        'Prepared-only linked DT tree; exactly two lifetime files replaced; CPU imports/configuration only;',
        'Prepared-only linked DT tree; exactly runner/decoder replaced; CPU imports/configuration only;')
    code = once(code,
        'plus explicit unchanged head and candidate runner/GDN imports',
        'plus explicit candidate runner/decoder and unchanged head/GDN imports')
    code = once(code,
        'mlp_token_chunk_preparation=dict(changed_dt_files=changed,previous_driver=previous,',
        'mlp_token_chunk_preparation=dict(changed_dt_files=changed,mlp_token_chunk_size=2048,previous_driver=previous,')
    code = once(code,
        "retained_head=binding(dt/'clean/qwen35/qwen35_answer_finite.py'),",
        "retained_head=binding(dt/'clean/qwen35/qwen35_answer_finite.py'),\n"
        "                retained_gdn=binding(dt/'clean/qwen35/qwen35_gdn_finite.py'),")
    code = once(code,
        "assert sha(dt/'clean/qwen35/qwen35_answer_finite.py')=='" + lifetime.HEAD_SHA + "'",
        "assert sha(dt/'clean/qwen35/qwen35_answer_finite.py')=='" + lifetime.HEAD_SHA + "'\n"
        "assert sha(dt/'clean/qwen35/qwen35_gdn_finite.py')=='" + RETAINED_GDN_SHA + "'")
    code = once(code,
        "gdn=inspection['imports']['qwen35_gdn_finite'],",
        "gdn=inspection['imports']['qwen35_gdn_finite'],decoder=inspection['imports']['qwen35_decoder_finite'],")
    ast.parse(code)
    return code


def transport_owner_namespace():
    """Retain the frozen transport/argument parser; change only output/archive names."""
    lifetime = lifetime_owner()
    main_source = inspect.getsource(lifetime.main)
    main_source = lifetime.replace_once(main_source,
        "'/candidates/direct-target-gpu-lifetime-20261007-v1'",
        "'/candidates/direct-target-mlp-token-chunk-20261007-v1'")
    main_source = lifetime.replace_once(main_source,
        "arcname='prepare_appworld_gpu_lifetime.py'",
        "arcname='prepare_appworld_mlp_token_chunk.py'")
    namespace = dict(vars(lifetime), HERE=HERE, REPLACEMENTS=REPLACEMENTS,
                     render_owner_code=render_owner_code, __file__=__file__)
    exec(compile(ast.parse(main_source), str(LIFETIME_PREPARE), 'exec'), namespace)
    return namespace


def main():
    transport_owner_namespace()['main']()


if __name__ == '__main__':
    main()
