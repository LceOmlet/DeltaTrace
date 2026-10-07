"""Prepare only the isolated two-file GPU lifetime candidate.

This reuses the frozen head-memory preparation owner and the existing prepared
submit validator. Running this helper stages a linked DT tree and checks CPU
imports/configuration; it never submits/stops training, mutates active
manifests, restores checkpoints, or constructs a model/DT/optimizer/episode.
Preparation is not numerical, memory-capacity, throughput, or runtime approval.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
BASE_PREPARE = AUDIT / 'direct-target-head-memory-20261007/v2/prepare_appworld_head_memory.py'
BASE_PREPARE_SHA = 'aaaaddd13aef5b37c4b3d7d0cadf5807fc14f5fe6356d5352e554a8a06608a26'
SUBMIT = AUDIT / 'direct-target-semantics-20261007/submit_prepared_direct_targets.py'
SUBMIT_SHA = '8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
PRIOR_SOURCE_SHA = 'ed3fdf7b3468116b65c96c4df915a4c549c0d9d9e55f72725f805751b2071cc6'
HEAD_SHA = '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
READOUT_SHA = '7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'
REPLACEMENTS = {
    'clean/qwen35/qwen35_dense_finite_runner.py': (
        '5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555',
        'ba639b2876827cc250f4806af278065181ef78d28c8b22da97377c34b9e168d7'),
    'clean/qwen35/qwen35_gdn_finite.py': (
        'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9',
        '448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(code, before, after):
    assert code.count(before) == 1, before
    return code.replace(before, after, 1)


def render_owner_code(owner, root, out, commit):
    """Reuse the exact preparation owner; replace only its candidate seams."""
    code = owner.CODE
    code = replace_once(code,
        "prior_path=R/'runs/direct-action-target-20261007-v3/appworld/appworld-dt/source.json'",
        "prior_path=R/'runs/direct-target-causal-prefix-20261007-v2/appworld/appworld-dt/source.json'")
    code = code.replace('70ffdcfd05fccb94c0cec70e5b1d83d8728e3335bb9f1a4e6bbec9299f53847b',
                        PRIOR_SOURCE_SHA)
    code = code.replace('1791344807.34', '1791352696.13').replace('167065', '996278')
    code = replace_once(code,
        "answer_relative='clean/qwen35/qwen35_answer_finite.py'\n"
        "assert prior['dt_source_sha256'][answer_relative]=='d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'\n"
        "assert sha(O/'setup/qwen35_answer_finite.py')=='@ANSWER_SHA@'",
        "replacements=" + repr(REPLACEMENTS) + "\n"
        "assert prior['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']=='@HEAD_SHA@'\n"
        "assert prior['entry_sha256']['reward_readout.py']=='@READOUT_SHA@'\n"
        "for name,(before,after) in replacements.items():\n"
        " assert prior['dt_source_sha256'][name]==before,name\n"
        " assert sha(O/'setup'/P(name).name)==after,name")
    code = replace_once(code,
        "answer=dt/answer_relative\nassert answer.is_symlink()\n"
        "answer.unlink();answer.write_bytes((O/'setup/qwen35_answer_finite.py').read_bytes())",
        "for name,(before,after) in replacements.items():\n"
        " replacement=dt/name\n assert replacement.is_symlink()\n"
        " replacement.unlink();replacement.write_bytes((O/'setup'/P(name).name).read_bytes())\n"
        " assert sha(replacement)==after,name")
    code = replace_once(code, "assert changed==[answer_relative],changed",
        "assert set(changed)==set(replacements) and len(changed)==2,changed\n"
        "assert sha(dt/'clean/qwen35/qwen35_answer_finite.py')=='@HEAD_SHA@'\n"
        "assert sha(P(prior['entry'])/'reward_readout.py')=='@READOUT_SHA@'")
    code = code.replace('runs/direct-target-head-memory-20261007-v2',
                        'runs/direct-target-gpu-lifetime-20261007-v1')
    code = replace_once(code, "options,sampling=launcher.options_for(output)",
        "options,sampling=launcher.options_for(output)\n"
        "baseline_options,baseline_sampling=launcher.options_for(P(sys.argv[3]))\n"
        "assert sampling==baseline_sampling")
    code = replace_once(code,
        "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner']:",
        "for name in ['qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite']:")
    code = code.replace("imports['qwen35_answer_finite']['sha256']=='@ANSWER_SHA@'",
                        "imports['qwen35_answer_finite']['sha256']=='@HEAD_SHA@'")
    code = replace_once(code,
        "assert imports['qwen35_dense_finite_runner']['sha256']=='5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'",
        "assert imports['qwen35_dense_finite_runner']['path']==str(dt/'clean/qwen35/qwen35_dense_finite_runner.py')\n"
        "assert imports['qwen35_dense_finite_runner']['sha256']=='@RUNNER_SHA@'\n"
        "assert imports['qwen35_gdn_finite']['path']==str(dt/'clean/qwen35/qwen35_gdn_finite.py')\n"
        "assert imports['qwen35_gdn_finite']['sha256']=='@GDN_SHA@'\n"
        "assert imports['reward_readout']['sha256']=='@READOUT_SHA@'")
    code = replace_once(code, "options=options,sampling=sampling,imports=imports,",
        "options=options,sampling=sampling,baseline_options=baseline_options,"
        "baseline_sampling=baseline_sampling,imports=imports,")
    code = replace_once(code,
        "[env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output)]",
        "[env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output),old['output']]")
    code = replace_once(code, "old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']",
        "old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']\n"
        "assert inspection['baseline_options']==old_options\n"
        "assert inspection['sampling']==inspection['baseline_sampling']\n"
        "prior_imports=prior['actual_CPU_imports']\n"
        "for name,before in prior_imports.items():\n"
        " after=inspection['imports'][name]\n"
        " if name=='qwen35_dense_finite_runner':continue\n"
        " assert after['sha256']==before['sha256'],name\n"
        " if name!='qwen35_answer_finite':\n"
        "  assert after['path']==before['path'] and after['resolved_path']==before['resolved_path'],name")
    code = replace_once(code, "assert set(differences)<=allowed,differences",
        "assert set(differences)<=allowed,differences\n"
        "for key,change in differences.items():\n"
        " assert change['after']==change['before'].replace(old['output'],str(output)),key\n"
        "assert env['DT_ENVIRONMENT_JSON']==prior['environment']['DT_ENVIRONMENT_JSON']\n"
        "environment_differences={key for key in env.keys()|prior['environment'].keys()\n"
        " if env.get(key)!=prior['environment'].get(key)}\n"
        "assert environment_differences<= {'DT_ROOT','PYTHONPATH'},environment_differences")
    code = replace_once(code,
        "('qwen35_answer_finite.py','prepare_appworld_head_memory.py','submit_prepared_direct_targets.py')",
        "('qwen35_dense_finite_runner.py','qwen35_gdn_finite.py',"
        "'prepare_appworld_gpu_lifetime.py','submit_prepared_direct_targets.py')")
    code = code.replace('head_memory_preparation=', 'gpu_lifetime_preparation=')
    code = replace_once(code, "candidate_answer=binding(answer),",
        "candidate_files={name:binding(dt/name) for name in replacements},\n"
        "                retained_head=binding(dt/'clean/qwen35/qwen35_answer_finite.py'),\n"
        "                retained_readout=binding(P(prior['entry'])/'reward_readout.py'),")
    code = code.replace('prepare_appworld_head_memory.py', 'prepare_appworld_gpu_lifetime.py')
    code = code.replace('Real executed joint action targets; isolated full-vocabulary target-row temporary tiling;',
        'Real executed joint action targets; isolated last-use release independent of capture transport;')
    code = code.replace('Preparation and CPU imports only;',
        'Prepared-only linked DT tree; exactly two lifetime files replaced; CPU imports/configuration only;')
    code = code.replace('plus explicit answer/runner imports',
        'plus explicit unchanged head and candidate runner/GDN imports')
    code = code.replace("answer=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],",
        "head=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],\n"
        "    gdn=inspection['imports']['qwen35_gdn_finite'],")
    values = {'@ROOT@': repr(root), '@OUT@': repr(out), '@COMMIT@': commit,
        '@HEAD_SHA@': HEAD_SHA, '@READOUT_SHA@': READOUT_SHA,
        '@RUNNER_SHA@': REPLACEMENTS['clean/qwen35/qwen35_dense_finite_runner.py'][1],
        '@GDN_SHA@': REPLACEMENTS['clean/qwen35/qwen35_gdn_finite.py'][1],
        '@ORIGINAL_PREPARE_PATH@': str(BASE_PREPARE.resolve()).replace('\\', '/'),
        '@ORIGINAL_PREPARE_SHA@': BASE_PREPARE_SHA}
    for token, value in values.items():
        code = code.replace(token, value)
    assert not re.search(r'@[A-Z_]+@', code), 'Unfilled preparation-owner template token'
    ast.parse(code)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.commit):
        parser.error('--commit must be the actual 40-character code commit')
    assert sha(BASE_PREPARE) == BASE_PREPARE_SHA
    assert sha(SUBMIT) == SUBMIT_SHA
    spec = importlib.util.spec_from_file_location('head_memory_preparation_owner', BASE_PREPARE)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    stage = owner.stage
    candidates = [HERE/'candidate'/Path(name).name for name in REPLACEMENTS]
    for name, path in zip(REPLACEMENTS, candidates):
        assert sha(path) == REPLACEMENTS[name][1], path
    out = stage.ROOT + '/candidates/direct-target-gpu-lifetime-20261007-v1'
    code = render_owner_code(owner, stage.ROOT, out, args.commit)
    bundle = HERE/'setup-source.tar'
    with tarfile.open(bundle, 'w') as archive:
        for path in candidates:
            archive.add(path, arcname=path.name)
        archive.add(Path(__file__), arcname='prepare_appworld_gpu_lifetime.py')
        archive.add(SUBMIT, arcname='submit_prepared_direct_targets.py')
    script = 'set -eu\nsource ' + stage.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
    (HERE/'prepare-command.sh').write_text(script, encoding='utf-8')
    # Existing preparation transport only; submit.main/submit_one are not called.
    subprocess.run(stage.SSH+['mkdir', '-p', out+'/setup'], check=True)
    subprocess.run(stage.SCP+[str(bundle), stage.SSH[-1]+':'+out+'/setup-source.tar'], check=True)
    subprocess.run(stage.SSH+['tar', '-xf', out+'/setup-source.tar', '-C', out+'/setup'], check=True)
    result = subprocess.run(stage.SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=240)
    (HERE/'prepare.stdout.txt').write_bytes(result.stdout)
    (HERE/'prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()


if __name__ == '__main__':
    main()
