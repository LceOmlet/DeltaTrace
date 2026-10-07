"""Prepare an isolated Phase1 scheduling comparison using the prior FA owner."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EFF = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = EFF / 'individual-prefix-owner-candidate-v1/candidate/vendor_fa_finite_p1_bf16_d256.cu'
    assert sha(source) == '9ebcef18cec94f45a875694acac0fed4b46064f13d3d0af1a4581c931df54c8b'
    original = source.read_text()
    candidate = original
    for old, new in (
        ('if constexpr(Phase!=0) {\n        auto load_owner=', 'if constexpr(Phase==2) {\n        auto load_owner='),
        ('if constexpr(Phase==0) flash::copy<false,true,true>(global_copy,fromGA',
         'if constexpr(Phase!=2) flash::copy<false,true,true>(global_copy,fromGA'),
        ('if constexpr(Phase==0) flash::gemm_opt(acc,regA',
         'if constexpr(Phase!=2) flash::gemm_opt(acc,regA'),
    ):
        assert candidate.count(old) == 1
        candidate = candidate.replace(old, new)
    target = HERE / 'candidate'
    target.mkdir(exist_ok=True)
    (target / source.name).write_text(candidate)
    wrapper = EFF / 'individual-prefix-owner-candidate-v1/candidate/vendor_fa_finite_bf16_d256.py'
    (target / wrapper.name).write_bytes(wrapper.read_bytes())
    assert sha(wrapper) == '3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'
    patch = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
        fromfile='current-4f42-owner', tofile='isolated-Phase1-original-shared-A'))
    (HERE / 'phase1-owner.patch').write_text(patch)
    assets = json.loads((EFF / 'finite-fa-current-owner-profile-v1/assets.json').read_bytes())
    prior = json.loads((EFF / 'finite-fa-center-query-fusion-candidate-v1/source-preparation.json').read_bytes())
    execution = json.loads((EFF / 'individual-prefix-owner-candidate-v1/native-representation-candidate/offline-checks-v1/independent-fa-results/execution.json').read_bytes())
    command = next(row['command'] for row in execution['checks'] if '--scalar-library' in row['command'])
    def item(flag, digest):
        return dict(path=command[command.index(flag) + 1], sha256=digest)
    assets.update(status='unaccepted_isolated_source_only_not_deployed',
        scope='Only Phase1 uses the original FA default shared-A GEMM instead of retaining three query fragments. Phase0/2, formulas, tile, dtype and all exported ABIs unchanged. No production overwrite.',
        original_source=dict(path=str(source), sha256=sha(source)),
        candidate_source=dict(path=str(target/source.name), sha256=sha(target/source.name)),
        patch=dict(path=str(HERE/'phase1-owner.patch'), sha256=sha(HERE/'phase1-owner.patch')),
        original_shared_A_commit='3524155570d9b5d33ac731dc9ca59172bd4dc0ec',
        register_retention_commit='0289dd29f52e87ebbfd5c40d3c8ef7e86ade3465',
        official_verifier=item('--saved-verifier','7ff11d9d800a2c233b019213ae0fa9a5b603dfea51dfffabe6b9296a74aab9a9'),
        official_sources=dict(path=command[command.index('--sources')+1]),
        scalar_wrapper=item('--scalar-wrapper','f5ea2f67af2a47b2a736545b22ff6683336be14be505060674fce6f01de3f15c'),
        scalar_library=item('--scalar-library','5d2af760abb2684401682029e41ac68aefc58ed2796082d435874a2b74bbcebb'),
        environment_json=dict(path=command[command.index('--environment-json')+1]),
        saved_operands=prior['actual_saved_assets'])
    (HERE/'source-preparation.json').write_text(json.dumps(assets,indent=2)+'\n')
    print(json.dumps(dict(candidate=assets['candidate_source'],patch=patch,status=assets['status'])))


if __name__ == '__main__':
    main()
