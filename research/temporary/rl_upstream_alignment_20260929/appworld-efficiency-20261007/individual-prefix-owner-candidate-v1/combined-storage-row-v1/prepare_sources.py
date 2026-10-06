"""Compose the frozen row extension with the verified boundary-row owner.

Preparation only: no Torch import, model call, remote operation or deployment.
"""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROW = ROOT.parent/'native-representation-candidate'
REPO = next(p for p in ROOT.parents if (p/'experiments/rl/PLAN.md').exists())
VERIFIED_COMMIT = 'ee0fcee4596ac60888fe0de5ab101a893ef10c6e'
VERIFIED = {
    'qwen35_native_prefix_artifacts.py': (
        REPO/'deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py',
        '4a461f2168d07b5ce88dbc8468e6a0d3eaae2a2c1d879cd5281347098e2ea5a8'),
    'native_prefix_leases.py': (REPO/'experiments/rl/native_prefix_leases.py',
        '6d2aecb0a47c3f4f6bd0e128b6da304ce7d53a602cbb331e18eea8edc745acc8'),
}
ROW_SHAS = {
    'qwen35_native_prefix_artifacts.py': '50af8daf2ce5beb44c474801dbdd3bc00e736d2f92c530a0ef17b1c0168d35ee',
    'native_prefix_leases.py': '54ad5ae1a07cefcbc4002fa63e7450d8281493a93567ad3df46cb7e9c2e74260',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    helper_path = ROW/'prepare_sources.py'
    assert sha(helper_path.read_bytes()) == 'abf5527d2af0daa7b14f67b7556f0666d33f405202467c4efc3158602c58475e'
    spec = importlib.util.spec_from_file_location('_frozen_row_source_extension', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    changes = []
    for name, (path, expected) in VERIFIED.items():
        raw = path.read_bytes()
        assert sha(raw) == expected, path
        row_raw = (ROW/'candidate'/name).read_bytes()
        assert sha(row_raw) == ROW_SHAS[name], name
        text = raw.decode('utf-8')
        if name.startswith('qwen35'):
            new = helper.patch_artifact(text)
            # Reuse the verified owner's original-row -> stored-row mapping.
            # Each consumer's boundary n replaces the scalar boundary here.
            old = "            conv = torch.cat([state[0][row:row+1] for state, (_, row) in zip(states, sources)])\n            recurrent = torch.cat([state[1][row:row+1] for state, (_, row) in zip(states, sources)])\n"
            replacement = "            if all(getattr(source, 'boundary_rows', None) is None for source, row in sources):\n                conv = torch.cat([state[0][row:row+1] for state, (_, row) in zip(states, sources)])\n                recurrent = torch.cat([state[1][row:row+1] for state, (_, row) in zip(states, sources)])\n            else:\n                rows = [row if getattr(source, 'boundary_rows', None) is None\n                        else source.boundary_rows[n][row] for (source, row), n in zip(sources, lengths)]\n                conv = torch.cat([state[0][row:row+1] for state, row in zip(states, rows)])\n                recurrent = torch.cat([state[1][row:row+1] for state, row in zip(states, rows)])\n"
            # The scalar storage owner's original None branch has the same
            # cat text. Restrict this replacement to the new row helper.
            start = new.index('def _compose_row_prefix_cache(')
            prefix, row_helper = new[:start], new[start:]
            new = prefix + helper.replace_once(row_helper, old, replacement)
        else:
            signature = 'def prepare_native_prefix_leases(runner, requests, *, minibatch_size, eos_token_id, boundary_row_storage=False, observe_boundary_rows=None):'
            bare = 'def prepare_native_prefix_leases(runner, requests, *, minibatch_size, eos_token_id):'
            # The frozen extension patches only the original row seams. Keep
            # all verified storage body bytes and restore its keyword ABI.
            new = helper.patch_lease(helper.replace_once(text, signature, bare))
            new = helper.replace_once(new,
                bare[:-2]+', individual_prefixes=False):',
                signature[:-2]+', individual_prefixes=False):')
        compile(ast.parse(new), name, 'exec')
        for directory, data in [('baseline', raw), ('candidate', new.encode('utf-8'))]:
            target = ROOT/directory/name
            target.parent.mkdir(exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(data)
        for label, old in [('verified-storage', text), ('uncompressed-row', row_raw.decode('utf-8'))]:
            patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                fromfile=label+'/'+name, tofile='combined-storage-row/'+name))
            (ROOT/(name+'.'+label+'.patch')).write_text(patch, encoding='utf-8', newline='\n')
        changes.append(dict(name=name, verified_storage_source=str(path), verified_storage_sha256=expected,
            uncompressed_row_sha256=ROW_SHAS[name], candidate_sha256=sha(new.encode('utf-8'))))
    manifest = dict(status='prepared_only_not_deployed', verified_storage_commit=VERIFIED_COMMIT,
        original_artifact_commit='8e7dd71258b2173ae0be0784e7783a49c9cd5b94',
        original_lease_commit='712795d1812b6b29b9626ab2ab88f0bd98564d1e',
        original_artifact_sha256='06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16',
        original_lease_sha256='d5b539c7ffb8a431374cf79f4b995ef6f4138c3e6e289890409b6b79dbad393e',
        frozen_row_preparer=dict(path=str(helper_path), sha256=sha(helper_path.read_bytes())),
        preparer_sha256=sha(Path(__file__).read_bytes()), source_changes=changes,
        default_contract='Unspecified individual_prefixes preserves the complete verified storage owner AST; boundary_row_storage remains False by default.',
        scope='Only existing storage selection and per-row boundary mapping composition. No runner, target, S/L, cuts, kernel, canonical grouping, PPO, GPU, remote or formal change.')
    (ROOT/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
