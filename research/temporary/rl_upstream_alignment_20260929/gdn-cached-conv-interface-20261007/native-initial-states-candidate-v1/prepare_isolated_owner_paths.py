"""Build only diagnostic owner paths using original-file symlinks.

No imports of torch/transformers, environment installation, model or GPU call.
The original owner tree and installed HF file are only read, never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt-root', type=Path, required=True)
    parser.add_argument('--hf-model', type=Path, required=True)
    parser.add_argument('--candidate-sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.dt_root.resolve()
    hf_source = args.hf_model.resolve()
    prepared = args.candidate_sources.resolve()
    output = args.output.resolve()
    assert not output.exists(), output
    originals = {
        source/'clean/qwen35/qwen35_dense_finite_runner.py': '6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa',
        source/'clean/qwen35/qwen35_gdn_finite.py': 'fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca',
        hf_source: 'f7e1a804fa12684bd1cc225c85cdf5f0b5996f30d66263f11de13449f53272be',
    }
    for path, expected in originals.items():
        assert digest(path) == expected, path
    candidates = {
        'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py': 'e9c7576486f742c26f895cd4078891a98d94c189e84a6e565fc8042a74aabcab',
        'deltatrace/clean/qwen35/qwen35_gdn_finite.py': 'ef55ce08dec9374304018b43b9f85510fca36054408c439ab1109dca8ac23ca9',
        'transformers/models/qwen3_5/modeling_qwen3_5.py': '59f9c339e3c01672b67ba09b0e56e2e960ba28c6c3d9975faadf00d15ee0b4c8',
    }
    for relative, expected in candidates.items():
        assert digest(prepared/relative) == expected, relative

    isolated_dt = output/'deltatrace'
    isolated_dt.mkdir(parents=True)

    def link_children(old, new, exception):
        for path in old.iterdir():
            if path.name in exception:
                continue
            (new/path.name).symlink_to(path, target_is_directory=path.is_dir())

    # Preserve every original owner file through links, including the original
    # factory, prefix-bank implementation, finite libraries and resource JSON.
    link_children(source, isolated_dt, {'clean'})
    (isolated_dt/'clean').mkdir()
    link_children(source/'clean', isolated_dt/'clean', {'qwen35'})
    (isolated_dt/'clean/qwen35').mkdir()
    names = {'qwen35_dense_finite_runner.py', 'qwen35_gdn_finite.py'}
    link_children(source/'clean/qwen35', isolated_dt/'clean/qwen35', names)
    for name in names:
        shutil.copyfile(prepared/'deltatrace/clean/qwen35'/name, isolated_dt/'clean/qwen35'/name)
    isolated_hf = output/'hf-qwen35'
    isolated_hf.mkdir()
    shutil.copyfile(prepared/'transformers/models/qwen3_5/modeling_qwen3_5.py',
                    isolated_hf/'modeling_qwen3_5.py')

    # This path-only opt-in occurs before the original AutoModel import. It
    # retains the canonical official module name, original lazy package and
    # all other original package paths. No module/class/forward replacement.
    site = output/'sitecustomize.py'
    site.write_bytes(b'''import os\nif os.environ.get('DT_PREFIX_NATIVE_CONV_INITIAL_STATES') == '1':\n    import importlib, sys\n    from pathlib import Path\n    root = Path(os.environ['DT_CONV_ISOLATED_IMPORT_ROOT'])\n    name = 'transformers.models.qwen3_5.modeling_qwen3_5'\n    if name in sys.modules:\n        raise RuntimeError('Original HF model imported before candidate namespace path setup')\n    package = importlib.import_module('transformers.models.qwen3_5')\n    package.__path__ = [str(root/'hf-qwen35'), *package.__path__]\n''')
    record = dict(status='prepared_only_not_model_imported_not_executed',
                  originals={str(path): digest(path) for path in originals},
                  isolated_dt_root=str(isolated_dt), hf_namespace_path=str(isolated_hf),
                  candidate_sources={str(isolated_dt/'clean/qwen35'/name): digest(isolated_dt/'clean/qwen35'/name)
                                     for name in names},
                  hf_candidate=dict(path=str(isolated_hf/'modeling_qwen3_5.py'),
                                    sha256=digest(isolated_hf/'modeling_qwen3_5.py')),
                  sitecustomize=dict(path=str(site), sha256=digest(site)),
                  env=dict(DT_ROOT=str(isolated_dt), DT_CONV_ISOLATED_IMPORT_ROOT=str(output),
                           DT_PREFIX_NATIVE_CONV_INITIAL_STATES='1'),
                  pythonpath_prepend=str(output), source_symlinks_not_copies=True,
                  original_owner_bytes_unchanged=True, installs=0, gpu_calls=0)
    (output/'isolated-owner-paths.json').write_text(json.dumps(record, indent=2)+'\n')
    for path, expected in originals.items():
        assert digest(path) == expected, path
    print(json.dumps(record))


if __name__ == '__main__':
    main()
