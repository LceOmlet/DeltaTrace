"""Prepare isolated transport seams and imports-only accelerated owners.

This module reads and patches source only. It imports no model, Torch, kernel,
offloader, stream or cache. The generated modules are private candidates, and
all new behavior defaults off. Their finite formulas remain the owner formulas.
"""
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping


MODULES = {
    'native_attention_capture.py': 'native_root_attention_transport_candidate.py',
    'native_dense_attention_capture.py': 'native_root_dense_attention_transport_candidate.py',
    'qwen35_decoder_finite.py': 'native_root_decoder_transport_candidate.py',
    'qwen35_gdn_finite.py': 'native_root_gdn_transport_candidate.py',
}
ACCELERATED_MODULES = {
    'qwen35_retained_capture.py': 'native_root_retained_transport_candidate.py',
    'qwen35_code_local_capture.py': 'native_root_code_local_transport_candidate.py',
}
EXPECTED_OWNER_SHA256 = {
    'native_attention_capture.py': '2a226ac35302e933920aa096fe962db7ed724c1a478c247ef5700b253461ab74',
    'native_dense_attention_capture.py': 'd02c51325e852289358b74f5756f285af0148915c9a3a0c2d8c535a945945604',
    'qwen35_decoder_finite.py': '047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197',
    'qwen35_gdn_finite.py': 'fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca',
}
EXPECTED_ACCELERATED_SHA256 = {
    'qwen35_retained_capture.py': 'c02ce3a54147dcf37e3f97d1d26cd38423ee1faaa3566ecc2999a2fe8ee59870',
    'qwen35_code_local_capture.py': '993ae72a49dc90901bfcbb7817e5fd32ad6004fbe630a190b775c6f1e77fbe72',
}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_once(text: str, before: str, after: str) -> str:
    count = text.count(before)
    if count != 1:
        raise ValueError(f'Fixed owner seam occurs {count} times: {before[:100]!r}')
    return text.replace(before, after, 1)


def _header(text: str, class_name: str, method_name: str) -> str:
    owner = next(n for n in ast.parse(text).body
                 if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == method_name)
    return text.splitlines(keepends=True)[method.lineno - 1]


def _method_text(text: str, class_name: str, method_name: str) -> str:
    owner = next(n for n in ast.parse(text).body
                 if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == method_name)
    return ''.join(text.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])


def _add_header_keywords(header: str, suffix: str) -> str:
    if not header.rstrip().endswith('):'):
        raise ValueError('Expected the pinned single-line owner signature.')
    return header.rstrip('\n')[:-2] + suffix + '):\n'


def _attention(text: str) -> str:
    header = _header(text, 'NativeAttentionCapture', '__init__')
    text = _replace_once(text, header,
                         _add_header_keywords(header, ', defer_host_sync=False'))
    text = _replace_once(text, '        self.pinned_host = pinned_host\n',
                         '        self.pinned_host = pinned_host\n'
                         '        self.defer_host_sync = defer_host_sync\n')
    return _replace_once(text, '        if self.pinned_host:\n',
                          '        if self.pinned_host and not self.defer_host_sync:\n')


def _dense(text: str) -> str:
    text = _replace_once(text,
        'from native_attention_capture import NativeAttentionCapture, _metadata_scalar\n',
        'from native_root_attention_transport_candidate import NativeAttentionCapture, _metadata_scalar\n')
    # This owner deliberately uses a multiline constructor signature.
    text = _replace_once(text,
        "                 native_dense_function, destination='cpu', *, copy_tensors=True, retained_names=None, preserve_strides=False, pinned_host=False):\n",
        "                 native_dense_function, destination='cpu', *, copy_tensors=True, retained_names=None, preserve_strides=False, pinned_host=False, defer_host_sync=False):\n")
    return _replace_once(text,
        '                         copy_tensors=copy_tensors, retained_names=retained_names, preserve_strides=preserve_strides,pinned_host=pinned_host)\n',
        '                         copy_tensors=copy_tensors, retained_names=retained_names, preserve_strides=preserve_strides,pinned_host=pinned_host,defer_host_sync=defer_host_sync)\n')


def _decoder(text: str) -> str:
    text = _replace_once(text, 'import torch\n',
        'import torch\nfrom native_root_attention_transport_candidate import copy_capture_tensor\n')
    header = _header(text, 'NativeDecoderCapture', '__init__')
    text = _replace_once(text, header, _add_header_keywords(
        header, ', preserve_strides=False, pinned_host=False, defer_host_sync=False'))
    text = _replace_once(text,
        '        self.copy_tensors=copy_tensors;self.retained_names=retained_names\n',
        '        self.copy_tensors=copy_tensors;self.retained_names=retained_names\n'
        '        self.preserve_strides=preserve_strides;self.pinned_host=pinned_host\n'
        '        self.defer_host_sync=defer_host_sync\n')
    retain = _method_text(text, 'NativeDecoderCapture', 'retain')
    original_assignment = '            self.values[name]=value.detach().to(self.destination,copy=self.copy_tensors)\n'
    changed_retain = _replace_once(retain, original_assignment,
        '            if self.preserve_strides or self.pinned_host:\n'
        '                self.values[name]=copy_capture_tensor(value,self.destination,\n'
        '                    copy=self.copy_tensors,preserve_strides=self.preserve_strides,pinned_host=self.pinned_host)\n'
        '            else:\n'
        '    ' + original_assignment)
    text = _replace_once(text, retain, changed_retain)
    exit_body = _method_text(text, 'NativeDecoderCapture', '__exit__')
    text = _replace_once(text, exit_body, exit_body +
        "        if self.pinned_host and torch.device(self.destination).type=='cpu' and not self.defer_host_sync:\n"
        '            torch.cuda.current_stream().synchronize()\n')
    header = next(line for line in text.splitlines(keepends=True)
                  if line.startswith('def decoder_finite_pullback('))
    text = _replace_once(text, header,
                         _add_header_keywords(header, ',restore_captures=False'))
    text = _replace_once(text, '    c=values\n    mnorm=boundaries.mlp(',
        '    c=values\n'
        '    def restore(*names):\n'
        '        if restore_captures:\n'
        '            for name in names:\n'
        "                c[name]=copy_capture_tensor(c[name],'cuda',preserve_strides=True)\n"
        "    restore('gate_output','up_output','silu_output')\n"
        '    mnorm=boundaries.mlp(')
    text = _replace_once(text, "    mmixer=boundaries.norm_residual(c['post_norm_input']",
        "    restore('post_norm_input')\n"
        "    mmixer=boundaries.norm_residual(c['post_norm_input']")
    return _replace_once(text, "    mx=boundaries.norm_residual(c['input_norm_input']",
        "    restore('input_norm_input')\n"
        "    mx=boundaries.norm_residual(c['input_norm_input']")


def _gdn(text: str) -> str:
    text = _replace_once(text,
        'from qwen35_decoder_finite import _linear_transpose, _linear_weights\n',
        'from native_root_decoder_transport_candidate import _linear_transpose, _linear_weights\n')
    text = _replace_once(text,
        'from native_attention_capture import copy_capture_tensor\n',
        'from native_root_attention_transport_candidate import copy_capture_tensor\n')
    header = _header(text, 'NativeGDNCapture', '__init__')
    text = _replace_once(text, header,
                         _add_header_keywords(header, ', defer_host_sync=False'))
    text = _replace_once(text, '        self.pinned_host=pinned_host\n',
        '        self.pinned_host=pinned_host\n        self.defer_host_sync=defer_host_sync\n')
    return _replace_once(text,
        "        if self.pinned_host and torch.device(self.device).type=='cpu':\n",
        "        if self.pinned_host and torch.device(self.device).type=='cpu' and not self.defer_host_sync:\n")


def _retained_imports(text: str) -> str:
    for before, after in (
            ('from native_attention_capture import copy_capture_tensor\n',
             'from native_root_attention_transport_candidate import copy_capture_tensor\n'),
            ('from qwen35_decoder_finite import NativeDecoderCapture as _Decoder\n',
             'from native_root_decoder_transport_candidate import NativeDecoderCapture as _Decoder\n'),
            ('from native_dense_attention_capture import NativeDenseAttentionCapture as _Attention\n',
             'from native_root_dense_attention_transport_candidate import NativeDenseAttentionCapture as _Attention\n'),
            ('from qwen35_gdn_finite import NativeGDNCapture as _GDN\n',
             'from native_root_gdn_transport_candidate import NativeGDNCapture as _GDN\n')):
        text = _replace_once(text, before, after)
    return text


def _code_local_imports(text: str) -> str:
    text = _replace_once(text,
        'from ..native_capture_events import LocalCaptureEvents\n',
        'from accelerated.native_capture_events import LocalCaptureEvents\n')
    for names in ('NativeDecoderCapture', 'NativeGDNCapture as _GDN',
                  'NativeDenseAttentionCapture as _Attention'):
        text = _replace_once(text, f'from .qwen35_retained_capture import {names}\n',
            f'from native_root_retained_transport_candidate import {names}\n')
    return text


def default_projection(candidate: ast.Module, owner_name: str) -> ast.Module:
    """Project only these disabled transport declarations back to owner AST.

    This is a source contract, not a numerical tolerance or a model execution.
    The no-op local restore calls are removed only for restore_captures=False.
    """
    private_imports = {Path(v).stem: Path(k).stem
                       for k, v in (MODULES | ACCELERATED_MODULES).items()}

    def self_attribute(node, name):
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
            and node.value.id == 'self' and node.attr == name

    class Project(ast.NodeTransformer):
        current_class = None
        current_function = None

        def visit_ClassDef(self, node):
            previous = self.current_class
            self.current_class = node.name
            node = self.generic_visit(node)
            self.current_class = previous
            return node

        def visit_FunctionDef(self, node):
            previous = self.current_function
            self.current_function = node.name
            if previous == 'decoder_finite_pullback' and node.name == 'restore':
                self.current_function = previous
                return None
            removed = {'defer_host_sync'}
            if self.current_class == 'NativeDecoderCapture' and node.name == '__init__':
                removed |= {'preserve_strides', 'pinned_host'}
            if node.name == 'decoder_finite_pullback':
                removed |= {'restore_captures'}
            pairs = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                     if a.arg not in removed]
            node.args.kwonlyargs = [a for a, _ in pairs]
            node.args.kw_defaults = [d for _, d in pairs]
            node = self.generic_visit(node)
            self.current_function = previous
            return node

        def visit_ImportFrom(self, node):
            if owner_name == 'qwen35_code_local_capture.py':
                if node.module == 'accelerated.native_capture_events':
                    node.module, node.level = 'native_capture_events', 2
                elif node.module == 'native_root_retained_transport_candidate':
                    node.module, node.level = 'qwen35_retained_capture', 1
                return node
            if owner_name == 'qwen35_decoder_finite.py' \
                    and node.module == 'native_root_attention_transport_candidate':
                return None
            if node.module in private_imports:
                node.module = private_imports[node.module]
            return node

        def visit_Assign(self, node):
            removed = {'defer_host_sync'}
            if self.current_class == 'NativeDecoderCapture':
                removed |= {'preserve_strides', 'pinned_host'}
            if node.targets and all(any(self_attribute(t, name) for name in removed)
                                    for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if self.current_class == 'NativeDecoderCapture':
                if self.current_function == 'retain' and isinstance(node.test, ast.BoolOp) \
                        and isinstance(node.test.op, ast.Or) \
                        and all(any(self_attribute(t, name) for name in
                                    ('preserve_strides', 'pinned_host')) for t in node.test.values):
                    return [self.visit(n) for n in node.orelse]
                if self.current_function == '__exit__' \
                        and any(self_attribute(n, 'defer_host_sync') for n in ast.walk(node.test)):
                    return None
            if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
                node.test.values = [n for n in node.test.values if not
                    (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
                     and self_attribute(n.operand, 'defer_host_sync'))]
                if len(node.test.values) == 1:
                    node.test = node.test.values[0]
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if self.current_function == 'decoder_finite_pullback' \
                    and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'restore':
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node.keywords = [k for k in node.keywords if k.arg != 'defer_host_sync']
            return self.generic_visit(node)

    return ast.fix_missing_locations(Project().visit(copy.deepcopy(candidate)))


def _source_contracts(owner_name: str, original: bytes, candidate: bytes):
    before, after = ast.parse(original), ast.parse(candidate)
    dump = lambda n: ast.dump(n, include_attributes=False)
    projected = default_projection(after, owner_name)
    if dump(before) != dump(projected):
        raise ValueError(f'Disabled transport projection changed owner AST: {owner_name}')
    after_functions = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    unchanged = [n.name for n in before.body if isinstance(n, ast.FunctionDef)
                 and dump(n) == dump(after_functions[n.name])]
    after_classes = {n.name: n for n in after.body if isinstance(n, ast.ClassDef)}
    unchanged_classes = [n.name for n in before.body if isinstance(n, ast.ClassDef)
                         and dump(n) == dump(after_classes[n.name])]
    imports_only = owner_name in ACCELERATED_MODULES
    if imports_only:
        before_nonimports = [n for n in before.body
                             if not isinstance(n, (ast.Import, ast.ImportFrom))]
        after_nonimports = [n for n in after.body
                            if not isinstance(n, (ast.Import, ast.ImportFrom))]
        if [dump(n) for n in before_nonimports] != [dump(n) for n in after_nonimports]:
            raise ValueError(f'Imports-only accelerated owner changed a non-import AST: {owner_name}')
    projected_functions = {n.name: n for n in projected.body if isinstance(n, ast.FunctionDef)}
    return dict(default_projection_equals_original_module_ast=True,
        default_projection_scope=(
            'Restore only original imports (including original relative-import level); all non-import AST is identical; not a numerical or runtime acceptance claim'
            if imports_only else
            'Remove disabled transport fields/keywords, private import names, and no-op staged restore calls only; not a numerical or runtime acceptance claim'),
        unchanged_top_level_function_ast=unchanged,
        unchanged_class_ast=unchanged_classes,
        imports_only=imports_only,
        all_non_import_ast_identical=True if imports_only else None,
        decoder_finite_default_projection_equals_original=(
            dump(next(n for n in before.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'decoder_finite_pullback')) == dump(
                projected_functions['decoder_finite_pullback'])
            if 'decoder_finite_pullback' in projected_functions else None))


def patched_sources(originals: Mapping[str, bytes], *,
                    accelerated_originals: Mapping[str, bytes] | None = None):
    """Return private owner candidates, preserving supplied original bytes."""
    if set(originals) != set(MODULES):
        raise ValueError('Supply exactly the four recorded original owner modules.')
    patches = {'native_attention_capture.py': _attention,
               'native_dense_attention_capture.py': _dense,
               'qwen35_decoder_finite.py': _decoder,
               'qwen35_gdn_finite.py': _gdn}
    module_names = dict(MODULES)
    supplied = dict(originals)
    if accelerated_originals is not None:
        if set(accelerated_originals) != set(ACCELERATED_MODULES):
            raise ValueError('Supply exactly the two recorded accelerated owner modules.')
        patches['qwen35_retained_capture.py'] = _retained_imports
        patches['qwen35_code_local_capture.py'] = _code_local_imports
        module_names.update(ACCELERATED_MODULES)
        supplied.update(accelerated_originals)
    candidates, files = {}, {}
    for name, original in supplied.items():
        source = original.decode('utf8')
        newline = '\r\n' if '\r\n' in source else '\n'
        after = patches[name](source.replace('\r\n', '\n'))
        compile(after, module_names[name], 'exec')
        candidate = after.replace('\n', newline).encode('utf8')
        candidates[module_names[name]] = candidate
        files[name] = dict(candidate_filename=module_names[name],
                           owner_sha256=sha256(original), candidate_sha256=sha256(candidate),
                           source_contracts=_source_contracts(name, original, candidate))
    return candidates, dict(status='prepared_only_not_deployed',
        scope='Original capture transport and exit-fence interfaces; accelerated owners use imports-only private bindings',
        accelerated_import_bindings_prepared=accelerated_originals is not None,
        accelerated_runtime_backend_verified=False,
        new_defaults=dict(preserve_strides=False, pinned_host=False,
                          defer_host_sync=False, restore_captures=False),
        finite_math_changed=False, parameter_prepare_release_changed=False,
        model_forwards_added=0, new_offloader_stream_or_cache=False, files=files)


def prepare(owner_root: Path, candidate_directory: Path,
            expected_owner_sha256: Mapping[str, str] | None = None, *,
            accelerated_owner_root: Path | None = None,
            expected_accelerated_sha256: Mapping[str, str] | None = None):
    owner_root, candidate_directory = Path(owner_root), Path(candidate_directory)
    accelerated_owner_root = (owner_root.parent.parent / 'accelerated/qwen35'
                              if accelerated_owner_root is None else Path(accelerated_owner_root))
    if candidate_directory.resolve() in (owner_root.resolve(), accelerated_owner_root.resolve()):
        raise ValueError('Private candidates cannot be written into the owner directory.')
    expected = EXPECTED_OWNER_SHA256 if expected_owner_sha256 is None else expected_owner_sha256
    if set(expected) != set(MODULES):
        raise ValueError('Expected SHA256 map must identify all four owner files.')
    originals = {name: (owner_root / name).read_bytes() for name in MODULES}
    accelerated_expected = (EXPECTED_ACCELERATED_SHA256 if expected_accelerated_sha256 is None
                            else expected_accelerated_sha256)
    if set(accelerated_expected) != set(ACCELERATED_MODULES):
        raise ValueError('Expected accelerated SHA256 map must identify both owner files.')
    accelerated_originals = {name: (accelerated_owner_root / name).read_bytes()
                             for name in ACCELERATED_MODULES}
    for name, original in originals.items():
        actual = sha256(original)
        if actual != expected[name]:
            raise ValueError(f'Owner SHA256 mismatch for {name}: {actual} != {expected[name]}')
    for name, original in accelerated_originals.items():
        actual = sha256(original)
        if actual != accelerated_expected[name]:
            raise ValueError(f'Owner SHA256 mismatch for {name}: {actual} != {accelerated_expected[name]}')
    candidates, manifest = patched_sources(originals, accelerated_originals=accelerated_originals)
    # A distinct directory preserves existing candidates and their receipts.
    candidate_directory.mkdir(parents=True, exist_ok=False)
    for filename, candidate in candidates.items():
        (candidate_directory / filename).write_bytes(candidate)
    for name, original in (originals | accelerated_originals).items():
        filename = (MODULES | ACCELERATED_MODULES)[name]
        source_root = owner_root if name in MODULES else accelerated_owner_root
        diff = ''.join(difflib.unified_diff(original.decode('utf8').splitlines(True),
            candidates[filename].decode('utf8').splitlines(True),
            fromfile=name, tofile=filename))
        (candidate_directory / (filename + '.patch')).write_text(diff, encoding='utf8', newline='')
        manifest['files'][name].update(owner_source=str((source_root / name).resolve()),
            candidate_source=str((candidate_directory / filename).resolve()))
    manifest.update(owner_root=str(owner_root.resolve()),
                    accelerated_owner_root=str(accelerated_owner_root.resolve()),
                    candidate_directory=str(candidate_directory.resolve()),
                    preparer_sha256=sha256(Path(__file__).read_bytes()))
    (candidate_directory / 'prepared.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-root', type=Path, required=True)
    parser.add_argument('--accelerated-owner-root', type=Path)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--expected-sha-manifest', type=Path)
    parser.add_argument('--expected-accelerated-sha-manifest', type=Path)
    args = parser.parse_args()
    expected = None if args.expected_sha_manifest is None else json.loads(
        args.expected_sha_manifest.read_text(encoding='utf8'))
    accelerated_expected = (None if args.expected_accelerated_sha_manifest is None else json.loads(
        args.expected_accelerated_sha_manifest.read_text(encoding='utf8')))
    print(json.dumps(prepare(args.owner_root, args.candidate_dir, expected,
        accelerated_owner_root=args.accelerated_owner_root,
        expected_accelerated_sha256=accelerated_expected), indent=2))


if __name__ == '__main__':
    main()
