"""Prepare an isolated, default-inert root-capture scheduling owner extension.

The original capture construction is moved once, not reimplemented. The same
attribute body, finite mathematics and FSDP parameter callbacks serve both
branches. This preparer never imports the model or launches a GPU operation.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import textwrap


RUNNER_NAME = 'qwen35_dense_finite_runner_root_tape_candidate.py'
HELPER_NAME = 'native_qwen35_root_tape.py'


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _replace_once(source: str, before: str, after: str) -> str:
    count = source.count(before)
    if count != 1:
        raise ValueError(f'Original owner boundary occurs {count} times: {before[:100]!r}')
    return source.replace(before, after, 1)


def _method(owner, name):
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def patched_owner(original: bytes):
    """Return candidate bytes and provenance; leave the supplied owner untouched."""
    source = original.decode('utf8')
    newline = '\r\n' if '\r\n' in source else '\n'
    text = source.replace('\r\n', '\n')
    parsed = ast.parse(text)
    owner = next(n for n in parsed.body if isinstance(n, ast.ClassDef)
                 and n.name == 'Qwen35DenseFiniteRunner')
    init = _method(owner, '__init__')
    attribute = _method(owner, 'attribute')
    reverse = next(n for n in attribute.body if isinstance(n, ast.For)
                   and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name)
                   and n.iter.func.id == 'reversed')
    capture_start = next(n for n in reverse.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'copy_captures'
                                 for t in n.targets))
    replay = next(n for n in reverse.body if isinstance(n, ast.FunctionDef) and n.name == 'replay')
    capture_nodes = reverse.body[reverse.body.index(capture_start):reverse.body.index(replay)]
    capture_end = capture_nodes[-1]
    lines = text.splitlines(keepends=True)
    capture_text = ''.join(lines[capture_start.lineno - 1:capture_end.end_lineno])
    factory_body = textwrap.indent(textwrap.dedent(capture_text), '        ')
    public_header = lines[attribute.lineno - 1]
    if not public_header.rstrip().endswith('):'):
        raise ValueError('Expected the fixed single-line attribute signature.')
    wrapper = (public_header +
        '        if not self.reuse_root_captures:\n'
        '            return self._attribute_owner_body(paired_ids,mask,selection,select_output_rows,observer,\n'
        '                prefix_cache_provider=prefix_cache_provider,_root_tape=None)\n'
        '        tape=NativeQwen35RootTape()\n'
        '        try:\n'
        '            return self._attribute_owner_body(paired_ids,mask,selection,select_output_rows,observer,\n'
        '                prefix_cache_provider=prefix_cache_provider,_root_tape=tape)\n'
        '        finally:\n'
        '            tape.clear()\n\n'
        '    def _make_layer_captures(self,layer,is_fa,observer,gdn_cut):\n' +
        factory_body + '        return dc,mc,offload_mixer\n\n')
    body_header = public_header.replace('def attribute(', 'def _attribute_owner_body(')
    body_header = body_header.rstrip('\n')[:-2] + ',_root_tape=None):\n'
    text = _replace_once(text, public_header, wrapper + body_header)

    init_header = lines[init.lineno - 1]
    text = _replace_once(text, init_header,
                         init_header.rstrip('\n')[:-2] + ',reuse_root_captures=False):\n')
    last_init_line = lines[init.end_lineno - 1]
    text = _replace_once(text, last_init_line,
                         last_init_line + '        self.reuse_root_captures=reuse_root_captures\n')
    text = _replace_once(text, 'import copy\n',
                         'import copy\nfrom contextlib import nullcontext\n'
                         'from native_qwen35_root_tape import NativeQwen35RootTape\n')

    cut_start = next(n for n in attribute.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'local_starts' for t in n.targets))
    cut_end = next(n for n in attribute.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'gdn_cut' for t in n.targets))
    cut_text = ''.join(lines[cut_start.lineno - 1:cut_end.end_lineno])
    text = _replace_once(text, cut_text, '')
    root_hook = '        for i,layer in enumerate(layers):\n'
    text = _replace_once(text, root_hook, cut_text +
        '        root_context=(nullcontext() if _root_tape is None else _root_tape.capture_scope(\n'
        '            layers,lambda index,layer:self._make_layer_captures(\n'
        "                layer,layer.block_type=='full_attention',observer,gdn_cut)))\n" + root_hook)
    text = _replace_once(text, "            with torch.no_grad():out=timed('native_root_with_CPU_checkpoints',",
                         "            with root_context,torch.no_grad():out=timed('native_root_with_CPU_checkpoints',")

    y_assign = next(n for n in reverse.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'y' for t in n.targets))
    old_replay = ''.join(lines[reverse.body[0].lineno - 1:y_assign.end_lineno])
    # The default statements/arguments stay exactly the original operations.
    # The enabled branch does not transport x merely to replay a removed call.
    new_replay = (
        "            layer=layers[i];is_fa=layer.block_type=='full_attention'\n"
        '            if _root_tape is None:\n'
        "                x=root[str(i)].to('cuda',non_blocking=self.pin_root_host);kw=_copy(kwargs[str(i)],'cuda',pinned_host=self.pin_root_host)\n"
        "                kw['past_key_values']=replay_cache\n"
        '                dc,mc,offload_mixer=self._make_layer_captures(layer,is_fa,observer,gdn_cut)\n'
        '                def replay():\n'
        '                    with torch.no_grad(),dc,mc:return layer(x,**kw)\n'
        "                owner_replay=getattr(model,'replay_finite_layer',None)\n"
        "                y=timed('native_replay_'+str(i),\n"
        '                        (lambda:owner_replay(layer,replay)) if callable(owner_replay) else replay)\n'
        '            else:\n'
        "                kw=_copy(kwargs[str(i)],'cuda',pinned_host=self.pin_root_host)\n"
        '                dc,mc,offload_mixer=_root_tape.pop(i)\n')
    text = _replace_once(text, old_replay, new_replay)
    row = next(n for n in reverse.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'row' for t in n.targets))
    deletion = next(n for n in reverse.body if isinstance(n, ast.Delete)
                    and any(isinstance(t, ast.Name) and t.id == 'expected' for t in n.targets))
    old_row = ''.join(lines[row.lineno - 1:deletion.end_lineno])
    row_branch = textwrap.indent(old_row, '    ')
    text = _replace_once(text, old_row,
        '            if _root_tape is None:\n' + row_branch +
        '            else:\n'
        "                row={'block_type':layer.block_type,'decoder_calls':dc.calls,'mixer_calls':mc.calls,\n"
        "                     'root_output_effect':effect(m,expected),'capture_source':'original_root',\n"
        "                     'native_decoder_replay_calls':0}\n"
        '                ledger[str(i)]=row;del expected\n')
    text = _replace_once(text, '        return signed,info\n',
                         "        if _root_tape is not None:info['root_capture_reuse']=_root_tape.report()\n"
                         '        return signed,info\n')
    ast.parse(text)
    compile(text, '<prepared-native-root-tape-owner>', 'exec')
    after = text.replace('\n', newline).encode('utf8')
    capture_ast = ast.dump(ast.Module(body=capture_nodes, type_ignores=[]), include_attributes=False)
    metadata = {
        'status': 'prepared_only_not_deployed', 'default_enabled': False,
        'owner_source_sha256': sha256(original), 'candidate_source_sha256': sha256(after),
        'capture_construction_ast_sha256': sha256(capture_ast.encode('utf8')),
        'candidate_runner_filename': RUNNER_NAME, 'helper_filename': HELPER_NAME,
        'new_flag': 'reuse_root_captures=False', 'all_original_layers': 32,
        'scope': 'original_root_capture_lifetime_and_decoder_replay_scheduling',
        'finite_math_changed': False, 'parameter_prepare_release_changed': False,
        'new_memory_policy': False, 'silent_fallback': False,
    }
    return after, metadata


def prepare(owner_source: Path, candidate_directory: Path, expected_owner_sha256: str):
    original = Path(owner_source).read_bytes()
    actual = sha256(original)
    if actual != expected_owner_sha256:
        raise ValueError(f'Owner SHA256 mismatch: expected {expected_owner_sha256}, actual {actual}')
    after, metadata = patched_owner(original)
    helper_path = Path(__file__).with_name(HELPER_NAME)
    helper = helper_path.read_bytes()
    destination = Path(candidate_directory)
    if destination.resolve() == Path(owner_source).parent.resolve():
        raise ValueError('Prepared candidate must remain outside the default owner directory.')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / RUNNER_NAME).write_bytes(after)
    (destination / HELPER_NAME).write_bytes(helper)
    metadata.update({'owner_source': str(Path(owner_source).resolve()),
                     'candidate_directory': str(destination.resolve()),
                     'helper_sha256': sha256(helper),
                     'preparer_sha256': sha256(Path(__file__).read_bytes())})
    diff = ''.join(difflib.unified_diff(original.decode('utf8').splitlines(True),
                                      after.decode('utf8').splitlines(True),
                                      fromfile='original_owner', tofile=RUNNER_NAME))
    (destination / 'owner.patch').write_text(diff, encoding='utf8', newline='')
    (destination / 'prepared.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-source', type=Path, required=True)
    parser.add_argument('--candidate-directory', type=Path, required=True)
    parser.add_argument('--expected-owner-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.owner_source, args.candidate_directory,
                             args.expected_owner_sha256), indent=2))


if __name__ == '__main__':
    main()
