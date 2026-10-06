"""Prepare isolated row-prefix representations; no model/cache transition runs."""
from pathlib import Path
import ast
import difflib
import hashlib
import json


ROOT = Path(__file__).resolve().parent
SOURCES = {
    'qwen35_native_prefix_artifacts.py': '06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16',
    'native_prefix_leases.py': 'd5b539c7ffb8a431374cf79f4b995ef6f4138c3e6e289890409b6b79dbad393e',
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Original source seam not unique: {old!r}')
    return text.replace(old, new, 1)


def patch_artifact(text):
    text = replace_once(text,
        'def compose_native_prefix_cache(config, sources, prefix_length, *, device):',
        'def compose_native_prefix_cache(config, sources, prefix_length, *, device, prefix_lengths=None):')
    text = replace_once(text,
        '    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin\n\n    result = DynamicCache(config=config)',
        '    if prefix_lengths is not None:\n'
        '        return _compose_row_prefix_cache(config, sources, prefix_length, prefix_lengths, device=device)\n'
        '    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin\n\n    result = DynamicCache(config=config)')
    helper = '''@torch.no_grad()
def _compose_row_prefix_cache(config, sources, prefix_length, prefix_lengths, *, device):
    """Represent literal row prefixes in the original Cache's public carrier.

    FA rows are padded after their real prefix to a shared physical width.
    The runner owns the mask and original positions for those padded columns.
    GDN rows select their real captured boundary; no padding enters recurrence.
    """
    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin

    lengths = tuple(int(n) for n in prefix_lengths)
    if (len(lengths) != len(sources) or not lengths or max(lengths) != prefix_length
            or any(n < 64 or n % 64 for n in lengths)):
        raise ValueError('Row prefix lengths must match exact positive captured boundaries')
    result = DynamicCache(config=config)
    for i, layer in enumerate(result.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            states = [source.layers[i]['boundaries'][n] for (source, row), n in zip(sources, lengths)]
            conv = torch.cat([state[0][row:row+1] for state, (_, row) in zip(states, sources)])
            recurrent = torch.cat([state[1][row:row+1] for state, (_, row) in zip(states, sources)])
            result.update_conv_state(conv.to(device), i)
            result.update_recurrent_state(recurrent.to(device), i)
        else:
            keys = torch.cat([torch.nn.functional.pad(
                source.layers[i]['keys'][row:row+1, :, :n], (0, 0, 0, prefix_length-n))
                for (source, row), n in zip(sources, lengths)])
            values = torch.cat([torch.nn.functional.pad(
                source.layers[i]['values'][row:row+1, :, :n], (0, 0, 0, prefix_length-n))
                for (source, row), n in zip(sources, lengths)])
            result.update(keys.to(device), values.to(device), i)
    return result


'''
    text = replace_once(text, '@dataclass\nclass NativePrefixLease:', helper + '@dataclass\nclass NativePrefixLease:')
    text = replace_once(text,
        '    prefix_length: int\n\n    def __call__(self, factual_ids):\n',
        '    prefix_length: int\n    prefix_lengths: tuple | None = None\n    context_lengths: tuple | None = None\n\n    def __call__(self, factual_ids):\n'
        '        if self.prefix_lengths is not None:\n'
        '            if factual_ids.shape != (len(self.sources), self.prefix_length):\n'
        "                raise ValueError('Prepared row prefix lease and factual input shapes differ')\n"
        '            actual = factual_ids.detach().cpu()\n'
        '            if len(self.prefix_lengths) != len(self.sources):\n'
        "                raise ValueError('One original prefix length is required per factual row')\n"
        '            for i, ((source, row), n) in enumerate(zip(self.sources, self.prefix_lengths)):\n'
        '                if not torch.equal(actual[i, :n], source.input_ids[row, :n]):\n'
        "                    raise ValueError('Prepared row prefix lease does not match exact factual token IDs')\n"
        '            return compose_native_prefix_cache(self.sources[0][0].config, self.sources,\n'
        '                self.prefix_length, device=factual_ids.device, prefix_lengths=self.prefix_lengths)\n')
    return text


def patch_lease(text):
    text = replace_once(text,
        'def prepare_native_prefix_leases(runner, requests, *, minibatch_size, eos_token_id):',
        'def prepare_native_prefix_leases(runner, requests, *, minibatch_size, eos_token_id, individual_prefixes=False):')
    text = replace_once(text, '    needed = {}\n',
        '    row_prefix_lengths = None\n'
        '    if individual_prefixes:\n'
        '        # Preserve the original all-rank zero/nonzero branch and its call count.\n'
        '        # A zero common boundary retains the original no-cache consumer.\n'
        "        row_prefix_lengths = [tuple(r['start']//64*64 for r in batch) if length else None\n"
        '                              for batch, length in zip(batches, prefix_lengths)]\n'
        '        carrier_prefix_lengths = [max(lengths) if lengths is not None else 0 for lengths in row_prefix_lengths]\n'
        '    needed = {}\n')
    text = replace_once(text,
        "    for batch, length in zip(batches, prefix_lengths):\n        if length:\n            for request in batch:\n                needed.setdefault(request['traj_uid'], set()).add(length)\n",
        "    if individual_prefixes:\n        for batch, lengths in zip(batches, row_prefix_lengths):\n            if lengths is not None:\n                for request, length in zip(batch, lengths):\n                    needed.setdefault(request['traj_uid'], set()).add(length)\n    else:\n        for batch, length in zip(batches, prefix_lengths):\n            if length:\n                for request in batch:\n                    needed.setdefault(request['traj_uid'], set()).add(length)\n")
    text = replace_once(text,
        "    for batch, length in zip(batches, prefix_lengths):\n        if length:\n            for request in batch:\n                if not torch.equal(request['prompt'][:length], canonical[request['traj_uid']][:length]):\n                    raise ValueError('Original histories do not share the requested factual prefix')\n",
        "    if individual_prefixes:\n        for batch, lengths in zip(batches, row_prefix_lengths):\n            if lengths is not None:\n                for request, length in zip(batch, lengths):\n                    if not torch.equal(request['prompt'][:length], canonical[request['traj_uid']][:length]):\n                        raise ValueError('Original histories do not share the requested factual row prefix')\n    else:\n        for batch, length in zip(batches, prefix_lengths):\n            if length:\n                for request in batch:\n                    if not torch.equal(request['prompt'][:length], canonical[request['traj_uid']][:length]):\n                        raise ValueError('Original histories do not share the requested factual prefix')\n")
    text = replace_once(text,
        "    leases = [NativePrefixLease([sources[r['traj_uid']] for r in batch], length)\n              if length else None for batch, length in zip(batches, prefix_lengths)]\n",
        "    if individual_prefixes:\n        leases = [NativePrefixLease([sources[r['traj_uid']] for r in batch], length, prefix_lengths=lengths,\n                                    context_lengths=tuple(r['context_tokens'] for r in batch))\n                  if lengths is not None else None\n                  for batch, length, lengths in zip(batches, carrier_prefix_lengths, row_prefix_lengths)]\n    else:\n        leases = [NativePrefixLease([sources[r['traj_uid']] for r in batch], length)\n                  if length else None for batch, length in zip(batches, prefix_lengths)]\n")
    return text


def main():
    changes = []
    for name, expected in SOURCES.items():
        baseline = ROOT/'baseline'/name
        raw = baseline.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f'Frozen baseline differs: {name}')
        text = raw.decode('utf-8')
        new = (patch_artifact if name.startswith('qwen35') else patch_lease)(text)
        ast.parse(new)
        compile(new, name, 'exec')
        path = ROOT/'candidate'/name
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(new.encode('utf-8'))
        changes.append(dict(name=name, baseline_sha256=expected,
                            candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        patch = ''.join(difflib.unified_diff(text.splitlines(True), new.splitlines(True),
                         fromfile='baseline/'+name, tofile='candidate/'+name))
        (ROOT/(name+'.patch')).write_text(patch, encoding='utf-8', newline='\n')
    (ROOT/'source-manifest.json').write_text(json.dumps(dict(
        status='prepared_only', source_changes=changes,
        default_contract='Unspecified row lengths preserve original scalar cache/lease bodies.',
        scope='Cache representation only; no runner, targets, FA/FLA math, model, training, GPU or deployment.'),
        indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
