"""Delegate actual saved operands to existing FA/FLA reference check owners.

Run only after the model/Ray diagnostic exits. B1 files are representation
slices of complete valid causal Q/K/V rows, not a smaller context or fixture.
No reference formula or tolerance is implemented here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--checker', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    out, rank = args.out, args.rank
    native_path = out / f'actual-native-fa3-rank{rank}.pt'
    finite_path = out / f'actual-finite-fa3-rank{rank}.pt'
    native = torch.load(native_path, map_location='cpu', weights_only=True)
    finite = torch.load(finite_path, map_location='cpu', weights_only=True)
    if finite.get('query_starts') is not None:
        raise ValueError('This thin checker packaging is for the actual scalar cached suffix capture')
    qstart = int(finite['query_start'])
    directory = out / ('official-checks-rank' + str(rank))
    directory.mkdir(exist_ok=False)
    metadata = dict(rank=rank, query_start=qstart, original_lengths=finite['lengths'],
        sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (native_path, finite_path, args.checker)},
        checks=[], scope=__doc__)
    for sample, length in enumerate(finite['lengths']):
        length = int(length)
        suffix = length - qstart
        pair = slice(2 * sample, 2 * sample + 2)
        attention = {}
        for name in ('dense_q', 'dense_k', 'dense_v'):
            value = native['tensors'][name]
            attention[name] = value[pair, :suffix if name == 'dense_q' else length].clone()
        operands = {}
        for name, value in finite['operands'].items():
            if name in ('q0', 'q1', 'u'):
                operands[name] = value[sample:sample+1, :, :suffix].clone()
            elif name in ('k0', 'k1', 'v0'):
                operands[name] = value[sample:sample+1, :, :length].clone()
            elif name in ('lse0', 'lse1'):
                operands[name] = value[sample:sample+1, :, :suffix].clone()
            else:
                raise ValueError('Unexpected finite FA tensor: ' + name)
        starts = finite['coefficient_starts']
        payload = dict(fa=dict(operands=operands, scale=finite['scale'], lengths=[length],
            padded_length=length, query_start=qstart,
            coefficient_starts=None if starts is None else [starts[sample]]),
            attention_values=attention)
        path = directory / f'actual-fa3-sample{sample}.pt'
        torch.save(payload, path)
        del payload, attention, operands
        output = directory / f'fa3-sample{sample}.json'
        command = [sys.executable, str(args.checker), '--operands', str(path),
                   '--sources', str(args.sources), '--output', str(output)]
        with (directory / f'fa3-sample{sample}.stdout.txt').open('xb') as stdout:
            result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT)
        item = dict(sample=sample, original_length=length, valid_suffix=suffix,
            operands=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
            original_checker_returncode=result.returncode, output=str(output))
        if output.exists():
            item['result'] = json.loads(output.read_bytes())
        metadata['checks'].append(item)
        (directory / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(dict(phase='original_FA_checker_complete', rank=rank, sample=sample,
                              returncode=result.returncode, output=str(output))), flush=True)
        if result.returncode or item.get('result', {}).get('status') != 'passed':
            raise SystemExit('Original FA assertion failed; preserve evidence, no correction')
    # The original GDN observer has already exported the exact original public
    # FLA input/output. Reuse its unchanged installed normalization + assertion.
    rank_report = json.loads((out / f'rank{rank}.json').read_bytes())
    observed = next(item for item in rank_report['reports'] if item['variant'] == 'observed')
    records = observed['actual_operands']['gdn0']['records']
    if len(records) != 1:
        raise ValueError('Expected one original GDN0 actual FLA capture')
    gdn_path = records[0]['path']
    command = [sys.executable, str(out / 'observe_native_gdn0_operands.py'),
        '--operands', gdn_path, '--official-source', str(args.sources / 'test_gated_delta_v041.py'),
        '--output', str(directory / 'gdn0.json')]
    with (directory / 'gdn0.stdout.txt').open('xb') as stdout:
        result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT)
    metadata['FLA'] = dict(returncode=result.returncode, operands=gdn_path,
                           output=str(directory / 'gdn0.json'))
    (directory / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(phase='original_FLA_checker_complete', rank=rank,
                          returncode=result.returncode)), flush=True)
    raise SystemExit(result.returncode)
