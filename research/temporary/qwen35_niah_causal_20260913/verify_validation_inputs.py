"""Decode actual validation tokens and audit replacements and gold spans on CPU."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.run / 'input_cases.json').read_bytes())
    protocol = json.loads((args.run / 'protocol.json').read_bytes())
    assert sha(args.run / 'input_cases.json') == protocol['validation_cases_sha256']
    source = {}
    for entry in data['sources']:
        file = args.source / (entry['dataset'] + '.jsonl')
        assert sha(file) == entry['sha256']
        source[entry['dataset']] = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
    tokenizer = json.loads(args.tokenizer.read_bytes())
    vocabulary = {v: k for k, v in tokenizer['model']['vocab'].items()}
    added = {r['id']: r['content'] for r in tokenizer['added_tokens']}
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = byte_values[:]
    n = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(256 + n)
            n += 1
    decoder = {chr(c): b for b, c in zip(byte_values, characters)}

    def token_bytes(index):
        if index in added:
            return added[index].encode('utf-8')
        return bytes(decoder[c] for c in vocabulary[index])

    count = 0
    for item in data['cases']:
        task, index = item['dataset'], item['index']
        original = source[task][item['source_index']]
        assert hashlib.sha256(original['prompt'].encode()).hexdigest() == item['source_prompt_sha256']
        assert hashlib.sha256(original['target'].encode()).hexdigest() == item['source_target_sha256']
        replacements = item['replacements']
        pattern = re.compile('|'.join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)))
        rewrite = lambda text: pattern.sub(lambda match: replacements[match.group()], text)
        assert rewrite(original['prompt']) == item['prompt']
        assert rewrite(original['target']) == item['target']
        case = json.loads((args.run / f'{task}_{index}' / 'results.json').read_bytes())
        ids = np.asarray(case['input_ids'], dtype=np.int64)
        assert hashlib.sha256(ids.tobytes()).hexdigest() == case['input_sha256']
        parts = [token_bytes(int(ids[j])) for j in case['user_positions']]
        text = ' ' + item['prompt']
        assert b''.join(parts) == text.encode('utf-8')
        assert b''.join(token_bytes(int(i)) for i in ids[case['prompt_length']:-1]) == item['target'].encode('utf-8')
        assert added[int(ids[-1])] == '<|im_end|>'
        ends = np.cumsum([len(part) for part in parts])
        starts = np.r_[0, ends[:-1]]
        gold = set()
        for old, new in zip(original['metadata']['needle_spans'], item['metadata']['needle_spans']):
            left, right = old['span']
            assert new['span'] == [len(rewrite(original['prompt'][:left])), len(rewrite(original['prompt'][:right]))]
            snippet = item['prompt'][new['span'][0]:new['span'][1]]
            assert new['key'] in snippet and new['answer'] in snippet
            left, right = (len(text[:n + 1].encode('utf-8')) for n in new['span'])
            gold.update(i for i in range(len(parts)) if starts[i] < right and ends[i] > left)
        assert sorted(gold) == case['gold']
        expected_keep = [i for i, part in enumerate(parts) if part.decode('utf-8', errors='replace').strip() not in ('', ',', '.')]
        assert expected_keep == case['keep']
        reference = ids.copy()
        reference[np.asarray(case['user_positions'])[case['keep']]] = ids[-1]
        assert hashlib.sha256(reference.tobytes()).hexdigest() == case['reference_sha256']
        count += 1
    assert count == protocol['expected_cases']
    result = dict(status='passed', cases=count, tokenizer_sha256=sha(args.tokenizer),
                  checks=['source identities', 'consistent key/value replacements', 'actual prompt and complete target token bytes',
                          'terminal EOS', 'gold span offsets', 'eligible tokens', 'reference and input hashes'])
    (args.run / 'input_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
