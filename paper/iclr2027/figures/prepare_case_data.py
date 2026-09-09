"""Freeze two fixed-index DT examples after checking actual model-input bytes.

No inference, reranking, or attribution recomputation is performed.
The tokenizer JSON files are used only to decode recorded token IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-root', type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_root
    numeric_path = REPO / 'research/temporary/development16_20260909/numeric.json'
    summary = json.loads(numeric_path.with_name('summary.json').read_bytes())
    numeric = json.loads(numeric_path.read_bytes())
    assert sha(numeric_path.read_bytes()) == summary['numeric_sha256']
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values[:]
    n = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + n)
            n += 1
    byte_decoder = {chr(c): b for b, c in zip(byte_values, codepoints)}
    result = {
        'selection': 'First released example (index 0) of each of two tasks, for both models.',
        'method': 'clean-v1-20260909; stored DT attribution, unchanged',
        'numeric_sha256': sha(numeric_path.read_bytes()),
        'score_target': 'Entire fixed released response plus EOS',
        'reference': 'Eligible input positions replaced by tokenizer EOS',
        'source_scope': numeric['scope'], 'tokenizer_sources': {}, 'cases': []}
    for family in ('qwen3', 'qwen35'):
        folder = audit / 'deltatrace-figures-20260909'
        tokenizer = json.loads((folder / f'{family}-tokenizer.json').read_bytes())
        source = json.loads((folder / f'{family}-source.json').read_bytes())
        assert sha((folder / f'{family}-tokenizer.json').read_bytes()) == source['sha256']
        result['tokenizer_sources'][family] = source
        vocab = {v: k for k, v in tokenizer['model']['vocab'].items()}
        special = {t['id']: t['content'] for t in tokenizer['added_tokens']}

        def token_bytes(token):
            return (special[token].encode('utf8') if token in special else
                    bytes(byte_decoder[c] for c in vocab[token]))

        for dataset in ('niah_mq_q2', 'morehopqa'):
            run = ('codex_clean_development16_20260909_mh_recovery_v1'
                   if family == 'qwen35' and dataset == 'morehopqa'
                   else 'codex_clean_development16_20260909_v1')
            report_path = audit / 'snapshot/tmp' / run / family / 'results.json'
            receipt = next(s for s in summary['sources'] if s['id'] == run + '/' + family)
            assert sha(report_path.read_bytes()) == receipt['raw_report_sha256']
            report = json.loads(report_path.read_bytes())
            raw = next(c for c in report['cases'] if c['dataset'] == dataset and c['index'] == 0)
            row = next(c for c in numeric['models'][family] if c['dataset'] == dataset and c['index'] == 0)
            ids = np.asarray(raw['input_ids'], dtype='<i8')
            assert sha(ids.tobytes()) == row['input_sha256'] == raw['input_sha256']
            data_path = audit / 'published_flashtrace/table1-data-v1/extracted/data' / (dataset + '.jsonl')
            cached = json.loads(data_path.read_text(encoding='utf8').splitlines()[0])
            protocol = json.loads((REPO / 'experiments/official/protocol.json').read_bytes())
            assert sha(data_path.read_bytes()) == protocol['tasks'][dataset]['cache_sha256']
            positions = row['user_positions']
            user_parts = [token_bytes(int(ids[j])) for j in positions]
            user_bytes = b''.join(user_parts)
            user_text = user_bytes.decode('utf8')
            assert user_text.strip() == cached['prompt'].strip()
            target = b''.join(token_bytes(int(t)) for t in ids[row['prompt_length']:-1]).decode('utf8')
            assert target == cached['target']
            signed = np.asarray(row['signed_full'])
            assert np.array_equal(np.maximum(signed[positions].astype(np.float32), 0), row['positive_prompt'])
            assert len(signed) == len(ids) and np.isfinite(signed).all()
            # Character byte boundaries also handle BPE splits inside UTF-8 characters.
            char_bytes = [0]
            for ch in user_text:
                char_bytes.append(char_bytes[-1] + len(ch.encode('utf8')))
            tokens, cursor = [], 0
            for j, (position, part) in enumerate(zip(positions, user_parts)):
                end = cursor + len(part)
                lo = int(np.searchsorted(char_bytes, cursor, side='right') - 1)
                hi = int(np.searchsorted(char_bytes, end, side='left'))
                tokens.append({'id': int(ids[position]), 'input_position': position,
                               'local_index': j, 'byte_span': [cursor, end], 'char_span': [lo, hi],
                               'score': float(signed[position]), 'eligible': j in row['keep'],
                               'gold': j in row['gold']})
                cursor = end
            assert cursor == len(user_bytes)
            curve = row['metrics']['DT']
            assert curve['actual_input_hashes'][0] == row['input_sha256']
            result['cases'].append({
                'model': family, 'dataset': dataset, 'index': 0,
                'run': run, 'raw_report_sha256': receipt['raw_report_sha256'],
                'input_sha256': row['input_sha256'], 'input_ids': ids.tolist(),
                'prompt_length': row['prompt_length'], 'target_length': row['target_length'],
                'user_text': user_text, 'target': target,
                'answer_excerpt': cached['metadata']['boxed_answer'],
                'tokens': tokens,
                'deletion': {k: curve[k] for k in ('normalized_model_response', 'deleted_user_indices', 'actual_input_hashes')},
                'signed_sum': float(signed.sum())})
    destination = HERE / 'data/cases.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'cases': len(result['cases']), 'actual_input_hashes_verified': True,
                      'fixed_response_text_verified': True, 'output': str(destination)}))


if __name__ == '__main__':
    main()
