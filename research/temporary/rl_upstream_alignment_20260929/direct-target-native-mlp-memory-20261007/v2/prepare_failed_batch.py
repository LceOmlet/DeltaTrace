"""Select the failed original B4 through the deployed readout owner, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    from verl import DataProto
    from agent_system.multi_turn_rollout.utils import to_list_of_dict
    from reward_readout import DirectActionTargetReadout
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_bytes())
    import reward_readout
    assert sha(reward_readout.__file__) == source['entry_sha256']['reward_readout.py']
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = []
    for rank in (0, 1):
        path = args.artifacts/f'rank{rank}-input-prepared.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        data = DataProto.from_dict(tensors=saved['batch'],
            non_tensors=saved['non_tensor_batch'], meta_info=saved['meta_info'])
        rows = to_list_of_dict(data)
        prepared = [DirectActionTargetReadout._prepare_row(row, i) for i, row in enumerate(rows)]
        requests = [item for item in prepared if item['reward'] != 0
                    and item['target_offsets'] and bool(item['prior'].any())]
        requests.sort(key=lambda item: item['selected'].numel())
        matched = []
        # Verify selection against every saved successful original batch, not
        # merely against the reconstructed ordering of the failed batch.
        for batch_number in range(1, 28):
            native_path = args.artifacts/f'rank{rank}-readout-native-batch-{batch_number}.pt'
            native = torch.load(native_path, map_location='cpu', weights_only=False)
            original = sorted(native['rows'], key=lambda row: row['batch_row'])
            selected = requests[(batch_number-1)*4:batch_number*4]
            assert len(original) == len(selected) == 4
            for actual, chosen in zip(original, selected):
                assert str(chosen['row']['traj_uid']) == actual['traj_uid']
                assert torch.equal(chosen['selected'], actual['selected'])
                assert chosen['target_offsets'] == actual['target_offsets']
            matched.append(dict(batch=batch_number, sha256=sha(native_path)))
        failed = requests[27*4:28*4]
        assert len(failed) == 4 and len(requests) == 112
        provenance = dict(source=str(args.source), source_sha256=sha(args.source),
            input_path=str(path), input_sha256=sha(path), rank=rank,
            owner_path=reward_readout.__file__, owner_sha256=sha(reward_readout.__file__),
            matched_original_successful_batches=matched,
            scope='Original CPU DataProto conversion and readout selection; zero model calls',
            failed_batch=28, selected_lengths=[item['selected'].numel() for item in failed],
            uids=[str(item['row']['traj_uid']) for item in failed])
        output = args.output/f'rank{rank}-failed-batch.pt'
        assert not output.exists()
        torch.save(dict(rows=[item['row'] for item in failed], provenance=provenance), output)
        receipts.append(dict(**provenance, output=str(output), output_sha256=sha(output)))
    assert not torch.cuda.is_initialized()
    (args.output/'selection.json').write_text(json.dumps(receipts, indent=2)+'\n')
    print(json.dumps(receipts))


if __name__ == '__main__':
    main()
