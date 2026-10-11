"""Compare saved CPU launch records; null pointers are not allocation addresses."""
from pathlib import Path
import hashlib
import json
import time

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'direct-credit-records-20261009-v1'

def normalized(row, anchor):
    def relative(pointer):
        return pointer - anchor if pointer not in (None, 0) else pointer
    return row[1:5] + [relative(row[5]), row[6], relative(row[7]), row[8]]

def main():
    report = []
    for rank in (0, 1):
        paths = [RAW / f'native-two-group-lifecycle-artifacts-20261011-v1/result/rank{rank}-two-group-lifecycle.json',
                 RAW / f'native-parameter-input-pointers-downloaded-20261011-v1/result/rank{rank}-two-group-lifecycle.json']
        a, b = [json.loads(p.read_bytes())['records'] for p in paths]
        anchors = [next(row[7] for row in records if row[7]) for records in (a, b)]
        count = min(len(a), len(b))
        report.append(dict(rank=rank,
            sources=[dict(path=p.as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
            failed_records=len(a), normal_records=len(b), compared_prefix=count,
            semantic_difference_indices=[i for i in range(count) if a[i][1:5] != b[i][1:5]],
            relative_address_geometry_difference_indices=[i for i in range(count)
                if normalized(a[i], anchors[0]) != normalized(b[i], anchors[1])]))
    out = dict(unix=time.time(), ranks=report, production_changes=0,
        scope='Relative addresses and CPU launch order do not establish asynchronous GPU completion or the faulty writer.',
        null_pointer_handling='Preserve 0/None; do not subtract live allocation anchors from null pointers.')
    path = RAW / 'native-lifecycle-failed-normal-compare-20261011-v1.json'
    path.write_bytes((json.dumps(out, indent=2) + '\n').encode())
    print(json.dumps(out))

if __name__ == '__main__':
    main()
