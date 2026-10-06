"""Read original stopped SQL logs; save their retained literal B4s only.

No model, tokenizer, task, Ray actor, tensor or training call is made.  The
retained per-rank minima are from the same 588-call report, but their original
owner batch indices differ.  They are not the original synchronized last B4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ENTRY, ROOT, SCP, SSH


REMOTE_BODY = r'''
from pathlib import Path
import hashlib,json,os,resource,time
root=Path(@ROOT@)
out=root/'receipts/sql-paired-real-b4-20261006-v1'
assert not out.exists(), 'Immutable extraction already exists; inspect it'
source=root/'runs/sql-padding-restart-20261001/sql-dt/source.json'
source_bytes=source.read_bytes()
source_sha=hashlib.sha256(source_bytes).hexdigest()
assert source_sha=='fe54f11730b1d9d06da6364edc7a09a6524fb7f8f02510c59e8e7ca4c99b04ff'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('MACA_VISIBLE_DEVICES')==''
session=Path('/tmp/ray/session_2026-10-01_05-44-42_236554_552842/logs')
expected_logs={
 562595:'31630f5ba4b794eba442d2cb978684edba61cf3d789e68528544db86ea5e7ce7',
 564368:'7f77661c933868a363d508b8e07733353e78fcaa0ed1e84c637a35e5da788a18',
}
ranks=[]
for rank,pid in enumerate((562595,564368)):
 logs=list(session.glob(f'worker-*-{pid}.out'))
 assert len(logs)==1
 path=logs[0];hasher=hashlib.sha256();offset=0;batch_lines=[];found=[]
 with path.open('rb') as handle:
  for line_number,raw in enumerate(handle,1):
   start=offset;offset+=len(raw);hasher.update(raw)
   if b'[DT EOS minibatch]' in raw:
    batch_lines.append(dict(line=line_number,byte_start=start,byte_end=offset,
     sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8').strip()))
    batch_lines=batch_lines[-2:]
   marker=b'[DeltaTrace readout]'
   if marker not in raw:continue
   report=json.loads(raw.split(marker,1)[1].strip())
   if report['finite_trace_calls']!=588:continue
   minimum=report['minimum_log_ratio_batch'];samples=[]
   for slot,sample in enumerate(minimum['samples']):
    selected=sample['selected_input_ids'];trace=sample['trace']
    matches=[index for index,value in enumerate(report['traces']) if value==trace]
    assert len(matches)==1
    trace_index=matches[0]
    assert trace_index//4==trace['owner_batch_index']
    assert trace_index%4==slot
    begin=sample['source_start'];end=sample['source_end'];query_end=end+trace['query_tokens']
    assert 0<=begin<end<query_end<len(selected)
    assert len(selected)==trace['context_tokens']
    assert len(selected[query_end:])==1 and selected[-1] in minimum['outcome_token_ids']
    # These are slices of original literal IDs at original recorded boundaries.
    # No text rendering, re-encoding, inferred query or reference is introduced.
    samples.append(dict(sample,original_trace_index=trace_index,original_slot=slot,
     selected_input_ids_sha256=hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest(),
     prompt_input_ids=selected[:begin],response_input_ids=selected[begin:end],
     query_input_ids=selected[end:query_end],target_input_ids=selected[query_end:],
     segment_source='Original source_start/source_end/query_tokens boundaries; literal list slicing'))
   owner_indices={sample['trace']['owner_batch_index'] for sample in samples}
   assert len(samples)==4 and len(owner_indices)==1
   found.append(dict(report_line=line_number,report_byte_start=start,report_byte_end=offset,
    report_line_sha256=hashlib.sha256(raw).hexdigest(),report_index=None,
    original_report=dict(task=report['task'],finite_trace_calls=report['finite_trace_calls'],
     event_contrasts=report['event_contrasts'],minibatch_size=report['minibatch_size'],
     seconds=report['seconds'],max_readout_length=report['max_readout_length'],
     original_trace_count=len(report['traces'])),
    original_last_two_batch_lines=batch_lines.copy(),
    original_owner_batch_index=next(iter(owner_indices)),
    minimum_log_ratio_batch=dict(minimum,samples=samples)))
 assert len(found)==1
 assert hasher.hexdigest()==expected_logs[pid], 'Original stopped log changed'
 ranks.append(dict(rank=rank,worker_pid=pid,original_log=dict(path=str(path),
  sha256=hasher.hexdigest(),bytes=offset,line_count=line_number),**found[0]))
assert ranks[0]['original_report']['event_contrasts']==ranks[1]['original_report']['event_contrasts']==2352
assert [x['original_owner_batch_index'] for x in ranks]==[366,256]
result=dict(observed_unix=time.time(),scope='Eight exact retained literal rows from the same original 588-call SQL RPC. Per-rank retained minimum B4s are different original batch indices, not an original synchronized B8.',
 source=dict(path=str(source),sha256=source_sha),driver_pid=552842,
 exact_requested_last_batch_available=False,
 missing='The 587th/588th B4 log lines retain lengths/time/d extrema, but no literal IDs. The only full row IDs in each structured report are its minimum_log_ratio_batch.',
 pairing=dict(report_finite_trace_calls=588,report_event_contrasts_per_rank=2352,
  original_synchronized_batch=False,original_owner_batch_indices=[366,256],
  requested_one_based_batches=[587,588],selection='The unique report immediately after original batch=588/588; retain its original per-rank minima without relabeling them.'),
 ranks=ranks,operations=dict(model=0,tokenizer=0,ray=0,rollout=0,DT=0,backward=0,optimizer=0),
 cpu=dict(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],maca_visible_devices=os.environ['MACA_VISIBLE_DEVICES'],
  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
 extractor_source=dict(local_path=@SCRIPT_PATH@,sha256=@SCRIPT_SHA@),
 limitations=['These eight literal rows are suitable as an explicitly saved-payload diagnostic, not a replay of the known 80-second synchronized B4.',
  'The original model at this report was not checkpointed; saved source/inputs do not preserve its exact current weights.',
  'Original trace conservation_tolerance metadata is preserved as historical data, not adopted as official numerical acceptance.'])
out.mkdir(parents=True)
target=out/'original-retained-sql-b4-inputs.json'
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
 samples=sum(len(x['minimum_log_ratio_batch']['samples']) for x in ranks),
 original_owner_batch_indices=result['pairing']['original_owner_batch_indices'],exact_requested_last_batch_available=False,
 selected_lengths=[[len(s['selected_input_ids']) for s in r['minimum_log_ratio_batch']['samples']] for r in ranks],
 report_lines=[r['report_line'] for r in ranks],max_rss_kib=result['cpu']['max_rss_kib'])))
'''


def main():
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    body = (REMOTE_BODY.replace('@ROOT@', repr(ROOT))
            .replace('@SCRIPT_PATH@', repr(str(Path(__file__).resolve())))
            .replace('@SCRIPT_SHA@', repr(script_sha)))
    compile(body, 'extract-original-sql-logs', 'exec')
    script = (f'source {ENTRY}/metax-entry.env.sh\n'
              'export CUDA_VISIBLE_DEVICES=""\nexport MACA_VISIBLE_DEVICES=""\n'
              'export OMP_NUM_THREADS=1\nexport MKL_NUM_THREADS=1\n'
              '"$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n')
    (HERE / 'capture.sh').write_text(script, encoding='utf-8', newline='\n')
    completed = subprocess.run(SSH + ['bash', '-s'], input=script.encode(),
                               capture_output=True, timeout=60)
    (HERE / 'capture.stdout.txt').write_bytes(completed.stdout)
    (HERE / 'capture.stderr.txt').write_bytes(completed.stderr)
    completed.check_returncode()
    records = [json.loads(line) for line in completed.stdout.decode().splitlines() if line.startswith('{')]
    record = records[-1]
    subprocess.run(SCP + [f'{SSH[-1]}:{record["path"]}',
                         str(HERE / 'original-retained-sql-b4-inputs.json')], check=True, timeout=45)
    actual = hashlib.sha256((HERE / 'original-retained-sql-b4-inputs.json').read_bytes()).hexdigest()
    assert actual == record['sha256']
    print(json.dumps(dict(record, local_path=str(HERE / 'original-retained-sql-b4-inputs.json'),
                         extractor_sha256=script_sha), ensure_ascii=False))


if __name__ == '__main__':
    main()
