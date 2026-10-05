"""Stream original stopped TextCraft stdout into complete-readout distributions.

This is a CPU-only log reader. It neither runs models nor reconstructs IDs.
Reports are associated with the following original completed-step metric line,
matching the original trainer's emission order. Missing reports are not zeros.
Negative roots are observations, not a numerical-error verdict.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
STEP = re.compile(r'\bstep:(\d+)\s+-\s+global_seqlen/')
PID = re.compile(r'\bpid=(\d+)\b')
METRIC = re.compile(r'(?:^| - )([A-Za-z0-9_./]+):([^ ]+)')
MARKER = '[DeltaTrace readout] '
EXPECTED_LOG_SHA = '54f4aef6d71bc864fca7f7e4e339e0ec55cb6b146e064f38370da6764185d933'
FIELDS = ('observed_return', 'root_effect', 'signed_sum', 'conservation_residual',
          'factual_target_logp', 'reference_target_logp',
          'source_log_ratio_min', 'source_log_ratio_max')
METRIC_FIELDS = ('critic/score/mean', 'episode/reward/mean', 'actor/entropy_loss',
    'actor/kl_loss', 'actor/ppo_kl', 'actor/grad_norm', 'actor/pg_clipfrac',
    'actor/pg_clipfrac_lower', 'response_length/clip_ratio',
    'episode/length/mean', 'timing_s/gen', 'timing_s/adv', 'timing_s/update_actor')


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_distribution(rows, field):
    values, missing, nonfinite, other = [], 0, 0, 0
    for row in rows:
        if field not in row or row[field] is None:
            missing += 1
        elif finite_number(row[field]):
            values.append(float(row[field]))
        elif isinstance(row[field], (int, float)) and not isinstance(row[field], bool):
            nonfinite += 1
        else:
            other += 1
    result = dict(finite_count=len(values), missing_count=missing,
                  nonfinite_count=nonfinite, nonnumeric_count=other)
    if not values:
        return result
    values.sort()
    def quantile(q):
        position = (len(values)-1)*q
        left = int(position)
        right = min(left+1, len(values)-1)
        return values[left]+(values[right]-values[left])*(position-left)
    result.update(mean=math.fsum(values)/len(values), min=values[0], max=values[-1],
        quantiles={str(q): quantile(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
    return result


def counts(rows, field):
    values = Counter()
    missing = 0
    for row in rows:
        if field not in row:
            missing += 1
        else:
            values[json.dumps(row[field], sort_keys=True)] += 1
    return dict(values=dict(values), missing_count=missing)


def summarize(rows):
    roots = [row['root_effect'] for row in rows if finite_number(row.get('root_effect'))]
    return dict(trace_count=len(rows),
        observed_return_counts=counts(rows, 'observed_return'),
        source_step_counts=counts(rows, 'source_step'),
        root_sign_counts=dict(negative=sum(value < 0 for value in roots),
            zero=sum(value == 0 for value in roots), positive=sum(value > 0 for value in roots)),
        distributions={field: scalar_distribution(rows, field) for field in FIELDS},
        trace_field_counts=dict(Counter(key for row in rows for key in row)),
        invalid_trace_values=0)


def compact_metadata(report):
    # Preserve actual scalar metadata; do not duplicate the enormous traces or
    # minimum-input arrays. Unknown fields are exposed by their original type.
    retained, types = {}, {}
    for key, value in report.items():
        if key in ('traces', 'minimum_log_ratio_batch', 'actual_row_lengths'):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            retained[key] = value if not isinstance(value, float) or math.isfinite(value) else repr(value)
        else:
            types[key] = type(value).__name__
    return dict(scalars=retained, non_scalar_field_types=types)


def summarize_reports(reports):
    rows, invalid, unreadable = [], 0, []
    for report in reports:
        value = report['data'].get('traces')
        if not isinstance(value, list):
            unreadable.append(dict(line=report['line'], actual_type=type(value).__name__))
            continue
        for row in value:
            if isinstance(row, dict):
                rows.append(row)
            else:
                invalid += 1
    result = summarize(rows)
    result['invalid_trace_values'] = invalid
    result['unreadable_trace_fields'] = unreadable
    result['report_count'] = len(reports)
    result['report_sources'] = [dict(line=r['line'], pid=r['pid'],
        original_line_sha256=r['original_line_sha256'],
        trace_payload_sha256=r['trace_payload_sha256'],
        original_metadata=compact_metadata(r['data'])) for r in reports]
    subset = [row for row in rows if row.get('observed_return') == 1 and row.get('source_step') == 0]
    result['return1_source_step0'] = summarize(subset)
    return result


def collect(log, rank_pids):
    pending, steps, errors = [], [], []
    sha = hashlib.sha256()
    decoder = json.JSONDecoder()
    total_lines = total_bytes = report_count = 0
    metric_steps_seen = set()
    with log.open('rb') as stream:
        for number, original in enumerate(stream, 1):
            sha.update(original)
            total_lines = number
            total_bytes += len(original)
            # Keep one original line/payload at a time, not the whole stdout.
            text = ANSI.sub('', original.decode('utf-8', errors='replace')).strip()
            if MARKER in text:
                prefix, raw = text.split(MARKER, 1)
                match = PID.search(prefix)
                try:
                    data, end = decoder.raw_decode(raw)
                    if not isinstance(data, dict):
                        raise ValueError('Original readout JSON is not a dictionary')
                    pending.append(dict(line=number, pid=int(match.group(1)) if match else None,
                        data=data, original_line_sha256=hashlib.sha256(original).hexdigest(),
                        trace_payload_sha256=hashlib.sha256(raw[:end].encode('utf-8')).hexdigest()))
                    report_count += 1
                except (ValueError, TypeError) as exc:
                    errors.append(dict(line=number, error=str(exc),
                        original_line_sha256=hashlib.sha256(original).hexdigest()))
            match = STEP.search(text)
            if match:
                step = int(match.group(1))
                parsed = dict(METRIC.findall(text))
                grouped = {}
                for report in pending:
                    grouped.setdefault(report['pid'], []).append(report)
                summaries = []
                for pid, reports in grouped.items():
                    summaries.append(dict(pid=pid, rank=rank_pids.get(pid), **summarize_reports(reports)))
                steps.append(dict(step=step, metric_line=number,
                    original_metric_sha256=hashlib.sha256(original).hexdigest(),
                    duplicate_completed_step=step in metric_steps_seen,
                    following_metric_values={key: parsed[key] for key in METRIC_FIELDS if key in parsed},
                    report_count=len(pending), rank_reports=summaries,
                    ranks_without_observed_report=[rank for pid, rank in rank_pids.items() if pid not in grouped]))
                metric_steps_seen.add(step)
                pending.clear()
    return dict(collected_unix=time.time(), source_log=dict(path=str(log.resolve()),
        sha256=sha.hexdigest(), bytes=total_bytes, lines=total_lines,
        expected_sha256=EXPECTED_LOG_SHA, matches_recorded_stopped_log=sha.hexdigest()==EXPECTED_LOG_SHA),
        original_rank_pid_mapping=[dict(pid=pid, rank=rank) for pid, rank in rank_pids.items()],
        rank_mapping_scope='Original worker PIDs bound to ranks from the stopped job; unknown PIDs remain null-rank.',
        parser_source=dict(path=str(Path(__file__).resolve()),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        emitted_readout_reports=report_count, completed_metric_records=len(steps),
        steps=steps, parse_errors=errors,
        unassigned_reports=summarize_reports(pending) if pending else None,
        scope=__doc__, quantile_method='Sorted finite values, linear interpolation at (n-1)*q.',
        interpretation='All original readout traces, not minimum-selected examples. Zero-return requests are normally skipped '
            'by the original owner, so this is not an all-rollout success distribution. Missing/nonfinite values '
            'are counted explicitly. A negative root alone does not prove a numerical or algorithmic error.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank-pids', type=int, nargs=2, default=[3236008, 3240133],
                        metavar=('RANK0_PID', 'RANK1_PID'))
    args = parser.parse_args()
    result = collect(args.log, {pid: rank for rank, pid in enumerate(args.rank_pids)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), source_log=result['source_log'],
        reports=result['emitted_readout_reports'], completed_steps=result['completed_metric_records'],
        parse_errors=len(result['parse_errors']))))
