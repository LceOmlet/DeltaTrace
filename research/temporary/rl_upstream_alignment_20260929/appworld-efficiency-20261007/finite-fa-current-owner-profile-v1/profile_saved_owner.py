"""Profile the unchanged deployed finite-FA owner on saved actual B4 operands."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    assets = json.loads(args.assets.read_bytes())

    def check(item):
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], str(path)
        return path

    helper_path = check(assets['transport_helper'])
    wrapper_path = check(assets['deployed_wrapper'])
    library_path = check(assets['deployed_library'])
    item = next(row for row in assets['saved_operands'] if row['rank'] == args.rank)
    operand_path = check(item['finite_operands'])
    spec = importlib.util.spec_from_file_location('_existing_actual_row_transport', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    import torch
    wrapper = helper.module('_unchanged_row_FA_profile_owner', wrapper_path)
    owner = wrapper.VendorFAFiniteP1BF16D256(library_path, assets['deployed_library']['sha256'])
    saved = torch.load(operand_path, map_location='cpu', weights_only=True, mmap=True)
    assert saved['decoder_index'] == 3 and saved['invocation'] == 8
    assert set(saved['operands']) == set(helper.OPERANDS)
    ops = {name: value.contiguous().to('cuda') for name, value in saved['operands'].items()}
    layout = wrapper.RightPaddedLengths(list(saved['lengths']), ops['k0'].shape[2], 'cuda',
        coefficient_starts=list(saved['coefficient_starts']), query_starts=list(saved['query_starts']),
        query_padded_length=ops['q0'].shape[2])
    report = dict(rank=args.rank, assets=assets, harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='One existing real B4 finite-FA call at decoder3. Default deployed ABI; no candidate, model, checkpoint, training, or numerical change. Profiling timings are diagnostic, not uninstrumented speed or full-DT costs.',
        operands={name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in ops.items()},
        lengths=list(saved['lengths']), query_starts=list(saved['query_starts']),
        coefficient_starts=list(saved['coefficient_starts']))
    with torch.no_grad():
        for _ in range(2):
            value = owner(ops, saved['scale'], layout)
            torch.cuda.synchronize()
            del value
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                with_stack=True, profile_memory=True) as profile:
            with torch.profiler.record_function('deployed_default_finite_FA_saved_B4'):
                value = owner(ops, saved['scale'], layout)
                torch.cuda.synchronize()
        report['instrumented_call_and_profiler_wall_seconds'] = time.perf_counter() - started
        report['outputs_finite'] = {name: bool(torch.isfinite(tensor).all()) for name, tensor in value.items()}
        report['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        trace = args.output / 'trace.json'
        profile.export_chrome_trace(str(trace))
        report['trace'] = dict(path=str(trace), bytes=trace.stat().st_size,
            sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
        report['original_profiler_events'] = [dict(name=e.key, count=e.count,
            self_cpu_us=e.self_cpu_time_total, self_device_us=getattr(e, 'self_device_time_total', 0))
            for e in profile.key_averages()]
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rank=args.rank, output=str(args.output / 'result.json'), trace=report['trace'])), flush=True)


if __name__ == '__main__':
    main()
