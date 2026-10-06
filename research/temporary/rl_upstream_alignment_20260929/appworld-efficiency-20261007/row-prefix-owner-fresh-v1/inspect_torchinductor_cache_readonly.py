"""Bounded stdlib-only read of this job's existing persistent compiler caches.

Run through the existing SSH transport on stdin. This writes nothing remotely.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


SINCE = 1791327400
EXPECTED_DRIVER_BIRTH = 1791325655.01
EXPECTED_SOURCE_SHA = 'c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
SOURCE = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/runs/appworld-fresh-row-prefix-20261007-v1/appworld-dt/source.json')
CACHE_KEYS = ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')
MAX_ENTRIES = 60000
MAX_DEPTH = 8
MAX_SCAN_SECONDS = 10
MAX_SOURCE_BYTES = 1024 * 1024
MAX_RECORDED_RECENT_FILES = 512


def identity(pid):
    proc = Path('/proc') / str(pid)
    boot = next(float(s.split()[1]) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('btime '))
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'birth_unix': boot + int(fields[19]) / os.sysconf('SC_CLK_TCK'), 'state': fields[0]}


def source_excerpt(path):
    raw = path.read_bytes()
    lines = raw.decode('utf-8', 'replace').splitlines()
    selected = []
    patterns = ('# AOT ID:', '# kernel path:', 'assert_size_stride(', 'reinterpret_tensor(',
                '# Source Nodes:', '# Original ATen:', '# Topologically Sorted Source Nodes:',
                '# File:', 'async_compile.', '@triton_heuristics.', 'def call(',
                'def benchmark_compiled_module(', "device_str=", 'torch.empty_strided(')
    for i, line in enumerate(lines, 1):
        if any(pattern in line for pattern in patterns):
            selected.append({'line': i, 'text': line[:360]})
            if len(selected) == 28:
                break
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'selected_generated_code_lines': selected}


def scan(root, observed):
    started = time.monotonic()
    entries = 0
    file_count = 0
    extensions = collections.Counter()
    recent_extensions = collections.Counter()
    minute_bins = collections.Counter()
    recent = []
    errors = []
    truncation = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if entries >= MAX_ENTRIES or time.monotonic() - started >= MAX_SCAN_SECONDS:
            truncation.append('entry_or_elapsed_scan_bound')
            break
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries >= MAX_ENTRIES or time.monotonic() - started >= MAX_SCAN_SECONDS:
                        truncation.append('entry_or_elapsed_scan_bound')
                        break
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth < MAX_DEPTH:
                            stack.append((Path(entry.path), depth + 1))
                        else:
                            truncation.append('depth_bound')
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    file_count += 1
                    suffix = Path(entry.name).suffix or '<no_suffix>'
                    extensions[suffix] += 1
                    if SINCE <= stat.st_mtime <= observed:
                        item = {'path': entry.path, 'bytes': stat.st_size, 'mtime_unix': stat.st_mtime,
                                'ctime_unix': stat.st_ctime, 'suffix': suffix}
                        if len(recent) < MAX_RECORDED_RECENT_FILES:
                            recent.append(item)
                        recent_extensions[suffix] += 1
                        minute_bins[str(int((stat.st_mtime - SINCE) // 60))] += 1
        except OSError as exc:
            errors.append({'path': str(directory), 'error': type(exc).__name__, 'errno': exc.errno})
    recent.sort(key=lambda item: (item['mtime_unix'], item['path']))
    python_files = [item for item in recent if item['suffix'] == '.py' and item['bytes'] <= MAX_SOURCE_BYTES]
    chosen = python_files[:2] + python_files[-2:] + sorted(python_files, key=lambda item: item['bytes'], reverse=True)[:2]
    excerpts = []
    seen = set()
    for item in chosen:
        if item['path'] in seen:
            continue
        seen.add(item['path'])
        try:
            detail = source_excerpt(Path(item['path']))
            detail.update({key: item[key] for key in ('mtime_unix', 'ctime_unix')})
            excerpts.append(detail)
        except OSError as exc:
            errors.append({'path': item['path'], 'error': type(exc).__name__, 'errno': exc.errno})
    return {'root': str(root), 'exists': root.exists(), 'entries_visited': entries,
            'regular_files': file_count, 'file_extensions': dict(extensions),
            'recent_file_count': sum(recent_extensions.values()), 'recent_extensions': dict(recent_extensions),
            'recorded_recent_file_count': len(recent),
            'recent_paths_truncated': sum(recent_extensions.values()) > len(recent),
            'recorded_recent_bytes': sum(item['bytes'] for item in recent),
            'recent_60s_bins_from_since': dict(sorted(minute_bins.items(), key=lambda item: int(item[0]))),
            'recent_files': recent, 'sample_generated_python': excerpts,
            'complete_within_bounds': not truncation, 'truncation_reasons': sorted(set(truncation)),
            'errors': errors, 'scan_elapsed_seconds': time.monotonic() - started}


def main():
    observed = time.time()
    driver = identity(2360541)
    assert driver['birth_unix'] == EXPECTED_DRIVER_BIRTH, driver
    raw_source = SOURCE.read_bytes()
    assert hashlib.sha256(raw_source).hexdigest() == EXPECTED_SOURCE_SHA
    config_source = json.loads(raw_source)
    env_path = config_source.get('resource_environment', {}).get('DT_ENVIRONMENT_JSON')
    execution_config = {'path': env_path}
    if env_path:
        raw_config = Path(env_path).read_bytes()
        execution_config['sha256'] = hashlib.sha256(raw_config).hexdigest()
        config = json.loads(raw_config)
        config_keys = ('dt_dynamic_shapes', 'dt_compiler_options', 'dt_compile_gdn_scalar_rules', 'dt_dtype', 'dt_pin_root_host', 'dt_native_conv_initial_states', 'dt_individual_prefixes', 'dt_boundary_row_storage')
        execution_config['whitelisted_execution_fields'] = {key: config[key] for key in config_keys if key in config}
    workers = []
    roots = {}
    for pid in (2367855, 2369144):
        item = identity(pid)
        raw = (Path('/proc') / str(pid) / 'environ').read_bytes()
        env = dict(part.split(b'=', 1) for part in raw.split(b'\0') if b'=' in part)
        item['cache_environment'] = {key: env[key.encode()].decode() for key in CACHE_KEYS if key.encode() in env}
        item['environ_sha256_only'] = hashlib.sha256(raw).hexdigest()
        for key, value in item['cache_environment'].items():
            path = Path(value)
            allowed = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache')
            assert path.resolve().is_relative_to(allowed.resolve()), value
            roots.setdefault(str(path), {'root': path, 'keys': [], 'workers': []})
            roots[str(path)]['keys'].append(key)
            roots[str(path)]['workers'].append(pid)
        workers.append(item)
    scans = []
    for value in roots.values():
        result = scan(value['root'], observed)
        result['environment_keys'] = sorted(set(value['keys']))
        result['worker_pids'] = sorted(set(value['workers']))
        scans.append(result)
    print(json.dumps({'observed_unix': observed, 'since_unix': SINCE, 'driver': driver, 'workers': workers,
                      'source': {'path': str(SOURCE), 'sha256': EXPECTED_SOURCE_SHA, 'bytes': len(raw_source)},
                      'execution_config': execution_config,
                      'bounds': {'max_entries_per_root': MAX_ENTRIES, 'max_depth': MAX_DEPTH,
                                 'max_scan_seconds_per_root': MAX_SCAN_SECONDS, 'max_source_bytes': MAX_SOURCE_BYTES},
                      'scans': scans, 'collector_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'torch_imported': 'torch' in sys.modules, 'remote_writes': 0,
                      'model_gpu_rpc_profiler_operations': 0,
                      'limitations': ['mtime/ctime are artifact timestamps, not compiler-duration measurements',
                                      'shared cache filenames do not establish producing PID or one file per compiled graph',
                                      'new cache writes may include autotune/metadata and loaded/restored artifacts',
                                      'no production parameters, cache files, or live methods modified']}))


if __name__ == '__main__':
    main()
