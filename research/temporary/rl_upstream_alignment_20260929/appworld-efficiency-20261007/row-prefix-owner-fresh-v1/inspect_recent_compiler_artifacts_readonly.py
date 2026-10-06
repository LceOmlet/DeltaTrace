"""One bounded system-find pass over the exact two worker-owned cache paths."""
import collections
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

SINCE = 1791327400
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
SOURCE = ROOT / 'runs/appworld-fresh-row-prefix-20261007-v1/appworld-dt/source.json'
SOURCE_SHA = 'c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
CACHE_KEYS = ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')
CONFIG_KEYS = ('dt_dynamic_shapes', 'dt_compiler_options', 'dt_compile_gdn_scalar_rules',
               'dt_pin_root_host', 'dt_native_conv_initial_states', 'dt_individual_prefixes',
               'dt_boundary_row_storage')


def config_fields(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix + '.' + key if prefix else key
            if key in CONFIG_KEYS:
                result[name] = item
            elif isinstance(item, (dict, list)):
                result.update(config_fields(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(config_fields(item, f'{prefix}[{index}]'))
    return result


def excerpt(item):
    path = Path(item['path'])
    raw = path.read_bytes()
    patterns = ('# AOT ID:', 'assert_size_stride(', '# Source Nodes:', '# Original ATen:',
                '# Topologically Sorted Source Nodes:', '# File:', 'async_compile.',
                '@triton_heuristics.', 'def call(', 'def benchmark_compiled_module(')
    selected = []
    for line_number, line in enumerate(raw.decode('utf-8', 'replace').splitlines(), 1):
        if any(pattern in line for pattern in patterns):
            selected.append({'line': line_number, 'text': line[:360]})
            if len(selected) == 28:
                break
    return dict(item, sha256=hashlib.sha256(raw).hexdigest(), selected_generated_lines=selected)


def main():
    observed = time.time()
    boot = next(float(s.split()[1]) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('btime '))
    hz = os.sysconf('SC_CLK_TCK')
    workers, roots = [], set()
    for pid in (2360541, 2367855, 2369144):
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        item = {'pid': pid, 'birth_unix': boot + int(fields[19]) / hz}
        if pid == 2360541:
            assert item['birth_unix'] == 1791325655.01
        else:
            raw = (proc / 'environ').read_bytes()
            env = dict(part.split(b'=', 1) for part in raw.split(b'\0') if b'=' in part)
            item['environ_sha256_only'] = hashlib.sha256(raw).hexdigest()
            item['cache_environment'] = {key: env[key.encode()].decode() for key in CACHE_KEYS if key.encode() in env}
            for value in item['cache_environment'].values():
                assert Path(value).resolve().is_relative_to(Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/cache').resolve())
                roots.add(value)
        workers.append(item)
    raw_source = SOURCE.read_bytes()
    assert hashlib.sha256(raw_source).hexdigest() == SOURCE_SHA
    source = json.loads(raw_source)
    env_path = source['resource_environment']['DT_ENVIRONMENT_JSON']
    raw_config = Path(env_path).read_bytes()
    execution = {'path': env_path, 'sha256': hashlib.sha256(raw_config).hexdigest(),
                 'whitelisted_execution_fields': config_fields(json.loads(raw_config))}
    scans = []
    for root in sorted(roots):
        command = ['timeout', '20s', 'find', root, '-xdev', '-maxdepth', '8', '-type', 'f',
                   '-newermt', '@' + str(SINCE), '!', '-newermt', '@' + str(observed),
                   '-print0']
        begin = time.monotonic()
        run = subprocess.run(command, capture_output=True, timeout=23)
        items, errors = [], []
        for part in run.stdout.split(b'\0'):
            if not part:
                continue
            try:
                path = part.decode('utf-8')
                stat = Path(path).stat()
                items.append({'path': path, 'mtime_unix': stat.st_mtime, 'bytes': stat.st_size,
                              'suffix': Path(path).suffix or '<no_suffix>'})
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                errors.append(type(exc).__name__)
        items.sort(key=lambda item: (item['mtime_unix'], item['path']))
        py = [item for item in items if item['suffix'] == '.py' and item['bytes'] <= 1024 * 1024]
        chosen = {item['path']: item for item in py[:2] + py[-2:] + sorted(py, key=lambda item: item['bytes'], reverse=True)[:2]}
        samples = []
        for item in chosen.values():
            try:
                samples.append(excerpt(item))
            except OSError as exc:
                errors.append({'path': item['path'], 'type': type(exc).__name__, 'errno': exc.errno})
        scans.append({'root': root, 'command': command, 'returncode': run.returncode,
                      'complete_within_bounds': run.returncode == 0, 'elapsed_seconds': time.monotonic() - begin,
                      'recent_file_count': len(items), 'recent_extensions': dict(collections.Counter(item['suffix'] for item in items)),
                      'recent_bytes': sum(item['bytes'] for item in items),
                      'recent_60s_bins_from_since': dict(sorted(collections.Counter(int((item['mtime_unix'] - SINCE) // 60) for item in items).items())),
                      'recent_files': items[:512], 'recent_paths_truncated': len(items) > 512,
                      'sample_generated_python': samples, 'errors': errors,
                      'stderr': run.stderr.decode('utf-8', 'replace')[:1500]})
    print(json.dumps({'observed_unix': observed, 'since_unix': SINCE, 'workers': workers,
                      'source': {'path': str(SOURCE), 'sha256': SOURCE_SHA}, 'execution_config': execution,
                      'scans': scans, 'torch_imported': 'torch' in sys.modules,
                      'collector_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'remote_writes': 0, 'model_gpu_rpc_profiler_operations': 0,
                      'limitations': ['mtime is not compiler duration', 'shared cache is not PID-attributed',
                                      'a new artifact is not necessarily a new graph or fresh kernel compilation',
                                      'returncode 124 is an incomplete bounded traversal, so missing matches cannot establish absence']}))


if __name__ == '__main__':
    main()
