"""Derive BF16/D256/right-padding support from the preserved finite FA extension.

No vendor header changes, numerical rule changes, new attention implementation,
tile search, or model/package patch. This script prepares one compile only.
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

A = Path(__file__).resolve().parent
R = A.parent / 'DeltaTrace'
sha = lambda b: hashlib.sha256(b).hexdigest()
old = R / 'research/prototypes/vendor_fa_finite_p1_shared_mean_reuse.cu'
s = old.read_text(encoding='utf-8')

def once(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)

s = s.replace('FP16', 'BF16').replace('mctlass::half_t', 'mctlass::bfloat16_t')
once('Flash_fwd_kernel_traits<128,', 'Flash_fwd_kernel_traits<256,')
once('    int batch,heads,kv_heads,length;',
     '    const int *valid_lengths;\n    int batch,heads,kv_heads,length;')
once('    const int length=p.length;', '''    // Padded length remains the storage stride; valid bounds are per sample.
    const int length=p.length, valid=p.valid_lengths[bh/p.heads];''')
once('    extern __shared__ char scratch[];', '''    // Every padding output is written, including entire empty tail tiles.
    // This branch is uniform within the block, before any synchronization.
    if(row0>=valid) {
        if constexpr(Phase==0) {
            for(int j=tid;j<M;j+=blockDim.x) if(row0+j<length) {
                const int64_t pos=int64_t(bh)*length+row0+j;
                p.tau[pos]=0.f;p.center[pos]=0.f;
            }
        } else {
            E *out=Phase==1?p.dq:p.dk;
            for(int j=tid;j<M*D;j+=blockDim.x) if(row0+j/D<length) {
                const int64_t pos=head_offset+int64_t(row0)*D+j;
                out[pos]=E(0.f);
                if constexpr(Phase==2) p.dv[pos]=E(0.f);
            }
        }
        return;
    }
    extern __shared__ char scratch[];''')
once('const int end=Phase==2 ? length : min(length,row0+M);',
     'const int end=Phase==2 ? valid : min(valid,row0+M);')
assert s.count('predA,length-row0') == 1 and s.count('predB,length-col0') == 2
s = s.replace('predA,length-row0', 'predA,valid-row0')
s = s.replace('predB,length-col0', 'predB,valid-col0')
once('query<length && key<length && key<=query', 'query<valid && key<valid && key<=query')
once('p.tau[int64_t(bh)*length+r]=denominator(i);',
     'p.tau[int64_t(bh)*length+r]=r<valid?denominator(i):0.f;')
once('p.center[int64_t(bh)*length+r]=numerator(i)/denominator(i);',
     'p.center[int64_t(bh)*length+r]=r<valid?numerator(i)/denominator(i):0.f;')
once('out[pos]=E(accQ(i)*p.scale);', 'out[pos]=E(r<valid?accQ(i)*p.scale:0.f);')
once('p.dv[pos]=E(accV(i));', 'p.dv[pos]=E(r<valid?accV(i):0.f);')
once('extern "C" int deltatrace_fa_finite_p1_shared_mean_reuse(',
     'extern "C" int deltatrace_fa_finite_p1_bf16_d256(')
once('void *dk,void *dv,int batch,', 'void *dk,void *dv,const void *valid_lengths,int batch,')
once('(E*)dq,(E*)dk,(E*)dv,batch,', '(E*)dq,(E*)dk,(E*)dv,(const int*)valid_lengths,batch,')
once(' * No integral quadrature,', ' * D256; right-padding lengths; full padding tiles skip arithmetic and write zeros.\n * No integral quadrature,')
target = R / 'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu'
target.write_text(s, encoding='utf-8', newline='\n')

build = (A / 'fa_shared_mean_reuse_build_20260907.py').read_text(encoding='utf-8')
build = build.replace('vendor_fa_finite_p1_shared_mean_reuse.cu', target.name)
build = build.replace('libdeltatrace_fa_finite_shared_mean_reuse.so', 'libdeltatrace_fa_finite_bf16_d256.so')
build = build.replace("assert 'deltatrace_fa_finite_p1_shared_mean_reuse'", "assert 'deltatrace_fa_finite_p1_bf16_d256'")
ast.parse(build)
study = A / 'qwen35_finite_fa_build_20260908.py'
study.write_text(build, encoding='utf-8', newline='\n')
previous = json.loads((A / 'fa_shared_mean_reuse_operator_protocol_20260907.json').read_bytes())
p = {k: previous[k] for k in ('vendor_source_parent', 'vendor_parent_sha256', 'vendor_git_commit',
     'vendor_source_version', 'installed_model_FA_version', 'same_version_source_claim', 'max_compile_seconds')}
p.update(study_sha256=sha(study.read_bytes()), extension_sha256=sha(target.read_bytes()),
    derived_from_sha256=sha(old.read_bytes()), purpose='BF16/D256 and true right-padding support; preserved finite content_P1 math, symmetric Q/K interaction and three native-framework tile sweeps.',
    budget={'compiles': 1, 'tile_candidates': 1, 'model_calls': 0, 'GPU_kernel_launches': 0, 'quality_queries': 0},
    traits={'head_dim': 256, 'tile_m': 32, 'tile_n': 32, 'warps': 2, 'shared_bytes_per_phase': 32768},
    stop='One compiler attempt. Build success alone does not validate actual operands, gradient limit, model attribution, quality or speed.')
protocol = A / 'qwen35_finite_fa_build_protocol_20260908.json'
protocol.write_text(json.dumps(p, indent=2), encoding='utf-8', newline='\n')
subprocess.run([sys.executable, str(A / 'prepare_remote_experiment.py'),
    '${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v1', 'study.py=' + str(study),
    'protocol.json=' + str(protocol), target.name + '=' + str(target),
    '--request', str(A / 'launch_qwen35_finite_fa_build_20260908.json')], check=True)
print(json.dumps({'extension_sha256': p['extension_sha256'], 'protocol_sha256': sha(protocol.read_bytes())}))
