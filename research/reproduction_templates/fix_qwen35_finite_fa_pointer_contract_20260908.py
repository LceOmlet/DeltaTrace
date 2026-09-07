"""Fix host/device trait type mismatch using the actual FA void-pointer ABI style."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
failed=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v1'
with zipfile.ZipFile(failed/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name;(failed/name).write_bytes(z.read(name))
previous=json.loads((failed/'results.json').read_bytes())
assert previous['status']=='finite_extension_compile_failed'
path=R/'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu'
s=path.read_text(encoding='utf-8')
assert sha(path.read_bytes())==previous['protocol']['extension_sha256']
s=s.replace('const mctlass::bfloat16_t *q0,*k0,*q1,*k1,*v0,*u;', 'const void *q0,*k0,*q1,*k1,*v0,*u;')
s=s.replace('mctlass::bfloat16_t *dq,*dk,*dv;', 'void *dq,*dk,*dv;')
needle='    using E=typename Traits::Element;'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
    // Like native Flash_fwd_params: opaque storage is typed inside each
    // compilation pass. The vendor traits use half_t on the host pass and
    // BF16 on the actual MACA device pass; do not override the vendor header.
    const E *q0=static_cast<const E*>(p.q0), *k0=static_cast<const E*>(p.k0);
    const E *q1=static_cast<const E*>(p.q1), *k1=static_cast<const E*>(p.k1);
    const E *v0=static_cast<const E*>(p.v0), *u=static_cast<const E*>(p.u);
    E *dq=static_cast<E*>(p.dq), *dk=static_cast<E*>(p.dk), *dv=static_cast<E*>(p.dv);''')
# Replace uses only below the declarations, retaining their original ABI reads.
head,body=s.split('    constexpr int M=',1)
kernel,tail=body.split('\nextern "C"',1)
for name in ('q0','k0','q1','k1','v0','u','dq','dk','dv'):
    import re
    kernel=re.sub(r'\bp\.'+name+r'\b',name,kernel)
s=head+'    constexpr int M='+kernel+'\nextern "C"'+tail
path.write_text(s,encoding='utf-8',newline='\n')
p=dict(previous['protocol']);p.update(extension_sha256=sha(path.read_bytes()),
    prior_failed_build_sha256=sha((failed/'results.json').read_bytes()),
    pointer_correction='Use opaque pointers in the finite parameter struct and vendor Traits::Element casts inside the kernel, as native FA does. No vendor-source edits or BF16-to-FP16 device substitution.',
    combined_build_attempts=2,stop='One corrected compile after the preserved host-pass pointer-type failure. No tile search or numerical rule change.')
protocol=A/'qwen35_finite_fa_build_v2_protocol_20260908.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8',newline='\n')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v2',
    'study.py='+str(A/'qwen35_finite_fa_build_20260908.py'),'protocol.json='+str(protocol),path.name+'='+str(path),
    '--request',str(A/'launch_qwen35_finite_fa_build_v2_20260908.json')],check=True)
print(json.dumps({'extension_sha256':p['extension_sha256'],'protocol_sha256':sha(protocol.read_bytes())}))
