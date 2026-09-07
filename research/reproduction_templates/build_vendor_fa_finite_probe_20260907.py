"""Freeze actual FA-framework extension build; no quadrature/Flex runtime."""
import hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_source_gpu_language_20260907_v1/results.json'
p={'purpose':'Compile genuine vendor FA framework finite-P1 extension. FP16 operands and output, FP32 MMA/softmax/row reductions. No quadrature, no FlexAttention, no dense global NxN buffers, no native model/package/backward replacement. No runtime or quality claim from compilation.',
 'study_sha256':sha(A/'vendor_fa_finite_build_20260907.py'),'extension_sha256':sha(A/'vendor_fa_finite_p1.cu'),
 'vendor_source_parent':'${ARTIFACT_ROOT}/codex_vendor_fa_source_gpu_language_20260907_v1','vendor_parent_sha256':sha(parent),
 'vendor_git_commit':'aef88de756194e077460a1c8b8b341baa429259a','vendor_source_version':'FA2.5.3',
 'installed_model_FA_version':'2.6.3+metax3.5.3.9torch2.8','same_version_source_claim':False,
 'max_compile_seconds':600,'budget':{'native_model_forwards':0,'native_model_backwards':0,'attribution_calls':0,'GPU_kernel_launches':0},
 'precision_scope':'Ordinary FA-style FP16 matrix operands / FP32 accumulation and nonlinear operations / FP16 emitted multipliers. Real arithmetic finite rule unchanged; native LSE vs explicit re-normalization and half rounding require actual numerical/quality checks.',
 'next_if_compiled':'Validate full original NI0 actual attention tensors and multipliers before any whole-model use; compare current dense P1, FP64 rule and true kernel counts/time/allocated shapes. Compilation alone cannot promote the extension.'}
path=A/'vendor_fa_finite_build_protocol_20260907.json';path.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_finite_build_20260907_v1',
 f'study.py={A/"vendor_fa_finite_build_20260907.py"}',f'protocol.json={path}',f'vendor_fa_finite_p1.cu={A/"vendor_fa_finite_p1.cu"}',
 '--request',str(A/'vendor_fa_finite_build_launch_20260907.json')],check=True)
print('Prepared finite FA extension build; zero GPU kernel executions planned.')
