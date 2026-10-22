"""Wire the existing native SSM-only cache option into the pinned GDN owner.

The original shared MambaStateDtypeCalculator resolves both native cache
options. No dtype algorithm, kernel, model, sampler, credit, or optimizer is
implemented here. Installed files remain unchanged. The candidate is bound
only in a diagnostic process before original model/cache initialization.
"""
import ast
import hashlib
import importlib
import inspect
from pathlib import Path

SPECS = {
    'qwen3_next': {
        'original_SHA256': '1ddb0915e8a61fbb98411665d769e25d0cb28d753ead30751dbacd643027c03b',
        'candidate': 'candidate_metax_qwen3_next_ssm_dtype.py',
        'methods': (('Qwen3NextGatedDeltaNet', 'get_state_dtype'),
                    ('Qwen3NextForCausalLM', 'get_mamba_state_dtype_from_config')),
    },
    'qwen3_5': {
        'original_SHA256': 'd282661ff26a3b213cef3eb11eed63c6a4102546e4dee156f2b3977c69c5a287',
        'candidate': 'candidate_metax_qwen3_5_ssm_dtype.py',
        'methods': (('Qwen3_5ForConditionalGeneration', 'get_mamba_state_dtype_from_config'),),
    },
}
RESOLVER_SHA256 = '750767c9168fee80ec2da1e9b2e35a16cedeae59568fb010331398991e3135ea'


def build_source(original, module_name):
    spec = SPECS[module_name]
    patched = original
    if module_name == 'qwen3_next':
        before = '''MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )'''
        after = '''MambaStateDtypeCalculator._mamba_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype
        )'''
        assert patched.count(before) == 1
        patched = patched.replace(before, after)
    before = '''MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )'''
    after = '''MambaStateDtypeCalculator._mamba_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype
        )'''
    assert patched.count(before) == 1
    patched = patched.replace(before, after)
    old_tree, new_tree = ast.parse(original), ast.parse(patched)
    for tree in (old_tree, new_tree):
        for class_name, method_name in spec['methods']:
            cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
            method, = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name]
            method.body = [ast.Pass()]
    assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
    return patched


def install():
    records = []
    for module_name, spec in SPECS.items():
        module = importlib.import_module('vllm_metax.models.' + module_name)
        original_path = Path(module.__file__)
        assert hashlib.sha256(original_path.read_bytes()).hexdigest() == spec['original_SHA256']
        resolver = module.MambaStateDtypeCalculator
        resolver_path = Path(inspect.getfile(resolver))
        assert hashlib.sha256(resolver_path.read_bytes()).hexdigest() == RESOLVER_SHA256
        candidate_path = Path(__file__).with_name(spec['candidate'])
        candidate_source = build_source(original_path.read_text(), module_name)
        assert candidate_path.read_bytes() == candidate_source.encode()
        tree = ast.parse(candidate_source)
        for class_name, method_name in spec['methods']:
            cls = getattr(module, class_name)
            old_descriptor = cls.__dict__[method_name]
            old_function = old_descriptor.__func__ if isinstance(old_descriptor, classmethod) else old_descriptor
            assert Path(inspect.getfile(old_function)).resolve() == original_path.resolve()
            cls_node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
            method, = [n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == method_name]
            namespace = dict(old_function.__globals__)
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(candidate_path), 'exec'), namespace)
            new_descriptor = namespace[method_name]
            assert isinstance(old_descriptor, classmethod) == isinstance(new_descriptor, classmethod)
            new_function = new_descriptor.__func__ if isinstance(new_descriptor, classmethod) else new_descriptor
            assert inspect.signature(old_function) == inspect.signature(new_function)
            setattr(cls, method_name, new_descriptor)
            records.append(dict(
                module=module.__name__, owner_class=class_name, method=method_name,
                original_path=str(original_path), original_SHA256=spec['original_SHA256'],
                candidate_path=str(candidate_path),
                candidate_SHA256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                original_shared_resolver=str(resolver_path),
                original_shared_resolver_SHA256=RESOLVER_SHA256,
                change='Forward existing CacheConfig.mamba_ssm_cache_dtype to the original shared dtype resolver',
                kernel_implementations_changed=0, installed_files_changed=0,
                formal_training_deployed=False))
    return dict(status='candidate_native_SSM_dtype_interface_bound_in_this_process_only',
                methods=records,
                default_auto_behavior='Unchanged; the launch must explicitly pass the documented native SSM dtype option from the provisioned model declaration.',
                model_kernel_changes=0, numerical_tolerance_verdict=None)
