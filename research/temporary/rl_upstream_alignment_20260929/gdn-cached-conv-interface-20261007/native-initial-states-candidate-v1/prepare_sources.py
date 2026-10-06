"""Prepare a default-disabled, patched-owner candidate; no remote/GPU work.

Change only the cached native call's public interface and capture representation.
The original finite secants, seed placement, native input VJP, FA/FLA and Q/V/A
arithmetic remain owned by their original code. No model forward is copied into
a replacement module. These files are prepared-only, not a deployment.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
REPO = AUDIT.parents[2]


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def load(path, expected):
    if identity(path)['sha256'] != expected:
        raise RuntimeError('Frozen source differs: ' + str(path))
    return path.read_bytes().decode('utf-8')


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Expected one existing owner seam: ' + old[:100])
    return text.replace(old, new, 1)


model_path = AUDIT / 'textcraft-degradation-20261005/owner-sources/modeling_qwen3_5.py'
model_before = load(model_path, 'f7e1a804fa12684bd1cc225c85cdf5f0b5996f30d66263f11de13449f53272be')
model = replace(model_before,
    '''            if use_precomputed_states:
                # Cached chunked-tokens decode: prepend the cached conv context so the causal conv
                # sees the correct left-context rather than zero-padding. Dropped from the output
                # at the end of this branch.
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
''',
    '''            conv_initial_states = None
            if use_precomputed_states:
                if (
                    getattr(self, "_deltatrace_native_conv_initial_states", False)
                    and self.causal_conv1d_fn is not None
                    and seq_len >= self.conv_kernel_size
                    and kwargs.get("seq_idx") is None
                ):
                    # The original cache update below uses copy_ in-place.
                    # Snapshot the small prior window before that owner update.
                    conv_initial_states = conv_state[..., -(self.conv_kernel_size - 1) :].transpose(1, 2).clone(
                        memory_format=torch.contiguous_format
                    ).transpose(1, 2)
                else:
                    # Unchanged default, single-/short-token and fallback paths.
                    mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
''')
model = replace(model,
    '                    seq_idx=kwargs.get("seq_idx"),\n',
    '                    seq_idx=kwargs.get("seq_idx"),\n'
    '                    initial_states=conv_initial_states,\n')
model = replace(model,
    '            if use_precomputed_states:\n                mixed_qkv = mixed_qkv[:, :, -seq_len:]\n',
    '            if use_precomputed_states and conv_initial_states is None:\n'
    '                mixed_qkv = mixed_qkv[:, :, -seq_len:]\n')

gdn_path = REPO / 'deltatrace/clean/qwen35/qwen35_gdn_finite.py'
gdn_before = load(gdn_path, 'fcbfd9e74a6c40c3bf970d7d24ec42e0f08e5d513ff49eb1ade6a5fcd52e9fca')
gdn = replace(gdn_before,
    '''            self.cached_conv_context=f['x'].shape[2]-self.input_shape[1]
            self.values['projected_qkv']=self.copy(self.select_time(f['x'],2,start=self.conv_context_start))
''',
    '''            if f.get('initial_states') is None:
                self.cached_conv_context=f['x'].shape[2]-self.input_shape[1]
                self.values['projected_qkv']=self.copy(self.select_time(f['x'],2,start=self.conv_context_start))
            else:
                # Public initial_states consumes fixed history separately.
                # Preserve actual paired operands, with compact channel-last
                # storage when the existing capture range starts after zero.
                self.cached_conv_context=0
                start=self.coefficient_start
                projected=f['x']
                initial=f['initial_states']
                if start:
                    width=f['weight'].shape[-1]
                    initial=projected[...,start-width+1:start].transpose(1,2).contiguous().transpose(1,2)
                    projected=projected[...,start:].transpose(1,2).contiguous().transpose(1,2)
                self.values['projected_qkv']=self.copy(projected)
                self.values['conv_initial_states']=self.copy(initial)
''')
gdn = replace(gdn,
    "    restore(c,'projected_qkv','conv_output')\n",
    "    restore(c,'projected_qkv','conv_output')\n"
    "    if c.get('conv_initial_states') is not None:\n"
    "        restore(c,'conv_initial_states')\n")
gdn = replace(gdn,
    '        pre=module.causal_conv1d_fn(projected,module.conv1d.weight.squeeze(1),activation=None)\n',
    "        pre=module.causal_conv1d_fn(projected,module.conv1d.weight.squeeze(1),\n"
    "            initial_states=c.get('conv_initial_states'),activation=None)\n")

runner_path = REPO / 'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
runner_before = load(runner_path, '6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa')
runner = replace(runner_before, 'import time\n', 'import time\nfrom contextlib import contextmanager\n')
runner = replace(runner, 'class Qwen35DenseFiniteRunner:\n',
    '''@contextmanager
def _native_conv_initial_states_scope(layers,enabled):
    """Set only the patched owner's execution option, then restore it."""
    if not enabled:
        yield
        return
    name='_deltatrace_native_conv_initial_states'
    saved=[]
    try:
        for layer in layers:
            if layer.block_type!='linear_attention':continue
            module=layer.linear_attn
            saved.append((module,name in vars(module),getattr(module,name,None)))
            setattr(module,name,True)
        yield
    finally:
        for module,had,value in reversed(saved):
            if had:setattr(module,name,value)
            else:delattr(module,name)


class Qwen35DenseFiniteRunner:
''')
runner = replace(runner, 'reuse_native_prefix=False):',
                 'reuse_native_prefix=False,native_conv_initial_states=False):')
runner = replace(runner, '        self.reuse_native_prefix=reuse_native_prefix\n',
    '        self.reuse_native_prefix=reuse_native_prefix\n'
    '        self.native_conv_initial_states=native_conv_initial_states\n')
runner = replace(runner,
    "            with torch.no_grad():out=timed('native_root_with_CPU_checkpoints',lambda:model(",
    "            with torch.no_grad(),_native_conv_initial_states_scope(layers,self.native_conv_initial_states):out=timed('native_root_with_CPU_checkpoints',lambda:model(")
runner = replace(runner,
    '                with torch.no_grad(),dc,mc:return layer(x,**kw)\n',
    '                with torch.no_grad(),dc,mc,_native_conv_initial_states_scope([layer],self.native_conv_initial_states):return layer(x,**kw)\n')

# Prove that the finite function changed only transport/API representation.
# No secant, seed-placement, contraction or dtype statement may change.
before_tree, after_tree = ast.parse(gdn_before), ast.parse(gdn)
def function(tree, name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
finite_before = function(before_tree,'gdn_finite_pullback')
finite_after = function(after_tree,'gdn_finite_pullback')
class OriginalRepresentation(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.unparse(node.test)=="c.get('conv_initial_states') is not None":return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        if ast.unparse(node.func)=='module.causal_conv1d_fn':
            node.keywords=[kw for kw in node.keywords if kw.arg!='initial_states']
        return self.generic_visit(node)
assert ast.dump(finite_before)==ast.dump(OriginalRepresentation().visit(finite_after))
for name in ('_scalar_secant','_silu_derivative','_l2_pullback','_norm_gate_finite_rule','_conv_silu_finite_rule'):
    assert ast.dump(function(before_tree,name))==ast.dump(function(after_tree,name))

outputs=[]
for relative, before, after in (
    ('transformers/models/qwen3_5/modeling_qwen3_5.py',model_before,model),
    ('deltatrace/clean/qwen35/qwen35_gdn_finite.py',gdn_before,gdn),
    ('deltatrace/clean/qwen35/qwen35_dense_finite_runner.py',runner_before,runner),
):
    ast.parse(after,filename=relative)
    path=HERE/'candidate_sources'/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(after.encode('utf-8'))
    patch=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
                                     fromfile='a/'+relative,tofile='b/'+relative))
    patch_path=HERE/(Path(relative).name+'.patch')
    patch_path.write_bytes(patch.encode('utf-8'))
    outputs.append(dict(relative=relative,prepared=identity(path),patch=identity(patch_path),
                        before_sha256=hashlib.sha256(before.encode()).hexdigest()))

costs=[]
for rank in range(2):
    path=HERE.parent/'long-v1'/f'rank{rank}-operator-result.json'
    data=json.loads(path.read_bytes())
    native={key:item['median_cuda_ms'] for key,item in data['native_benchmark']['operations'].items()}
    finite={key:item['median_cuda_ms'] for key,item in data['finite_benchmark']['operations'].items()}
    saving=native['native_actual_concat_then_kernel']-native['initial_states']
    extra=finite['initial_states']-finite['existing_materialized_cat']
    costs.append(dict(rank=rank,source=identity(path),native_ms=native,finite_ms=finite,
        root_only_24_layers_kernel_budget_saved_seconds=24*saving/1000,
        root_and_replay_48_layers_less_24_finite_increments_seconds=(48*saving-24*extra)/1000,
        caveat='Conditional same-shape kernel budget only; excludes capture transport, compaction, cache-window snapshot, full DT scheduling and FSDP/peer waits'))

report=dict(status='prepared_only_not_deployed_not_GPU_verified',script=identity(Path(__file__)),
    sources=[identity(model_path),identity(gdn_path),identity(runner_path)],outputs=outputs,
    default=dict(runner_native_conv_initial_states=False,model_instance_option_default=False,
        unmodified_paths=['no cache','single token','short suffix smaller than conv_kernel_size','no official conv kernel','seq_idx present','prefix-bank preparation']),
    owner_api='Original causal_conv1d_fn(initial_states=...) and original autograd input VJP',
    lifecycle=dict(original_cache_update='Unchanged F.pad and update_conv_state; for suffix>=kernel_size, tail is the same original projection tail',
        inplace_state_source=dict(path='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/lib/python3.12/site-packages/transformers/cache_utils.py',
            sha256='16bbbba2394c0151af53009800e5a4052479417a8c7e5edfbd5ef128a46fcbc1',line=961,
            observation='Original LinearAttentionLayer uses self.conv_states.copy_ in-place'),
        initial_window='Clone actual last kernel_size-1 states in channel-last format before original cache update; fixed per endpoint',
        finite_context='New suffix preactivation has context0; original code computes context, seed and suffix dx without a new propagation formula',
        compact_capture='At positive original 64-token coefficient start, use actual prior kernel_size-1 projected tokens as fixed initial state; capture exact remaining suffix compact channel-last',
        capture_contract='One original conv/FLA call; no reconstruction of a full cached concat; no extra model forward/backward'),
    arithmetic_AST_checks=dict(gdn_finite_body_identical_after_removing_only_optional_state_restore_and_API_keyword=True,
        scalar_secant_silu_l2_norm_gate_conv_silu_functions_identical=True),
    performance_evidence=costs,
    numerical_scope=dict(owner_test='causal-conv1d v1.5.0 test_causal_conv1d original output/dx assertions',
        test_sha256='c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf',
        tolerances_AST_unchanged=True,actual_dtype='BF16 including frozen weights; outside the original FP32-weight fixture',
        current_operator_evidence='Both original real long cases pass all nine original output/dx assertion expressions',
        missing='Integrated cached root/replay and compact capture not run; no whole-DT official tolerance invented'),
    next_smallest_checks=['Default-disabled frozen owner path behavior unchanged',
        'Same original base-model B8/current saved request bank, resume disabled, no checkpoint or update; enable candidate runner only in diagnostic',
        'Record actual source hashes and original effective rank8/alpha16/B4/32k config before first DT call',
        'At actual conv calls retain official original output/input-VJP reference checks; report actual dtype scope',
        'Check cache-window bytes, context0 capture/output lengths, and compact start64+ representation against actual owner operands',
        'Measure root, replay, finite, capture copies, physical VRAM and phase PSS/cgroup separately; count all relocated work'],
    operations=dict(remote_writes=0,gpu_calls=0,checkpoint_loads=0,production_edits=0,installs=0))
(HERE/'prepared-sources.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],output=str(HERE/'prepared-sources.json'),
                     sources=[item['prepared'] for item in outputs],performance=costs)))
