"""WITHDRAWN, UNEXECUTED DRAFT. Not an official FT baseline or accepted runtime.

Stopped on the user's 2026-09-08 correction: extracting the author's controller
and feeding custom aggregate values does not establish original FT equivalence.
Preserved only as a record of the rejected approach. Do not launch or integrate.

Stream the pinned author's Both control block to a real batched aggregator.

Only CPU hop bookkeeping is extracted. This module contains no model, attention,
forward or backward implementation. Native model/input preparation is external.
Source capture/parameter construction is replaced by supplied, validated facts;
aggregate calls are yielded to the caller. The original arithmetic, masks and
branching are retained. Synchronous replay checks scheduling, not an independent
mathematical reference. No full generation-by-sequence score matrix is retained.
"""
import ast
import copy
import hashlib
import types
import typing
from pathlib import Path
import torch


class AuthorBothProgram:
    def __init__(self, source_path, expected_sha256, author_core):
        raw = Path(source_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Pinned author Both source mismatch.')
        tree = ast.parse(raw)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMIFRAttributionBoth')
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_ifr_multi_hop_both')
        def assigned(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        begin = next(i for i,n in enumerate(original.body) if assigned(n, 'end_no_eos'))
        end = next(i for i,n in enumerate(original.body) if assigned(n, 'eval_vector'))
        block = copy.deepcopy(original.body[begin:end+1])
        removed = []
        kept = []
        for node in block:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ['_capture_model_state','_build_ifr_params']:
                removed.append({'line':node.lineno, 'call':node.value.func.attr})
            else: kept.append(node)
        assert {x['call'] for x in removed} == {'_capture_model_state','_build_ifr_params'}
        helpers = {'is_stop_token','keep_token_indices','_last_attributable_generation_index','_stop_keep_mask','_build_stop_keep_mask_full'}
        constants = {'STOP_TOKENS','SKIP_WHITESPACE','STRIP_BEFORE_MATCH'}
        nodes = [copy.deepcopy(n) for n in tree.body if
                 isinstance(n, ast.FunctionDef) and n.name in helpers or
                 isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in constants]
        assert len(nodes) == len(helpers)+len(constants)
        namespace = {'torch':torch,'__name__':'pinned_author_both_control',
                     'IFRAggregate':author_core.IFRAggregate,'MultiHopIFRResult':author_core.MultiHopIFRResult,
                     'types':types, **{name:getattr(typing,name) for name in ['Any','Dict','List','Optional','Sequence','Tuple']}}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source_path),'exec'),namespace)
        functions = {}
        for streaming in [False,True]:
            class Boundary(ast.NodeTransformer):
                count = 0
                def visit_Call(self, node):
                    if isinstance(node.func,ast.Name) and node.func.id == 'compute_ifr_sentence_aggregate':
                        self.count += 1
                        names = ['sink_start','sink_end','sink_weights','renorm_threshold']
                        kws = [copy.deepcopy(k) for k in node.keywords if k.arg in names]
                        if streaming:
                            return ast.copy_location(ast.Yield(value=ast.Dict(keys=[ast.Constant(k.arg) for k in kws],values=[k.value for k in kws])),node)
                        return ast.copy_location(ast.Call(func=ast.Name(id='aggregate',ctx=ast.Load()),args=[],keywords=kws),node)
                    return self.generic_visit(node)
            transform = Boundary()
            body = [transform.visit(copy.deepcopy(n)) for n in kept]
            assert transform.count == 2
            template = ast.parse('def run(context, sink_span, thinking_span, n_hops, observation_mask=None, renorm_threshold=0.0, aggregate=None):\n    pass').body[0]
            prefix = ast.parse('self=context\nprompt_len_full=context.prompt_length\ngen_len=context.generation_ids.shape[1]\ntotal_len=prompt_len_full+gen_len\nparams=types.SimpleNamespace(model_dtype=context.model_dtype)').body
            # Return the author's factorized observation; wrapper projection and
            # repeated answer rows are outside hop bookkeeping and not materialized.
            suffix = ast.parse("return {'multi_hop':multi_hop,'stop_keep_mask':stop_keep_mask_full,'all_gen_span':(all_gen_start_abs,all_gen_end_abs),'sink_span':(sink_start_abs,sink_end_abs)}").body
            template.body = prefix+body+suffix
            module = ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[]))
            scope = dict(namespace); exec(compile(module,str(source_path),'exec'),scope)
            functions[streaming] = scope['run']
        self.stream = functions[True]; self.synchronous = functions[False]
        self.namespace = namespace
        self.receipt = {'source_sha256':expected_sha256,'method_lines':[original.lineno,original.end_lineno],
                        'control_lines':[original.body[begin].lineno,original.body[end].end_lineno],
                        'excluded_native_preparation_statements':removed,'yielded_aggregate_sites':2,
                        'retained_control_ast_sha256':hashlib.sha256(ast.dump(ast.Module(body=kept,type_ignores=[]),include_attributes=False).encode()).hexdigest()}

    def context(self, tokenizer, generation_ids, prompt_length, user_positions, user_tokens, generation_tokens, model_dtype):
        ids = generation_ids.detach().cpu()
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2 or prompt_length < 1:
            raise ValueError('This adapter requires one nonempty, fixed response and prompt per context.')
        if len(generation_tokens) != ids.shape[1] or len(user_positions) != len(user_tokens):
            raise ValueError('Token facts have inconsistent lengths.')
        end = self.namespace['_last_attributable_generation_index'](tokenizer,ids,generation_tokens)
        if end < 0: raise ValueError('No attributable generation token.')
        if any(j < 0 or j >= prompt_length for j in user_positions): raise ValueError('User position outside prompt.')
        return types.SimpleNamespace(tokenizer=tokenizer,generation_ids=ids,prompt_length=prompt_length,
                                     user_prompt_indices=list(user_positions),user_prompt_tokens=list(user_tokens),
                                     generation_tokens=list(generation_tokens),model_dtype=model_dtype,renorm_threshold_default=0.0)


def pack_author_aggregate_requests(requests, lengths, dtype=torch.bfloat16, device='cpu'):
    """Use core.py's native-dtype weight cast and normalization before aggregation."""
    if len(requests) != len(lengths) or not lengths: raise ValueError('One request per real batch item is required.')
    width = max(lengths)
    weights = torch.zeros((len(lengths),width),dtype=torch.float32,device=device)
    limits = []
    for b,(request,length) in enumerate(zip(requests,lengths)):
        start,end = request['sink_start'],request['sink_end']
        if not 0 <= start <= end < length or request['renorm_threshold'] != 0.0:
            raise ValueError('Invalid target or unsupported threshold in native adapter.')
        w = request.get('sink_weights')
        if w is None: weights[b,start:end+1] = 1
        else:
            if w.numel() != end-start+1 or not torch.isfinite(w).all(): raise ValueError('Invalid source weights.')
            w = w.detach().to(device=device,dtype=dtype);w = w/(w.sum()+1e-12)
            weights[b,start:end+1] = w.float()
        limits.append(end+1)
    return weights,limits
