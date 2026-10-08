"""Unaccepted local log-softmax finite-rule experiment; never a launch default.

Derivation: endpoint-head-derivation.json. The original FiniteAnswerOps owns
packing, projection, diagnostics, compilation and reverse propagation. This
module implements only the candidate mathematical seed, not those interfaces.
"""
import copy
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import torch


def seed_with_checks(z0, z1, target):
    lp0, lp1 = z0.log_softmax(-1), z1.log_softmax(-1)
    valid = torch.isfinite(lp0) & torch.isfinite(lp1)
    same_mask = (torch.isfinite(lp0) == torch.isfinite(lp1)).all()
    u = torch.where(valid, lp1-lp0, torch.zeros_like(lp0))
    a = u.abs()
    safe = torch.where(a > 0, a, torch.ones_like(a))
    # Positive KL decomposition avoids cancellation of signed KL terms.
    # The series is a local numerical evaluation of the same analytic rule.
    small = 0.5+a*(-1/12+a.square()*(1/720+a.square()*(-1/30240+a.square()/1209600)))
    large = 1/safe-torch.exp(-safe)/(-torch.expm1(-safe))
    fraction = torch.where(a <= 0.5, small, large)
    maximum = torch.where(valid, torch.maximum(lp0, lp1), torch.full_like(lp0, -torch.inf)).exp()
    jeffreys = maximum*a*(-torch.expm1(-a))
    d0 = (jeffreys*torch.where(u >= 0, fraction, 1-fraction)).sum(-1, keepdim=True)
    d1 = (jeffreys*torch.where(u >= 0, 1-fraction, fraction)).sum(-1, keepdim=True)
    total = d0+d1
    weight = torch.where(total > 0, d0/torch.where(total > 0, total, torch.ones_like(total)),
                         torch.full_like(total, 0.5))
    p0 = torch.where(valid, lp0.exp(), torch.zeros_like(lp0))
    p1 = torch.where(valid, lp1.exp(), torch.zeros_like(lp1))
    q = (1-weight)*p0+weight*p1
    checks = same_mask & torch.isfinite(q).all() & (q >= 0).all() & (q.sum(-1) > 0).all()
    seed = -q
    seed = seed.clone()
    seed.scatter_add_(-1, target.unsqueeze(-1), seed.new_ones((*target.shape, 1)))
    return seed, checks


def make_candidate(original, spec, environment):
    """Reuse an isolated instance of the actual answer owner, with one seed seam."""
    path = spec['head_owner']['path']
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest == spec['head_owner']['sha256']
    name = 'research_endpoint_head_owner'
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    native_seed = module.seed_with_checks
    module.seed_with_checks = seed_with_checks
    candidate = copy.copy(original)
    candidate.answer = module.FiniteAnswerOps(environment.get('dt_answer_compiled', True),
        dynamic_shapes=environment.get('dt_dynamic_shapes', False),
        compiler_options=environment.get('dt_compiler_options', {}))
    audit = {}
    compiled_projection = candidate.answer.seed
    def first_actual_dtype_check(z0, z1, target, weight):
        result = compiled_projection(z0, z1, target, weight)
        if not audit:
            # Actual model inputs, a bounded primitive diagnostic only. This
            # is not a synthetic quality set or a borrowed FA/FLA tolerance.
            n = min(128, len(target))
            x0, x1, labels = z0[:n].float(), z1[:n].float(), target[:n]
            current, valid = seed_with_checks(x0, x1, labels)
            previous, previous_valid = native_seed(x0, x1, labels)
            ref, ref_valid = seed_with_checks(x0.double(), x1.double(), labels)
            delta = x1.double()-x0.double()
            endpoint = x1.double().log_softmax(-1).gather(-1, labels[:,None]).squeeze(-1)
            endpoint -= x0.double().log_softmax(-1).gather(-1, labels[:,None]).squeeze(-1)
            reference_seed_error = float((current.double()-ref).abs().max())
            original_endpoint_error = float(((previous.double()*delta).sum(-1)-endpoint).abs().max())
            new_endpoint_error = float(((current.double()*delta).sum(-1)-endpoint).abs().max())
            compiled_endpoint_error = float((result[-1][:n].double()-endpoint).abs().max())
            audit.update(actual_rows=n, total_first_call_rows=len(target), input_dtype=str(z0.dtype),
                seed_dtype=str(current.dtype), reference_seed_dtype=str(ref.dtype),
                new_seed_max_abs_FP32_vs_FP64=reference_seed_error,
                original_seed_endpoint_max_abs_FP64_reference=original_endpoint_error,
                new_seed_endpoint_max_abs_FP64_reference=new_endpoint_error,
                compiled_allocated_endpoint_max_abs_FP64_reference=compiled_endpoint_error,
                owner_valid=bool(valid & previous_valid & ref_valid),
                not_an_official_tolerance_test=True,
                reference='Same mathematical seed at FP64; finite small-series truncation remains below 4.1e-11 per scalar.',
                quality_scope='Only primitive numeric evaluation; whole-collection author curves remain primary.')
        return result
    candidate.answer.seed = first_actual_dtype_check
    assert candidate.model is original.model and candidate.capture_backend is original.capture_backend
    assert candidate.finite_fa is original.finite_fa and candidate.finite_fla is original.finite_fla
    assert original.answer.__class__.__module__ != module.__name__
    return candidate, dict(kind='endpoint_head', head_owner=spec['head_owner'], actual_dtype_check=audit,
        seed_owner_path=inspect.getsourcefile(seed_with_checks),
        seed_owner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_seed_path=inspect.getsourcefile(native_seed),
        original_runner_and_kernels=True, same_model_object=True,
        same_capture_backend=True, extra_model_forwards=0, extra_DT_calls=0,
        production_modified=False, accepted_candidate=False)
