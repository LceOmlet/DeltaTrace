"""Official dynamic-shape compilation of existing DT finite expressions.

No formula copies and no native model/FA/FLA changes. Compilation failures are
errors through fullgraph=True; no eager fallback or alternate attention exists.
"""
import torch
from qwen35_decoder_finite import FiniteBoundaryOps
from qwen35_answer_finite import FiniteAnswerOps
from finite_fla_gpu import mixed_coefficients,native_input_adjoints


def configure_dynamic_finite(runner):
    def compile_existing(function):
        return torch.compile(function,fullgraph=True,dynamic=True,
                             options={'triton.cudagraphs':False,'max_autotune':False})
    boundaries=FiniteBoundaryOps(compiled=False)
    for name in ('mlp','norm_residual','attention_gate','attention_input'):
        setattr(boundaries,name,compile_existing(getattr(boundaries,name)))
    answer=FiniteAnswerOps(compiled=False)
    answer.seed=compile_existing(answer.seed)
    mixed=compile_existing(mixed_coefficients)
    def finite_fla(endpoints,upstream,scale):
        return mixed(endpoints,native_input_adjoints(endpoints,upstream,scale),scale,reuse_scalar_products=False)
    runner.boundaries=boundaries;runner.answer=answer;runner.finite_fla=finite_fla
    return runner
