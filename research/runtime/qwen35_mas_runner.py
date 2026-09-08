"""Named Qwen3.5 profile for the measured conditional-MAS repair.

This only assembles the exact configuration evaluated in the preserved pilots.
The caller supplies the same validated native-backed finite FA/FLA operators.
The historical Qwen35DenseFiniteRunner defaults remain independently usable.
"""
from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner
from layer0_fla_endpoint_average_20260908 import Layer0FLAEndpointAverage


def make_qwen35_mas_runner(model, finite_fa, finite_fla):
    """Symmetric layer-zero norm-gate plus complete FLA endpoint-order average.

    No deletion-state data, model replacement, score calibration or extra model
    forward is used. All other layers retain the supplied original backends.
    """
    average = Layer0FLAEndpointAverage(finite_fla)
    return Qwen35DenseFiniteRunner(
        model,
        finite_fa,
        finite_fla,
        norm_gate_rules={0: 'symmetric'},
        finite_fla_by_layer={0: average},
    )
