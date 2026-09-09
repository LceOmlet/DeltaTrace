"""Clean Qwen3.5 DT profile, before the MAS-driven layer-specific changes."""
from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner


def make_qwen35_clean_runner(model, finite_fa, finite_fla):
    """Use the established P1 FA and content1 GDN propagation at every layer."""
    return Qwen35DenseFiniteRunner(model, finite_fa, finite_fla)
