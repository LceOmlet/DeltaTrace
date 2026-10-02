"""Compatibility import for the prepared artifact diagnostic.

The single implementation now belongs to the Qwen DT owner module. Frozen
remote diagnostics preserve the prior source and receipts without alteration.
"""
from qwen35_native_prefix_artifacts import (
    NativePrefixArtifacts, NativePrefixLease, compose_native_prefix_cache,
)
