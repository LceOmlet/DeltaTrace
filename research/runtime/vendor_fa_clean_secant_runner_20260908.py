"""Explicit all-FA finite backend adapter; no layer/call-count rule dispatch."""
import torch
from vendor_fa_clean_secant_20260908 import VendorFACleanSecant, ROW_FIELDS


class AllFACleanSecantBackend(VendorFACleanSecant):
    def __init__(self, library, expected_sha256):
        super().__init__(library, expected_sha256)
        self.row_receipts = []

    def __call__(self, operands, scale, layout, activity=None):
        result = super().__call__(operands, scale, layout)
        # Explicit diagnostic overhead remains inside actual attribution timing.
        assert all(bool(torch.isfinite(value).all()) for value in result.values())
        rows = result['rows'].detach().cpu()
        names = {name: rows[i].double() for i, name in enumerate(ROW_FIELDS)}
        zero = names['variance_M2'] == 0
        assert bool((names['variance_M2'] >= 0).all()) and bool((names['alpha'][zero] == 0).all())
        assert bool((names['raw_p0_sum'] > 0).all()) and bool((names['raw_p1_sum'] > 0).all())
        self.row_receipts.append({'shape': list(rows.shape), 'row_state_bytes': rows.numel() * rows.element_size(),
            'zero_variance_count': int(zero.sum()), 'max_absolute_alpha': float(names['alpha'].abs().max()),
            'max_correction_L2': float((names['alpha'].abs() * names['variance_M2'].sqrt()).max()),
            'degenerate_target_residual_sum': float(names['normalized_route_target'][zero].sum()),
            'normalization_target_change_sum': float((names['normalized_route_target'] - names['raw_route_target']).sum()),
            'endpoint_positive_sum': float(names['endpoint_positive'].sum()),
            'endpoint_negative_sum': float(names['endpoint_negative'].sum()),
            'row_constraint_residual_sum': float((names['endpoint_positive'] + names['endpoint_negative'] - names['normalized_route_target']).sum())})
        if activity is not None:
            _, heads, length, _ = operands['q0'].shape
            activity.update(query_heads=heads, kv_heads=operands['k0'].shape[1], valid_lengths=list(layout.lengths),
                padded_length=length, GQA_input_expansion=False, global_attention_matrix=False,
                kernel_launches_per_call=3, finite_rule='normalized_clean_jacobian_minimum_norm_secant')
        return result
