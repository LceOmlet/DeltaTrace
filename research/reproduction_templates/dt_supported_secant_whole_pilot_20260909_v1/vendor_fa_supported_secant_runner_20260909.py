"""Signature/count adapter for the existing supported-secant finite operator.

No input-dependent rule selection, coefficient arithmetic, fallback, extra
operator, row contraction or attention implementation is introduced here.
"""
from vendor_fa_supported_secant_20260908 import VendorFASupportedSecant, ROW_FIELDS


class AllFASupportedSecantBackend(VendorFASupportedSecant):
    def __init__(self, library, expected_sha256):
        super().__init__(library, expected_sha256)
        self.receipts = []

    def __call__(self, operands, scale, layout, activity=None):
        row = {'status': 'entered', 'kernel_phases_on_success': 3}
        self.receipts.append(row)
        if activity is not None:
            activity['calls_attempted'] = activity.get('calls_attempted', 0) + 1
        result = super().__call__(operands, scale, layout)
        rows = result['rows']
        row.update(status='returned', row_state_shape=list(rows.shape),
                   row_state_bytes=rows.numel() * rows.element_size(),
                   row_field_count=len(ROW_FIELDS), original_wrapper_synchronization=True)
        if activity is not None:
            activity.update(kernel_launches_per_call=3, global_attention_matrix=False,
                            GQA_input_expansion=False, finite_rule='supported_secant',
                            row_state_bytes=row['row_state_bytes'])
            activity['calls_enqueued'] = activity.get('calls_enqueued', 0) + 1
        return result
