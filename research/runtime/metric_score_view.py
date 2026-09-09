"""The user-selected positive score view for original needle/RISE/MAS.

Keep the finite attribution's signed output separately. This view changes both
the deletion order and MAS density; old signed-order curves are not reusable by
default. It is evaluation preprocessing, not a change to finite propagation.
"""

def positive_metric_scores(signed_scores):
    return signed_scores.float().clamp_min(0)
