"""Select DT's metric inputs without replacing any original metric function."""


def signed_rise_equals_positive_curve(signed_prompt, keep, positive_curve):
    """Sufficient condition to reuse an original RISE result without extra calls.

    Strictly ordered positive scores have the same order after clamping. If
    the author's running-minimum response reaches zero while deletion still
    contains only positive scores, later ordering cannot change its RISE.
    This is an identity check on a returned original curve, not a new metric.
    """
    import torch
    if not torch.isfinite(signed_prompt).all():
        return None
    weights=signed_prompt[keep]
    positive=weights[weights>0]
    if len(positive)!=len(torch.unique(positive)):
        return None
    response=positive_curve['normalized_model_response']
    zero=next((i for i,value in enumerate(response) if value==0),None)
    if zero is None:
        return None
    deleted=positive_curve['deleted_user_indices'][zero]
    # Also require the entire budget through this step to be within positives.
    # This handles originally-EOS tokens that do not appear as changed inputs.
    steps=len(response)-1
    if steps<1:
        return None
    base,remainder=divmod(len(keep),steps)
    budget=zero*base+min(zero,remainder)
    if budget>len(positive) or any(float(signed_prompt[j])<=0 for j in deleted):
        return None
    return {'kind':'identical_positive_prefix_then_original_monotone_zero',
            'zero_step':zero,'positive_count':len(positive),'deletion_budget_at_zero':budget}
