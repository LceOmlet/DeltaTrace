"""Adapt the existing CPU native-input verifier to v3's approved boundary.

The official GRPO and deployed compute_advantage functions are called directly.
No new model, DT, backward, optimizer or rollout call is performed.
"""
from pathlib import Path

source = Path(__file__).with_name("verify_live_update_input_v62.py")
text = source.read_text()
changes = {
    "textcraft-dt-grpo-multiplier-no-whiten-20261014-v1": "textcraft-dt-grpo-multiplier-variance-only-20261014-v3",
    "live-native-update-input-v61": "live-native-update-input-variance-v3",
    "DT-group-multiplier-v40-job.json": "DTPO-variance-v3-job.json",
    "actual-native-input-verification-v62.json": "actual-native-input-variance-v3-verification.json",
    "63b0d623380bab3173eaf2c7148bddf21b557827625bd5eff25c5b4faa05daa4":
        "b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943",
    "{1426979: 1791977192.37, 1429032: 1791977210.63}":
        "{3151306: 1791993626.33, 3153050: 1791993643.98}",
    "from dt_training_batch import CREDIT_KEYS":
        "from dt_training_batch import CREDIT_KEYS\nfrom verl.utils import torch_functional as verl_F",
    "checks = dict(": """raw_DT = data.batch['dt_token_advantages'].clone()
variance_scale = torch.rsqrt(verl_F.masked_var(raw_DT, mask) + 1e-8)
native_DT_view = data.select(deepcopy=True)
native_DT_view.batch = data.batch.clone(recurse=False)
native_DT = compute_advantage(native_DT_view,
    adv_estimator=AdvantageEstimator.DELTATRACE, multi_turn=True)
checks = dict(""",
    "actual_advantages_equal_DT_output=torch.equal(original_advantages, data.batch['dt_token_advantages']),":
        """actual_advantages_equal_deployed_native_boundary=torch.equal(original_advantages, native_DT.batch['advantages']),
    actual_variance_only_contract=torch.equal(original_advantages, torch.where(mask, raw_DT*variance_scale, 0.0)),
    actual_nonzero_signs_preserved=torch.equal(original_advantages[mask & raw_DT.ne(0)].sign(), raw_DT[mask & raw_DT.ne(0)].sign()),
    actual_finite_advantages=bool(torch.isfinite(original_advantages).all()),""",
    "actual_advantage_norm=float(original_advantages.double().norm()),":
        """actual_advantage_norm=float(original_advantages.double().norm()),
    raw_DT_masked_mean=float(verl_F.masked_mean(raw_DT, mask)),
    raw_DT_masked_std=float(verl_F.masked_var(raw_DT, mask).sqrt()),
    variance_scale=float(variance_scale),
    actual_advantage_masked_mean=float(verl_F.masked_mean(original_advantages, mask)),
    actual_advantage_masked_std=float(verl_F.masked_var(original_advantages, mask).sqrt()),""",
    "Actual zero-group transport and no-whitening regression":
        "Actual zero-group transport and variance-only boundary regression",
}
for old, new in changes.items():
    assert old in text, old
    text = text.replace(old, new)
exec(compile(text, str(source), "exec"), globals())
