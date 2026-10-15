"""Compose the accepted passive recorder with the tested v5 owner, AST exact."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'rl_textcraft_weighted_reward_targets_20261015_v4'
source=PREVIOUS/'prepare_passive_v4.py'
code=source.read_text(encoding='utf-8')
code=code.replace("prior=(HERE/'baseline/passive_reward_readout.py').read_text()",
                  "prior=(PREVIOUS/'baseline/passive_reward_readout.py').read_text()")
code=code.replace("installer=(HERE/'baseline/install_passive_records_native_graph.py').read_text()",
                  "installer=(PREVIOUS/'baseline/install_passive_records_native_graph.py').read_text()")
code=code.replace('textcraft-weighted-reward-targets-20261015-v4','textcraft-reward-target-exponential-20261016-v5')
code=code.replace('passive_reward_readout_v4.py','passive_reward_readout_v5.py')
code=code.replace('install_passive_records_v4.py','install_passive_records_v5.py')
code=code.replace("credit_transform='linear_finite_allocation'","credit_transform='per_event_exponential_then_credit_sum'")
code=code.replace('source_log_ratios_is_legacy_field_name=True','source_log_ratios_is_legacy_field_name=False')
code=code.replace("coefficient_kind='official_GRPO_group_multiplier'", "coefficient_kind='stable_target_credit; native_TextCraft_uses_official_GRPO'")
code=code.replace('tested_v4_owner','tested_v5_owner')
exec(compile(code,str(source),'exec'),globals())
